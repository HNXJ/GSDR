import jax.numpy as jnp
import jaxley as jx
import numpy as np
import jaxley.optimize.transforms as jt
import jax
import jax.scipy.signal

from jax import jit, vmap, value_and_grad
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap # Import for custom colormap
from scipy import signal # Import scipy.signal for spectrogram
from scipy import ndimage # Import scipy.ndimage for smoothing

import optax
from flax.struct import dataclass # For GSDR
from typing import Any, Callable, NamedTuple, Optional, Tuple # Added Tuple
from scipy.ndimage import zoom, gaussian_filter

from scipy import signal
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np


class Dataset:
    """
    A simple Dataloader which returns batches of the data.

    Instead of using this simple dataloader, you can also just use one from
    PyTorch or Tensorflow. You do not have to understand what is going on here
    to follow this tutorial.
    """

    def __init__(self, inputs: np.ndarray, labels: np.ndarray):
        """
        Initialize the dataloader.

        Args:
            inputs: Array of shape (num_samples, num_dim)
            labels: Array of shape (num_samples,)
        """
        assert len(inputs) == len(labels), "Inputs and labels must have same length"
        self.inputs = inputs
        self.labels = labels
        self.num_samples = len(inputs)
        self._rng_state = None
        self.batch_size = 1

    def shuffle(self, seed=None):
        """
        Shuffle the dataset in-place
        """
        self._rng_state = np.random.get_state()[1][0] if seed is None else seed
        np.random.seed(self._rng_state)
        indices = np.random.permutation(self.num_samples)
        self.inputs = self.inputs[indices]
        self.labels = self.labels[indices]
        return self

    def batch(self, batch_size):
        """
        Create batches of the data.
        """
        self.batch_size = batch_size
        return self

    def __iter__(self):
        """
        Iterate over the dataset.
        """
        self.shuffle(seed=self._rng_state)
        for start in range(0, self.num_samples, self.batch_size):
            end = min(start + self.batch_size, self.num_samples)
            yield self.inputs[start:end], self.labels[start:end]
        self._rng_state += 1


class ClampTransform(jt.Transform):
    def __init__(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def forward(self, x):
        return jnp.nan_to_num(jnp.clip(x, self.lower_bound, self.upper_bound), 0.0)

    def inverse(self, x):
        return x


@dataclass
class SDRState:
    momentum_accum: Any
    step_count: int

def SDR(
    learning_rate: float = 1e-3,
    momentum: float = 0.1,
    change_lower_bound: float = -1.0,
    change_upper_bound: float = 1.0,
    delta_distribution: Callable = jax.random.uniform,
    sigma: float = 1.0
) -> optax.GradientTransformation:
    """
    Optax-compliant implementation of the Stochastic Delta Rule (SDR).
    Includes x32 stability transform (Clamp/OmitNaN).
    """
    def init_fn(params):
        momentum_accum = jax.tree.map(lambda p: jnp.zeros_like(p), params)
        return SDRState(
            momentum_accum=momentum_accum,
            step_count=0
        )

    def update_fn(updates, state, params=None, value=None, key=None):
        if key is None:
            raise ValueError("SDR requires a random 'key' to be passed to update().")

        grads = updates

        new_momentum_accum = jax.tree.map(
            lambda m, g: momentum * m + g,
            state.momentum_accum, grads
        )

        grad_signs = jax.tree.map(jnp.sign, new_momentum_accum)

        param_leaves, treedef = jax.tree.flatten(grads)
        subkeys = jax.random.split(key, len(param_leaves))
        param_keys_tree = jax.tree.unflatten(treedef, subkeys)

        random_factors = jax.tree.map(
            lambda g, k: sigma * delta_distribution(k, g.shape),
            grads, param_keys_tree
        )

        def smooth_factor(x):
            if x.ndim == 2:
                n, m = x.shape
                kn = int(np.sqrt(n))
                km = int(np.sqrt(m))
                kn = max(1, kn)
                km = max(1, km)
                kernel = jnp.ones((kn, km)) / (kn * km)
                return jax.scipy.signal.convolve2d(x, kernel, mode='same')
            elif x.ndim == 1:
                n = x.shape[0]
                k = int(np.sqrt(n))
                k = max(1, k)
                kernel = jnp.ones((k,)) / k
                return jax.scipy.signal.convolve(x, kernel, mode='same')
            else:
                return x

        random_factors = jax.tree.map(smooth_factor, random_factors)

        raw_updates = jax.tree.map(
            lambda s, r: -learning_rate * s * r,
            grad_signs, random_factors
        )

        boundTransform = ClampTransform(change_lower_bound, change_upper_bound)
        final_updates = jax.tree.map(lambda x: boundTransform.forward(x), raw_updates)

        new_state = SDRState(
            momentum_accum=new_momentum_accum,
            step_count=state.step_count + 1
        )

        return final_updates, new_state

    return optax.GradientTransformation(init_fn, update_fn)


@dataclass
class GSDRState:
    inner_state: Any
    params_opt: Any
    inner_state_opt: Any
    loss_opt: float
    a: float
    a_opt: float
    lambda_d: float
    step_count: int
    consecutive_unchanged_epochs: int
    last_optimal_change_step: int

def GSDR(
    inner_optimizer: optax.GradientTransformation,
    delta_distribution: Callable = jax.random.normal,
    deselection_threshold: float = 2.0,
    a_init: float = 0.5,
    lambda_d: float = 1.0,
    checkpoint_n: int = 10,
    tau_a_growth: float = 10.0,
    mcdp: bool = True,
    a_dynamic: bool = False
) -> optax.GradientTransformation:
    """
    Optax-compliant implementation of the Genetic-Stochastic Delta Rule.
    """

    def init_fn(params):
        inner_state = inner_optimizer.init(params)
        return GSDRState(
            inner_state=inner_state,
            params_opt=params,
            inner_state_opt=inner_state,
            loss_opt=jnp.inf,
            a=a_init,
            a_opt=a_init,
            lambda_d=lambda_d,
            step_count=0,
            consecutive_unchanged_epochs=0,
            last_optimal_change_step=0
        )

    def update_fn(updates, state, params=None, value=None, key=None, mcdp_factors=None):

        if params is None:
            raise ValueError("GSDR requires 'params' to be passed to update().")
        if value is None:
            raise ValueError("GSDR requires current loss 'value' to be passed to update().")
        if key is None:
            raise ValueError("GSDR requires a random 'key' to be passed to update().")

        grads = updates
        loss = value

        # 1. Update Best-Known State (Optimality Check)
        is_new_opt = (loss < state.loss_opt)

        new_params_opt = jax.tree.map(
            lambda cur, opt: jnp.where(is_new_opt, cur, opt),
            params, state.params_opt
        )
        new_loss_opt = jnp.where(is_new_opt, loss, state.loss_opt)
        new_a_opt = jnp.where(is_new_opt, state.a, state.a_opt)
        new_inner_state_opt = jax.tree.map(
            lambda cur, opt: jnp.where(is_new_opt, cur, opt),
            state.inner_state, state.inner_state_opt
        )

        next_consecutive_unchanged_epochs = jnp.where(
            is_new_opt, 0, state.consecutive_unchanged_epochs + 1
        )
        step_of_last_optimal_change = jnp.where(
            is_new_opt, state.step_count + 1, state.last_optimal_change_step
        )

        # 2. Check Reset Conditions
        is_deselect = ((loss > (new_loss_opt * deselection_threshold)) & (new_loss_opt != jnp.inf)) | (jnp.isnan(loss))
        is_reset_due_to_checkpoint = (state.step_count > 0) & \
                                     (next_consecutive_unchanged_epochs >= checkpoint_n) & \
                                     (new_loss_opt != jnp.inf)

        should_reset = is_deselect | is_reset_due_to_checkpoint

        def reset_branch(operand):
            _params, _new_params_opt, _new_inner_state_opt, _current_step = operand

            reset_updates = jax.tree.map(
                lambda opt_p, cur_p: opt_p - cur_p,
                _new_params_opt, _params
            )

            reset_state = GSDRState(
                inner_state=_new_inner_state_opt,
                params_opt=_new_params_opt,
                inner_state_opt=_new_inner_state_opt,
                loss_opt=new_loss_opt,
                a=new_a_opt,
                a_opt=new_a_opt,
                lambda_d=state.lambda_d,
                step_count=_current_step,
                consecutive_unchanged_epochs=0,
                last_optimal_change_step=_current_step
            )

            if is_reset_due_to_checkpoint:
                print(">Deselection(Checkpoint)")
            elif is_deselect:
                print(">Deselection(Divergence)")

            return reset_updates, reset_state

        def normal_branch(operand):
            _params, _new_params_opt, _new_inner_state_opt, _current_step = operand

            time_since_last_change = jnp.maximum(0, _current_step - step_of_last_optimal_change)
            effective_lambda_d = (time_since_last_change**2) * (1.0 - jnp.exp(-(time_since_last_change) / tau_a_growth))

            inner_opt_key, delta_dist_key, a_key, noise_key = jax.random.split(key, 4)

            if a_dynamic:
                delta_a = jax.random.uniform(a_key, minval=-.1, maxval=.1)
                a_candidate = state.a + delta_a
                next_a = jax.numpy.clip(a_candidate, 0.0, 1.0)
            else:
                next_a = state.a

            boundTransform = ClampTransform(-1.0, 1.0)

            inner_updates, updated_inner_state = inner_optimizer.update(grads, state.inner_state, _params, key=inner_opt_key)
            inner_updates = jax.tree.map(lambda x: boundTransform.forward(x), inner_updates)

            # Fixed typo here: jax.tree_util instead of jax.tree.util
            inner_updates_flat = jax.tree_util.tree_leaves(inner_updates)
            avg_inner_update = jnp.mean(jnp.concatenate([x.flatten() for x in inner_updates_flat])) if inner_updates_flat else jnp.array(0.0)
            std_inner_update = jnp.std(jnp.concatenate([x.flatten() for x in inner_updates_flat])) if inner_updates_flat else jnp.array(0.0)
            jax.debug.print("GSDR: Avg Inner Update: {}, Std Inner Update: {}", avg_inner_update, std_inner_update)

            param_leaves, treedef = jax.tree.flatten(_params)
            subkeys = jax.random.split(noise_key, len(param_leaves))
            param_keys_tree = jax.tree.unflatten(treedef, subkeys)

            delta_d = jax.tree.map(
                lambda p, k: delta_distribution(k, p.shape),
                _params, param_keys_tree
            )

            if mcdp and mcdp_factors is not None:
                delta = jax.tree.map(lambda n, p, r: n * p * r, delta_d, _params, mcdp_factors)
            else:
                delta = jax.tree.map(lambda n, p: n * p, delta_d, _params)

            delta = jax.tree.map(lambda x: boundTransform.forward(x), delta)

            # Fixed typo here
            delta_flat = jax.tree_util.tree_leaves(delta)
            avg_delta = jnp.mean(jnp.concatenate([x.flatten() for x in delta_flat])) if delta_flat else jnp.array(0.0)
            std_delta = jnp.std(jnp.concatenate([x.flatten() for x in delta_flat])) if delta_flat else jnp.array(0.0)
            jax.debug.print("GSDR: Avg Delta (MC): {}, Std Delta (MC): {}", avg_delta, std_delta)

            combined_updates = jax.tree.map(
                lambda d, g: effective_lambda_d * (next_a * d + (1 - next_a) * g),
                delta, inner_updates
            )

            # Fixed typo here
            combined_updates_flat = jax.tree_util.tree_leaves(combined_updates)
            avg_combined_update = jnp.mean(jnp.concatenate([x.flatten() for x in combined_updates_flat])) if combined_updates_flat else jnp.array(0.0)
            std_combined_update = jnp.std(jnp.concatenate([x.flatten() for x in combined_updates_flat])) if combined_updates_flat else jnp.array(0.0)
            jax.debug.print("GSDR: Avg Combined Update: {}, Std Combined Update: {}", avg_combined_update, std_combined_update)

            new_normal_state = GSDRState(
                inner_state=updated_inner_state,
                params_opt=_new_params_opt,
                inner_state_opt=_new_inner_state_opt,
                loss_opt=new_loss_opt,
                a=next_a,
                a_opt=new_a_opt,
                lambda_d=state.lambda_d,
                step_count=_current_step,
                consecutive_unchanged_epochs=next_consecutive_unchanged_epochs,
                last_optimal_change_step=step_of_last_optimal_change
            )

            return combined_updates, new_normal_state

        current_step = state.step_count + 1
        operand = (params, new_params_opt, new_inner_state_opt, current_step)

        final_updates, new_state = jax.lax.cond(
            should_reset,
            reset_branch,
            normal_branch,
            operand
        )

        return final_updates, new_state

    return optax.GradientTransformation(init_fn, update_fn)


