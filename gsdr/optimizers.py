import jax
import jax.numpy as jnp
import optax
from flax.struct import dataclass
from typing import Any, Callable, Optional, Tuple
import numpy as np

# --- Transforms ---

class ClampTransform:
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper
    def forward(self, x):
        return jnp.clip(x, self.lower, self.upper)

# --- SDR (Stochastic Delta Rule) ---

@dataclass
class SDRState:
    momentum_accum: Any
    step_count: int

def SDR(
    learning_rate: float = 1e-2,
    momentum: float = 0.9,
    sigma: float = 0.1,
    change_lower_bound: float = -1.0,
    change_upper_bound: float = 1.0,
    delta_distribution: Callable = jax.random.normal
) -> optax.GradientTransformation:
    """
    Stochastic Delta Rule (SDR) optimizer.
    """
    def init_fn(params):
        momentum_accum = jax.tree.map(jnp.zeros_like, params)
        return SDRState(momentum_accum=momentum_accum, step_count=0)

    def update_fn(updates, state, params=None, key=None):
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
                kn = max(1, int(np.sqrt(n)))
                km = max(1, int(np.sqrt(m)))
                kernel = jnp.ones((kn, km)) / (kn * km)
                return jax.scipy.signal.convolve2d(x, kernel, mode='same')
            elif x.ndim == 1:
                n = x.shape[0]
                k = max(1, int(np.sqrt(n)))
                kernel = jnp.ones((k,)) / k
                return jnp.convolve(x, kernel, mode='same')
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

# --- GSDR (Genetic Stochastic Delta Rule) ---

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
        if params is None or value is None or key is None:
            raise ValueError("GSDR requires 'params', 'value' (loss), and 'key' to be passed to update().")

        grads = updates
        loss = value

        is_new_opt = (loss < state.loss_opt)
        new_params_opt = jax.tree.map(lambda cur, opt: jnp.where(is_new_opt, cur, opt), params, state.params_opt)
        new_loss_opt = jnp.where(is_new_opt, loss, state.loss_opt)
        new_a_opt = jnp.where(is_new_opt, state.a, state.a_opt)
        new_inner_state_opt = jax.tree.map(lambda cur, opt: jnp.where(is_new_opt, cur, opt), state.inner_state, state.inner_state_opt)

        next_consecutive_unchanged_epochs = jnp.where(is_new_opt, 0, state.consecutive_unchanged_epochs + 1)
        step_of_last_optimal_change = jnp.where(is_new_opt, state.step_count + 1, state.last_optimal_change_step)

        is_deselect = ((loss > (new_loss_opt * deselection_threshold)) & (new_loss_opt != jnp.inf)) | (jnp.isnan(loss))
        is_reset_due_to_checkpoint = (state.step_count > 0) & (next_consecutive_unchanged_epochs >= checkpoint_n) & (new_loss_opt != jnp.inf)
        should_reset = is_deselect | is_reset_due_to_checkpoint

        def reset_branch(operand):
            _params, _new_params_opt, _new_inner_state_opt, _current_step = operand
            reset_updates = jax.tree.map(lambda opt_p, cur_p: opt_p - cur_p, _new_params_opt, _params)
            reset_state = GSDRState(
                inner_state=_new_inner_state_opt, params_opt=_new_params_opt, 
                inner_state_opt=_new_inner_state_opt, loss_opt=new_loss_opt,
                a=new_a_opt, a_opt=new_a_opt, lambda_d=state.lambda_d,
                step_count=_current_step, consecutive_unchanged_epochs=0,
                last_optimal_change_step=_current_step
            )
            return reset_updates, reset_state

        def normal_branch(operand):
            _params, _new_params_opt, _new_inner_state_opt, _current_step = operand
            time_since_last_change = jnp.maximum(0, _current_step - step_of_last_optimal_change)
            effective_lambda_d = (time_since_last_change**2) * (1.0 - jnp.exp(-(time_since_last_change) / tau_a_growth))

            inner_opt_key, _, a_key, noise_key = jax.random.split(key, 4)
            next_a = jnp.clip(state.a + jax.random.uniform(a_key, minval=-.1, maxval=.1), 0.0, 1.0) if a_dynamic else state.a

            boundTransform = ClampTransform(-1.0, 1.0)
            inner_updates, updated_inner_state = inner_optimizer.update(grads, state.inner_state, _params, key=inner_opt_key)
            inner_updates = jax.tree.map(lambda x: boundTransform.forward(x), inner_updates)

            param_leaves, treedef = jax.tree.flatten(_params)
            subkeys = jax.random.split(noise_key, len(param_leaves))
            delta_d = jax.tree.map(lambda p, k: delta_distribution(k, p.shape), _params, jax.tree.unflatten(treedef, subkeys))

            if mcdp and mcdp_factors is not None:
                delta = jax.tree.map(lambda n, p, r: n * p * r, delta_d, _params, mcdp_factors)
            else:
                delta = jax.tree.map(lambda n, p: n * p, delta_d, _params)

            delta = jax.tree.map(lambda x: boundTransform.forward(x), delta)
            combined_updates = jax.tree.map(lambda d, g: effective_lambda_d * (next_a * d + (1 - next_a) * g), delta, inner_updates)

            return combined_updates, GSDRState(
                inner_state=updated_inner_state, params_opt=_new_params_opt,
                inner_state_opt=_new_inner_state_opt, loss_opt=new_loss_opt,
                a=next_a, a_opt=new_a_opt, lambda_d=state.lambda_d,
                step_count=_current_step, consecutive_unchanged_epochs=next_consecutive_unchanged_epochs,
                last_optimal_change_step=step_of_last_optimal_change
            )

        current_step = state.step_count + 1
        return jax.lax.cond(should_reset, reset_branch, normal_branch, (params, new_params_opt, new_inner_state_opt, current_step))

    return optax.GradientTransformation(init_fn, update_fn)
