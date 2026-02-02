import jaxley.optimize.transforms as jt
import numpy as np
import optax
import jax

import jax.scipy.signal
import jax.numpy as jnp

from typing import NamedTuple, Optional, Any
from flax.struct import dataclass
from typing import Any, Callable, NamedTuple, Optional, Tuple


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
    prev_loss: float # New: Store previous loss
    prev_params: Any # New: Store previous parameters

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
            step_count=0,
            prev_loss=jnp.inf, # Initialize prev_loss to infinity
            prev_params=params # Initialize prev_params with the initial parameters
        )

    def update_fn(updates, state, params=None, value=None, key=None):
        if key is None:
            raise ValueError("SDR requires a random 'key' to be passed to update().")
        if params is None:
            raise ValueError("SDR requires 'params' to be passed to update().")
        if value is None:
            raise ValueError("SDR requires current loss 'value' to be passed to update().")

        grads = updates
        current_loss = value

        # Handle the very first step (step_count == 0)
        if state.step_count == 0:
            initial_updates = jax.tree.map(jnp.zeros_like, params)
            new_state = SDRState(
                momentum_accum=jax.tree.map(lambda g: momentum * g, grads), # Update momentum accum with first gradients
                step_count=state.step_count + 1,
                prev_loss=current_loss,
                prev_params=params
            )
            return initial_updates, new_state


        # Calculate temporal difference components
        loss_diff = current_loss - state.prev_loss
        params_diff_tree = jax.tree.map(lambda p_curr, p_prev: p_curr - p_prev, params, state.prev_params)

        # Compute temporal_diff_tree
        temporal_diff_tree = jax.tree.map(lambda pd: loss_diff * pd, params_diff_tree)

        # Generate raw_updates using jax.random.uniform based on temporal_diff_tree
        # The range for uniform distribution is [min(0, td), max(0, td)]
        param_leaves, treedef = jax.tree.flatten(temporal_diff_tree)
        subkeys = jax.random.split(key, len(param_leaves))
        param_keys_tree = jax.tree.unflatten(treedef, subkeys)

        raw_updates = jax.tree.map(
            lambda td, k: jax.random.uniform(k, shape=td.shape, minval=jnp.minimum(0.0, td), maxval=jnp.maximum(0.0, td)),
            temporal_diff_tree, param_keys_tree
        )
        # Scale by -learning_rate
        raw_updates = jax.tree.map(lambda ru: -learning_rate * ru, raw_updates)


        # Apply clamping transform
        boundTransform = ClampTransform(change_lower_bound, change_upper_bound)
        final_updates = jax.tree.map(lambda x: boundTransform.forward(x), raw_updates)

        # Update state for the next iteration
        new_momentum_accum = jax.tree.map(
            lambda m, g: momentum * m + g,
            state.momentum_accum, grads
        )
        new_state = SDRState(
            momentum_accum=new_momentum_accum,
            step_count=state.step_count + 1,
            prev_loss=current_loss,
            prev_params=params
        )

        return final_updates, new_state

    return optax.GradientTransformation(init_fn, update_fn)


@dataclass
class GSDRState:
    inner_state: Any
    params_opt: Any
    inner_state_opt: Any
    loss_opt: float
    alpha: float
    a_opt: float
    lambda_d: float
    step_count: int
    consecutive_unchanged_epochs: int
    last_optimal_change_step: int

def GSDR(
    inner_optimizer: optax.GradientTransformation,
    delta_distribution: Callable = jax.random.normal,
    deselection_threshold: float = 2.0,
    lambda_d: float = 1.0,
    checkpoint_n: int = 10,
    tau_a_growth: float = 10.0,
    mcdp: bool = True
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
            alpha=0.5, # Initialized to 0.5, will be dynamically calculated
            a_opt=0.5, # Initialized to 0.5
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
        new_a_opt = jnp.where(is_new_opt, state.alpha, state.a_opt)
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
                alpha=new_a_opt,
                a_opt=new_a_opt,
                lambda_d=state.lambda_d,
                step_count=_current_step,
                consecutive_unchanged_epochs=0,
                last_optimal_change_step=_current_step
            )

            if is_reset_due_to_checkpoint:
                jax.debug.print(">Deselection(Checkpoint)")
            elif is_deselect:
                jax.debug.print(">Deselection(Divergence)")

            return reset_updates, reset_state

        def normal_branch(operand):
            _params, _new_params_opt, _new_inner_state_opt, _current_step = operand

            time_since_last_change = jnp.maximum(0, _current_step - step_of_last_optimal_change)
            effective_lambda_d = (time_since_last_change**2) * (1.0 - jnp.exp(-(time_since_last_change) / tau_a_growth))

            # Split key for inner optimizer and delta distribution, and noise (for delta_d_for_variance)
            inner_opt_key, delta_dist_key, noise_key_for_variance = jax.random.split(key, 3)

            # --- Dynamic alpha calculation based on variance --- #
            inner_updates_raw, updated_inner_state = inner_optimizer.update(grads, state.inner_state, _params, value=loss, key=inner_opt_key)
            # Ensure inner_updates are transformed for variance calculation consistency
            boundTransform = ClampTransform(-1.0, 1.0)
            transformed_inner_updates = jax.tree.map(lambda x: boundTransform.forward(x), inner_updates_raw)

            # Generate delta_d for variance calculation
            param_leaves_for_variance, treedef_for_variance = jax.tree.flatten(_params)
            subkeys_for_delta_variance = jax.random.split(noise_key_for_variance, len(param_leaves_for_variance))
            param_keys_tree_for_delta_variance = jax.tree.unflatten(treedef_for_variance, subkeys_for_delta_variance)

            delta_d_for_variance = jax.tree.map(
                lambda p, k: delta_distribution(k, p.shape),
                _params, param_keys_tree_for_delta_variance
            )
            transformed_delta_for_variance = jax.tree.map(lambda x: boundTransform.forward(x), delta_d_for_variance)

            # Flatten updates and deltas to compute variances
            flat_inner_updates, _ = jax.flatten_util.ravel_pytree(transformed_inner_updates)
            flat_delta, _ = jax.flatten_util.ravel_pytree(transformed_delta_for_variance)

            var_inner_updates = jnp.var(flat_inner_updates)
            var_delta = jnp.var(flat_delta)

            # Calculate alpha based on inverse variance weighting
            sum_var = var_inner_updates + var_delta
            next_a = jnp.where(sum_var > 1e-9, var_inner_updates / sum_var, 0.5) # Avoid division by zero
            # Clamp alpha between 0 and 1
            next_a = jnp.clip(next_a, 0.0, 1.0)
            # ---------------------------------------------------- #

            # Now, generate the delta_d for the actual update, potentially using mcdp_factors
            # We need a new key split for this to ensure statistical independence if desired,
            # or just use `delta_dist_key` for this purpose. Let's use `delta_dist_key` for consistency.
            param_leaves_for_update, treedef_for_update = jax.tree.flatten(_params)
            subkeys_for_delta = jax.random.split(delta_dist_key, len(param_leaves_for_update))
            param_keys_tree_for_delta_update = jax.tree.unflatten(treedef_for_update, subkeys_for_delta)

            delta_d_for_combined = jax.tree.map(
                lambda p, k: delta_distribution(k, p.shape),
                _params, param_keys_tree_for_delta_update
            )

            if mcdp and mcdp_factors is not None:
                delta_for_combined = jax.tree.map(lambda n, p, r: n * p * r, delta_d_for_combined, _params, mcdp_factors)
            else:
                delta_for_combined = jax.tree.map(lambda n, p: n * p, delta_d_for_combined, _params)

            delta_for_combined = jax.tree.map(lambda x: boundTransform.forward(x), delta_for_combined)

            delta_flat = jax.tree.util.tree_leaves(delta_for_combined)
            avg_delta = jnp.mean(jnp.concatenate([x.flatten() for x in delta_flat])) if delta_flat else jnp.array(0.0)
            std_delta = jnp.std(jnp.concatenate([x.flatten() for x in delta_flat])) if delta_flat else jnp.array(0.0)
            jax.debug.print("GSDR: Avg Delta (Noise): {}, Std Delta (Noise): {}", avg_delta, std_delta)


            combined_updates = jax.tree.map(
                lambda d, g: effective_lambda_d * (next_a * d + (1 - next_a) * g),
                delta_for_combined, transformed_inner_updates # Use transformed inner updates
            )

            combined_updates_flat = jax.tree.util.tree_leaves(combined_updates)
            avg_combined_update = jnp.mean(jnp.concatenate([x.flatten() for x in combined_updates_flat])) if combined_updates_flat else jnp.array(0.0)
            std_combined_update = jnp.std(jnp.concatenate([x.flatten() for x in combined_updates_flat])) if combined_updates_flat else jnp.array(0.0)
            jax.debug.print("GSDR: Avg Combined Update: {}, Std Combined Update: {}", avg_combined_update, std_combined_update)

            new_normal_state = GSDRState(
                inner_state=updated_inner_state,
                params_opt=_new_params_opt,
                inner_state_opt=_new_inner_state_opt,
                loss_opt=new_loss_opt,
                alpha=next_a, # Use the dynamically calculated next_a
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


@dataclass
class AdagenState:
    opt1_state: Any
    opt2_state: Any
    best_params: Any         # Stores the best parameters found so far
    best_loss: jnp.ndarray   # Stores the lowest loss value
    patience_count: jnp.ndarray # Counter for steps since last improvement
    just_reset: jnp.ndarray  # Flag: True if previous step was a reset/improvement
def adagen(
    optimizer1: optax.GradientTransformation,
    optimizer2: optax.GradientTransformation,
    max_checkpoints: int
) -> optax.GradientTransformation:
    """
    Adaptive Genetic Optimizer (ADAGEN) Optimizer.

    Combines two optimizers, weighting them by the inverse variance of their updates.
    Reverts to best_params if no improvement after max_checkpoints.
    """

    def init_fn(params):
        return AdagenState(
            opt1_state=optimizer1.init(params),
            opt2_state=optimizer2.init(params),
            best_params=params,
            best_loss=jnp.inf,
            patience_count=jnp.asarray(0, dtype=jnp.int32),
            just_reset=jnp.asarray(False, dtype=jnp.bool_)
        )

    def update_fn(updates, state, params=None, value=None, key=None):
        if params is None or value is None:
            # Fallback for standard optax chains lacking context (warn user ideally)
            return updates, state

        loss = value
        # 1. Get updates from internal optimizers
        updates1, new_opt1_state = optimizer1.update(updates, state.opt1_state, params)
        updates2, new_opt2_state = optimizer2.update(updates, state.opt2_state, params)

        # 2. Calculate variances (flatten updates to compute global scalar variance)
        # Helper to flatten pytree into single vector
        flat_u1, _ = jax.flatten_util.ravel_pytree(updates1)
        flat_u2, _ = jax.flatten_util.ravel_pytree(updates2)

        var1 = jnp.var(flat_u1)
        var2 = jnp.var(flat_u2)

        # 3. Calculate Adaptive Alpha
        # Equation: alpha * var1 = (1-alpha) * var2  => alpha = var2 / (var1 + var2)
        # Logic: More variance = less contribution.
        sum_var = var1 + var2
        # Handle 0 variance edge case (identical values) -> 0.5
        alpha = jnp.where(sum_var > 0, var2 / sum_var, 0.5)

        # 4. Apply Genetic Rounding (if just reset/improved)
        # If just_reset is True, round alpha to 0.0 or 1.0
        alpha = jnp.where(state.just_reset, jnp.round(alpha), alpha)

        # 5. Combine Updates: alpha * d1 + (1-alpha) * d2
        scaled_u1 = jax.tree.map(lambda x: x * alpha, updates1)
        scaled_u2 = jax.tree.map(lambda x: x * (1.0 - alpha), updates2)
        combined_update = jax.tree.map(lambda x, y: x + y, scaled_u1, scaled_u2)

        # 6. Check Logic: Improvement vs Reversion
        is_improvement = loss < state.best_loss
        is_patience_exceeded = state.patience_count >= max_checkpoints

        # Logic Branch A: Improvement found
        # Save new best, reset counter, flag just_reset for NEXT step
        new_best_loss = jnp.where(is_improvement, loss, state.best_loss)
        new_best_params = jax.tree.map(
            lambda n, o: jnp.where(is_improvement, n, o),
            params, state.best_params
        )
        new_count = jnp.where(is_improvement, 0, state.patience_count + 1)

        # Logic Branch B: Patience Exceeded (Revert)
        # If reverting, the 'update' must transform current params back to best_params
        # Revert update = best_params - current_params
        revert_update = jax.tree.map(lambda b, p: b - p, state.best_params, params)

        # Final selection of update to apply
        final_update = jax.tree.map(
            lambda reg, rev: jnp.where(is_patience_exceeded, rev, reg),
            combined_update, revert_update
        )

        # If we reverted, we reset count and flag just_reset
        new_count = jnp.where(is_patience_exceeded, 0, new_count)

        # just_reset is True if we Improved OR Reverted in this step
        new_just_reset = jnp.logical_or(is_improvement, is_patience_exceeded)

        new_state = AdagenState(
            opt1_state=new_opt1_state,
            opt2_state=new_opt2_state,
            best_params=new_best_params,
            best_loss=new_best_loss,
            patience_count=new_count,
            just_reset=new_just_reset
        )

        return final_update, new_state

    return optax.GradientTransformation(init_fn, update_fn)
