import jax
import jax.numpy as jnp
import numpy as np
from typing import Optional

def noise_current(
    i_delay: float,
    i_dur: float,
    i_amp: float,
    delta_t: float,
    t_max: float,
    seed: Optional[int] = None, 
    noise_standard_deviation: Optional[float] = None, 
    noise_correlation_tau: Optional[float] = None, 
    noise_mean: Optional[float] = None
) -> jnp.ndarray:
    """
    Generates a random noise current pulse using an Ornstein-Uhlenbeck process.
    """
    if noise_standard_deviation is None: noise_standard_deviation = 0.1
    if noise_correlation_tau is None: noise_correlation_tau = 10.0
    if noise_mean is None: noise_mean = 0.1

    if seed is None:
        seed = int(np.random.randint(0, 2**31 - 1))
    key = jax.random.PRNGKey(seed)
    num_steps = int(t_max / delta_t) + 1
    time_axis_array = jnp.arange(0, t_max + delta_t, delta_t)

    def ou_step_fn(n_prev, key_t):
        key_noise, _ = jax.random.split(key_t)
        xi = jax.random.normal(key_noise)
        drift = (noise_mean - n_prev) / noise_correlation_tau * delta_t
        diffusion = noise_standard_deviation * jnp.sqrt(2.0 / noise_correlation_tau) * xi * jnp.sqrt(delta_t)
        new_n = n_prev + drift + diffusion
        return new_n, new_n

    initial_state_ou = noise_mean
    keys_series = jax.random.split(key, num_steps - 1)
    _, full_noise_trace_raw = jax.lax.scan(ou_step_fn, initial_state_ou, keys_series)
    full_noise_trace = jnp.concatenate([jnp.array([initial_state_ou]), full_noise_trace_raw])

    time_off = i_delay + i_dur
    pulse_mask = (time_axis_array >= i_delay) & (time_axis_array <= time_off)
    return jnp.where(pulse_mask, full_noise_trace * i_amp, 0.0)

def ramp_current(i_delay: float, i_dur: float, i_amp: float, delta_t: float, t_max: float) -> jnp.ndarray:
    """Generates a ramping current pulse."""
    time_axis_array = jnp.arange(0, t_max + delta_t, delta_t)
    t_ramp_end = i_delay + i_dur
    ramp_mask = (time_axis_array >= i_delay) & (time_axis_array <= t_ramp_end)
    ramp_factor = (time_axis_array - i_delay) / i_dur
    return jnp.where(ramp_mask, i_amp * ramp_factor, 0.0)

def step_current(i_delay: float, i_dur: float, i_amp: float, delta_t: float, t_max: float) -> jnp.ndarray:
    """Generates a step current pulse."""
    time_axis_array = jnp.arange(0, t_max + delta_t, delta_t)
    t_step_end = i_delay + i_dur
    step_mask = (time_axis_array >= i_delay) & (time_axis_array <= t_step_end)
    return jnp.where(step_mask, i_amp, 0.0)

def noise_current_ac(
    i_delay: float, i_dur: float, amp_n: float, amp_b: float, 
    spect: jnp.ndarray, delta_t: float, t_max: float, seed: Optional[int] = None
) -> jnp.ndarray:
    """Generates combined OU noise and AC sinusoidal waves."""
    num_steps = int(t_max / delta_t) + 1
    time_axis_array = jnp.arange(0, t_max + delta_t, delta_t)
    
    # Noise part
    noise_trace = noise_current(i_delay, i_dur, amp_n, delta_t, t_max, seed=seed, noise_mean=0.0)
    
    # AC part
    t_reshaped = time_axis_array[None, :]
    freqs_reshaped = spect[:, None]
    phases = 2 * jnp.pi * freqs_reshaped * (t_reshaped / 1000.0)
    ac_trace = jnp.sum(jnp.sin(phases), axis=0)
    
    time_off = i_delay + i_dur
    pulse_mask = (time_axis_array >= i_delay) & (time_axis_array <= time_off)
    
    return jnp.where(pulse_mask, noise_trace + (ac_trace * amp_b), 0.0)
