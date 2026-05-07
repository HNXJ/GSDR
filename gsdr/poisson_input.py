#!/usr/bin/env python
"""
Poisson-like spike train generator with refractory period constraints.

Generates background input noise for neural simulations.
ISI (inter-spike interval) drawn from exponential with refractory period enforced.
"""

import jax
import jax.numpy as jnp
import numpy as np


def generate_poisson_spikes_numpy(
    duration_ms: float,
    dt_ms: float,
    rate_hz: float = 1.0,
    min_isi_ms: float = 200.0,
    seed: int = None,
) -> np.ndarray:
    """
    Generate Poisson spike train with refractory period constraint.

    Args:
      duration_ms: total simulation duration (ms)
      dt_ms: timestep (ms)
      rate_hz: baseline spike rate (Hz) — mean ISI = 1/rate
      min_isi_ms: minimum ISI / refractory period (ms)
      seed: random seed

    Returns:
      spikes: shape (T,), binary spike times {0, 1}
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()

    T = int(np.round(duration_ms / dt_ms))
    spikes = np.zeros(T, dtype=np.float32)

    mean_isi = 1000.0 / rate_hz  # ms
    min_isi_samples = int(np.round(min_isi_ms / dt_ms))

    t = 0
    while t < T:
        # Exponential ISI (Poisson process)
        isi_samples = int(np.maximum(
            min_isi_samples,
            np.round(np.random.exponential(mean_isi / dt_ms))
        ))
        t += isi_samples

        if t < T:
            spikes[t] = 1.0

    return spikes


def generate_poisson_spikes_jax(
    duration_ms: float,
    dt_ms: float,
    rate_hz: float = 1.0,
    min_isi_ms: float = 200.0,
    key: jnp.ndarray = None,
) -> jnp.ndarray:
    """
    JAX-compatible Poisson spike train generator (for simulation use).

    Note: JAX version is approximate (fully differentiable but less precise).
    For data generation, use numpy version; for simulation input, use this.

    Args:
      duration_ms, dt_ms, rate_hz, min_isi_ms: as above
      key: jax.random.PRNGKey

    Returns:
      spikes: shape (T,), binary {0, 1}
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    T = int(jnp.round(duration_ms / dt_ms))
    mean_isi = 1000.0 / rate_hz  # ms
    min_isi_samples = int(jnp.round(min_isi_ms / dt_ms))

    # Approximate: Poisson with threshold
    # Draw random values, threshold to get ~rate_hz firing
    key, subkey = jax.random.split(key)
    u = jax.random.uniform(subkey, shape=(T,))

    # Probability of spike at each timestep (Poisson approximation)
    p_spike = rate_hz / 1000.0  # dt_ms converted from ms to s
    spikes_raw = (u < p_spike).astype(jnp.float32)

    # Enforce refractory period (simple: suppress within min_isi)
    def enforce_refractory(spikes, min_isi):
        """Simple refractory enforcement: zero out spikes within min_isi of previous."""
        spikes_out = spikes.copy()
        last_spike = -jnp.inf

        for t in range(len(spikes)):
            if spikes[t] > 0.5:
                if t - last_spike >= min_isi_samples:
                    last_spike = t
                else:
                    spikes_out = spikes_out.at[t].set(0.0)

        return spikes_out

    # JAX version: use scan for loop-like behavior
    def step(carry, x):
        last_spike = carry
        t = x[0]
        spike = x[1]

        # If spike and enough time since last spike
        keep = spike * jnp.where(t - last_spike >= min_isi_samples, 1.0, 0.0)
        new_last = jnp.where(keep > 0.5, t, last_spike)

        return new_last, keep

    init_last_spike = -jnp.inf
    t_indices = jnp.arange(T)
    _, spikes_refractory = jax.lax.scan(
        step, init_last_spike, (t_indices, spikes_raw)
    )

    return spikes_refractory


def poisson_input_current(
    spikes: np.ndarray,
    amplitude: float = 5.0,
    duration_ms: float = None,
    dt_ms: float = 0.1,
) -> np.ndarray:
    """
    Convert spike train to input current (rectangular pulses).

    Args:
      spikes: shape (T,), binary spike times
      amplitude: current magnitude per spike (mV/ms)
      duration_ms: pulse duration (ms); if None, use dt_ms
      dt_ms: timestep

    Returns:
      current: shape (T,), time-varying input current
    """
    if duration_ms is None:
        duration_ms = dt_ms

    pulse_samples = max(1, int(np.round(duration_ms / dt_ms)))
    T = len(spikes)

    current = np.zeros(T, dtype=np.float32)

    for t in np.where(spikes > 0.5)[0]:
        t_end = min(t + pulse_samples, T)
        current[t:t_end] += amplitude

    return current


# ==============================================================================
# Test / Demo
# ==============================================================================

if __name__ == "__main__":
    duration_ms = 1000.0
    dt_ms = 0.1
    rate_hz = 1.0
    min_isi_ms = 200.0

    print("=" * 80)
    print("POISSON SPIKE TRAIN GENERATOR")
    print("=" * 80)

    print(f"\nParameters:")
    print(f"  Duration: {duration_ms} ms")
    print(f"  Timestep: {dt_ms} ms")
    print(f"  Target rate: {rate_hz} Hz (mean ISI = {1000/rate_hz} ms)")
    print(f"  Min ISI (refractory): {min_isi_ms} ms")

    # Generate with numpy
    spikes_np = generate_poisson_spikes_numpy(
        duration_ms, dt_ms, rate_hz, min_isi_ms, seed=42
    )

    n_spikes = int(spikes_np.sum())
    actual_rate = n_spikes / (duration_ms / 1000.0)
    spike_times = np.where(spikes_np > 0.5)[0] * dt_ms

    if len(spike_times) > 1:
        isis = np.diff(spike_times)
        mean_isi = np.mean(isis)
        min_isi = np.min(isis)
    else:
        mean_isi = np.nan
        min_isi = np.nan

    print(f"\nNumPy Generator Results:")
    print(f"  Total spikes: {n_spikes}")
    print(f"  Actual rate: {actual_rate:.2f} Hz")
    print(f"  Mean ISI: {mean_isi:.1f} ms")
    print(f"  Min ISI: {min_isi:.1f} ms (constraint: {min_isi_ms} ms)")
    print(f"  First 10 spike times (ms): {spike_times[:10]}")

    # Convert to input current
    current = poisson_input_current(spikes_np, amplitude=5.0, dt_ms=dt_ms)
    print(f"\nInput Current:")
    print(f"  Max current: {current.max():.2f} mV/ms")
    print(f"  Mean current: {current.mean():.4f} mV/ms")

    # JAX version
    print(f"\nJAX Generator Results:")
    key = jax.random.PRNGKey(42)
    spikes_jax = np.array(
        generate_poisson_spikes_jax(duration_ms, dt_ms, rate_hz, min_isi_ms, key)
    )
    n_spikes_jax = int(spikes_jax.sum())
    actual_rate_jax = n_spikes_jax / (duration_ms / 1000.0)
    print(f"  Total spikes: {n_spikes_jax}")
    print(f"  Actual rate: {actual_rate_jax:.2f} Hz")

    print("\n" + "=" * 80)
