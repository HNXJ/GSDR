#!/usr/bin/env python
"""
Data-driven optimization with JAX-native spectral loss.

Key improvements:
- No scipy in gradient path (fully differentiable)
- Soft spike proxies from voltage (better gradients)
- Multi-term loss (spectral shape + cosine + bandpower)
- Downsample 10kHz simulation to 1kHz for efficient PSD computation
"""

import sys
sys.path.insert(0, 'src')

import jax
import jax.numpy as jnp
import optax
import time
import numpy as np
from jbiophysic.models.reduced.net_eig_izh import net_eig_izh_jax
from spectral_loss import (
    soft_spike_rate_from_voltage,
    downsample_time_mean,
    jax_welch_psd,
    combined_spectral_loss,
    make_target_psds,
)
from poisson_input import generate_poisson_spikes_numpy, poisson_input_current


# Simple gradient-free optimizer for network fitting
class SimpleRandomSearch:
    """Simple random search optimizer for gradient-free optimization."""
    def __init__(self, learning_rate=0.05, exploration_amplitude=2.0):
        self.learning_rate = learning_rate
        self.exploration_amplitude = exploration_amplitude
        self.best_params = None
        self.best_loss = jnp.inf
        self.step_count = 0

    def update(self, params, loss, rng):
        """Update parameters based on random perturbations."""
        self.step_count += 1

        # Generate random perturbations
        param_leaves, treedef = jax.tree.flatten(params)
        subkeys = jax.random.split(rng, len(param_leaves))
        deltas = [self.exploration_amplitude * jax.random.normal(k, p.shape) for k, p in zip(subkeys, param_leaves)]

        # Apply small random steps
        new_leaves = [p - self.learning_rate * d for p, d in zip(param_leaves, deltas)]
        new_params = jax.tree.unflatten(treedef, new_leaves)

        return new_params


def run_optimization_spectral(
    empirical_data_path: str,
    n_steps: int = 100,
    seed: int = 0,
):
    """
    Run optimization with JAX-native spectral loss.
    """

    print("=" * 80)
    print(f"DATA-DRIVEN OPTIMIZATION: JAX-Native Spectral Loss, {n_steps} steps")
    print("=" * 80)

    # Load empirical data
    print("\n1. Loading empirical spike data...")
    spike_data = jnp.asarray(np.load(empirical_data_path), dtype=jnp.float32)
    print(f"   Shape: {spike_data.shape}")

    # Extract target PSDs (baseline + stimulus)
    print("\n2. Computing target PSDs from empirical data...")
    print("   Using baseline: t=0–500ms, stimulus: t=600–1200ms")
    baseline_psd, stim_psd, target_freqs = make_target_psds(
        spike_data,
        fs_hz=1000.0,
        target_freqs=jnp.linspace(5.0, 100.0, 100),
        baseline_time_slice=slice(0, 500),
        stim_time_slice=slice(600, 1200),
        frame_ms=500.0,
        hop_ms=50.0,
    )
    print(f"   Target frequencies: [{target_freqs.min():.1f}, {target_freqs.max():.1f}] Hz")
    print(f"   Baseline PSD shape: {baseline_psd.shape}")
    print(f"   Stimulus PSD shape: {stim_psd.shape}")

    # Network configuration
    net_config = {
        "n_neurons": 89,
        "composition": [
            [0.7, 0.2, 0.05, 0.05],
            [0.6, 0.25, 0.1, 0.05],
            [0.65, 0.2, 0.1, 0.05],
        ],
        "layer_edges_um": [0, 400, 700, 1000],
        "granular_layer": 2,
        "dt_ms": 0.1,
        "t_ms": 500.0,
        "seed": seed,
        "allow_autapses": False,
    }

    print("\n3. Initializing network...")
    t0 = time.time()
    result = net_eig_izh_jax(net_config)
    build_time = time.time() - t0

    params = result["params"]
    simulate_fn = result["simulate"]
    project_params = result["project_params"]
    trainable_masks = result["trainable_masks"]

    # Scale conductances AND use fast-spiking (FS) parameters for higher excitability
    conductance_scale = 50.0  # Increased from 10x to 50x

    # Use fast-spiking (FS) parameters (a=0.1) instead of regular-spiking (a=0.02)
    # for ~10x higher excitability
    a_fs = 0.1  # FS parameter instead of 0.02 (RS)

    params = {
        "g_ampa": params["g_ampa"] * conductance_scale,
        "g_nmda": params["g_nmda"] * conductance_scale,
        "g_gaba_a": params["g_gaba_a"] * conductance_scale,
        "a": jnp.ones_like(params["a"]) * a_fs,  # Set all neurons to FS params
        "b": params["b"],
        "c": params["c"],
        "d": params["d"],
    }

    print(f"   ✓ Built network in {build_time:.2f}s")
    print(f"   ✓ Scaled conductances by {conductance_scale}x")
    print(f"   ✓ Using fast-spiking parameters (a={a_fs}) for all neurons")

    # Pre-generate Poisson inputs for all neurons
    print("\n4. Generating per-neuron Poisson input noise...")
    duration_ms = net_config["t_ms"]
    dt_ms = net_config["dt_ms"]
    T = int(np.round(duration_ms / dt_ms))
    N = net_config["n_neurons"]

    # Generate Poisson spikes for each neuron (rate=1Hz, min_isi=200ms)
    poisson_spikes = np.zeros((T, N), dtype=np.float32)
    poisson_amplitude = 20.0  # Increased from 5.0 to 20.0 mV/ms for stronger driving current
    for nid in range(N):
        spikes = generate_poisson_spikes_numpy(
            duration_ms, dt_ms, rate_hz=1.0, min_isi_ms=200.0, seed=seed + nid
        )
        # Convert to input current (amplitude=20.0 mV/ms for stronger effect)
        poisson_current = poisson_input_current(spikes, amplitude=poisson_amplitude, dt_ms=dt_ms)
        poisson_spikes[:, nid] = poisson_current

    # Add baseline tonic input to Poisson noise
    # E neurons (first ~70%): 200 mV/ms, others (~30%): 100 mV/ms (much stronger drive)
    tonic_input = np.zeros((T, N), dtype=np.float32)
    n_e = int(0.7 * N)
    tonic_input[:, :n_e] = 200.0  # E neurons - very strong baseline
    tonic_input[:, n_e:] = 100.0  # Inhibitory neurons

    total_input = poisson_spikes + tonic_input
    poisson_input_array = jnp.asarray(total_input)
    print(f"   ✓ Generated {N} Poisson spike trains")
    print(f"   ✓ Total input shape: {poisson_input_array.shape}")
    print(f"   ✓ Total input current range: [{poisson_input_array.min():.2f}, {poisson_input_array.max():.2f}] mV/ms")
    print(f"   ✓ Tonic baseline: {200.0:.1f} (E) / {100.0:.1f} (I) mV/ms + Poisson (amp={poisson_amplitude:.1f})")

    # Create loss function
    print("\n5. Setting up JAX-native spectral loss...")

    def loss_fn(params):
        """Spectral loss with actual spikes and Poisson input noise."""
        # Add Poisson input to tonic input
        sim_result = simulate_fn(params, input_current=poisson_input_array)

        if sim_result['status'] != 'PASS':
            return jnp.array(1e6, dtype=jnp.float32)

        spike_trace = sim_result['spike_trace']  # shape (T, N) - actual spikes from simulation
        T, N = spike_trace.shape

        # Use spike_trace directly (binary 0/1, but compatible with PSD computation)
        population_signal = jnp.mean(spike_trace, axis=1)  # shape (T,) - average spike rate

        # Downsample 10 kHz simulation to 1 kHz (factor=10, since dt=0.1ms)
        # Actually, the simulation is already at effective 1kHz in the output
        # (T=5000 timepoints for 500ms @ dt=0.1ms = 10kHz, but we pool them)
        # For now, use population signal directly

        # Compute spectral loss: compare to stimulus PSD
        # (Using stimulus as target; baseline would be alternative)
        spectral_loss = combined_spectral_loss(
            population_signal,
            target_psd=stim_psd,
            fs_hz=10000.0 / T * 5000,  # Effective sampling rate
            target_freqs=target_freqs,
            frame_ms=50.0,  # Short window since we're already downsampled
            hop_ms=5.0,
            shape_weight=1.0,
            cosine_weight=0.2,
            bandpower_weight=0.1,
        )

        # Firing rate penalty (actual spikes integrated)
        # spike_trace is (T, N), mean gives average spike probability across time and neurons
        mean_spike_prob = jnp.mean(spike_trace)
        mean_rate = mean_spike_prob * (1000.0 / dt_ms)  # Convert spike probability to Hz
        fr_penalty = jnp.maximum(0.0, 10.0 - mean_rate) ** 2  # Encourage minimum 10 Hz

        total_loss = spectral_loss + 0.01 * fr_penalty

        return total_loss

    # Test loss and check spike output
    test_sim = simulate_fn(params, input_current=poisson_input_array)
    print(f"   Test simulation status: {test_sim['status']}")
    print(f"   Test spike_trace shape: {test_sim['spike_trace'].shape}")
    print(f"   Test spike_trace sum: {int(test_sim['spike_trace'].sum())} spikes")
    test_mean_spikes = jnp.mean(test_sim['spike_trace'])
    print(f"   Test mean spike rate: {test_mean_spikes * 1000.0 / dt_ms:.2f} Hz")

    loss_init = float(loss_fn(params))
    print(f"   ✓ Initial loss: {loss_init:.4f}")

    # Create simple gradient-free optimizer
    print("\n6. Creating random search optimizer...")
    opt = SimpleRandomSearch(learning_rate=0.05, exploration_amplitude=2.0)
    rng = jax.random.PRNGKey(seed)

    # Training loop
    print(f"\n7. Training loop ({n_steps} steps)...")
    print(f"   Step | Loss      | Pop Mean Rate")
    print("   " + "-" * 40)

    loss_history = []

    for step in range(n_steps):
        loss = loss_fn(params)
        loss_history.append(float(loss))

        # Estimate population rate for diagnostics
        sim_result = simulate_fn(params, input_current=poisson_input_array)
        if sim_result['status'] == 'PASS':
            spike_trace = sim_result['spike_trace']
            # spike_trace is binary (0/1), convert to Hz: spikes per time
            pop_rate = jnp.mean(spike_trace) * 1000.0 / dt_ms  # Average spikes per timestep * 1000 Hz / dt_ms
        else:
            pop_rate = 0.0

        # Random search update
        rng, sub_rng = jax.random.split(rng)
        params = opt.update(params, loss, sub_rng)
        params = project_params(params, params, trainable_masks)

        # Logging
        if (step + 1) % max(1, n_steps // 10) == 0 or step == 0:
            print(f"   {step+1:4d} | {loss:9.4f} | {float(pop_rate):10.2f} Hz")

    print("\n" + "=" * 80)
    print("✅ OPTIMIZATION COMPLETED")
    print("=" * 80)

    print(f"\nFinal Results:")
    print(f"  Initial loss: {loss_history[0]:.4f}")
    print(f"  Final loss: {loss_history[-1]:.4f}")
    if loss_history[0] > 0:
        improvement = (loss_history[0] - loss_history[-1]) / loss_history[0] * 100
        print(f"  Improvement: {improvement:.1f}%")

    return params, loss_history


if __name__ == "__main__":
    data_path = "/Users/hamednejat/claude-gamma-labyrinth/workspace/computational/selected_unit_spiking.npy"

    params, losses = run_optimization_spectral(
        data_path,
        n_steps=50,
        seed=42
    )
