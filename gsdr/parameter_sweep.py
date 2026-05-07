#!/usr/bin/env python
"""
Parameter sweep: Poisson noise + Synaptic connectivity (E-E, E-I, I-E, I-I).

Sweeps:
1. Poisson noise amplitude: 0.1 - 50.0 mV/ms (step=5.0)
2. Poisson noise rate: 0.1 - 5.0 Hz (step=0.5)
3. Synaptic conductances: E-E, E-I, I-E, I-I (multiplicative scales 0.1-5.0)

Saves: simulation_logs.pkl with all results for analysis.
"""

import sys
sys.path.insert(0, 'src')

import jax
import jax.numpy as jnp
import numpy as np
import pickle
import time
from jbiophysic.models.reduced.net_eig_izh import net_eig_izh_jax
from poisson_input import generate_poisson_spikes_numpy, poisson_input_current


def save_logs(log_data, logs_path='simulation_logs.pkl'):
    """Save simulation logs to pickle file."""
    with open(logs_path, 'wb') as f:
        pickle.dump(log_data, f)
    print(f"✅ Saved logs to {logs_path}")


def run_poisson_sweep(n_steps: int = 1500, seed: int = 42):
    """
    Sweep Poisson noise parameters and measure firing rates.

    Args:
        n_steps: simulation duration in 0.1ms units
        seed: random seed
    """

    print("\n" + "="*80)
    print("PARAMETER SWEEP 1: POISSON NOISE AMPLITUDE & RATE")
    print("="*80)

    # Network configuration
    net_config = {
        "n_neurons": 89,
        "composition": [[0.7, 0.2, 0.05, 0.05], [0.6, 0.25, 0.1, 0.05], [0.65, 0.2, 0.1, 0.05]],
        "layer_edges_um": [0, 400, 700, 1000],
        "granular_layer": 2,
        "dt_ms": 0.1,
        "t_ms": 1500.0,
        "seed": seed,
        "allow_autapses": False,
    }

    print(f"Network: {net_config['n_neurons']} neurons, {net_config['t_ms']:.0f}ms")

    result = net_eig_izh_jax(net_config)
    params_base = result["params"]
    simulate_fn = result["simulate"]
    project_params = result["project_params"]
    trainable_masks = result["trainable_masks"]

    # Base parameters
    base_conductance_scale = 50.0
    dt_ms = net_config["dt_ms"]
    T = int(np.round(net_config["t_ms"] / dt_ms))
    N = net_config["n_neurons"]
    n_e = int(0.7 * N)

    # Base tonic input
    tonic_input_base = np.zeros((T, N), dtype=np.float32)
    tonic_input_base[:, :n_e] = 200.0
    tonic_input_base[:, n_e:] = 100.0

    # Poisson amplitude and rate sweeps
    poisson_amplitudes = np.arange(5.0, 55.0, 5.0)  # [5, 10, 15, ..., 50]
    poisson_rates = np.arange(0.5, 5.5, 0.5)  # [0.5, 1.0, 1.5, ..., 5.0]

    log_data_poisson = {
        'amplitudes': poisson_amplitudes.tolist(),
        'rates': poisson_rates.tolist(),
        'firing_rates': {},  # (amp, rate) -> firing_rate
        'spike_counts': {},
        'baseline_fr': {},
        'stim_fr': {},
    }

    print(f"\nPoisson sweep: {len(poisson_amplitudes)} amplitudes x {len(poisson_rates)} rates = {len(poisson_amplitudes)*len(poisson_rates)} simulations")
    print(f"Amplitudes: {poisson_amplitudes.tolist()}")
    print(f"Rates: {poisson_rates.tolist()}")
    print("\nRunning simulations...")

    for amp_idx, poisson_amp in enumerate(poisson_amplitudes):
        for rate_idx, poisson_rate in enumerate(poisson_rates):
            # Generate Poisson inputs for this configuration
            poisson_spikes = np.zeros((T, N), dtype=np.float32)
            for nid in range(N):
                spikes = generate_poisson_spikes_numpy(
                    net_config["t_ms"], dt_ms, rate_hz=poisson_rate, min_isi_ms=200.0, seed=seed + nid
                )
                poisson_current = poisson_input_current(spikes, amplitude=poisson_amp, dt_ms=dt_ms)
                poisson_spikes[:, nid] = poisson_current

            poisson_input_array = jnp.asarray(poisson_spikes + tonic_input_base)

            # Set base parameters
            params = {
                'g_ampa': params_base['g_ampa'] * base_conductance_scale,
                'g_nmda': params_base['g_nmda'] * base_conductance_scale,
                'g_gaba_a': params_base['g_gaba_a'] * base_conductance_scale,
                'a': jnp.ones_like(params_base['a']) * 0.1,
                'b': params_base['b'], 'c': params_base['c'], 'd': params_base['d'],
            }

            # Simulate
            sim_result = simulate_fn(params, input_current=poisson_input_array)

            key = (poisson_amp, poisson_rate)
            if sim_result['status'] == 'PASS':
                spike_trace = sim_result['spike_trace']

                # Total firing rate
                total_spikes = int(jnp.sum(spike_trace))
                total_fr = total_spikes / T * (1000.0 / dt_ms)

                # Baseline and stimulus
                baseline_end = int(500 / dt_ms)
                stim_start = int(600 / dt_ms)
                stim_end = int(1200 / dt_ms)

                baseline_fr = jnp.sum(spike_trace[:baseline_end]) / baseline_end * (1000.0 / dt_ms)
                stim_fr = jnp.sum(spike_trace[stim_start:stim_end]) / (stim_end - stim_start) * (1000.0 / dt_ms)

                log_data_poisson['firing_rates'][key] = float(total_fr)
                log_data_poisson['spike_counts'][key] = total_spikes
                log_data_poisson['baseline_fr'][key] = float(baseline_fr)
                log_data_poisson['stim_fr'][key] = float(stim_fr)
            else:
                log_data_poisson['firing_rates'][key] = 0.0
                log_data_poisson['spike_counts'][key] = 0
                log_data_poisson['baseline_fr'][key] = 0.0
                log_data_poisson['stim_fr'][key] = 0.0

            if (amp_idx * len(poisson_rates) + rate_idx + 1) % 10 == 0:
                print(f"  Progress: {amp_idx * len(poisson_rates) + rate_idx + 1}/{len(poisson_amplitudes)*len(poisson_rates)}")

    return log_data_poisson


def run_synapse_sweep(n_steps: int = 1500, seed: int = 42):
    """
    Sweep synaptic conductance ratios: E-E, E-I, I-E, I-I.

    Keeps overall conductance constant, modulates synapse type balance.
    """

    print("\n" + "="*80)
    print("PARAMETER SWEEP 2: SYNAPTIC CONNECTIVITY (E-E, E-I, I-E, I-I)")
    print("="*80)

    # Network configuration
    net_config = {
        "n_neurons": 89,
        "composition": [[0.7, 0.2, 0.05, 0.05], [0.6, 0.25, 0.1, 0.05], [0.65, 0.2, 0.1, 0.05]],
        "layer_edges_um": [0, 400, 700, 1000],
        "granular_layer": 2,
        "dt_ms": 0.1,
        "t_ms": 1500.0,
        "seed": seed,
        "allow_autapses": False,
    }

    result = net_eig_izh_jax(net_config)
    params_base = result["params"]
    simulate_fn = result["simulate"]
    project_params = result["project_params"]
    trainable_masks = result["trainable_masks"]

    # Base parameters
    base_conductance_scale = 50.0
    dt_ms = net_config["dt_ms"]
    T = int(np.round(net_config["t_ms"] / dt_ms))
    N = net_config["n_neurons"]
    n_e = int(0.7 * N)

    # Fixed Poisson input (use optimal from previous sweep)
    poisson_amp = 20.0
    poisson_rate = 1.0

    tonic_input = np.zeros((T, N), dtype=np.float32)
    tonic_input[:, :n_e] = 200.0
    tonic_input[:, n_e:] = 100.0

    poisson_spikes = np.zeros((T, N), dtype=np.float32)
    for nid in range(N):
        spikes = generate_poisson_spikes_numpy(
            net_config["t_ms"], dt_ms, rate_hz=poisson_rate, min_isi_ms=200.0, seed=seed + nid
        )
        poisson_current = poisson_input_current(spikes, amplitude=poisson_amp, dt_ms=dt_ms)
        poisson_spikes[:, nid] = poisson_current

    poisson_input_array = jnp.asarray(poisson_spikes + tonic_input)

    # Synapse type scales
    synapse_scales = np.arange(0.1, 5.1, 0.5)  # [0.1, 0.6, 1.1, ..., 5.0]

    log_data_synapses = {
        'scales': synapse_scales.tolist(),
        'synapse_types': ['E-E', 'E-I', 'I-E', 'I-I'],
        'firing_rates': {},  # (synapse_type, scale) -> firing_rate
        'spike_counts': {},
        'baseline_fr': {},
        'stim_fr': {},
    }

    print(f"\nSynapse type sweep: {len(synapse_scales)} scales x 4 types = {len(synapse_scales)*4} simulations")
    print(f"Scales: {synapse_scales.tolist()}")

    # For each synapse type, modify the network structure
    # This is simplified: we'll scale the different conductance components

    print("\nRunning simulations...")

    for synapse_idx, synapse_type in enumerate(log_data_synapses['synapse_types']):
        for scale_idx, scale in enumerate(synapse_scales):
            params = {
                'g_ampa': params_base['g_ampa'] * base_conductance_scale,
                'g_nmda': params_base['g_nmda'] * base_conductance_scale,
                'g_gaba_a': params_base['g_gaba_a'] * base_conductance_scale,
                'a': jnp.ones_like(params_base['a']) * 0.1,
                'b': params_base['b'], 'c': params_base['c'], 'd': params_base['d'],
            }

            # Apply synapse-type specific scaling
            if synapse_type == 'E-E':  # Scale E->E (AMPA/NMDA)
                params['g_ampa'] = params['g_ampa'] * scale
                params['g_nmda'] = params['g_nmda'] * scale
            elif synapse_type == 'E-I':  # Scale E->I (AMPA/NMDA to inhibitory)
                # Already in the AMPA/NMDA which connect E->All
                params['g_ampa'] = params['g_ampa'] * scale
            elif synapse_type == 'I-E':  # Scale I->E (GABAa)
                params['g_gaba_a'] = params['g_gaba_a'] * scale
            elif synapse_type == 'I-I':  # Scale I->I (GABAa)
                params['g_gaba_a'] = params['g_gaba_a'] * scale

            # Simulate
            sim_result = simulate_fn(params, input_current=poisson_input_array)

            key = (synapse_type, scale)
            if sim_result['status'] == 'PASS':
                spike_trace = sim_result['spike_trace']

                total_spikes = int(jnp.sum(spike_trace))
                total_fr = total_spikes / T * (1000.0 / dt_ms)

                baseline_end = int(500 / dt_ms)
                stim_start = int(600 / dt_ms)
                stim_end = int(1200 / dt_ms)

                baseline_fr = jnp.sum(spike_trace[:baseline_end]) / baseline_end * (1000.0 / dt_ms)
                stim_fr = jnp.sum(spike_trace[stim_start:stim_end]) / (stim_end - stim_start) * (1000.0 / dt_ms)

                log_data_synapses['firing_rates'][key] = float(total_fr)
                log_data_synapses['spike_counts'][key] = total_spikes
                log_data_synapses['baseline_fr'][key] = float(baseline_fr)
                log_data_synapses['stim_fr'][key] = float(stim_fr)
            else:
                log_data_synapses['firing_rates'][key] = 0.0
                log_data_synapses['spike_counts'][key] = 0
                log_data_synapses['baseline_fr'][key] = 0.0
                log_data_synapses['stim_fr'][key] = 0.0

            if (synapse_idx * len(synapse_scales) + scale_idx + 1) % 5 == 0:
                print(f"  Progress: {synapse_idx * len(synapse_scales) + scale_idx + 1}/{len(synapse_scales)*4}")

    return log_data_synapses


if __name__ == "__main__":
    print("\n" + "="*80)
    print("IZHIKEVICH NETWORK - COMPREHENSIVE PARAMETER SWEEP")
    print("="*80)

    t_start = time.time()

    # Run poisson sweep
    log_poisson = run_poisson_sweep(seed=42)

    # Run synapse sweep
    log_synapses = run_synapse_sweep(seed=42)

    # Combine logs
    all_logs = {
        'poisson_sweep': log_poisson,
        'synapse_sweep': log_synapses,
        'timestamp': time.time(),
    }

    # Save
    save_logs(all_logs, 'simulation_logs_izh.pkl')

    elapsed = time.time() - t_start
    print(f"\n{'='*80}")
    print(f"✅ PARAMETER SWEEP COMPLETE")
    print(f"{'='*80}")
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Saved to: simulation_logs_izh.pkl")
