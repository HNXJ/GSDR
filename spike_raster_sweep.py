#!/usr/bin/env python
"""
Spike raster sweep: Find noise + drive combinations for ~10 spikes/neuron/1000ms.

Generates 8 example spike rasters with varying Poisson noise and tonic drive.
Creates HTML report with all rasters visualized.
"""

import sys
sys.path.insert(0, 'src')

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import base64
from io import BytesIO
from jbiophysic.models.reduced.net_eig_izh import net_eig_izh_jax
from poisson_input import generate_poisson_spikes_numpy, poisson_input_current


def run_single_simulation(poisson_amp, poisson_rate, tonic_drive_e, tonic_drive_i,
                         seed=42):
    """
    Run a single 1000ms simulation and return spike trace + firing rate.

    Args:
        poisson_amp: Poisson noise amplitude (mV/ms)
        poisson_rate: Poisson noise rate (Hz)
        tonic_drive_e: E neuron tonic input (mV/ms)
        tonic_drive_i: I neuron tonic input (mV/ms)
        seed: random seed

    Returns:
        spike_trace: (T, N) binary spike matrix
        firing_rate: average firing rate (Hz)
    """

    net_config = {
        "n_neurons": 89,
        "composition": [[0.7, 0.2, 0.05, 0.05], [0.6, 0.25, 0.1, 0.05], [0.65, 0.2, 0.1, 0.05]],
        "layer_edges_um": [0, 400, 700, 1000],
        "granular_layer": 2,
        "dt_ms": 0.1,
        "t_ms": 1000.0,  # 1000ms simulation
        "seed": seed,
        "allow_autapses": False,
    }

    result = net_eig_izh_jax(net_config)
    params = result["params"]
    simulate_fn = result["simulate"]

    # Base conductance scale
    base_conductance_scale = 50.0
    params = {
        'g_ampa': params['g_ampa'] * base_conductance_scale,
        'g_nmda': params['g_nmda'] * base_conductance_scale,
        'g_gaba_a': params['g_gaba_a'] * base_conductance_scale,
        'a': jnp.ones_like(params['a']) * 0.1,  # Fast-spiking
        'b': params['b'], 'c': params['c'], 'd': params['d'],
    }

    # Generate inputs
    dt_ms = 0.1
    T = int(np.round(1000.0 / dt_ms))  # 10000 timesteps
    N = 89

    # Poisson input
    poisson_spikes = np.zeros((T, N), dtype=np.float32)
    for nid in range(N):
        spikes = generate_poisson_spikes_numpy(1000.0, dt_ms, rate_hz=poisson_rate,
                                               min_isi_ms=200.0, seed=seed + nid)
        poisson_current = poisson_input_current(spikes, amplitude=poisson_amp, dt_ms=dt_ms)
        poisson_spikes[:, nid] = poisson_current

    # Tonic input
    tonic_input = np.zeros((T, N), dtype=np.float32)
    n_e = int(0.7 * N)
    tonic_input[:, :n_e] = tonic_drive_e
    tonic_input[:, n_e:] = tonic_drive_i

    poisson_input_array = jnp.asarray(poisson_spikes + tonic_input)

    # Simulate
    sim_result = simulate_fn(params, input_current=poisson_input_array)

    if sim_result['status'] == 'PASS':
        spike_trace = sim_result['spike_trace']
        total_spikes = int(jnp.sum(spike_trace))
        firing_rate = total_spikes / T * (1000.0 / dt_ms)  # Convert to Hz
        return spike_trace, firing_rate
    else:
        return None, 0.0


def find_optimal_configurations():
    """
    Search for parameter combinations that yield ~10 spikes/neuron/1000ms (10 Hz).

    Returns list of (poisson_amp, poisson_rate, tonic_e, tonic_i, firing_rate, spike_trace).
    """

    print("\n" + "="*80)
    print("SEARCHING FOR OPTIMAL NOISE + DRIVE CONFIGURATIONS")
    print("="*80)
    print("\nTarget: 10 Hz average firing rate (10 spikes/neuron/1000ms)")
    print("\nParameter ranges:")
    print("  Poisson amplitude: 5-25 mV/ms")
    print("  Poisson rate: 0.5-3.0 Hz")
    print("  Tonic drive E: 50-200 mV/ms")
    print("  Tonic drive I: 25-100 mV/ms")

    # Coarse search first
    configs = []
    target_fr = 10.0

    poisson_amps = np.arange(5, 26, 5)  # [5, 10, 15, 20, 25]
    poisson_rates = np.arange(0.5, 3.5, 1.0)  # [0.5, 1.5, 2.5, 3.5]
    tonic_drives_e = np.arange(50, 201, 50)  # [50, 100, 150, 200]
    tonic_drives_i = np.arange(25, 101, 25)  # [25, 50, 75, 100]

    total_sims = len(poisson_amps) * len(poisson_rates) * len(tonic_drives_e) * len(tonic_drives_i)
    print(f"\nTotal simulations to run: {total_sims}")
    print("Running simulations (this may take several minutes)...\n")

    count = 0
    for p_amp in poisson_amps:
        for p_rate in poisson_rates:
            for t_drive_e in tonic_drives_e:
                for t_drive_i in tonic_drives_i:
                    count += 1
                    spike_trace, fr = run_single_simulation(p_amp, p_rate, t_drive_e, t_drive_i)

                    if spike_trace is not None:
                        configs.append({
                            'poisson_amp': p_amp,
                            'poisson_rate': p_rate,
                            'tonic_e': t_drive_e,
                            'tonic_i': t_drive_i,
                            'firing_rate': fr,
                            'spike_trace': spike_trace,
                        })

                    if count % 20 == 0:
                        print(f"  {count}/{total_sims} simulations completed")

    # Sort by distance from target firing rate
    configs.sort(key=lambda x: abs(x['firing_rate'] - target_fr))

    print(f"\n✅ Found {len(configs)} valid configurations")
    print(f"\nTop 8 configurations closest to {target_fr} Hz:")
    print("-" * 80)
    print(f"{'#':>2} | {'Pois Amp':>8} | {'Pois Rate':>10} | {'Tonic E':>8} | {'Tonic I':>8} | {'FR (Hz)':>8}")
    print("-" * 80)

    for i, cfg in enumerate(configs[:8]):
        print(f"{i+1:2d} | {cfg['poisson_amp']:8.1f} | {cfg['poisson_rate']:10.1f} | "
              f"{cfg['tonic_e']:8.1f} | {cfg['tonic_i']:8.1f} | {cfg['firing_rate']:8.2f}")

    return configs[:8]


def plot_spike_raster(spike_trace, title, max_neurons=89):
    """Create a spike raster plot."""
    fig, ax = plt.subplots(figsize=(12, 6))

    spike_times, neuron_ids = np.where(spike_trace[:, :max_neurons] > 0)

    # Color code by neuron type (first 62 are E, rest are I)
    n_e = int(0.7 * 89)
    colors = ['#667eea' if nid < n_e else '#f093fb' for nid in neuron_ids]

    ax.scatter(spike_times * 0.1, neuron_ids, s=8, alpha=0.7, c=colors)

    ax.set_xlabel('Time (ms)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Neuron ID', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlim(0, 1000)
    ax.set_ylim(-1, max_neurons)

    # Legend
    e_patch = mpatches.Patch(color='#667eea', label='Excitatory')
    i_patch = mpatches.Patch(color='#f093fb', label='Inhibitory')
    ax.legend(handles=[e_patch, i_patch], loc='upper right', fontsize=9)

    ax.grid(True, alpha=0.2)

    return fig


def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string."""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=80, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return img_str


def generate_html_report(configs, output_file='spike_raster_examples.html'):
    """Generate HTML with 8 spike raster examples."""

    print(f"\nGenerating spike raster plots for 8 configurations...")

    # Create raster plots
    raster_images = []
    raster_info = []

    for i, cfg in enumerate(configs):
        fig = plot_spike_raster(
            cfg['spike_trace'],
            f"Config {i+1}: Pois={cfg['poisson_amp']:.0f} mV/ms, "
            f"Rate={cfg['poisson_rate']:.1f} Hz, "
            f"Drive={cfg['tonic_e']:.0f}/{cfg['tonic_i']:.0f} mV/ms "
            f"(FR={cfg['firing_rate']:.2f} Hz)"
        )
        img = fig_to_base64(fig)
        raster_images.append(img)
        raster_info.append(cfg)

    # Generate HTML
    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Spike Raster Examples - 1000ms Simulations</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }
        .container {
            max-width: 1600px;
            margin: 0 auto;
        }
        .header {
            background: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        }
        .header h1 {
            color: #667eea;
            margin-bottom: 10px;
            font-size: 28px;
        }
        .header p {
            color: #666;
            font-size: 14px;
            line-height: 1.6;
        }
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }
        .stat-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }
        .stat-box .value {
            font-size: 24px;
            font-weight: bold;
        }
        .stat-box .label {
            font-size: 12px;
            opacity: 0.9;
            margin-top: 5px;
        }
        .raster-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
            margin-bottom: 30px;
        }
        .raster-item {
            background: white;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            transition: transform 0.3s, box-shadow 0.3s;
        }
        .raster-item:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 40px rgba(0,0,0,0.15);
        }
        .raster-image {
            width: 100%;
            display: block;
            background: #f9f9f9;
        }
        .raster-info {
            padding: 15px;
            border-top: 2px solid #667eea;
        }
        .raster-info h3 {
            color: #667eea;
            font-size: 14px;
            margin-bottom: 8px;
        }
        .raster-params {
            font-size: 12px;
            color: #666;
            line-height: 1.8;
            font-family: 'Courier New', monospace;
        }
        .raster-params strong {
            color: #333;
        }
        .legend {
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        }
        .legend h3 {
            color: #667eea;
            margin-bottom: 15px;
            font-size: 16px;
        }
        .legend-items {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }
        .legend-item {
            padding: 15px;
            background: #f9f9f9;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }
        .legend-item h4 {
            color: #667eea;
            margin-bottom: 8px;
            font-size: 13px;
        }
        .legend-item p {
            color: #666;
            font-size: 12px;
            line-height: 1.6;
            margin: 5px 0;
        }
        .color-box {
            display: inline-block;
            width: 16px;
            height: 16px;
            margin-right: 8px;
            vertical-align: middle;
            border-radius: 2px;
        }
        .footer {
            background: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            color: #666;
            font-size: 12px;
            margin-top: 30px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔬 Spike Raster Examples - 1000ms Simulations</h1>
            <p>Izhikevich spiking network with varying Poisson noise and tonic drive levels.
               Each configuration targets ~10 spikes per neuron per 1000ms (10 Hz average firing rate).</p>
            <div class="stats">
                <div class="stat-box">
                    <div class="value">89</div>
                    <div class="label">Total Neurons</div>
                </div>
                <div class="stat-box">
                    <div class="value">62:27</div>
                    <div class="label">E:I Ratio</div>
                </div>
                <div class="stat-box">
                    <div class="value">1000</div>
                    <div class="label">Simulation Duration (ms)</div>
                </div>
                <div class="stat-box">
                    <div class="value">0.1</div>
                    <div class="label">Time Step (ms)</div>
                </div>
            </div>
        </div>

        <div class="legend">
            <h3>📋 Configuration Details & Parameters</h3>
            <div class="legend-items">
                <div class="legend-item">
                    <h4>Neuron Types</h4>
                    <p><span class="color-box" style="background: #667eea;"></span> Excitatory (E): 62 neurons</p>
                    <p><span class="color-box" style="background: #f093fb;"></span> Inhibitory (I): 27 neurons</p>
                </div>
                <div class="legend-item">
                    <h4>Network Properties</h4>
                    <p><strong>Model:</strong> Izhikevich spiking neurons</p>
                    <p><strong>Connectivity:</strong> Fully dense (all-to-all)</p>
                    <p><strong>Base Conductance Scale:</strong> 50x</p>
                </div>
                <div class="legend-item">
                    <h4>Input Parameters</h4>
                    <p><strong>Poisson Amplitude:</strong> Background noise strength (mV/ms)</p>
                    <p><strong>Poisson Rate:</strong> Spike frequency of noise (Hz)</p>
                    <p><strong>Tonic Drive E/I:</strong> Constant baseline input (mV/ms)</p>
                </div>
            </div>
        </div>

        <div class="raster-grid">
"""

    for i, (img, info) in enumerate(zip(raster_images, raster_info)):
        html += f"""
            <div class="raster-item">
                <img src="data:image/png;base64,{img}" alt="Raster {i+1}" class="raster-image">
                <div class="raster-info">
                    <h3>Configuration {i+1}</h3>
                    <div class="raster-params">
                        <p><strong>Poisson Amplitude:</strong> {info['poisson_amp']:.1f} mV/ms</p>
                        <p><strong>Poisson Rate:</strong> {info['poisson_rate']:.1f} Hz</p>
                        <p><strong>Tonic Drive (E):</strong> {info['tonic_e']:.1f} mV/ms</p>
                        <p><strong>Tonic Drive (I):</strong> {info['tonic_i']:.1f} mV/ms</p>
                        <p><strong>Firing Rate:</strong> {info['firing_rate']:.2f} Hz</p>
                        <p><strong>Total Spikes:</strong> {int(jnp.sum(info['spike_trace']))}</p>
                    </div>
                </div>
            </div>
"""

    html += """
        </div>

        <div class="footer">
            <p>🧠 Izhikevich Spiking Network Analysis | 8 Representative Configurations | Target: 10 Hz Average Firing Rate</p>
            <p>Simulation Parameters: dt=0.1ms, Duration=1000ms, Neurons=89 (62E+27I), Full Connectivity</p>
        </div>
    </div>
</body>
</html>"""

    with open(output_file, 'w') as f:
        f.write(html)

    print(f"✅ HTML report saved to {output_file}")
    return output_file


if __name__ == "__main__":
    print("\n" + "="*80)
    print("SPIKE RASTER SWEEP - FINDING OPTIMAL CONFIGURATIONS")
    print("="*80)

    # Find optimal configurations
    configs = find_optimal_configurations()

    # Generate HTML report
    output = generate_html_report(configs)

    print("\n" + "="*80)
    print("✅ SPIKE RASTER SWEEP COMPLETE")
    print("="*80)
    print(f"\n📊 Report: {output}")
    print("\nOpen in browser to view all 8 spike raster examples!")
