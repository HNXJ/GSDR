#!/usr/bin/env python
"""
Conductance sweep analysis: measure firing rate vs conductance scaling.

Parameters swept: conductance scales [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
Simulation duration: 1500 ms (baseline 0-500ms, stimulus 600-1200ms, post-stimulus 1200-1500ms)
No optimization - just measure baseline firing rates.
"""

import sys
sys.path.insert(0, 'src')

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import time
from jbiophysic.models.reduced.net_eig_izh import net_eig_izh_jax
from poisson_input import generate_poisson_spikes_numpy, poisson_input_current


def run_conductance_sweep(n_steps: int = 1500, seed: int = 42):
    """
    Run conductance sweep: measure firing rate across conductance scales.

    Args:
        n_steps: simulation duration in 0.1ms units (1500 = 1500ms)
        seed: random seed
    """

    print("=" * 80)
    print("CONDUCTANCE SWEEP ANALYSIS")
    print("=" * 80)

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
        "t_ms": 1500.0,  # Extended to 1500 ms for stimulus response
        "seed": seed,
        "allow_autapses": False,
    }

    print(f"\nNetwork config: {net_config['n_neurons']} neurons, {net_config['t_ms']:.0f}ms simulation")
    print("\nBuilding network...")
    result = net_eig_izh_jax(net_config)
    params_base = result["params"]
    simulate_fn = result["simulate"]
    project_params = result["project_params"]
    trainable_masks = result["trainable_masks"]

    # Base conductance (50x scaled as in optimization)
    base_scale = 50.0

    # Generate Poisson inputs for extended duration
    dt_ms = net_config["dt_ms"]
    T = int(np.round(net_config["t_ms"] / dt_ms))
    N = net_config["n_neurons"]

    print(f"\nGenerating Poisson inputs for {T} timesteps ({net_config['t_ms']:.0f}ms)...")
    poisson_spikes = np.zeros((T, N), dtype=np.float32)
    poisson_amplitude = 20.0
    for nid in range(N):
        spikes = generate_poisson_spikes_numpy(
            net_config["t_ms"], dt_ms, rate_hz=1.0, min_isi_ms=200.0, seed=seed + nid
        )
        poisson_current = poisson_input_current(spikes, amplitude=poisson_amplitude, dt_ms=dt_ms)
        poisson_spikes[:, nid] = poisson_current

    # Add tonic baseline input
    tonic_input = np.zeros((T, N), dtype=np.float32)
    n_e = int(0.7 * N)
    tonic_input[:, :n_e] = 200.0  # E neurons
    tonic_input[:, n_e:] = 100.0  # I neurons

    poisson_input_array = jnp.asarray(poisson_spikes + tonic_input)

    # Conductance scales to test
    g_scales = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]

    print(f"\nConductance scales to test: {g_scales}")
    print(f"Base scale: {base_scale}x")
    print("\nRunning simulations...")
    print("=" * 80)
    print(f"{'Scale':>6} | {'Total FR':>9} | {'Baseline':>9} | {'Stimulus':>9} | {'Post-Stim':>9}")
    print("-" * 80)

    results = {
        'scales': g_scales,
        'total_firing_rates': [],
        'baseline_firing_rates': [],
        'stimulus_firing_rates': [],
        'post_stimulus_firing_rates': [],
        'spike_traces': {},
    }

    for scale in g_scales:
        # Scale all conductances
        params = {
            'g_ampa': params_base['g_ampa'] * base_scale * scale,
            'g_nmda': params_base['g_nmda'] * base_scale * scale,
            'g_gaba_a': params_base['g_gaba_a'] * base_scale * scale,
            'a': jnp.ones_like(params_base['a']) * 0.1,  # Fast-spiking
            'b': params_base['b'],
            'c': params_base['c'],
            'd': params_base['d'],
        }

        # Run simulation
        t0 = time.time()
        sim_result = simulate_fn(params, input_current=poisson_input_array)
        sim_time = time.time() - t0

        if sim_result['status'] != 'PASS':
            print(f" {scale:5.1f}x | {'ERROR':>9} | {'ERROR':>9} | {'ERROR':>9} | {'ERROR':>9}")
            continue

        spike_trace = sim_result['spike_trace']  # (T, N)
        results['spike_traces'][scale] = spike_trace

        # Calculate firing rates for different periods
        # Baseline: 0-500ms (indices 0-5000)
        baseline_start, baseline_end = 0, int(500 / dt_ms)
        baseline_spikes = jnp.sum(spike_trace[baseline_start:baseline_end, :])
        baseline_fr = baseline_spikes / (baseline_end - baseline_start) * (1000.0 / dt_ms)

        # Stimulus: 600-1200ms (indices 6000-12000)
        stim_start, stim_end = int(600 / dt_ms), int(1200 / dt_ms)
        stim_spikes = jnp.sum(spike_trace[stim_start:stim_end, :])
        stim_fr = stim_spikes / (stim_end - stim_start) * (1000.0 / dt_ms)

        # Post-stimulus: 1200-1500ms (indices 12000-15000)
        post_start, post_end = int(1200 / dt_ms), T
        post_spikes = jnp.sum(spike_trace[post_start:post_end, :])
        post_fr = post_spikes / (post_end - post_start) * (1000.0 / dt_ms)

        # Total firing rate
        total_spikes = jnp.sum(spike_trace)
        total_fr = total_spikes / T * (1000.0 / dt_ms)

        results['total_firing_rates'].append(float(total_fr))
        results['baseline_firing_rates'].append(float(baseline_fr))
        results['stimulus_firing_rates'].append(float(stim_fr))
        results['post_stimulus_firing_rates'].append(float(post_fr))

        print(f" {scale:5.1f}x | {total_fr:8.2f} Hz | {baseline_fr:8.2f} Hz | {stim_fr:8.2f} Hz | {post_fr:8.2f} Hz")

    print("=" * 80)
    return results


def plot_conductance_analysis(results):
    """Create comprehensive conductance sweep visualization."""
    scales = results['scales']
    total_fr = results['total_firing_rates']
    baseline_fr = results['baseline_firing_rates']
    stim_fr = results['stimulus_firing_rates']
    post_fr = results['post_stimulus_firing_rates']

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    # Plot 1: Firing rate vs conductance scale
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(scales, total_fr, 'o-', linewidth=2.5, markersize=8, label='Total', color='#667eea')
    ax1.plot(scales, baseline_fr, 's-', linewidth=2.5, markersize=8, label='Baseline (0-500ms)', color='#764ba2')
    ax1.plot(scales, stim_fr, '^-', linewidth=2.5, markersize=8, label='Stimulus (600-1200ms)', color='#f093fb')
    ax1.plot(scales, post_fr, 'd-', linewidth=2.5, markersize=8, label='Post-Stim (1200-1500ms)', color='#4facfe')
    ax1.set_xlabel('Conductance Scale Factor', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Firing Rate (Hz)', fontsize=12, fontweight='bold')
    ax1.set_title('Firing Rate vs Conductance Scale', fontsize=13, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    ax1.set_yscale('log')

    # Plot 2: Spike count per period
    ax2 = fig.add_subplot(gs[0, 1])
    width = 0.2
    x_pos = np.arange(len(scales))

    baseline_counts = [f * 5000.0 / 10000.0 for f in baseline_fr]  # 500ms duration
    stim_counts = [f * 6000.0 / 10000.0 for f in stim_fr]  # 600ms duration
    post_counts = [f * 3000.0 / 10000.0 for f in post_fr]  # 300ms duration

    ax2.bar(x_pos - width, baseline_counts, width, label='Baseline', color='#764ba2', alpha=0.8)
    ax2.bar(x_pos, stim_counts, width, label='Stimulus', color='#f093fb', alpha=0.8)
    ax2.bar(x_pos + width, post_counts, width, label='Post-Stim', color='#4facfe', alpha=0.8)

    ax2.set_xlabel('Conductance Scale Factor', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Total Spikes', fontsize=12, fontweight='bold')
    ax2.set_title('Total Spike Count by Period', fontsize=13, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f'{s:.1f}x' for s in scales])
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    # Plot 3: Stimulus response (ratio)
    ax3 = fig.add_subplot(gs[1, 0])
    stim_response = np.array(stim_fr) / (np.array(baseline_fr) + 1e-6)  # Avoid division by zero
    ax3.plot(scales, stim_response, 'o-', linewidth=2.5, markersize=8, color='#f093fb')
    ax3.axhline(y=1.0, color='k', linestyle='--', alpha=0.3, label='No change')
    ax3.set_xlabel('Conductance Scale Factor', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Stimulus Response Ratio', fontsize=12, fontweight='bold')
    ax3.set_title('Stimulus-Induced Firing Rate Increase', fontsize=13, fontweight='bold')
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log')

    # Plot 4: Summary statistics table
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')

    table_data = []
    table_data.append(['Scale', 'Total FR', 'Baseline', 'Stimulus', 'Post-Stim', 'Stim/Base'])
    for i, scale in enumerate(scales):
        ratio = stim_fr[i] / (baseline_fr[i] + 1e-6)
        table_data.append([
            f'{scale:.1f}x',
            f'{total_fr[i]:.2f}',
            f'{baseline_fr[i]:.2f}',
            f'{stim_fr[i]:.2f}',
            f'{post_fr[i]:.2f}',
            f'{ratio:.2f}',
        ])

    table = ax4.table(cellText=table_data, cellLoc='center', loc='center',
                      colWidths=[0.12, 0.14, 0.14, 0.14, 0.14, 0.14])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Header styling
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#667eea')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(table_data)):
        color = '#f0f0f0' if i % 2 == 0 else 'white'
        for j in range(len(table_data[0])):
            table[(i, j)].set_facecolor(color)

    ax4.set_title('Summary Statistics (All Firing Rates in Hz)', fontsize=13, fontweight='bold', pad=20)

    plt.suptitle('Conductance Sweep Analysis: Izhikevich Network', fontsize=16, fontweight='bold', y=0.995)

    return fig


def plot_spike_rasters(results, max_neurons=50):
    """Plot spike rasters for selected conductance scales."""
    scales = results['scales']
    spike_traces = results['spike_traces']

    # Select scales to visualize (first, middle, last)
    viz_indices = [0, len(scales)//2, -1]
    viz_scales = [scales[i] for i in viz_indices]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, scale in enumerate(viz_scales):
        if scale not in spike_traces:
            continue

        spike_trace = spike_traces[scale]
        spike_times, neuron_ids = np.where(spike_trace[:, :max_neurons] > 0)

        ax = axes[idx]
        ax.scatter(spike_times * 0.1, neuron_ids, s=5, alpha=0.6, color='#667eea')

        # Add stimulus period marker
        ax.axvline(x=600, color='red', linestyle='--', alpha=0.3, linewidth=2, label='Stimulus start')
        ax.axvline(x=1200, color='orange', linestyle='--', alpha=0.3, linewidth=2, label='Stimulus end')

        ax.set_xlabel('Time (ms)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Neuron ID', fontsize=11, fontweight='bold')
        ax.set_title(f'Spike Raster (Scale {scale:.1f}x)', fontsize=12, fontweight='bold')
        ax.set_xlim(0, 1500)
        ax.set_ylim(-1, max_neurons)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)

    plt.suptitle('Network Spike Rasters Across Conductance Scales', fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig


def generate_html_report(results, output_file='conductance_sweep_report.html'):
    """Generate HTML report with all visualizations."""
    import base64
    from io import BytesIO

    print(f"\n{'='*80}")
    print("GENERATING HTML REPORT")
    print('='*80)

    # Create visualizations
    print("Creating conductance analysis plot...")
    fig1 = plot_conductance_analysis(results)

    print("Creating spike raster plots...")
    fig2 = plot_spike_rasters(results)

    # Convert to base64
    def fig_to_base64(fig):
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode()
        plt.close(fig)
        return img_str

    img1 = fig_to_base64(fig1)
    img2 = fig_to_base64(fig2)

    # Generate HTML
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Conductance Sweep Analysis Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            color: #333;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }}
        .header h1 {{
            margin: 0;
            font-size: 32px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}
        .header p {{
            margin: 10px 0 0 0;
            font-size: 14px;
            opacity: 0.9;
        }}
        .section {{
            background: white;
            padding: 25px;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .section h2 {{
            color: #667eea;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
            margin-top: 0;
        }}
        .figure {{
            text-align: center;
            margin: 20px 0;
        }}
        .figure img {{
            max-width: 100%;
            height: auto;
            border-radius: 5px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .figure p {{
            color: #666;
            font-size: 12px;
            margin-top: 10px;
            font-style: italic;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .stat-box {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        }}
        .stat-box .value {{
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        .stat-box .label {{
            font-size: 12px;
            opacity: 0.9;
        }}
        .footer {{
            text-align: center;
            color: #666;
            font-size: 12px;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ccc;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        th, td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #667eea;
            color: white;
            font-weight: bold;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔬 Conductance Sweep Analysis Report</h1>
        <p>Izhikevich Spiking Network - Firing Rate vs Conductance Scaling</p>
    </div>

    <div class="section">
        <h2>📊 Analysis Summary</h2>
        <div class="stats">
            <div class="stat-box">
                <div class="value">{len(results['scales'])}</div>
                <div class="label">Conductance Scales Tested</div>
            </div>
            <div class="stat-box">
                <div class="value">{results['scales'][0]:.1f}x - {results['scales'][-1]:.1f}x</div>
                <div class="label">Conductance Range</div>
            </div>
            <div class="stat-box">
                <div class="value">{results['total_firing_rates'][0]:.2f} - {max(results['total_firing_rates']):.2f}</div>
                <div class="label">Firing Rate Range (Hz)</div>
            </div>
            <div class="stat-box">
                <div class="value">1500</div>
                <div class="label">Simulation Duration (ms)</div>
            </div>
        </div>
    </div>

    <div class="section">
        <h2>📈 Conductance Analysis</h2>
        <div class="figure">
            <img src="data:image/png;base64,{img1}" alt="Conductance Analysis">
            <p>Firing rates and spike counts across different conductance scaling factors. Top-left shows log-log relationship. Top-right shows spike counts by period. Bottom-left shows stimulus response ratio. Bottom-right shows summary statistics table.</p>
        </div>
    </div>

    <div class="section">
        <h2>🔷 Spike Rasters</h2>
        <div class="figure">
            <img src="data:image/png;base64,{img2}" alt="Spike Rasters">
            <p>Network spike rasters for low (1.0x), medium ({results['scales'][len(results['scales'])//2]:.1f}x), and high (32.0x) conductance scales. Red dashed line marks stimulus start (600ms), orange marks stimulus end (1200ms).</p>
        </div>
    </div>

    <div class="section">
        <h2>📋 Detailed Results</h2>
        <table>
            <tr>
                <th>Conductance Scale</th>
                <th>Total FR (Hz)</th>
                <th>Baseline FR (Hz)</th>
                <th>Stimulus FR (Hz)</th>
                <th>Post-Stim FR (Hz)</th>
                <th>Stimulus Response (Ratio)</th>
            </tr>
"""

    for i, scale in enumerate(results['scales']):
        ratio = results['stimulus_firing_rates'][i] / (results['baseline_firing_rates'][i] + 1e-6)
        html_content += f"""            <tr>
                <td>{scale:.1f}x</td>
                <td>{results['total_firing_rates'][i]:.2f}</td>
                <td>{results['baseline_firing_rates'][i]:.2f}</td>
                <td>{results['stimulus_firing_rates'][i]:.2f}</td>
                <td>{results['post_stimulus_firing_rates'][i]:.2f}</td>
                <td>{ratio:.2f}</td>
            </tr>
"""

    html_content += """        </table>
    </div>

    <div class="section">
        <h2>📝 Configuration</h2>
        <ul>
            <li><strong>Network Size:</strong> 89 neurons (70% excitatory, 30% inhibitory)</li>
            <li><strong>Connectivity:</strong> Fully dense (all-to-all synapses)</li>
            <li><strong>Neuron Model:</strong> Izhikevich with fast-spiking parameters (a=0.1)</li>
            <li><strong>Input:</strong> Per-neuron Poisson noise (1 Hz, 200ms min ISI) + Tonic baseline</li>
            <li><strong>Simulation Duration:</strong> 1500 ms total</li>
            <li><strong>Base Conductance Scale:</strong> 50x (before sweep multiplication)</li>
            <li><strong>Simulation Resolution:</strong> 0.1 ms (10 kHz effective sampling)</li>
        </ul>
    </div>

    <div class="footer">
        <p>Report generated: Izhikevich Spiking Network Conductance Sweep Analysis</p>
        <p>Baseline: 0-500ms | Stimulus: 600-1200ms | Post-Stimulus: 1200-1500ms</p>
    </div>
</body>
</html>"""

    with open(output_file, 'w') as f:
        f.write(html_content)

    print(f"\n✅ Report saved to {output_file}")
    return output_file


if __name__ == "__main__":
    print("\n" + "="*80)
    print("IZHIKEVICH NETWORK - CONDUCTANCE SWEEP ANALYSIS")
    print("="*80)

    # Run conductance sweep
    results = run_conductance_sweep(n_steps=15000, seed=42)  # 1500ms

    # Generate report
    output_file = generate_html_report(results)

    print("\n" + "="*80)
    print("✅ CONDUCTANCE SWEEP ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {output_file}")
