#!/usr/bin/env python
"""
Generate comprehensive HTML report with all simulation visualizations:
- Spike rasters
- PSDs (Power Spectral Density)
- Network activity traces
- Loss curves over optimization
- Firing rate statistics
- Optimizer comparisons
"""

import sys
sys.path.insert(0, 'src')

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime
import base64
from io import BytesIO

from jbiophysic.models.reduced.net_eig_izh import net_eig_izh_jax
from spectral_loss import combined_spectral_loss, make_target_psds, jax_welch_psd
from poisson_input import generate_poisson_spikes_numpy, poisson_input_current

# Configure matplotlib
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

def fig_to_base64(fig):
    """Convert matplotlib figure to base64 for HTML embedding."""
    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    plt.close(fig)
    return image_base64

class SimpleRandomSearch:
    """Gradient-free random search optimizer."""
    def __init__(self, learning_rate=0.05, exploration_amplitude=2.0):
        self.learning_rate = learning_rate
        self.exploration_amplitude = exploration_amplitude
        self.step_count = 0

    def update(self, params, loss, rng):
        self.step_count += 1
        param_leaves, treedef = jax.tree.flatten(params)
        subkeys = jax.random.split(rng, len(param_leaves))
        deltas = [self.exploration_amplitude * jax.random.normal(k, p.shape) for k, p in zip(subkeys, param_leaves)]
        new_leaves = [p - self.learning_rate * d for p, d in zip(param_leaves, deltas)]
        return jax.tree.unflatten(treedef, new_leaves)

def run_optimization(opt_name, opt_config, n_steps=100, seed=42):
    """Run complete optimization pipeline."""
    print(f"\n{'='*80}")
    print(f"Running {opt_name} optimization ({n_steps} steps)...")
    print('='*80)

    # Load data
    spike_data = jnp.asarray(np.load('/Users/hamednejat/claude-gamma-labyrinth/workspace/computational/selected_unit_spiking.npy'), dtype=jnp.float32)
    baseline_psd, stim_psd, target_freqs = make_target_psds(
        spike_data, fs_hz=1000.0, target_freqs=jnp.linspace(5.0, 100.0, 100),
        baseline_time_slice=slice(0, 500), stim_time_slice=slice(600, 1200),
        frame_ms=500.0, hop_ms=50.0,
    )

    # Build network
    net_config = {
        'n_neurons': 89, 'composition': [[0.7, 0.2, 0.05, 0.05], [0.6, 0.25, 0.1, 0.05], [0.65, 0.2, 0.1, 0.05]],
        'layer_edges_um': [0, 400, 700, 1000], 'granular_layer': 2,
        'dt_ms': 0.1, 't_ms': 500.0, 'seed': seed, 'allow_autapses': False,
    }

    result = net_eig_izh_jax(net_config)
    params = result['params']
    simulate_fn = result['simulate']
    project_params = result['project_params']
    trainable_masks = result['trainable_masks']

    # Scale parameters
    params = {
        'g_ampa': params['g_ampa'] * 50.0,
        'g_nmda': params['g_nmda'] * 50.0,
        'g_gaba_a': params['g_gaba_a'] * 50.0,
        'a': jnp.ones_like(params['a']) * 0.1,
        'b': params['b'], 'c': params['c'], 'd': params['d'],
    }

    # Generate inputs
    T = 5000
    N = 89
    poisson_spikes = np.zeros((T, N), dtype=np.float32)
    for nid in range(N):
        spikes = generate_poisson_spikes_numpy(500.0, 0.1, rate_hz=1.0, min_isi_ms=200.0, seed=seed + nid)
        poisson_current = poisson_input_current(spikes, amplitude=20.0, dt_ms=0.1)
        poisson_spikes[:, nid] = poisson_current

    tonic_input = np.zeros((T, N), dtype=np.float32)
    tonic_input[:, :int(0.7*N)] = 200.0
    tonic_input[:, int(0.7*N):] = 100.0
    poisson_input_array = jnp.asarray(poisson_spikes + tonic_input)

    # Loss function
    def loss_fn(params):
        sim_result = simulate_fn(params, input_current=poisson_input_array)
        if sim_result['status'] != 'PASS':
            return jnp.array(1e6, dtype=jnp.float32)
        spike_trace = sim_result['spike_trace']
        population_signal = jnp.mean(spike_trace, axis=1)
        spectral_loss = combined_spectral_loss(
            population_signal, target_psd=stim_psd, fs_hz=10000.0 / T * 5000,
            target_freqs=target_freqs, frame_ms=50.0, hop_ms=5.0,
            shape_weight=1.0, cosine_weight=0.2, bandpower_weight=0.1,
        )
        mean_spike_prob = jnp.mean(spike_trace)
        mean_rate = mean_spike_prob * (1000.0 / 0.1)
        fr_penalty = jnp.maximum(0.0, 10.0 - mean_rate) ** 2
        return spectral_loss + 0.01 * fr_penalty

    # Initialize optimizer
    opt = SimpleRandomSearch(**opt_config)
    rng = jax.random.PRNGKey(seed)

    # Run optimization
    loss_history = []
    firing_rates = []
    spike_traces = []
    voltages = []

    for step in range(n_steps):
        loss = loss_fn(params)
        loss_history.append(float(loss))

        sim_result = simulate_fn(params, input_current=poisson_input_array)
        if sim_result['status'] == 'PASS':
            spike_trace = sim_result['spike_trace']
            v_trace = sim_result['v_trace']

            if step % 20 == 0 or step == n_steps - 1:
                spike_traces.append(spike_trace)
                voltages.append(v_trace)

            pop_rate = jnp.mean(spike_trace) * 1000.0 / 0.1
            firing_rates.append(float(pop_rate))
        else:
            firing_rates.append(0.0)

        rng, sub_rng = jax.random.split(rng)
        params = opt.update(params, loss, sub_rng)
        params = project_params(params, params, trainable_masks)

        if (step + 1) % 20 == 0:
            print(f"  Step {step+1:3d}: loss={loss:7.4f}, firing_rate={firing_rates[-1]:6.2f} Hz")

    # Get final simulation for visualization
    final_sim = simulate_fn(params, input_current=poisson_input_array)
    final_spikes = final_sim['spike_trace']
    final_voltages = final_sim['v_trace']
    final_pop_signal = jnp.mean(final_spikes, axis=1)

    # Compute final PSD
    target_freqs = jnp.linspace(5.0, 100.0, 100)
    fs = 10000.0  # 10 kHz simulation sampling rate
    psd = jax_welch_psd(final_pop_signal, fs_hz=fs, target_freqs=target_freqs, frame_ms=500.0, hop_ms=50.0)
    freqs = target_freqs

    results = {
        'name': opt_name,
        'loss_history': loss_history,
        'firing_rates': firing_rates,
        'final_spikes': final_spikes,
        'final_voltages': final_voltages,
        'final_psd': (freqs, psd),
        'target_psd': (target_freqs, stim_psd),
        'spike_traces': spike_traces,
        'voltages': voltages,
    }

    return results

def plot_loss_curves(results_list):
    """Plot loss curves for all optimizers."""
    fig, ax = plt.subplots(figsize=(12, 6))

    for results in results_list:
        steps = np.arange(len(results['loss_history']))
        ax.plot(steps, results['loss_history'], linewidth=2, label=results['name'], marker='o', markersize=3, markevery=10)

    ax.set_xlabel('Optimization Step', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Loss Convergence Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    return fig_to_base64(fig)

def plot_firing_rates(results_list):
    """Plot firing rates over optimization."""
    fig, ax = plt.subplots(figsize=(12, 6))

    for results in results_list:
        steps = np.arange(len(results['firing_rates']))
        ax.plot(steps, results['firing_rates'], linewidth=2, label=results['name'], marker='s', markersize=3, markevery=10)

    ax.set_xlabel('Optimization Step', fontsize=12)
    ax.set_ylabel('Population Firing Rate (Hz)', fontsize=12)
    ax.set_title('Network Activity During Optimization', fontsize=14, fontweight='bold')
    ax.axhline(y=10.0, color='r', linestyle='--', linewidth=1, label='Target (10 Hz)', alpha=0.7)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    return fig_to_base64(fig)

def plot_spike_raster(spike_trace, title="Spike Raster", max_neurons=50):
    """Plot spike raster."""
    fig, ax = plt.subplots(figsize=(14, 8))

    n_neurons = min(max_neurons, spike_trace.shape[1])
    spike_times, neuron_ids = np.where(spike_trace[:, :n_neurons])

    ax.scatter(spike_times * 0.1, neuron_ids, s=1, alpha=0.6, color='black')
    ax.set_xlabel('Time (ms)', fontsize=12)
    ax.set_ylabel('Neuron ID', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim(-1, n_neurons)

    return fig_to_base64(fig)

def plot_psd_comparison(freqs, psd, target_freqs, target_psd, title="PSD Comparison"):
    """Plot PSD comparison."""
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.semilogy(freqs, psd, linewidth=2, label='Fitted Network', marker='o', markersize=4, markevery=10)
    ax.semilogy(target_freqs, target_psd, linewidth=2, label='Empirical Data', marker='s', markersize=4, markevery=10)

    ax.set_xlabel('Frequency (Hz)', fontsize=12)
    ax.set_ylabel('Power (log scale)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which='both')

    return fig_to_base64(fig)

def plot_voltage_trace(v_trace, title="Population Voltage Trace"):
    """Plot population voltage."""
    fig, ax = plt.subplots(figsize=(14, 6))

    pop_voltage = np.mean(v_trace, axis=1)
    time = np.arange(len(pop_voltage)) * 0.1

    ax.plot(time, pop_voltage, linewidth=1, color='blue', alpha=0.7)
    ax.fill_between(time, pop_voltage.min(), pop_voltage, alpha=0.3, color='blue')

    ax.set_xlabel('Time (ms)', fontsize=12)
    ax.set_ylabel('Population Voltage (mV)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    return fig_to_base64(fig)

def generate_html_report(results_list, output_file='optimization_report.html'):
    """Generate comprehensive HTML report."""

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Spiking Network Optimization Report</title>
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
            .stats {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin-bottom: 20px;
            }}
            .stat-box {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 15px;
                border-radius: 8px;
                text-align: center;
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
            .figure {{
                margin: 20px 0;
                text-align: center;
            }}
            .figure img {{
                max-width: 100%;
                border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }}
            .figure p {{
                margin: 10px 0 0 0;
                font-size: 12px;
                color: #666;
                font-style: italic;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 15px 0;
            }}
            table th {{
                background: #667eea;
                color: white;
                padding: 12px;
                text-align: left;
            }}
            table td {{
                padding: 10px;
                border-bottom: 1px solid #eee;
            }}
            table tr:hover {{
                background: #f9f9f9;
            }}
            .footer {{
                text-align: center;
                padding: 20px;
                color: #999;
                font-size: 12px;
                margin-top: 40px;
                border-top: 1px solid #ddd;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🧠 Spiking Network Optimization Report</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>

        <div class="section">
            <h2>📊 Optimization Summary</h2>
            <table>
                <tr>
                    <th>Optimizer</th>
                    <th>Initial Loss</th>
                    <th>Final Loss</th>
                    <th>Improvement</th>
                    <th>Steps</th>
                    <th>Final Firing Rate</th>
                </tr>
    """

    for results in results_list:
        initial_loss = results['loss_history'][0]
        final_loss = results['loss_history'][-1]
        improvement = (initial_loss - final_loss) / initial_loss * 100
        final_fr = results['firing_rates'][-1]

        html_content += f"""
                <tr>
                    <td><strong>{results['name']}</strong></td>
                    <td>{initial_loss:.4f}</td>
                    <td>{final_loss:.4f}</td>
                    <td>{improvement:.1f}%</td>
                    <td>{len(results['loss_history'])}</td>
                    <td>{final_fr:.2f} Hz</td>
                </tr>
        """

    html_content += """
            </table>
        </div>

        <div class="section">
            <h2>📈 Loss Convergence</h2>
            <div class="figure">
    """
    html_content += f'<img src="data:image/png;base64,{plot_loss_curves(results_list)}" alt="Loss Curves">'
    html_content += """
                <p>Loss curves for all optimizers showing convergence dynamics</p>
            </div>
        </div>

        <div class="section">
            <h2>🔥 Network Activity</h2>
            <div class="figure">
    """
    html_content += f'<img src="data:image/png;base64,{plot_firing_rates(results_list)}" alt="Firing Rates">'
    html_content += """
                <p>Population firing rate throughout optimization (target: 10 Hz)</p>
            </div>
        </div>
    """

    for i, results in enumerate(results_list):
        html_content += f"""
        <div class="section">
            <h2>🔬 {results['name']} - Detailed Results</h2>

            <div class="stats">
                <div class="stat-box">
                    <div class="value">{results['loss_history'][0]:.2f}</div>
                    <div class="label">Initial Loss</div>
                </div>
                <div class="stat-box">
                    <div class="value">{results['loss_history'][-1]:.2f}</div>
                    <div class="label">Final Loss</div>
                </div>
                <div class="stat-box">
                    <div class="value">{(results['loss_history'][0] - results['loss_history'][-1]) / results['loss_history'][0] * 100:.1f}%</div>
                    <div class="label">Improvement</div>
                </div>
                <div class="stat-box">
                    <div class="value">{np.mean(results['firing_rates']):.2f}</div>
                    <div class="label">Avg Firing Rate (Hz)</div>
                </div>
            </div>

            <h3>Spike Raster (Initial vs Final)</h3>
            <div class="figure">
        """

        # Initial spikes (from first recorded spike trace)
        if results['spike_traces']:
            initial_img = plot_spike_raster(results["spike_traces"][0], "Initial Network Activity (Step 0)", 50)
            html_content += f'<img src="data:image/png;base64,{initial_img}" alt="Initial Raster">'

        # Final spikes
        final_step = len(results['loss_history'])
        final_img = plot_spike_raster(results["final_spikes"], f"Final Network Activity (Step {final_step})", 50)
        html_content += f'<img src="data:image/png;base64,{final_img}" alt="Final Raster">'

        html_content += """
                <p>Spike raster showing network spiking pattern</p>
            </div>

            <h3>Population Voltage Trace</h3>
            <div class="figure">
        """

        voltage_title = f"Final Population Voltage - {results['name']}"
        voltage_img = plot_voltage_trace(results["final_voltages"], voltage_title)
        html_content += f'<img src="data:image/png;base64,{voltage_img}" alt="Voltage Trace">'

        html_content += """
                <p>Population-averaged voltage dynamics</p>
            </div>

            <h3>Power Spectral Density (PSD)</h3>
            <div class="figure">
        """

        freqs, psd = results['final_psd']
        target_freqs, target_psd = results['target_psd']
        psd_title = f"PSD Comparison - {results['name']}"
        psd_img = plot_psd_comparison(freqs, psd, target_freqs, target_psd, psd_title)
        html_content += f'<img src="data:image/png;base64,{psd_img}" alt="PSD">'

        html_content += """
                <p>Comparison of fitted network PSD vs empirical data</p>
            </div>
        </div>
        """

    html_content += f"""
        <div class="footer">
            <p>Report generated on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}</p>
            <p>Izhikevich Spiking Network Optimization Pipeline</p>
        </div>
    </body>
    </html>
    """

    with open(output_file, 'w') as f:
        f.write(html_content)

    print(f"\n✅ Report saved to {output_file}")

if __name__ == "__main__":
    print("="*80)
    print("SPIKING NETWORK OPTIMIZATION - COMPREHENSIVE REPORT GENERATION")
    print("="*80)

    # Run optimizations with different configurations
    configs = [
        ("SimpleRandomSearch (lr=0.05)", {'learning_rate': 0.05, 'exploration_amplitude': 2.0}),
        ("SimpleRandomSearch (lr=0.03)", {'learning_rate': 0.03, 'exploration_amplitude': 2.0}),
    ]

    results_list = []
    for name, config in configs:
        results = run_optimization(name, config, n_steps=100, seed=42)
        results_list.append(results)

    print("\n" + "="*80)
    print("GENERATING HTML REPORT...")
    print("="*80)

    generate_html_report(results_list, 'optimization_report.html')

    print("\n✅ ALL TASKS COMPLETED!")
