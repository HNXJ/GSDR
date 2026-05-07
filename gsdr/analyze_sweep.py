#!/usr/bin/env python
"""
Analyze parameter sweep results from simulation_logs_izh.pkl.

Generates:
1. 2D heatmaps for Poisson amplitude vs rate
2. Synaptic type response curves
3. HTML report with all visualizations
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import base64
from io import BytesIO


def load_logs(logs_path='simulation_logs_izh.pkl'):
    """Load simulation logs from pickle."""
    with open(logs_path, 'rb') as f:
        data = pickle.load(f)
    print(f"✅ Loaded logs from {logs_path}")
    return data


def analyze_poisson_sweep(log_data):
    """Create heatmap and analysis for Poisson sweep."""

    amplitudes = np.array(log_data['amplitudes'])
    rates = np.array(log_data['rates'])
    firing_rates = log_data['firing_rates']
    baseline_fr = log_data['baseline_fr']
    stim_fr = log_data['stim_fr']

    # Create 2D arrays for heatmap
    fr_matrix = np.zeros((len(amplitudes), len(rates)))
    baseline_matrix = np.zeros((len(amplitudes), len(rates)))
    stim_matrix = np.zeros((len(amplitudes), len(rates)))

    for i, amp in enumerate(amplitudes):
        for j, rate in enumerate(rates):
            key = (amp, rate)
            fr_matrix[i, j] = firing_rates.get(key, 0.0)
            baseline_matrix[i, j] = baseline_fr.get(key, 0.0)
            stim_matrix[i, j] = stim_fr.get(key, 0.0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Total firing rate
    im1 = axes[0].imshow(fr_matrix, cmap='viridis', aspect='auto', origin='lower')
    axes[0].set_xlabel('Poisson Rate (Hz)', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Poisson Amplitude (mV/ms)', fontsize=11, fontweight='bold')
    axes[0].set_title('Total Firing Rate (Hz)', fontsize=12, fontweight='bold')
    axes[0].set_xticks(range(len(rates)))
    axes[0].set_xticklabels([f'{r:.1f}' for r in rates], rotation=45)
    axes[0].set_yticks(range(len(amplitudes)))
    axes[0].set_yticklabels([f'{a:.0f}' for a in amplitudes])
    plt.colorbar(im1, ax=axes[0])

    # Baseline firing rate
    im2 = axes[1].imshow(baseline_matrix, cmap='plasma', aspect='auto', origin='lower')
    axes[1].set_xlabel('Poisson Rate (Hz)', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Poisson Amplitude (mV/ms)', fontsize=11, fontweight='bold')
    axes[1].set_title('Baseline Firing Rate (Hz)', fontsize=12, fontweight='bold')
    axes[1].set_xticks(range(len(rates)))
    axes[1].set_xticklabels([f'{r:.1f}' for r in rates], rotation=45)
    axes[1].set_yticks(range(len(amplitudes)))
    axes[1].set_yticklabels([f'{a:.0f}' for a in amplitudes])
    plt.colorbar(im2, ax=axes[1])

    # Stimulus firing rate
    im3 = axes[2].imshow(stim_matrix, cmap='coolwarm', aspect='auto', origin='lower')
    axes[2].set_xlabel('Poisson Rate (Hz)', fontsize=11, fontweight='bold')
    axes[2].set_ylabel('Poisson Amplitude (mV/ms)', fontsize=11, fontweight='bold')
    axes[2].set_title('Stimulus Firing Rate (Hz)', fontsize=12, fontweight='bold')
    axes[2].set_xticks(range(len(rates)))
    axes[2].set_xticklabels([f'{r:.1f}' for r in rates], rotation=45)
    axes[2].set_yticks(range(len(amplitudes)))
    axes[2].set_yticklabels([f'{a:.0f}' for a in amplitudes])
    plt.colorbar(im3, ax=axes[2])

    plt.suptitle('Poisson Noise Parameter Sweep: Firing Rates', fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig, (amplitudes, rates, fr_matrix, baseline_matrix, stim_matrix)


def analyze_synapse_sweep(log_data):
    """Create curves for synapse type sweep."""

    scales = np.array(log_data['scales'])
    synapse_types = log_data['synapse_types']
    firing_rates = log_data['firing_rates']
    baseline_fr = log_data['baseline_fr']
    stim_fr = log_data['stim_fr']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    colors = ['#667eea', '#764ba2', '#f093fb', '#4facfe']

    for idx, synapse_type in enumerate(synapse_types):
        ax = axes[idx]

        fr_list = []
        baseline_list = []
        stim_list = []

        for scale in scales:
            key = (synapse_type, scale)
            fr_list.append(firing_rates.get(key, 0.0))
            baseline_list.append(baseline_fr.get(key, 0.0))
            stim_list.append(stim_fr.get(key, 0.0))

        ax.plot(scales, fr_list, 'o-', linewidth=2.5, markersize=8,
                label='Total', color=colors[idx])
        ax.plot(scales, baseline_list, 's--', linewidth=2, markersize=6,
                label='Baseline', alpha=0.7, color=colors[idx])
        ax.plot(scales, stim_list, '^--', linewidth=2, markersize=6,
                label='Stimulus', alpha=0.7, color=colors[idx])

        ax.set_xlabel('Conductance Scale', fontsize=11, fontweight='bold')
        ax.set_ylabel('Firing Rate (Hz)', fontsize=11, fontweight='bold')
        ax.set_title(f'{synapse_type} Synapse Scaling', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Synaptic Conductance Sweep: Firing Rates by Synapse Type',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig


def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string."""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return img_str


def generate_html_report(log_data, output_file='parameter_sweep_report.html'):
    """Generate comprehensive HTML report."""

    print("\nGenerating visualizations...")
    fig1, poisson_data = analyze_poisson_sweep(log_data['poisson_sweep'])
    img1 = fig_to_base64(fig1)

    fig2 = analyze_synapse_sweep(log_data['synapse_sweep'])
    img2 = fig_to_base64(fig2)

    # Summary statistics
    poisson_log = log_data['poisson_sweep']
    synapse_log = log_data['synapse_sweep']

    max_poisson_fr = max(poisson_log['firing_rates'].values()) if poisson_log['firing_rates'] else 0
    max_synapse_fr = max(synapse_log['firing_rates'].values()) if synapse_log['firing_rates'] else 0

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Parameter Sweep Analysis Report</title>
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
        }}
        .stat-box .value {{
            font-size: 24px;
            font-weight: bold;
        }}
        .stat-box .label {{
            font-size: 12px;
            opacity: 0.9;
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
        }}
        .footer {{
            text-align: center;
            color: #666;
            font-size: 12px;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #ccc;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔬 Parameter Sweep Analysis Report</h1>
        <p>Izhikevich Spiking Network - Poisson Noise & Synaptic Conductance Sweep</p>
    </div>

    <div class="section">
        <h2>📊 Analysis Overview</h2>
        <div class="stats">
            <div class="stat-box">
                <div class="value">{len(poisson_log['amplitudes'])*len(poisson_log['rates'])}</div>
                <div class="label">Poisson Simulations</div>
            </div>
            <div class="stat-box">
                <div class="value">{len(synapse_log['scales'])*len(synapse_log['synapse_types'])}</div>
                <div class="label">Synapse Simulations</div>
            </div>
            <div class="stat-box">
                <div class="value">{max_poisson_fr:.2f}</div>
                <div class="label">Max Poisson FR (Hz)</div>
            </div>
            <div class="stat-box">
                <div class="value">{max_synapse_fr:.2f}</div>
                <div class="label">Max Synapse FR (Hz)</div>
            </div>
        </div>
    </div>

    <div class="section">
        <h2>📈 Poisson Noise Parameter Sweep</h2>
        <p><strong>Amplitudes:</strong> {', '.join([f'{a:.0f}' for a in poisson_log['amplitudes']])} mV/ms</p>
        <p><strong>Rates:</strong> {', '.join([f'{r:.1f}' for r in poisson_log['rates']])} Hz</p>
        <div class="figure">
            <img src="data:image/png;base64,{img1}" alt="Poisson Sweep">
            <p>Left: Total firing rate. Center: Baseline (0-500ms) firing rate. Right: Stimulus (600-1200ms) firing rate. Heatmaps show how firing rate varies with Poisson amplitude and rate.</p>
        </div>
    </div>

    <div class="section">
        <h2>🔗 Synaptic Conductance Sweep</h2>
        <p><strong>Synapse Types:</strong> E-E (excitatory-to-excitatory), E-I (excitatory-to-inhibitory), I-E (inhibitory-to-excitatory), I-I (inhibitory-to-inhibitory)</p>
        <p><strong>Scales:</strong> {', '.join([f'{s:.1f}' for s in synapse_log['scales']])} (multiplicative factor)</p>
        <div class="figure">
            <img src="data:image/png;base64,{img2}" alt="Synapse Sweep">
            <p>Firing rate response curves for each synapse type. Shows how total, baseline, and stimulus-evoked firing rates change with synaptic conductance scaling. Each subplot represents one synapse type.</p>
        </div>
    </div>

    <div class="section">
        <h2>⚙️ Experimental Configuration</h2>
        <ul>
            <li><strong>Network:</strong> 89 neurons (70% E, 30% I)</li>
            <li><strong>Connectivity:</strong> Fully dense (all-to-all synapses)</li>
            <li><strong>Neuron Model:</strong> Izhikevich (fast-spiking parameters, a=0.1)</li>
            <li><strong>Simulation Duration:</strong> 1500 ms (baseline + stimulus + post-stimulus)</li>
            <li><strong>Base Conductance Scale:</strong> 50x (before parameter sweep)</li>
            <li><strong>Baseline Input:</strong> Tonic current (200 mV/ms E, 100 mV/ms I)</li>
            <li><strong>Stimulus:</strong> Poisson spike trains with variable amplitude/rate</li>
        </ul>
    </div>

    <div class="footer">
        <p>Report generated: Parameter sweep analysis of Izhikevich spiking network</p>
        <p>Time periods: Baseline (0-500ms) | Stimulus (600-1200ms) | Post-Stimulus (1200-1500ms)</p>
    </div>
</body>
</html>"""

    with open(output_file, 'w') as f:
        f.write(html)

    print(f"✅ Report saved to {output_file}")
    return output_file


if __name__ == "__main__":
    print("\n" + "="*80)
    print("ANALYZING PARAMETER SWEEP RESULTS")
    print("="*80)

    # Load
    try:
        log_data = load_logs('simulation_logs_izh.pkl')
    except FileNotFoundError:
        print("❌ simulation_logs_izh.pkl not found. Run parameter_sweep.py first.")
        exit(1)

    # Generate report
    output = generate_html_report(log_data)

    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE")
    print("="*80)
    print(f"Report: {output}")
