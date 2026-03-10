import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List
from scipy import signal

def traces_to_spike_matrix(traces: np.ndarray, threshold: float = -20.0) -> np.ndarray:
    """Converts voltage traces to a binary spike matrix."""
    num_neurons, num_steps = traces.shape
    spike_matrix = np.zeros_like(traces)
    for i in range(num_neurons):
        spikes = (traces[i, :-1] < threshold) & (traces[i, 1:] >= threshold)
        spike_matrix[i, 1:][spikes] = 1.0
    return spike_matrix

def detect_spikes(neuron_trace, threshold=-20.0):
    """Detects upward crossings of a threshold."""
    # Exclude first point if it's V_init
    data = neuron_trace[1:]
    spikes = (data[:-1] < threshold) & (data[1:] >= threshold)
    return jnp.sum(spikes)

def calculate_firing_rates(traces, dt, threshold=-20.0):
    """Computes firing rates (Hz) for a batch of traces (Batch, Cells, Time)."""
    if traces.ndim == 2: # Single batch
        traces = traces[None, ...]
    num_batches, num_cells, num_timepoints = traces.shape
    duration_s = (num_timepoints * dt) / 1000.0
    
    # Nested vmaps for spike detection
    batched_detect = jax.vmap(lambda t: detect_spikes(t, threshold))
    total_spikes = jax.vmap(batched_detect)(traces)
    
    return total_spikes / duration_s

def compute_psd(signal_1d, dt, target_freqs=None):
    """Computes PSD for a 1D signal."""
    N = signal_1d.shape[-1]
    fs = 1000.0 / dt
    signal_fft = jnp.fft.rfft(signal_1d)
    freqs = jnp.fft.rfftfreq(N, d=dt/1000.0)
    psd_raw = jnp.abs(signal_fft)**2 / (N * fs)
    
    if target_freqs is None:
        target_freqs = jnp.linspace(1.0, 100.0, 100)
    
    interpolated_psd = jnp.interp(target_freqs, freqs, psd_raw)
    return target_freqs, interpolated_psd

def calculate_psd_bands(psd, freqs, band_defs):
    """Calculates average power in specified frequency bands."""
    band_powers = {}
    for band_name, (f_min, f_max) in band_defs.items():
        indices = (freqs >= f_min) & (freqs < f_max)
        if not jnp.any(indices):
            band_powers[band_name] = 0.0
        else:
            band_powers[band_name] = jnp.mean(psd[..., indices])
    return band_powers

def compute_correlations(traces, pre_inds, post_inds):
    """
    Computes mutual Pearson correlation (est., time adjusted) between pre and post traces.
    traces: (Batch, Cells, Time)
    pre_inds, post_inds: (Num_Edges,)
    Returns: (Num_Edges,)
    """
    if pre_inds.size == 0 or post_inds.size == 0:
        return jnp.array([])

    # Shifted correlation
    pre_v = traces[:, pre_inds, :-10]
    post_v = traces[:, post_inds, 10:]

    pre_mean = jnp.mean(pre_v, axis=-1, keepdims=True)
    post_mean = jnp.mean(post_v, axis=-1, keepdims=True)
    
    num = jnp.sum((pre_v - pre_mean) * (post_v - post_mean), axis=-1)
    den = jnp.sqrt(jnp.sum((pre_v - pre_mean)**2, axis=-1) * jnp.sum((post_v - post_mean)**2, axis=-1))
    
    # Square correlation to get mutual information proxy
    return jnp.mean(num**2 / (den + 1e-6), axis=0)

def calculate_mcdp(traces, ampa_pre_inds, ampa_post_inds, gaba_pre_inds, gaba_post_inds):
    """
    Calculates normalized Mutual-correlation dependent plasticity (MCDP) factors 
    for all trainable parameters.
    """
    r_ampa = compute_correlations(traces, ampa_pre_inds, ampa_post_inds)
    r_gaba = compute_correlations(traces, gaba_pre_inds, gaba_post_inds)

    r_ampa_norm = (r_ampa - jnp.mean(r_ampa)) / (jnp.std(r_ampa) + 1e-6) if r_ampa.size > 0 else jnp.array([])
    r_gaba_norm = (r_gaba - jnp.mean(r_gaba)) / (jnp.std(r_gaba) + 1e-6) if r_gaba.size > 0 else jnp.array([])

    return [{'gAMPA': r_ampa_norm}, {'gGABAa': r_gaba_norm}]

def compute_unscaled_psd_from_trace(trace, dt_global):
    """Computes unscaled PSD for a 1D trace."""
    N = trace.shape[-1]
    fs = 1000.0 / dt_global
    signal_fft = jnp.fft.rfft(trace)
    freqs = jnp.fft.rfftfreq(N, d=dt_global/1000.0)
    psd_raw = jnp.abs(signal_fft)**2 / (N * fs)
    target_freqs = jnp.linspace(1.0, 100.0, 100)
    interpolated_psd = jnp.interp(target_freqs, freqs, psd_raw)
    return target_freqs, interpolated_psd

def compute_kappa(spike_matrix, fs, bin_size_ms=5.0):
    """
    Computes Fleiss' Kappa for inter-neuron synchrony.
    Input: (Cells, Time) binary matrix.
    """
    N, T = spike_matrix.shape
    samples_per_bin = int(fs * (bin_size_ms / 1000.0)) or 1
    num_bins = T // samples_per_bin
    
    # Use a slice that is a multiple of samples_per_bin
    # We use dynamic_slice or just regular slicing if indices are known
    valid_T = num_bins * samples_per_bin
    binned_data = spike_matrix[:, :valid_T]
    
    # Reshape and take max across bin
    # Note: reshape with -1 or concrete values
    # To keep it JIT-friendly, we use the fact that valid_T is num_bins * samples_per_bin
    binned = jnp.max(binned_data.reshape(N, num_bins, samples_per_bin), axis=2)
    
    spikes_per_bin = binned.sum(axis=0)
    silences_per_bin = N - spikes_per_bin
    
    # P_i: relative observed agreement in bin i
    # Avoid division by zero if N < 2
    den_Pi = jnp.maximum(1.0, float(N * (N - 1)))
    P_i = (spikes_per_bin * (spikes_per_bin - 1) + silences_per_bin * (silences_per_bin - 1)) / den_Pi
    P_o = jnp.mean(P_i)
    
    # P_e: relative agreement expected by chance
    den_Pe = jnp.maximum(1.0, float(N * num_bins))
    p_spike = jnp.sum(spikes_per_bin) / den_Pe
    P_e = (p_spike**2) + ((1 - p_spike)**2)
    
    # Final Kappa
    return (P_o - P_e) / jnp.maximum(1e-6, 1.0 - P_e)

def plot_full_simulation_summary(recorded_voltages, time_axis, dt_global,
                                 spike_threshold=-20.0,
                                 window_size=250.0, overlap=0.95,
                                 f_min=1.0, f_max=100.0,
                                 title_suffix="", figsize=(16, 14), save=False, savename="fig3p.svg",
                                 aperiodic_correct: float = 0.0,
                                 baseline_relative: Optional[Tuple[float, float]] = None,
                                 interval1: Tuple[float, float] = (1, 500),
                                 interval2: Tuple[float, float] = (500, 1000)):
    """
    Combines voltage image, raster plot, and time-frequency response into a single multi-subplot figure.
    """
    recorded_voltages = jnp.nan_to_num(recorded_voltages, nan=0.0)
    recorded_voltages = jnp.clip(recorded_voltages, -100, +100)

    fig = plt.figure(figsize=figsize)
    
    # Subplot 1: Raster Plot as Image
    plt.subplot(4, 1, 1)
    num_neurons = recorded_voltages.shape[0]
    downsampling_factor = int(jnp.ceil(1.0 / dt_global)) or 1
    neuron_traces_data = recorded_voltages[:, 1:]
    original_num_timepoints = neuron_traces_data.shape[1]
    
    num_timepoints_raster = (original_num_timepoints + downsampling_factor - 1) // downsampling_factor
    spike_image = jnp.zeros((num_neurons, num_timepoints_raster))

    for i in range(num_neurons):
        neuron_trace_down = recorded_voltages[i, 1:][::downsampling_factor]
        spikes = (neuron_trace_down[:-1] < spike_threshold) & (neuron_trace_down[1:] >= spike_threshold)
        spike_indices = jnp.where(spikes)[0]
        spike_image = spike_image.at[i, spike_indices].set(1.0)

    plt.imshow(spike_image, aspect='auto', cmap='Greys', origin='lower',
               extent=[time_axis[0], time_axis[-1], 0, num_neurons], vmin=0, vmax=1)
    plt.colorbar(label='Spike Activity', ticks=[0.25, 0.75]).set_ticklabels(['No Spike', 'Spike'])
    plt.title(f'Raster Plot (Image) {title_suffix}')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron Index')
    plt.axvline(x=500, color='red', linestyle='--', linewidth=1.5)

    # Subplot 2: Time-Frequency Response
    plt.subplot(4, 1, 2)
    dt_raster = dt_global * downsampling_factor
    fs_raster = 1000.0 / dt_raster
    sigma_samples = 5.0 / dt_raster
    
    k_size = int(7 * sigma_samples + 1)
    x_grid = jnp.linspace(-3.0 * sigma_samples, 3.0 * sigma_samples, k_size)
    kernel = jnp.exp(-x_grid**2 / (2 * sigma_samples**2))
    kernel = kernel / jnp.sum(kernel)
    
    recorded_spiking_w = jax.scipy.signal.convolve2d(spike_image, kernel[None, :], mode='same')
    mean_signal = jnp.mean(recorded_spiking_w, axis=0)
    
    # Spectrogram with mirroring
    nperseg = int(window_size * fs_raster / 1000.0) or 1
    half_win = nperseg // 2
    mean_signal_np = np.asarray(mean_signal)
    
    if len(mean_signal_np) > half_win:
        prefix = mean_signal_np[1 : half_win + 1][::-1]
        suffix = mean_signal_np[-(half_win + 1) : -1][::-1]
        mean_signal_np = np.concatenate((prefix, mean_signal_np, suffix))

    frequencies, times, Sxx = signal.spectrogram(mean_signal_np, fs=fs_raster, window='hann', 
                                                 nperseg=nperseg, noverlap=int(nperseg * overlap))
    
    adjusted_times = times * 1000 - (half_win * dt_raster)
    freq_mask = (frequencies >= f_min) & (frequencies <= f_max)
    frequencies_f = frequencies[freq_mask]
    Sxx_f = Sxx[freq_mask, :]

    if frequencies_f.size > 0:
        if aperiodic_correct != 0.0:
            Sxx_f = Sxx_f / (frequencies_f[:, None]**aperiodic_correct + 1e-9)
        
        if baseline_relative:
            t_start, t_end = baseline_relative
            idx_s = jnp.argmin(jnp.abs(adjusted_times - t_start))
            idx_e = jnp.argmin(jnp.abs(adjusted_times - t_end))
            baseline = jnp.mean(Sxx_f[:, idx_s:idx_e+1], axis=1, keepdims=True)
            Sxx_f = Sxx_f / (baseline + 1e-9)

        Sxx_scaled = jnp.sqrt((Sxx_f - jnp.min(Sxx_f)) / (jnp.max(Sxx_f) - jnp.min(Sxx_f) + 1e-9))
        plt.imshow(Sxx_scaled, aspect='auto', cmap='jet', origin='lower',
                   extent=[adjusted_times[0], adjusted_times[-1], frequencies_f[0], frequencies_f[-1]])
        plt.colorbar(label='Normalized Power')
        plt.title(f'Mean Spectrogram {title_suffix}')
        plt.ylabel('Frequency (Hz)')
        plt.axvline(x=500, color='red', linestyle='--', linewidth=1.5)

    # Subplots 3 & 4: Average PSDs
    for i, interval in enumerate([interval1, interval2], 3):
        plt.subplot(4, 1, i)
        if frequencies_f.size > 0:
            t_mask = (adjusted_times >= interval[0]) & (adjusted_times <= interval[1])
            if np.any(t_mask):
                avg_psd = jnp.mean(Sxx_f[:, t_mask], axis=1)
                avg_psd_scaled = (avg_psd - jnp.min(avg_psd)) / (jnp.max(avg_psd) - jnp.min(avg_psd) + 1e-9)
                plt.plot(frequencies_f, avg_psd_scaled, color='blue')
                plt.title(f'Average PSD ({interval[0]}-{interval[1]} ms)')
                plt.ylabel('Norm Power')
                plt.ylim([0, 1.1])

    plt.tight_layout()
    if save: plt.savefig(savename, format='svg')
    plt.show()


def calculate_axial_current(traces_soma, traces_dend, ra=100.0):
    """Calculates the axial current (MEG dipole approximation)."""
    return (traces_dend - traces_soma) / ra
