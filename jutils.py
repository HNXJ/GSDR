import jax.numpy as jnp
import jaxley as jx
import numpy as np

import jax
import jax.scipy.signal
from jax import jit, vmap, value_and_grad

from scipy import signal # For signal.spectrogram
from scipy.signal import detrend # Import detrend specifically
from scipy.ndimage import zoom, gaussian_filter # For spectrogram smoothing and upsampling

import matplotlib.pyplot as plt
import os
from typing import Optional, Tuple


class Dataset:
    """
    A simple Dataloader which returns batches of the data.

    Instead of using this simple dataloader, you can also just use one from
    PyTorch or Tensorflow. You do not have to understand what is going on here
    to follow this tutorial.
    """

    def __init__(self, inputs: np.ndarray, labels: np.ndarray):
        """
        Initialize the dataloader.

        Args:
            inputs: Array of shape (num_samples, num_dim)
            labels: Array of shape (num_samples,)
        """
        assert len(inputs) == len(labels), "Inputs and labels must have same length"
        self.inputs = inputs
        self.labels = labels
        self.num_samples = len(inputs)
        self._rng_state = None
        self.batch_size = 1

    def shuffle(self, seed=None):
        """
        Shuffle the dataset in-place
        """
        self._rng_state = np.random.get_state()[1][0] if seed is None else seed
        np.random.seed(self._rng_state)
        indices = np.random.permutation(self.num_samples)
        self.inputs = self.inputs[indices]
        self.labels = self.labels[indices]
        return self

    def batch(self, batch_size):
        """
        Create batches of the data.
        """
        self.batch_size = batch_size
        return self

    def __iter__(self):
        """
        Iterate over the dataset.
        """
        self.shuffle(seed=self._rng_state)
        for start in range(0, self.num_samples, self.batch_size):
            end = min(start + self.batch_size, self.num_samples)
            yield self.inputs[start:end], self.labels[start:end]
        self._rng_state += 1


def extend_traces_for_spectrogram(traces, nperseg):
    """
    Mirrors the beginning and end of the traces based on the spectrogram window size.
    Input: traces (N, T)
    Output: traces_extended (N, T + nperseg)
    """
    half_window = int(nperseg // 2)
    # Mirror the beginning (reverse first half_window samples)
    prefix = traces[:, 1 : half_window + 1][:, ::-1]
    suffix = traces[:, -half_window - 1 : -1][:, ::-1]

    traces_extended = jnp.concatenate([prefix, traces, suffix], axis=1)
    return traces_extended


def compute_summed_spectrogram(traces_extended, fs, nperseg, noverlap):
    """
    Calculates spectrogram on each row separately and sums them up.
    """
    # Convert to numpy for scipy.signal
    traces_np = np.array(traces_extended)

    total_Sxx = None
    freqs = None
    times = None

    for i in range(traces_np.shape[0]):
        f, t, Sxx = signal.spectrogram(traces_np[i], fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap)
        # Apply detrend before summing
        Sxx_detrended = Sxx / Sxx[0]
        Sxx_detrended = Sxx
        if total_Sxx is None:
            total_Sxx = Sxx_detrended
            freqs = f
            times = t
        else:
            total_Sxx += Sxx_detrended

    # Normalize each time slice (column) by its total power, as requested
    # Add a small epsilon to avoid division by zero
    total_power_at_each_time = np.sum(total_Sxx, axis=0, keepdims=True)
    # Ensure total_power_at_each_time is not zero to prevent division by zero
    total_power_at_each_time = np.where(total_power_at_each_time == 0, 1e-9, total_power_at_each_time)
    normalized_Sxx = total_Sxx / total_power_at_each_time

    return freqs, times, normalized_Sxx


def vis_smoothed_spectrogram(ax, data, t_range, f_range, zoom_fac=3, sigma=1.5):
    """
    Helper to upsample, smooth, and visualize the spectrogram image.
    """
    # Upsample
    data_up = zoom(data, zoom_fac, order=1)
    # Smooth
    data_smooth = gaussian_filter(data_up, sigma=sigma)

    # Plot
    # Note: imshow extent is [left, right, bottom, top]
    extent = [t_range[0], t_range[-1], f_range[0], f_range[1]]
    ax.imshow(data_smooth, aspect='auto', origin='lower', cmap='jet', extent=extent, interpolation='bicubic')


def test_net_vis_comp_psd(net, params, simulate_fn, input_scalar, label_psd, label_psd_2, global_psd_interval, dt_global, lower_c, upper_c, psd_weight, firing_rate_weight, interval1=(1, 500), interval2=(501, 1000), savename=None):
    """
    Visualizes the PSD prediction and loss for a single input/label pair over two specific intervals.
    """
    # 1. Simulate (Scalar input)
    traces = simulate_fn(params, input_scalar) # Shape (Cells, Time)

    # Helper to calculate PSD and stats for a specific interval
    def analyze_interval(traces_segment, dt):
        # PSD
        signal = jnp.mean(traces_segment, axis=0)
        N = signal.shape[-1]
        if N == 0:
            return jnp.zeros_like(global_psd_interval), 0.0

        fs = 1000.0 / dt
        signal_fft = jnp.fft.rfft(signal)
        freqs = jnp.fft.rfftfreq(N, d=dt/1000)
        psd_raw = jnp.abs(signal_fft)**2 / (N * fs)

        target_freqs = global_psd_interval
        interpolated_psd = jnp.interp(target_freqs, freqs, psd_raw)
        max_psd = jnp.max(interpolated_psd)
        prediction_psd = interpolated_psd / (max_psd + 1e-6)

        # Firing Rate
        threshold = -20.0
        spikes = (traces_segment[:, :-1] < threshold) & (traces_segment[:, 1:] >= threshold)
        count = jnp.sum(spikes)
        n_cells = traces_segment.shape[0]
        duration_sec = (traces_segment.shape[1] * dt) / 1000.0
        mean_fr = count / n_cells / duration_sec

        return prediction_psd, mean_fr

    # Indices for slicing
    idx_start1 = int(interval1[0] / dt_global)
    idx_end1 = int(interval1[1] / dt_global)
    idx_start2 = int(interval2[0] / dt_global)
    idx_end2 = int(interval2[1] / dt_global)

    # Analyze Intervals
    psd1, mean_fr1 = analyze_interval(traces[:, idx_start1:idx_end1], dt_global)
    psd2, mean_fr2 = analyze_interval(traces[:, idx_start2:idx_end2], dt_global)

    # Calculate Loss for Interval 1
    epsilon = 1e-6
    psd_loss1 = jnp.sum(jnp.square(label_psd * jnp.log((psd1 + epsilon) / (label_psd + epsilon))))
    psd_loss2 = jnp.sum(jnp.square(label_psd_2 * jnp.log((psd2 + epsilon) / (label_psd_2 + epsilon))))

    # Firing rate penalty approximation for single scalar mean FR (assuming uniform distribution roughly)
    # Note: This differs slightly from calculating per neuron then averaging penalty, but fits the "mean_fr" scalar we have
    penalty_lower = jnp.exp(lower_c - mean_fr1)
    penalty_upper = jnp.exp(mean_fr1 - upper_c)
    firing_rate_penalty1 = penalty_lower + penalty_upper

    total_loss1 = psd_loss1 * psd_weight + firing_rate_weight * firing_rate_penalty1

    # Visualization
    fig, axs = plt.subplots(2, 1, figsize=(12, 6))

    # Upsample and smoothen prediction lines
    upsample_factor = 10
    smoothing_sigma = 1.5

    global_psd_interval_upsampled = jnp.linspace(
        global_psd_interval[0], global_psd_interval[-1],
        len(global_psd_interval) * upsample_factor
    )

    psd1_upsampled = jnp.interp(global_psd_interval_upsampled, global_psd_interval, psd1)
    psd1_smoothed = gaussian_filter(psd1_upsampled, sigma=smoothing_sigma)

    psd2_upsampled = jnp.interp(global_psd_interval_upsampled, global_psd_interval, psd2)
    psd2_smoothed = gaussian_filter(psd2_upsampled, sigma=smoothing_sigma)


    # Subplot 211: Interval 1
    axs[0].plot(global_psd_interval, label_psd, label='Target (Label_pre)', linestyle='--', color='black')
    axs[0].plot(global_psd_interval_upsampled, psd1_smoothed, label=f'Prediction ({interval1}ms) Smoothed', color='red')
    axs[0].set_title(f"Interval {interval1}ms - Total Loss: {total_loss1:.4f}")
    axs[0].set_xlabel("Frequency (Hz)")
    axs[0].set_ylabel("Normalized Power")
    axs[0].legend()

    stats_text1 = (f"Total Loss: {total_loss1:.4f}\n"
                   f"PSD Loss: {psd_loss1*psd_weight:.4f}\n"
                   f"FR Penalty: {firing_rate_penalty1*firing_rate_weight:.4f}\n"
                   f"Mean FR: {mean_fr1:.2f} Hz")
    axs[0].text(0.05, 0.95, stats_text1,
                transform=axs[0].transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    # Subplot 212: Interval 2
    axs[1].plot(global_psd_interval, label_psd_2, label='Target (Label_stim)', linestyle='--', color='black')
    axs[1].plot(global_psd_interval_upsampled, psd2_smoothed, label=f'Prediction ({interval2}ms) Smoothed', color='blue')
    axs[1].set_title(f"Interval {interval2}ms")
    axs[1].set_xlabel("Frequency (Hz)")
    axs[1].set_ylabel("Normalized Power")
    axs[1].legend()
    axs[1].text(0.05, 0.95, f"Mean FR: {mean_fr2:.2f} Hz",
                transform=axs[1].transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    plt.tight_layout()
    if savename:
        plt.savefig(savename, format='svg')
    plt.show()

    return total_loss1

def test_net_vis_comp_tfr(net, params, simulate_fn, input_scalar, dt_global, label_psd=None, interval1=(1, 500), interval2=(501, 1000), savename=None):
    """
    Visualizes the model raster and spectrogram.
    """
    # 1. Simulate
    traces = simulate_fn(params, input_scalar)

    # 2. Raster
    spike_threshold = -20.0
    num_neurons = traces.shape[0]

    downsampling_factor = int(jnp.ceil(1.0 / dt_global))
    if downsampling_factor <= 0:
        downsampling_factor = 1

    neuron_traces_data = traces[:, 1:] # Shape (num_neurons, original_num_timepoints)
    original_num_timepoints = neuron_traces_data.shape[1]

    full_res_spike_trains = jnp.zeros((num_neurons, original_num_timepoints), dtype=jnp.float32)
    for i in range(num_neurons):
        neuron_trace = traces[i, 1:]
        # Fix: Changed neuron_trace[:, 1:] to neuron_trace[1:] for 1D array
        spikes_detected = (neuron_trace[:-1] < spike_threshold) & (neuron_trace[1:] >= spike_threshold)
        spike_indices = jnp.where(spikes_detected)[0] + 1
        full_res_spike_trains = full_res_spike_trains.at[i, spike_indices].set(1.0)

    num_timepoints_raster = (original_num_timepoints + downsampling_factor - 1) // downsampling_factor
    spike_image = jnp.zeros((num_neurons, num_timepoints_raster), dtype=jnp.float32)

    for i in range(num_neurons):
        neuron_trace = traces[i, 1:]
        neuron_trace_down = neuron_trace[::downsampling_factor]

        spikes = (neuron_trace_down[:-1] < spike_threshold) & (neuron_trace_down[1:] >= spike_threshold)

        spike_indices = jnp.where(spikes)[0]
        spike_image = spike_image.at[i, spike_indices].set(1.0)

    print(spike_image.shape)

    # Cumulative spiking
    cumulative_spiking = jnp.sum(spike_image, axis=0)

    # 3. Spectrogram (Summed across neurons)
    fs = 1000.0 / dt_global
    nperseg = int(200.0 * fs / 1000.0) # 200ms window
    noverlap = int(nperseg * 0.99)

    # Get raw data (exclude V_init)
    traces_data = traces[:, 1:]

    # Extend traces with mirroring
    traces_extended = extend_traces_for_spectrogram(traces_data, nperseg)

    # Compute summed spectrogram
    f, t, Sxx_sum = compute_summed_spectrogram(traces_extended, fs, nperseg, noverlap)

    # Normalize summed Sxx for plotting
    Sxx_plot = Sxx_sum #np.sqrt(Sxx_sum)

    t_shift = (nperseg / 2) / fs
    t_ms = (t - t_shift) * 1000.0

    # 4. Plot
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 0.1, 1], hspace=0.15)

    axs = []
    axs.append(fig.add_subplot(gs[0]))
    axs.append(fig.add_subplot(gs[1], sharex=axs[0]))
    axs.append(fig.add_subplot(gs[2], sharex=axs[0]))

    # Raster
    t_max = traces.shape[1] * dt_global
    axs[0].imshow(spike_image, aspect='auto', cmap='Greys', origin='lower', interpolation='none',
                  extent=[0, t_max, 0, spike_image.shape[0]])
    axs[0].set_title("Model Raster")
    axs[0].set_ylabel("Neuron Index")
    axs[0].tick_params(labelbottom=False)

    # Cumulative Spiking
    raster_time_axis = jnp.linspace(0, t_max, num_timepoints_raster)
    axs[1].plot(raster_time_axis, cumulative_spiking, color='black', linewidth=0.8)
    axs[1].set_ylabel("Spike Count")
    axs[1].tick_params(labelbottom=False)

    # Spectrogram
    # Limit freq axis to interest (e.g. 0-100Hz)
    f_lim_idx = f <= 100.0
    f_plot = f[f_lim_idx]
    Sxx_plot_lim = Sxx_plot[f_lim_idx, :]

    # Spectrogram
    vis_smoothed_spectrogram(axs[2], Sxx_plot_lim, [t_ms[0], t_ms[-1]], [f_plot[0], f_plot[-1]])

    axs[2].set_title("Summed Spectrogram (Population Power)")
    axs[2].set_ylabel("Frequency (Hz)")
    axs[2].set_xlabel("Time (ms)")
    axs[2].set_ylim(0, 100)

    # Add red dashed lines
    red_line_freqs = [3.0, 8.0, 12.0, 32.0, 90.0]
    for freq in red_line_freqs:
        axs[2].axhline(y=freq, color='red', linestyle='--', linewidth=1.0, alpha=0.8)

    plt.tight_layout()
    if savename:
        plt.savefig(savename, format='svg')
    plt.show()


def trace_vis_tfr(traces, label_psd=None, interval1=(1, 500), interval2=(501, 1000), dt_trace=0.1, savename: Optional[str] = None, spike_threshold=-20.0):
    """
    Visualizes the model raster and spectrogram.
    """
    # 2. Raster
    num_neurons = traces.shape[0]

    downsampling_factor = int(jnp.ceil(1.0 / dt_trace))
    if downsampling_factor <= 0:
        downsampling_factor = 1

    neuron_traces_data = traces[:, 1:] # Shape (num_neurons, original_num_timepoints)
    original_num_timepoints = neuron_traces_data.shape[1]

    full_res_spike_trains = jnp.zeros((num_neurons, original_num_timepoints), dtype=jnp.float32)
    for i in range(num_neurons):
        neuron_trace = traces[i, 1:]
        # Fix: Changed neuron_trace[:, 1:] to neuron_trace[1:] for 1D array
        spikes_detected = (neuron_trace[:-1] < spike_threshold) & (neuron_trace[1:] >= spike_threshold)
        spike_indices = jnp.where(spikes_detected)[0] + 1
        full_res_spike_trains = full_res_spike_trains.at[i, spike_indices].set(1.0)

    num_timepoints_raster = (original_num_timepoints + downsampling_factor - 1) // downsampling_factor
    spike_image = jnp.zeros((num_neurons, num_timepoints_raster), dtype=jnp.float32)

    for i in range(num_neurons):
        neuron_trace = traces[i, 1:]
        neuron_trace_down = neuron_trace[::downsampling_factor]

        spikes = (neuron_trace_down[:-1] < spike_threshold) & (neuron_trace_down[1:] >= spike_threshold)

        spike_indices = jnp.where(spikes)[0]
        spike_image = spike_image.at[i, spike_indices].set(1.0)

    print(spike_image.shape)

    # Cumulative spiking
    cumulative_spiking = jnp.sum(spike_image, axis=0)

    # 3. Spectrogram (Summed across neurons)
    fs = 1000.0 / dt_trace
    nperseg = int(200.0 * fs / 1000.0) # 200ms window
    noverlap = int(nperseg * 0.99)

    # Get raw data (exclude V_init)
    traces_data = traces[:, 1:]

    # Extend traces with mirroring
    traces_extended = extend_traces_for_spectrogram(traces_data, nperseg)

    # Compute summed spectrogram
    f, t, Sxx_sum = compute_summed_spectrogram(traces_extended, fs, nperseg, noverlap)

    # Normalize summed Sxx for plotting
    Sxx_plot = Sxx_sum #np.sqrt(Sxx_sum)

    t_shift = (nperseg / 2) / fs
    t_ms = (t - t_shift) * 1000.0

    # 4. Plot
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 0.1, 1], hspace=0.15)

    axs = []
    axs.append(fig.add_subplot(gs[0]))
    axs.append(fig.add_subplot(gs[1], sharex=axs[0]))
    axs.append(fig.add_subplot(gs[2], sharex=axs[0]))

    # Raster
    t_max = traces.shape[1] * dt_trace
    axs[0].imshow(spike_image, aspect='auto', cmap='Greys', origin='lower', interpolation='none',
                  extent=[0, t_max, 0, spike_image.shape[0]])
    axs[0].set_title("Model Raster")
    axs[0].set_ylabel("Neuron Index")
    axs[0].tick_params(labelbottom=False)

    # Cumulative Spiking
    raster_time_axis = jnp.linspace(0, t_max, num_timepoints_raster)
    axs[1].plot(raster_time_axis, cumulative_spiking, color='black', linewidth=0.8)
    axs[1].set_ylabel("Spike Count")
    axs[1].tick_params(labelbottom=False)

    # Spectrogram
    # Limit freq axis to interest (e.g. 0-100Hz)
    f_lim_idx = f <= 100.0
    f_plot = f[f_lim_idx]
    Sxx_plot_lim = Sxx_plot[f_lim_idx, :]

    # Spectrogram
    vis_smoothed_spectrogram(axs[2], Sxx_plot_lim, [t_ms[0], t_ms[-1]], [f_plot[0], f_plot[-1]])

    axs[2].set_title("Summed Spectrogram (Population Power)")
    axs[2].set_ylabel("Frequency (Hz)")
    axs[2].set_xlabel("Time (ms)")
    axs[2].set_ylim(0, 100)

    # Add red dashed lines
    red_line_freqs = [3.0, 8.0, 12.0, 32.0, 90.0]
    for freq in red_line_freqs:
        axs[2].axhline(y=freq, color='red', linestyle='--', linewidth=1.0, alpha=0.8)

    plt.tight_layout()
    if savename:
        plt.savefig(savename, format='svg')
    plt.show()


def save_jnn(filename, filepath, net_object, initial_params, mid_params, final_params, log_net, Ne, Nig, Nil):
    """
    Saves various components of the JAXley model to a single .pkl file using pickle.
    Args:
        filename (str):
        filepath (str): Directory path to save the file.
        net_object (jx.Network): The JAXley network object.
        initial_params: Initial parameters of the network.
        mid_params: Intermediate parameters (if any).
        final_params: Final trained parameters of the network.
        log_net: Training log data.
        Ne, Nig, Nil: Network size parameters.
    """
    full_path = os.opath.join(filepath, filename + ".pkl") # Changed extension to .pkl

    # Collect all data into a dictionary
    model_data = {
        'initial_params': initial_params,
        'mid_params': mid_params,
        'final_params': final_params,
        'log_net': log_net,
        'net_params': net_object.get_parameters(), # Save the current parameters of the net
        'Ne': Ne, # Using global Ne variable
        'Nig': Nig, # Using global Nig variable
        'Nil': Nil, # Using global Nil variable
    }

    with open(full_path, "wb") as handle:
        pickle.dump(model_data, handle) # Use pickle.dump directly
    print(f"Model components saved to {full_path}")


def load_jnn(filename, filepath, net_eig_fn):
    """
    Loads various components of the JAXley model from a .pkl file using pickle.
    Args:
        filename (str): Base name for the file (e.g., "model_001").
        filepath (str): Directory path to load the file from.
        net_eig_fn: The function to reconstruct the network.
    Returns:
        tuple: (rebuilt_net_object, initial_params, mid_params, final_params, log_net)
              where rebuilt_net_object is recreated and its parameters set.
    """
    full_path = os.path.join(filepath, filename) # .pkl
    with open(full_path, "rb") as handle:
        loaded_data = pickle.load(handle) # Use pickle.load directly

    initial_params = loaded_data['initial_params']
    mid_params = loaded_data['mid_params']
    final_params = loaded_data['final_params']
    log_net = loaded_data['log_net']
    net_params = loaded_data['net_params']

    rebuilt_net = net_eig_fn(loaded_data['Ne'], loaded_data['Nig'], loaded_data['Nil'])
    print(f"Model components loaded from {full_path}")
    return rebuilt_net, initial_params, mid_params, final_params, log_net


def noise_current(
    i_delay: float,
    i_dur: float,
    i_amp: float,
    delta_t: float,
    t_max: float,
    seed: int = 0, noise_standard_deviation: Optional[float] = None, noise_correlation_tau: Optional[float] = None, noise_mean: Optional[float] = None) -> jnp.ndarray:
    """
    Generates a random noise current pulse using an Ornstein-Uhlenbeck process.

    Args:
        i_delay: Start time of the current pulse in ms.
        i_dur: Duration of the current pulse in ms.
        i_amp: Overall amplitude scaling of the noise pulse (nA).
        delta_t: Time step for the simulation in ms.
        t_max: Total simulation time in ms.
        seed: Random seed for reproducibility (default: 0).

    Returns:
        jnp.ndarray: Array representing the noise current pulse over time, with shape (int(t_max / delta_t) + 1,).R
    """
    # Parameters for the Ornstein-Uhlenbeck noise process
    if noise_standard_deviation is None:
        noise_standard_deviation = 0.1 # Sigma for OU (nA) - a default value
    if noise_correlation_tau is None:
        noise_correlation_tau = 10.0     # Correlation time constant in ms
    if noise_mean is None:
        noise_mean = 0.1                 # Mean of the OU process

    key = jax.random.PRNGKey(seed)

    # Calculate number of steps to include initial state (for JAXley compatibility)
    num_steps = int(t_max / delta_t) + 1
    time_axis_array = jnp.arange(0, t_max + delta_t, delta_t)

    def ou_step_fn(n_prev, key_t):
        key_noise, _ = jax.random.split(key_t)
        xi = jax.random.normal(key_noise)

        drift = (noise_mean - n_prev) / noise_correlation_tau * delta_t
        diffusion = noise_standard_deviation * jnp.sqrt(2.0 / noise_correlation_tau) * xi * jnp.sqrt(delta_t) # FIX: Changed dt to delta_t

        new_n = n_prev + drift + diffusion # FIX: Changed 'n' to 'n_prev'

        # Return new state AND increment step
        return new_n, new_n

    initial_state_ou = noise_mean
    # Split keys for num_steps-1 iterations, as initial_state is already handled
    keys_series = jax.random.split(key, num_steps - 1)

    # Run the scan to generate the raw noise trace
    _, full_noise_trace_raw = jax.lax.scan(ou_step_fn, initial_state_ou, keys_series)
    # Prepend the initial state to get the full trace length
    full_noise_trace = jnp.concatenate([jnp.array([initial_state_ou]), full_noise_trace_raw])

    # Apply the pulse window based on i_delay and i_dur
    time_off = i_delay + i_dur
    pulse_mask = (time_axis_array >= i_delay) & (time_axis_array <= time_off)

    # Scale the noise by i_amp and apply the pulse mask
    noise_current_pulse_array = jnp.where(pulse_mask, full_noise_trace * i_amp, 0.0)

    return noise_current_pulse_array


def ramp_current(
    i_delay: float,
    i_dur: float,
    i_amp: float,
    delta_t: float,
    t_max: float
) -> jnp.ndarray:
    """
    Generates a ramping current pulse.

    The current is 0 up to i_delay, then ramps up linearly to i_amp over i_dur,
    and then is 0 again after i_delay + i_dur.

    Args:
        i_delay: Start time of the current ramp in ms.
        i_dur: Duration of the current ramp in ms.
        i_amp: Peak amplitude of the ramped current (nA).
        delta_t: Time step for the simulation in ms.
        t_max: Total simulation time in ms.

    Returns:
        jnp.ndarray: Array representing the ramping current pulse over time.
    """
    num_steps = int(t_max / delta_t) + 1
    time_axis_array = jnp.arange(0, t_max + delta_t, delta_t)

    current = jnp.zeros_like(time_axis_array)

    # Define the ramp start and end times
    t_ramp_start = i_delay
    t_ramp_end = i_delay + i_dur

    # Create a mask for the ramping period
    ramp_mask = (time_axis_array >= t_ramp_start) & (time_axis_array <= t_ramp_end)

    # Calculate the linear ramp within the masked period
    # (time - t_ramp_start) / i_dur gives a value from 0 to 1 over the duration
    ramp_factor = (time_axis_array - t_ramp_start) / i_dur
    ramp_current_segment = i_amp * ramp_factor

    # Apply the ramp current only within the masked period
    current = jnp.where(ramp_mask, ramp_current_segment, current)

    return current

def step_current(
    i_delay: float,
    i_dur: float,
    i_amp: float,
    delta_t: float,
    t_max: float
) -> jnp.ndarray:
    """
    Generates a step current pulse.

    The current is 0 up to i_delay, then steps up to i_amp for i_dur,
    and then is 0 again after i_delay + i_dur.

    Args:
        i_delay: Start time of the current step in ms.
        i_dur: Duration of the current step in ms.
        i_amp: Amplitude of the step current (nA).
        delta_t: Time step for the simulation in ms.
        t_max: Total simulation time in ms.

    Returns:
        jnp.ndarray: Array representing the step current pulse over time.
    """
    num_steps = int(t_max / delta_t) + 1
    time_axis_array = jnp.arange(0, t_max + delta_t, delta_t)

    current = jnp.zeros_like(time_axis_array)

    # Define the step start and end times
    t_step_start = i_delay
    t_step_end = i_delay + i_dur

    # Create a mask for the step period
    step_mask = (time_axis_array >= t_step_start) & (time_axis_array <= t_step_end)

    # Apply the step current only within the masked period
    current = jnp.where(step_mask, i_amp, current)

    return current


def noise_current_ac(
    i_delay: float,
    i_dur: float,
    amp_n: float,
    amp_b: float,
    spect: jnp.ndarray,
    delta_t: float,
    t_max: float,
    seed: int = 0
) -> jnp.ndarray:
    """
    Generates a current composed of an Ornstein-Uhlenbeck noise pulse and
    a superposition of sinusoidal waves (AC).

    Formula: I_total = (OU_Noise * amp_n) + (Sum(Sin(spect)) * amp_b)

    Args:
        i_delay: Start time of the pulse window in ms.
        i_dur: Duration of the pulse window in ms.
        amp_n: Amplitude scaling for the OU noise component (nA).
        amp_b: Amplitude scaling for the AC/Sinusoidal component (nA).
        spect: 1D array of frequencies (Hz) to construct the AC signal.
               Example: jnp.array([10.0, 40.0, 100.0]) for 10, 40, and 100Hz.
        delta_t: Time step for the simulation in ms.
        t_max: Total simulation time in ms.
        seed: Random seed for reproducibility.

    Returns:
        jnp.ndarray: The combined current trace (nA).
    """

    num_steps = int(t_max / delta_t) + 1
    time_axis_array = jnp.arange(0, t_max + delta_t, delta_t)

    noise_standard_deviation = 0.1
    noise_correlation_tau = 10.0
    noise_mean = 0.0 # Often 0 for pure fluctuations, user had 0.2 previously

    key = jax.random.PRNGKey(seed)

    def ou_step_fn(n_prev, key_t):
        key_noise, _ = jax.random.split(key_t)
        xi = jax.random.normal(key_noise)

        # Euler-Maruyama method
        drift = (noise_mean - n_prev) / noise_correlation_tau * delta_t
        diffusion = noise_standard_deviation * jnp.sqrt(2.0 / noise_correlation_tau) * xi * jnp.sqrt(delta_t)

        new_n = n_prev + drift + diffusion
        return new_n, new_n

    initial_state_ou = noise_mean
    keys_series = jax.random.split(key, num_steps - 1)

    _, full_noise_trace_raw = jax.lax.scan(ou_step_fn, initial_state_ou, keys_series)
    full_noise_trace = jnp.concatenate([jnp.array([initial_state_ou]), full_noise_trace_raw])

    # Conversion: t is in ms, spect is in Hz (1/s).
    # Argument = 2*pi * f * (t / 1000)

    t_reshaped = time_axis_array[None, :]
    freqs_reshaped = spect[:, None]

    # Calculate phases
    phases = 2 * jnp.pi * freqs_reshaped * (t_reshaped / 1000.0)

    # Sum sines across the frequency axis (axis 0)
    # Result is shape (T,)
    ac_trace = jnp.sum(jnp.sin(phases), axis=0)

    # --- 4. Combine and Gate ---
    # Define the time window mask
    time_off = i_delay + i_dur
    pulse_mask = (time_axis_array >= i_delay) & (time_axis_array <= time_off)

    # Apply scaling
    weighted_noise = full_noise_trace * amp_n
    weighted_ac = ac_trace * amp_b

    # Combine
    # Note: Depending on requirements, you might want the AC to run continuously
    # or only during the pulse. Here I apply the mask to BOTH components.
    total_current = jnp.where(pulse_mask, weighted_noise + weighted_ac, 0.0)

    return total_current


def plot_full_simulation_summary(recorded_voltages, time_axis, dt_global,
                                 spike_threshold=-20.0,
                                 window_size=250.0, overlap=0.95,
                                 f_min=1.0, f_max=100.0,
                                 title_suffix="", figsize=(16, 14), save=False, savename="fig3p.svg",
                                 aperiodic_correct: float = 0.0,
                                 baseline_relative: Optional[Tuple[float, float]] = None,
                                 interval1: Tuple[float, float] = (1, 500),
                                 interval2: Tuple[float, float] = (500, 1000)): # Added baseline_relative
    """
    Combines voltage image, raster plot, and time-frequency response into a single 3x1 subplot figure.
    """
    # Replace NaN values with 0 at the very beginning
    recorded_voltages = jnp.nan_to_num(recorded_voltages, nan=0.0)
    recorded_voltages = jnp.clip(recorded_voltages, -100, +100)

    fig = plt.figure(figsize=figsize) # Adjust figure size for 3 tall subplots

    # Subplot 1: Raster Plot as Image
    raster_ax = plt.subplot(4, 1, 1) # Changed to 3, 1, 1
    num_neurons = recorded_voltages.shape[0]

    # Calculate downsampling factor for spike detection prior to thresholding
    # e.g., if dt_global=0.1ms, factor=10, meaning spikes are detected at ~1ms resolution.
    downsampling_factor = int(jnp.ceil(1.0 / dt_global))
    if downsampling_factor <= 0:
        downsampling_factor = 1

    # Get the voltage data, excluding the initial V_init column
    neuron_traces_data = recorded_voltages[:, 1:] # Shape (num_neurons, original_num_timepoints)
    original_num_timepoints = neuron_traces_data.shape[1]

    # Create a binary spike train for each neuron at full resolution
    full_res_spike_trains = jnp.zeros((num_neurons, original_num_timepoints), dtype=jnp.float32)
    for i in range(num_neurons):
        neuron_trace = recorded_voltages[i, 1:] # Exclude V_init
        spikes_detected = (neuron_trace[:-1] < spike_threshold) & (neuron_trace[1:] >= spike_threshold)
        spike_indices = jnp.where(spikes_detected)[0] + 1 # +1 to mark the point after crossing
        full_res_spike_trains = full_res_spike_trains.at[i, spike_indices].set(1.0)


    # The spike_image creation for raster plotting:
    # Calculate the number of time points in the downsampled raster
    num_timepoints_raster = (original_num_timepoints + downsampling_factor - 1) // downsampling_factor

    # Create a 2D array for the downsampled spike image
    spike_image = jnp.zeros((num_neurons, num_timepoints_raster), dtype=jnp.float32)

    for i in range(num_neurons):
        # Extract and downsample the voltage trace for the current neuron
        # We take samples at intervals of 'downsampling_factor'
        neuron_trace = recorded_voltages[i, 1:]
        neuron_trace_down = neuron_trace[::downsampling_factor]

        # Detect spikes on downsampled
        spikes = (neuron_trace_down[:-1] < spike_threshold) & (neuron_trace_down[1:] >= spike_threshold)

        # Map downsampled spike indices to image
        spike_indices = jnp.where(spikes)[0]
        spike_image = spike_image.at[i, spike_indices].set(1.0)


    # The `time_axis` provided is for the original, full-resolution data. When plotting the
    # downsampled `spike_image`, its visual extent should still cover the full original time range.
    # The `extent` parameter of `imshow` handles this scaling automatically.
    plt.imshow(
        spike_image,
        aspect='auto',
        cmap='Greys',
        origin='lower',
        extent=[time_axis[0], time_axis[-1], 0, num_neurons], # Use original time extent for display
        vmin=0, vmax=1
    )
    # Add a custom colorbar to indicate spikes
    cbar = plt.colorbar(label='Spike Activity', ticks=[0.25, 0.75]) # Position ticks to be in the middle of bands
    cbar.set_ticklabels(['No Spike', 'Spike'])

    plt.title(f'Raster Plot (Image) {title_suffix}')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron Index')

    # Generate y-ticks at approximately 10% intervals
    if num_neurons > 0:
        y_tick_interval = max(1, int(np.ceil(num_neurons / 10)))
        y_ticks = np.arange(0, num_neurons + 1, y_tick_interval)
        # Ensure 0 is always included and last tick is within bounds or is the last neuron
        y_ticks = np.unique(np.clip(y_ticks, 0, num_neurons - 1)).astype(int)
        plt.yticks(y_ticks)
    else:
        plt.yticks([])

    plt.ylim(-0.5, num_neurons - 0.5)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.xlim([time_axis[0], time_axis[-1]])
    plt.axvline(x=500, color='red', linestyle='--', linewidth=1.5)

    # Subplot 2: Time-Frequency Response
    plt.subplot(4, 1, 2)
    recorded_voltages = jnp.nan_to_num(recorded_voltages, nan=0.0, posinf=0.0, neginf=0.0)

    # --- NEW: Calculate recorded_spiking_w and prepare for spectrogram ---
    w = 5.0 # Gaussian kernel width, as per request default
    # Sampling frequency of the SPIKE IMAGE (1 kHz if downsampling_factor ~ 1ms)
    dt_raster = dt_global * downsampling_factor
    fs_raster = 1000.0 / dt_raster

    sigma_samples = w / dt_raster # Sigma in samples of the raster

    # Apply Gaussian filter to each neuron's full-resolution binary spike train
    # Using convolve2d as per user request to avoid ndimage dependency issues
    k_size_gauss = int(7 * sigma_samples + 1)
    x_grid_gauss = jnp.linspace(-3.0 * sigma_samples, 3.0 * sigma_samples, k_size_gauss)
    kernel_1d_gauss = jnp.exp(-x_grid_gauss**2 / (2 * sigma_samples**2))
    kernel_1d_gauss = kernel_1d_gauss / jnp.sum(kernel_1d_gauss) # Normalize
    kernel_2d_gauss = kernel_1d_gauss[None, :]

    recorded_spiking_w = jax.scipy.signal.convolve2d(spike_image, kernel_2d_gauss, mode='same')

    # Compute the mean signal from recorded_spiking_w for spectrogram
    mean_signal_for_spectrogram = jnp.mean(recorded_spiking_w, axis=0)

    # Convert JAX numpy array to numpy array for scipy compatibility
    # And apply uniform filter (smoothing) as in original code for LFP proxy
    fs = fs_raster # Sampling frequency of the raster
    # smoothing_window_samples = int(1.0 / dt_global) # 1ms window for uniform filter
    mean_signal_np = np.asarray(mean_signal_for_spectrogram).astype(np.float64)
    # if smoothing_window_samples > 1:
    #     mean_signal_np = ndimage.uniform_filter1d(mean_signal_np, size=smoothing_window_samples)

    # Apply mirroring to the mean signal before spectrogram calculation
    nperseg = int(window_size * fs / 1000.0)
    if nperseg == 0: nperseg = 1
    half_window_samples = nperseg // 2

    extended_mean_signal_np = mean_signal_np
    if len(mean_signal_np) > half_window_samples:
        mirrored_prefix = mean_signal_np[1 : half_window_samples + 1][::-1]
        extended_mean_signal_np = np.concatenate((mirrored_prefix, extended_mean_signal_np))
    if len(mean_signal_np) > half_window_samples:
        mirrored_suffix = mean_signal_np[-(half_window_samples + 1) : -1][::-1]
        extended_mean_signal_np = np.concatenate((extended_mean_signal_np, mirrored_suffix))

    # Spectrogram parameters and computation
    noverlap = int(nperseg * overlap)
    if noverlap >= nperseg: noverlap = nperseg - 1
    if noverlap < 0: noverlap = 0

    frequencies, times, Sxx = signal.spectrogram(extended_mean_signal_np, fs=fs,
                                                 window='hann', nperseg=nperseg,
                                                 noverlap=noverlap)

    # Adjust the times for plotting to align with the original time_axis
    time_shift_ms = half_window_samples * dt_raster
    adjusted_times = times * 1000 - time_shift_ms

    freq_indices = (frequencies >= f_min) & (frequencies <= f_max)
    frequencies_filtered = frequencies[freq_indices]
    Sxx_filtered = (Sxx[freq_indices, :])

    # Apply aperiodic correction
    if aperiodic_correct != 0.0 and frequencies_filtered.size > 0:
        # Add a small epsilon to avoid division by zero for frequency=0
        Sxx_filtered = Sxx_filtered / (frequencies_filtered[:, jnp.newaxis]**aperiodic_correct + 1e-9)

    # Apply baseline relative normalization if requested
    if baseline_relative is not None and frequencies_filtered.size > 0:
        t_start_ms, t_end_ms = baseline_relative
        # Convert ms to time indices for adjusted_times
        t_start_idx = jnp.argmin(jnp.abs(adjusted_times - t_start_ms))
        t_end_idx = jnp.argmin(jnp.abs(adjusted_times - t_end_ms))

        # Ensure indices are valid and in order
        if t_start_idx > t_end_idx:
            t_start_idx, t_end_idx = t_end_idx, t_start_idx
        # Ensure the slice is not empty
        if t_start_idx == t_end_idx and t_end_idx < Sxx_filtered.shape[1] -1:
             t_end_idx +=1 # Extend slice to at least 2 points
        elif t_start_idx == t_end_idx and t_end_idx > 0:
             t_start_idx -=1 # Extend slice to at least 2 points

        baseline_power_per_freq = jnp.mean(Sxx_filtered[:, t_start_idx:t_end_idx + 1], axis=1, keepdims=True)
        # Avoid division by zero
        Sxx_filtered = Sxx_filtered / (baseline_power_per_freq + 1e-9)

    if frequencies_filtered.size == 0:
        print("Warning: No frequencies found within the specified f_min and f_max range for spectrogram in combined plot.")
    else:

        n_shape_sxx = int(Sxx_filtered.shape[1]*0.8)
        min_val = jnp.min(Sxx_filtered[:n_shape_sxx])
        max_val = jnp.max(Sxx_filtered[:n_shape_sxx])
        Sxx_scaled = (Sxx_filtered - min_val) / (max_val - min_val) if (max_val - min_val) != 0 else jnp.zeros_like(Sxx_filtered)
        Sxx_scaled = jnp.sqrt(Sxx_scaled)

        plt.imshow(Sxx_scaled, aspect='auto', cmap='jet', origin='lower',
                   extent=[adjusted_times[0], adjusted_times[-1], frequencies_filtered[0], frequencies_filtered[-1]])
        plt.colorbar(label='Normalized Power (0-1)')
        plt.title(f'Mean Spectrogram of Individual Neuronal Potentials (Convolved Spikes) {title_suffix}')
        plt.xlabel('Time (ms)')
        plt.ylabel('Frequency (Hz)')
        plt.ylim([frequencies_filtered[0], frequencies_filtered[-1]])
        plt.xlim([time_axis[0], time_axis[-1]])
        plt.axvline(x=500, color='red', linestyle='--', linewidth=1.5)
        red_line_freqs = [3.0, 8.0, 12.0, 20.0, 30.0, 70.0]
        for freq_line in red_line_freqs:
            if frequencies_filtered[0] <= freq_line <= frequencies_filtered[-1]:
                plt.axhline(y=freq_line, color='red', linestyle='--', linewidth=1.0)

    # Subplot 3: Average PSD Interval 1
    plt.subplot(4, 1, 3)
    if frequencies_filtered.size > 0:
        t_mask1 = (adjusted_times >= interval1[0]) & (adjusted_times <= interval1[1])
        if np.any(t_mask1):
            avg_psd1 = jnp.mean(Sxx_filtered[:, t_mask1], axis=1)
            min_psd1 = jnp.min(avg_psd1)
            max_psd1 = jnp.max(avg_psd1)
            avg_psd1_scaled = (avg_psd1 - min_psd1) / (max_psd1 - min_psd1) if (max_psd1 - min_psd1) != 0 else jnp.zeros_like(avg_psd1)

            plt.plot(frequencies_filtered, avg_psd1_scaled, color='blue')
            plt.title(f'Average PSD ({interval1[0]}-{interval1[1]} ms) {title_suffix}')
            plt.ylabel('Norm Power')
            plt.xlim([frequencies_filtered[0], frequencies_filtered[-1]])
            plt.ylim([0, 1.1])
            for freq_line in red_line_freqs:
                if frequencies_filtered[0] <= freq_line <= frequencies_filtered[-1]:
                    plt.axvline(x=freq_line, color='red', linestyle='--', linewidth=1.0)

    # Subplot 4: Average PSD Interval 2
    plt.subplot(4, 1, 4)
    if frequencies_filtered.size > 0:
        t_mask2 = (adjusted_times >= interval2[0]) & (adjusted_times <= interval2[1])
        if np.any(t_mask2):
            avg_psd2 = jnp.mean(Sxx_filtered[:, t_mask2], axis=1)
            min_psd2 = jnp.min(avg_psd2)
            max_psd2 = jnp.max(avg_psd2)
            avg_psd2_scaled = (avg_psd2 - min_psd2) / (max_psd2 - min_psd2) if (max_psd2 - min_psd2) != 0 else jnp.zeros_like(avg_psd2)

            plt.plot(frequencies_filtered, avg_psd2_scaled, color='blue')
            plt.title(f'Average PSD ({interval2[0]}-{interval2[1]} ms) {title_suffix}')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Norm Power')
            plt.xlim([frequencies_filtered[0], frequencies_filtered[-1]])
            plt.ylim([0, 1.1])
            for freq_line in red_line_freqs:
                if frequencies_filtered[0] <= freq_line <= frequencies_filtered[-1]:
                    plt.axvline(x=freq_line, color='red', linestyle='--', linewidth=1.0)

    plt.tight_layout()

    # No separate `subplots_adjust` needed for `right` anymore if using `add_axes` for cbar, just for overall spacing.
    plt.subplots_adjust(right=0.9)

    if save:
        plt.savefig(savename, format='svg')

    plt.show()

                                     
