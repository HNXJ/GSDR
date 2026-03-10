%pip install jaxley

import os
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=16'

from jax import config
config.update("jax_enable_x64", True)
config.update("jax_platform_name", "cpu") # 'gpu' / 'cpu'
# config.update("jax_debug_nans", True)

import jax
import numpy as np
import jaxley as jx
import jax.numpy as jnp
import matplotlib.pyplot as plt
import jaxley.optimize.transforms as jt

from jax import jit, vmap, value_and_grad
from jaxley.channels import Leak, HH
from jaxley.synapses import IonotropicSynapse
from jaxley.connect import fully_connect, sparse_connect, connect
from scipy import ndimage
import scipy
import pickle


figure_save_path = "/content/data/Figures"
model_save_path = "/content/data/NeuralCircuits"
save_dir = "/content/data/SignalArrays/"

# from google.colab import drive
# drive.mount('/content/drive/')
# figure_save_path = "/content/drive/MyDrive/Workspace/Figures" # Drive
# model_save_path = "/content/drive/MyDrive/Workspace/Models" # Drive
# save_dir = "/content/drive/MyDrive/Workspace/" # Drive


file_name = "selected_unit_spiking.npy"

save_path = os.path.join(save_dir, file_name)
selected_unit_spiking = jnp.load(save_path)
print(f"'selected_unit_spiking' loaded from {save_path}")
print(f"Shape of loaded_selected_unit_spiking: {selected_unit_spiking.shape}")




def net_eig(num_e: int, num_ig: int, num_il: int):
    """
    Constructs a JAXley neural network with specified numbers of excitatory and inhibitory neurons.

    Args:
        num_e (int): Number of Excitatory neurons.
        num_ig (int): Number of Global inhibitory interneurons (SST-like).
        num_il (int): Number of Local inhibitory interneurons (PV-like).

    Returns:
        jx.Network: The constructed JAXley network object.
    """
    comp = jx.Compartment()
    branch = jx.Branch(comp, ncomp=2) # Changed to ncomp=2 for distinct pre/post branches

    e_cells = [jx.Cell(branch, parents=[-1, 0]) for _ in range(num_e)]
    i_cells_g = [jx.Cell(branch, parents=[-1, 0]) for _ in range(num_ig)]
    i_cells_l = [jx.Cell(branch, parents=[-1, 0]) for _ in range(num_il)]

    for cell in e_cells + i_cells_g + i_cells_l:
        r_d_tau_diff = np.random.uniform(10.0, 50.0)
        r_amp_diff = np.clip(np.random.uniform(-0.1, 0.1), 0.0, 0.05)
        cell.insert(HH())
        cell.insert(Inoise(initial_amp_noise=r_amp_diff, initial_tau=r_d_tau_diff, initial_mean=0.0))
        cell.radius = 1.0  # uniform constant geometry
        cell.length = 1.0

    net = jx.Network(e_cells + i_cells_g + i_cells_l)
    net.cell(np.arange(0, num_e)).add_to_group("E")
    net.cell(np.arange(num_e,num_e+num_ig)).add_to_group("Ing")
    net.cell(np.arange(num_e+num_ig,num_e+num_ig+num_il)).add_to_group("Inl")

    # 3. Connectivity (All-to-All between populations)
    # E -> all (AMPA)
    pre = net.cell([ik for ik in range(num_e)]).branch(0).loc(0.0) # From branch 0, loc 0.0
    post = net.cell([ik for ik in range(num_e+num_ig+num_il)]).branch(1).loc(1.0) # To branch 1, loc 1.0
    fully_connect(pre, post, GradedAMPA(tauD_AMPA=2.0))

    # Ig -> all (GABAa)
    pre = net.cell([ik for ik in range(num_e, num_e+num_ig)]).branch(0).loc(0.0)
    post = net.cell([ik for ik in range(num_e+num_ig+num_il)]).branch(1).loc(1.0)
    fully_connect(pre, post, GradedGABAa(tauD_GABAa=5.0))

    # Il -> subset of all (GABAa)
    pre_il = net.cell([ik for ik in range(num_e+num_ig, num_e+num_ig+num_il)]).branch(0).loc(0.0)
    num_posts_to_select = int((num_e+num_ig+num_il)*0.1)
    posts_pool_indices = jnp.arange(0, num_e + num_ig + num_il)
    key_conn = jax.random.PRNGKey(1)
    selected_post_indices = np.array(jax.random.choice(key_conn, posts_pool_indices, shape=(num_posts_to_select,), replace=False))
    post_il = net.cell(selected_post_indices).branch(1).loc(1.0) # Connect to branch 1
    fully_connect(pre_il, post_il, GradedGABAa(tauD_GABAa=2.0))

    # net.compute_xyz()
    # net.arrange_in_layers(layers=[num_e + num_ig + num_il], within_layer_offset=100.0, between_layer_offset=100.0)
    # fig, ax = plt.subplots(1, 1, figsize=(6, 12))
    # _ = net.vis(ax=ax, detail="full")

    return net



def get_signal_psd(traces, fs=1000):
  signal = jnp.nanmean(jnp.squeeze(traces), axis=1) # Average voltage across all recorded neurons over time
  # signal = ndimage.uniform_filter1d(signal, size=10)
  N = signal.shape[-1]

  # Compute one-sided FFT and corresponding frequencies for real signals
  signal_fft = jnp.fft.rfft(signal)
  freqs = jnp.fft.rfftfreq(N, 1/fs)
  psd_raw = jnp.abs(signal_fft)**2 / (N * fs)

  target_freqs = global_psd_interval
  interpolated_psd = jnp.interp(target_freqs, freqs, psd_raw)
  interpolated_psd = ndimage.uniform_filter1d(interpolated_psd, size=jnp.ceil(5*jnp.sqrt(fs/1000)).astype(int))

  interpolated_psd = interpolated_psd - jnp.min(interpolated_psd)
  max_psd = jnp.max(interpolated_psd)
  scaled_psd = interpolated_psd / (max_psd + 1e-6)
  return scaled_psd


global_psd_interval = jnp.linspace(15.0, 60.0, 40)
nbin_freq = global_psd_interval.shape[0]
inputs = jnp.array([0.0, 1.0, 1.0])*0.6 # nA stim amp for the three groups
labels = jnp.zeros((inputs.shape[0], nbin_freq))

for ik in range(1):
  base_units_spectrum = get_signal_psd(jnp.nanmean(selected_unit_spiking[5:6, :200, :], 0))
  labels = labels.at[ik].set(base_units_spectrum)

for ik in range(1, 3):
  stim_units_spectrum = get_signal_psd(jnp.nanmean(selected_unit_spiking[5:6, 600:1200, :], 0))
  labels = labels.at[ik].set(stim_units_spectrum)

batch_size = 3
dataloader = Dataset(inputs, labels)
dataloader = dataloader.shuffle(seed=0).batch(batch_size)

# Visualization of the labels (PSDs)
plt.figure(figsize=(10, 6))
freq_axis = global_psd_interval

for i in range(0, labels.shape[0]):
    label_name = f"Baseline" if i < 1 else f"Stim on"
    plt.plot(freq_axis, labels[i], label=label_name)

plt.title("Target Power Spectral Densities (Labels)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Normalized Power")
plt.xlim([5, 60])
plt.ylim([0, 1])
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# temp_trace = np.squeeze(np.mean(selected_unit_spiking[5:6, 0:1500, :], 0)).T
# # temp_trace.shape
# trace_vis_tfr(temp_trace, dt_trace=1.0)




band_definitions = {
    'gamma': (35.0, 60.0),
    'beta': (13.0, 30.0),
    'alpha': (8.0, 12.0),
    'theta': (4.0, 8.0)
}

lower_c = 5.0
upper_c = 50.0
firing_rate_weight = 9.0
psd_weight = 990.0

dt_global = 0.1
t_max_global = 1500
t_on_global = 500

Ne = 36 # Number of Excitatory neurons
Nig = 8 # Number of Global inhibitory interneurons (SST-like)
Nil = 6 # Number of Local inhibitory interneurons (PV-like))
net = net_eig(Ne, Nig, Nil)

net.edges.gAMPA = np.clip(np.random.uniform(-4.0, 10.0, net.edges.gAMPA.shape), 0, 8.0)
net.edges.gGABAa = np.clip(np.random.uniform(2.0, 10.0, net.edges.gGABAa.shape), 0, 10.0)

gAMPA_lower_bounds = 0.00
gAMPA_upper_bounds = 10.00

gGABA_lower_bounds = 0.00
gGABA_upper_bounds = 10.00

# Correctly define transform as ParamTransform for the entire parameter tree
transform = jx.ParamTransform([
    {"gAMPA": ClampTransform(gAMPA_lower_bounds, gAMPA_upper_bounds)},
    {"gGABAa": ClampTransform(gGABA_lower_bounds, gGABA_upper_bounds)}
])

net.delete_recordings()
net.delete_stimuli()
net.delete_trainables()

net.GradedAMPA.edge("all").make_trainable("gAMPA")
net.GradedGABAa.edge("all").make_trainable("gGABAa")
view_ampa = net.GradedAMPA.edge("all")
view_gaba = net.GradedGABAa.edge("all")

# Helper to safely extract column
def get_edge_indices(view, col_candidates):
    for col in col_candidates:
        if col in view.edges.columns:
            return jnp.array(view.edges[col].values.astype(int))
    # If not found, print available columns for debugging and return empty array
    print(f"DEBUG: No candidate column found. Available columns: {list(view.edges.columns)}")
    return jnp.array([]) # Return an empty JAX array instead of raising an error

pre_candidates = ["pre_cell_idx", "pre_cell_index", "pre", "source", "pre_index"]
post_candidates = ["post_cell_idx", "post_cell_index", "post", "target", "post_index"]

ampa_pre_inds = get_edge_indices(view_ampa, pre_candidates)
ampa_post_inds = get_edge_indices(view_ampa, post_candidates)
gaba_pre_inds = get_edge_indices(view_gaba, pre_candidates)
gaba_post_inds = get_edge_indices(view_gaba, post_candidates)


@jit
def compute_correlations(traces, pre_inds, post_inds):
    """
    Computes Mutual Pearson correlation (est., time adjusted) between pre and post traces.
    traces: (Batch, Cells, Time)
    pre_inds, post_inds: (Num_Edges,)
    Returns: (Num_Edges,)
    """
    if pre_inds.size == 0 or post_inds.size == 0:
        return jnp.array([])

    # 1. Compute Raw Correlations
    pre_v = traces[:, pre_inds, :-10]
    post_v = traces[:, post_inds, 10:]

    pre_mean = jnp.mean(pre_v, axis=-1, keepdims=True)
    post_mean = jnp.mean(post_v, axis=-1, keepdims=True)
    pre_c = pre_v - pre_mean
    post_c = post_v - post_mean

    num = jnp.sum(pre_c * post_c, axis=-1)
    den = jnp.sqrt(jnp.sum(pre_c**2, axis=-1) * jnp.sum(post_c**2, axis=-1))
    corr = num**2 / (den + 1e-6) # (Batch, Num_Edges)
    num_total_neurons = traces.shape[1]

    k = int(np.sqrt(num_total_neurons))
    x_grid = jnp.linspace(-2.0, 2.0, k)
    window = jnp.exp(-x_grid**2 / 2.)
    kernel = jnp.outer(window, window)
    kernel = kernel / jnp.sum(kernel)  # Normalize

    def smooth_single_batch(c_flat):
        # Map sparse edges to dense matrix (N_cells, N_cells)
        mat = jnp.zeros((num_total_neurons, num_total_neurons))
        mat = mat.at[pre_inds, post_inds].set(c_flat)

        # Convolve with 2D kernel
        smoothed_mat = jax.scipy.signal.convolve2d(mat, kernel, mode='same')

        # Extract values back to sparse edge list
        return smoothed_mat[pre_inds, post_inds]

    corr_smoothed = jax.vmap(smooth_single_batch)(corr)

    return jnp.mean(corr_smoothed, axis=0)


def calculate_mcdp(traces):
    """
    Calculates normalized MCDP factors for all trainable parameters.
    """
    # Compute raw correlations
    r_ampa = compute_correlations(traces, ampa_pre_inds, ampa_post_inds)
    r_gaba = compute_correlations(traces, gaba_pre_inds, gaba_post_inds)

    # Normalize: (val - mean) / std
    # Note: Normalization is per parameter group
    # Handle cases where r_ampa or r_gaba might be empty
    r_ampa_norm = (r_ampa - jnp.mean(r_ampa)) / (jnp.std(r_ampa) + 1e-6) if r_ampa.size > 0 else jnp.array([])
    r_gaba_norm = (r_gaba - jnp.mean(r_gaba)) / (jnp.std(r_gaba) + 1e-6) if r_gaba.size > 0 else jnp.array([])

    # Return structure matching the params PyTree
    # [{gAMPA: ...}, {gGABAa: ...}]
    return [{'gAMPA': r_ampa_norm}, {'gGABAa': r_gaba_norm}] # , {'amp_noise': jnp.array([0])}


levels = 2
time_points = t_max_global / dt_global + 1
checkpoints = [int(np.ceil(time_points**(1/levels))) for _ in range(levels)]

net.delete_recordings()
net.cell([i for i in range(Ne+Nig+Nil)]).branch(0).loc(0.0).record()

def calculate_firing_rates(traces, spike_threshold=-20.0):
    """
    Calculates the average firing rate (spikes/second) for each neuron in the given voltage traces.

    Args:
        traces (jnp.ndarray): Array of voltage traces with shape (num_batches, num_neurons, num_timepoints).
                              The first column of each neuron's trace is assumed to be V_init and skipped.
        spike_threshold (float): The voltage threshold for detecting a spike (in mV).

    Returns:
        jnp.ndarray: Array of average firing rates (in Hz) for each neuron, with shape (num_batches, num_neurons).
    """

    # Determine the actual simulation duration excluding the initial V_init point
    num_timepoints_actual = traces.shape[-1] - 1
    # Total simulation time in milliseconds (since t_max_global is in ms)
    sim_duration_ms = t_max_global
    # Total simulation time in seconds
    sim_duration_s = sim_duration_ms / 1000.0

    def detect_spikes_for_neuron(neuron_trace):
        # Exclude the initial V_init point from the trace
        neuron_trace_data = neuron_trace[1:]
        # Detect upward crossings of the spike_threshold
        spikes = (neuron_trace_data[:-1] < spike_threshold) & (neuron_trace_data[1:] >= spike_threshold)
        return jnp.sum(spikes)

    # Vmap over neurons to get spike counts for all neurons in a single batch
    # input_axes=(0, None) means the first axis of neuron_traces is mapped, second (spike_threshold) is broadcast
    batched_detect_spikes_for_neurons = vmap(detect_spikes_for_neuron, in_axes=(0))

    # Vmap over batches to get spike counts for all batches
    # input_axes=(0) means the first axis of traces (batches) is mapped
    total_spike_counts_per_neuron_per_batch = vmap(batched_detect_spikes_for_neurons, in_axes=(0))(traces)

    # Calculate firing rate (spikes/second)
    firing_rates_hz = total_spike_counts_per_neuron_per_batch / sim_duration_s

    return firing_rates_hz


def simulate(params, input_amp):
    """
    Simulates the network for a given scalar input amplitude.
    """
    current_amp_nA = input_amp
    ac_currents = noise_current_ac(
        i_delay=t_on_global,
        i_dur=500.0,
        amp_n=0.0,
        amp_b=current_amp_nA,
        delta_t=dt_global,
        t_max=t_max_global,
        spect=jnp.array([120.0])
    )

    net.delete_stimuli()
    data_stimuli = net.cell([ik for ik in range(0, Ne, 2)]).branch(1).loc(0.0).data_stimulate(ac_currents)

    # Integrate the network dynamics.
    traces = jx.integrate(net, params=params, data_stimuli=data_stimuli, checkpoint_lengths=checkpoints)
    return traces


batched_simulate = vmap(simulate, in_axes=(None, 0))


def predict(params, input_amp):
    """
    Predicts a scaled PSD array.
    """
    traces = simulate(params, input_amp)
    signal = jnp.mean(traces[:, 1:], axis=0)

    N = signal.shape[-1]
    fs = 1000.0 / dt_global

    signal_fft = jnp.fft.rfft(signal)
    freqs = jnp.fft.rfftfreq(N, d=dt_global/1000)
    psd_raw = jnp.abs(signal_fft)**2 / (N * fs)

    target_freqs = global_psd_interval
    interpolated_psd = jnp.interp(target_freqs, freqs, psd_raw)

    max_psd = jnp.max(interpolated_psd)
    scaled_psd = interpolated_psd / (max_psd + 1e-6)

    return scaled_psd


def loss(opt_params, inputs, labels):
    """
    Loss function for PSD fitting with firing rate penalty. Returns total loss and traces (auxiliary).
    """
    params = transform.forward(opt_params)

    # Get traces directly via batched_simulate to use for both loss and MCDP
    traces = batched_simulate(params, inputs) # Shape (Batch, Cells, Time)

    # Calculate PSDs from traces for loss
    # Helper to calculate PSD for a single trace (mean over cells)
    def compute_psd_from_trace(trace):
        # trace: (Cells, Time)
        temp_l = int(100/dt_global)
        temp_r = int(1400/dt_global)
        signal = jnp.mean(trace[:, temp_l:temp_r], axis=0)
        N = signal.shape[-1]
        fs = 1000.0 / dt_global
        signal_fft = jnp.fft.rfft(signal)
        freqs = jnp.fft.rfftfreq(N, d=dt_global/1000)
        psd_raw = jnp.abs(signal_fft)**2 / (N * fs)

        target_freqs = global_psd_interval
        interpolated_psd = jnp.interp(target_freqs, freqs, psd_raw)
        max_psd = jnp.max(interpolated_psd)
        return interpolated_psd / (max_psd + 1e-6)

    predictions = vmap(compute_psd_from_trace)(traces)

    epsilon = 1e-6
    # psd_loss_per_sample = jnp.sum(jnp.square(labels * jnp.log((predictions + epsilon) / (labels + epsilon))), axis=1)
    psd_loss_per_sample = jnp.sum(jnp.square(labels * jnp.square((predictions + epsilon) - (labels + epsilon))), axis=1)

    # Calculate firing rates
    firing_rates = calculate_firing_rates(traces)

    # Compute firing rate penalty
    # The penalty is mean(exp(lower_c - F) + exp(F - upper_c)) over all neurons and batches
    penalty_lower = jnp.exp(lower_c - firing_rates)
    penalty_upper = jnp.exp(firing_rates - upper_c)
    firing_rate_penalty = jnp.mean(penalty_lower + penalty_upper) # Mean over all neurons and batches
    # jax.debug.print("Firing Rate Penalty: {}", firing_rate_penalty)

    # Combine PSD loss and firing rate penalty
    total_loss = jnp.mean(psd_loss_per_sample) * psd_weight + firing_rate_weight * firing_rate_penalty

    return total_loss, traces


# JIT the loss function and its gradient for efficiency
# has_aux=True because loss now returns (value, auxiliary_data)
jitted_grad = jit(value_and_grad(loss, argnums=0, has_aux=True))


def compute_unscaled_psd_from_trace(trace, dt_global):
    # Assumes trace is 1D (mean signal over neurons) and already smoothed
    N = trace.shape[-1]
    fs = 1000.0 / dt_global  # dt_global is in ms, fs in Hz

    # trace is already the mean signal over neurons, 1D array of shape (Time,)
    signal = trace

    signal_fft = jnp.fft.rfft(signal)
    freqs = jnp.fft.rfftfreq(N, d=dt_global/1000)
    psd_raw = jnp.abs(signal_fft)**2 / (N * fs)

    target_freqs = jnp.linspace(1.0, 100.0, 100)
    interpolated_psd = jnp.interp(target_freqs, freqs, psd_raw)

    return target_freqs, interpolated_psd


def calculate_psd_bands(psd, freqs, band_defs):
    """
    Calculates the average absolute power within specified frequency bands from PSD data.
    Handles multi-dimensional PSD arrays (e.g. Batch x Cells x Freqs) by averaging.
    Returns a list of powers corresponding to the order of keys in band_defs.
    """
    band_powers = []
    for band_name, (f_min, f_max) in band_defs.items():
        freq_indices = (freqs >= f_min) & (freqs < f_max)

        if not jnp.any(freq_indices):
            avg_power = 0.0
        else:
            # Handle ND arrays by slicing the last dimension (frequencies)
            if psd.ndim > 1:
                band_psd_values = psd[..., freq_indices]
            else:
                band_psd_values = psd[freq_indices]

            # Average over all dimensions (Batch, Cells, Freqs in band)
            avg_power = jnp.mean(band_psd_values)

        band_powers.append(avg_power)

    return band_powers


def net_trainer_temp1(net, optimizer, dataloader, epoch_n=100, initial_params=None, optimal_params=None, training_log=None):

    if initial_params is None:
        initial_params = net.get_parameters()

    if optimal_params is None:
        print("Starting training...")
        opt_params = initial_params
    else:
        print("Continuing training...")
        opt_params = optimal_params

    if training_log is None:
        training_log = {
                "loss": [],
                "gsdr_alpha": [], # Renamed from alpha
                "loss_opt": [],
                "avg_bounded_params": [],
                "std_bounded_params": [],
                "avg_gAMPA": [],
                "std_gAMPA": [],
                "avg_gGABAa": [],
                "std_gGABAa": [],
                "mcdp_factors": [],
                "gamma_band": [],
                "beta_band": [],
                "alpha_band": [],
                "theta_band": []
              }
        pre_cnt = 0

    else:
        pre_cnt = len(training_log["loss"])

    opt_state = optimizer.init(opt_params)
    key = jax.random.PRNGKey(0)

    mid_params = []
    epoch_cnt = 0

    for epoch in range(epoch_n):

        key, step_key = jax.random.split(key)
        epoch_loss_accum = 0.0
        batch_count_for_avg = 0
        current_epoch_had_nan = False

        # Accumulate band powers for the epoch
        epoch_band_powers = [0.0, 0.0, 0.0, 0.0]

        for batch_ind, batch in enumerate(dataloader):
            # print("/")
            current_batch, label_batch = batch
            (loss_val, traces), gradient = jitted_grad(opt_params, current_batch, label_batch)

            # Calculate PSD and Bands for logging
            mean_traces = jnp.mean(traces, axis=1) # (Batch, Time)
            mean_trace_batch = jnp.mean(mean_traces, axis=0) # (Time,)
            freq_t, psd_t = compute_unscaled_psd_from_trace(mean_trace_batch, dt_global)
            band_powers_list = calculate_psd_bands(psd_t, freq_t, band_definitions)
            # print(band_powers_list)
            # print(freq_t)
            # print()

            for i in range(4):
                epoch_band_powers[i] += band_powers_list[i]

            if jnp.isnan(loss_val):
                print(f"Epoch {epoch + pre_cnt}, Batch {batch_ind}: epoch voided due to nan in ODE/PDE. Reverting to last optimal state.")
                current_epoch_had_nan = True
                # GSDR.update_fn handles the reset when value=NaN is passed.
                mcdp_factors = None # No MCDP update on NaN
            else:
                epoch_loss_accum += loss_val
                batch_count_for_avg += 1
                # Calculate MCDP factors from traces
                mcdp_factors = calculate_mcdp(traces)

            current_network_params = transform.forward(opt_params)
            updates, opt_state = optimizer.update(gradient, opt_state,
                params=current_network_params,
                value=loss_val,
                key=step_key,
                mcdp_factors=mcdp_factors) # Pass MCDP factors
            opt_params = optax.apply_updates(opt_params, updates)

        # After all batches in an epoch
        if current_epoch_had_nan:
            epoch_cnt -= 1
            current_avg_loss = jnp.nan
            avg_physical_params = jnp.nan
            std_physical_params = jnp.nan
            avg_gAMPA = jnp.nan
            std_gAMPA = jnp.nan
            avg_gGABAa = jnp.nan
            std_gGABAa = jnp.nan
            avg_band_powers = [jnp.nan]*4
        elif batch_count_for_avg > 0:
            current_avg_loss = epoch_loss_accum / batch_count_for_avg

            # Params Stats
            physical_params = transform.forward(opt_params)
            all_physical_params_flat = []
            gAMPA_vals = []
            gGABAa_vals = []

            for param_group in physical_params:
                for name, param_array in param_group.items():
                    flat = param_array.flatten()
                    all_physical_params_flat.append(flat)
                    if name == 'gAMPA':
                        gAMPA_vals.append(flat)
                    elif name == 'gGABAa':
                        gGABAa_vals.append(flat)

            all_flat = jnp.concatenate(all_physical_params_flat)
            avg_physical_params = jnp.mean(all_flat)
            std_physical_params = jnp.std(all_flat)

            gAMPA_flat = jnp.concatenate(gAMPA_vals) if gAMPA_vals else jnp.array([])
            avg_gAMPA = jnp.mean(gAMPA_flat) if gAMPA_flat.size > 0 else 0.0
            std_gAMPA = jnp.std(gAMPA_flat) if gAMPA_flat.size > 0 else 0.0

            gGABAa_flat = jnp.concatenate(gGABAa_vals) if gGABAa_vals else jnp.array([])
            avg_gGABAa = jnp.mean(gGABAa_flat) if gGABAa_flat.size > 0 else 0.0
            std_gGABAa = jnp.std(gGABAa_flat) if gGABAa_flat.size > 0 else 0.0

            # Average band powers
            avg_band_powers = [p / batch_count_for_avg for p in epoch_band_powers]
        else:
            current_avg_loss = jnp.inf
            avg_physical_params = jnp.nan
            std_physical_params = jnp.nan
            avg_gAMPA = jnp.nan
            std_gAMPA = jnp.nan
            avg_gGABAa = jnp.nan
            std_gGABAa = jnp.nan
            avg_band_powers = [jnp.nan]*4

        if not current_epoch_had_nan:
          print(f"epoch {epoch_cnt}, avg. loss {current_avg_loss:.4f}, alpha {opt_state.a:.4f}, loss_opt {opt_state.loss_opt:.4f}")
          print(f"  Params: Avg={avg_physical_params:.4f}, Std={std_physical_params:.4f}")
          print(f"  gAMPA: Avg={avg_gAMPA:.4f}, Std={std_gAMPA:.4f}")
          print(f"  gGABAa: Avg={avg_gGABAa:.4f}, Std={std_gGABAa:.4f}")

          training_log["loss"].append(current_avg_loss)
          training_log["gsdr_alpha"].append(opt_state.a)
          training_log["loss_opt"].append(opt_state.loss_opt)
          training_log["avg_bounded_params"].append(avg_physical_params)
          training_log["std_bounded_params"].append(std_physical_params)

          training_log["avg_gAMPA"].append(avg_gAMPA)
          training_log["std_gAMPA"].append(std_gAMPA)
          training_log["avg_gGABAa"].append(avg_gGABAa)
          training_log["std_gGABAa"].append(std_gGABAa)

          training_log["mcdp_factors"].append(mcdp_factors)
          training_log["gamma_band"].append(avg_band_powers[0])
          training_log["beta_band"].append(avg_band_powers[1])
          training_log["alpha_band"].append(avg_band_powers[2])
          training_log["theta_band"].append(avg_band_powers[3])

        if epoch_cnt % 10 == 0 and epoch_cnt > 10:
            mid_params.append(transform.forward(opt_params))

        epoch_cnt += 1

    final_params = transform.forward(opt_params)

    return final_params, training_log, mid_params



key = jax.random.PRNGKey(421)
initial_params_0 = net.get_parameters()
model_save_name = "model_sgd_dynamic_a000_3"
optimizer_inner = SDR(learning_rate=1e-1, momentum=0.4)

optimizer = GSDR(
    inner_optimizer=optimizer_inner,
    delta_distribution=jax.random.normal,
    deselection_threshold=3.0,
    a_init=0.0,
    a_dynamic=True,
    lambda_d=1.0,
    checkpoint_n=5,
    mcdp=True
)

final_params_0, log_net_0, mid_params_0 = net_trainer_temp1(net, optimizer, dataloader, 500)
# final_params_0, log_net_0, mid_params_0 = net_trainer_temp1(net, optimizer, dataloader, 20, initial_params=final_params_0, optimal_params=final_params_0, training_log=log_net_0)
save_jnn(model_save_name, model_save_path, net, initial_params_0, mid_params_0, final_params_0, log_net_0)



loaded_nets = []
loaded_initial_params = []
loaded_mid_params = []
loaded_final_params = []
loaded_log_nets = []
loaded_log_names = []
cnt = 0

# Sort files for consistent ordering
files = sorted([f for f in os.listdir(model_save_path) if f.startswith("refn_")])

for file in files:
    try:
        loaded_net, loaded_initial_param, loaded_mid_param, loaded_final_param, loaded_log_net = load_jnn(file, model_save_path)
        loaded_nets.append(loaded_net)
        loaded_initial_params.append(loaded_initial_param)
        loaded_mid_params.append(loaded_mid_param)
        loaded_final_params.append(loaded_final_param)
        loaded_log_nets.append(loaded_log_net)
        loaded_log_names.append(file)
        # print(f"Loaded {file}")
    except Exception as e:
        print(f"Error loading {file}: {e}")




def plot_trajectory_bandpowers(log_nets, save_fname=None, band1=None, band2=None, lim1=None, lim2=None, ksw=1):

    if band1 is None:
        band1 = 'gamma_band'
    if band2 is None:
        band2 = 'beta_band'

    if lim1 is None:
        lim1 = (0, 2)
    if lim2 is None:
        lim2 = (0, 2)

    fig, axes = plt.subplots(1, 1, figsize=(10, 10))

    for i, log_net in enumerate(log_nets):

        if 'gsdr_alpha' in log_net:
            alph_t = log_net['gsdr_alpha'][0]
            alpha_key = 'gsdr_alpha'
        elif 'alpha' in log_net:
            alph_t = log_net['alpha'][0]
            alpha_key = 'alpha'
        else:
            alph_t = 0.0
            alpha_key = None

        label = loaded_log_names[i]


        if band1 in log_net and len(log_net[band1]) > 0:

            a_params = jnp.array(log_net[band1])
            b_params = jnp.array(log_net[band2])

            if len(a_params) > ksw:
                a_params = ndimage.uniform_filter1d(a_params, ksw)
                b_params = ndimage.uniform_filter1d(b_params, ksw)

            if a_params[0] != 0:
                a_params = a_params / a_params[0]
            if b_params[0] != 0:
                b_params = b_params / b_params[0]

            color = plt.colormaps['turbo'](i / len(log_nets))

            for j in range(len(a_params) - 1):
                axes.annotate(
                    '', xy=(a_params[j+1], b_params[j+1]), xytext=(a_params[j], b_params[j]),
                    arrowprops=dict(arrowstyle='->', color=color, linewidth=0.5)
                )

            axes.plot(a_params[0], b_params[0], marker='x', markersize=10, color=(0, 0, 0), label=None)
            axes.plot(a_params[-1], b_params[-1], marker='o', markersize=10, color=color, label=f'{i}: {label}')
            axes.text(a_params[-1], b_params[-1], str(i), fontsize=12, fontweight='bold', color='black')

    axes.set_title('RelParamsChange')
    axes.set_xlabel(band1)
    axes.set_ylabel(band2)

    # axes.plot(jnp.linspace(0, 30, 10), jnp.linspace(0, 30, 10), linestyle="--", label=None, color=(0, 0, 0))
    axes.set_xlim(lim1)
    axes.set_ylim(lim2)

    axes.legend(loc='lower left')
    axes.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()

    if save_fname is not None:
        plt.savefig(save_fname, format='svg')

    plt.show()


def plot_trajectory_ratio(log_nets, save_fname=None, band11=None, band12=None, band21=None, band22=None, lim1=None, lim2=None, ksw=1):

    if lim1 is None:
        lim1 = (0, 2)
    if lim2 is None:
        lim2 = (0, 2)

    fig, axes = plt.subplots(1, 1, figsize=(10, 10))

    for i, log_net in enumerate(log_nets):

        if 'gsdr_alpha' in log_net:
            alph_t = log_net['gsdr_alpha'][0]
            alpha_key = 'gsdr_alpha'
        elif 'alpha' in log_net:
            alph_t = log_net['alpha'][0]
            alpha_key = 'alpha'
        else:
            alph_t = 0.0
            alpha_key = None

        label = loaded_log_names[i]

        if band11 in log_net and len(log_net[band11]) > 0:

            a_params = jnp.array(log_net[band11])
            b_params = jnp.array(log_net[band12])
            c_params = jnp.array(log_net[band21])
            d_params = jnp.array(log_net[band22])

            if len(a_params) > ksw:
                a_params = ndimage.uniform_filter1d(a_params, ksw)
                b_params = ndimage.uniform_filter1d(b_params, ksw)
                c_params = ndimage.uniform_filter1d(c_params, ksw)
                d_params = ndimage.uniform_filter1d(d_params, ksw)

            # Calculate Ratios element-wise
            # Add epsilon to denominator to prevent division by zero
            ratio_x = a_params / (b_params + 1e-9)
            ratio_y = c_params / (d_params + 1e-9)

            # Normalize Ratios to start at 1.0
            if ratio_x[0] != 0:
                ratio_x = ratio_x / ratio_x[0]
            if ratio_y[0] != 0:
                ratio_y = ratio_y / ratio_y[0]

            color = plt.colormaps['turbo'](i / len(log_nets))

            for j in range(len(ratio_x) - 1):
                axes.annotate(
                    '', xy=(ratio_x[j+1], ratio_y[j+1]), xytext=(ratio_x[j], ratio_y[j]),
                    arrowprops=dict(arrowstyle='->', color=color, linewidth=0.5)
                )

            axes.plot(ratio_x[0], ratio_y[0], marker='x', markersize=10, color=(0, 0, 0), label=None)
            axes.plot(ratio_x[-1], ratio_y[-1], marker='o', markersize=10, color=color, label=f'{i}: {label}')
            axes.text(ratio_x[-1], ratio_y[-1], str(i), fontsize=12, fontweight='bold', color='black')

    axes.set_title('Relative Parameter Ratio Change')
    axes.set_xlabel(f"{band11} / {band12} (Normalized)")
    axes.set_ylabel(f"{band21} / {band22} (Normalized)")

    axes.set_xlim(lim1)
    axes.set_ylim(lim2)

    axes.legend(loc='lower right')
    axes.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()

    if save_fname is not None:
        plt.savefig(save_fname, format='svg')

    plt.show()



def plot_training_logs(log_nets, sweep_data=None, save_fname=None):

    fig, axes = plt.subplots(4, 1, figsize=(20, 32))
    
    final_stats = [] # Store data for post-loop annotation
    model_optimals = [] # Store (name, optimal_loss)

    total_models = len(log_nets) + (len(sweep_data) if sweep_data else 0)
    current_idx = 0

    # --- Process Training Logs ---
    for i, log_net in enumerate(log_nets):

        if 'gsdr_alpha' in log_net:
            alph_t = log_net['gsdr_alpha'][0]
            alpha_key = 'gsdr_alpha'
        elif 'alpha' in log_net:
            alph_t = log_net['alpha'][0]
            alpha_key = 'alpha'
        else:
            alph_t = 0.0
            alpha_key = None

        if alpha_key and len(log_net[alpha_key]) > 10 and alph_t != log_net[alpha_key][10]:
             label = f"{current_idx}: {loaded_log_names[i]}"
        else:
             label = f"{current_idx}: Static Alph-{alph_t}"

        epochs = jnp.arange(len(log_net['loss']))
        loss_data = jnp.array(log_net['loss'])
        loss_data = jnp.where(jnp.isfinite(loss_data), loss_data, 1e-6) # Replace inf/nan
        loss_t = loss_data

        smoothing_window = 5
        color = plt.colormaps['turbo'](current_idx / total_models)

        # Plot 1: Instantaneous Loss
        if len(loss_t) > smoothing_window:
            loss_tm = ndimage.uniform_filter1d(loss_t, size=smoothing_window)
            std_loss_t = ndimage.uniform_filter1d(jnp.abs(loss_t - loss_tm), size=smoothing_window*3)

            axes[0].plot(epochs, loss_tm, label=label, color=color)
            axes[0].fill_between(epochs, loss_tm - std_loss_t, loss_tm + std_loss_t, alpha=0.1, color=color)
        else:
             axes[0].plot(epochs, loss_t, label=label, color=color)

        axes[0].set_yscale('log')
        axes[0].set_title('Loss per Trial')
        axes[0].set_ylabel('Loss')
        axes[0].set_yticks([10000, 5000, 2500, 1000, 750, 500])
        axes[0].get_yaxis().set_major_formatter(plt.ScalarFormatter())
        axes[0].set_ylim(500, 7500)

        # Plot 2: Optimal Loss up to Trial
        cum_min_loss = np.minimum.accumulate(loss_data)
        axes[1].plot(epochs, cum_min_loss, label=label, color=color, linestyle='-', linewidth=4.0)

        if len(cum_min_loss) > 0:
            final_val = cum_min_loss[-1]
            final_stats.append({
                'val': final_val,
                'color': color,
                'id': current_idx,
                'x': epochs[-1]
            })
            
            last_100_avg = np.mean(loss_data[-100:]) if len(loss_data) >= 100 else np.mean(loss_data)
            model_optimals.append({
                'name': loaded_log_names[i],
                'optimal_loss': float(final_val),
                'last100_loss': float(last_100_avg),
                'color': color,
                'id': current_idx
            })
        
        current_idx += 1

    # --- Process Parameter Sweeps ---
    if sweep_data:
        for j, sweep in enumerate(sweep_data):
            label = f"{current_idx}: {sweep['file_name']}"
            color = plt.colormaps['turbo'](current_idx / total_models)
            
            # Extract losses (handle tuples from loss function output if present)
            raw_losses = sweep['log_loss']
            clean_losses = []
            for l in raw_losses:
                if isinstance(l, (tuple, list)):
                    clean_losses.append(float(l[0]))
                else:
                    clean_losses.append(float(l))
            
            loss_data = jnp.array(clean_losses)
            loss_data = jnp.where(jnp.isfinite(loss_data), loss_data, 1e-6)
            
            # Sort descending to mimic a convergence trajectory
            loss_data = jnp.sort(loss_data)[::-1]
            epochs = jnp.arange(len(loss_data))
            
            # Plot 1: Sorted Loss (mimics trajectory)
            # axes[0].plot(epochs, loss_data, label=label, color=color, linestyle='--')
            
            # Plot 2: Cumulative Min (tracks the sorted curve)
            cum_min_loss = np.minimum.accumulate(loss_data)
            # axes[1].plot(epochs, cum_min_loss, label=label, color=color, linestyle='--', linewidth=4.0)
            
            if len(cum_min_loss) > 0:
                # Do not add to final_stats to avoid text labels on subplot 2
                # final_stats.append({ ... })
                
                loss_np = np.array(loss_data)
                
                # 1. Top 5 minimums (data is sorted desc, so last 5)
                top_5_mins = loss_np[-5:]
                
                # 2. 5 chunks of 20 means
                if loss_np.shape[0] == 100:
                    chunk_means = loss_np.reshape(5, 20).mean(axis=1)
                else:
                    # Fallback if not exactly 100 (unlikely based on description)
                    chunk_means = np.array([np.mean(loss_np)] * 5)

                for k in range(5):
                    model_optimals.append({
                        'name': sweep['file_name'], 
                        'optimal_loss': float(top_5_mins[k]),
                        'last100_loss': float(chunk_means[k]),
                        'color': color,
                        'id': current_idx
                    })
            
            current_idx += 1
    else:
        print("DEBUG: No sweep_data provided or empty.")

    # Configure Axes 0 & 1
    for ax in axes[:2]:
        ax.set_yscale('log')
        ax.set_yticks([10000, 5000, 2500, 1000, 750, 500])
        ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_ylim(500, 7500)
    
    axes[0].set_title('Loss per Trial')
    axes[0].set_ylabel('Loss')
    axes[1].set_title('Optimal Loss per Trial (Log Scale)')
    axes[1].set_ylabel('Min Loss')

    # Sort stats and place evenly spaced text for Subplot 2
    if final_stats:
        final_stats.sort(key=lambda x: x['val'])
        y_min_log = np.log10(500)
        y_max_log = np.log10(7500)
        log_positions = np.linspace(y_min_log, y_max_log, len(final_stats) + 2)[1:-1]
        y_positions = 10**log_positions

        for idx, item in enumerate(final_stats):
            axes[1].text(1.02, y_positions[idx], f"{item['id']}. {item['val']:.0f}", 
                         color=item['color'], fontsize=12, fontweight='bold', 
                         va='center', transform=axes[1].get_yaxis_transform())

    axes[1].set_yscale('log')
    axes[1].set_title('Optimal Loss per Trial (Log Scale)')
    axes[1].set_ylabel('Min Loss')
    axes[1].set_yticks([10000, 5000, 2500, 1000, 750, 500])
    axes[1].get_yaxis().set_major_formatter(plt.ScalarFormatter())
    axes[1].set_ylim(500, 7500)

    # --- Group Statistics Logic ---
    # Reordered Groups
    groups = {
        "No SSa": [],
        "SGD": [],
        "Dynamic SSa": [],
        "Alpha 0.2, 0.4": [],
        "Alpha 0.6, 0.8": [],
        "Alpha 1.0": [],
        "Parameter Sweep": []
    }
    
    for m in model_optimals:
        name = m['name']
        # 1. No SSa (Priority check)
        if "a000" in name and "dynamic" not in name and "simulation_logs" not in name:
             groups["No SSa"].append(m)
             continue
             
        # 2. SGD
        if "model_sgd" in name:
            groups["SGD"].append(m)
            
        # 3. Dynamic
        if "dynamic" in name:
            groups["Dynamic SSa"].append(m)
            
        # 4. Alpha 0.2, 0.4
        if any(x in name for x in ["static_a002", "static_a004"]): 
            groups["Alpha 0.2, 0.4"].append(m)
            
        # 5. Alpha 0.6, 0.8
        if any(x in name for x in ["static_a006", "static_a008"]): 
            groups["Alpha 0.6, 0.8"].append(m)
            
        # 6. Alpha 1.0
        if "static_a010" in name: 
            groups["Alpha 1.0"].append(m)
            
        # 7. Parameter Sweep
        if "simulation_logs" in name: 
            groups["Parameter Sweep"].append(m)
        
        # Removed "All" group as requested
             
    group_names = list(groups.keys())
    
    # --- Helper to plot stats ---
    def plot_stats_on_ax(ax, means, stds, mins, maxs, data_key, ylabel, title, labels, data_points_list):
        x_pos = np.arange(len(labels))

        # 1. Plot Min/Max Range (Line bars with wider caps)
        # yerr should be shape (2, N) where row 0 is (mean-min) and row 1 is (max-mean)
        yerr_range = np.array([means - mins, maxs - means])
        
        ax.errorbar(x_pos, means, yerr=yerr_range, fmt='none', 
                    ecolor='black', elinewidth=2, capsize=10, label='Min / Max', zorder=1)

        # 2. Plot +/- 1 STD (Rectangular bars)
        rect_width = 0.3
        # Ensure bottom is positive for log scale safety
        bar_bottom = np.maximum(means - stds, 1e-6)
        bar_height = (means + stds) - bar_bottom
        
        ax.bar(x_pos, height=bar_height, bottom=bar_bottom, width=rect_width, 
               color='lightgray', edgecolor='black', alpha=0.5, label='Mean ± 1 STD', zorder=2)
        
        # 3. Plot Mean Points
        ax.scatter(x_pos, means, s=100, marker='D', color='black', zorder=5, label='Group Mean')
        
        # 4. Plot Individual Points
        for idx, items in enumerate(data_points_list):
            if not items: continue
            ys = [it[data_key] for it in items]
            colors = [it['color'] for it in items]
            xs = np.random.normal(idx, 0.05, size=len(ys))
            ax.scatter(xs, ys, c=colors, s=50, alpha=0.8, zorder=4, edgecolors='black')
            
        ax.set_title(title)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=15, ha='right', fontsize=12)
        ax.set_ylabel(ylabel)
        ax.set_yscale('log') 
        ax.grid(True, axis='y', linestyle='--', alpha=0.6)

    # --- Subplot 3: Group Statistics (Optimal Loss) ---
    # Process ALL groups for Subplot 3
    group_means_opt = []
    group_stds_opt = []
    group_mins_opt = []
    group_maxs_opt = []
    group_data_points_opt = []
    
    for g_name in group_names:
        items = groups[g_name]
        if not items:
            group_means_opt.append(np.nan)
            group_stds_opt.append(np.nan)
            group_mins_opt.append(np.nan)
            group_maxs_opt.append(np.nan)
            group_data_points_opt.append([])
            continue
            
        vals = [item['optimal_loss'] for item in items]
        group_means_opt.append(np.mean(vals))
        group_stds_opt.append(np.std(vals))
        group_mins_opt.append(np.min(vals))
        group_maxs_opt.append(np.max(vals))
        group_data_points_opt.append(items)
        
    plot_stats_on_ax(axes[2], np.array(group_means_opt), np.array(group_stds_opt), 
                     np.array(group_mins_opt), np.array(group_maxs_opt), 
                     'optimal_loss', 'Optimal Loss', 'Group Analysis: Optimal Loss', 
                     group_names, group_data_points_opt)
    
    # Add Improvement Arrow for Subplot 3
    try:
        if "Parameter Sweep" in group_names and "Alpha 0.2, 0.4" in group_names:
            sw_idx = group_names.index("Parameter Sweep")
            hi_idx = group_names.index("Alpha 0.2, 0.4")
            
            y_sw = group_means_opt[sw_idx]
            y_hi = group_means_opt[hi_idx]
            
            if not np.isnan(y_sw) and not np.isnan(y_hi):
                axes[2].annotate("Improvement",
                    xy=(hi_idx, y_hi), xycoords='data',
                    xytext=(sw_idx, y_sw), textcoords='data',
                    arrowprops=dict(arrowstyle="->", color="green", linewidth=2, connectionstyle="arc3,rad=-0.3"),
                    color="green", fontsize=12, fontweight='bold', ha='center', va='bottom'
                )
    except ValueError:
        pass

    # --- Subplot 4: Group Statistics (Last 100 Avg Loss) ---
    # Process Filtered groups for Subplot 4 (exclude Parameter Sweep)
    group_names_l100 = [g for g in group_names if g != "Parameter Sweep"]
    
    group_means_l100 = []
    group_stds_l100 = []
    group_mins_l100 = []
    group_maxs_l100 = []
    group_data_points_l100 = []
    
    for g_name in group_names_l100:
        items = groups[g_name]
        if not items:
            group_means_l100.append(np.nan)
            group_stds_l100.append(np.nan)
            group_mins_l100.append(np.nan)
            group_maxs_l100.append(np.nan)
            group_data_points_l100.append([])
            continue
            
        vals = [item['last100_loss'] for item in items]
        group_means_l100.append(np.mean(vals))
        group_stds_l100.append(np.std(vals))
        group_mins_l100.append(np.min(vals))
        group_maxs_l100.append(np.max(vals))
        group_data_points_l100.append(items)

    plot_stats_on_ax(axes[3], np.array(group_means_l100), np.array(group_stds_l100), 
                     np.array(group_mins_l100), np.array(group_maxs_l100), 
                     'last100_loss', 'Avg Loss', 'Group Analysis: Avg Loss (Last 100 Trials)', 
                     group_names_l100, group_data_points_l100)

    for ax in axes[:2]:
        ax.legend(loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()

    if save_fname is not None:
        plt.savefig(save_fname, format='svg')

    plt.show()

plot_training_logs(loaded_log_nets, sweep_data=all_loaded_data, save_fname="GSDR_alpha_logs_rw.svg")
