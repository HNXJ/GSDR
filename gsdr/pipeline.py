import jax
import jax.numpy as jnp
import jaxley as jx
import optax
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple
from .analysis import calculate_firing_rates, compute_psd, calculate_mcdp, calculate_psd_bands, compute_unscaled_psd_from_trace, compute_kappa
from .simulation import noise_current_ac

def get_loss_fn(net, transform, dt_global, global_psd_interval, 
                lower_c, upper_c, firing_rate_weight, psd_weight,
                num_e, checkpoints, kappa_weight=100.0):
    """
    Returns a configured loss function targeting Gamma during stim, 1/f off-stim, and minimal overall Kappa.
    """
    def simulate_wrapper(params, input_amp):
        ac_currents = noise_current_ac(
            i_delay=500.0, i_dur=500.0, amp_n=0.0, amp_b=input_amp,
            spect=jnp.array([120.0]), delta_t=dt_global, t_max=1500.0
        )
        net.delete_stimuli()
        net.delete_recordings()
        data_stimuli = net.cell(list(range(0, num_e, 2))).branch(1).loc(0.0).data_stimulate(ac_currents)
        net.cell("all").branch(0).loc(0.0).record()
        return jx.integrate(net, params=params, data_stimuli=data_stimuli, checkpoint_lengths=checkpoints)

    batched_simulate = jax.vmap(simulate_wrapper, in_axes=(None, 0))

    def loss_fn(opt_params, inputs, labels):
        # labels should now contain [target_stim, target_off]
        target_stim = labels[:, 0, :]
        target_off = labels[:, 1, :]

        params = transform.forward(opt_params)
        traces = batched_simulate(params, inputs)
        fs = 1000.0 / dt_global

        def compute_psd_scaled(trace, t_l, t_r):
            signal = jnp.mean(trace[:, t_l:t_r], axis=0)
            _, psd = compute_psd(signal, dt_global, target_freqs=global_psd_interval)
            return psd / (jnp.max(psd) + 1e-6)

        def get_trace_loss(trace, t_stim, t_off):
            # PSDs
            psd_pre = compute_psd_scaled(trace, int(100/dt_global), int(500/dt_global))
            psd_stim = compute_psd_scaled(trace, int(500/dt_global), int(1000/dt_global))
            psd_post = compute_psd_scaled(trace, int(1000/dt_global), int(1400/dt_global))

            loss_pre = jnp.sum(jnp.square(t_off * jnp.square(psd_pre - t_off)))
            loss_stim = jnp.sum(jnp.square(t_stim * jnp.square(psd_stim - t_stim)))
            loss_post = jnp.sum(jnp.square(t_off * jnp.square(psd_post - t_off)))
            
            # Kappas
            threshold = -20.0
            spikes = (trace[:, :-1] < threshold) & (trace[:, 1:] >= threshold)
            spike_matrix = jnp.zeros_like(trace).at[:, 1:].set(spikes.astype(jnp.float32))
            
            k_pre = compute_kappa(spike_matrix[:, int(100/dt_global):int(500/dt_global)], fs)
            k_stim = compute_kappa(spike_matrix[:, int(500/dt_global):int(1000/dt_global)], fs)
            k_post = compute_kappa(spike_matrix[:, int(1000/dt_global):int(1400/dt_global)], fs)
            
            kappa_total = jnp.abs(k_pre) + jnp.abs(k_stim) + jnp.abs(k_post)
            psd_total = loss_pre + loss_stim + loss_post
            
            return psd_total, kappa_total

        batched_losses = jax.vmap(get_trace_loss)(traces, target_stim, target_off)
        psd_loss = jnp.mean(batched_losses[0])
        kappa_loss = jnp.mean(batched_losses[1])

        # 3. Firing Rate Penalty
        firing_rates = calculate_firing_rates(traces, dt_global)
        penalty = jnp.mean(jnp.exp(lower_c - firing_rates) + jnp.exp(firing_rates - upper_c))
        
        total_loss = psd_loss * psd_weight + kappa_loss * kappa_weight + penalty * firing_rate_weight
        return total_loss, traces

    return loss_fn

def get_scz_loss_fn(net, target_psd, dt_global=0.1, ra=100.0):
    """
    Returns a loss function targeting MEG dipole spectra (Axial Current).
    """
    from .analysis import calculate_axial_current

    def loss_fn(opt_params):
        # 1. Simulate
        # Note: ScZ model uses multi-compartment cells. 
        # net.cell('all').branch(0) is Soma, branch(1) is Dendrite.
        net.delete_stimuli()
        net.delete_recordings()
        
        # Stimulate Soma of E-cells (indices 0 to 35)
        ac_currents = noise_current_ac(
            i_delay=500.0, i_dur=500.0, amp_n=0.0, amp_b=0.2,
            spect=jnp.array([120.0]), delta_t=dt_global, t_max=1500.0
        )
        net.cell(range(36)).branch(0).loc(0.0).data_stimulate(ac_currents)
        
        # Record from Soma and Dendrite of E-cells
        net.cell(range(36)).branch(0).loc(0.0).record()
        net.cell(range(36)).branch(1).loc(1.0).record()
        
        traces = jx.integrate(net, params=opt_params, delta_t=dt_global, t_max=1500.0)
        
        # 2. Calculate Axial Current (Dipole)
        # traces shape: (num_recordings, num_time_points)
        # indices 0-35 are Soma, 36-71 are Dendrite
        traces_soma = traces[:36, :]
        traces_dend = traces[36:, :]
        
        i_axial = calculate_axial_current(traces_soma, traces_dend, ra=ra)
        pop_dipole = jnp.mean(i_axial, axis=0) # Population average dipole
        
        # 3. PSD of Dipole
        # FFT of the stimulus window (500-1000ms)
        start_idx, end_idx = int(500/dt_global), int(1000/dt_global)
        window_signal = pop_dipole[start_idx:end_idx]
        
        sim_psd = jnp.abs(jnp.fft.rfft(window_signal))**2
        # Normalize
        sim_psd = sim_psd / (jnp.max(sim_psd) + 1e-6)
        
        # Trim target_psd to match sim_psd shape if necessary
        t_psd = target_psd[:sim_psd.shape[0]]
        
        loss = jnp.mean(jnp.square(sim_psd - t_psd))
        return loss, traces

    return loss_fn

def train_net(net, optimizer, transform, dataloader, loss_fn, 
              ampa_pre_inds, ampa_post_inds, gaba_pre_inds, gaba_post_inds,
              dt_global, band_definitions, epoch_n=100, initial_params=None):
    """
    Main training loop for the NetEIG model using GSDR.
    """
    if initial_params is None:
        opt_params = net.get_parameters()
    else:
        opt_params = initial_params
        
    opt_state = optimizer.init(opt_params)
    seed = int(np.random.randint(0, 2**31 - 1))
    key = jax.random.PRNGKey(seed)
    jitted_grad = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))

    training_log = {k: [] for k in ["loss", "alpha", "avg_gAMPA", "avg_gGABAa", "gamma", "beta", "alpha_band", "theta"]}

    for epoch in range(epoch_n):
        key, step_key = jax.random.split(key)
        for batch in dataloader:
            inputs, labels = batch
            (loss_val, traces), grads = jitted_grad(opt_params, inputs, labels)
            
            if jnp.isnan(loss_val):
                mcdp_factors = None
            else:
                mcdp_factors = calculate_mcdp(traces, ampa_pre_inds, ampa_post_inds, gaba_pre_inds, gaba_post_inds)

            updates, opt_state = optimizer.update(grads, opt_state, params=transform.forward(opt_params), 
                                                 value=loss_val, key=step_key, mcdp_factors=mcdp_factors)
            opt_params = optax.apply_updates(opt_params, updates)

        # Logging logic (simplified for package)
        params_f = transform.forward(opt_params)
        print(f"Epoch {epoch}: Loss {loss_val:.4f}")
        training_log["loss"].append(float(loss_val))

    return transform.forward(opt_params), training_log
