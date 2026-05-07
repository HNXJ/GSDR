#!/usr/bin/env python
"""
JAX-native differentiable spectral loss functions.

Key insight: Empirical spectrum is below Nyquist (5–100 Hz target).
No need for scipy.signal.welch in loss computation.
Use soft spike proxies for gradient flow through voltage.
"""

import jax
import jax.numpy as jnp


# ==============================================================================
# Soft Spike Proxy (Differentiable Alternative to Hard Spikes)
# ==============================================================================

def soft_spike_rate_from_voltage(
    v_trace: jnp.ndarray,
    threshold: float = -20.0,
    temperature: float = 2.0,
) -> jnp.ndarray:
    """
    Differentiable soft spike rate from voltage trace.

    Args:
      v_trace: shape (T, N) — voltage per neuron over time
      threshold: voltage spike threshold (approx -20 mV for Izh)
      temperature: sigmoid sharpness (lower = sharper spike-like)

    Returns:
      soft_spikes: shape (T, N), range [0, 1], differentiable w.r.t. v_trace
    """
    return jax.nn.sigmoid((v_trace - threshold) / temperature)


# ==============================================================================
# Time Pooling
# ==============================================================================

def downsample_time_mean(x: jnp.ndarray, factor: int) -> jnp.ndarray:
    """
    Downsample by averaging (for rate signals).

    x: (..., T, ...)
    factor: e.g. 10 to downsample 10kHz -> 1kHz
    """
    t = x.shape[-2]
    usable = (t // factor) * factor
    x = x[..., :usable, :]

    # Reshape to (batch, usable//factor, factor, neurons)
    new_shape = x.shape[:-2] + (usable // factor, factor, x.shape[-1])
    x_reshaped = x.reshape(new_shape)

    return jnp.mean(x_reshaped, axis=-2)


def downsample_time_sum(x: jnp.ndarray, factor: int) -> jnp.ndarray:
    """
    Downsample by summing (for spike counts).

    x: (T, N) or (T,)
    factor: downsampling factor
    """
    if x.ndim == 1:
        t = x.shape[0]
        usable = (t // factor) * factor
        x = x[:usable]
        return jnp.sum(x.reshape(usable // factor, factor), axis=1)
    else:
        t = x.shape[0]
        usable = (t // factor) * factor
        x = x[:usable]
        new_shape = (usable // factor, factor) + x.shape[1:]
        return jnp.sum(x.reshape(new_shape), axis=1)


# ==============================================================================
# JAX-Native PSD Computation
# ==============================================================================

def normalize_spectrum(psd: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    """Normalize PSD to [0, 1] range."""
    psd = jnp.nan_to_num(psd, nan=0.0, posinf=0.0, neginf=0.0)
    psd = jnp.maximum(psd, 0.0)  # Ensure non-negative
    psd_min = jnp.min(psd)
    psd_centered = psd - psd_min
    psd_max = jnp.max(psd_centered)
    return psd_centered / (psd_max + eps)


def jax_psd_rfft(
    signal: jnp.ndarray,
    fs_hz: float = 1000.0,
    target_freqs: jnp.ndarray = None,
    eps: float = 1e-8,
) -> jnp.ndarray:
    """
    Single-window one-sided PSD using JAX rFFT. Fully differentiable.

    Args:
      signal: shape (T,) — 1D time series
      fs_hz: sampling rate (Hz)
      target_freqs: shape (F,) — frequency bins to interpolate to
      eps: numerical stability

    Returns:
      psd: shape (F,) — normalized power spectral density
    """
    signal = signal - jnp.mean(signal)
    n = signal.shape[0]

    # Hann window for spectral leakage reduction
    window = jnp.hanning(n)
    signal_w = signal * window

    # Real FFT
    fft = jnp.fft.rfft(signal_w)
    freqs = jnp.fft.rfftfreq(n, d=1.0 / fs_hz)

    # Window energy correction
    scale = fs_hz * jnp.sum(window**2) + eps
    psd = (jnp.abs(fft) ** 2) / scale

    # Interpolate to target frequencies
    if target_freqs is not None:
        psd = jnp.interp(target_freqs, freqs, psd)

    return normalize_spectrum(psd, eps=eps)


def frame_signal_1d(x: jnp.ndarray, frame_len: int, hop: int) -> jnp.ndarray:
    """
    Frame a 1D signal for Welch-style averaging.

    Args:
      x: shape (T,)
      frame_len: window length (samples)
      hop: hop size (samples)

    Returns:
      frames: shape (n_frames, frame_len)
    """
    n = x.shape[0]
    n_frames = (n - frame_len) // hop + 1
    starts = jnp.arange(n_frames) * hop
    return jax.vmap(lambda s: jax.lax.dynamic_slice(x, (s,), (frame_len,)))(starts)


def jax_welch_psd(
    signal: jnp.ndarray,
    fs_hz: float = 1000.0,
    target_freqs: jnp.ndarray = None,
    frame_ms: float = 500.0,
    hop_ms: float = 50.0,
    eps: float = 1e-8,
) -> jnp.ndarray:
    """
    Welch PSD using overlapping windows. Fully differentiable.

    For 5–100 Hz target, use frame_ms=500 (gives ~2 Hz resolution).
    For 100+ Hz, frame_ms=200 is okay (~5 Hz resolution).

    Args:
      signal: shape (T,)
      fs_hz: sampling rate
      target_freqs: shape (F,), e.g. linspace(5, 100, 100)
      frame_ms: window duration (milliseconds)
      hop_ms: overlap duration (milliseconds)
      eps: numerical stability

    Returns:
      psd: shape (F,) — normalized averaged PSD
    """
    signal = signal - jnp.mean(signal)

    frame_len = int(jnp.round(frame_ms * fs_hz / 1000.0))
    hop = int(jnp.round(hop_ms * fs_hz / 1000.0))

    frames = frame_signal_1d(signal, frame_len, hop)
    window = jnp.hanning(frame_len)

    # Apply window and compute FFT
    frames = frames * window[None, :]
    fft = jnp.fft.rfft(frames, axis=-1)
    freqs = jnp.fft.rfftfreq(frame_len, d=1.0 / fs_hz)

    # Window energy correction
    scale = fs_hz * jnp.sum(window**2) + eps
    psd = jnp.mean((jnp.abs(fft) ** 2) / scale, axis=0)

    # Interpolate to target frequencies
    if target_freqs is not None:
        psd = jnp.interp(target_freqs, freqs, psd)

    return normalize_spectrum(psd, eps=eps)


# ==============================================================================
# Spectral Loss Functions
# ==============================================================================

def spectral_shape_loss(
    pred_psd: jnp.ndarray,
    target_psd: jnp.ndarray,
    eps: float = 1e-6,
) -> jnp.ndarray:
    """
    Log-spectral shape loss (normalized).

    Compares log-normalized spectrum, robust to amplitude differences.
    Good for matching spectral *shape* regardless of overall power.
    """
    pred = jnp.log(pred_psd + eps)
    targ = jnp.log(target_psd + eps)

    # Normalize to zero mean, unit variance
    pred = (pred - jnp.mean(pred)) / (jnp.std(pred) + eps)
    targ = (targ - jnp.mean(targ)) / (jnp.std(targ) + eps)

    return jnp.mean((pred - targ) ** 2)


def spectral_cosine_loss(
    pred_psd: jnp.ndarray,
    target_psd: jnp.ndarray,
    eps: float = 1e-8,
) -> jnp.ndarray:
    """
    Cosine distance in spectrum space.

    1.0 - cosine_sim, so 0 = perfect match, 1 = orthogonal.
    Less sensitive to outliers than MSE.
    """
    pred = pred_psd / (jnp.linalg.norm(pred_psd) + eps)
    targ = target_psd / (jnp.linalg.norm(target_psd) + eps)
    return 1.0 - jnp.sum(pred * targ)


def bandpower_ratio_loss(
    pred_psd: jnp.ndarray,
    target_psd: jnp.ndarray,
    freqs: jnp.ndarray,
    bands: dict = None,
    eps: float = 1e-8,
) -> jnp.ndarray:
    """
    Compare bandpower ratios (gamma/theta, beta/alpha, etc).

    Args:
      pred_psd, target_psd: shape (F,)
      freqs: shape (F,) — frequency bins
      bands: dict of {name: (lo_hz, hi_hz), ...}
      eps: numerical stability

    Returns:
      loss: scalar, mean squared error of log bandpower ratios
    """
    if bands is None:
        bands = {
            "theta": (4.0, 8.0),
            "alpha": (8.0, 12.0),
            "beta": (13.0, 30.0),
            "gamma": (35.0, 60.0),
        }

    loss = 0.0
    for name, (lo, hi) in bands.items():
        mask = (freqs >= lo) & (freqs < hi)
        count = jnp.sum(mask) + eps

        pred_bp = jnp.sum(jnp.where(mask, pred_psd, 0.0)) / count
        targ_bp = jnp.sum(jnp.where(mask, target_psd, 0.0)) / count

        loss = loss + (jnp.log(pred_bp + eps) - jnp.log(targ_bp + eps)) ** 2

    return loss / len(bands)


def firing_rate_penalty(
    rate_hz: jnp.ndarray,
    lower_bound: float = 1.0,
    upper_bound: float = 100.0,
) -> jnp.ndarray:
    """
    Soft penalty for firing rates outside bounds.

    Lower penalty: exp(lower - rate) when rate < lower
    Upper penalty: exp(rate - upper) when rate > upper
    """
    lower_pen = jnp.sum(jnp.exp(lower_bound - rate_hz))
    upper_pen = jnp.sum(jnp.exp(rate_hz - upper_bound))
    return (lower_pen + upper_pen) / len(rate_hz)


# ==============================================================================
# Combined Spectral Loss
# ==============================================================================

def combined_spectral_loss(
    pred_signal: jnp.ndarray,
    target_psd: jnp.ndarray,
    fs_hz: float = 1000.0,
    target_freqs: jnp.ndarray = None,
    frame_ms: float = 500.0,
    hop_ms: float = 50.0,
    shape_weight: float = 1.0,
    cosine_weight: float = 0.2,
    bandpower_weight: float = 0.1,
    eps: float = 1e-8,
) -> jnp.ndarray:
    """
    Multi-term spectral loss: shape + cosine + bandpower.

    Good default for broadband spectral matching.
    """
    if target_freqs is None:
        target_freqs = jnp.linspace(5.0, 100.0, 100)

    # Compute predicted PSD
    pred_psd = jax_welch_psd(
        pred_signal,
        fs_hz=fs_hz,
        target_freqs=target_freqs,
        frame_ms=frame_ms,
        hop_ms=hop_ms,
        eps=eps,
    )

    # Multi-term loss
    shape_loss = spectral_shape_loss(pred_psd, target_psd, eps=eps)
    cosine_loss = spectral_cosine_loss(pred_psd, target_psd, eps=eps)
    bandpower_loss = bandpower_ratio_loss(
        pred_psd, target_psd, target_freqs, eps=eps
    )

    total = (
        shape_weight * shape_loss
        + cosine_weight * cosine_loss
        + bandpower_weight * bandpower_loss
    )

    return total


# ==============================================================================
# Target Generation from Empirical Data
# ==============================================================================

def empirical_population_signal(
    spike_data: jnp.ndarray,
    time_slice: slice,
    trial_slice: slice = slice(None),
    neuron_slice: slice = slice(None),
    binary: bool = True,
) -> jnp.ndarray:
    """
    Extract population-averaged signal from spike data.

    Args:
      spike_data: shape (trials, time, neurons)
      time_slice, trial_slice, neuron_slice: numpy slices
      binary: if True, threshold to 0/1; else use raw values

    Returns:
      signal: shape (T,) — population mean over trials and neurons
    """
    x = spike_data[trial_slice, time_slice, neuron_slice]
    x = jnp.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    if binary:
        x = (x > 0.0).astype(jnp.float32)

    # Average across trials and neurons
    return jnp.mean(x, axis=tuple(range(x.ndim - 1)))


def make_target_psds(
    spike_data: jnp.ndarray,
    fs_hz: float = 1000.0,
    target_freqs: jnp.ndarray = None,
    baseline_time_slice: slice = slice(0, 500),
    stim_time_slice: slice = slice(600, 1200),
    frame_ms: float = 500.0,
    hop_ms: float = 50.0,
) -> tuple:
    """
    Extract target baseline and stimulus PSDs from empirical data.

    Args:
      spike_data: shape (trials, time, neurons)
      fs_hz: sampling rate (typically 1000 Hz)
      target_freqs: frequency bins for interpolation
      baseline_time_slice, stim_time_slice: time windows (in samples)
      frame_ms, hop_ms: Welch parameters

    Returns:
      (baseline_psd, stim_psd): each shape (F,)
    """
    if target_freqs is None:
        target_freqs = jnp.linspace(5.0, 100.0, 100)

    # Extract population signals
    baseline_signal = empirical_population_signal(
        spike_data,
        time_slice=baseline_time_slice,
        trial_slice=slice(None),
        neuron_slice=slice(None),
        binary=True,
    )

    stim_signal = empirical_population_signal(
        spike_data,
        time_slice=stim_time_slice,
        trial_slice=slice(None),
        neuron_slice=slice(None),
        binary=True,
    )

    # Compute Welch PSDs
    baseline_psd = jax_welch_psd(
        baseline_signal,
        fs_hz=fs_hz,
        target_freqs=target_freqs,
        frame_ms=frame_ms,
        hop_ms=hop_ms,
    )

    stim_psd = jax_welch_psd(
        stim_signal,
        fs_hz=fs_hz,
        target_freqs=target_freqs,
        frame_ms=frame_ms,
        hop_ms=hop_ms,
    )

    return baseline_psd, stim_psd, target_freqs
