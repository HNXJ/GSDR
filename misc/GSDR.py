"""
GSDR: Genetic Stochastic Delta Rule
Root-level bridge to the modular gsdr package.
"""

from .gsdr import build_net_eig, Inoise, GradedAMPA, GradedGABAa, GradedGABAb
from .gsdr import SDR, GSDR, ClampTransform
from .gsdr import Dataset
from .gsdr import (
    calculate_firing_rates, compute_psd, plot_full_simulation_summary, 
    calculate_mcdp, compute_kappa, compute_unscaled_psd_from_trace,
    traces_to_spike_matrix
)
from .gsdr import noise_current, ramp_current, step_current, noise_current_ac
from .gsdr import get_loss_fn, train_net
