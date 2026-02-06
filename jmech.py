import jax.numpy as jnp
import jaxley as jx
import numpy as np
import jax

from jax import vmap
from jaxley.channels import Channel
from jaxley.synapses import Synapse

from typing import Optional, Tuple


class GradedAMPA(Synapse):
    """
    Graded Excitatory Synapse (AMPA).
    Ref: Traub et al. (1991), Borg-Graham (1998).
    """
    def __init__(self, tauD_AMPA: Optional[float] = None):
        super().__init__()
        r_d_gSyn = np.random.uniform(2.0, 3.0) # Conductance range
        self.synapse_params = {
            "gAMPA": r_d_gSyn,       # Reduced initial conductance
            "EAMPA": 0.0,        # Reversal Potential (mV) - Excitatory
            "tauDAMPA": 5.0,         # Faster decay for AMPA (ms)
            "tauRAMPA": 0.2,         # Fast rise (ms)
            "slopeAMPA": 5.0,
            "V_thAMPA": -20.0        # Threshold (mV)
        }

        if tauD_AMPA is not None:
            self.synapse_params["tauDAMPA"] = tauD_AMPA

        self.synapse_states = {"sAMPA": 0.1}

    def update_states(self, states, dt, pre_v, post_v, params):
        s = states["sAMPA"]
        activation = 0.5 * (1 + jnp.tanh((pre_v - params["V_thAMPA"]) / params["slopeAMPA"])) # Sigmoidal activation
        d_s = (-s / params["tauDAMPA"]) + activation * ((1 - s) / params["tauRAMPA"])
        return {"sAMPA": s + d_s * dt}

    def compute_current(self, states, pre_v, post_v, params):
        return params["gAMPA"] * states["sAMPA"] * (post_v - params["EAMPA"])


class GradedGABAa(Synapse):
    """
    Graded fast Inhibitory Synapse. (GABAa)
    Ref: Golowasch et al. (1999) ; Prinz, A. A., et al. (2004).
    """
    def __init__(self, tauD_GABAa: Optional[float] = None):
        super().__init__()

        r_d_gSyn = np.random.uniform(4.0, 6.0) # Conductance range

        self.synapse_params = {
            "gGABAa": r_d_gSyn,       # Reduced initial conductance
            "EGABAa": -80.0,     # Reversal Potential (mV)
            "tauDGABAa": 5.0,        # Decay (ms)
            "tauRGABAa": 0.5,         # Rise (ms)
            "slopeGABAa": 5.0,        # Steepness of activation
            "V_thGABAa": -20.0        # Threshold (mV) - Tuned to activate during spikes
        }

        if tauD_GABAa is not None:
            self.synapse_params["tauDGABAa"] = tauD_GABAa

        self.synapse_states = {"sGABAa": 0.1}

    def update_states(self, states, dt, pre_v, post_v, params):
        # s' = -s/tauD + 0.5*(1+tanh((V-Vth)/slope)) * ((1-s)/tauR)
        s = states["sGABAa"]
        activation = 0.5 * (1 + jnp.tanh((pre_v - params["V_thGABAa"]) / params["slopeGABAa"]))
        d_s = (-s / params["tauDGABAa"]) + activation * ((1 - s) / params["tauRGABAa"])
        return {"sGABAa": s + d_s * dt}

    def compute_current(self, states, pre_v, post_v, params):
        return params["gGABAa"] * states["sGABAa"] * (post_v - params["EGABAa"])


class GradedGABAb(Synapse):
    """
    Graded slow Inhibitory Synapse. (GABAb)
    Ref: Golowasch et al. (1999) ; Prinz, A. A., et al. (2004).
    """
    def __init__(self, tauD_GABAb: Optional[float] = None):
        super().__init__()
        r_d_gSyn = np.random.uniform(0.5, 9.5) # Conductance range
        self.synapse_params = {
            "gGABAb": r_d_gSyn,       # Conductance (uS)
            "EGABAb": -95.0,     # Reversal Potential (mV)
            "tauDGABAb": 200.0,        # Decay (ms)
            "tauRGABAb": 10.0,         # Rise (ms)
            "slopeGABAb": 5.0,        # Steepness of activation
            "V_thGABAb": -20.0        # Threshold (mV) - Tuned to activate during spikes
        }

        if tauD_GABAb is not None:
            self.synapse_params["tauDGABAb"] = tauD_GABAb

        self.synapse_states = {"sGABAb": 0.01}

    def update_states(self, states, dt, pre_v, post_v, params):
        # s' = -s/tauD + 0.5*(1+tanh((V-Vth)/slope)) * ((1-s)/tauR)
        s = states["sGABAb"]
        activation = 0.5 * (1 + jnp.tanh((pre_v - params["V_thGABAb"]) / params["slopeGABAb"]))
        d_s = (-s / params["tauDGABAb"]) + activation * ((1 - s) / params["tauRGABAb"])
        return {"sGABAb": s + d_s * dt}

    def compute_current(self, states, pre_v, post_v, params):
        return params["gGABAb"] * states["sGABAb"] * (post_v - params["EGABAb"])


class GradedNMDA(Synapse):
    """
    Graded slow Excitatory Synapse. (NMDA)
    Ref: Traub et al. (1991), Borg-Graham (1998).
    """
    def __init__(self, tauD_NMDA: Optional[float] = None):
        super().__init__()
        r_d_gSyn = np.random.uniform(0.1, 1.0) # Often smaller conductance but longer lasting
        self.synapse_params = {
            "gNMDA": r_d_gSyn,
            "ENMDA": 0.0,         # Excitatory reversal potential
            "tauDNMDA": 75.0,     # Slower decay (ms)
            "tauRNMDA": 7.0,      # Slower rise (ms)
            "slopeNMDA": 5.0,
            "V_thNMDA": -20.0
        }
        if tauD_NMDA is not None:
            self.synapse_params["tauDNMDA"] = tauD_NMDA
        self.synapse_states = {"sNMDA": 0.1}

    def update_states(self, states, dt, pre_v, post_v, params):
        s = states["sNMDA"]
        activation = 0.5 * (1 + jnp.tanh((pre_v - params["V_thNMDA"]) / params["slopeNMDA"])) # Simplified activation for graded
        d_s = (-s / params["tauDNMDA"]) + activation * ((1 - s) / params["tauRNMDA"])
        return {"sNMDA": s + d_s * dt}

    def compute_current(self, states, pre_v, post_v, params):
        # NMDA often has a voltage-dependent magnesium block. For a simple graded model,
        # we might omit explicit voltage dependence or add it as a separate factor.
        # For now, keeping it similar to other graded synapses.
        return params["gNMDA"] * states["sNMDA"] * (post_v - params["ENMDA"])


class Inoise(Channel):
    """
    Stochastic Ornstein-Uhlenbeck noise channel.
    """
    def __init__(self, name: str = None, initial_seed: Optional[int] = None, initial_amp_noise: Optional[float] = None, initial_tau: Optional[float] = None, initial_mean: Optional[float] = None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)

        self.channel_params = {
            "amp_noise": 0.01,  # Reduced initial noise amplitude
            "mean": 0.00,        # The baseline current drive [mA/cm^2]
            "tau": 20.0         # Correlation time constant [ms]
        }
        # If no initial_seed is provided, generate a random one using numpy.
        # This ensures each Inoise instance gets a unique starting seed.
        if initial_seed is None:
            self.channel_params["seed"] = float(np.random.randint(0, 2**16 - 1))
        else:
            self.channel_params["seed"] = float(initial_seed)

        if initial_amp_noise is None:
            self.channel_params["amp_noise"] = float(0.01) # Ensure this default is also reduced
        else:
            self.channel_params["amp_noise"] = float(initial_amp_noise)

        if initial_tau is None:
            self.channel_params["tau"] = float(20.0)
        else:
            self.channel_params["tau"] = float(initial_tau)

        if initial_mean is None:
            self.channel_params["mean"] = float(0.00)
        else:
            self.channel_params["mean"] = float(initial_mean)

        self.channel_states = {"n": 0.00, "step": 0.0}
        self.current_name = "i_noise"

    def update_states(self, states, dt, v, params):
        """
        Updates the noise state 'n' using an Ornstein-Uhlenbeck process.
        """
        n = states["n"]
        step = states["step"]

        # 1. RNG Handling
        # When JAXley vmaps the update_states function, params["seed"] will be an array.
        # All other inputs (n, step, v, dt) will also be batched (arrays).
        # We need to vmap the PRNGKey and fold_in calls.

        # Ensure seed is an integer type for PRNGKey
        seeds_int = params["seed"].astype(int)

        # Create base keys (potentially batched)
        if seeds_int.ndim == 0:
            base_key = jax.random.PRNGKey(seeds_int)
        else:
            base_key = jax.vmap(jax.random.PRNGKey)(seeds_int)

        # Fold in step (potentially batched)
        if step.ndim == 0:
            step_key = jax.random.fold_in(base_key, step.astype(int))
        else: # if step is an array
            step_key = jax.vmap(jax.random.fold_in)(base_key, step.astype(int))

        # Generate normal random numbers (potentially batched)
        # A single JAX PRNGKey has shape (2,) and ndim=1.
        # An array of N JAX PRNGKeys has shape (N, 2) and ndim=2.
        if step_key.ndim == 1: # if step_key is a single key
            xi = jax.random.normal(step_key)
        else: # if step_key is an array of keys (ndim 2 for key array)
            xi = jax.vmap(jax.random.normal)(step_key)

        # 2. Physics (Ornstein-Uhlenbeck)
        # dn = -(n - mean)/tau * dt + sigma*sqrt(2/tau)*dW
        mu = params["mean"]
        sigma = params["amp_noise"]
        tau = params["tau"]

        drift = (mu - n) / tau * dt
        diffusion = sigma * jnp.sqrt(2.0 / tau) * xi * jnp.sqrt(dt) # FIX: Changed dt_global to dt

        new_n = n + drift + diffusion # FIX: Changed 'n' to 'n_prev'

        # Return new state AND increment step
        return {"n": new_n, "step": step + 1.0}

    def compute_current(self, states, v, params):
        """
        Returns the current.
        Note: We return negative 'n' so that a positive mean acts
        as an excitatory (depolarizing) injection in the cable equation.
        """
        return -states["n"]

    def init_state(self, states, v, params, delta_t):
        """
        Initialize to the mean value.
        This needs to handle batched parameters if JAXley vmaps init_state.
        """
        # If params["mean"] is a scalar, jnp.zeros_like(params["mean"]) is a scalar 0.0.
        # If params["mean"] is an array, jnp.zeros_like(params["mean"]) is an array of 0.0s.
        # This handles both cases correctly.
        return {"n": jnp.zeros_like(params["mean"]) + params["mean"], "step": jnp.zeros_like(params["mean"]) + 0.0}


class GradedDRD1(Synapse):
    """
    Graded Dopamine D1 Receptor Synapse.
    Ref: K.J. Wager, J.E. Raymond, M.J. Frank (2018) - often modeled as increasing excitability (excitatory reversal)
    """
    def __init__(self, tauD_DRD1: Optional[float] = None):
        super().__init__()
        r_d_gSyn = np.random.uniform(0.01, 0.5) # Modulatory conductance
        self.synapse_params = {
            "gDRD1": r_d_gSyn,
            "EDRD1": 0.0,         # Excitatory reversal potential
            "tauDDRD1": 200.0,    # Slower decay (ms)
            "tauRDRD1": 50.0,     # Slower rise (ms)
            "slopeDRD1": 5.0,
            "V_thDRD1": -20.0
        }
        if tauD_DRD1 is not None:
            self.synapse_params["tauDDRD1"] = tauD_DRD1
        self.synapse_states = {"sDRD1": 0.01}

    def update_states(self, states, dt, pre_v, post_v, params):
        s = states["sDRD1"]
        activation = 0.5 * (1 + jnp.tanh((pre_v - params["V_thDRD1"]) / params["slopeDRD1"])) # Sigmoidal activation
        d_s = (-s / params["tauDDRD1"]) + activation * ((1 - s) / params["tauRDRD1"])
        return {"sDRD1": s + d_s * dt}

    def compute_current(self, states, pre_v, post_v, params):
        return params["gDRD1"] * states["sDRD1"] * (post_v - params["EDRD1"])


class GradedDRD2(Synapse):
    """
    Graded Dopamine D2 Receptor Synapse.
    Ref: K.J. Wager, J.E. Raymond, M.J. Frank (2018) - often modeled as decreasing excitability (inhibitory reversal)
    """
    def __init__(self, tauD_DRD2: Optional[float] = None):
        super().__init__()
        r_d_gSyn = np.random.uniform(0.01, 0.5) # Modulatory conductance
        self.synapse_params = {
            "gDRD2": r_d_gSyn,
            "EDRD2": -90.0,       # Inhibitory reversal potential
            "tauDDRD2": 250.0,    # Slower decay (ms)
            "tauRDRD2": 60.0,     # Slower rise (ms)
            "slopeDRD2": 5.0,
            "V_thDRD2": -20.0
        }
        if tauD_DRD2 is not None:
            self.synapse_params["tauDDRD2"] = tauD_DRD2
        self.synapse_states = {"sDRD2": 0.01}

    def update_states(self, states, dt, pre_v, post_v, params):
        s = states["sDRD2"]
        activation = 0.5 * (1 + jnp.tanh((pre_v - params["V_thDRD2"]) / params["slopeDRD2"])) # Sigmoidal activation
        d_s = (-s / params["tauDDRD2"]) + activation * ((1 - s) / params["tauRDRD2"])
        return {"sDRD2": s + d_s * dt}

    def compute_current(self, states, pre_v, post_v, params):
        return params["gDRD2"] * states["sDRD2"] * (post_v - params["EDRD2"])


class Graded5HT1(Synapse):
    """
    Graded Serotonin 5-HT1 Receptor Synapse (e.g., 5-HT1A).
    Ref: Azhari et al. (2006) - often inhibitory via Gi coupling.
    """
    def __init__(self, tauD_5HT1: Optional[float] = None):
        super().__init__()
        r_d_gSyn = np.random.uniform(0.01, 0.5) # Modulatory conductance
        self.synapse_params = {
            "g5HT1": r_d_gSyn,
            "E5HT1": -90.0,       # Inhibitory reversal potential
            "tauD5HT1": 300.0,    # Slower decay (ms)
            "tauR5HT1": 70.0,     # Slower rise (ms)
            "slope5HT1": 5.0,
            "V_th5HT1": -20.0
        }
        if tauD_5HT1 is not None:
            self.synapse_params["tauD5HT1"] = tauD_5HT1
        self.synapse_states = {"s5HT1": 0.01}

    def update_states(self, states, dt, pre_v, post_v, params):
        s = states["s5HT1"]
        activation = 0.5 * (1 + jnp.tanh((pre_v - params["V_th5HT1"]) / params["slope5HT1"])) # Sigmoidal activation
        d_s = (-s / params["tauD5HT1"]) + activation * ((1 - s) / params["tauR5HT1"])
        return {"s5HT1": s + d_s * dt}

    def compute_current(self, states, pre_v, post_v, params):
        return params["g5HT1"] * states["s5HT1"] * (post_v - params["E5HT1"])


class Graded5HT2a(Synapse):
    """
    Graded Serotonin 5-HT2A Receptor Synapse.
    Ref: K.J. Wager, J.E. Raymond, M.J. Frank (2018) - often excitatory via Gq coupling.
    """
    def __init__(self, tauD_5HT2a: Optional[float] = None):
        super().__init__()
        r_d_gSyn = np.random.uniform(0.01, 0.5) # Modulatory conductance
        self.synapse_params = {
            "g5HT2a": r_d_gSyn,
            "E5HT2a": 0.0,        # Excitatory reversal potential
            "tauD5HT2a": 200.0,   # Slower decay (ms)
            "tauR5HT2a": 50.0,    # Slower rise (ms)
            "slope5HT2a": 5.0,
            "V_th5HT2a": -20.0
        }
        if tauD_5HT2a is not None:
            self.synapse_params["tauD5HT2a"] = tauD_5HT2a
        self.synapse_states = {"s5HT2a": 0.01}

    def update_states(self, states, dt, pre_v, post_v, params):
        s = states["s5HT2a"]
        activation = 0.5 * (1 + jnp.tanh((pre_v - params["V_th5HT2a"]) / params["slope5HT2a"])) # Sigmoidal activation
        d_s = (-s / params["tauD5HT2a"]) + activation * ((1 - s) / params["tauR5HT2a"])
        return {"s5HT2a": s + d_s * dt}

    def compute_current(self, states, pre_v, post_v, params):
        return params["g5HT2a"] * states["s5HT2a"] * (post_v - params["E5HT2a"])


class Graded5HT3(Synapse):
    """
    Graded Serotonin 5-HT3 Receptor Synapse.
    Ref: Siegelbaum et al. (2014) - ionotropic, excitatory.
    """
    def __init__(self, tauD_5HT3: Optional[float] = None):
        super().__init__()
        r_d_gSyn = np.random.uniform(0.5, 2.0) # Conductance range
        self.synapse_params = {
            "g5HT3": r_d_gSyn,
            "E5HT3": 0.0,         # Excitatory reversal potential
            "tauD5HT3": 10.0,     # Faster decay (ms) than metabotropic
            "tauR5HT3": 0.5,      # Fast rise (ms)
            "slope5HT3": 5.0,
            "V_th5HT3": -20.0
        }
        if tauD_5HT3 is not None:
            self.synapse_params["tauD5HT3"] = tauD_5HT3
        self.synapse_states = {"s5HT3": 0.1}

    def update_states(self, states, dt, pre_v, post_v, params):
        s = states["s5HT3"]
        activation = 0.5 * (1 + jnp.tanh((pre_v - params["V_th5HT3"]) / params["slope5HT3"])) # Sigmoidal activation
        d_s = (-s / params["tauD5HT3"]) + activation * ((1 - s) / params["tauR5HT3"])
        return {"s5HT3": s + d_s * dt}

    def compute_current(self, states, pre_v, post_v, params):
        return params["g5HT3"] * states["s5HT3"] * (post_v - params["E5HT3"])


class GradedM1(Synapse):
    """
    Graded Muscarinic Acetylcholine M1 Receptor Synapse.
    Ref: M. Destexhe, J.M. Fellous, T.J. Sejnowski (2001) - often excitatory via Gq coupling, reducing K+ currents.
    """
    def __init__(self, tauD_M1: Optional[float] = None):
        super().__init__()
        r_d_gSyn = np.random.uniform(0.01, 0.5) # Modulatory conductance
        self.synapse_params = {
            "gM1": r_d_gSyn,
            "EM1": 0.0,           # Excitatory reversal potential (indirectly via K+ channel block)
            "tauDM1": 400.0,      # Very slow decay (ms)
            "tauRM1": 100.0,      # Slow rise (ms)
            "slopeM1": 5.0,
            "V_thM1": -20.0
        }
        if tauD_M1 is not None:
            self.synapse_params["tauDM1"] = tauD_M1
        self.synapse_states = {"sM1": 0.01}

    def update_states(self, states, dt, pre_v, post_v, params):
        s = states["sM1"]
        activation = 0.5 * (1 + jnp.tanh((pre_v - params["V_thM1"]) / params["slopeM1"])) # Sigmoidal activation
        d_s = (-s / params["tauDM1"]) + activation * ((1 - s) / params["tauRM1"])
        return {"sM1": s + d_s * dt}

    def compute_current(self, states, pre_v, post_v, params):
        return params["gM1"] * states["sM1"] * (post_v - params["EM1"])


class GradednACH(Synapse):
    """
    Graded Nicotinic Acetylcholine Receptor Synapse.
    Ref: Siegelbaum et al. (2014) - ionotropic, excitatory.
    """
    def __init__(self, tauD_nACH: Optional[float] = None):
        super().__init__()
        r_d_gSyn = np.random.uniform(0.5, 2.0) # Conductance range
        self.synapse_params = {
            "gnACH": r_d_gSyn,
            "EnACH": 0.0,         # Excitatory reversal potential
            "tauDnACH": 5.0,      # Fast decay (ms)
            "tauRnACH": 0.2,      # Fast rise (ms)
            "slopenACH": 5.0,
            "V_thnACH": -20.0
        }
        if tauD_nACH is not None:
            self.synapse_params["tauDnACH"] = tauD_nACH
        self.synapse_states = {"snACH": 0.1}

    def update_states(self, states, dt, pre_v, post_v, params):
        s = states["snACH"]
        activation = 0.5 * (1 + jnp.tanh((pre_v - params["slopenACH"]) / params["slopenACH"])) # Sigmoidal activation
        d_s = (-s / params["tauDnACH"]) + activation * ((1 - s) / params["tauRnACH"])
        return {"snACH": s + d_s * dt}

    def compute_current(self, states, pre_v, post_v, params):
        return params["gnACH"] * states["snACH"] * (post_v - params["EnACH"])


class GradedCustomMechanism(Synapse):
    """
    Template for a custom graded synaptic mechanism.
    """
    def __init__(self, g_custom: float, E_custom: float, tauD_custom: float, tauR_custom: float, slope_custom: float, V_th_custom: float, s_init: float = 0.1):
        super().__init__()
        self.synapse_params = {
            "gCustom": g_custom,
            "ECustom": E_custom,
            "tauDCustom": tauD_custom,
            "tauRCustom": tauR_custom,
            "slopeCustom": slope_custom,
            "V_thCustom": V_th_custom
        }
        self.synapse_states = {"sCustom": s_init}

    def update_states(self, states, dt, pre_v, post_v, params):
        s = states["sCustom"]
        activation = 0.5 * (1 + jnp.tanh((pre_v - params["V_thCustom"]) / params["slopeCustom"]))
        d_s = (-s / params["tauDCustom"]) + activation * ((1 - s) / params["tauRCustom"])
        return {"sCustom": s + d_s * dt}

    def compute_current(self, states, pre_v, post_v, params):
        return params["gCustom"] * states["sCustom"] * (post_v - params["ECustom"])
        

class Diffusion(Channel):
    def __init__(self, g_max: float = 100.0, v_th: float = 50.0, slope: float = 1.0, e_rev: float = -70.0):
        self.current_is_in_mA_per_cm2 = True # Required for Jaxley 0.5.0+
        super().__init__()
        self.channel_params = {
            "Diffusion_g_max": g_max,
            "Diffusion_v_th": v_th,
            "Diffusion_slope": slope,
            "Diffusion_e_rev": e_rev
        }
        self.channel_states = {}

    def compute_current(self, states, voltage, params):
        # Exponentially increasing conductance when V > V_th
        # acts as a "soft" voltage clamp or diode
        v_diff = (voltage - params["Diffusion_v_th"]) / params["Diffusion_slope"]
        g = params["Diffusion_g_max"] * jnp.exp(v_diff)
        # Conductance (mS/cm^2) * Voltage (mV) = Current (uA/cm^2)
        # Divide by 1000 to get mA/cm^2
        return (g * (voltage - params["Diffusion_e_rev"])) / 1000.0
    
    def init_state(self, states, voltage, params):
        return {}


class STDPGradedGABAa(Synapse):
    def __init__(self, tauD_GABAa: Optional[float] = None, stdp_rate: float = 0.1):
        super().__init__()
        r_d_gSyn = np.random.uniform(4.0, 6.0)

        self.synapse_params = {
            "STDP_EGABAa": -80.0,
            "STDP_tauDGABAa": 5.0,
            "STDP_tauRGABAa": 0.5,
            "STDP_slopeGABAa": 5.0,
            "STDP_V_thGABAa": -20.0,
            # STDP Parameters - Prefixed
            "GABAa_stdp_rate": stdp_rate,
            "GABAa_tau_p": 30.0,
            "GABAa_c_p": 0.06,   # Scaled down from 60.0
            "GABAa_c_d": 0.1,    # Scaled down from 100.0
            "GABAa_V_spike": 0.0,
        }

        if tauD_GABAa is not None:
            self.synapse_params["STDP_tauDGABAa"] = tauD_GABAa

        self.synapse_states = {
            "STDP_sGABAa": 0.1,
            "STDP_gGABAa": r_d_gSyn,
            # States - Prefixed
            "GABAa_pre_trace": 0.0,
            "GABAa_post_trace": 0.0
        }

    def update_states(self, states, dt, pre_v, post_v, params):
        s = states["STDP_sGABAa"]
        g = states["STDP_gGABAa"]
        tr_pre = states["GABAa_pre_trace"]
        tr_post = states["GABAa_post_trace"]

        activation = 0.5 * (1 + jnp.tanh((pre_v - params["STDP_V_thGABAa"]) / params["STDP_slopeGABAa"]))
        d_s = (-s / params["STDP_tauDGABAa"]) + activation * ((1 - s) / params["STDP_tauRGABAa"])

        pre_spike = jnp.where(pre_v > params["GABAa_V_spike"], 1.0, 0.0)
        post_spike = jnp.where(post_v > params["GABAa_V_spike"], 1.0, 0.0)

        d_tr_pre = -tr_pre / params["GABAa_tau_p"] + pre_spike
        d_tr_post = -tr_post / params["GABAa_tau_p"] + post_spike

        dw_plus = params["GABAa_c_p"] * tr_pre * post_spike
        dw_minus = params["GABAa_c_p"] * tr_post * pre_spike
        ltd = params["GABAa_c_d"] * pre_spike

        total_dw = (dw_plus - dw_minus - ltd) * params["GABAa_stdp_rate"] * dt

        return {
            "STDP_sGABAa": s + d_s * dt,
            "STDP_gGABAa": jnp.clip(g + total_dw, a_min=0.0, a_max=2.0e-4), # Clip max conductance
            "GABAa_pre_trace": tr_pre + d_tr_pre * dt,
            "GABAa_post_trace": tr_post + d_tr_post * dt
        }

    def compute_current(self, states, pre_v, post_v, params):
        return states["STDP_gGABAa"] * states["STDP_sGABAa"] * (post_v - params["STDP_EGABAa"])


class STDPGradedAMPA(Synapse):
    def __init__(self, tauD_AMPA: Optional[float] = None, stdp_rate: float = 0.1):
        super().__init__()
        r_d_gSyn = np.random.uniform(2.0, 3.0)

        self.synapse_params = {
            "STDP_EAMPA": 0.0,
            "STDP_tauDAMPA": 5.0,
            "STDP_tauRAMPA": 0.2,
            "STDP_slopeAMPA": 5.0,
            "STDP_V_thAMPA": -20.0,
            # Scaled down STDP params
            "AMPA_stdp_rate": stdp_rate,
            "AMPA_tau_p": 30.0,
            "AMPA_c_p": 0.06,
            "AMPA_c_d": 0.1,
            "AMPA_V_spike": 0.0
        }

        if tauD_AMPA is not None:
            self.synapse_params["STDP_tauDAMPA"] = tauD_AMPA

        self.synapse_states = {
            "STDP_sAMPA": 0.1,
            "STDP_gAMPA": r_d_gSyn,
            "AMPA_pre_trace": 0.0,
            "AMPA_post_trace": 0.0
        }

    def update_states(self, states, dt, pre_v, post_v, params):
        s = states["STDP_sAMPA"]
        g = states["STDP_gAMPA"]
        tr_pre = states["AMPA_pre_trace"]
        tr_post = states["AMPA_post_trace"]

        activation = 0.5 * (1 + jnp.tanh((pre_v - params["STDP_V_thAMPA"]) / params["STDP_slopeAMPA"]))
        d_s = (-s / params["STDP_tauDAMPA"]) + activation * ((1 - s) / params["STDP_tauRAMPA"])

        pre_spike = jnp.where(pre_v > params["AMPA_V_spike"], 1.0, 0.0)
        post_spike = jnp.where(post_v > params["AMPA_V_spike"], 1.0, 0.0)

        d_tr_pre = -tr_pre / params["AMPA_tau_p"] + pre_spike
        d_tr_post = -tr_post / params["AMPA_tau_p"] + post_spike

        dw_plus = params["AMPA_c_p"] * tr_pre * post_spike
        dw_minus = params["AMPA_c_p"] * tr_post * pre_spike
        ltd_baseline = params["AMPA_c_d"] * pre_spike

        total_dw = (dw_plus - dw_minus - ltd_baseline) * params["AMPA_stdp_rate"] * dt

        return {
            "STDP_sAMPA": s + d_s * dt,
            "STDP_gAMPA": jnp.clip(g + total_dw, a_min=0.0, a_max=1.0e-4),
            "AMPA_pre_trace": tr_pre + d_tr_pre * dt,
            "AMPA_post_trace": tr_post + d_tr_post * dt
        }

    def compute_current(self, states, pre_v, post_v, params):
        return states["STDP_gAMPA"] * states["STDP_sAMPA"] * (post_v - params["STDP_EAMPA"])


class STDPGradedNMDA(Synapse):
    def __init__(self, tauD_NMDA: Optional[float] = None, stdp_rate: float = 1.0):
        super().__init__()
        # Reduced g drastically to 1e-6 range
        r_d_gSyn = np.random.uniform(0.1, 1.0)

        self.synapse_params = {
            "STDP_ENMDA": 0.0,
            "STDP_tauDNMDA": 75.0,
            "STDP_tauRNMDA": 7.0,
            "STDP_slopeNMDA": 5.0,
            "STDP_V_thNMDA": -20.0,
            "NMDA_Mg2": 1.0,
            "NMDA_stdp_rate": stdp_rate,
            "NMDA_tau_p": 30.0,
            "NMDA_c_p": 0.06,
            "NMDA_c_d": 0.1,
            "NMDA_Th": 2.5,
            "NMDA_V_spike": 0.0
        }

        if tauD_NMDA is not None:
            self.synapse_params["STDP_tauDNMDA"] = tauD_NMDA

        self.synapse_states = {
            "STDP_sNMDA": 0.1,
            "STDP_gNMDA": r_d_gSyn,
            "NMDA_pre_trace": 0.0,
            "NMDA_post_trace": 0.0
        }

    def update_states(self, states, dt, pre_v, post_v, params):
        s = states["STDP_sNMDA"]
        g = states["STDP_gNMDA"]
        tr_pre = states["NMDA_pre_trace"]
        tr_post = states["NMDA_post_trace"]

        activation = 0.5 * (1 + jnp.tanh((pre_v - params["STDP_V_thNMDA"]) / params["STDP_slopeNMDA"]))
        d_s = (-s / params["STDP_tauDNMDA"]) + activation * ((1 - s) / params["STDP_tauRNMDA"])

        pre_event = jnp.where(pre_v > params["NMDA_V_spike"], 1.0, 0.0)
        post_event = jnp.where(post_v > params["NMDA_V_spike"], 1.0, 0.0)

        d_tr_pre = -tr_pre / params["NMDA_tau_p"] + pre_event
        d_tr_post = -tr_post / params["NMDA_tau_p"] + post_event

        i_nmda = jnp.abs(self.compute_current(states, pre_v, post_v, params))
        ca_gate = jnp.maximum(i_nmda - params["NMDA_Th"], 0.0)

        dw_plus = params["NMDA_c_p"] * ca_gate * tr_pre * post_event
        dw_minus = params["NMDA_c_p"] * ca_gate * tr_post * pre_event
        ltd = params["NMDA_c_d"] * pre_event

        total_dw = (dw_plus - dw_minus - ltd) * params["NMDA_stdp_rate"] * dt

        return {
            "STDP_sNMDA": s + d_s * dt,
            "STDP_gNMDA": jnp.clip(g + total_dw, a_min=0.0, a_max=1.0e-4),
            "NMDA_pre_trace": tr_pre + d_tr_pre * dt,
            "NMDA_post_trace": tr_post + d_tr_post * dt
        }

    def compute_current(self, states, pre_v, post_v, params):
        mg_block = 1.0 / (1.0 + params["NMDA_Mg2"] * jnp.exp(-0.062 * post_v / 3.57))
        return states["STDP_gNMDA"] * states["STDP_sNMDA"] * mg_block * (post_v - params["STDP_ENMDA"])


