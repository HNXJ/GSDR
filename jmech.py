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


class GradedGABab(Synapse):
    """
    Graded slow Inhibitory Synapse. (GABab)
    Ref: Golowasch et al. (1999) ; Prinz, A. A., et al. (2004).
    """
    def __init__(self, tauD_GABab: Optional[float] = None):
        super().__init__()
        r_d_gSyn = np.random.uniform(0.5, 9.5) # Conductance range
        self.synapse_params = {
            "gGABab": r_d_gSyn,       # Conductance (uS)
            "EGABab": -95.0,     # Reversal Potential (mV)
            "tauDGABab": 200.0,        # Decay (ms)
            "tauRGABab": 10.0,         # Rise (ms)
            "slopeGABab": 5.0,        # Steepness of activation
            "V_thGABab": -20.0        # Threshold (mV) - Tuned to activate during spikes
        }

        if tauD_GABab is not None:
            self.synapse_params["tauDGABab"] = tauD_GABab

        self.synapse_states = {"sGABab": 0.01}

    def update_states(self, states, dt, pre_v, post_v, params):
        # s' = -s/tauD + 0.5*(1+tanh((V-Vth)/slope)) * ((1-s)/tauR)
        s = states["sGABab"]
        activation = 0.5 * (1 + jnp.tanh((pre_v - params["V_thGABab"]) / params["slopeGABab"]))
        d_s = (-s / params["tauDGABab"]) + activation * ((1 - s) / params["tauRGABab"])
        return {"sGABab": s + d_s * dt}

    def compute_current(self, states, pre_v, post_v, params):
        return params["gGABab"] * states["sGABab"] * (post_v - params["EGABab"])


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
        activation = 0.5 * (1 + jnp.tanh((pre_v - params["V_thnACH"]) / params["slopenACH"])) # Sigmoidal activation
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
