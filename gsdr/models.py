import jax
import jax.numpy as jnp
import numpy as np
import jaxley as jx
from jaxley.channels import HH, Leak
from jaxley.synapses import Synapse
from jaxley.connect import fully_connect
from typing import Optional

# These dependencies (Inoise, GradedAMPA, GradedGABAa) were extracted 
# from the local biophys_jx_gsdr.py and are now part of the AAE package.

class Inoise(jx.channels.Channel):
    """Stochastic Ornstein-Uhlenbeck noise channel."""
    def __init__(self, name: str = None, initial_seed: Optional[int] = None, initial_amp_noise: Optional[float] = None, initial_tau: Optional[float] = None, initial_mean: Optional[float] = None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        self.channel_params = {"amp_noise": 0.01, "mean": 0.00, "tau": 20.0}
        if initial_seed is None:
            self.channel_params["seed"] = float(np.random.randint(0, 2**16 - 1))
        else:
            self.channel_params["seed"] = float(initial_seed)
        self.channel_params["amp_noise"] = float(initial_amp_noise) if initial_amp_noise is not None else 0.01
        self.channel_params["tau"] = float(initial_tau) if initial_tau is not None else 20.0
        self.channel_params["mean"] = float(initial_mean) if initial_mean is not None else 0.00
        self.channel_states = {"n": 0.00, "step": 0.0}
        self.current_name = "i_noise"

    def update_states(self, states, dt, v, params):
        n, step, seeds_int = states["n"], states["step"], params["seed"].astype(int)
        if seeds_int.ndim == 0:
            base_key = jax.random.PRNGKey(seeds_int)
        else:
            base_key = jax.vmap(jax.random.PRNGKey)(seeds_int)
        if step.ndim == 0:
            step_key = jax.random.fold_in(base_key, step.astype(int))
        else:
            step_key = jax.vmap(jax.random.fold_in)(base_key, step.astype(int))
        xi = jax.random.normal(step_key) if step_key.ndim == 1 else jax.vmap(jax.random.normal)(step_key)
        mu, sigma, tau = params["mean"], params["amp_noise"], params["tau"]
        drift = (mu - n) / tau * dt
        diffusion = sigma * jnp.sqrt(2.0 / tau) * xi * jnp.sqrt(dt)
        new_n = n + drift + diffusion
        new_n = jnp.where(jnp.isnan(new_n) | jnp.isinf(new_n), n, new_n)
        return {"n": new_n, "step": step + 1.0}

    def compute_current(self, states, v, params): return -states["n"]
    def init_state(self, states, v, params, delta_t): return {"n": params["mean"], "step": 0.0}

class GradedAMPA(jx.synapses.Synapse):
    def __init__(self, g: float = 2.5, tauD_AMPA: Optional[float] = None):
        super().__init__()
        self.synapse_params = {"gAMPA": g, "EAMPA": 0.0, "tauDAMPA": 5.0, "tauRAMPA": 0.2, "slopeAMPA": 5.0, "V_thAMPA": -20.0}
        if tauD_AMPA is not None: self.synapse_params["tauDAMPA"] = tauD_AMPA
        self.synapse_states = {"sAMPA": 0.1}
    def update_states(self, states, dt, pre_v, post_v, params):
        s = states["sAMPA"]
        activation = 0.5 * (1 + jnp.tanh((pre_v - params["V_thAMPA"]) / params["slopeAMPA"]))
        d_s = (-s / params["tauDAMPA"]) + activation * ((1 - s) / params["tauRAMPA"])
        new_s = s + d_s * dt
        # Physical realisticity barrier for float32 stability
        new_s = jnp.where(jnp.isnan(new_s) | jnp.isinf(new_s), s, new_s)
        return {"sAMPA": new_s}
    def compute_current(self, states, pre_v, post_v, params): return params["gAMPA"] * states["sAMPA"] * (post_v - params["EAMPA"])

class GradedGABAa(jx.synapses.Synapse):
    def __init__(self, g: float = 5.0, tauD_GABAa: Optional[float] = None):
        super().__init__()
        self.synapse_params = {"gGABAa": g, "EGABAa": -80.0, "tauDGABAa": 5.0, "tauRGABAa": 0.5, "slopeGABAa": 5.0, "V_thGABAa": -20.0}
        if tauD_GABAa is not None: self.synapse_params["tauDGABAa"] = tauD_GABAa
        self.synapse_states = {"sGABAa": 0.1}
    def update_states(self, states, dt, pre_v, post_v, params):
        s = states["sGABAa"]
        activation = 0.5 * (1 + jnp.tanh((pre_v - params["V_thGABAa"]) / params["slopeGABAa"]))
        d_s = (-s / params["tauDGABAa"]) + activation * ((1 - s) / params["tauRGABAa"])
        new_s = s + d_s * dt
        new_s = jnp.where(jnp.isnan(new_s) | jnp.isinf(new_s), s, new_s)
        return {"sGABAa": new_s}
    def compute_current(self, states, pre_v, post_v, params): return params["gGABAa"] * states["sGABAa"] * (post_v - params["EGABAa"])

class GradedGABAb(jx.synapses.Synapse):
    """Graded slow Inhibitory Synapse (GABAb)."""
    def __init__(self, g: float = 1.0, tauD_GABAb: Optional[float] = None):
        super().__init__()
        self.synapse_params = {
            "gGABAb": g, "EGABAb": -95.0, "tauDGABAb": 200.0, "tauRGABAb": 10.0, 
            "slopeGABAb": 5.0, "V_thGABAb": -20.0
        }
        if tauD_GABAb is not None: self.synapse_params["tauDGABAb"] = tauD_GABAb
        self.synapse_states = {"sGABAb": 0.01}
    def update_states(self, states, dt, pre_v, post_v, params):
        s = states["sGABAb"]
        activation = 0.5 * (1 + jnp.tanh((pre_v - params["V_thGABAb"]) / params["slopeGABAb"]))
        d_s = (-s / params["tauDGABAb"]) + activation * ((1 - s) / params["tauRGABAb"])
        new_s = s + d_s * dt
        new_s = jnp.where(jnp.isnan(new_s) | jnp.isinf(new_s), s, new_s)
        return {"sGABAb": new_s}
    def compute_current(self, states, pre_v, post_v, params): 
        return params["gGABAb"] * states["sGABAb"] * (post_v - params["EGABAb"])

def build_net_eig(num_e: int, num_ig: int, num_il: int, seed: Optional[int] = None):
    """
    Constructs a JAXley neural network with specified numbers of excitatory and inhibitory neurons.
    If seed is None, a random seed is generated for connectivity.
    """
    if seed is None:
        seed = int(np.random.randint(0, 2**31 - 1))
    np.random.seed(seed) # Set numpy seed for connectivity
    
    comp = jx.Compartment()
    branch = jx.Branch(comp, ncomp=2)

    e_cells = [jx.Cell(branch, parents=[-1, 0]) for _ in range(num_e)]
    i_cells_g = [jx.Cell(branch, parents=[-1, 0]) for _ in range(num_ig)]
    i_cells_l = [jx.Cell(branch, parents=[-1, 0]) for _ in range(num_il)]

    for cell in e_cells + i_cells_g + i_cells_l:
        r_d_tau_diff = np.random.uniform(5.0, 50.0)
        r_amp_diff = np.clip(np.random.uniform(-0.1, 0.1), 0.0, 0.1)
        cell.insert(HH())
        cell.insert(Inoise(initial_amp_noise=r_amp_diff, initial_tau=r_d_tau_diff, initial_mean=0.0))
        cell.radius = 1.0
        cell.length = 1.0

    net = jx.Network(e_cells + i_cells_g + i_cells_l)
    
    # E -> all (AMPA)
    pre = net.cell([ik for ik in range(num_e)]).branch(0).loc(0.0)
    post = net.cell([ik for ik in range(num_e+num_ig+num_il)]).branch(1).loc(1.0)
    fully_connect(pre, post, GradedAMPA(tauD_AMPA=2.0))

    # Ig -> all (GABAa)
    pre = net.cell([ik for ik in range(num_e, num_e+num_ig)]).branch(0).loc(0.0)
    post = net.cell([ik for ik in range(num_e+num_ig+num_il)]).branch(1).loc(1.0)
    fully_connect(pre, post, GradedGABAa(g=5.0, tauD_GABAa=5.0))

    # Il -> subset of all (GABAa)
    pre_il = net.cell([ik for ik in range(num_e+num_ig, num_e+num_ig+num_il)]).branch(0).loc(0.0)
    num_posts_to_select = int((num_e+num_ig+num_il)*0.1)
    posts_pool_indices = jnp.arange(0, num_e + num_ig + num_il)
    
    # Randomize connectivity key using the main seed
    key_conn = jax.random.PRNGKey(seed)
    selected_post_indices = np.array(jax.random.choice(key_conn, posts_pool_indices, shape=(num_posts_to_select,), replace=False))
    post_il = net.cell(selected_post_indices).branch(1).loc(1.0)
    fully_connect(pre_il, post_il, GradedGABAa(g=5.0, tauD_GABAa=5.0)) # Increased to 5.0ms

    return net


def build_pyramidal_mc(seed=None):
    """Multi-compartment Pyramidal cell (Soma + Distal Dendrite)."""
    if seed is None: seed = 42
    comp_soma = jx.Compartment()
    comp_dend = jx.Compartment()
    # Simple cable: Soma (0) -> Dendrite (1)
    cell = jx.Cell([comp_soma, comp_dend], parents=[-1, 0])
    cell.radius = 1.0
    cell.length = 100.0 # Length drives axial resistance
    return cell


# --- Cell Models (Pyramidal, PV, SST, VIP) ---

def build_pyramidal_cell():
    """Regular Spiking (RS) Pyramidal Cell with adaptation."""
    comp_soma = jx.Compartment()
    comp_dend = jx.Compartment()
    cell = jx.Cell([comp_soma, comp_dend], parents=[-1, 0])
    cell.radius = 1.0
    cell.length = 100.0
    # High adaptation, standard HH
    cell.insert(jx.channels.HH())
    # Assuming Jaxley HH has gK, gNa, gl. 
    # For RS: gK is moderate, gNa is moderate.
    return cell

def build_pv_cell():
    """Fast Spiking (FS) Parvalbumin Interneuron."""
    comp_soma = jx.Compartment()
    cell = jx.Cell([comp_soma], parents=[-1])
    cell.radius = 1.0
    cell.length = 10.0
    cell.insert(jx.channels.HH())
    # FS typically has very high gNa and gK for fast repolarization
    return cell

def build_sst_cell():
    """Low-Threshold Spiking (LTS) Somatostatin Interneuron."""
    comp_soma = jx.Compartment()
    cell = jx.Cell([comp_soma], parents=[-1])
    cell.radius = 1.0
    cell.length = 10.0
    cell.insert(jx.channels.HH())
    return cell

def build_vip_cell():
    """Irregular Spiking / Bursting VIP Interneuron."""
    comp_soma = jx.Compartment()
    cell = jx.Cell([comp_soma], parents=[-1])
    cell.radius = 0.5 # Smaller size, lower capacitance
    cell.length = 10.0
    cell.insert(jx.channels.HH())
    return cell
