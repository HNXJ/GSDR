"""
Pure-JAX Laminar Izhikevich Network Builder

net_eig_izh_jax: Implements a reduced-spiking cortical microcircuit builder
using Izhikevich point neurons in JAX, without dependency on Jaxley, Equinox,
Diffrax, Optax, or MLX.

Izhikevich model: a reduced spiking abstraction preserving spike timing, reset
dynamics, and nonlinear firing regimes. NOT a claim of biological optimality.

This builder returns JAX pytrees compatible with GSDR/AGSDR workflows, with
tagged connectivity masks for selective freezing during optimization.
"""

import jax
import jax.numpy as jnp
from typing import Dict, List, Optional, Tuple, Any
import dataclasses


# ============================================================================
# CELL PRESETS (smoke-test calibration only, not fitted biological identities)
# ============================================================================

CELL_PRESETS = {
    "E": {
        "a": 0.02,
        "b": 0.20,
        "c": -65.0,
        "d": 8.0,
    },
    "PV": {
        "a": 0.10,
        "b": 0.20,
        "c": -65.0,
        "d": 2.0,
    },
    "SST": {
        "a": 0.02,
        "b": 0.25,
        "c": -65.0,
        "d": 2.0,
    },
    "VIP": {
        "a": 0.02,
        "b": 0.20,
        "c": -60.0,
        "d": 4.0,
    },
}

CELL_TYPES = ["E", "PV", "SST", "VIP"]


# ============================================================================
# CONFIG VALIDATION AND ROUNDING
# ============================================================================

def _largest_remainder_rounding(
    raw_counts: List[float],
) -> List[int]:
    """
    Distribute counts deterministically using largest-remainder method.

    Args:
        raw_counts: fractional counts, sum ≈ total

    Returns:
        integer counts, sum == total (integer part of sum)
    """
    total = int(jnp.round(jnp.sum(jnp.array(raw_counts))))
    base_counts = [int(c) for c in raw_counts]
    remainders = [raw_counts[i] - base_counts[i] for i in range(len(raw_counts))]

    remaining = total - sum(base_counts)

    # Sort by remainder descending, with index tiebreak (lower index wins)
    indices_by_remainder = sorted(
        range(len(remainders)),
        key=lambda i: (-remainders[i], i)
    )

    result = base_counts.copy()
    for i in range(remaining):
        idx = indices_by_remainder[i]
        result[idx] += 1

    return result


def _validate_config(config: Dict) -> Tuple[Dict, Dict]:
    """
    Validate and resolve config, return (resolved_config, warnings_metadata).
    """
    warnings = []

    # --- Defaults ---
    resolved = {
        "n_neurons": config.get("n_neurons", 100),
        "composition": config.get("composition", [[0.8, 0.2, 0.0, 0.0]]),
        "layer_edges_um": config.get("layer_edges_um", [0.0, 1000.0]),
        "granular_layer": config.get("granular_layer", 1),
        "dt_ms": config.get("dt_ms", 0.1),
        "t_ms": config.get("t_ms", 1000.0),
        "seed": config.get("seed", 0),
        "allow_autapses": config.get("allow_autapses", False),
        "train_intrinsic": config.get("train_intrinsic", False),
        "v_init": config.get("v_init", -65.0),
    }

    n_neurons = resolved["n_neurons"]
    composition = resolved["composition"]
    layer_edges_um = resolved["layer_edges_um"]
    granular_layer = resolved["granular_layer"]

    # ---- Validate composition ----
    composition = [list(row) for row in composition]
    n_layers = len(composition)

    for i, row in enumerate(composition):
        if len(row) != 4:
            raise ValueError(f"composition row {i} has {len(row)} columns, expected 4")

        if any(c < 0 for c in row):
            raise ValueError(f"composition row {i} has negative values")

        row_sum = sum(row)
        if row_sum == 0:
            raise ValueError(f"composition row {i} has zero sum (all types have zero count)")

        # Normalize if sum != 1.0
        if abs(row_sum - 1.0) > 1e-6:
            composition[i] = [c / row_sum for c in row]
            warnings.append(f"composition row {i} normalized from sum={row_sum:.4f} to 1.0")

    resolved["composition"] = composition

    # ---- Validate layer_edges ----
    if len(layer_edges_um) != n_layers + 1:
        raise ValueError(
            f"layer_edges_um has {len(layer_edges_um)} elements, "
            f"expected {n_layers + 1} (n_layers + 1)"
        )

    for i in range(len(layer_edges_um) - 1):
        if layer_edges_um[i] >= layer_edges_um[i + 1]:
            raise ValueError(f"layer_edges_um not strictly increasing at ({i}, {i+1})")

    # ---- Validate granular_layer ----
    if not (1 <= granular_layer <= n_layers):
        raise ValueError(
            f"granular_layer={granular_layer} out of range [1, {n_layers}]"
        )

    granular_layer_index0 = granular_layer - 1
    resolved["granular_layer_index0"] = granular_layer_index0

    return resolved, {"warnings": warnings}


def _allocate_neurons(
    n_neurons: int,
    layer_edges: List[float],
    composition: List[List[float]],
    seed: int,
) -> Tuple[List[int], List[List[int]]]:
    """
    Allocate neurons to layers and cell types using largest-remainder rounding.

    Returns:
        (layer_counts, cell_counts_by_layer)
        - layer_counts: [n_neurons_per_layer]
        - cell_counts_by_layer: [[n_E, n_PV, n_SST, n_VIP] for each layer]
    """
    total_thickness = layer_edges[-1] - layer_edges[0]

    # Allocate neurons to layers
    layer_raw = [
        n_neurons * (layer_edges[i + 1] - layer_edges[i]) / total_thickness
        for i in range(len(layer_edges) - 1)
    ]
    layer_counts = _largest_remainder_rounding(layer_raw)

    # Allocate cells within each layer
    cell_counts_by_layer = []
    for layer_idx, layer_n in enumerate(layer_counts):
        comp_row = composition[layer_idx]
        cell_raw = [layer_n * comp_row[ct_idx] for ct_idx in range(4)]
        cell_counts = _largest_remainder_rounding(cell_raw)
        cell_counts_by_layer.append(cell_counts)

    return layer_counts, cell_counts_by_layer


# ============================================================================
# NETWORK BUILDER
# ============================================================================

def net_eig_izh_jax(config: Optional[Dict] = None) -> Dict:
    """
    Build a pure-JAX laminar Izhikevich network.

    If config is None, returns an E-PV (80% E, 20% PV) 100-neuron smoke network.

    Args:
        config: configuration dict with keys:
            - n_neurons: total neurons (default 100)
            - composition: [[E, PV, SST, VIP], ...] per layer (default [[0.8, 0.2, 0, 0]])
            - layer_edges_um: layer boundaries in μm (default [0, 1000])
            - granular_layer: 1-based layer index (default 1)
            - dt_ms: integration timestep (default 0.1)
            - t_ms: simulation duration (default 1000)
            - seed: RNG seed (default 0)
            - allow_autapses: allow self-connections (default False)
            - train_intrinsic: optimize a,b,c,d parameters (default False)

    Returns:
        dict with:
            - config: resolved configuration
            - metadata: warnings, layer info, counts
            - cell_table: list of cell info dicts
            - params: JAX pytree with g_ampa, g_nmda, g_gaba_a, a, b, c, d
            - tag_masks: dict of boolean [N, N] masks per connectivity tag
            - trainable_masks: dict of trainable parameter masks
            - simulate: callable(params, input_current=None, key=None) -> traces
            - project_params: callable for projecting parameters to trainable subset
    """

    if config is None:
        print("config was empty, returning an E-PV (80%E-20%PV) 100-neuron pure-JAX Izhikevich network")
        config = {}

    # ---- Validate and resolve config ----
    resolved_config, config_metadata = _validate_config(config)

    n_neurons = resolved_config["n_neurons"]
    composition = resolved_config["composition"]
    layer_edges = resolved_config["layer_edges_um"]
    granular_layer_idx = resolved_config["granular_layer_index0"]
    dt_ms = resolved_config["dt_ms"]
    t_ms = resolved_config["t_ms"]
    seed = resolved_config["seed"]
    allow_autapses = resolved_config["allow_autapses"]
    train_intrinsic = resolved_config["train_intrinsic"]
    v_init = resolved_config["v_init"]

    n_layers = len(composition)

    # ---- Allocate neurons ----
    layer_counts, cell_counts_by_layer = _allocate_neurons(
        n_neurons, layer_edges, composition, seed
    )

    # ---- Build cell table ----
    cell_table = []
    neuron_id = 0

    for layer_idx in range(n_layers):
        layer_num = layer_idx + 1
        layer_n = layer_counts[layer_idx]
        cell_counts = cell_counts_by_layer[layer_idx]

        # Determine layer role
        if n_layers == 1:
            layer_role = "granular"
        elif layer_idx == granular_layer_idx:
            layer_role = "granular"
        elif layer_idx < granular_layer_idx:
            layer_role = "supragranular"
        else:
            layer_role = "infragranular"

        # Allocate cells
        layer_min_um = layer_edges[layer_idx]
        layer_max_um = layer_edges[layer_idx + 1]

        for ct_idx, ct_name in enumerate(CELL_TYPES):
            ct_count = cell_counts[ct_idx]

            for ct_pos in range(ct_count):
                # Depth within layer (deterministic)
                depth_frac = ct_pos / max(ct_count, 1)
                depth_um = layer_min_um + depth_frac * (layer_max_um - layer_min_um)

                preset = CELL_PRESETS[ct_name]

                cell_table.append({
                    "neuron_id": neuron_id,
                    "layer": layer_num,
                    "layer_index0": layer_idx,
                    "layer_role": layer_role,
                    "cell_type": ct_name,
                    "depth_um": depth_um,
                    "a": preset["a"],
                    "b": preset["b"],
                    "c": preset["c"],
                    "d": preset["d"],
                })

                neuron_id += 1

    # ---- Build parameter arrays ----
    a_vals = jnp.array([c["a"] for c in cell_table], dtype=jnp.float32)
    b_vals = jnp.array([c["b"] for c in cell_table], dtype=jnp.float32)
    c_vals = jnp.array([c["c"] for c in cell_table], dtype=jnp.float32)
    d_vals = jnp.array([c["d"] for c in cell_table], dtype=jnp.float32)

    # Conductance initialization (conservative, normalized by network size)
    norm = max(n_neurons, 1)
    g_ampa = jnp.zeros((n_neurons, n_neurons), dtype=jnp.float32)
    g_nmda = jnp.zeros((n_neurons, n_neurons), dtype=jnp.float32)
    g_gaba_a = jnp.zeros((n_neurons, n_neurons), dtype=jnp.float32)

    # Fill conductances based on cell type
    for pre_id in range(n_neurons):
        pre_type = cell_table[pre_id]["cell_type"]

        for post_id in range(n_neurons):
            # Skip autapses if disabled
            if not allow_autapses and pre_id == post_id:
                continue

            # E source: AMPA and NMDA
            if pre_type == "E":
                g_ampa = g_ampa.at[pre_id, post_id].set(0.5 / norm)
                g_nmda = g_nmda.at[pre_id, post_id].set(0.1 / norm)

            # Inhibitory source: GABA_A
            elif pre_type in ["PV", "SST", "VIP"]:
                g_gaba_a = g_gaba_a.at[pre_id, post_id].set(1.0 / norm)

    params = {
        "g_ampa": g_ampa,
        "g_nmda": g_nmda,
        "g_gaba_a": g_gaba_a,
        "a": a_vals,
        "b": b_vals,
        "c": c_vals,
        "d": d_vals,
    }

    # ---- Build tag masks ----
    tag_masks = {}

    for pre_id in range(n_neurons):
        pre_type = cell_table[pre_id]["cell_type"]
        pre_layer = cell_table[pre_id]["layer"]

        for post_id in range(n_neurons):
            post_type = cell_table[post_id]["cell_type"]
            post_layer = cell_table[post_id]["layer"]

            # Skip if no valid connection
            if pre_type == "E":
                synapses = ["AMPA", "NMDA"]
            elif pre_type in ["PV", "SST", "VIP"]:
                synapses = ["GABA_A"]
            else:
                synapses = []

            for syn_type in synapses:
                tag = f"L{pre_layer}:{pre_type}->L{post_layer}:{post_type}:{syn_type}"

                if tag not in tag_masks:
                    tag_masks[tag] = jnp.zeros((n_neurons, n_neurons), dtype=jnp.bool_)

                tag_masks[tag] = tag_masks[tag].at[pre_id, post_id].set(True)

    # Remove empty tags
    tag_masks = {k: v for k, v in tag_masks.items() if v.sum() > 0}

    # ---- Trainable masks ----
    trainable_masks = {
        "g_ampa": jnp.ones_like(params["g_ampa"], dtype=jnp.bool_),
        "g_nmda": jnp.ones_like(params["g_nmda"], dtype=jnp.bool_),
        "g_gaba_a": jnp.ones_like(params["g_gaba_a"], dtype=jnp.bool_),
        "a": jnp.ones_like(params["a"], dtype=jnp.bool_) if train_intrinsic else jnp.zeros_like(params["a"], dtype=jnp.bool_),
        "b": jnp.ones_like(params["b"], dtype=jnp.bool_) if train_intrinsic else jnp.zeros_like(params["b"], dtype=jnp.bool_),
        "c": jnp.ones_like(params["c"], dtype=jnp.bool_) if train_intrinsic else jnp.zeros_like(params["c"], dtype=jnp.bool_),
        "d": jnp.ones_like(params["d"], dtype=jnp.bool_) if train_intrinsic else jnp.zeros_like(params["d"], dtype=jnp.bool_),
    }

    # ---- Simulation function ----
    def simulate_fn(
        params: Dict,
        input_current: Optional[jnp.ndarray] = None,
        key: Optional[jnp.ndarray] = None,
    ) -> Dict:
        """
        Simulate network using Izhikevich dynamics with jax.lax.scan.

        Args:
            params: pytree with g_ampa, g_nmda, g_gaba_a, a, b, c, d
            input_current: [T, N] external current, or None for smoke input
            key: optional RNG key

        Returns:
            dict with v_trace, u_trace, spike_trace, input_current, status
        """

        T = int(jnp.round(t_ms / dt_ms))
        N = n_neurons

        # Generate or validate input current
        if input_current is None:
            # Deterministic smoke input: cell-type-specific tonic drive
            input_current = jnp.zeros((T, N), dtype=jnp.float32)

            for nid in range(N):
                cell_type = cell_table[nid]["cell_type"]
                # Tonic drive sufficient to produce spikes
                if cell_type == "E":
                    drive = 10.0
                else:
                    drive = 5.0

                input_current = input_current.at[:, nid].set(drive)

        # Initial state
        v0 = jnp.full(N, v_init, dtype=jnp.float32)
        u0 = params["b"] * v0
        s_ampa0 = jnp.zeros(N, dtype=jnp.float32)
        s_nmda0 = jnp.zeros(N, dtype=jnp.float32)
        s_gaba_a0 = jnp.zeros(N, dtype=jnp.float32)

        state0 = (v0, u0, s_ampa0, s_nmda0, s_gaba_a0)

        # Simulation step
        def step_fn(state, inp_t):
            v, u, s_ampa, s_nmda, s_gaba_a = state
            I_ext = inp_t

            # Synaptic currents
            def _syn_current(g, s_pre, v_post, e_rev):
                current_matrix = g * s_pre[:, None] * (e_rev - v_post[None, :])
                return jnp.sum(current_matrix, axis=0)

            I_ampa = _syn_current(params["g_ampa"], s_ampa, v, 0.0)
            I_nmda = _syn_current(params["g_nmda"], s_nmda, v, 0.0)
            I_gaba_a = _syn_current(params["g_gaba_a"], s_gaba_a, v, -70.0)

            I_syn = I_ampa + I_nmda + I_gaba_a

            # Izhikevich dynamics
            dv = 0.04 * v**2 + 5.0 * v + 140.0 - u + I_ext + I_syn
            du = params["a"] * (params["b"] * v - u)

            v_new = v + dv * (dt_ms / 1000.0)
            u_new = u + du * (dt_ms / 1000.0)

            # Spike reset
            spikes = (v_new >= 30.0).astype(jnp.float32)
            v_new = jnp.where(v_new >= 30.0, params["c"], v_new)
            u_new = jnp.where(v_new >= 30.0, u_new + params["d"], u_new)

            # Synaptic state decay
            tau_ampa = 5.0
            tau_nmda = 100.0
            tau_gaba_a = 10.0

            dt_s = dt_ms / 1000.0
            s_ampa_new = s_ampa * jnp.exp(-dt_s / tau_ampa) + spikes
            s_nmda_new = s_nmda * jnp.exp(-dt_s / tau_nmda) + spikes
            s_gaba_a_new = s_gaba_a * jnp.exp(-dt_s / tau_gaba_a) + spikes

            s_ampa_new = jnp.clip(s_ampa_new, 0.0, 1.0)
            s_nmda_new = jnp.clip(s_nmda_new, 0.0, 1.0)
            s_gaba_a_new = jnp.clip(s_gaba_a_new, 0.0, 1.0)

            state_new = (v_new, u_new, s_ampa_new, s_nmda_new, s_gaba_a_new)

            return state_new, (v_new, u_new, spikes)

        # Scan
        final_state, (v_trace, u_trace, spike_trace) = jax.lax.scan(
            step_fn, state0, input_current
        )

        # Check for invalid traces
        invalid_v = ~jnp.isfinite(v_trace).all()
        invalid_u = ~jnp.isfinite(u_trace).all()
        invalid_spike = ~jnp.isfinite(spike_trace).all()
        invalid_input = ~jnp.isfinite(input_current).all()

        status = "PASS"
        invalid_reason = None

        if invalid_v or invalid_u or invalid_spike or invalid_input:
            status = "REJECT_INVALID"
            reasons = []
            if invalid_v:
                reasons.append("v_trace")
            if invalid_u:
                reasons.append("u_trace")
            if invalid_spike:
                reasons.append("spike_trace")
            if invalid_input:
                reasons.append("input_current")
            invalid_reason = ", ".join(reasons)

        return {
            "v_trace": v_trace,
            "u_trace": u_trace,
            "spike_trace": spike_trace,
            "input_current": input_current,
            "status": status,
            "invalid_reason": invalid_reason,
            "metadata": {
                "n_neurons": N,
                "n_timesteps": T,
                "dt_ms": dt_ms,
                "total_duration_ms": t_ms,
            },
        }

    # ---- Parameter projection function ----
    def project_params(
        old_params: Dict,
        proposed_params: Dict,
        trainable_masks: Dict,
    ) -> Dict:
        """
        Project proposed parameters onto trainable subset.

        Args:
            old_params: original parameters
            proposed_params: new parameter values
            trainable_masks: dict of boolean masks

        Returns:
            projected parameters
        """
        projected = {}

        for key in old_params:
            if key in ["g_ampa", "g_nmda", "g_gaba_a"]:
                # Conductance matrices: clip to [0, 10]
                proposed_clipped = jnp.clip(proposed_params[key], 0.0, 10.0)
                mask = trainable_masks[key]
                projected[key] = jnp.where(mask, proposed_clipped, old_params[key])

            elif key in ["a", "b", "c", "d"]:
                # Intrinsic parameters: respect trainable mask
                mask = trainable_masks[key]
                projected[key] = jnp.where(mask, proposed_params[key], old_params[key])

        return projected

    # ---- Metadata ----
    metadata = {
        "n_neurons": n_neurons,
        "n_layers": n_layers,
        "layer_counts": layer_counts,
        "cell_counts_by_layer": cell_counts_by_layer,
        "n_tags": len(tag_masks),
        "warnings": config_metadata["warnings"],
        "dt_ms": dt_ms,
        "t_ms": t_ms,
        "allow_autapses": allow_autapses,
        "train_intrinsic": train_intrinsic,
    }

    return {
        "config": resolved_config,
        "metadata": metadata,
        "cell_table": cell_table,
        "params": params,
        "tag_masks": tag_masks,
        "trainable_masks": trainable_masks,
        "simulate": simulate_fn,
        "project_params": project_params,
    }
