"""Quick d=3 comparison: what does the d=3 fully-plugged graph look like?"""
import sys, time
from pathlib import Path
from copy import deepcopy
from fractions import Fraction

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))

import tsim
import pyzx_param as zx
from tsim.core.graph import prepare_graph, connected_components, get_params
from tsim_cutting import _plug_outputs, _find_zx_components, _remove_phase_terms
from stab_rank_cut import tcount

from d_3_circuit_definitions import circuit_source_injection_T, circuit_source_projection_proj

# Build d=3 circuit exactly like run.py does
import re, stim
from run import replace_t_with_s, replace_s_with_t, add_noise, compute_projection_obs_include

noise_strength = 0.001

# Same logic as run.py main()
clifford_source = replace_t_with_s(circuit_source_injection_T)
clifford_circuit = stim.Circuit(clifford_source)
noisy_clifford = add_noise(circuit=clifford_circuit, noise_strength=noise_strength)
noisy_injection_str = replace_s_with_t(noisy_clifford)

c_injection = tsim.Circuit(noisy_injection_str)
c_projection = tsim.Circuit(circuit_source_projection_proj)
circ = c_injection + c_projection

print("=== d=3 Circuit Analysis ===")
stim_circ = stim.Circuit(str(noisy_clifford))  # Clifford version for stats
print(f"Qubits: {stim_circ.num_qubits}")

# Prepare graph
prepared = prepare_graph(circ, sample_detectors=True)
print(f"\nAfter prepare_graph:")
print(f"  Num outputs: {prepared.num_outputs}")
print(f"  Num vertices: {len(list(prepared.graph.vertices()))}")
print(f"  Num edges: {len(list(prepared.graph.edges()))}")

f_params = [p for p in get_params(prepared.graph) if p.startswith('f')]
print(f"  Num f-params: {len(f_params)}")

# Connected components
components = connected_components(prepared.graph)
components_sorted = sorted(components, key=lambda c: len(c.output_indices))
print(f"\nConnected components: {len(components_sorted)}")

for i, cc in enumerate(components_sorted):
    g = cc.graph
    n_verts = len(list(g.vertices()))
    n_outputs = len(g.outputs())
    n_f = sum(1 for p in get_params(g) if p.startswith('f'))
    if n_outputs > 1:
        print(f"\n  Component {i}: {n_verts} verts, {n_outputs} outputs, {n_f} f-params  *** NON-TRIVIAL ***")

        # Plug all outputs
        output_indices = cc.output_indices
        component_m_chars = [f"m{idx}" for idx in output_indices]
        plugged_graphs = _plug_outputs(g, component_m_chars, [0, n_outputs])

        g_level0 = deepcopy(plugged_graphs[0])
        zx.full_reduce(g_level0, paramSafe=True)
        g_level0.normalize()
        power2_base = g_level0.scalar.power2

        g_plugged = plugged_graphs[1]
        g_copy = deepcopy(g_plugged)
        zx.full_reduce(g_copy, paramSafe=True)
        g_copy.normalize()
        g_copy.scalar.add_power(-power2_base)
        _remove_phase_terms(g_copy)

        tc = tcount(g_copy)
        n_plug_verts = len(list(g_copy.vertices()))
        print(f"    Fully-plugged graph: {n_plug_verts} verts, T-count={tc}")

        zx_comps = _find_zx_components(g_copy)
        print(f"    Sub-components after plugging: {len(zx_comps)}")
        for ci, cv in enumerate(zx_comps):
            print(f"      Sub-comp {ci}: {len(cv)} vertices")
    else:
        pass  # skip trivial components

print(f"\n  (Trivial 1-output components: {sum(1 for cc in components_sorted if len(cc.graph.outputs()) == 1)})")
