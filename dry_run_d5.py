"""d=5 compile-only dry run.

Loads Gidney's d=5 cultivation circuit, runs it through the compilation
pipeline (graph prep → connected components → cutting decomposition),
and reports the key metrics that determine computational tractability:
  - Number of connected components
  - Outputs (measurements) per component
  - Clifford terms after cutting decomposition per component
  - Total enumeration cost
"""

import sys, time
from pathlib import Path
from copy import deepcopy

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))

from fractions import Fraction
import tsim
import pyzx_param as zx
from tsim.core.graph import prepare_graph, connected_components, get_params
from tsim_cutting import find_stab_cutting, _plug_outputs, _find_zx_components, _remove_phase_terms

# ============================================================================
# Load the d=5 circuit
# ============================================================================

CIRCUIT_PATH = Path("/Users/ritesh/Downloads/prx/gidney-circuits/circuits/"
                    "for_perfectionist_decoding/"
                    "c=inject[unitary]+cultivate,p=0.001,noise=uniform,"
                    "g=css,q=42,b=Y,r=10,d1=5.stim")

print(f"Loading d=5 circuit from:\n  {CIRCUIT_PATH}")
circuit_str = CIRCUIT_PATH.read_text()

# Gidney's circuit uses S/S_DAG (Clifford proxy). For exact simulation,
# replace with T/T_DAG (non-Clifford). This is the same approach as
# run.py's replace_s_with_t() applied in reverse.
import re
def replace_s_with_t(s):
    s = re.sub(r'^(\s*)S_DAG(\s)', r'\1T_DAG\2', s, flags=re.MULTILINE)
    return re.sub(r'^(\s*)S(\s)', r'\1T\2', s, flags=re.MULTILINE)

circuit_str_t = replace_s_with_t(circuit_str)

# Verify replacement
n_s = len(re.findall(r'^\s*S[ _]', circuit_str, re.MULTILINE))
n_t = len(re.findall(r'^\s*T[ _]', circuit_str_t, re.MULTILINE))
print(f"  Replaced {n_s} S/S_DAG gates → {n_t} T/T_DAG gates")

circuit = tsim.Circuit(circuit_str_t)

import stim
stim_circ = stim.Circuit(circuit_str)
print(f"\nCircuit stats:")
print(f"  Qubits: {stim_circ.num_qubits}")
print(f"  Measurements: {stim_circ.num_measurements}")
print(f"  Detectors: {stim_circ.num_detectors}")
print(f"  Observables: {stim_circ.num_observables}")

# Count T gates
t_count = 0
for inst in stim_circ.flattened():
    if inst.name in ('S_DAG', 'S'):
        # In Gidney's circuits, S/S_DAG are Clifford (not T proxies)
        pass
    # In the actual circuit, non-Clifford gates would be T/T_DAG
    # But Gidney's cultivation circuits use S_DAG as Clifford proxy
    # The actual non-Clifford behavior comes from the MPP at the end

# ============================================================================
# Step 1: Prepare graph
# ============================================================================

print(f"\n{'='*60}")
print("Step 1: Preparing ZX graph (parse → build → reduce)")
print(f"{'='*60}")

t0 = time.time()
prepared = prepare_graph(circuit, sample_detectors=True)
t_prep = time.time() - t0

print(f"  Time: {t_prep:.2f}s")
print(f"  Num outputs: {prepared.num_outputs}")
print(f"  Num detectors: {prepared.num_detectors}")
print(f"  Num vertices in graph: {len(list(prepared.graph.vertices()))}")
print(f"  Num edges in graph: {len(list(prepared.graph.edges()))}")

# Count f-params (noise channels)
f_params = [p for p in get_params(prepared.graph) if p.startswith('f')]
m_params = [p for p in get_params(prepared.graph) if p.startswith('m')]
print(f"  Num f-params (noise): {len(f_params)}")
print(f"  Num m-params (measurements): {len(m_params)}")

# ============================================================================
# Step 2: Connected components
# ============================================================================

print(f"\n{'='*60}")
print("Step 2: Finding connected components")
print(f"{'='*60}")

components = connected_components(prepared.graph)
components_sorted = sorted(components, key=lambda c: len(c.output_indices))

print(f"  Total components: {len(components_sorted)}")
print()

for i, cc in enumerate(components_sorted):
    g = cc.graph
    n_verts = len(list(g.vertices()))
    n_edges = len(list(g.edges()))
    n_outputs = len(g.outputs())
    n_out_idx = len(cc.output_indices)
    cc_params = get_params(g)
    n_f = sum(1 for p in cc_params if p.startswith('f'))
    n_m = sum(1 for p in cc_params if p.startswith('m'))
    print(f"  Component {i}: {n_verts} verts, {n_edges} edges, "
          f"{n_outputs} outputs, {n_f} f-params, {n_m} m-params")

# ============================================================================
# Step 3: Cutting decomposition (the key step)
# ============================================================================

print(f"\n{'='*60}")
print("Step 3: Cutting decomposition per component")
print(f"{'='*60}")

total_terms = 0
total_combos = 0
max_outputs = 0

for i, cc in enumerate(components_sorted):
    g = cc.graph
    n_outputs = len(g.outputs())
    output_indices = cc.output_indices

    print(f"\n  --- Component {i} ({n_outputs} outputs) ---")

    if n_outputs <= 1:
        print(f"    ≤1 output → autoregressive (safe, trivial)")
        # These are simple Bernoulli components — count terms for reference
        g_copy = deepcopy(g)
        zx.full_reduce(g_copy, paramSafe=True)
        # Count non-Clifford phases
        n_tcount = sum(1 for v in g_copy.vertices()
                       if hasattr(g_copy, 'phase') and
                       g_copy.phase(v) not in (0, 1, Fraction(1,2), Fraction(3,2)))
        print(f"    T-count after reduce: {n_tcount}")
        total_terms += 1  # at least 1 term per component
        continue

    max_outputs = max(max_outputs, n_outputs)
    n_combos = 2 ** n_outputs
    total_combos += n_combos

    # Replicate what _compile_component_enum_general does
    component_m_chars = [f"m{idx}" for idx in output_indices]

    # Plug all outputs
    print(f"    Plugging {n_outputs} outputs...")
    t0 = time.time()
    plugged_graphs = _plug_outputs(g, component_m_chars, [0, n_outputs])

    # Level-0 normalization
    g_level0 = deepcopy(plugged_graphs[0])
    zx.full_reduce(g_level0, paramSafe=True)
    g_level0.normalize()
    power2_base = g_level0.scalar.power2

    # Fully-plugged with normalization
    g_plugged = plugged_graphs[1]
    g_copy = deepcopy(g_plugged)
    zx.full_reduce(g_copy, paramSafe=True)
    g_copy.normalize()
    g_copy.scalar.add_power(-power2_base)
    _remove_phase_terms(g_copy)
    t_plug = time.time() - t0
    print(f"    Plugging + reduce time: {t_plug:.2f}s")

    # Check for disconnection
    zx_comps = _find_zx_components(g_copy)
    n_subcomps = len(zx_comps)

    if n_subcomps >= 2:
        print(f"    DISCONNECTS into {n_subcomps} sub-components!")
    else:
        print(f"    Does NOT disconnect (monolithic)")

    # Count vertices per sub-component
    for ci, comp_verts in enumerate(zx_comps):
        print(f"      Sub-component {ci}: {len(comp_verts)} vertices")

    # Cutting decomposition on the fully-plugged graph
    print(f"    Running cutting decomposition...")
    t0 = time.time()

    if n_subcomps >= 2:
        # Process each sub-component separately
        from tsim_cutting import _extract_subgraph
        f_indices_global = [int(p[1:]) for p in f_params]
        param_names = [f"f{i}" for i in f_indices_global]
        param_names += [f"m{output_indices[j]}" for j in range(n_outputs)]

        for ci, comp_verts in enumerate(zx_comps):
            sub_g = _extract_subgraph(g_copy, comp_verts, reset_scalar=(ci > 0))
            sub_params_set = set(get_params(sub_g))
            sub_param_names = [p for p in param_names if p in sub_params_set]

            try:
                sub_terms = find_stab_cutting(sub_g, max_cut_iterations=10, debug=False)
                print(f"      Sub-comp {ci}: {len(sub_terms)} Clifford terms, "
                      f"{len(sub_param_names)} params")
                total_terms += len(sub_terms)
            except Exception as e:
                print(f"      Sub-comp {ci}: ERROR - {e}")
    else:
        # Monolithic
        try:
            terms = find_stab_cutting(g_copy, max_cut_iterations=10, debug=True)
            t_cut = time.time() - t0
            print(f"    Terms after cutting: {len(terms)}")
            print(f"    Cutting time: {t_cut:.2f}s")
            total_terms += len(terms)
        except Exception as e:
            t_cut = time.time() - t0
            print(f"    ERROR in cutting: {e}")
            print(f"    Time before error: {t_cut:.2f}s")

    print(f"    Enumeration combos: {n_combos}")
    if n_outputs > 12:
        print(f"    ⚠ EXCEEDS MAX_ENUM_OUTPUTS_GENERAL=12!")

# ============================================================================
# Summary
# ============================================================================

print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"  Total components: {len(components_sorted)}")
print(f"  Total Clifford terms: {total_terms}")
print(f"  Max outputs in any component: {max_outputs}")
print(f"  Total enumeration combos: {total_combos}")
print(f"\n  Feasibility assessment:")
if max_outputs <= 12:
    print(f"    ✓ All components within MAX_ENUM_OUTPUTS_GENERAL=12")
else:
    print(f"    ✗ Some components exceed MAX_ENUM_OUTPUTS_GENERAL=12!")
if total_terms <= 100:
    print(f"    ✓ Total Clifford terms manageable ({total_terms})")
else:
    print(f"    ⚠ Total Clifford terms may be challenging ({total_terms})")
