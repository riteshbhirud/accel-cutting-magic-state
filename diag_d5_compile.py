"""Diagnostic: d=5 compilation — step by step.

Traces the actual compilation path to understand what happens
at each stage. Uses a timeout-safe approach.
"""
import sys, time, re, os
os.environ["JAX_PLATFORMS"] = "cpu"

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

import numpy as np
from copy import deepcopy
import tsim
import stim
import pyzx_param as zx
from tsim.core.graph import prepare_graph, connected_components, get_params
from stab_rank_cut import tcount, decompose as stab_rank_decompose, can_cut, find_best_cut

# ============================================================================
# Load d=5 noiseless circuit
# ============================================================================
CIRCUIT_PATH = "/Users/ritesh/Downloads/prx/gidney-circuits/circuits/" \
    "for_perfectionist_decoding/" \
    "c=inject[unitary]+cultivate,p=0.001,noise=uniform," \
    "g=css,q=42,b=Y,r=10,d1=5.stim"

raw = open(CIRCUIT_PATH).read()
lines = raw.split('\n')
clean = []
for line in lines:
    s = line.strip()
    if any(s.startswith(p) for p in ['X_ERROR', 'Z_ERROR', 'DEPOLARIZE1', 'DEPOLARIZE2']):
        continue
    line = re.sub(r'M\([\d.]+\)', 'M', line)
    line = re.sub(r'MX\([\d.]+\)', 'MX', line)
    clean.append(line)

noiseless_str = '\n'.join(clean)

def replace_s_with_t(c):
    p = str(c) if not isinstance(c, str) else c
    p = re.sub(r'^(\s*)S_DAG(\s)', r'\1T_DAG\2', p, flags=re.MULTILINE)
    return re.sub(r'^(\s*)S(\s)', r'\1T\2', p, flags=re.MULTILINE)

t_str = replace_s_with_t(noiseless_str)
circ = tsim.Circuit(t_str)

print("=" * 60)
print("d=5 Step-by-Step Compilation Diagnostic")
print("=" * 60)

# ============================================================================
# Step 1: Get non-trivial component
# ============================================================================
prepared = prepare_graph(circ, sample_detectors=True)
components = connected_components(prepared.graph)
components_sorted = sorted(components, key=lambda c: len(c.output_indices))

nontrivial = [cc for cc in components_sorted if len(cc.graph.outputs()) > 1]
cc = nontrivial[0]
g = cc.graph
n_outputs = len(g.outputs())
print(f"Non-trivial component: {len(list(g.vertices()))} verts, "
      f"{n_outputs} outputs, T-count={tcount(g)}")
sys.stdout.flush()

# ============================================================================
# Step 2: Properly plug outputs (mirror _compile_component_enum_general)
# ============================================================================
print(f"\nStep 2: Plug all {n_outputs} outputs (using apply_effect)")

# Mirroring the actual code in _plug_outputs:
# Level 0: all "+" (normalization)
g_level0 = deepcopy(g)
effect_0 = "+" * n_outputs
g_level0.apply_effect(effect_0)
g_level0.scalar.add_power(n_outputs)

zx.full_reduce(g_level0, paramSafe=True)
g_level0.normalize()
power2_base = g_level0.scalar.power2
print(f"  Level 0 (normalization): power2_base={power2_base}")
print(f"    Vertices after reduce: {len(list(g_level0.vertices()))}")
print(f"    T-count: {tcount(g_level0)}")
sys.stdout.flush()

# Level N: all "0" (fully plugged), with m-params
g_plugged = deepcopy(g)
output_vertices = list(g_plugged.outputs())
effect_N = "0" * n_outputs
g_plugged.apply_effect(effect_N)

# Set m-param phases on output vertices
m_chars = [f"m{i}" for i in range(n_outputs)]
for i, v in enumerate(output_vertices):
    g_plugged.set_phase(v, m_chars[i])

# No add_power(0) needed (it's 0)

print(f"\n  Level N (fully plugged, with m-params):")
print(f"    Vertices before reduce: {len(list(g_plugged.vertices()))}")
sys.stdout.flush()

t0 = time.time()
zx.full_reduce(g_plugged, paramSafe=True)
t_reduce = time.time() - t0
g_plugged.normalize()
g_plugged.scalar.add_power(-power2_base)

print(f"    full_reduce: {t_reduce:.2f}s")
print(f"    Vertices after reduce: {len(list(g_plugged.vertices()))}")
print(f"    T-count: {tcount(g_plugged)}")
sys.stdout.flush()

# ============================================================================
# Step 3: Check disconnection
# ============================================================================
print(f"\nStep 3: Check graph disconnection")

from tsim_cutting import _find_zx_components

zx_comps = _find_zx_components(g_plugged)
print(f"  Sub-components: {len(zx_comps)}")

for i, comp_verts in enumerate(zx_comps):
    # Extract subgraph to check T-count
    from tsim_cutting import _extract_subgraph
    sub_g = _extract_subgraph(g_plugged, comp_verts, reset_scalar=(i > 0))
    tc = tcount(sub_g)
    n_verts = len(comp_verts)
    print(f"    Sub-component {i}: {n_verts} verts, T-count={tc}")
sys.stdout.flush()

# ============================================================================
# Step 4: Test cutting on each sub-component
# ============================================================================
print(f"\nStep 4: Test cutting decomposition")

from tsim_cutting import find_stab_cutting

for i, comp_verts in enumerate(zx_comps):
    sub_g = _extract_subgraph(g_plugged, comp_verts, reset_scalar=(i > 0))
    tc = tcount(sub_g)

    if tc == 0:
        print(f"  Sub-component {i}: already Clifford (T-count=0)")
        continue

    print(f"  Sub-component {i}: T-count={tc}, attempting cutting+BSS...")
    sys.stdout.flush()

    t0 = time.time()
    try:
        terms = find_stab_cutting(
            sub_g,
            max_cut_iterations=50,
            debug=True,
            cut_strategy="fewest_neighbors",
            use_tsim_bss=True,
        )
        t_cut = time.time() - t0
        n_clifford = sum(1 for t in terms if tcount(t) == 0)
        n_nonclif = sum(1 for t in terms if tcount(t) > 0)
        print(f"    Result: {len(terms)} terms ({n_clifford} Clifford, {n_nonclif} non-Clifford)")
        print(f"    Time: {t_cut:.1f}s")
    except Exception as e:
        print(f"    Error: {e}")
        import traceback
        traceback.print_exc()
    sys.stdout.flush()

# ============================================================================
# Step 5: Also try without m-params (pure noiseless, combo 0)
# ============================================================================
print(f"\n{'=' * 60}")
print("Step 5: Plug without m-params (combo=0, pure noiseless)")
print(f"{'=' * 60}")

g_pure = deepcopy(g)
output_verts = list(g_pure.outputs())
g_pure.apply_effect("0" * n_outputs)
# Don't set m-params — this is a specific combo (all 0)

t0 = time.time()
zx.full_reduce(g_pure, paramSafe=True)
t_reduce = time.time() - t0
g_pure.normalize()

print(f"  Vertices after reduce: {len(list(g_pure.vertices()))}")
print(f"  T-count: {tcount(g_pure)}")
print(f"  full_reduce: {t_reduce:.2f}s")

zx_comps_pure = _find_zx_components(g_pure)
print(f"  Sub-components: {len(zx_comps_pure)}")

for i, comp_verts in enumerate(zx_comps_pure):
    sub_g = _extract_subgraph(g_pure, comp_verts, reset_scalar=(i > 0))
    tc = tcount(sub_g)
    print(f"    Sub-component {i}: {len(comp_verts)} verts, T-count={tc}")

# Test cutting on the pure graph if it has non-zero T
for i, comp_verts in enumerate(zx_comps_pure):
    sub_g = _extract_subgraph(g_pure, comp_verts, reset_scalar=(i > 0))
    tc = tcount(sub_g)
    if tc > 0:
        print(f"\n  Cutting sub-component {i} (T-count={tc})...")
        sys.stdout.flush()
        t0 = time.time()
        try:
            terms = find_stab_cutting(
                sub_g, max_cut_iterations=50, debug=True, use_tsim_bss=True,
            )
            t_cut = time.time() - t0
            print(f"    Result: {len(terms)} Clifford terms in {t_cut:.1f}s")
        except Exception as e:
            print(f"    Error: {e}")
    sys.stdout.flush()

print("\nDone.")
