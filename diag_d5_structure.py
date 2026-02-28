"""Diagnostic: d=5 plugged graph structure.

Key question: does the d=5 fully-plugged sampling graph disconnect
into sub-components after full_reduce? This determines whether
compilation is tractable.

For d=3: plugged graph disconnects into K=2 sub-components (32+32 graphs).
For d=5: unknown — this script finds out.
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
from stab_rank_cut import tcount, decompose as stab_rank_decompose

# ============================================================================
# Load d=5 circuit
# ============================================================================
CIRCUIT_PATH = os.path.join(_PRX_ROOT, "gidney-circuits", "circuits", \
    "for_perfectionist_decoding/" \
    "c=inject[unitary]+cultivate,p=0.001,noise=uniform," \
    "g=css,q=42,b=Y,r=10,d1=5.stim"

raw = open(CIRCUIT_PATH).read()

# Strip noise
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

# Replace S→T
def replace_s_with_t(c):
    p = str(c) if not isinstance(c, str) else c
    p = re.sub(r'^(\s*)S_DAG(\s)', r'\1T_DAG\2', p, flags=re.MULTILINE)
    return re.sub(r'^(\s*)S(\s)', r'\1T\2', p, flags=re.MULTILINE)

t_str = replace_s_with_t(noiseless_str)
circ = tsim.Circuit(t_str)

print("=" * 60)
print("d=5 Noiseless Circuit Structure Diagnostic")
print("=" * 60)
print(f"  Qubits: {circ.num_qubits}")

# ============================================================================
# Step 1: Build sampling graph and find components
# ============================================================================
print("\nStep 1: Prepare sampling graph")
t0 = time.time()
prepared = prepare_graph(circ, sample_detectors=True)
print(f"  prepare_graph: {time.time()-t0:.2f}s")
print(f"  Outputs: {prepared.num_outputs}, Detectors: {prepared.num_detectors}")

components = connected_components(prepared.graph)
components_sorted = sorted(components, key=lambda c: len(c.output_indices))

n_trivial = sum(1 for cc in components_sorted if len(cc.graph.outputs()) <= 1)
print(f"  Components: {len(components_sorted)} ({n_trivial} trivial)")

# Find the non-trivial component
nontrivial = [cc for cc in components_sorted if len(cc.graph.outputs()) > 1]
if not nontrivial:
    print("  ERROR: No non-trivial component found!")
    sys.exit(1)

cc = nontrivial[0]
g = cc.graph
n_outputs = len(g.outputs())
tc = tcount(g)
print(f"\n  Non-trivial component: {len(list(g.vertices()))} verts, "
      f"{n_outputs} outputs, T-count={tc}")

# ============================================================================
# Step 2: Plug all outputs and check for disconnection
# ============================================================================
print(f"\nStep 2: Plug all {n_outputs} outputs and check disconnection")
sys.stdout.flush()

# Import the internal function that finds ZX components
from tsim_cutting import _find_zx_components, _extract_subgraph

# We need to plug the outputs. In the enumeration pipeline, outputs are
# replaced with 0/1 values. Let's plug all outputs to 0 (first combo).
g_plugged = deepcopy(g)

# Get output vertices
outputs = g_plugged.outputs()
print(f"  Output vertices: {outputs}")
sys.stdout.flush()

# Plug outputs to 0 (identity) — this just removes the output boundary
# and sets the spider phase to 0
for o in outputs:
    neighbors = list(g_plugged.neighbors(o))
    if neighbors:
        # Remove the boundary vertex and its edge
        g_plugged.remove_vertex(o)

# Reduce
print("  Running full_reduce...")
sys.stdout.flush()
t0 = time.time()
zx.full_reduce(g_plugged, paramSafe=True)
t_reduce = time.time() - t0
print(f"  full_reduce: {t_reduce:.2f}s")
print(f"  After reduce: {len(list(g_plugged.vertices()))} verts, T-count={tcount(g_plugged)}")

# Check disconnection
print("\n  Checking for disconnected sub-components...")
sys.stdout.flush()

try:
    subcomps = _find_zx_components(g_plugged)
    print(f"  Sub-components found: {len(subcomps)}")
    for i, sg in enumerate(subcomps):
        tc_sub = tcount(sg)
        n_verts = len(list(sg.vertices()))
        print(f"    Sub-component {i}: {n_verts} verts, T-count={tc_sub}")
except Exception as e:
    print(f"  _find_zx_components failed: {e}")
    # Manual check: use networkx or manual BFS
    import traceback
    traceback.print_exc()

# ============================================================================
# Step 3: Also try the actual compilation path (mirror _compile_component_enum_general)
# ============================================================================
print(f"\n{'=' * 60}")
print("Step 3: Mirror the actual compilation path")
print(f"{'=' * 60}")
sys.stdout.flush()

# The actual code in _compile_component_enum_general does:
# 1. Get graph from component
# 2. For each output combo: plug outputs, full_reduce, find_zx_components, then cut each
# 3. Let's just do it for combo 0 (all outputs=0)

# Actually let's look at how _compile_component_enum_general plugs outputs.
# It uses get_params() and manipulates the graph differently.

# Let's use a fresh copy and follow the code more closely.
g_fresh = deepcopy(cc.graph)
params = get_params(g_fresh)
print(f"  Params: outputs={len(params.output_vertices)}, "
      f"f_params={len(params.f_param_indices)}")

# The actual output plugging in the enum code replaces output boundaries
# with Z-spiders of phase 0 or pi. Let's try the same.
# From tsim_cutting.py _compile_component_enum_general, it calls
# _plug_outputs_and_reduce which is defined somewhere...

# Let's try a simpler approach: trace the compilation itself
print("\n  Attempting direct compilation (may take a while)...")
sys.stdout.flush()

from tsim_cutting import compile_detector_sampler_subcomp_enum_general

# First try with noiseless, small max_cut_iterations to see what happens
try:
    t0 = time.time()
    sampler = compile_detector_sampler_subcomp_enum_general(
        circ, seed=42, max_cut_iterations=50, max_enum_outputs=15
    )
    t_compile = time.time() - t0
    print(f"  Compiled in {t_compile:.1f}s")
    print(f"  {sampler}")

    # Quick sample
    det, obs = sampler.sample(shots=1000, batch_size=1000, separate_observables=True)
    trivial = np.all(det == 0, axis=1)
    kept = int(np.sum(trivial))
    errs = int(np.sum(obs[trivial, 0].astype(int) != 0))
    print(f"  Quick test: PSR={kept/1000:.4f}, errors={errs}/{kept}")

except Exception as e:
    print(f"  Compilation failed: {e}")
    import traceback
import os
_PRX_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    traceback.print_exc()

print("\nDone.")
