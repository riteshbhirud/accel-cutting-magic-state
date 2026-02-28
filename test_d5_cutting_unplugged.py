"""Test: Apply Algorithm 1 (cutting) to the UNPLUGGED d=5 sampling graph component.

The paper's Figures 1-5 show ~8 terms for d=5 at the circuit level.
The external repo plugs outputs THEN cuts (fails for d=5: monolithic T-count=106).
Let's try cutting the unplugged component and see if inter-cut full_reduce
is more effective when boundary vertices are present.
"""
import sys, time
from pathlib import Path
from copy import deepcopy
from fractions import Fraction

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))

import re
import tsim
import pyzx_param as zx
from tsim.core.graph import prepare_graph, connected_components, get_params
from stab_rank_cut import tcount, decompose, find_best_cut, cut_spider, can_cut
import os
_PRX_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

# Load d=5 circuit with T gates
CIRCUIT_PATH = Path(os.path.join(_PRX_ROOT, "gidney-circuits", "circuits", ""
                    "for_perfectionist_decoding/"
                    "c=inject[unitary]+cultivate,p=0.001,noise=uniform,"
                    "g=css,q=42,b=Y,r=10,d1=5.stim")

circuit_str = CIRCUIT_PATH.read_text()

def replace_s_with_t(s):
    s = re.sub(r'^(\s*)S_DAG(\s)', r'\1T_DAG\2', s, flags=re.MULTILINE)
    return re.sub(r'^(\s*)S(\s)', r'\1T\2', s, flags=re.MULTILINE)

circuit_str_t = replace_s_with_t(circuit_str)
circuit = tsim.Circuit(circuit_str_t)
prepared = prepare_graph(circuit, sample_detectors=True)

# Get the non-trivial component
components = connected_components(prepared.graph)
components_sorted = sorted(components, key=lambda c: len(c.output_indices))
big_cc = components_sorted[-1]
g = big_cc.graph

print(f"Unplugged component: {len(list(g.vertices()))} verts, "
      f"{len(g.outputs())} outputs, T-count={tcount(g)}")
print(f"Boundary vertices: {len(list(g.inputs()))} inputs, {len(list(g.outputs()))} outputs")

# Try Algorithm 1 on the UNPLUGGED graph
# with higher max_iterations and debug to see what happens
print(f"\n{'='*60}")
print("Attempting cutting decomposition on UNPLUGGED component")
print(f"{'='*60}")

g_copy = deepcopy(g)

# Manual implementation of Algorithm 1 with detailed tracking
zx.full_reduce(g_copy, paramSafe=True)
tc_init = tcount(g_copy)
print(f"After initial full_reduce: T-count={tc_init}, verts={len(list(g_copy.vertices()))}")

terms = [g_copy]
clifford_terms = []

MAX_ITERS = 20  # enough to see the pattern

for iteration in range(MAX_ITERS):
    new_terms = []
    zero_scalar_count = 0
    became_clifford = 0

    for term in terms:
        zx.full_reduce(term, paramSafe=True)

        if term.scalar.is_zero:
            zero_scalar_count += 1
            continue

        tc = tcount(term)
        if tc == 0:
            clifford_terms.append(term)
            became_clifford += 1
            continue

        cut_v = find_best_cut(term, strategy="fewest_neighbors")
        if cut_v is None:
            # No cuttable vertex - try BSS or keep
            clifford_terms.append(term)
            became_clifford += 1
            continue

        g_left, g_right = cut_spider(term, cut_v)
        new_terms.extend([g_left, g_right])

    # Track T-count distribution of surviving terms
    tc_counts = [tcount(t) for t in new_terms]
    tc_min = min(tc_counts) if tc_counts else 0
    tc_max = max(tc_counts) if tc_counts else 0
    tc_avg = sum(tc_counts)/len(tc_counts) if tc_counts else 0

    print(f"Iter {iteration+1:2d}: {len(new_terms):5d} non-Clifford, "
          f"{became_clifford} became Clifford, {zero_scalar_count} zero-scalar, "
          f"T-count range [{tc_min}-{tc_max}] avg={tc_avg:.1f}")

    if not new_terms:
        break

    terms = new_terms

    # Safety: stop if terms explode
    if len(terms) > 2048:
        print(f"  STOPPING: {len(terms)} terms, too many")
        break

print(f"\nFinal: {len(clifford_terms)} Clifford terms, {len(terms)} remaining non-Clifford")
