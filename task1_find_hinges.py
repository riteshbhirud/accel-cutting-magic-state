"""Task 1b: Find the GHZ hinge spiders for structured cutting.

The paper's Appendix C (equations 26-29) says:
- Each cultivation layer has a double-checking sub-circuit with GHZ structure
- Cutting 2 specific Z-spiders (the GHZ hinges) → 4 terms per layer
- Simplifies to 2 Clifford terms per layer
- d=5 has 2 cultivation layers → 2×2 = 4 Clifford terms (eq 12)

Strategy: Find spiders whose removal maximally disconnects the graph
and separates T-containing regions from T-free regions.
"""
import sys, os
from pathlib import Path
from copy import deepcopy
from fractions import Fraction
from collections import defaultdict

os.environ["JAX_PLATFORMS"] = "cpu"

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))

import re
import pyzx_param as zx
from stab_rank_cut import tcount, is_t_like, can_cut, cut_spider

# ============================================================================
# Load and prepare the circuit-level ZX
# ============================================================================

CIRCUIT_PATH = Path("/Users/ritesh/Downloads/prx/gidney-circuits/circuits/"
                    "for_perfectionist_decoding/"
                    "c=inject[unitary]+cultivate,p=0.001,noise=uniform,"
                    "g=css,q=42,b=Y,r=10,d1=5.stim")

circuit_str = CIRCUIT_PATH.read_text()

# Strip noise
lines = circuit_str.split('\n')
noiseless_lines = []
for line in lines:
    stripped = line.strip()
    if any(stripped.startswith(prefix) for prefix in [
        'X_ERROR', 'Z_ERROR', 'DEPOLARIZE1', 'DEPOLARIZE2',
    ]):
        continue
    line = re.sub(r'M\([\d.]+\)', 'M', line)
    line = re.sub(r'MX\([\d.]+\)', 'MX', line)
    noiseless_lines.append(line)

noiseless_str = '\n'.join(noiseless_lines)

def replace_s_with_t(s):
    s = re.sub(r'^(\s*)S_DAG(\s)', r'\1T_DAG\2', s, flags=re.MULTILINE)
    return re.sub(r'^(\s*)S(\s)', r'\1T\2', s, flags=re.MULTILINE)

noiseless_t_str = replace_s_with_t(noiseless_str)

import tsim
circuit = tsim.Circuit(noiseless_t_str)
raw_graph = circuit.get_graph()

# full_reduce
g = deepcopy(raw_graph)
zx.full_reduce(g, paramSafe=True)

# Get the non-trivial component
from tsim.core.graph import connected_components
comps = connected_components(g)
big_cc = max(comps, key=lambda c: len(list(c.graph.vertices())))
g_main = big_cc.graph

print(f"Main component: {len(list(g_main.vertices()))} verts, T-count={tcount(g_main)}")

# ============================================================================
# Analysis 1: Spider types and phases
# ============================================================================
print(f"\n{'='*60}")
print("Spider analysis")
print(f"{'='*60}")

z_spiders = []
x_spiders = []
t_spiders = []
clifford_spiders = []

for v in g_main.vertices():
    vtype = g_main.type(v)
    phase = g_main.phase(v)

    if vtype == 1:  # Z-spider
        z_spiders.append(v)
        if is_t_like(phase):
            t_spiders.append(v)
        else:
            clifford_spiders.append(v)
    elif vtype == 2:  # X-spider
        x_spiders.append(v)
        if is_t_like(phase):
            t_spiders.append(v)
        else:
            clifford_spiders.append(v)

print(f"Z-spiders: {len(z_spiders)}")
print(f"X-spiders: {len(x_spiders)}")
print(f"T-spiders (non-Clifford): {len(t_spiders)}")
print(f"Clifford spiders: {len(clifford_spiders)}")

# ============================================================================
# Analysis 2: Neighbor counts and edge types of T-spiders
# ============================================================================
print(f"\n{'='*60}")
print("T-spider connectivity")
print(f"{'='*60}")

neighbor_hist = defaultdict(int)
for v in t_spiders:
    n = len(list(g_main.neighbors(v)))
    neighbor_hist[n] += 1

print("Neighbor count distribution of T-spiders:")
for n_nb in sorted(neighbor_hist):
    print(f"  {n_nb} neighbors: {neighbor_hist[n_nb]} spiders")

# ============================================================================
# Analysis 3: Find bridge-like T-spiders (articulation points)
# ============================================================================
print(f"\n{'='*60}")
print("Bridge analysis: T-spiders whose removal disconnects the graph")
print(f"{'='*60}")

# BFS-based articulation point detection
def count_components_without(g, v_remove):
    """Count connected components after removing vertex v."""
    verts = set(g.vertices()) - {v_remove}
    visited = set()
    n_comps = 0
    for start in verts:
        if start in visited:
            continue
        n_comps += 1
        queue = [start]
        while queue:
            curr = queue.pop()
            if curr in visited:
                continue
            visited.add(curr)
            for nb in g.neighbors(curr):
                if nb != v_remove and nb not in visited:
                    queue.append(nb)
    return n_comps

# Check all T-spiders
bridge_t_spiders = []
for v in t_spiders:
    nc = count_components_without(g_main, v)
    if nc > 1:
        bridge_t_spiders.append((v, nc))
        nb_count = len(list(g_main.neighbors(v)))
        print(f"  T-spider {v}: removal creates {nc} components, "
              f"phase={g_main.phase(v)}, type={'Z' if g_main.type(v)==1 else 'X'}, "
              f"neighbors={nb_count}")

if not bridge_t_spiders:
    print("  No T-spiders are articulation points")

# ============================================================================
# Analysis 4: Find ALL Clifford articulation points
# ============================================================================
print(f"\n{'='*60}")
print("Bridge analysis: Clifford spiders whose removal disconnects")
print(f"{'='*60}")

bridge_clifford = []
for v in clifford_spiders:
    nc = count_components_without(g_main, v)
    if nc > 1:
        phase = g_main.phase(v)
        vtype = 'Z' if g_main.type(v) == 1 else 'X'
        nb_count = len(list(g_main.neighbors(v)))
        bridge_clifford.append((v, nc, phase, vtype, nb_count))

print(f"Found {len(bridge_clifford)} Clifford articulation points")
for v, nc, phase, vtype, nb in bridge_clifford[:20]:
    print(f"  Spider {v}: {nc} components, phase={phase}, type={vtype}, neighbors={nb}")

# ============================================================================
# Analysis 5: Try cutting articulation points first
# ============================================================================
if bridge_clifford or bridge_t_spiders:
    print(f"\n{'='*60}")
    print("Testing: Cut bridge spiders then reduce")
    print(f"{'='*60}")

    # Sort bridge spiders by number of components created (most first)
    all_bridges = [(v, nc) for v, nc, *_ in bridge_clifford]
    all_bridges += bridge_t_spiders
    all_bridges.sort(key=lambda x: -x[1])

    for v, nc in all_bridges[:5]:
        g_test = deepcopy(g_main)
        phase = g_test.phase(v)
        vtype = 'Z' if g_test.type(v) == 1 else 'X'

        if not can_cut(g_test, v):
            print(f"  Spider {v} ({vtype}, phase={phase}): not cuttable (boundary-adjacent)")
            continue

        g_left, g_right = cut_spider(g_test, v)

        # Reduce both
        zx.full_reduce(g_left, paramSafe=True)
        zx.full_reduce(g_right, paramSafe=True)

        tc_l = tcount(g_left)
        tc_r = tcount(g_right)
        nv_l = len(list(g_left.vertices()))
        nv_r = len(list(g_right.vertices()))

        print(f"  Cut spider {v} ({vtype}, phase={phase}):")
        print(f"    Left:  {nv_l} verts, T-count={tc_l}, zero_scalar={g_left.scalar.is_zero}")
        print(f"    Right: {nv_r} verts, T-count={tc_r}, zero_scalar={g_right.scalar.is_zero}")

# ============================================================================
# Analysis 6: Try cutting ALL cuttable Clifford spiders
# ============================================================================
print(f"\n{'='*60}")
print("Testing: Single cuts of Clifford spiders (phase=0 or π)")
print(f"{'='*60}")

clifford_cut_results = []
for v in clifford_spiders:
    if not can_cut(g_main, v):
        continue

    phase = g_main.phase(v)
    vtype = g_main.type(v)

    g_test = deepcopy(g_main)
    g_left, g_right = cut_spider(g_test, v)

    zx.full_reduce(g_left, paramSafe=True)
    zx.full_reduce(g_right, paramSafe=True)

    tc_l = tcount(g_left)
    tc_r = tcount(g_right)

    # Record cases where one half becomes Clifford
    if tc_l == 0 or tc_r == 0:
        clifford_cut_results.append((v, phase, vtype, tc_l, tc_r))

print(f"Clifford spiders where cutting produces a Clifford half: {len(clifford_cut_results)}")
for v, phase, vtype, tc_l, tc_r in clifford_cut_results[:20]:
    vt = 'Z' if vtype == 1 else 'X'
    print(f"  Spider {v} ({vt}, phase={phase}): left T={tc_l}, right T={tc_r}")

# ============================================================================
# Analysis 7: Try cutting pairs of T-spiders
# ============================================================================
print(f"\n{'='*60}")
print("Testing: Pairs of T-spider cuts (looking for 4 Clifford terms)")
print(f"{'='*60}")

# For efficiency, first try cutting the T-spiders with fewest neighbors
t_by_neighbors = sorted(t_spiders, key=lambda v: len(list(g_main.neighbors(v))))

# Try first 10 T-spiders paired with each other
best_pair = None
best_total_tc = float('inf')
tested = 0

for i, v1 in enumerate(t_by_neighbors[:15]):
    for v2 in t_by_neighbors[i+1:15]:
        g_test = deepcopy(g_main)

        # Cut v1 first
        if not can_cut(g_test, v1):
            continue
        g_l1, g_r1 = cut_spider(g_test, v1)

        # Cut v2 in each branch
        terms = []
        for branch in [g_l1, g_r1]:
            zx.full_reduce(branch, paramSafe=True)
            if branch.scalar.is_zero:
                continue

            # Check if v2 still exists in this branch
            if v2 not in set(branch.vertices()):
                terms.append(branch)
                continue

            if not can_cut(branch, v2):
                terms.append(branch)
                continue

            b_l, b_r = cut_spider(branch, v2)
            zx.full_reduce(b_l, paramSafe=True)
            zx.full_reduce(b_r, paramSafe=True)

            if not b_l.scalar.is_zero:
                terms.append(b_l)
            if not b_r.scalar.is_zero:
                terms.append(b_r)

        tc_total = sum(tcount(t) for t in terms)
        n_clifford = sum(1 for t in terms if tcount(t) == 0)
        tested += 1

        if n_clifford > 0 or tc_total < best_total_tc:
            best_total_tc = tc_total
            best_pair = (v1, v2)

            if n_clifford >= 2 or (tested % 20 == 0):
                print(f"  Pair ({v1},{v2}): {len(terms)} terms, "
                      f"{n_clifford} Clifford, total T={tc_total}")

        if n_clifford == len(terms):
            print(f"  *** ALL CLIFFORD! Pair ({v1},{v2}): {len(terms)} Clifford terms ***")
            for j, t in enumerate(terms):
                print(f"    Term {j}: {len(list(t.vertices()))} verts, T-count={tcount(t)}")
            break
    else:
        continue
    break

if best_pair:
    print(f"\nBest pair found: {best_pair}, total T-count={best_total_tc}")
print(f"Total pairs tested: {tested}")
