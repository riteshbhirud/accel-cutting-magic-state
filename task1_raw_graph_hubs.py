"""Task 1e: Find GHZ hub spiders in the RAW (unreduced) circuit ZX.

From Appendix C: The two Z-spiders to cut are phase-0 hubs connecting all
T-spider wires. They're visible in the raw ZX but get fused during full_reduce.

Strategy: In the raw graph, find phase-0 Z-spiders that have many
T-phase (±1/4) neighbors. These are the GHZ hubs.
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

# Load circuit
CIRCUIT_PATH = Path("/Users/ritesh/Downloads/prx/gidney-circuits/circuits/"
                    "for_perfectionist_decoding/"
                    "c=inject[unitary]+cultivate,p=0.001,noise=uniform,"
                    "g=css,q=42,b=Y,r=10,d1=5.stim")

circuit_str = CIRCUIT_PATH.read_text()
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

# ============================================================================
# Analysis of raw graph
# ============================================================================
print(f"Raw circuit ZX: {len(list(raw_graph.vertices()))} verts, "
      f"{len(list(raw_graph.edges()))} edges, T-count={tcount(raw_graph)}")

# Count vertex types
type_counts = defaultdict(int)
for v in raw_graph.vertices():
    vtype = raw_graph.type(v)
    type_counts[vtype] += 1

print(f"\nVertex types:")
for vtype, count in sorted(type_counts.items()):
    type_name = {0: 'Boundary', 1: 'Z-spider', 2: 'X-spider'}.get(vtype, f'Unknown({vtype})')
    print(f"  {type_name}: {count}")

# ============================================================================
# Find phase-0 Z-spiders with T-phase neighbors (hub candidates)
# ============================================================================
print(f"\n{'='*60}")
print("Hub candidates: Phase-0 Z-spiders with many T-phase neighbors")
print(f"{'='*60}")

hub_candidates = []

for v in raw_graph.vertices():
    vtype = raw_graph.type(v)
    phase = raw_graph.phase(v)

    if vtype != 1:  # Z-spider only
        continue
    if phase != 0:  # Phase 0 only
        continue

    # Count T-phase neighbors
    neighbors = list(raw_graph.neighbors(v))
    n_total = len(neighbors)
    n_t_neighbors = 0
    n_z_neighbors = 0
    n_x_neighbors = 0
    n_boundary = 0

    for nb in neighbors:
        nb_type = raw_graph.type(nb)
        nb_phase = raw_graph.phase(nb)
        if nb_type == 0:
            n_boundary += 1
        elif nb_type == 1:
            n_z_neighbors += 1
            if is_t_like(nb_phase):
                n_t_neighbors += 1
        elif nb_type == 2:
            n_x_neighbors += 1
            if is_t_like(nb_phase):
                n_t_neighbors += 1

    if n_t_neighbors >= 2:  # At least 2 T-phase neighbors
        hub_candidates.append({
            'v': v, 'n_total': n_total,
            'n_t': n_t_neighbors, 'n_z': n_z_neighbors,
            'n_x': n_x_neighbors, 'n_boundary': n_boundary,
            'qubit': raw_graph.qubit(v), 'row': raw_graph.row(v),
        })

# Sort by number of T-phase neighbors (most first)
hub_candidates.sort(key=lambda h: -h['n_t'])

print(f"\nFound {len(hub_candidates)} candidates with ≥2 T-phase neighbors")
print(f"\nTop 30 (sorted by # T-phase neighbors):")
print(f"{'Vertex':>8} {'#Total':>6} {'#T_nb':>5} {'#Z':>4} {'#X':>4} {'#Bnd':>5} "
      f"{'Qubit':>6} {'Row':>6}")
print("-" * 60)

for h in hub_candidates[:30]:
    print(f"{h['v']:>8} {h['n_total']:>6} {h['n_t']:>5} {h['n_z']:>4} "
          f"{h['n_x']:>4} {h['n_boundary']:>5} {h['qubit']:>6.0f} {h['row']:>6.0f}")

# ============================================================================
# Find the specific hub pattern: phase-0 Z connected to hub on one side
# and multiple T-spiders on the other
# ============================================================================
print(f"\n{'='*60}")
print("Looking for hub PAIRS (two phase-0 Z connected to each other + T-spiders)")
print(f"{'='*60}")

# For each pair of hub candidates that are connected to each other
hub_pairs = []
for i, h1 in enumerate(hub_candidates):
    for h2 in hub_candidates[i+1:]:
        v1, v2 = h1['v'], h2['v']
        # Check if connected
        if v2 in set(raw_graph.neighbors(v1)):
            hub_pairs.append((h1, h2))

print(f"Found {len(hub_pairs)} connected hub candidate pairs")
for h1, h2 in hub_pairs[:20]:
    print(f"  ({h1['v']}, {h2['v']}): "
          f"T_nb=({h1['n_t']},{h2['n_t']}), "
          f"total_nb=({h1['n_total']},{h2['n_total']}), "
          f"qubits=({h1['qubit']:.0f},{h2['qubit']:.0f}), "
          f"rows=({h1['row']:.0f},{h2['row']:.0f})")

# ============================================================================
# Try cutting hub pairs in the raw graph, then reduce
# ============================================================================
if hub_pairs:
    print(f"\n{'='*60}")
    print("Testing: Cut hub pairs in RAW graph, then full_reduce")
    print(f"{'='*60}")

    # Test top pairs (sorted by total T-neighbors)
    hub_pairs.sort(key=lambda p: -(p[0]['n_t'] + p[1]['n_t']))

    for h1, h2 in hub_pairs[:10]:
        v1, v2 = h1['v'], h2['v']
        g_test = deepcopy(raw_graph)

        # Check cuttability
        can1 = can_cut(g_test, v1)
        can2 = can_cut(g_test, v2)

        if not can1 or not can2:
            print(f"  Pair ({v1},{v2}): v1_cuttable={can1}, v2_cuttable={can2}, SKIP")
            continue

        # Cut v1 first
        g_l1, g_r1 = cut_spider(g_test, v1)

        # For each branch, check if v2 exists and cut it
        final_terms = []
        for branch_name, branch in [("L", g_l1), ("R", g_r1)]:
            if v2 not in set(branch.vertices()):
                # v2 doesn't exist (shouldn't happen, but check)
                zx.full_reduce(branch, paramSafe=True)
                if not branch.scalar.is_zero:
                    final_terms.append(branch)
                continue

            if not can_cut(branch, v2):
                zx.full_reduce(branch, paramSafe=True)
                if not branch.scalar.is_zero:
                    final_terms.append(branch)
                continue

            b_l, b_r = cut_spider(branch, v2)

            zx.full_reduce(b_l, paramSafe=True)
            zx.full_reduce(b_r, paramSafe=True)

            if not b_l.scalar.is_zero:
                final_terms.append(b_l)
            if not b_r.scalar.is_zero:
                final_terms.append(b_r)

        tc_terms = [tcount(t) for t in final_terms]
        n_clifford = sum(1 for tc in tc_terms if tc == 0)
        n_zero_scalar = 4 - len(final_terms)  # Terms that became zero
        total_tc = sum(tc_terms)

        print(f"  Pair ({v1},{v2}): T_nb=({h1['n_t']},{h2['n_t']}), "
              f"{len(final_terms)} terms (TCs={tc_terms}), "
              f"{n_clifford} Clifford, {n_zero_scalar} zero-scalar, total_T={total_tc}")

        if n_clifford == len(final_terms) and len(final_terms) > 0:
            print(f"  *** ALL CLIFFORD! ***")
            for j, t in enumerate(final_terms):
                nv = len(list(t.vertices()))
                print(f"    Term {j}: {nv} verts, T={tcount(t)}")
