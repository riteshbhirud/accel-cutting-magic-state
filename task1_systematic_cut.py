"""Task 1c: Systematically test cutting EVERY cuttable spider.

For each cuttable spider (Clifford or T), cut it, full_reduce both branches,
and report the T-count of each branch. Look for spiders where cutting causes
a massive T-count reduction (the GHZ hinges from Appendix C).

The GHZ hinge spiders should cause T-count to drop by many (not just 1)
because they sit at a topological chokepoint. When cut, the remaining
T-spiders form conjugation pairs that cancel during full_reduce.
"""
import sys, os
from pathlib import Path
from copy import deepcopy
from fractions import Fraction

os.environ["JAX_PLATFORMS"] = "cpu"

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))

import re
import pyzx_param as zx
from stab_rank_cut import tcount, can_cut, cut_spider, is_t_like

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

g = deepcopy(raw_graph)
zx.full_reduce(g, paramSafe=True)

# Get main component
from tsim.core.graph import connected_components
comps = connected_components(g)
big_cc = max(comps, key=lambda c: len(list(c.graph.vertices())))
g_main = big_cc.graph

print(f"Main component: {len(list(g_main.vertices()))} verts, T-count={tcount(g_main)}")

# ============================================================================
# Test ALL cuttable vertices
# ============================================================================
print(f"\n{'='*60}")
print("Cutting every cuttable vertex, reporting T-count of branches")
print(f"{'='*60}")

results = []

for v in sorted(g_main.vertices()):
    if not can_cut(g_main, v):
        continue

    phase = g_main.phase(v)
    vtype = g_main.type(v)
    n_nb = len(list(g_main.neighbors(v)))
    is_t = is_t_like(phase)

    g_test = deepcopy(g_main)
    g_left, g_right = cut_spider(g_test, v)

    zx.full_reduce(g_left, paramSafe=True)
    zx.full_reduce(g_right, paramSafe=True)

    tc_l = tcount(g_left)
    tc_r = tcount(g_right)
    zero_l = g_left.scalar.is_zero
    zero_r = g_right.scalar.is_zero

    # T-count reduction: original is 53
    reduction = 53 - min(tc_l if not zero_l else 999, tc_r if not zero_r else 999)

    results.append({
        'v': v, 'phase': phase, 'type': 'Z' if vtype == 1 else 'X',
        'n_nb': n_nb, 'is_t': is_t,
        'tc_l': tc_l, 'tc_r': tc_r,
        'zero_l': zero_l, 'zero_r': zero_r,
        'reduction': reduction,
    })

# Sort by reduction (most reduction first)
results.sort(key=lambda r: -r['reduction'])

print(f"\nTop 20 spiders by T-count reduction:")
print(f"{'Spider':>8} {'Type':>4} {'Phase':>8} {'#Nb':>4} {'T?':>3} "
      f"{'TC_L':>5} {'TC_R':>5} {'Zero_L':>6} {'Zero_R':>6} {'Reduce':>7}")
print("-" * 75)

for r in results[:20]:
    print(f"{r['v']:>8} {r['type']:>4} {str(r['phase']):>8} {r['n_nb']:>4} "
          f"{'Y' if r['is_t'] else 'N':>3} "
          f"{r['tc_l']:>5} {r['tc_r']:>5} {str(r['zero_l']):>6} {str(r['zero_r']):>6} "
          f"{r['reduction']:>7}")

# ============================================================================
# Show full table for non-T (Clifford) spiders only
# ============================================================================
print(f"\n{'='*60}")
print("All Clifford spider cuts (sorted by reduction)")
print(f"{'='*60}")

clifford_results = [r for r in results if not r['is_t']]
clifford_results.sort(key=lambda r: -r['reduction'])

for r in clifford_results:
    print(f"  Spider {r['v']:>4} ({r['type']}, phase={str(r['phase']):>5}, "
          f"nb={r['n_nb']:>2}): L_T={r['tc_l']:>3}, R_T={r['tc_r']:>3}, "
          f"L_zero={r['zero_l']}, R_zero={r['zero_r']}")

# ============================================================================
# Test sequential cuts of top Clifford reducers
# ============================================================================
print(f"\n{'='*60}")
print("Sequential cutting of top Clifford reducers")
print(f"{'='*60}")

# Pick top 5 Clifford spiders that reduce T-count most
if clifford_results:
    top_cliffords = [r['v'] for r in clifford_results[:10]]
    print(f"Top Clifford candidates: {top_cliffords}")

    # Try sequential pairs
    for i, v1 in enumerate(top_cliffords[:5]):
        for v2 in top_cliffords[i+1:6]:
            g_test = deepcopy(g_main)

            # Cut v1
            if v1 not in set(g_test.vertices()) or not can_cut(g_test, v1):
                continue
            g_l1, g_r1 = cut_spider(g_test, v1)

            # Process all 4 paths: cut v2 in both branches
            final_terms = []
            for branch_name, branch in [("L", g_l1), ("R", g_r1)]:
                zx.full_reduce(branch, paramSafe=True)
                if branch.scalar.is_zero:
                    continue

                if v2 not in set(branch.vertices()) or not can_cut(branch, v2):
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
            total_tc = sum(tc_terms)

            if n_clifford > 0 or total_tc < 200:
                print(f"  Pair ({v1},{v2}): {len(final_terms)} terms, "
                      f"TCs={tc_terms}, {n_clifford} Clifford, total={total_tc}")

# ============================================================================
# Test: What if we cut PAIRS where one is S-phase (1/2)?
# ============================================================================
print(f"\n{'='*60}")
print("Testing: Pairs involving S-phase (1/2) Clifford spiders")
print(f"{'='*60}")

s_phase_spiders = [r['v'] for r in clifford_results if r['phase'] == Fraction(1, 2)]
print(f"S-phase spiders: {s_phase_spiders}")

for v1 in s_phase_spiders:
    for v2 in s_phase_spiders:
        if v1 >= v2:
            continue

        g_test = deepcopy(g_main)

        if not can_cut(g_test, v1):
            continue
        g_l1, g_r1 = cut_spider(g_test, v1)

        final_terms = []
        for branch in [g_l1, g_r1]:
            zx.full_reduce(branch, paramSafe=True)
            if branch.scalar.is_zero:
                continue

            if v2 not in set(branch.vertices()) or not can_cut(branch, v2):
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
        total_tc = sum(tc_terms)

        print(f"  Pair ({v1},{v2}): {len(final_terms)} terms, "
              f"TCs={tc_terms}, {n_clifford} Clifford, total={total_tc}")
