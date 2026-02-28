"""Task 1f: Cut boundary-adjacent hub spiders.

Key finding: Spiders 18, 19, 20 in the reduced graph have:
- Phase = 1/2 (S-phase)
- 17 neighbors each
- Are boundary-adjacent (can't cut with standard can_cut)

These are likely the fused GHZ hub spiders from Appendix C eq (26).
The boundary adjacency is because the circuit-level ZX has open qubit wires.

Strategy: Modify cut to allow boundary-adjacent vertices, then test
cutting pairs of these high-degree spiders.
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
from pyzx_param.utils import VertexType, EdgeType
from stab_rank_cut import tcount, is_t_like

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
# Analyze boundary vertices and their neighbors
# ============================================================================
print(f"\n{'='*60}")
print("Boundary vertex analysis")
print(f"{'='*60}")

boundary_verts = []
for v in g_main.vertices():
    if g_main.type(v) == 0:  # Boundary
        boundary_verts.append(v)

print(f"Boundary vertices: {len(boundary_verts)}")
for bv in sorted(boundary_verts):
    nbs = list(g_main.neighbors(bv))
    nb_info = []
    for nb in nbs:
        nb_type = 'Z' if g_main.type(nb) == 1 else ('X' if g_main.type(nb) == 2 else 'B')
        nb_phase = g_main.phase(nb)
        nb_info.append(f"{nb}({nb_type},ph={nb_phase})")
    print(f"  Boundary {bv}: neighbors = {nb_info}")

# ============================================================================
# Identify non-cuttable Clifford spiders (boundary-adjacent)
# ============================================================================
print(f"\n{'='*60}")
print("Non-cuttable Clifford spiders (boundary-adjacent)")
print(f"{'='*60}")

boundary_set = set(boundary_verts)
non_cuttable = []
for v in g_main.vertices():
    if g_main.type(v) not in [1, 2]:
        continue
    phase = g_main.phase(v)
    if is_t_like(phase):
        continue  # T-spider, not Clifford
    # Check if boundary-adjacent
    for nb in g_main.neighbors(v):
        if nb in boundary_set:
            non_cuttable.append(v)
            break

print(f"Non-cuttable Clifford spiders: {len(non_cuttable)}")
for v in sorted(non_cuttable):
    phase = g_main.phase(v)
    n_nb = len(list(g_main.neighbors(v)))
    n_boundary_nb = sum(1 for nb in g_main.neighbors(v) if nb in boundary_set)
    n_t_nb = sum(1 for nb in g_main.neighbors(v)
                 if g_main.type(nb) in [1,2] and is_t_like(g_main.phase(nb)))
    print(f"  Spider {v}: phase={phase}, total_nb={n_nb}, "
          f"boundary_nb={n_boundary_nb}, T_nb={n_t_nb}")

# ============================================================================
# Modified cut_spider that allows boundary-adjacent vertices
# ============================================================================
def cut_spider_boundary_ok(g, v):
    """Like cut_spider but allows boundary-adjacent vertices."""
    v_type = g.type(v)
    if v_type not in [1, 2]:
        raise ValueError(f"Vertex {v} is not a Z or X spider (type={v_type})")

    v_phase = g.phase(v)
    v_params = g.get_params(v)
    neighbors = list(g.neighbors(v))
    n_neighbors = len(neighbors)

    # Record edge types
    edge_types = {}
    for nb in neighbors:
        e = g.edge(v, nb)
        edge_types[nb] = g.edge_type(e)

    base_create_type = 2 if v_type == 1 else 1

    g_left = deepcopy(g)
    g_right = deepcopy(g)

    g_left.remove_vertex(v)
    g_right.remove_vertex(v)

    for nb in neighbors:
        et = edge_types[nb]

        if g.type(nb) == 0:
            # Boundary neighbor - DON'T create new spider, just leave disconnected
            # Actually, we should still apply the cutting identity:
            # create opposite-color spider connected to boundary
            pass

        if et == 2:  # Hadamard edge
            spider_type = v_type
        else:
            spider_type = base_create_type

        new_edge_type = 1

        v_left = g_left.add_vertex(
            spider_type,
            g.qubit(v),
            (g.row(v) + g.row(nb)) / 2,
            0
        )
        g_left.add_edge((v_left, nb), new_edge_type)

        v_right = g_right.add_vertex(
            spider_type,
            g.qubit(v),
            (g.row(v) + g.row(nb)) / 2,
            1
        )
        g_right.add_edge((v_right, nb), new_edge_type)

    g_left.scalar.add_power(-n_neighbors)
    g_right.scalar.add_power(-n_neighbors)

    if isinstance(v_phase, Fraction):
        g_right.scalar.add_phase(v_phase)
    else:
        g_right.scalar.add_phase(Fraction(v_phase).limit_denominator(1000))
    if v_params:
        g_right.scalar.add_phase_vars_pi(set(v_params))

    return g_left, g_right

# ============================================================================
# Test cutting high-degree non-cuttable spiders
# ============================================================================
print(f"\n{'='*60}")
print("Testing: Cut high-degree boundary-adjacent spiders")
print(f"{'='*60}")

# Sort non-cuttable spiders by degree (highest first)
non_cuttable_sorted = sorted(
    non_cuttable,
    key=lambda v: -len(list(g_main.neighbors(v)))
)

for v in non_cuttable_sorted[:10]:
    phase = g_main.phase(v)
    n_nb = len(list(g_main.neighbors(v)))

    g_test = deepcopy(g_main)
    g_left, g_right = cut_spider_boundary_ok(g_test, v)

    zx.full_reduce(g_left, paramSafe=True)
    zx.full_reduce(g_right, paramSafe=True)

    tc_l = tcount(g_left)
    tc_r = tcount(g_right)
    nv_l = len(list(g_left.vertices()))
    nv_r = len(list(g_right.vertices()))
    zero_l = g_left.scalar.is_zero
    zero_r = g_right.scalar.is_zero

    print(f"\n  Cut spider {v} (phase={phase}, {n_nb} neighbors):")
    print(f"    Left:  {nv_l} verts, T={tc_l}, zero={zero_l}")
    print(f"    Right: {nv_r} verts, T={tc_r}, zero={zero_r}")

# ============================================================================
# Test pairs of the highest-degree non-cuttable spiders
# ============================================================================
print(f"\n{'='*60}")
print("Testing: PAIRS of boundary-adjacent spider cuts")
print(f"{'='*60}")

top_nc = non_cuttable_sorted[:10]

for i, v1 in enumerate(top_nc):
    for v2 in top_nc[i+1:]:
        g_test = deepcopy(g_main)

        # Cut v1
        g_l1, g_r1 = cut_spider_boundary_ok(g_test, v1)

        final_terms = []
        for branch in [g_l1, g_r1]:
            zx.full_reduce(branch, paramSafe=True)
            if branch.scalar.is_zero:
                continue

            if v2 not in set(branch.vertices()):
                final_terms.append(branch)
                continue

            try:
                b_l, b_r = cut_spider_boundary_ok(branch, v2)
                zx.full_reduce(b_l, paramSafe=True)
                zx.full_reduce(b_r, paramSafe=True)

                if not b_l.scalar.is_zero:
                    final_terms.append(b_l)
                if not b_r.scalar.is_zero:
                    final_terms.append(b_r)
            except Exception as e:
                final_terms.append(branch)

        tc_terms = [tcount(t) for t in final_terms]
        n_clifford = sum(1 for tc in tc_terms if tc == 0)
        n_zero = 4 - len(final_terms)
        total_tc = sum(tc_terms)

        p1, p2 = g_main.phase(v1), g_main.phase(v2)
        nb1, nb2 = len(list(g_main.neighbors(v1))), len(list(g_main.neighbors(v2)))

        if n_clifford > 0 or total_tc < 150:
            print(f"  ({v1}[ph={p1},nb={nb1}], {v2}[ph={p2},nb={nb2}]): "
                  f"{len(final_terms)} terms, TCs={tc_terms}, "
                  f"{n_clifford} Clifford, {n_zero} zero-scalar")

        if n_clifford == len(final_terms) and len(final_terms) > 0:
            print(f"  *** ALL CLIFFORD! ***")
