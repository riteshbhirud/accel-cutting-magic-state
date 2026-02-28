"""Task 1: Find hub spiders via PARTIAL ZX reduction.

The hub structure is destroyed by full_reduce but invisible in the raw graph
(T-spiders connect to hubs via intermediate CNOT spiders).

Strategy: Apply only spider fusion (no T-consuming rules) then search for hubs.
"""
import sys, os, re
from pathlib import Path
from copy import deepcopy
from fractions import Fraction
from collections import defaultdict

os.environ["JAX_PLATFORMS"] = "cpu"

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))

import tsim
import pyzx_param as zx
from stab_rank_cut import tcount, is_t_like, can_cut, cut_spider

# Load circuit
CIRCUIT_PATH = Path("/Users/ritesh/Downloads/prx/gidney-circuits/circuits/"
                    "for_perfectionist_decoding/"
                    "c=inject[unitary]+cultivate,p=0.001,noise=uniform,"
                    "g=css,q=42,b=Y,r=10,d1=5.stim")
circuit_str = CIRCUIT_PATH.read_text()
lines = circuit_str.split('\n')
noiseless = []
for line in lines:
    s = line.strip()
    if any(s.startswith(p) for p in ['X_ERROR', 'Z_ERROR', 'DEPOLARIZE1', 'DEPOLARIZE2']):
        continue
    line = re.sub(r'M\([\d.]+\)', 'M', line)
    line = re.sub(r'MX\([\d.]+\)', 'MX', line)
    noiseless.append(line)

def replace_s_with_t(s):
    s = re.sub(r'^(\s*)S_DAG(\s)', r'\1T_DAG\2', s, flags=re.MULTILINE)
    return re.sub(r'^(\s*)S(\s)', r'\1T\2', s, flags=re.MULTILINE)

noiseless_t = replace_s_with_t('\n'.join(noiseless))
circuit = tsim.Circuit(noiseless_t)
raw_graph = circuit.get_graph()

print(f"Raw graph: {len(list(raw_graph.vertices()))} verts, "
      f"{len(list(raw_graph.edges()))} edges, T-count={tcount(raw_graph)}")

# ============================================================================
# Step 1: Apply spider_simp only (basic fusion, no T-consuming rules)
# ============================================================================
print(f"\n{'='*60}")
print("Applying spider_simp only (spider fusion)")
print(f"{'='*60}")

g = deepcopy(raw_graph)

# pyzx simplification steps:
# spider_simp: fuse adjacent same-type spiders
# id_simp: remove identity spiders
# These are the basic simplifications that don't consume T gates.

# Apply spider fusion iteratively
changed = True
rounds = 0
while changed:
    changed = False
    # Spider fusion: merge adjacent same-type spiders
    n_before = len(list(g.vertices()))
    zx.simplify.spider_simp(g)
    zx.simplify.id_simp(g)
    n_after = len(list(g.vertices()))
    if n_after < n_before:
        changed = True
        rounds += 1

print(f"After {rounds} rounds of spider_simp + id_simp:")
print(f"  Vertices: {len(list(g.vertices()))}, T-count: {tcount(g)}")

# Count vertex types
type_counts = defaultdict(int)
for v in g.vertices():
    vtype = g.type(v)
    type_counts[vtype] += 1

print(f"\nVertex types:")
for vtype, count in sorted(type_counts.items()):
    type_name = {0: 'Boundary', 1: 'Z-spider', 2: 'X-spider'}.get(vtype, f'Unknown({vtype})')
    print(f"  {type_name}: {count}")

# ============================================================================
# Step 2: Find spiders with many T-phase neighbors
# ============================================================================
print(f"\n{'='*60}")
print("Looking for hub candidates (any type, any phase, ≥3 T-neighbors)")
print(f"{'='*60}")

hub_candidates = []

for v in g.vertices():
    vtype = g.type(v)
    phase = g.phase(v)

    if vtype == 0:  # Skip boundary
        continue

    neighbors = list(g.neighbors(v))
    n_t_neighbors = 0
    for nb in neighbors:
        nb_phase = g.phase(nb)
        if g.type(nb) != 0 and is_t_like(nb_phase):
            n_t_neighbors += 1

    if n_t_neighbors >= 3:
        hub_candidates.append({
            'v': v, 'type': vtype, 'phase': phase,
            'degree': len(neighbors), 'n_t_nb': n_t_neighbors,
        })

hub_candidates.sort(key=lambda h: -h['n_t_nb'])

print(f"\nFound {len(hub_candidates)} candidates")
if hub_candidates:
    print(f"\nTop candidates:")
    print(f"{'Vertex':>8} {'Type':>5} {'Phase':>10} {'Degree':>6} {'#T_nb':>5}")
    print("-" * 40)
    for h in hub_candidates[:30]:
        type_name = {1: 'Z', 2: 'X'}[h['type']]
        print(f"{h['v']:>8} {type_name:>5} {str(h['phase']):>10} {h['degree']:>6} {h['n_t_nb']:>5}")

# ============================================================================
# Step 3: Also look for X-spider hubs (MX creates X-spiders)
# ============================================================================
print(f"\n{'='*60}")
print("X-spider hub candidates (MX creates X-type measurement)")
print(f"{'='*60}")

x_hubs = []
for v in g.vertices():
    if g.type(v) != 2:  # X-spider only
        continue

    neighbors = list(g.neighbors(v))
    n_t_neighbors = 0
    n_z_nb = 0
    for nb in neighbors:
        nb_type = g.type(nb)
        nb_phase = g.phase(nb)
        if nb_type == 1:  # Z-spider neighbor
            n_z_nb += 1
            if is_t_like(nb_phase):
                n_t_neighbors += 1

    if len(neighbors) >= 3:
        x_hubs.append({
            'v': v, 'phase': g.phase(v),
            'degree': len(neighbors), 'n_t_nb': n_t_neighbors,
            'n_z_nb': n_z_nb,
        })

x_hubs.sort(key=lambda h: -h['degree'])

print(f"\nFound {len(x_hubs)} X-spiders with degree ≥ 3")
print(f"\nTop by degree:")
for h in x_hubs[:20]:
    print(f"  v={h['v']}: phase={h['phase']}, degree={h['degree']}, "
          f"T_nb={h['n_t_nb']}, Z_nb={h['n_z_nb']}")

# ============================================================================
# Step 4: Try additional simplification steps
# ============================================================================
print(f"\n{'='*60}")
print("After pivot_simp (Clifford pivoting)")
print(f"{'='*60}")

g2 = deepcopy(g)
changed = True
while changed:
    changed = False
    n_before = len(list(g2.vertices()))
    zx.simplify.pivot_simp(g2)
    zx.simplify.spider_simp(g2)
    zx.simplify.id_simp(g2)
    n_after = len(list(g2.vertices()))
    if n_after < n_before:
        changed = True

print(f"Vertices: {len(list(g2.vertices()))}, T-count: {tcount(g2)}")

# Re-search for hubs
hub_candidates2 = []
for v in g2.vertices():
    vtype = g2.type(v)
    phase = g2.phase(v)
    if vtype == 0:
        continue
    neighbors = list(g2.neighbors(v))
    n_t_neighbors = sum(1 for nb in neighbors
                        if g2.type(nb) != 0 and is_t_like(g2.phase(nb)))
    if n_t_neighbors >= 2:
        hub_candidates2.append({
            'v': v, 'type': vtype, 'phase': phase,
            'degree': len(neighbors), 'n_t_nb': n_t_neighbors,
        })

hub_candidates2.sort(key=lambda h: -h['n_t_nb'])

print(f"\nHub candidates with ≥2 T-neighbors: {len(hub_candidates2)}")
for h in hub_candidates2[:20]:
    type_name = {1: 'Z', 2: 'X'}[h['type']]
    print(f"  v={h['v']}: {type_name}, phase={h['phase']}, "
          f"degree={h['degree']}, T_nb={h['n_t_nb']}")


# ============================================================================
# Step 5: Try lcomp_simp (local complementation)
# ============================================================================
print(f"\n{'='*60}")
print("After lcomp_simp + pivot_simp + spider_simp + id_simp")
print(f"{'='*60}")

g3 = deepcopy(g)
changed = True
while changed:
    changed = False
    n_before = len(list(g3.vertices()))
    zx.simplify.lcomp_simp(g3)
    zx.simplify.pivot_simp(g3)
    zx.simplify.spider_simp(g3)
    zx.simplify.id_simp(g3)
    n_after = len(list(g3.vertices()))
    if n_after < n_before:
        changed = True

print(f"Vertices: {len(list(g3.vertices()))}, T-count: {tcount(g3)}")

# Re-search
hub_candidates3 = []
for v in g3.vertices():
    vtype = g3.type(v)
    phase = g3.phase(v)
    if vtype == 0:
        continue
    neighbors = list(g3.neighbors(v))
    n_t_neighbors = sum(1 for nb in neighbors
                        if g3.type(nb) != 0 and is_t_like(g3.phase(nb)))
    if n_t_neighbors >= 3:
        hub_candidates3.append({
            'v': v, 'type': vtype, 'phase': phase,
            'degree': len(neighbors), 'n_t_nb': n_t_neighbors,
        })

hub_candidates3.sort(key=lambda h: -h['n_t_nb'])

print(f"\nHub candidates with ≥3 T-neighbors: {len(hub_candidates3)}")
for h in hub_candidates3[:20]:
    type_name = {1: 'Z', 2: 'X'}[h['type']]
    print(f"  v={h['v']}: {type_name}, phase={h['phase']}, "
          f"degree={h['degree']}, T_nb={h['n_t_nb']}")


# ============================================================================
# Step 6: Direct test - cut highest-T-neighbor spiders in partial-reduced graph
# ============================================================================
if hub_candidates3:
    print(f"\n{'='*60}")
    print("Testing: Cut top hub candidates (partial-reduced graph)")
    print(f"{'='*60}")

    for h in hub_candidates3[:10]:
        v = h['v']
        g_test = deepcopy(g3)

        if not can_cut(g_test, v):
            print(f"  v={v}: NOT cuttable (boundary-adjacent)")
            continue

        g_l, g_r = cut_spider(g_test, v)

        zx.full_reduce(g_l, paramSafe=True)
        zx.full_reduce(g_r, paramSafe=True)

        tc_l = tcount(g_l)
        tc_r = tcount(g_r)

        print(f"  v={v} (T_nb={h['n_t_nb']}, deg={h['degree']}): "
              f"T_left={tc_l}, T_right={tc_r}, "
              f"zero_l={g_l.scalar.is_zero}, zero_r={g_r.scalar.is_zero}")

        if tc_l == 0 and tc_r == 0:
            print(f"  *** BOTH CLIFFORD! ***")
