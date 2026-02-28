"""Task 1: Circuit-level cutting decomposition for d=5.

The key insight: the paper's ~8 terms (Figures 1-5) are for the CIRCUIT-level
ZX diagram, NOT the sampling graph. The sampling graph doubles the circuit
(compose with adjoint) creating T-count=106. The circuit itself has T-count=53.

This script:
1. Creates noiseless d=5 circuit with T gates
2. Gets the circuit-level ZX diagram via tsim's get_graph() (NOT sampling graph)
3. Applies full_reduce then cutting decomposition (Algorithm 1)
4. Reports number of Clifford terms and T-count of each

Target: 4 Clifford terms for error-free d=5 (equation 12 of the paper).
"""
import sys, time, os
from pathlib import Path
from copy import deepcopy
from fractions import Fraction

os.environ["JAX_PLATFORMS"] = "cpu"

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))

import re
import pyzx_param as zx
from stab_rank_cut import tcount, decompose, find_best_cut, cut_spider, can_cut

# ============================================================================
# Step 1: Create noiseless d=5 circuit with T gates
# ============================================================================

CIRCUIT_PATH = Path("/Users/ritesh/Downloads/prx/gidney-circuits/circuits/"
                    "for_perfectionist_decoding/"
                    "c=inject[unitary]+cultivate,p=0.001,noise=uniform,"
                    "g=css,q=42,b=Y,r=10,d1=5.stim")

circuit_str = CIRCUIT_PATH.read_text()

# Strip ALL noise instructions via regex
lines = circuit_str.split('\n')
noiseless_lines = []
for line in lines:
    stripped = line.strip()
    if any(stripped.startswith(prefix) for prefix in [
        'X_ERROR', 'Z_ERROR', 'DEPOLARIZE1', 'DEPOLARIZE2',
    ]):
        continue
    # Remove noise parameter from M and MX (e.g., M(0.001) -> M)
    line = re.sub(r'M\([\d.]+\)', 'M', line)
    line = re.sub(r'MX\([\d.]+\)', 'MX', line)
    noiseless_lines.append(line)

noiseless_str = '\n'.join(noiseless_lines)

# Replace S/S_DAG with T/T_DAG (Gidney uses S as Clifford proxy for T)
def replace_s_with_t(s):
    s = re.sub(r'^(\s*)S_DAG(\s)', r'\1T_DAG\2', s, flags=re.MULTILINE)
    return re.sub(r'^(\s*)S(\s)', r'\1T\2', s, flags=re.MULTILINE)

noiseless_t_str = replace_s_with_t(noiseless_str)

# Count T gates in text
t_count = 0
for line in noiseless_t_str.split('\n'):
    stripped = line.strip()
    if stripped.startswith('T_DAG ') or stripped.startswith('T '):
        t_count += len(stripped.split()[1:])
print(f"Noiseless d=5 circuit: {t_count} T/T_DAG qubit-operations")

# ============================================================================
# Step 2: Get circuit-level ZX diagram (NOT sampling graph)
# ============================================================================

import tsim

# tsim.Circuit handles T→S[T] conversion internally via shorthand_to_stim
circuit = tsim.Circuit(noiseless_t_str)

# get_graph() returns the circuit-level ZX (before sampling graph doubling)
raw_graph = circuit.get_graph()

tc_raw = tcount(raw_graph)
n_verts = len(list(raw_graph.vertices()))
n_edges = len(list(raw_graph.edges()))
n_inputs = len(list(raw_graph.inputs()))
n_outputs = len(list(raw_graph.outputs()))

print(f"\nCircuit-level ZX diagram (NOT sampling graph):")
print(f"  Vertices: {n_verts}")
print(f"  Edges: {n_edges}")
print(f"  Inputs: {n_inputs}")
print(f"  Outputs: {n_outputs}")
print(f"  T-count: {tc_raw}")

# Reduce
g = deepcopy(raw_graph)
zx.full_reduce(g, paramSafe=True)
tc_reduced = tcount(g)
n_verts_reduced = len(list(g.vertices()))
n_edges_reduced = len(list(g.edges()))

print(f"\nAfter full_reduce:")
print(f"  Vertices: {n_verts_reduced}")
print(f"  Edges: {n_edges_reduced}")
print(f"  T-count: {tc_reduced}")

# ============================================================================
# Step 3: Graph structure analysis
# ============================================================================
print(f"\n{'='*60}")
print("Graph structure analysis")
print(f"{'='*60}")

# Count boundary-adjacent vs non-boundary-adjacent T-spiders
t_boundary_adj = 0
t_cuttable = 0
for v in g.vertices():
    phase = g.phase(v)
    if phase in (Fraction(1,4), Fraction(3,4), Fraction(5,4), Fraction(7,4)):
        if can_cut(g, v):
            t_cuttable += 1
        else:
            t_boundary_adj += 1

print(f"  T-spiders adjacent to boundary: {t_boundary_adj}")
print(f"  T-spiders cuttable: {t_cuttable}")

# Check connectivity
from tsim.core.graph import connected_components, ConnectedComponent
comps = connected_components(g)
print(f"  Connected components: {len(comps)}")
for i, cc in enumerate(comps):
    tc_cc = tcount(cc.graph)
    nv = len(list(cc.graph.vertices()))
    print(f"    Component {i}: {nv} verts, T-count={tc_cc}, outputs={len(cc.output_indices)}")

# ============================================================================
# Step 4: Apply cutting decomposition (Algorithm 1)
# ============================================================================

print(f"\n{'='*60}")
print("Applying cutting decomposition (Algorithm 1)")
print(f"{'='*60}")

# Manual step-by-step to see what happens
g_work = deepcopy(g)
zx.full_reduce(g_work, paramSafe=True)

terms = [g_work]
clifford_terms = []
MAX_ITERS = 30

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
            # No cuttable vertex - try BSS
            print(f"  Iter {iteration+1}: No cuttable T-gate (T-count={tc}), trying BSS...")
            try:
                bss = zx.simulate.find_stabilizer_decomp(term)
                for b in bss:
                    if not b.scalar.is_zero:
                        clifford_terms.append(b)
                        became_clifford += 1
                print(f"  BSS decomposed T-count={tc} into {len([b for b in bss if not b.scalar.is_zero])} terms")
            except Exception as e:
                print(f"  BSS failed: {e}, keeping as-is")
                clifford_terms.append(term)
                became_clifford += 1
            continue

        g_left, g_right = cut_spider(term, cut_v)
        new_terms.extend([g_left, g_right])

    # Track T-count distribution
    tc_counts = [tcount(t) for t in new_terms]
    tc_min = min(tc_counts) if tc_counts else 0
    tc_max = max(tc_counts) if tc_counts else 0

    print(f"Iter {iteration+1:2d}: {len(new_terms):5d} non-Clifford, "
          f"{became_clifford} Clifford, {zero_scalar_count} zero-scalar, "
          f"T-count range [{tc_min}-{tc_max}]")

    if not new_terms:
        break

    terms = new_terms

    # Safety: stop if terms explode
    if len(terms) > 4096:
        print(f"  STOPPING: {len(terms)} terms, too many")
        break

print(f"\n{'='*60}")
print("RESULTS")
print(f"{'='*60}")
print(f"Clifford terms: {len(clifford_terms)}")
print(f"Remaining non-Clifford: {len(terms) if terms else 0}")

for i, term in enumerate(clifford_terms[:20]):
    tc = tcount(term)
    n_v = len(list(term.vertices()))
    print(f"  Clifford term {i}: {n_v} verts, T-count={tc}")

print(f"\nTarget: 4 Clifford terms (equation 12 of paper)")
