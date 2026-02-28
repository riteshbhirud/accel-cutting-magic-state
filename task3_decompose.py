"""Task 3: Use the actual ZX-diagram spider cutting to decompose d=3 and d=5.

Uses the stab_rank_cut.decompose() function which iteratively cuts T-spiders
in the ZX diagram, producing Clifford terms.
"""
import sys, os, re
from pathlib import Path
from copy import deepcopy
import numpy as np

os.environ["JAX_PLATFORMS"] = "cpu"

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))

import tsim
import pyzx_param as zx
from stab_rank_cut import tcount, can_cut, find_cuttable_t, find_best_cut
from stab_rank_cut import cut_spider, decompose

# ============================================================================
# Load d=3 circuit
# ============================================================================
D3_PATH = Path(os.path.join(_PRX_ROOT, "gidney-circuits", "circuits", ""
               "for_perfectionist_decoding/"
               "c=inject[unitary]+cultivate,p=0.001,noise=uniform,"
               "g=css,q=15,b=Y,r=4,d1=3.stim")

circuit_str = D3_PATH.read_text()
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


# ============================================================================
# d=3: Circuit-level ZX analysis
# ============================================================================
print("="*60)
print("d=3: Circuit-level ZX analysis")
print("="*60)

d3_t_text = replace_s_with_t('\n'.join(noiseless))
d3_circuit = tsim.Circuit(d3_t_text)
d3_g = d3_circuit.get_graph()
d3_g2 = deepcopy(d3_g)
zx.full_reduce(d3_g2, paramSafe=True)

tc_raw = tcount(d3_g)
tc_red = tcount(d3_g2)
n_verts = len(list(d3_g2.vertices()))

print(f"  T-count: raw={tc_raw}, reduced={tc_red}")
print(f"  Vertices after reduce: {n_verts}")

# Check which T-spiders can be cut
n_cuttable = 0
n_boundary = 0
for v in d3_g2.vertices():
    if d3_g2.type(v) not in [1, 2]:
        continue
    phase = d3_g2.phase(v)
    from stab_rank_cut import is_t_like
    if not is_t_like(phase):
        continue
    if can_cut(d3_g2, v):
        n_cuttable += 1
    else:
        n_boundary += 1
        # Show why not cuttable
        boundary_nbs = [n for n in d3_g2.neighbors(v) if d3_g2.type(n) == 0]
        print(f"    T-spider {v}: boundary-adj ({len(boundary_nbs)} boundary neighbors), "
              f"phase={d3_g2.phase(v)}, degree={len(list(d3_g2.neighbors(v)))}")

print(f"  Cuttable T-spiders: {n_cuttable}")
print(f"  Boundary-adjacent T-spiders: {n_boundary}")


# ============================================================================
# d=3: Decompose with spider cutting
# ============================================================================
print(f"\n{'='*60}")
print("d=3: Decomposing circuit-level ZX with spider cutting")
print(f"{'='*60}")

d3_g3 = deepcopy(d3_g)
terms = decompose(d3_g3, max_iterations=50, use_bss_fallback=True,
                  debug=True, cut_strategy="fewest_neighbors")

print(f"\nResult: {len(terms)} Clifford terms")
for i, t in enumerate(terms):
    tc = tcount(t)
    is_zero = t.scalar.is_zero
    nv = len(list(t.vertices()))
    print(f"  Term {i}: T-count={tc}, zero={is_zero}, verts={nv}")


# ============================================================================
# d=3: Sampling graph analysis
# ============================================================================
print(f"\n{'='*60}")
print("d=3: Sampling graph analysis")
print(f"{'='*60}")

from tsim.core.graph import prepare_graph, connected_components
import os
_PRX_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

d3_circuit2 = tsim.Circuit(d3_t_text)
prepared = prepare_graph(d3_circuit2, sample_detectors=True)

print(f"  Sampling graph vertices: {len(list(prepared.graph.vertices()))}")
print(f"  Sampling graph T-count: {tcount(prepared.graph)}")

components = connected_components(prepared.graph)
print(f"  Connected components: {len(components)}")

for i, cc in enumerate(components):
    g_cc = cc.graph
    tc_cc = tcount(g_cc)
    n_out = len(cc.output_indices)
    nv = len(list(g_cc.vertices()))
    print(f"    Component {i}: {nv} verts, {n_out} outputs, T-count={tc_cc}")


# ============================================================================
# d=3: Decompose non-trivial components of sampling graph
# ============================================================================
print(f"\n{'='*60}")
print("d=3: Decomposing sampling graph components")
print(f"{'='*60}")

nontrivial = [cc for cc in components if tcount(cc.graph) > 0]
print(f"Non-trivial components: {len(nontrivial)}")

for i, cc in enumerate(nontrivial):
    g_cc = deepcopy(cc.graph)
    tc = tcount(g_cc)
    print(f"\n  Component {i}: T-count={tc}")

    terms_cc = decompose(g_cc, max_iterations=50, use_bss_fallback=True,
                         debug=True, cut_strategy="fewest_neighbors")
    print(f"  → {len(terms_cc)} Clifford terms")


# ============================================================================
# d=5: Circuit-level ZX analysis
# ============================================================================
print(f"\n{'='*60}")
print("d=5: Circuit-level ZX analysis")
print(f"{'='*60}")

D5_PATH = Path(os.path.join(_PRX_ROOT, "gidney-circuits", "circuits", ""
               "for_perfectionist_decoding/"
               "c=inject[unitary]+cultivate,p=0.001,noise=uniform,"
               "g=css,q=42,b=Y,r=10,d1=5.stim")

d5_str = D5_PATH.read_text()
d5_lines = d5_str.split('\n')
d5_noiseless = []
for line in d5_lines:
    s = line.strip()
    if any(s.startswith(p) for p in ['X_ERROR', 'Z_ERROR', 'DEPOLARIZE1', 'DEPOLARIZE2']):
        continue
    line = re.sub(r'M\([\d.]+\)', 'M', line)
    line = re.sub(r'MX\([\d.]+\)', 'MX', line)
    d5_noiseless.append(line)

d5_t_text = replace_s_with_t('\n'.join(d5_noiseless))
d5_circuit = tsim.Circuit(d5_t_text)
d5_g = d5_circuit.get_graph()
d5_g2 = deepcopy(d5_g)
zx.full_reduce(d5_g2, paramSafe=True)

tc_raw5 = tcount(d5_g)
tc_red5 = tcount(d5_g2)
n_verts5 = len(list(d5_g2.vertices()))

print(f"  T-count: raw={tc_raw5}, reduced={tc_red5}")
print(f"  Vertices after reduce: {n_verts5}")

n_cuttable5 = 0
n_boundary5 = 0
for v in d5_g2.vertices():
    if d5_g2.type(v) not in [1, 2]:
        continue
    if not is_t_like(d5_g2.phase(v)):
        continue
    if can_cut(d5_g2, v):
        n_cuttable5 += 1
    else:
        n_boundary5 += 1

print(f"  Cuttable T-spiders: {n_cuttable5}")
print(f"  Boundary-adjacent T-spiders: {n_boundary5}")


# ============================================================================
# d=5: Decompose circuit-level ZX
# ============================================================================
print(f"\n{'='*60}")
print("d=5: Decomposing circuit-level ZX with spider cutting")
print(f"{'='*60}")

d5_g3 = deepcopy(d5_g)
terms5 = decompose(d5_g3, max_iterations=50, use_bss_fallback=True,
                   debug=True, cut_strategy="fewest_neighbors")

print(f"\nResult: {len(terms5)} Clifford terms")
for i, t in enumerate(terms5[:20]):  # Show first 20
    tc = tcount(t)
    is_zero = t.scalar.is_zero
    nv = len(list(t.vertices()))
    print(f"  Term {i}: T-count={tc}, zero={is_zero}, verts={nv}")

if len(terms5) > 20:
    print(f"  ... ({len(terms5) - 20} more terms)")


# ============================================================================
# d=5: Sampling graph analysis
# ============================================================================
print(f"\n{'='*60}")
print("d=5: Sampling graph analysis")
print(f"{'='*60}")

d5_circuit2 = tsim.Circuit(d5_t_text)
prepared5 = prepare_graph(d5_circuit2, sample_detectors=True)

print(f"  Sampling graph T-count: {tcount(prepared5.graph)}")

components5 = connected_components(prepared5.graph)
print(f"  Connected components: {len(components5)}")

for i, cc in enumerate(components5):
    g_cc = cc.graph
    tc_cc = tcount(g_cc)
    n_out = len(cc.output_indices)
    nv = len(list(g_cc.vertices()))
    print(f"    Component {i}: {nv} verts, {n_out} outputs, T-count={tc_cc}")

# Try decomposing the non-trivial components
nontrivial5 = [cc for cc in components5 if tcount(cc.graph) > 0]
print(f"\n  Non-trivial components: {len(nontrivial5)}")

for i, cc in enumerate(nontrivial5):
    g_cc = deepcopy(cc.graph)
    tc = tcount(g_cc)

    # Check cuttable vs boundary
    nc = sum(1 for v in g_cc.vertices()
             if g_cc.type(v) in [1,2] and is_t_like(g_cc.phase(v)) and can_cut(g_cc, v))
    nb = sum(1 for v in g_cc.vertices()
             if g_cc.type(v) in [1,2] and is_t_like(g_cc.phase(v)) and not can_cut(g_cc, v))

    print(f"\n  Component {i}: T-count={tc}, cuttable={nc}, boundary-adj={nb}")

    if tc <= 30:  # Only try if reasonable
        terms_cc = decompose(g_cc, max_iterations=50, use_bss_fallback=True,
                             debug=True, cut_strategy="fewest_neighbors")
        print(f"  → {len(terms_cc)} Clifford terms")
    else:
        print(f"  → T-count too high for full decomposition, skipping")
