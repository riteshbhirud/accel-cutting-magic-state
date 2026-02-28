"""Quick d=5 analysis: T-count at different stages."""
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
from stab_rank_cut import tcount

# Load and replace S→T
CIRCUIT_PATH = Path("/Users/ritesh/Downloads/prx/gidney-circuits/circuits/"
                    "for_perfectionist_decoding/"
                    "c=inject[unitary]+cultivate,p=0.001,noise=uniform,"
                    "g=css,q=42,b=Y,r=10,d1=5.stim")

circuit_str = CIRCUIT_PATH.read_text()

def replace_s_with_t(s):
    s = re.sub(r'^(\s*)S_DAG(\s)', r'\1T_DAG\2', s, flags=re.MULTILINE)
    return re.sub(r'^(\s*)S(\s)', r'\1T\2', s, flags=re.MULTILINE)

circuit_str_t = replace_s_with_t(circuit_str)

# Count T/T_DAG gates by text parsing (stim can't parse T gates)
t_gates = 0
for line in circuit_str_t.split('\n'):
    stripped = line.strip()
    if stripped.startswith('T_DAG ') or stripped.startswith('T '):
        t_gates += len(stripped.split()[1:])
print(f"T/T_DAG qubit-operations in circuit: {t_gates}")

# Parse with tsim
circuit = tsim.Circuit(circuit_str_t)
prepared = prepare_graph(circuit, sample_detectors=True)

# Find the non-trivial component
components = connected_components(prepared.graph)
components_sorted = sorted(components, key=lambda c: len(c.output_indices))

big_cc = components_sorted[-1]  # largest component
g = big_cc.graph
n_outputs = len(g.outputs())
print(f"\nNon-trivial component:")
print(f"  Vertices: {len(list(g.vertices()))}")
print(f"  Edges: {len(list(g.edges()))}")
print(f"  Outputs: {n_outputs}")

# T-count of unplugged graph
tc_unplugged = tcount(g)
print(f"  T-count (unplugged, unreduced): {tc_unplugged}")

# Reduce unplugged and check
g_reduced = deepcopy(g)
zx.full_reduce(g_reduced, paramSafe=True)
tc_reduced = tcount(g_reduced)
n_verts_reduced = len(list(g_reduced.vertices()))
print(f"  T-count (unplugged, reduced): {tc_reduced} ({n_verts_reduced} verts)")

# Also check: how many S_DAG and S gates were there originally
n_sdag = len(re.findall(r'^\s*S_DAG\s', circuit_str, re.MULTILINE))
n_s = len(re.findall(r'^\s*S\s', circuit_str, re.MULTILINE))
print(f"\nOriginal S_DAG lines: {n_sdag}, S lines: {n_s}")

# Count individual qubit targets in S_DAG/S lines
n_sdag_qubits = 0
n_s_qubits = 0
for line in circuit_str.split('\n'):
    stripped = line.strip()
    if stripped.startswith('S_DAG '):
        n_sdag_qubits += len(stripped.split()[1:])
    elif stripped.startswith('S '):
        n_s_qubits += len(stripped.split()[1:])
print(f"S_DAG qubit-targets: {n_sdag_qubits}, S qubit-targets: {n_s_qubits}")
print(f"Total non-Clifford qubit-operations after replacement: {n_sdag_qubits + n_s_qubits}")

# For comparison: d=3 numbers
print(f"\n=== Comparison ===")
print(f"d=3: 64 verts, 5 outputs, T-count=32 (plugged), DISCONNECTS into 2 sub-components")
print(f"d=5: {len(list(g.vertices()))} verts, {n_outputs} outputs, T-count={tc_unplugged} (unplugged)")
print(f"d=5 reduced: T-count={tc_reduced}")
