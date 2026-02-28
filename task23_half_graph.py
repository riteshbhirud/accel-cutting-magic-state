import sys
import re
import os
os.environ["JAX_PLATFORMS"] = "cpu"

sys.path.insert(0, '/Users/ritesh/Downloads/prx/accel-cutting-magic-state')
sys.path.insert(0, '/Users/ritesh/Downloads/prx/tsim/src')

import stim
import numpy as np
import tsim
from tsim.core.graph import prepare_graph, evaluate_graph
from tsim_cutting import find_stab_cutting

# Step 1: Build noisy T-gate circuit (exactly as run.py does it)
D3_CLIFFORD = "/Users/ritesh/Downloads/prx/gidney-circuits/circuits/for_perfectionist_decoding/c=inject[unitary]+cultivate,p=0.001,noise=uniform,g=css,q=15,b=Y,r=4,d1=3.stim"
with open(D3_CLIFFORD) as f:
    content = f.read()

# The Gidney circuit already has noise baked in at p=0.001
# Just replace S->T to get the actual non-Clifford circuit
t_content = re.sub(r'\bS_DAG\b', 'T_DAG', content)
t_content = re.sub(r'\bS\b', 'T', t_content)

tsim_circuit = tsim.Circuit(t_content)
print(f"tsim circuit: {tsim_circuit.num_qubits} qubits, {tsim_circuit.num_measurements} measurements")

# Step 2: Get the HALF graph (circuit only, not doubled)
print("\ntsim.Circuit methods related to graph:")
print([m for m in dir(tsim_circuit) if 'graph' in m.lower() or 'zx' in m.lower()])

# Try get_graph()
try:
    half_graph = tsim_circuit.get_graph()
    print(f"\nhalf_graph type: {type(half_graph)}")
    # pyzx T-count
    import pyzx_param as zx
    t_count = zx.tcount(half_graph)
    print(f"pyzx T-count: {t_count}")
    print(f"Vertices: {len(list(half_graph.vertices()))}")
    print(f"Inputs: {len(half_graph.inputs())}")
    print(f"Outputs: {len(half_graph.outputs())}")
except Exception as e:
    print(f"get_graph failed: {e}")
    import traceback
    traceback.print_exc()
