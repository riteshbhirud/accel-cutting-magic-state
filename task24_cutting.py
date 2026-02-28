import sys
import re
import os
import time
os.environ["JAX_PLATFORMS"] = "cpu"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PRX_ROOT, 'tsim', 'src'))

import stim
import numpy as np
import tsim
from tsim.core.graph import prepare_graph, evaluate_graph
from tsim_cutting import find_stab_cutting
import pyzx_param as zx

# Rebuild the half graph
D3_CLIFFORD = os.path.join(_PRX_ROOT, "gidney-circuits", "circuits", "for_perfectionist_decoding", "c=inject[unitary]+cultivate,p=0.001,noise=uniform,g=css,q=15,b=Y,r=4,d1=3.stim"
with open(D3_CLIFFORD) as f:
    content = f.read()
t_content = re.sub(r'\bS_DAG\b', 'T_DAG', content)
t_content = re.sub(r'\bS\b', 'T', t_content)
tsim_circuit = tsim.Circuit(t_content)
half_graph = tsim_circuit.get_graph()

print(f"Half graph: {len(list(half_graph.vertices()))} vertices, T-count={zx.tcount(half_graph)}")
print(f"Inputs: {len(half_graph.inputs())}, Outputs: {len(half_graph.outputs())}")

print("\nRunning find_stab_cutting on d=3 half graph...")
print("(This may take 1-5 minutes for d=3)")

start = time.time()
try:
    clifford_terms = find_stab_cutting(half_graph, max_cut_iterations=20, debug=True)
    elapsed = time.time() - start
    print(f"\nCompleted in {elapsed:.1f}s")
    print(f"Number of Clifford terms: {len(clifford_terms)}")
    print(f"Expected: ~120 terms for d=3")

    # Inspect first term
    if len(clifford_terms) > 0:
        term0 = clifford_terms[0]
        print(f"\nFirst term type: {type(term0)}")
        print(f"First term attrs (first 20): {[a for a in dir(term0) if not a.startswith('_')][:20]}")

        # Check T-count of first term
        tc0 = zx.tcount(term0)
        print(f"First term T-count: {tc0} (should be 0 for Clifford)")

        # Check vertices/inputs/outputs
        print(f"First term vertices: {len(list(term0.vertices()))}")
        print(f"First term inputs: {len(term0.inputs())}")
        print(f"First term outputs: {len(term0.outputs())}")

        # Try evaluate_graph on first term
        try:
            val2 = evaluate_graph(term0)
            print(f"\nevaluate_graph result type: {type(val2)}")
            print(f"evaluate_graph result shape: {val2.shape}")
            print(f"evaluate_graph result: {val2}")
        except Exception as e:
            print(f"\nevaluate_graph failed: {e}")
            import traceback
            traceback.print_exc()

        # Try pyzx tensorfy
        try:
            val = zx.tensorfy(term0)
            print(f"\ntensorfy result shape: {val.shape}")
            print(f"tensorfy result (first 4): {val.flat[:4]}")
        except Exception as e:
            print(f"\ntensorfy failed: {e}")

except Exception as e:
    print(f"find_stab_cutting failed: {e}")
    import traceback
_PRX_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    traceback.print_exc()
