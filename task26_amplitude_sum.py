import sys
import re
import numpy as np
import os
os.environ["JAX_PLATFORMS"] = "cpu"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PRX_ROOT, 'tsim', 'src'))

import tsim
import pyzx_param as zx
from tsim.core.graph import evaluate_graph
from tsim_cutting import find_stab_cutting

# Build T-gate circuit
D3_CLIFFORD = os.path.join(_PRX_ROOT, "gidney-circuits", "circuits", "for_perfectionist_decoding", "c=inject[unitary]+cultivate,p=0.001,noise=uniform,g=css,q=15,b=Y,r=4,d1=3.stim"
with open(D3_CLIFFORD) as f:
    content = f.read()
t_content = re.sub(r'\bS_DAG\b', 'T_DAG', content)
t_content = re.sub(r'\bS\b', 'T', t_content)

tsim_circuit = tsim.Circuit(t_content)
half_graph = tsim_circuit.get_graph()

# Decompose into Clifford terms
clifford_terms = find_stab_cutting(half_graph, max_cut_iterations=20)
print(f"Got {len(clifford_terms)} Clifford terms")

# Fix IO and evaluate each term
amplitudes = []
errors = []
for i, term in enumerate(clifford_terms):
    try:
        term.auto_detect_io()
        tensor = zx.tensorfy(term)
        amplitudes.append(tensor)
        if i < 3:
            print(f"Term {i}: tensor shape={tensor.shape}, dtype={tensor.dtype}")
            print(f"  sample values: {tensor.flat[:4]}")
    except Exception as e:
        errors.append((i, str(e)))
        if i < 3:
            import traceback
_PRX_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
            traceback.print_exc()

print(f"\nSuccessfully evaluated: {len(amplitudes)} / {len(clifford_terms)} terms")
if errors:
    print(f"Errors ({len(errors)} total): {errors[:5]}")

# Check tensor shapes
shapes = [a.shape for a in amplitudes]
unique_shapes = set(str(s) for s in shapes)
print(f"Unique shapes: {unique_shapes}")

# Sum all amplitudes (the superposition)
if len(unique_shapes) == 1:
    total = sum(amplitudes)
    print(f"\nTotal amplitude tensor shape: {total.shape}")
    probs = np.abs(total)**2
    print(f"Sum of probabilities: {probs.sum():.6f}  (should be ~1.0)")
    print(f"Nonzero entries: {np.count_nonzero(probs > 1e-8)}")
    print(f"Max probability: {probs.max():.6f}")
else:
    print("Shapes differ — need to understand tensor index structure before summing")
    print("All shapes:", shapes[:10])
