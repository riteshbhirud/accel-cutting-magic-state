import sys
import re
import numpy as np
import os
os.environ["JAX_PLATFORMS"] = "cpu"

sys.path.insert(0, '/Users/ritesh/Downloads/prx/accel-cutting-magic-state')
sys.path.insert(0, '/Users/ritesh/Downloads/prx/tsim/src')

import stim
import tsim
import pyzx_param as zx
from tsim_cutting import find_stab_cutting

D3_CLIFFORD = "/Users/ritesh/Downloads/prx/gidney-circuits/circuits/for_perfectionist_decoding/c=inject[unitary]+cultivate,p=0.001,noise=uniform,g=css,q=15,b=Y,r=4,d1=3.stim"
with open(D3_CLIFFORD) as f:
    content = f.read()

# Step 1: Decompose the NOISELESS T-gate circuit
# Strip noise instructions
noiseless_content = '\n'.join(
    line for line in content.split('\n')
    if not any(x in line for x in ['DEPOLARIZE', 'PAULI_CHANNEL', 'X_ERROR', 'Z_ERROR', 'ELSE_CORRELATED'])
)
# Also fix measurement noise: M(0.001) -> M
noiseless_content = re.sub(r'M\([\d.]+\)', 'M', noiseless_content)
noiseless_content = re.sub(r'MX\([\d.]+\)', 'MX', noiseless_content)

t_noiseless = re.sub(r'\bS_DAG\b', 'T_DAG', noiseless_content)
t_noiseless = re.sub(r'\bS\b', 'T', t_noiseless)

print("Noiseless T-gate circuit (first 20 lines):")
for line in t_noiseless.strip().split('\n')[:20]:
    print(f"  {line}")

# Step 2: Decompose noiseless circuit
try:
    tsim_circuit = tsim.Circuit(t_noiseless)
    half_graph = tsim_circuit.get_graph()
    half_graph.auto_detect_io()
    n_inputs = len(half_graph.inputs())
    n_outputs = len(half_graph.outputs())
    print(f"\nNoiseless half graph: {n_inputs} inputs, {n_outputs} outputs")
    print(f"Vertices: {len(list(half_graph.vertices()))}")
    print(f"T-count: {zx.tcount(half_graph)}")
except Exception as e:
    print(f"\nFailed to build noiseless circuit: {e}")
    import traceback
    traceback.print_exc()
    # Fall back to full circuit with noise
    print("\nFalling back to noisy circuit...")
    t_noisy = re.sub(r'\bS_DAG\b', 'T_DAG', content)
    t_noisy = re.sub(r'\bS\b', 'T', t_noisy)
    tsim_circuit = tsim.Circuit(t_noisy)
    half_graph = tsim_circuit.get_graph()
    half_graph.auto_detect_io()
    n_inputs = len(half_graph.inputs())
    n_outputs = len(half_graph.outputs())
    print(f"Noisy half graph: {n_inputs} inputs, {n_outputs} outputs")
    print(f"T-count: {zx.tcount(half_graph)}")

clifford_terms = find_stab_cutting(half_graph, max_cut_iterations=20)
print(f"Clifford terms: {len(clifford_terms)}")

# Precompute all term tensors
term_tensors = []
for term in clifford_terms:
    term.auto_detect_io()
    t = zx.tensorfy(term, preserve_scalar=True)
    # contract inputs with |0>
    for _ in range(n_inputs):
        t = t[0]
    term_tensors.append(t)

# Sum and check normalization
total = sum(term_tensors)
probs = np.abs(total)**2
norm = probs.sum()
print(f"\nNormalization: {norm:.6e}")
print(f"Output tensor shape: {term_tensors[0].shape}")
print(f"Nonzero entries (threshold norm*1e-10): {np.count_nonzero(probs > norm * 1e-10)}")

# Step 3: Sample one outcome from the distribution
if norm > 0:
    probs_normalized = probs.flatten() / norm
    outcome_idx = np.random.choice(len(probs_normalized), p=probs_normalized)
    n_out = len(term_tensors[0].shape)
    outcome_bits = np.array(list(np.binary_repr(outcome_idx, width=n_out)), dtype=int)
    print(f"\nSampled outcome index: {outcome_idx}")
    print(f"Sampled outcome bits: {outcome_bits}")
    print(f"Outcome probability: {probs_normalized[outcome_idx]:.6f}")

    # Show top 10 most probable outcomes
    top_idx = np.argsort(probs_normalized)[-10:][::-1]
    print(f"\nTop 10 outcomes:")
    for idx in top_idx:
        bits = np.binary_repr(idx, width=n_out)
        print(f"  {bits}: p={probs_normalized[idx]:.6f}")
else:
    print("Norm is zero — cannot sample")
