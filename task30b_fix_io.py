import sys
import re
import numpy as np
import os
os.environ["JAX_PLATFORMS"] = "cpu"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PRX_ROOT, 'tsim', 'src'))

import stim
import tsim
import pyzx_param as zx
from tsim_cutting import find_stab_cutting
from pyzx_param.utils import VertexType
_PRX_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

D3_CLIFFORD = os.path.join(_PRX_ROOT, "gidney-circuits", "circuits", "for_perfectionist_decoding", "c=inject[unitary]+cultivate,p=0.001,noise=uniform,g=css,q=15,b=Y,r=4,d1=3.stim"
with open(D3_CLIFFORD) as f:
    content = f.read()

# Strip noise, fix measurement noise
noiseless_content = '\n'.join(
    line for line in content.split('\n')
    if not any(x in line for x in ['DEPOLARIZE', 'PAULI_CHANNEL', 'X_ERROR', 'Z_ERROR', 'ELSE_CORRELATED'])
)
noiseless_content = re.sub(r'M\([\d.]+\)', 'M', noiseless_content)
noiseless_content = re.sub(r'MX\([\d.]+\)', 'MX', noiseless_content)

t_noiseless = re.sub(r'\bS_DAG\b', 'T_DAG', noiseless_content)
t_noiseless = re.sub(r'\bS\b', 'T', t_noiseless)

tsim_circuit = tsim.Circuit(t_noiseless)
half_graph = tsim_circuit.get_graph()
half_graph.auto_detect_io()
n_inputs = len(half_graph.inputs())
n_outputs = len(half_graph.outputs())
print(f"Noiseless half graph: {n_inputs} inputs, {n_outputs} outputs, T-count={zx.tcount(half_graph)}")

clifford_terms = find_stab_cutting(half_graph, max_cut_iterations=20)
print(f"Clifford terms: {len(clifford_terms)}")

def fix_io(graph):
    """Manually set all BOUNDARY vertices as outputs (no inputs for this circuit)."""
    boundaries = [v for v in graph.vertices() if graph.type(v) == VertexType.BOUNDARY]
    graph.set_inputs(())
    graph.set_outputs(tuple(boundaries))
    return len(boundaries)

# Precompute all term tensors
term_tensors = []
for i, term in enumerate(clifford_terms):
    n_boundary = fix_io(term)
    t = zx.tensorfy(term, preserve_scalar=True)
    term_tensors.append(t)
    if i == 0:
        print(f"Term 0: {n_boundary} boundary vertices, tensor shape={t.shape}")

# Sum and check normalization
total = sum(term_tensors)
probs = np.abs(total)**2
norm = probs.sum()
n_out = len(term_tensors[0].shape)
print(f"\nOutput tensor shape: {term_tensors[0].shape} ({n_out} axes)")
print(f"Normalization: {norm:.6e}")
print(f"Nonzero entries (threshold norm*1e-10): {np.count_nonzero(probs > norm * 1e-10)}")

# Sample from normalized distribution
if norm > 0:
    probs_flat = probs.flatten()
    probs_normalized = probs_flat / norm

    # Show top 10 most probable outcomes
    top_idx = np.argsort(probs_normalized)[-10:][::-1]
    print(f"\nTop 10 outcomes (out of {len(probs_flat)}):")
    for idx in top_idx:
        bits = np.binary_repr(idx, width=n_out)
        print(f"  {bits}: p={probs_normalized[idx]:.6f}")

    # Sample 100 outcomes
    samples = np.random.choice(len(probs_flat), size=100, p=probs_normalized)
    unique, counts = np.unique(samples, return_counts=True)
    print(f"\n100 sampled outcomes ({len(unique)} unique):")
    for u, c in sorted(zip(unique, counts), key=lambda x: -x[1])[:10]:
        bits = np.binary_repr(u, width=n_out)
        print(f"  {bits}: count={c}, expected_p={probs_normalized[u]:.4f}")
else:
    print("Norm is zero — cannot sample")

# Also verify: original graph gives same normalization
print(f"\n--- Verification: original graph ---")
half_graph2 = tsim_circuit.get_graph()
half_graph2.auto_detect_io()
t_orig = zx.tensorfy(half_graph2, preserve_scalar=True)
probs_orig = np.abs(t_orig)**2
norm_orig = probs_orig.sum()
print(f"Original norm: {norm_orig:.6e}")
print(f"Ratio decomposed/original: {norm/norm_orig:.6f} (should be ~1.0)")
