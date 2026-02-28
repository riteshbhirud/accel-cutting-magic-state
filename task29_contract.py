import sys
import re
import numpy as np
import os
os.environ["JAX_PLATFORMS"] = "cpu"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PRX_ROOT, 'tsim', 'src'))

import tsim
import pyzx_param as zx
from tsim_cutting import find_stab_cutting
_PRX_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

D3_CLIFFORD = os.path.join(_PRX_ROOT, "gidney-circuits", "circuits", "for_perfectionist_decoding", "c=inject[unitary]+cultivate,p=0.001,noise=uniform,g=css,q=15,b=Y,r=4,d1=3.stim"
with open(D3_CLIFFORD) as f:
    content = f.read()
t_content = re.sub(r'\bS_DAG\b', 'T_DAG', content)
t_content = re.sub(r'\bS\b', 'T', t_content)

tsim_circuit = tsim.Circuit(t_content)
half_graph = tsim_circuit.get_graph()
clifford_terms = find_stab_cutting(half_graph, max_cut_iterations=20)
print(f"Got {len(clifford_terms)} Clifford terms")

# For each term: evaluate tensor, contract inputs with |0>
term0 = clifford_terms[0]
term0.auto_detect_io()
inputs = list(term0.inputs())
outputs = list(term0.outputs())
print(f"Inputs: {inputs} (count={len(inputs)})")
print(f"Outputs: {outputs[:5]}... (count={len(outputs)})")

# Get full tensor
tensor = zx.tensorfy(term0, preserve_scalar=True)
print(f"Tensor shape: {tensor.shape}")
print(f"Ndim: {tensor.ndim}, total axes: {tensor.ndim}")

def contract_inputs_with_zero(tensor, n_inputs=2):
    """Select index 0 along each input axis."""
    t = tensor
    for _ in range(n_inputs):
        t = t[0]  # select |0> on first remaining axis
    return t

# Sum all 32 terms FIRST (sum amplitudes), THEN square
print("\n--- Approach 1: contract first 2 axes (inputs first) ---")
total_amplitude = np.zeros((2,)*16, dtype=complex)
for i, term in enumerate(clifford_terms):
    term.auto_detect_io()
    t = zx.tensorfy(term, preserve_scalar=True)
    t_contracted = contract_inputs_with_zero(t, n_inputs=2)
    total_amplitude += t_contracted

probs = np.abs(total_amplitude)**2
print(f"After summing {len(clifford_terms)} terms then squaring:")
print(f"Sum of probabilities: {probs.sum():.6e}  (target: ~1.0)")
print(f"Nonzero entries: {np.count_nonzero(probs > 1e-10)}")
print(f"Max probability: {probs.max():.6e}")

# Alternative: sum |ai|^2 without cross terms
diag_sum = np.zeros((2,)*16, dtype=float)
for term in clifford_terms:
    term.auto_detect_io()
    t = zx.tensorfy(term, preserve_scalar=True)
    t_contracted = contract_inputs_with_zero(t, n_inputs=2)
    diag_sum += np.abs(t_contracted)**2

print(f"\nAlternative (sum |ai|^2 without cross terms):")
print(f"Sum: {diag_sum.sum():.6e}")

# Try contracting last 2 axes instead
print("\n--- Approach 2: contract last 2 axes (inputs last) ---")
total_amplitude2 = np.zeros((2,)*16, dtype=complex)
for term in clifford_terms:
    term.auto_detect_io()
    t = zx.tensorfy(term, preserve_scalar=True)
    t2 = t[..., 0, 0]
    total_amplitude2 += t2

probs2 = np.abs(total_amplitude2)**2
print(f"Sum of probabilities: {probs2.sum():.6e}")
print(f"Nonzero entries: {np.count_nonzero(probs2 > 1e-10)}")

# Debug: check what the original undecomposed graph gives
print("\n--- Debug: original undecomposed graph ---")
half_graph2 = tsim_circuit.get_graph()
half_graph2.auto_detect_io()
print(f"Original half_graph inputs: {len(half_graph2.inputs())}")
print(f"Original half_graph outputs: {len(half_graph2.outputs())}")
t_orig = zx.tensorfy(half_graph2, preserve_scalar=True)
print(f"Original tensor shape: {t_orig.shape}")
t_orig_contracted = contract_inputs_with_zero(t_orig, n_inputs=len(half_graph2.inputs()))
probs_orig = np.abs(t_orig_contracted)**2
print(f"Original sum of probs: {probs_orig.sum():.6e}")
print(f"Original nonzero: {np.count_nonzero(probs_orig > 1e-10)}")
print(f"Original max prob: {probs_orig.max():.6e}")

# Also try contracting last N axes for the original
n_in = len(half_graph2.inputs())
t_orig_last = t_orig
for _ in range(n_in):
    t_orig_last = t_orig_last[..., 0]
probs_orig_last = np.abs(t_orig_last)**2
print(f"\nOriginal (contract last {n_in} axes):")
print(f"Sum of probs: {probs_orig_last.sum():.6e}")
print(f"Nonzero: {np.count_nonzero(probs_orig_last > 1e-10)}")
