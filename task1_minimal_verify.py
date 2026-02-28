"""Task 1g: Verify the Appendix C decomposition on minimal circuits.

Build minimal double-checking circuits matching eq (26) structure.
For n=2,3: verify T†^⊗n · C_check · T^⊗n ∝ I^⊗n + (XS†)^⊗n.

The check circuit has the GHZ structure:
  T†^⊗n → [CX fan-out to ancilla] → [CX fan-in from ancilla] → T^⊗n

After ZX conversion, cut the hub spiders and verify 2 Clifford terms.
Also verify the decomposition numerically via tensor contraction.
"""
import sys, os
from pathlib import Path
from copy import deepcopy
from fractions import Fraction
import numpy as np

os.environ["JAX_PLATFORMS"] = "cpu"

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))

import tsim
import pyzx_param as zx
from stab_rank_cut import tcount

# ============================================================================
# Numerical verification: T†^⊗n · C · T^⊗n = c₁·I + c₂·(XS†)^⊗n
# ============================================================================
print("="*60)
print("Numerical verification of Appendix C decomposition")
print("="*60)

# XS† = X · S† = [[0,1],[1,0]] · [[1,0],[0,-i]] = [[0,-i],[1,0]]
X = np.array([[0, 1], [1, 0]])
S_dag = np.array([[1, 0], [0, -1j]])
XSd = X @ S_dag
print(f"XS† = \n{XSd}")
I2 = np.eye(2)

# I + XS† = [[1,-i],[1,1]] (matches paper)
print(f"I + XS† = \n{I2 + XSd}")

T_gate = np.diag([1, np.exp(1j * np.pi / 4)])
Td_gate = np.diag([1, np.exp(-1j * np.pi / 4)])
H_gate = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
CX_gate = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]]).reshape(2,2,2,2)

def kron_n(matrices):
    """Tensor product of a list of matrices."""
    result = matrices[0]
    for m in matrices[1:]:
        result = np.kron(result, m)
    return result

def check_decomposition(n, C_check_func, label):
    """Check if T†^⊗n · C · T^⊗n ∝ I^⊗n + (XS†)^⊗n.

    C_check_func: function that takes n and returns the 2^n × 2^n matrix
    for the checking circuit on data qubits only (ancillas traced out).
    """
    print(f"\n--- {label} (n={n}) ---")

    Td_n = kron_n([Td_gate] * n)
    T_n = kron_n([T_gate] * n)

    C = C_check_func(n)
    block = Td_n @ C @ T_n

    # Check if block ∝ I + (XS†)^⊗n
    I_n = np.eye(2**n)
    XSd_n = kron_n([XSd] * n)

    # block = c₁ I + c₂ (XS†)^⊗n
    # Solve for c₁, c₂ from two matrix elements
    # block[0,0] = c₁ + c₂ · (XS†)^⊗n[0,0]
    # We can use least squares

    target = I_n.reshape(-1)
    target2 = XSd_n.reshape(-1)
    A = np.column_stack([target, target2])
    b = block.reshape(-1)

    coeffs, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
    c1, c2 = coeffs

    reconstructed = c1 * I_n + c2 * XSd_n
    error = np.linalg.norm(block - reconstructed) / np.linalg.norm(block)

    print(f"  c₁ = {c1:.6f}")
    print(f"  c₂ = {c2:.6f}")
    print(f"  Reconstruction error: {error:.2e}")
    print(f"  |c₁| = {abs(c1):.6f}, |c₂| = {abs(c2):.6f}")
    print(f"  Match (I + XS†)^⊗n decomposition: {error < 1e-10}")

    return error < 1e-10, c1, c2

# ============================================================================
# C_check version 1: Simple CX fan-out/fan-in with ancilla
# ============================================================================
def c_check_cx_fan(n):
    """CX fan-out to ancilla, measure, CX fan-in.

    data qubits: 0..n-1
    ancilla qubit: n
    Circuit: CX(0,n), CX(1,n), ..., CX(n-1,n), M(n), CX(0,n), ..., CX(n-1,n)

    For the operator on data qubits (tracing out ancilla with result 0):
    """
    dim_data = 2**n
    dim_anc = 2

    # Build full unitary on data + ancilla
    dim = dim_data * dim_anc

    # CX from data qubit i to ancilla (qubit n)
    def cx_data_anc(i, n_total):
        """CX with control=qubit i, target=qubit n_total-1 (last qubit = ancilla)."""
        dim_t = 2**n_total
        result = np.zeros((dim_t, dim_t), dtype=complex)
        for state in range(dim_t):
            control_bit = (state >> (n_total - 1 - i)) & 1
            if control_bit == 0:
                result[state, state] = 1
            else:
                # Flip ancilla (last qubit)
                target_mask = 1  # qubit n_total-1 is last bit
                new_state = state ^ target_mask
                result[new_state, state] = 1
        return result

    n_total = n + 1  # data + 1 ancilla

    # Fan-out: CX(0,n), CX(1,n), ..., CX(n-1,n)
    fan_out = np.eye(dim, dtype=complex)
    for i in range(n):
        fan_out = cx_data_anc(i, n_total) @ fan_out

    # Fan-in: same gates
    fan_in = fan_out.copy()  # CX is self-inverse for fan-out/fan-in

    # Full circuit: fan-out, then fan-in
    full = fan_in @ fan_out

    # Trace out ancilla with measurement result 0:
    # C_data = <0_anc| U |0_anc> (unnormalized)
    # ancilla is last qubit (least significant bit)
    C_data = np.zeros((dim_data, dim_data), dtype=complex)
    for i in range(dim_data):
        for j in range(dim_data):
            # ancilla starts in |0⟩ (bit 0), project onto |0⟩
            C_data[i, j] = full[i * 2 + 0, j * 2 + 0]

    return C_data

# ============================================================================
# C_check version 2: GHZ-style controlled-H_XY
# ============================================================================
def c_check_ghz(n):
    """GHZ-style double-checking: prepare GHZ on ancillas, controlled-H_XY to data.

    For simplicity, use the paper's structure: controlled-CZ between GHZ and data.
    """
    # For the paper's eq (26), the checking circuit is:
    # n controlled-H_XY gates from GHZ state to data qubits
    # In ZX this creates the hub structure
    #
    # Simplified version: CZ (controlled-Z) from ancilla to each data qubit
    # This is the simplest circuit with the GHZ hub structure

    dim_data = 2**n

    # CZ from ancilla to data qubit i
    # CZ|00⟩=|00⟩, CZ|01⟩=|01⟩, CZ|10⟩=|10⟩, CZ|11⟩=-|11⟩
    # On data (tracing ancilla with |+⟩ input, project |+⟩):
    # <+|anc CZ^⊗n |+⟩anc = (I + Z^⊗n) / 2  (GHZ measurement)

    # The operator on data qubits (ancilla in |+⟩, projected to |+⟩):
    I_n = np.eye(dim_data, dtype=complex)
    Z_n = kron_n([np.diag([1, -1])] * n) if n > 0 else np.array([[1]])

    return (I_n + Z_n) / 2

# ============================================================================
# C_check version 3: Simple CNOT parity check
# ============================================================================
def c_check_parity(n):
    """Simple parity check: ancilla measures parity of all data qubits.

    CX(0,anc), CX(1,anc), ..., CX(n-1,anc), measure ancilla, project to 0.
    Result: projection onto even parity subspace.
    """
    dim_data = 2**n
    C = np.zeros((dim_data, dim_data), dtype=complex)
    for i in range(dim_data):
        parity = bin(i).count('1') % 2
        if parity == 0:
            C[i, i] = 1
    return C

# Test each version
for n in [2, 3, 4]:
    print(f"\n{'='*60}")
    print(f"n = {n}")
    print(f"{'='*60}")

    check_decomposition(n, c_check_cx_fan, "CX fan-out/fan-in")
    check_decomposition(n, c_check_ghz, "GHZ CZ measurement")
    check_decomposition(n, c_check_parity, "Parity projection")

# ============================================================================
# Now verify with tsim: build actual stim circuits and check ZX T-count
# ============================================================================
print(f"\n\n{'='*60}")
print("tsim verification: Build circuits and decompose")
print(f"{'='*60}")

# Build a minimal cultivation-like circuit for n=3
n = 3
data_qubits = list(range(n))
ancilla = n

# Original circuit: T†^⊗n · CX_fan · CX_fan · T^⊗n
# (CX fan-out to ancilla, then fan-in)
circ_lines = []
# T† on data
circ_lines.append(f"T_DAG {' '.join(str(q) for q in data_qubits)}")
circ_lines.append("TICK")
# CX fan-out
for q in data_qubits:
    circ_lines.append(f"CX {q} {ancilla}")
    circ_lines.append("TICK")
# CX fan-in (same gates)
for q in data_qubits:
    circ_lines.append(f"CX {q} {ancilla}")
    circ_lines.append("TICK")
# T on data
circ_lines.append(f"T {' '.join(str(q) for q in data_qubits)}")
circ_lines.append("TICK")
# Measure ancilla
circ_lines.append(f"M {ancilla}")

circ_str = "\n".join(circ_lines)
print(f"\nMinimal circuit (n={n}):")
print(circ_str)

circuit = tsim.Circuit(circ_str)
graph = circuit.get_graph()
tc = tcount(graph)
print(f"\nZX graph: {len(list(graph.vertices()))} verts, T-count={tc}")

# Reduce
g = deepcopy(graph)
zx.full_reduce(g, paramSafe=True)
tc_r = tcount(g)
nv_r = len(list(g.vertices()))
print(f"After full_reduce: {nv_r} verts, T-count={tc_r}")

# Now build the 2 Clifford variants:
# Term 1: Remove T gates, keep checking circuit
circ1_lines = []
circ1_lines.append("TICK")  # skip T†
for q in data_qubits:
    circ1_lines.append(f"CX {q} {ancilla}")
    circ1_lines.append("TICK")
for q in data_qubits:
    circ1_lines.append(f"CX {q} {ancilla}")
    circ1_lines.append("TICK")
circ1_lines.append("TICK")  # skip T
circ1_lines.append(f"M {ancilla}")

circ1_str = "\n".join(circ1_lines)
circuit1 = tsim.Circuit(circ1_str)
graph1 = circuit1.get_graph()
tc1 = tcount(graph1)
print(f"\nTerm 1 (T removed, C kept): T-count={tc1}")

# Term 2: Remove entire block, insert (XS†)^⊗n on data
circ2_lines = []
for q in data_qubits:
    circ2_lines.append(f"X {q}")
    circ2_lines.append(f"S_DAG {q}")
circ2_lines.append("TICK")
circ2_lines.append(f"M {ancilla}")

circ2_str = "\n".join(circ2_lines)
circuit2 = tsim.Circuit(circ2_str)
graph2 = circuit2.get_graph()
tc2 = tcount(graph2)
print(f"Term 2 (Block → XS†): T-count={tc2}")

# Term 3: Remove entire block, insert I (identity)
circ3_lines = []
circ3_lines.append("TICK")
circ3_lines.append(f"M {ancilla}")

circ3_str = "\n".join(circ3_lines)
circuit3 = tsim.Circuit(circ3_str)
graph3 = circuit3.get_graph()
tc3 = tcount(graph3)
print(f"Term 3 (Block → I): T-count={tc3}")

# ============================================================================
# Tensor verification for the minimal circuit
# ============================================================================
print(f"\n{'='*60}")
print(f"Tensor verification for minimal circuit (n={n})")
print(f"{'='*60}")

# Original tensor
tensor_orig = circuit.to_tensor()
print(f"Original tensor shape: {tensor_orig.shape}")

# Check if T†^⊗n · C · T^⊗n ∝ I + (XS†)^⊗n
# by checking the operator on data qubits
# The circuit acts on n+1 qubits (data + ancilla)
# After tracing out ancilla (projecting to measurement result 0):
dim = 2**(n+1)
dim_data = 2**n

# The tensor from to_tensor() represents the circuit as a matrix/tensor
# For n+1 qubits, it's a 2^(n+1) vector or 2^(n+1) x 2^(n+1) matrix
print(f"Tensor:\n{tensor_orig}")
print(f"Tensor dtype: {tensor_orig.dtype}")
print(f"Tensor shape: {tensor_orig.shape}")

# If it's a matrix, extract the operator on data qubits
# by projecting ancilla to |0⟩
if len(tensor_orig.shape) == 2:
    # It's a matrix: 2^(n+1) x 2^(n+1)
    # Data = first n qubits, ancilla = last qubit
    # Project ancilla input to |0⟩, ancilla output to |0⟩
    op_data = np.zeros((dim_data, dim_data), dtype=complex)
    for i in range(dim_data):
        for j in range(dim_data):
            # ancilla in |0⟩ (bit 0), out |0⟩
            op_data[i, j] = tensor_orig[i*2 + 0, j*2 + 0]

    print(f"\nOperator on data qubits (ancilla traced):")
    print(op_data)

    # Check I + XS† decomposition
    ok, c1, c2 = check_decomposition(n, lambda _: op_data, "Minimal circuit")
