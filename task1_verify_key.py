"""Task 1: Verify decomposition on minimal circuits where hub qubit = data qubit.

Key insight from Gidney: qubit 7 receives T†/T AND is the MX measurement hub.
Build small circuits capturing this feature and verify decomposition identity.
"""
import sys, os
from pathlib import Path
import numpy as np

os.environ["JAX_PLATFORMS"] = "cpu"

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))

import tsim

def get_amps(circ_str, n_qubits):
    """Get amplitudes for all-zero input state."""
    circuit = tsim.Circuit(circ_str)
    tensor = np.array(circuit.to_tensor())
    idx = tuple([slice(None)] * n_qubits + [0] * n_qubits)
    return tensor[idx].reshape(-1)

def try_decomposition(orig_amps, basis_circuits, n_qubits, labels):
    """Try least-squares decomposition of orig into basis circuits."""
    basis_amps = [get_amps(c, n_qubits) for c in basis_circuits]
    A = np.column_stack(basis_amps)
    coeffs, _, _, _ = np.linalg.lstsq(A, orig_amps, rcond=None)
    recon = A @ coeffs
    error = np.linalg.norm(orig_amps - recon) / max(np.linalg.norm(orig_amps), 1e-15)

    for label, c in zip(labels, coeffs):
        print(f"    {label}: {c:.6f} (|c|={abs(c):.6f})")
    print(f"    Error: {error:.2e}")
    return error, coeffs


# ============================================================================
# Circuit 1: 2 qubits, hub=data qubit
# Data: {0, 1}, hub=1 (receives T and gets MX)
# ============================================================================
print("="*60)
print("Circuit A: 2 data qubits, hub=qubit 1")
print("="*60)

origA = """T_DAG 0 1
TICK
CX 0 1
TICK
MX 1
TICK
RX 1
TICK
CX 0 1
TICK
T 0 1
TICK
M 0 1"""

print(f"Circuit:\n{origA}\n")
amps_origA = get_amps(origA, 2)
print(f"Amplitudes for |00⟩ input:")
for i in range(4):
    bits = format(i, '02b')
    if abs(amps_origA[i]) > 1e-10:
        print(f"  |{bits}⟩: {amps_origA[i]:.6f}")

# Identity variant: remove entire block
idA = """TICK
TICK
TICK
TICK
TICK
TICK
TICK
M 0 1"""

# XS† variant
xsdA = """S_DAG 0 1
TICK
X 0 1
TICK
TICK
TICK
TICK
TICK
M 0 1"""

# Z variant
zA = """Z 0 1
TICK
TICK
TICK
TICK
TICK
TICK
M 0 1"""

# X variant
xA = """X 0 1
TICK
TICK
TICK
TICK
TICK
TICK
M 0 1"""

# S variant
sA = """S 0 1
TICK
TICK
TICK
TICK
TICK
TICK
M 0 1"""

# SX variant (= S · X)
sxA = """X 0 1
TICK
S 0 1
TICK
TICK
TICK
TICK
TICK
M 0 1"""

print("\n2-term fit (I, XS†):")
try_decomposition(amps_origA, [idA, xsdA], 2, ["I", "XS†"])

print("\n3-term fit (I, XS†, Z):")
try_decomposition(amps_origA, [idA, xsdA, zA], 2, ["I", "XS†", "Z"])

print("\n4-term fit (I, XS†, Z, X):")
try_decomposition(amps_origA, [idA, xsdA, zA, xA], 2, ["I", "XS†", "Z", "X"])

print("\n6-term fit (I, XS†, Z, X, S, SX):")
try_decomposition(amps_origA, [idA, xsdA, zA, xA, sA, sxA], 2,
                   ["I", "XS†", "Z", "X", "S", "SX"])


# ============================================================================
# Circuit B: 3 data + 1 helper, tree structure, hub=data
# Data: {0, 1, 3}, hub=1 (data+hub), helper=2
# Tree: CX 2 0, CX 3 1 (helpers→data), CX 1 2 (hub→helper)...
# Actually let me match Gidney more closely
# ============================================================================
print(f"\n{'='*60}")
print("Circuit B: 3 data + 1 helper, tree cascade, hub=data")
print("="*60)

# Data: {0, 2, 3}, hub=2 (data+hub), helper=1
# Layer 1: CX 1 0, CX 3 2 (two pairs)
# Layer 2: CX 2 1 (merge)
# MX 2, RX 2
# Reverse
origB = """T_DAG 0 2 3
TICK
CX 1 0 3 2
TICK
CX 2 1
TICK
MX 2
TICK
RX 2
TICK
CX 2 1
TICK
CX 1 0 3 2
TICK
T 0 2 3
TICK
M 0 1 2 3"""

print(f"Circuit:\n{origB}\n")
amps_origB = get_amps(origB, 4)
n_nonzero = np.count_nonzero(np.abs(amps_origB) > 1e-10)
print(f"Non-zero amplitudes: {n_nonzero} / {len(amps_origB)}")

# Variants for n=3 data qubits
data_str_B = "0 2 3"

idB = """TICK
TICK
TICK
TICK
TICK
TICK
TICK
TICK
M 0 1 2 3"""

xsdB = f"""S_DAG {data_str_B}
TICK
X {data_str_B}
TICK
TICK
TICK
TICK
TICK
TICK
M 0 1 2 3"""

zB = f"""Z {data_str_B}
TICK
TICK
TICK
TICK
TICK
TICK
TICK
M 0 1 2 3"""

xB = f"""X {data_str_B}
TICK
TICK
TICK
TICK
TICK
TICK
TICK
M 0 1 2 3"""

print("2-term fit (I, XS†):")
try_decomposition(amps_origB, [idB, xsdB], 4, ["I", "XS†"])

print("\n3-term fit (I, XS†, Z):")
try_decomposition(amps_origB, [idB, xsdB, zB], 4, ["I", "XS†", "Z"])

print("\n4-term fit (I, XS†, Z, X):")
try_decomposition(amps_origB, [idB, xsdB, zB, xB], 4, ["I", "XS†", "Z", "X"])


# ============================================================================
# Circuit C: 5 data + 2 helper, more realistic tree
# ============================================================================
print(f"\n{'='*60}")
print("Circuit C: 5 data + 2 helper")
print("="*60)

# Data: {0, 2, 4, 5, 6}, hub=4 (data+hub), helpers={1, 3}
# Layer 1: CX 1 0, CX 3 2, CX 5 4, CX 6 4  (but 6→4 is same as other)
# Actually let's use a simpler tree:
# Data: {0, 1, 2, 3, 4}, hub=2 (data+hub), helpers={5, 6}
# Layer 1: CX 5 0, CX 6 1, CX 3 2, CX 4 2 ... this is getting complicated

# Let me use the simplest possible tree that matches the DEGREE of the Gidney hub
# In Gidney, hub 7 is connected to many spiders through the tree
# The key is: hub receives T AND gets MX

# Simple chain: all data qubits fan into hub via cascade
# Data: {0,1,2,3,4}, hub=4
origC = """T_DAG 0 1 2 3 4
TICK
CX 0 4
TICK
CX 1 4
TICK
CX 2 4
TICK
CX 3 4
TICK
MX 4
TICK
RX 4
TICK
CX 3 4
TICK
CX 2 4
TICK
CX 1 4
TICK
CX 0 4
TICK
T 0 1 2 3 4
TICK
M 0 1 2 3 4"""

amps_origC = get_amps(origC, 5)
n_nonzero = np.count_nonzero(np.abs(amps_origC) > 1e-10)
print(f"Non-zero amplitudes: {n_nonzero} / {len(amps_origC)}")

data_str_C = "0 1 2 3 4"

idC = "TICK\n" * 13 + "M 0 1 2 3 4"
xsdC = f"S_DAG {data_str_C}\nTICK\nX {data_str_C}\nTICK\n" + "TICK\n" * 11 + "M 0 1 2 3 4"
zC = f"Z {data_str_C}\nTICK\n" + "TICK\n" * 12 + "M 0 1 2 3 4"
xC = f"X {data_str_C}\nTICK\n" + "TICK\n" * 12 + "M 0 1 2 3 4"

print("2-term (I, XS†):")
try_decomposition(amps_origC, [idC, xsdC], 5, ["I", "XS†"])

print("\n3-term (I, XS†, Z):")
try_decomposition(amps_origC, [idC, xsdC, zC], 5, ["I", "XS†", "Z"])

print("\n4-term (I, XS†, Z, X):")
try_decomposition(amps_origC, [idC, xsdC, zC, xC], 5, ["I", "XS†", "Z", "X"])


# ============================================================================
# Key test: what IS the operator? Compute T†·C·T directly
# ============================================================================
print(f"\n{'='*60}")
print("Direct operator computation for 2-qubit circuit")
print("="*60)

T = np.diag([1, np.exp(1j * np.pi / 4)])
Td = np.diag([1, np.exp(-1j * np.pi / 4)])
H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
I2 = np.eye(2)
XSd = np.array([[0, -1j], [1, 0]])
X = np.array([[0, 1], [1, 0]])
Z = np.diag([1, -1])
S = np.diag([1, 1j])
Sd = np.diag([1, -1j])

# For 2 qubits (0, 1), hub=1:
# CX with control=0, target=1
CX01 = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])

# T_DAG on both: Td⊗Td
Td2 = np.kron(Td, Td)
T2 = np.kron(T, T)

# The checking circuit: CX(0→1), MX(1), RX(1), CX(0→1)
# MX(1): measure qubit 1 in X basis → project onto |+⟩ or |−⟩
# RX(1): reset to |+⟩

# For MX outcome m=0 (|+⟩):
# Projector: |+⟩⟨+| on qubit 1
plus = np.array([1, 1]) / np.sqrt(2)
minus = np.array([1, -1]) / np.sqrt(2)
P_plus_1 = np.kron(I2, np.outer(plus, plus))
P_minus_1 = np.kron(I2, np.outer(minus, minus))

# Reset to |+⟩: replace qubit 1 state with |+⟩
# Reset operator: |+⟩⟨anything| → but this isn't unitary
# RX: prepare qubit 1 in |+⟩ state regardless
# After MX, qubit 1 is in |+⟩ or |−⟩. RX resets to |+⟩.
# Net effect for outcome m: qubit 1 → |+⟩, with projection ⟨m_X|

# For outcome m=0: operator = CX · (I⊗|+⟩)(I⊗⟨+|) · CX · (on 2 qubits)
# = CX · (I⊗|+⟩⟨+|) · CX

# Full block for outcome 0:
# T†⊗T† · CX · |+⟩⟨+|_1 · CX · T⊗T
#
# Actually: the circuit is:
# Td^⊗2 → CX(0,1) → MX(1)[→m] → RX(1)[→|+⟩] → CX(0,1) → T^⊗2
#
# For outcome m=0:
# The state after CX(0,1) is: CX|ψ⟩
# MX(1) projects qubit 1 to |+⟩: (I⊗⟨+|)CX|ψ⟩
# RX(1) prepares |+⟩: resulting state is (I⊗|+⟩)(I⊗⟨+|)CX|ψ⟩ = (I⊗|+⟩) α(ψ)
# Then CX(0,1): CX(I⊗|+⟩)α(ψ)
# But CX|a⟩|+⟩ = (1/√2)(|a⟩|0⊕a⟩ + |a⟩|1⊕a⟩) = |a⟩|+⟩ (since CX preserves |+⟩ on target)
# Wait, that's not right. CX|a,b⟩ = |a, b⊕a⟩
# CX|a,+⟩ = CX|a⟩(|0⟩+|1⟩)/√2 = (|a,a⟩+|a,1-a⟩)/√2 = |a⟩(|a⟩+|1-a⟩)/√2
# = |a⟩|+⟩ when a=0 (gives (|0⟩+|1⟩)/√2 = |+⟩)
# = |a⟩|+⟩ when a=1 (gives (|1⟩+|0⟩)/√2 = |+⟩)
# So CX|a,+⟩ = |a,+⟩! CX preserves |+⟩ on target.
#
# So after RX(1) which gives |+⟩ on qubit 1:
# CX(0,1) acts on |ψ_0⟩|+⟩ = |ψ_0⟩|+⟩ (no change!)
# Then T^⊗2 gives: T|ψ_0⟩ ⊗ T|+⟩
#
# Meanwhile, before MX/RX:
# The state is Td^⊗2 CX(0,1) |ψ_in⟩
# After CX(0,1): entangled state
# After MX(1) with outcome 0: (I⊗⟨+|) Td^⊗2 CX(0,1) |ψ_in⟩
# This gives a 1-qubit state for qubit 0: ψ_0 = ⟨+| Td CX(0,1) |ψ_in⟩
# The factor Td^⊗2 on both qubits...
#
# Actually let me just compute this numerically.

print("\n--- Numerical computation ---")

# Input state |00⟩
psi_in = np.array([1, 0, 0, 0])  # |00⟩

# Step 1: T†^⊗2
psi1 = Td2 @ psi_in
print(f"After T†⊗T†: {psi1}")

# Step 2: CX(0,1)
psi2 = CX01 @ psi1
print(f"After CX: {psi2}")

# Step 3: MX(1) with outcome m=0 (project qubit 1 to |+⟩)
# ⟨+|_1 on qubit 1
psi3_unnorm_0 = np.zeros(2, dtype=complex)
for q0 in range(2):
    for q1 in range(2):
        amp = psi2[q0 * 2 + q1]
        psi3_unnorm_0[q0] += plus[q1] * amp

print(f"After ⟨+|_1: qubit 0 state = {psi3_unnorm_0}")

# Step 4: RX(1) → qubit 1 = |+⟩
psi4_0 = np.kron(psi3_unnorm_0, plus)
print(f"After RX: {psi4_0}")

# Step 5: CX(0,1)
psi5_0 = CX01 @ psi4_0
print(f"After 2nd CX: {psi5_0}")

# Step 6: T^⊗2
psi6_0 = T2 @ psi5_0
print(f"After T⊗T: {psi6_0}")

# Compare with identity (just measure |00⟩)
psi_id = np.array([1, 0, 0, 0])
print(f"\nIdentity: {psi_id}")

# Compare with XS†^⊗2
# XS† = [[0,-i],[1,0]], (XS†)^⊗2 on |00⟩
XSd2 = np.kron(XSd, XSd)
psi_xsd = XSd2 @ np.array([1, 0, 0, 0])
print(f"(XS†)^⊗2 |00⟩: {psi_xsd}")

# Check: psi6_0 = a * psi_id + b * psi_xsd ?
A_fit = np.column_stack([psi_id, psi_xsd])
c_fit, _, _, _ = np.linalg.lstsq(A_fit, psi6_0, rcond=None)
recon = A_fit @ c_fit
err = np.linalg.norm(psi6_0 - recon)
print(f"\nFit: psi6 = {c_fit[0]:.6f}·I|00⟩ + {c_fit[1]:.6f}·XS†|00⟩")
print(f"Error: {err:.2e}")

# Also check outcome m=1
psi3_unnorm_1 = np.zeros(2, dtype=complex)
for q0 in range(2):
    for q1 in range(2):
        amp = psi2[q0 * 2 + q1]
        psi3_unnorm_1[q0] += minus[q1] * amp

psi4_1 = np.kron(psi3_unnorm_1, plus)
psi5_1 = CX01 @ psi4_1
psi6_1 = T2 @ psi5_1

print(f"\nOutcome m=1: {psi6_1}")
c_fit1, _, _, _ = np.linalg.lstsq(A_fit, psi6_1, rcond=None)
recon1 = A_fit @ c_fit1
err1 = np.linalg.norm(psi6_1 - recon1)
print(f"Fit: psi6 = {c_fit1[0]:.6f}·I|00⟩ + {c_fit1[1]:.6f}·XS†|00⟩")
print(f"Error: {err1:.2e}")

# ============================================================================
# Key: what operator does the block implement on DATA qubits?
# ============================================================================
print(f"\n{'='*60}")
print("Full operator on data qubits (2-qubit block)")
print("="*60)

# For each input |i,j⟩, compute output for both MX outcomes
# The block operator (including MX/RX) acts as:
# O|ψ_data⟩ = Σ_m p(m) · O_m|ψ_data⟩
# where O_m is the conditional operator for outcome m

# O_m = T^⊗2 · CX · (|+⟩ on q1) · ⟨m_X|_q1 · CX · T†^⊗2
# But qubit 1 is ALSO a data qubit, so the operator acts on BOTH qubits

# Let's compute the SUPEROPERATOR: trace over measurement outcomes
# ρ_out = Σ_m K_m ρ_in K_m†
# where K_m = T^⊗2 · CX · (I⊗|+⟩)(I⊗⟨m_X|) · CX · T†^⊗2

# Each K_m: 4→4 operator (but rank 2 since qubit 1 is projected)
K0 = np.zeros((4, 4), dtype=complex)
K1 = np.zeros((4, 4), dtype=complex)

for in_state in range(4):
    psi = np.zeros(4, dtype=complex)
    psi[in_state] = 1

    p1 = Td2 @ psi
    p2 = CX01 @ p1

    # Outcome 0: project q1 to |+⟩
    p3_0 = np.zeros(2, dtype=complex)
    for q0 in range(2):
        for q1 in range(2):
            p3_0[q0] += plus[q1] * p2[q0 * 2 + q1]
    p4_0 = np.kron(p3_0, plus)
    p5_0 = CX01 @ p4_0
    K0[:, in_state] = T2 @ p5_0

    # Outcome 1: project q1 to |−⟩
    p3_1 = np.zeros(2, dtype=complex)
    for q0 in range(2):
        for q1 in range(2):
            p3_1[q0] += minus[q1] * p2[q0 * 2 + q1]
    p4_1 = np.kron(p3_1, plus)
    p5_1 = CX01 @ p4_1
    K1[:, in_state] = T2 @ p5_1

print("Kraus operator K0 (MX=0):")
print(K0)
print(f"\nKraus operator K1 (MX=1):")
print(K1)

# Check: K0 = a0*I + b0*(XS†)^⊗2 ?
I4 = np.eye(4)
XSd2_mat = np.kron(XSd, XSd)

print(f"\n--- Decomposing K0 ---")
A_K = np.column_stack([I4.reshape(-1), XSd2_mat.reshape(-1)])
c_K0, _, _, _ = np.linalg.lstsq(A_K, K0.reshape(-1), rcond=None)
recon_K0 = (c_K0[0] * I4 + c_K0[1] * XSd2_mat)
err_K0 = np.linalg.norm(K0 - recon_K0) / max(np.linalg.norm(K0), 1e-15)
print(f"K0 = {c_K0[0]:.6f}·I + {c_K0[1]:.6f}·(XS†)^⊗2")
print(f"Error: {err_K0:.2e}")

print(f"\n--- Decomposing K1 ---")
c_K1, _, _, _ = np.linalg.lstsq(A_K, K1.reshape(-1), rcond=None)
recon_K1 = (c_K1[0] * I4 + c_K1[1] * XSd2_mat)
err_K1 = np.linalg.norm(K1 - recon_K1) / max(np.linalg.norm(K1), 1e-15)
print(f"K1 = {c_K1[0]:.6f}·I + {c_K1[1]:.6f}·(XS†)^⊗2")
print(f"Error: {err_K1:.2e}")

# Try with more basis: I, XS†, Z, X, S, Sd, SX, XSd_individual...
print(f"\n--- Decomposing K0 with more basis operators ---")
# Single-qubit Cliffords on 2 qubits
bases_2q = {
    "I": I4,
    "XS†⊗2": XSd2_mat,
    "Z⊗2": np.kron(Z, Z),
    "X⊗2": np.kron(X, X),
    "S⊗2": np.kron(S, S),
    "S†⊗2": np.kron(Sd, Sd),
    "I⊗XS†": np.kron(I2, XSd),
    "XS†⊗I": np.kron(XSd, I2),
    "I⊗X": np.kron(I2, X),
    "X⊗I": np.kron(X, I2),
    "I⊗Z": np.kron(I2, Z),
    "Z⊗I": np.kron(Z, I2),
}

A_big = np.column_stack([v.reshape(-1) for v in bases_2q.values()])
labels_big = list(bases_2q.keys())
c_big, _, _, _ = np.linalg.lstsq(A_big, K0.reshape(-1), rcond=None)
recon_big = A_big @ c_big
err_big = np.linalg.norm(K0.reshape(-1) - recon_big) / np.linalg.norm(K0.reshape(-1))

for label, coeff in zip(labels_big, c_big):
    if abs(coeff) > 1e-10:
        print(f"    {label}: {coeff:.6f}")
print(f"  Error: {err_big:.2e}")

print(f"\n--- Decomposing K1 with more basis operators ---")
c_big1, _, _, _ = np.linalg.lstsq(A_big, K1.reshape(-1), rcond=None)
recon_big1 = A_big @ c_big1
err_big1 = np.linalg.norm(K1.reshape(-1) - recon_big1) / np.linalg.norm(K1.reshape(-1))

for label, coeff in zip(labels_big, c_big1):
    if abs(coeff) > 1e-10:
        print(f"    {label}: {coeff:.6f}")
print(f"  Error: {err_big1:.2e}")

# If individual terms don't work, the block might map to non-product operators
# Let's check: is K0 itself Clifford?
print(f"\n--- Check if K0, K1 are proportional to Clifford unitaries ---")
print(f"K0 singular values: {np.linalg.svd(K0)[1]}")
print(f"K1 singular values: {np.linalg.svd(K1)[1]}")
print(f"K0 rank: {np.linalg.matrix_rank(K0, tol=1e-10)}")
print(f"K1 rank: {np.linalg.matrix_rank(K1, tol=1e-10)}")
