"""Task 1: Numerical verification of the block replacement decomposition.

Build a small cultivation-like circuit matching the Gidney structure:
  T† on data qubits → CNOT cascade → MX ancilla → RX ancilla → CNOT cascade reverse → T on data

Verify: Original = c₁·(I variant) + c₂·(XS† variant) as operator identity.
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
from stab_rank_cut import tcount

# ============================================================================
# Helper: build circuit tensor
# ============================================================================
def circuit_tensor(circ_str):
    """Get the full tensor of a circuit as a numpy array."""
    circuit = tsim.Circuit(circ_str)
    return circuit.to_tensor()

def circuit_zx_tcount(circ_str):
    """Get T-count after ZX full_reduce."""
    circuit = tsim.Circuit(circ_str)
    g = circuit.get_graph()
    g2 = deepcopy(g)
    zx.full_reduce(g2, paramSafe=True)
    return tcount(g2)

# ============================================================================
# Build minimal cultivation circuit: 2 data qubits + 1 helper
# Structure mirrors first cultivation in Gidney:
#   T_DAG data → CX cascade → MX hub → RX hub → CX cascade reverse → T data
#
# Minimal version (n=2 data qubits, qubit 0 and 1 are data, qubit 2 is hub):
#   T_DAG 0 1
#   CX 0 2
#   CX 1 2  (or a tree: CX 1 0, CX 0 2)
#   MX 2
#   RX 2
#   CX 1 2
#   CX 0 2
#   T 0 1
# ============================================================================

print("="*60)
print("Minimal cultivation circuit: 2 data + 1 hub qubit")
print("="*60)

# Version 1: Simple fan-out/fan-in
orig_circ = """T_DAG 0 1
TICK
CX 0 2
TICK
CX 1 2
TICK
MX 2
TICK
RX 2
TICK
CX 1 2
TICK
CX 0 2
TICK
T 0 1
TICK
M 0 1 2"""

print(f"\nOriginal circuit:\n{orig_circ}")
tc = circuit_zx_tcount(orig_circ)
print(f"ZX T-count after reduce: {tc}")

# Identity variant: remove T†, cascade, MX, RX, cascade, T → just measure
id_circ = """TICK
TICK
TICK
TICK
TICK
TICK
TICK
TICK
M 0 1 2"""

print(f"\nIdentity variant:\n{id_circ}")
tc_id = circuit_zx_tcount(id_circ)
print(f"ZX T-count: {tc_id}")

# XS† variant: replace entire block with S_DAG + X on data qubits
xsd_circ = """S_DAG 0 1
TICK
X 0 1
TICK
TICK
TICK
TICK
TICK
TICK
TICK
M 0 1 2"""

print(f"\nXS† variant:\n{xsd_circ}")
tc_xsd = circuit_zx_tcount(xsd_circ)
print(f"ZX T-count: {tc_xsd}")

# ============================================================================
# Tensor comparison
# ============================================================================
print(f"\n{'='*60}")
print("Tensor comparison")
print(f"{'='*60}")

t_orig = np.array(circuit_tensor(orig_circ))
t_id = np.array(circuit_tensor(id_circ))
t_xsd = np.array(circuit_tensor(xsd_circ))

print(f"Original tensor shape: {t_orig.shape}")
print(f"Identity tensor shape: {t_id.shape}")
print(f"XS† tensor shape: {t_xsd.shape}")

# Flatten for comparison
v_orig = t_orig.reshape(-1)
v_id = t_id.reshape(-1)
v_xsd = t_xsd.reshape(-1)

print(f"\nOriginal (non-zero): {np.count_nonzero(np.abs(v_orig) > 1e-10)}")
print(f"Identity (non-zero): {np.count_nonzero(np.abs(v_id) > 1e-10)}")
print(f"XS† (non-zero): {np.count_nonzero(np.abs(v_xsd) > 1e-10)}")

# Try: Original = a * Identity + b * XS†
# Use least squares
A = np.column_stack([v_id, v_xsd])
coeffs, residuals, rank, sv = np.linalg.lstsq(A, v_orig, rcond=None)
a, b = coeffs

reconstructed = a * v_id + b * v_xsd
error = np.linalg.norm(v_orig - reconstructed) / np.linalg.norm(v_orig)

print(f"\nFitting: Original = a·Identity + b·XS†")
print(f"  a = {a}")
print(f"  b = {b}")
print(f"  |a| = {abs(a):.8f}")
print(f"  |b| = {abs(b):.8f}")
print(f"  Reconstruction error: {error:.2e}")
print(f"  Match: {error < 1e-10}")

# ============================================================================
# Version 2: Tree cascade (more like Gidney's structure)
# 4 data qubits (0,1,2,3), hub qubit 4
# Cascade: CX 0 4, CX 1 0, CX 2 4, CX 3 2 → tree merging into qubit 4
# Actually, let's use the exact Gidney pattern with a smaller example
# ============================================================================
print(f"\n{'='*60}")
print("Larger circuit: 4 data qubits + tree cascade")
print("="*60)

# Gidney-style tree: CX pairs, then merge
# data = {0,1,2,3}, helpers = none, hub = qubit 2 (gets MX)
# Cascade: CX 1 0, CX 3 2; then CX 0 2; (fan-in binary tree to qubit 2)
# MX 2; RX 2; reverse cascade

orig4 = """T_DAG 0 1 2 3
TICK
CX 1 0 3 2
TICK
CX 0 2
TICK
MX 2
TICK
RX 2
TICK
CX 0 2
TICK
CX 1 0 3 2
TICK
T 0 1 2 3
TICK
M 0 1 2 3"""

print(f"\nOriginal (4 data):\n{orig4}")
tc4 = circuit_zx_tcount(orig4)
print(f"ZX T-count: {tc4}")

id4 = """TICK
TICK
TICK
TICK
TICK
TICK
TICK
TICK
M 0 1 2 3"""

xsd4 = """S_DAG 0 1 2 3
TICK
X 0 1 2 3
TICK
TICK
TICK
TICK
TICK
TICK
TICK
M 0 1 2 3"""

tc4_id = circuit_zx_tcount(id4)
tc4_xsd = circuit_zx_tcount(xsd4)
print(f"Identity T-count: {tc4_id}")
print(f"XS† T-count: {tc4_xsd}")

t4_orig = np.array(circuit_tensor(orig4)).reshape(-1)
t4_id = np.array(circuit_tensor(id4)).reshape(-1)
t4_xsd = np.array(circuit_tensor(xsd4)).reshape(-1)

A4 = np.column_stack([t4_id, t4_xsd])
coeffs4, _, _, _ = np.linalg.lstsq(A4, t4_orig, rcond=None)
a4, b4 = coeffs4
recon4 = a4 * t4_id + b4 * t4_xsd
error4 = np.linalg.norm(t4_orig - recon4) / np.linalg.norm(t4_orig)

print(f"\nFitting: Original = a·Identity + b·XS†")
print(f"  a = {a4}")
print(f"  b = {b4}")
print(f"  |a| = {abs(a4):.8f}, |b| = {abs(b4):.8f}")
print(f"  Reconstruction error: {error4:.2e}")
print(f"  Match: {error4 < 1e-10}")

if error4 > 1e-5:
    # Try with more basis functions
    print("\n  Trying 3-term fit: Original = a·Id + b·XS† + c·(other)")
    # Try with Z^⊗n
    Z = np.diag([1, -1])
    Zn = Z
    for _ in range(3):
        Zn = np.kron(Zn, Z)
    # Actually that's 16x16, let me make it for 4 qubits
    Z4 = np.diag([1, -1])
    for _ in range(3):
        Z4 = np.kron(Z4, np.diag([1, -1]))

    # Build Z variant circuit
    z4_circ = """Z 0 1 2 3
TICK
TICK
TICK
TICK
TICK
TICK
TICK
TICK
M 0 1 2 3"""
    t4_z = np.array(circuit_tensor(z4_circ)).reshape(-1)

    A4_3 = np.column_stack([t4_id, t4_xsd, t4_z])
    coeffs4_3, _, _, _ = np.linalg.lstsq(A4_3, t4_orig, rcond=None)
    recon4_3 = A4_3 @ coeffs4_3
    error4_3 = np.linalg.norm(t4_orig - recon4_3) / np.linalg.norm(t4_orig)
    print(f"  3-term coefficients: {coeffs4_3}")
    print(f"  3-term error: {error4_3:.2e}")


# ============================================================================
# Version 3: Direct operator comparison (without measurements)
# Build circuits WITHOUT final measurements, get unitary tensor
# ============================================================================
print(f"\n{'='*60}")
print("Direct operator comparison (no measurements)")
print("="*60)

# Minimal: 2 data qubits (0,1), hub qubit 2
orig_no_m = """T_DAG 0 1
TICK
CX 0 2
TICK
CX 1 2
TICK
MX 2
TICK
RX 2
TICK
CX 1 2
TICK
CX 0 2
TICK
T 0 1"""

id_no_m = ""

xsd_no_m = """S_DAG 0 1
TICK
X 0 1"""

# Can't easily build "no measurement" circuits for tsim (it expects measurements?)
# Let's compute the FULL operator including measurement and check outcome-by-outcome

print("\nChecking decomposition per measurement outcome:")
print("For each bit pattern of M 0 1 2:")

# The tensor from tsim has shape (2,2,2, 2,2,2) = (outputs, inputs)
# Actually it has shape based on qubits...
# Let me check what to_tensor actually returns for a circuit with measurements

t_check = circuit_tensor(orig_circ)
print(f"  Circuit with M: tensor shape = {np.array(t_check).shape}")

# The shape is (2,2,2,2,2,2) for 3 qubits measured
# First 3 indices are measurement outcomes (outputs)
# Last 3 indices are initial states (inputs)
# Actually, for a circuit with M at the end, the tensor represents
# the projective measurement operator: output[m] = <m|U|input>

# For our decomposition, we care about the measurement statistics
# when all qubits start in |0⟩:
t_arr = np.array(t_check)
# Probability of each outcome for |000⟩ input:
if len(t_arr.shape) == 6:  # 3 qubits: (2,2,2,2,2,2)
    for m0 in range(2):
        for m1 in range(2):
            for m2 in range(2):
                amp_orig = t_arr[m0, m1, m2, 0, 0, 0]
                amp_id = np.array(circuit_tensor(id_circ))[m0, m1, m2, 0, 0, 0]
                amp_xsd = np.array(circuit_tensor(xsd_circ))[m0, m1, m2, 0, 0, 0]

                p_orig = abs(amp_orig)**2
                p_recon = abs(a * amp_id + b * amp_xsd)**2

                if p_orig > 1e-15 or p_recon > 1e-15:
                    print(f"  |{m0}{m1}{m2}⟩: p_orig={p_orig:.6f}, "
                          f"amp_orig={amp_orig:.4f}, "
                          f"a·amp_id+b·amp_xsd={a*amp_id+b*amp_xsd:.4f}")

# ============================================================================
# Version 4: Match Gidney structure more closely with H gate
# The Gidney circuit uses MX which is M in X basis = H M H
# In ZX terms, MX creates X-basis measurement spider
# ============================================================================
print(f"\n{'='*60}")
print("Version 4: Trying with H-M-H (explicit X basis)")
print("="*60)

# Alternative: use H before M to simulate MX
orig_hm = """T_DAG 0 1
TICK
CX 0 2
TICK
CX 1 2
TICK
H 2
M 2
TICK
R 2
H 2
TICK
CX 1 2
TICK
CX 0 2
TICK
T 0 1
TICK
M 0 1"""

print(f"Circuit:\n{orig_hm}")
tc_hm = circuit_zx_tcount(orig_hm)
print(f"T-count: {tc_hm}")

try:
    t_hm = np.array(circuit_tensor(orig_hm))
    print(f"Tensor shape: {t_hm.shape}")
except Exception as e:
    print(f"Tensor error: {e}")

# Identity and XS† variants with M 0 1 only
id_hm = """TICK
TICK
TICK
TICK
TICK
TICK
TICK
TICK
TICK
M 0 1"""

xsd_hm = """S_DAG 0 1
TICK
X 0 1
TICK
TICK
TICK
TICK
TICK
TICK
TICK
M 0 1"""

# But wait - we need qubit 2 to exist. Let me add a trivial measurement
id_hm2 = """TICK
TICK
TICK
TICK
TICK
TICK
TICK
TICK
TICK
M 0 1 2"""

xsd_hm2 = """S_DAG 0 1
TICK
X 0 1
TICK
TICK
TICK
TICK
TICK
TICK
TICK
M 0 1 2"""

try:
    t_id_hm = np.array(circuit_tensor(id_hm2))
    t_xsd_hm = np.array(circuit_tensor(xsd_hm2))

    v_hm = t_hm.reshape(-1) if 't_hm' in dir() else None
    v_id_hm = t_id_hm.reshape(-1)
    v_xsd_hm = t_xsd_hm.reshape(-1)

    if v_hm is not None and v_hm.shape == v_id_hm.shape:
        A_hm = np.column_stack([v_id_hm, v_xsd_hm])
        c_hm, _, _, _ = np.linalg.lstsq(A_hm, v_hm, rcond=None)
        recon_hm = A_hm @ c_hm
        err_hm = np.linalg.norm(v_hm - recon_hm) / max(np.linalg.norm(v_hm), 1e-15)
        print(f"\nFitting: a={c_hm[0]:.6f}, b={c_hm[1]:.6f}")
        print(f"Error: {err_hm:.2e}")
    else:
        print("Shape mismatch, skipping fit")
except Exception as e:
    print(f"Error: {e}")
