"""Task 1: Extract decomposition coefficients from ZX scalars.

All 4 variants are Clifford (T-count=0). Now extract the complex scalars
to determine: Circuit = Σ_i c_i · Clifford_i

From the ZX scalar tracking: each graph has a Scalar object that tracks
the global phase and normalization.
"""
import sys, os, re
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
import os
_PRX_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

# Load circuit
CIRCUIT_PATH = Path(os.path.join(_PRX_ROOT, "gidney-circuits", "circuits", ""
                    "for_perfectionist_decoding/"
                    "c=inject[unitary]+cultivate,p=0.001,noise=uniform,"
                    "g=css,q=42,b=Y,r=10,d1=5.stim")
circuit_str = CIRCUIT_PATH.read_text()

# Strip noise
lines = circuit_str.split('\n')
noiseless = []
for line in lines:
    s = line.strip()
    if any(s.startswith(p) for p in ['X_ERROR', 'Z_ERROR', 'DEPOLARIZE1', 'DEPOLARIZE2']):
        continue
    line = re.sub(r'M\([\d.]+\)', 'M', line)
    line = re.sub(r'MX\([\d.]+\)', 'MX', line)
    noiseless.append(line)

def replace_s_with_t(s):
    s = re.sub(r'^(\s*)S_DAG(\s)', r'\1T_DAG\2', s, flags=re.MULTILINE)
    return re.sub(r'^(\s*)S(\s)', r'\1T\2', s, flags=re.MULTILINE)

t_lines = replace_s_with_t('\n'.join(noiseless)).split('\n')

# Block definitions
block1_start, block1_end = 105, 127
block1_data = [0, 3, 7, 9, 11, 13, 17]
block2_start, block2_end = 238, 272
block2_data = [0, 3, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 32, 34, 36, 38, 40]
injection_line = 63

def make_xsd_clifford(data_qubits):
    result = []
    result.append(f"S_DAG {' '.join(str(q) for q in data_qubits)}")
    result.append("TICK")
    result.append(f"X {' '.join(str(q) for q in data_qubits)}")
    result.append("TICK")
    return result

def build_variant(t_lines, block1_mode, block2_mode, inject_mode="keep"):
    result = []
    for i, line in enumerate(t_lines):
        if i == injection_line:
            if inject_mode == "identity":
                result.append("TICK")
                continue
            elif inject_mode == "xsd":
                result.append("S_DAG 3")
                result.append("X 3")
                result.append("TICK")
                continue
        if block1_start <= i <= block1_end:
            if i == block1_start:
                if block1_mode == "identity":
                    result.append("TICK")
                elif block1_mode == "xsd":
                    result.extend(make_xsd_clifford(block1_data))
                else:
                    result.append(line)
            elif block1_mode != "keep":
                continue
            else:
                result.append(line)
            continue
        if block2_start <= i <= block2_end:
            if i == block2_start:
                if block2_mode == "identity":
                    result.append("TICK")
                elif block2_mode == "xsd":
                    result.extend(make_xsd_clifford(block2_data))
                else:
                    result.append(line)
            elif block2_mode != "keep":
                continue
            else:
                result.append(line)
            continue
        result.append(line)
    return '\n'.join(result)

# ============================================================================
# Extract scalars from reduced variants
# ============================================================================
print("="*60)
print("Extracting ZX scalars from 4 Clifford variants")
print("="*60)

variants = [
    ("I ⊗ I", "identity", "identity"),
    ("I ⊗ XS†", "identity", "xsd"),
    ("XS† ⊗ I", "xsd", "identity"),
    ("XS† ⊗ XS†", "xsd", "xsd"),
]

scalars = {}
graphs_reduced = {}

for name, b1, b2 in variants:
    circ_text = build_variant(t_lines, b1, b2, inject_mode="keep")
    circuit = tsim.Circuit(circ_text)
    graph = circuit.get_graph()

    g = deepcopy(graph)
    zx.full_reduce(g, paramSafe=True)

    # Extract scalar info
    sc = g.scalar
    print(f"\nVariant ({name}):")
    print(f"  T-count: {tcount(g)}")
    print(f"  Scalar object: {sc}")
    print(f"  Scalar attributes: {[a for a in dir(sc) if not a.startswith('_')]}")

    # Try to get numerical value
    try:
        sc_val = sc.to_number()
        print(f"  Scalar value: {sc_val}")
        print(f"  |scalar|: {abs(sc_val):.8f}")
        print(f"  phase: {np.angle(sc_val)/np.pi:.6f}π")
    except Exception as e:
        print(f"  Scalar to_number() error: {e}")

    # Try accessing components
    try:
        print(f"  power2: {sc.power2}")
        print(f"  phase: {sc.phase}")
        print(f"  phasef: {float(sc.phase) if hasattr(sc.phase, '__float__') else sc.phase}")
        print(f"  floatfactor: {sc.floatfactor}")
        print(f"  is_zero: {sc.is_zero}")
    except Exception as e:
        print(f"  Component access error: {e}")

    scalars[name] = sc
    graphs_reduced[name] = g

# ============================================================================
# Also get original scalar
# ============================================================================
print(f"\n{'='*60}")
print("Original circuit scalar")
print(f"{'='*60}")

orig_text = build_variant(t_lines, "keep", "keep", inject_mode="keep")
orig_circuit = tsim.Circuit(orig_text)
orig_graph = orig_circuit.get_graph()
g_orig = deepcopy(orig_graph)
zx.full_reduce(g_orig, paramSafe=True)

sc_orig = g_orig.scalar
print(f"  T-count: {tcount(g_orig)}")
print(f"  Scalar: {sc_orig}")
try:
    sc_val = sc_orig.to_number()
    print(f"  Value: {sc_val}")
except Exception as e:
    print(f"  to_number() error: {e}")
try:
    print(f"  power2: {sc_orig.power2}")
    print(f"  phase: {sc_orig.phase}")
    print(f"  floatfactor: {sc_orig.floatfactor}")
except Exception as e:
    print(f"  Error: {e}")

# ============================================================================
# Compute tensors for small verification
# ============================================================================
print(f"\n{'='*60}")
print("Tensor verification (if feasible)")
print(f"{'='*60}")

# Check how many qubits
for name, b1, b2 in variants:
    circ_text = build_variant(t_lines, b1, b2, inject_mode="keep")
    circuit = tsim.Circuit(circ_text)
    n_qubits = circuit.num_qubits
    print(f"  ({name}): {n_qubits} qubits")

# For the original
orig_circuit = tsim.Circuit(build_variant(t_lines, "keep", "keep", inject_mode="keep"))
print(f"  Original: {orig_circuit.num_qubits} qubits")
print(f"  (42 qubits → 2^42 ≈ 4T elements, tensor too large for direct computation)")

# ============================================================================
# Spider cutting coefficient analysis
# ============================================================================
print(f"\n{'='*60}")
print("Spider cutting coefficient analysis")
print(f"{'='*60}")

# From the spider cutting identity (equation 18):
# Z_α(n) = (1/√2)^n * Σ_{k=0}^{1} e^{ikα} Z_{kπ}(n)
#
# For n=1 (degree 1 spider), the T gate (phase π/4):
# T = (1/√2) [I + e^{iπ/4} Z]   ... this is in ZX, not matrix form
#
# But we're cutting the HUB spiders, not individual T spiders.
# The hub spider has degree n (number of T-spider connections).
#
# For a phase-0 Z-spider with degree n:
# Z_0(n) = (1/√2)^? × [G_0 + G_π]   where G_0,G_π are the 0/π replacement terms
#
# The paper's equation (26-29) shows:
# Cut 2 hub spiders → 4 terms → 2 survive → I + XS†
#
# For our circuit-level approach, each cultivation block contributes:
# Block_i = α_i · I + β_i · (XS†)^⊗n_i
#
# where α_i, β_i are the coefficients from the 2-spider cut.

print("From Appendix C (eq 29):")
print("  T†^⊗n · C · T^⊗n ∝ I^⊗n + (XS†)^⊗n")
print("")
print("  XS† = [[0,-i],[1,0]]")
print("  (XS†)^2 = [[0,-i],[1,0]]^2 = [[-i,0],[0,-i]] = -iI")
print("  So (XS†)^⊗n has eigenvalues that are n-th tensor powers of ±1, ±i")
print("")
XSd = np.array([[0, -1j], [1, 0]])
print(f"  XS† = {XSd}")
print(f"  (XS†)^2 = {XSd @ XSd}")
print(f"  trace(XS†) = {np.trace(XSd)}")
print(f"  det(XS†) = {np.linalg.det(XSd)}")
print(f"  eigenvalues: {np.linalg.eigvals(XSd)}")

# The coefficients for the decomposition:
# Each block gives 2 terms with equal magnitude |c|
# The full circuit = Σ c_i Clifford_i with 4 terms
# For sampling: P(m) = |Σ c_i <m|φ_i>|²

# Let's determine coefficients from the ZX scalar structure.
# In ZX, cutting a degree-d phase-0 spider gives:
# (1/√2) × [G_0 + G_π]
# The normalization (1/√2) applies PER CUT.
#
# For 2 cultivation blocks, we make 2 cuts total (one hub per block):
# Full = (1/√2)² × [G_{00} + G_{0π} + G_{π0} + G_{ππ}]
# = (1/2) × [all 4 variants]
#
# But equation (28) shows that 2 of 4 terms vanish when cutting 2 spiders
# in the same block (the "hub pair"). In our case, the 2 blocks are
# SEPARATE, so all 4 terms survive.

print(f"\n--- Coefficient structure ---")
print("Each cultivation block's structured cut: (1/√2) × [I + (XS†)]")
print("Two blocks: (1/√2)² × [I₁+XS₁†] × [I₂+XS₂†]")
print("= (1/2) × [I₁I₂ + I₁·XS₂† + XS₁†·I₂ + XS₁†·XS₂†]")
print("= (1/2) × [4 Clifford terms]")
print("")
print("So c_i = 1/2 for each term (equal weight)")
print("χ = 4 Clifford terms (stabilizer rank)")
print("")
print("For Born probability:")
print("P(m|f) = |Σ_i (1/2) <m|φ_i(f)>|²")
print("       = (1/4) |Σ_i <m|φ_i(f)>|²")
print("")
print("This means χ² = 16 amplitude evaluations per shot.")
