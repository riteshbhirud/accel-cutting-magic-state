"""Task 1d: Construct 4 Clifford circuits from the d=5 cultivation circuit.

The analytical decomposition from Appendix C:
- T = (I + e^{iπ/4}Z)/√2 (spider cutting identity)
- For T†^⊗n · C · T^⊗n where C has GHZ structure:
  Only 2 substitution patterns survive: T/T† → I and T/T† → Z
- d=5 has 2 cultivation layers → 2×2 = 4 Clifford circuits

Construction:
1. Identify the T-gate blocks in the circuit
2. For each block, substitute T/T† with I (remove) or Z
3. 4 cross-product variants = 4 Clifford circuits
4. Verify T-count=0 for each
"""
import sys, os
from pathlib import Path
from copy import deepcopy
from fractions import Fraction

os.environ["JAX_PLATFORMS"] = "cpu"

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))

import re
import pyzx_param as zx
from stab_rank_cut import tcount

# ============================================================================
# Load original circuit (with noise stripped but S gates intact)
# ============================================================================
CIRCUIT_PATH = Path(os.path.join(_PRX_ROOT, "gidney-circuits", "circuits", ""
                    "for_perfectionist_decoding/"
                    "c=inject[unitary]+cultivate,p=0.001,noise=uniform,"
                    "g=css,q=42,b=Y,r=10,d1=5.stim")

circuit_str = CIRCUIT_PATH.read_text()

# Strip noise
lines = circuit_str.split('\n')
noiseless_lines = []
for line in lines:
    stripped = line.strip()
    if any(stripped.startswith(prefix) for prefix in [
        'X_ERROR', 'Z_ERROR', 'DEPOLARIZE1', 'DEPOLARIZE2',
    ]):
        continue
    line = re.sub(r'M\([\d.]+\)', 'M', line)
    line = re.sub(r'MX\([\d.]+\)', 'MX', line)
    noiseless_lines.append(line)

noiseless_str = '\n'.join(noiseless_lines)

# ============================================================================
# Identify T-gate blocks (S/S_DAG lines that we'd replace with T)
# ============================================================================
print("Identifying S/S_DAG blocks (= T gate proxy lines):")
print("="*60)

s_lines = []  # (line_number, original_text, gate_type, qubits)
for i, line in enumerate(noiseless_str.split('\n')):
    stripped = line.strip()
    if stripped.startswith('S_DAG ') and not stripped.startswith('S_DAG['):
        qubits = stripped.split()[1:]
        s_lines.append((i, line, 'S_DAG', qubits))
        print(f"  Line {i}: {stripped}  ({len(qubits)} qubits)")
    elif stripped.startswith('S ') and not stripped.startswith('S['):
        qubits = stripped.split()[1:]
        s_lines.append((i, line, 'S', qubits))
        print(f"  Line {i}: {stripped}  ({len(qubits)} qubits)")

# Group into blocks:
# Block 0 (injection): S_DAG 3 (line ~85 = 1 qubit)
# Block 1a (first cult T†): S_DAG 0 3 7 9 11 13 17 (7 qubits)
# Block 1b (first cult T): S 0 3 7 9 11 13 17 (7 qubits)
# Block 2a (second cult T†): S_DAG 0 3 ... 38 40 (19 qubits)
# Block 2b (second cult T): S 0 3 ... 38 40 (19 qubits)

print(f"\nFound {len(s_lines)} S/S_DAG lines")

# Classify into injection + 2 cultivation blocks
injection = s_lines[0]  # S_DAG 3 (1 qubit)
block1_tdagger = s_lines[1]  # S_DAG on 7 qubits
block1_t = s_lines[2]  # S on 7 qubits
block2_tdagger = s_lines[3]  # S_DAG on 19 qubits
block2_t = s_lines[4]  # S on 19 qubits

print(f"\nBlock structure:")
print(f"  Injection: {injection[2]} on {len(injection[3])} qubit(s) (line {injection[0]})")
print(f"  Block 1 T†: {block1_tdagger[2]} on {len(block1_tdagger[3])} qubits (line {block1_tdagger[0]})")
print(f"  Block 1 T:  {block1_t[2]} on {len(block1_t[3])} qubits (line {block1_t[0]})")
print(f"  Block 2 T†: {block2_tdagger[2]} on {len(block2_tdagger[3])} qubits (line {block2_tdagger[0]})")
print(f"  Block 2 T:  {block2_t[2]} on {len(block2_t[3])} qubits (line {block2_t[0]})")

# ============================================================================
# Construct 4 circuit variants
# ============================================================================
print(f"\n{'='*60}")
print("Constructing 4 Clifford circuit variants")
print(f"{'='*60}")

import tsim

def make_variant(noiseless_str, block1_sub, block2_sub, label):
    """Create a variant circuit with specified substitutions.

    block1_sub: 'remove' or 'Z' for block 1 (injection + first cultivation)
    block2_sub: 'remove' or 'Z' for block 2 (second cultivation)
    """
    source_lines = noiseless_str.split('\n')

    # Lines to modify (by line index in noiseless_str)
    injection_idx = injection[0]
    b1_td_idx = block1_tdagger[0]
    b1_t_idx = block1_t[0]
    b2_td_idx = block2_tdagger[0]
    b2_t_idx = block2_t[0]

    result_lines = []
    for i, line in enumerate(source_lines):
        if i in (injection_idx, b1_td_idx, b1_t_idx):
            # Block 1 (injection + first cultivation)
            if block1_sub == 'remove':
                continue  # Remove line entirely
            elif block1_sub == 'Z':
                # Replace S/S_DAG with Z
                stripped = line.strip()
                qubits = stripped.split()[1:]
                result_lines.append(f"Z {' '.join(qubits)}")
            else:
                result_lines.append(line)
        elif i in (b2_td_idx, b2_t_idx):
            # Block 2 (second cultivation)
            if block2_sub == 'remove':
                continue
            elif block2_sub == 'Z':
                stripped = line.strip()
                qubits = stripped.split()[1:]
                result_lines.append(f"Z {' '.join(qubits)}")
            else:
                result_lines.append(line)
        else:
            result_lines.append(line)

    variant_str = '\n'.join(result_lines)
    return variant_str

variants = [
    ('remove', 'remove', 'C1: Remove all T gates'),
    ('remove', 'Z',      'C2: Remove block1, Z block2'),
    ('Z',      'remove', 'C3: Z block1, remove block2'),
    ('Z',      'Z',      'C4: Z both blocks'),
]

for block1_sub, block2_sub, label in variants:
    variant_str = make_variant(noiseless_str, block1_sub, block2_sub, label)

    # Parse as tsim Circuit (S gates stay as S = Clifford, Z gates are Clifford)
    # The variant should have NO T gates
    try:
        circuit = tsim.Circuit(variant_str)
        graph = circuit.get_graph()
        tc = tcount(graph)

        # Also reduce
        g = deepcopy(graph)
        zx.full_reduce(g, paramSafe=True)
        tc_reduced = tcount(g)
        nv = len(list(g.vertices()))

        print(f"\n{label}:")
        print(f"  T-count (raw): {tc}")
        print(f"  T-count (reduced): {tc_reduced}")
        print(f"  Vertices (reduced): {nv}")

        if tc == 0:
            print(f"  ✓ CLIFFORD (T-count=0)")
        else:
            print(f"  ✗ NOT Clifford (T-count={tc})")

    except Exception as e:
        print(f"\n{label}: ERROR - {e}")

# ============================================================================
# Also verify original circuit T-count
# ============================================================================
print(f"\n{'='*60}")
print("Original circuit (with T gates):")

def replace_s_with_t(s):
    s = re.sub(r'^(\s*)S_DAG(\s)', r'\1T_DAG\2', s, flags=re.MULTILINE)
    return re.sub(r'^(\s*)S(\s)', r'\1T\2', s, flags=re.MULTILINE)

noiseless_t_str = replace_s_with_t(noiseless_str)
circuit_orig = tsim.Circuit(noiseless_t_str)
graph_orig = circuit_orig.get_graph()
tc_orig = tcount(graph_orig)
g_orig = deepcopy(graph_orig)
zx.full_reduce(g_orig, paramSafe=True)
tc_orig_reduced = tcount(g_orig)

print(f"  T-count (raw): {tc_orig}")
print(f"  T-count (reduced): {tc_orig_reduced}")

# ============================================================================
# Now test: does the weighted sum of the 4 Clifford circuits
# equal the original circuit?
# ============================================================================
print(f"\n{'='*60}")
print("Verifying: weighted sum of 4 Clifford circuits = original")
print(f"{'='*60}")

# For a small test, compute the tensor (matrix) of each variant
# and verify the sum matches the original.
# But d=5 has 42 qubits - WAY too big for tensor computation.
# So let's verify on a small toy example instead.

print("\nTesting on toy circuit: T on single qubit")
print("-"*40)

import numpy as np
import os
_PRX_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

# Toy: H, T, H, M on 1 qubit
toy_t = tsim.Circuit("H 0\nT 0\nH 0")
toy_remove = tsim.Circuit("H 0\nH 0")  # T → removed
toy_z = tsim.Circuit("H 0\nZ 0\nH 0")  # T → Z

# Get tensors
tensor_t = toy_t.to_tensor()
tensor_remove = toy_remove.to_tensor()
tensor_z = toy_z.to_tensor()

# T = (I + e^{iπ/4} Z) / √2
# So: circuit_with_T = (1/√2) * circuit_without_T + (e^{iπ/4}/√2) * circuit_with_Z
coeff_remove = 1 / np.sqrt(2)
coeff_z = np.exp(1j * np.pi / 4) / np.sqrt(2)

tensor_sum = coeff_remove * tensor_remove + coeff_z * tensor_z

print(f"Original T tensor:\n{tensor_t}")
print(f"Weighted sum (I + e^{{iπ/4}}Z)/√2:\n{tensor_sum}")
print(f"Match: {np.allclose(tensor_t, tensor_sum)}")

# Test T† too
print("\nTesting on toy circuit: T† on single qubit")
print("-"*40)

toy_td = tsim.Circuit("H 0\nT_DAG 0\nH 0")
tensor_td = toy_td.to_tensor()

coeff_remove_dag = 1 / np.sqrt(2)
coeff_z_dag = np.exp(-1j * np.pi / 4) / np.sqrt(2)

tensor_sum_dag = coeff_remove_dag * tensor_remove + coeff_z_dag * tensor_z

print(f"Original T† tensor:\n{tensor_td}")
print(f"Weighted sum (I + e^{{-iπ/4}}Z)/√2:\n{tensor_sum_dag}")
print(f"Match: {np.allclose(tensor_td, tensor_sum_dag)}")

# Test conjugation T†·H·T
print("\nTesting conjugation: T†·H·T on single qubit")
print("-"*40)

toy_conj = tsim.Circuit("T_DAG 0\nH 0\nT 0")
tensor_conj = toy_conj.to_tensor()

# T†·H·T decomposition:
# Each T/T† gives 2 terms (I or Z), so 4 terms total
# T† = (I + e^{-iπ/4}Z)/√2, T = (I + e^{iπ/4}Z)/√2

toy_II = tsim.Circuit("H 0")  # Both T→I
toy_IZ = tsim.Circuit("H 0\nZ 0")  # T†→I, T→Z
toy_ZI = tsim.Circuit("Z 0\nH 0")  # T†→Z, T→I
toy_ZZ = tsim.Circuit("Z 0\nH 0\nZ 0")  # Both T→Z

t_II = toy_II.to_tensor()
t_IZ = toy_IZ.to_tensor()
t_ZI = toy_ZI.to_tensor()
t_ZZ = toy_ZZ.to_tensor()

# Coefficients: (1/√2)^2 × phase factors
# (s,t) = (0,0): (1/2) × 1 × 1 = 1/2
# (s,t) = (0,1): (1/2) × 1 × e^{iπ/4} = e^{iπ/4}/2
# (s,t) = (1,0): (1/2) × e^{-iπ/4} × 1 = e^{-iπ/4}/2
# (s,t) = (1,1): (1/2) × e^{-iπ/4} × e^{iπ/4} = 1/2

tensor_sum4 = (1/2) * t_II + (np.exp(1j*np.pi/4)/2) * t_IZ + (np.exp(-1j*np.pi/4)/2) * t_ZI + (1/2) * t_ZZ

print(f"Original T†·H·T:\n{tensor_conj}")
print(f"Sum of 4 terms:\n{tensor_sum4}")
print(f"Match: {np.allclose(tensor_conj, tensor_sum4)}")
print(f"# non-zero terms: 4 (all survive for generic H)")

# Now test with CNOT (GHZ-like structure)
print("\nTesting conjugation: T†⊗2 · CX · T⊗2 on 2 qubits")
print("-"*40)

toy_conj2 = tsim.Circuit("T_DAG 0\nT_DAG 1\nCX 0 1\nT 0\nT 1")
tensor_conj2 = toy_conj2.to_tensor()

# 4 substitution patterns for 2 qubits: (s,t) where s,t ∈ {0,1}^2
# Full expansion: 16 terms. But let's check which survive.

patterns = []
for s0 in [0, 1]:
    for s1 in [0, 1]:
        for t0 in [0, 1]:
            for t1 in [0, 1]:
                # Build circuit: Z^s · CX · Z^t
                parts = []
                if s0: parts.append("Z 0")
                if s1: parts.append("Z 1")
                parts.append("CX 0 1")
                if t0: parts.append("Z 0")
                if t1: parts.append("Z 1")

                circ_str = "\n".join(parts) if parts else "CX 0 1"
                circ = tsim.Circuit(circ_str)
                tensor = circ.to_tensor()

                # Coefficient
                s_sum = s0 + s1
                t_sum = t0 + t1
                coeff = (1/4) * np.exp(-1j*np.pi*s_sum/4) * np.exp(1j*np.pi*t_sum/4)

                patterns.append({
                    's': (s0, s1), 't': (t0, t1),
                    'coeff': coeff, 'tensor': tensor
                })

tensor_sum16 = sum(p['coeff'] * p['tensor'] for p in patterns)
print(f"Match (16-term sum): {np.allclose(tensor_conj2, tensor_sum16)}")

# Check which patterns have matching s,t (i.e., s=t)
print(f"\nPatterns with s=t only:")
tensor_sum_st = sum(
    p['coeff'] * p['tensor'] for p in patterns if p['s'] == p['t']
)
print(f"Match (s=t only): {np.allclose(tensor_conj2, tensor_sum_st)}")

# Check s=(0,0),t=(0,0) and s=(1,1),t=(1,1) only
tensor_sum_2 = sum(
    p['coeff'] * p['tensor'] for p in patterns
    if (p['s'] == (0,0) and p['t'] == (0,0)) or
       (p['s'] == (1,1) and p['t'] == (1,1))
)
print(f"Match (only 00↔00, 11↔11): {np.allclose(tensor_conj2, tensor_sum_2)}")

# Show all non-negligible terms
print(f"\nAll 16 terms with coefficients:")
for p in patterns:
    norm = abs(p['coeff'])
    if norm > 1e-10:
        print(f"  s={p['s']}, t={p['t']}: |coeff|={norm:.4f}")
