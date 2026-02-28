"""Task 1: Correct Appendix C decomposition via spider cutting identity.

Spider cutting identity: T = (1/√2)(I + e^{iπ/4} Z)
                          T† = (1/√2)(I + e^{-iπ/4} Z)

For each cultivation block: replace T_DAG^⊗n with {I^⊗n, Z^⊗n}
                            replace T^⊗n with {I^⊗n, Z^⊗n}
Keep the cascade + MX + RX structure intact.

4 terms per block:
  (I,I): coeff = (1/2)^n
  (I,Z): coeff = (1/2)^n * e^{inπ/4}
  (Z,I): coeff = (1/2)^n * e^{-inπ/4}
  (Z,Z): coeff = (1/2)^n

Appendix C says 2 of 4 cancel → 2 surviving terms per block.
For 2 blocks: 2×2 = 4 total Clifford terms.
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

# ============================================================================
# Load d=5 circuit
# ============================================================================
CIRCUIT_PATH = Path(os.path.join(_PRX_ROOT, "gidney-circuits", "circuits", ""
                    "for_perfectionist_decoding/"
                    "c=inject[unitary]+cultivate,p=0.001,noise=uniform,"
                    "g=css,q=42,b=Y,r=10,d1=5.stim")
circuit_str = CIRCUIT_PATH.read_text()

lines = circuit_str.split('\n')
noiseless = []
for line in lines:
    s = line.strip()
    if any(s.startswith(p) for p in ['X_ERROR', 'Z_ERROR', 'DEPOLARIZE1', 'DEPOLARIZE2']):
        continue
    line = re.sub(r'M\([\d.]+\)', 'M', line)
    line = re.sub(r'MX\([\d.]+\)', 'MX', line)
    noiseless.append(line)

# Apply S→T to get circuit with actual T gates
def replace_s_with_t(s):
    s = re.sub(r'^(\s*)S_DAG(\s)', r'\1T_DAG\2', s, flags=re.MULTILINE)
    return re.sub(r'^(\s*)S(\s)', r'\1T\2', s, flags=re.MULTILINE)

t_text = replace_s_with_t('\n'.join(noiseless))
t_lines = t_text.split('\n')

# Block positions (from earlier analysis):
# Injection: line 63: T_DAG 3
# Block 1: line 105: T_DAG 0 3 7 9 11 13 17
#           line 127: T 0 3 7 9 11 13 17
# Block 2: line 238: T_DAG 0 3 7 9 11 13 15 17 19 21 23 25 27 29 32 34 36 38 40
#           line 272: T 0 3 7 9 11 13 15 17 19 21 23 25 27 29 32 34 36 38 40

injection_line = 63
b1_tdag_line, b1_t_line = 105, 127
b1_qubits = [0, 3, 7, 9, 11, 13, 17]

b2_tdag_line, b2_t_line = 238, 272
b2_qubits = [0, 3, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 32, 34, 36, 38, 40]

# Verify
print("Verifying block positions:")
print(f"  Injection: {t_lines[injection_line].strip()}")
print(f"  Block 1 T†: {t_lines[b1_tdag_line].strip()}")
print(f"  Block 1 T:  {t_lines[b1_t_line].strip()}")
print(f"  Block 2 T†: {t_lines[b2_tdag_line].strip()}")
print(f"  Block 2 T:  {t_lines[b2_t_line].strip()}")


def build_variant(t_lines, b1_tdag_mode, b1_t_mode, b2_tdag_mode, b2_t_mode, inj_mode="keep"):
    """Build a circuit variant by replacing T/T_DAG with I or Z.

    Modes: "keep" = original T/T_DAG
           "I" = remove (identity)
           "Z" = replace with Z gates
    Keep ALL other circuit structure (cascade, MX, RX, etc.) intact.
    """
    result = []
    for i, line in enumerate(t_lines):
        s = line.strip()

        # Injection
        if i == injection_line:
            if inj_mode == "I":
                result.append("TICK")  # Remove T_DAG
                continue
            elif inj_mode == "Z":
                result.append("Z 3")
                continue

        # Block 1 T†
        if i == b1_tdag_line:
            if b1_tdag_mode == "I":
                result.append("TICK")
                continue
            elif b1_tdag_mode == "Z":
                result.append(f"Z {' '.join(str(q) for q in b1_qubits)}")
                continue

        # Block 1 T
        if i == b1_t_line:
            if b1_t_mode == "I":
                result.append("TICK")
                continue
            elif b1_t_mode == "Z":
                result.append(f"Z {' '.join(str(q) for q in b1_qubits)}")
                continue

        # Block 2 T†
        if i == b2_tdag_line:
            if b2_tdag_mode == "I":
                result.append("TICK")
                continue
            elif b2_tdag_mode == "Z":
                result.append(f"Z {' '.join(str(q) for q in b2_qubits)}")
                continue

        # Block 2 T
        if i == b2_t_line:
            if b2_t_mode == "I":
                result.append("TICK")
                continue
            elif b2_t_mode == "Z":
                result.append(f"Z {' '.join(str(q) for q in b2_qubits)}")
                continue

        result.append(line)

    return '\n'.join(result)


# ============================================================================
# Test all 4 variants per block (keeping injection as-is for now)
# ============================================================================
print(f"\n{'='*60}")
print("Block 1: 4 variants (T†→{I,Z} × T→{I,Z})")
print("Block 2: kept as original T/T_DAG")
print(f"{'='*60}")

n1 = len(b1_qubits)
for tdag_m, t_m in [("I","I"), ("I","Z"), ("Z","I"), ("Z","Z")]:
    text = build_variant(t_lines, tdag_m, t_m, "keep", "keep", "keep")
    circ = tsim.Circuit(text)
    g = circ.get_graph()
    g2 = deepcopy(g)
    zx.full_reduce(g2, paramSafe=True)
    tc = tcount(g2)
    is_zero = g2.scalar.is_zero
    nv = len(list(g2.vertices()))

    # Coefficient
    if tdag_m == "I" and t_m == "I":
        coeff_str = f"(1/2)^{n1} = {0.5**n1:.6f}"
    elif tdag_m == "I" and t_m == "Z":
        coeff_str = f"(1/2)^{n1} * e^(i{n1}π/4)"
    elif tdag_m == "Z" and t_m == "I":
        coeff_str = f"(1/2)^{n1} * e^(-i{n1}π/4)"
    else:
        coeff_str = f"(1/2)^{n1}"

    print(f"  (T†→{tdag_m}, T→{t_m}): T-count={tc}, zero={is_zero}, "
          f"verts={nv}, coeff={coeff_str}")

# ============================================================================
# Test all 4 variants per block for Block 2
# ============================================================================
print(f"\n{'='*60}")
print("Block 2: 4 variants (T†→{I,Z} × T→{I,Z})")
print("Block 1: kept as original T/T_DAG")
print(f"{'='*60}")

n2 = len(b2_qubits)
for tdag_m, t_m in [("I","I"), ("I","Z"), ("Z","I"), ("Z","Z")]:
    text = build_variant(t_lines, "keep", "keep", tdag_m, t_m, "keep")
    circ = tsim.Circuit(text)
    g = circ.get_graph()
    g2 = deepcopy(g)
    zx.full_reduce(g2, paramSafe=True)
    tc = tcount(g2)
    is_zero = g2.scalar.is_zero
    nv = len(list(g2.vertices()))

    print(f"  (T†→{tdag_m}, T→{t_m}): T-count={tc}, zero={is_zero}, verts={nv}")

# ============================================================================
# Test ALL 16 combinations (4 per block × 4 per block), ignoring injection
# ============================================================================
print(f"\n{'='*60}")
print("All 16 combinations (block1 × block2)")
print("Injection kept as T_DAG")
print(f"{'='*60}")

modes = ["I", "Z"]
results = []

for b1_tdag in modes:
    for b1_t in modes:
        for b2_tdag in modes:
            for b2_t in modes:
                text = build_variant(t_lines, b1_tdag, b1_t, b2_tdag, b2_t, "keep")
                circ = tsim.Circuit(text)
                g = circ.get_graph()
                g2 = deepcopy(g)
                zx.full_reduce(g2, paramSafe=True)
                tc = tcount(g2)
                is_zero = g2.scalar.is_zero

                label = f"({b1_tdag},{b1_t})×({b2_tdag},{b2_t})"
                status = "CLIFFORD" if tc == 0 else f"T={tc}"
                if is_zero:
                    status = "ZERO"

                results.append({
                    'label': label, 'tc': tc, 'zero': is_zero,
                    'b1_tdag': b1_tdag, 'b1_t': b1_t,
                    'b2_tdag': b2_tdag, 'b2_t': b2_t,
                })

                print(f"  {label}: {status}")

# Summarize
n_clifford = sum(1 for r in results if r['tc'] == 0 and not r['zero'])
n_zero = sum(1 for r in results if r['zero'])
n_noncliff = sum(1 for r in results if r['tc'] > 0 and not r['zero'])

print(f"\nSummary: {n_clifford} Clifford, {n_zero} zero-scalar, {n_noncliff} non-Clifford")

# ============================================================================
# Also test with injection replaced
# ============================================================================
print(f"\n{'='*60}")
print("Same 16 combinations but injection T_DAG→I")
print(f"{'='*60}")

for b1_tdag in modes:
    for b1_t in modes:
        for b2_tdag in modes:
            for b2_t in modes:
                text = build_variant(t_lines, b1_tdag, b1_t, b2_tdag, b2_t, "I")
                circ = tsim.Circuit(text)
                g = circ.get_graph()
                g2 = deepcopy(g)
                zx.full_reduce(g2, paramSafe=True)
                tc = tcount(g2)
                is_zero = g2.scalar.is_zero

                label = f"({b1_tdag},{b1_t})×({b2_tdag},{b2_t})"
                status = "CLIFFORD" if tc == 0 else f"T={tc}"
                if is_zero:
                    status = "ZERO"

                print(f"  inj→I, {label}: {status}")
