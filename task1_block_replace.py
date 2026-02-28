"""Task 1: Replace full cultivation blocks with I or XS† to get Clifford circuits.

The cultivation block structure is:
  S_DAG data_qubits       (= T†)
  CNOT cascade in
  MX ancilla
  RX ancilla
  CNOT cascade out (reverse)
  S data_qubits            (= T)

Appendix C says: T†^⊗n · C_check · T^⊗n ∝ I^⊗n + (XS†)^⊗n

So each full block gets replaced by EITHER:
  - Identity (remove entire block)
  - X; S_DAG on each data qubit (= XS† on each data qubit)

For 2 cultivation blocks: 2² = 4 Clifford variants.
"""
import sys, os, re
from pathlib import Path
from copy import deepcopy
from fractions import Fraction

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

# Identify cultivation blocks
# Block 1: lines 105-127 (S_DAG on 7 qubits, cascade, MX, RX, cascade, S)
# Block 2: lines 238-272 (S_DAG on 19 qubits, cascade, MX, RX, cascade, S)
# Also: injection at line 63 (S_DAG 3, single T gate)

block1_start, block1_end = 105, 127  # inclusive
block1_data_qubits = [0, 3, 7, 9, 11, 13, 17]

block2_start, block2_end = 238, 272  # inclusive
block2_data_qubits = [0, 3, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 32, 34, 36, 38, 40]

injection_line = 63  # S_DAG 3

print(f"Noiseless circuit: {len(noiseless)} lines")
print(f"Block 1 (cultivation 1): lines {block1_start}-{block1_end}, data qubits: {block1_data_qubits}")
print(f"Block 2 (cultivation 2): lines {block2_start}-{block2_end}, data qubits: {block2_data_qubits}")
print(f"Injection: line {injection_line}")

# Print the blocks for verification
print(f"\n--- Block 1 ---")
for i in range(block1_start, block1_end + 1):
    print(f"  {i}: {noiseless[i].strip()}")

print(f"\n--- Block 2 ---")
for i in range(block2_start, block2_end + 1):
    print(f"  {i}: {noiseless[i].strip()}")

print(f"\n--- Injection ---")
print(f"  {injection_line}: {noiseless[injection_line].strip()}")


def replace_s_with_t(s):
    """Replace S/S_DAG with T/T_DAG."""
    s = re.sub(r'^(\s*)S_DAG(\s)', r'\1T_DAG\2', s, flags=re.MULTILINE)
    return re.sub(r'^(\s*)S(\s)', r'\1T\2', s, flags=re.MULTILINE)


def make_xsd_lines(data_qubits):
    """Generate circuit lines for (XS†)^⊗n on data_qubits."""
    result = []
    for q in data_qubits:
        result.append(f"S_DAG {q}")  # S† stays as S_DAG (Clifford)
    result.append("TICK")
    for q in data_qubits:
        result.append(f"X {q}")
    result.append("TICK")
    return result


def build_variant(noiseless_lines, block1_mode, block2_mode, inject_mode="keep"):
    """Build a circuit variant.

    block_mode can be:
      "keep" - keep the original block (with S→T substitution)
      "identity" - replace entire block with nothing
      "xsd" - replace entire block with X; S_DAG on each data qubit

    inject_mode can be:
      "keep" - keep the S_DAG 3 injection (becomes T_DAG)
      "identity" - remove it
      "xsd" - replace with X 3; S_DAG 3
    """
    result = []

    for i, line in enumerate(noiseless_lines):
        # Handle injection
        if i == injection_line:
            if inject_mode == "identity":
                result.append("TICK")  # placeholder
                continue
            elif inject_mode == "xsd":
                result.append("S_DAG 3")
                result.append("X 3")
                result.append("TICK")
                continue
            # else "keep" - fall through

        # Handle block 1
        if block1_start <= i <= block1_end:
            if i == block1_start:  # First line of block
                if block1_mode == "identity":
                    result.append("TICK")  # placeholder
                elif block1_mode == "xsd":
                    result.extend(make_xsd_lines(block1_data_qubits))
                else:  # "keep"
                    result.append(line)
            elif block1_mode != "keep":
                continue  # Skip rest of block
            else:
                result.append(line)
            continue

        # Handle block 2
        if block2_start <= i <= block2_end:
            if i == block2_start:  # First line of block
                if block2_mode == "identity":
                    result.append("TICK")
                elif block2_mode == "xsd":
                    result.extend(make_xsd_lines(block2_data_qubits))
                else:  # "keep"
                    result.append(line)
            elif block2_mode != "keep":
                continue  # Skip rest of block
            else:
                result.append(line)
            continue

        result.append(line)

    return '\n'.join(result)


# ============================================================================
# Build and test all 4 variants (injection kept as T)
# ============================================================================
print(f"\n{'='*60}")
print("Testing 4 variants (2 cultivation blocks × {I, XS†})")
print("Injection S_DAG 3 kept as-is (will become T_DAG after S→T)")
print(f"{'='*60}")

variants = [
    ("I ⊗ I", "identity", "identity"),
    ("I ⊗ XS†", "identity", "xsd"),
    ("XS† ⊗ I", "xsd", "identity"),
    ("XS† ⊗ XS†", "xsd", "xsd"),
]

for name, b1_mode, b2_mode in variants:
    circ_text = build_variant(noiseless, b1_mode, b2_mode, inject_mode="keep")
    circ_t = replace_s_with_t(circ_text)
    circuit = tsim.Circuit(circ_t)
    graph = circuit.get_graph()

    tc_raw = tcount(graph)
    nv_raw = len(list(graph.vertices()))

    g = deepcopy(graph)
    zx.full_reduce(g, paramSafe=True)
    tc_red = tcount(g)
    nv_red = len(list(g.vertices()))

    print(f"\nVariant ({name}):")
    print(f"  Raw: {nv_raw} verts, T-count={tc_raw}")
    print(f"  Reduced: {nv_red} verts, T-count={tc_red}")


# ============================================================================
# Also test with injection removed/replaced
# ============================================================================
print(f"\n{'='*60}")
print("Testing with injection also replaced (8 variants)")
print(f"{'='*60}")

all_variants = [
    ("I, I, I", "identity", "identity", "identity"),
    ("I, I, XS†", "identity", "identity", "xsd"),
    ("I, XS†, I", "identity", "xsd", "identity"),
    ("I, XS†, XS†", "identity", "xsd", "xsd"),
    ("XS†, I, I", "xsd", "identity", "identity"),
    ("XS†, I, XS†", "xsd", "identity", "xsd"),
    ("XS†, XS†, I", "xsd", "xsd", "identity"),
    ("XS†, XS†, XS†", "xsd", "xsd", "xsd"),
]

for name, b1_mode, b2_mode, inj_mode in all_variants:
    circ_text = build_variant(noiseless, b1_mode, b2_mode, inject_mode=inj_mode)
    circ_t = replace_s_with_t(circ_text)
    circuit = tsim.Circuit(circ_t)
    graph = circuit.get_graph()

    tc_raw = tcount(graph)

    g = deepcopy(graph)
    zx.full_reduce(g, paramSafe=True)
    tc_red = tcount(g)
    nv_red = len(list(g.vertices()))

    print(f"  ({name}): raw T={tc_raw}, reduced T={tc_red}, verts={nv_red}")


# ============================================================================
# Verify: original circuit T-count
# ============================================================================
print(f"\n{'='*60}")
print("Original circuit for reference")
print(f"{'='*60}")

orig_t = replace_s_with_t('\n'.join(noiseless))
orig_circuit = tsim.Circuit(orig_t)
orig_graph = orig_circuit.get_graph()
tc_orig_raw = tcount(orig_graph)

g_orig = deepcopy(orig_graph)
zx.full_reduce(g_orig, paramSafe=True)
tc_orig_red = tcount(g_orig)
nv_orig = len(list(g_orig.vertices()))

print(f"  Raw T-count: {tc_orig_raw}")
print(f"  Reduced: T-count={tc_orig_red}, verts={nv_orig}")
