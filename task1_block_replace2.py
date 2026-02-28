"""Task 1: Replace full cultivation blocks with Clifford operations.

Key fix: Apply S→T conversion FIRST, then replace T-containing blocks
with Clifford operations (S_DAG is Clifford, T_DAG is not).
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

# Apply S→T conversion to get circuit with ACTUAL T gates
def replace_s_with_t(s):
    s = re.sub(r'^(\s*)S_DAG(\s)', r'\1T_DAG\2', s, flags=re.MULTILINE)
    return re.sub(r'^(\s*)S(\s)', r'\1T\2', s, flags=re.MULTILINE)

t_lines = replace_s_with_t('\n'.join(noiseless)).split('\n')

# Identify blocks in T-converted circuit
# Block 1: lines 105-127 (T_DAG→cascade→MX→RX→cascade→T on 7 qubits)
# Block 2: lines 238-272 (T_DAG→cascade→MX→RX→cascade→T on 19 qubits)
# Injection: line 63 (T_DAG 3)

block1_start, block1_end = 105, 127
block1_data = [0, 3, 7, 9, 11, 13, 17]

block2_start, block2_end = 238, 272
block2_data = [0, 3, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 32, 34, 36, 38, 40]

injection_line = 63

# Verify block boundaries
print("Block boundary verification:")
print(f"  Block 1 start: {t_lines[block1_start].strip()}")
print(f"  Block 1 end:   {t_lines[block1_end].strip()}")
print(f"  Block 2 start: {t_lines[block2_start].strip()}")
print(f"  Block 2 end:   {t_lines[block2_end].strip()}")
print(f"  Injection:     {t_lines[injection_line].strip()}")


def make_xsd_clifford(data_qubits):
    """Generate Clifford lines for (XS†)^⊗n. Uses S_DAG (truly Clifford)."""
    result = []
    # XS† = X · S† so apply S† first, then X
    result.append(f"S_DAG {' '.join(str(q) for q in data_qubits)}")
    result.append("TICK")
    result.append(f"X {' '.join(str(q) for q in data_qubits)}")
    result.append("TICK")
    return result


def build_variant(t_lines, block1_mode, block2_mode, inject_mode="keep"):
    """Build a circuit variant from the T-converted lines.

    Modes: "keep" = original, "identity" = remove block, "xsd" = replace with XS†
    """
    result = []

    for i, line in enumerate(t_lines):
        # Handle injection
        if i == injection_line:
            if inject_mode == "identity":
                result.append("TICK")
                continue
            elif inject_mode == "xsd":
                result.append("S_DAG 3")
                result.append("X 3")
                result.append("TICK")
                continue

        # Handle block 1
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

        # Handle block 2
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
# Build and test 4 variants (injection kept as T)
# ============================================================================
print(f"\n{'='*60}")
print("4 variants: cultivation blocks → {I, XS†}")
print("Injection T_DAG 3 kept as-is")
print(f"{'='*60}")

variants_4 = [
    ("I ⊗ I", "identity", "identity"),
    ("I ⊗ XS†", "identity", "xsd"),
    ("XS† ⊗ I", "xsd", "identity"),
    ("XS† ⊗ XS†", "xsd", "xsd"),
]

for name, b1, b2 in variants_4:
    circ_text = build_variant(t_lines, b1, b2, inject_mode="keep")

    # Count T/T_DAG gates in the text
    t_count_text = sum(1 for l in circ_text.split('\n')
                       if l.strip().startswith('T_DAG ') or l.strip().startswith('T '))

    circuit = tsim.Circuit(circ_text)
    graph = circuit.get_graph()
    tc_raw = tcount(graph)

    g = deepcopy(graph)
    zx.full_reduce(g, paramSafe=True)
    tc_red = tcount(g)
    nv = len(list(g.vertices()))

    print(f"  ({name}): T-gates={t_count_text}, ZX raw T={tc_raw}, "
          f"reduced T={tc_red}, verts={nv}")


# ============================================================================
# 8 variants (injection also replaced)
# ============================================================================
print(f"\n{'='*60}")
print("8 variants: injection + 2 cultivation blocks → {I, XS†}")
print(f"{'='*60}")

variants_8 = [
    ("I, I, I", "identity", "identity", "identity"),
    ("I, I, XS†", "identity", "identity", "xsd"),
    ("I, XS†, I", "identity", "xsd", "identity"),
    ("I, XS†, XS†", "identity", "xsd", "xsd"),
    ("XS†, I, I", "xsd", "identity", "identity"),
    ("XS†, I, XS†", "xsd", "identity", "xsd"),
    ("XS†, XS†, I", "xsd", "xsd", "identity"),
    ("XS†, XS†, XS†", "xsd", "xsd", "xsd"),
]

for name, b1, b2, inj in variants_8:
    circ_text = build_variant(t_lines, b1, b2, inject_mode=inj)

    t_count_text = sum(1 for l in circ_text.split('\n')
                       if l.strip().startswith('T_DAG ') or l.strip().startswith('T '))

    circuit = tsim.Circuit(circ_text)
    graph = circuit.get_graph()
    tc_raw = tcount(graph)

    g = deepcopy(graph)
    zx.full_reduce(g, paramSafe=True)
    tc_red = tcount(g)
    nv = len(list(g.vertices()))

    print(f"  ({name}): T-gates={t_count_text}, ZX T={tc_raw}→{tc_red}, verts={nv}")


# ============================================================================
# Verify original
# ============================================================================
print(f"\n{'='*60}")
print("Original (keep all blocks)")
print(f"{'='*60}")

orig_text = build_variant(t_lines, "keep", "keep", inject_mode="keep")
orig_circuit = tsim.Circuit(orig_text)
orig_graph = orig_circuit.get_graph()
g_orig = deepcopy(orig_graph)
zx.full_reduce(g_orig, paramSafe=True)
print(f"  ZX T={tcount(orig_graph)}→{tcount(g_orig)}, verts={len(list(g_orig.vertices()))}")


# ============================================================================
# Verify: check if any XS† lines got mistakenly converted to T
# ============================================================================
print(f"\n{'='*60}")
print("Sanity check: S_DAG lines in variant (XS†,XS†,XS†)")
print(f"{'='*60}")

test_text = build_variant(t_lines, "xsd", "xsd", inject_mode="xsd")
for i, line in enumerate(test_text.split('\n')):
    s = line.strip()
    if s.startswith('S_DAG') or s.startswith('S '):
        print(f"  Line {i}: {s}   <-- Clifford S_DAG (good)")
    if s.startswith('T_DAG') or (s.startswith('T ') and not s.startswith('TICK')):
        print(f"  Line {i}: {s}   <-- T gate (should not exist)")
