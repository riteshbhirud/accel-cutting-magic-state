"""Task 1: Validate on d=3 using tsim's sampling (not full tensor).

Use tsim to sample from the original d=3 circuit,
and stim to simulate the Clifford variant circuits.
Compare measurement statistics.
"""
import sys, os, re
from pathlib import Path
from copy import deepcopy
from collections import Counter
import numpy as np

os.environ["JAX_PLATFORMS"] = "cpu"

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))

import tsim
import pyzx_param as zx
from stab_rank_cut import tcount
import stim

# Load d=3 circuit
CIRCUIT_PATH = Path(os.path.join(_PRX_ROOT, "gidney-circuits", "circuits", ""
                    "for_perfectionist_decoding/"
                    "c=inject[unitary]+cultivate,p=0.001,noise=uniform,"
                    "g=css,q=15,b=Y,r=4,d1=3.stim")

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

# Find T blocks
print("S/S_DAG lines in d=3 noiseless circuit:")
s_lines_found = []
for i, line in enumerate(noiseless):
    s = line.strip()
    if s.startswith('S_DAG') or s.startswith('S '):
        s_lines_found.append((i, s))
        print(f"  Line {i}: {s}")

# Find MX lines
print("\nMX lines:")
mx_lines = []
for i, line in enumerate(noiseless):
    s = line.strip()
    if s.startswith('MX'):
        mx_lines.append((i, s))
        print(f"  Line {i}: {s}")

# Find M lines (final measurements)
print("\nM lines:")
m_lines = []
for i, line in enumerate(noiseless):
    s = line.strip()
    if s.startswith('M ') or (s == 'M' and i == len(noiseless)-1):
        m_lines.append((i, s))
        print(f"  Line {i}: {s}")

print(f"\nTotal noiseless lines: {len(noiseless)}")

# Apply S→T and check ZX
def replace_s_with_t(s):
    s = re.sub(r'^(\s*)S_DAG(\s)', r'\1T_DAG\2', s, flags=re.MULTILINE)
    return re.sub(r'^(\s*)S(\s)', r'\1T\2', s, flags=re.MULTILINE)

t_text = replace_s_with_t('\n'.join(noiseless))
t_lines = t_text.split('\n')

circuit = tsim.Circuit(t_text)
g = circuit.get_graph()
g2 = deepcopy(g)
zx.full_reduce(g2, paramSafe=True)
print(f"\nOriginal d=3 circuit: {circuit.num_qubits} qubits, "
      f"ZX T-count raw={tcount(g)}, reduced={tcount(g2)}")

# Identify cultivation block(s) in d=3
# Pattern: S_DAG (injection) → Clifford ops → S_DAG (cultivation T†) → cascade → MX → RX → cascade → S (cultivation T)

# For d=3, based on s_lines_found:
# s_lines_found contains the S/S_DAG blocks
# Let's figure out the structure

print(f"\n{'='*60}")
print("Block structure analysis")
print(f"{'='*60}")

# Show ALL lines around S/S_DAG blocks
for line_num, _ in s_lines_found:
    start = max(0, line_num - 2)
    end = min(len(noiseless), line_num + 3)
    print(f"\n  Context around line {line_num}:")
    for i in range(start, end):
        marker = " >>>" if i == line_num else "    "
        print(f"  {marker} {i}: {noiseless[i].strip()}")

# Identify blocks by pairing S_DAG with subsequent S on same qubits
print(f"\n{'='*60}")
print("Cultivation blocks (paired S_DAG/S)")
print(f"{'='*60}")

blocks = []
used = set()
for i, (ln1, s1) in enumerate(s_lines_found):
    if i in used or not s1.startswith('S_DAG'):
        continue
    parts1 = s1.split()
    qubits1 = set(parts1[1:])

    # Find matching S line with same qubits
    for j, (ln2, s2) in enumerate(s_lines_found):
        if j in used or not s2.startswith('S '):
            continue
        parts2 = s2.split()
        qubits2 = set(parts2[1:])
        if qubits1 == qubits2 and ln2 > ln1:
            blocks.append({
                'start': ln1, 'end': ln2,
                'qubits': sorted([int(q) for q in qubits1]),
                's_dag_line': s1, 's_line': s2,
            })
            used.add(i)
            used.add(j)
            break

# Unpaired S_DAG = injection
for i, (ln, s) in enumerate(s_lines_found):
    if i not in used and s.startswith('S_DAG'):
        parts = s.split()
        blocks.append({
            'start': ln, 'end': ln,
            'qubits': sorted([int(q) for q in parts[1:]]),
            's_dag_line': s, 's_line': None,
            'is_injection': True,
        })

for b in blocks:
    if b.get('is_injection'):
        print(f"  Injection at line {b['start']}: qubits {b['qubits']}")
    else:
        print(f"  Cultivation lines {b['start']}-{b['end']}: "
              f"qubits {b['qubits']} ({len(b['qubits'])} qubits)")

# ============================================================================
# Build variant circuits
# ============================================================================
print(f"\n{'='*60}")
print("Building Clifford variants")
print(f"{'='*60}")

# Find cultivation blocks (those with S_DAG/S pairs)
cult_blocks = [b for b in blocks if not b.get('is_injection')]
inj_blocks = [b for b in blocks if b.get('is_injection')]

print(f"Cultivation blocks: {len(cult_blocks)}")
print(f"Injection blocks: {len(inj_blocks)}")

def make_xsd_lines(data_qubits):
    result = []
    result.append(f"S_DAG {' '.join(str(q) for q in data_qubits)}")
    result.append("TICK")
    result.append(f"X {' '.join(str(q) for q in data_qubits)}")
    result.append("TICK")
    return result

# Build variant by replacing cultivation blocks
def build_variant(t_lines, cult_blocks, modes, inj_blocks=None, inj_modes=None):
    """modes: list of "keep"/"identity"/"xsd" per cult block."""
    skip_ranges = set()
    insertions = {}  # line_num → replacement lines

    for block, mode in zip(cult_blocks, modes):
        if mode == "keep":
            continue
        for i in range(block['start'], block['end'] + 1):
            skip_ranges.add(i)
        if mode == "identity":
            insertions[block['start']] = ["TICK"]
        elif mode == "xsd":
            insertions[block['start']] = make_xsd_lines(block['qubits'])

    if inj_blocks and inj_modes:
        for block, mode in zip(inj_blocks, inj_modes):
            if mode == "keep":
                continue
            skip_ranges.add(block['start'])
            if mode == "identity":
                insertions[block['start']] = ["TICK"]
            elif mode == "xsd":
                insertions[block['start']] = make_xsd_lines(block['qubits'])

    result = []
    for i, line in enumerate(t_lines):
        if i in skip_ranges:
            if i in insertions:
                result.extend(insertions[i])
            continue
        result.append(line)

    return '\n'.join(result)

# For d=3 with 1 cultivation block: 2 variants
for name, mode in [("Identity", "identity"), ("XS†", "xsd"), ("Keep (original)", "keep")]:
    var_text = build_variant(t_lines, cult_blocks, [mode], inj_blocks, ["keep"])
    var_circuit = tsim.Circuit(var_text)
    var_g = var_circuit.get_graph()
    var_g2 = deepcopy(var_g)
    zx.full_reduce(var_g2, paramSafe=True)
    tc = tcount(var_g2)
    print(f"  {name}: {var_circuit.num_qubits} qubits, T-count={tc}")

# ============================================================================
# Sample from Clifford variants using stim
# ============================================================================
print(f"\n{'='*60}")
print("Sampling from Clifford variants (using stim)")
print(f"{'='*60}")

n_samples = 100000

for name, mode in [("Identity", "identity"), ("XS†", "xsd")]:
    var_text = build_variant(t_lines, cult_blocks, [mode], inj_blocks, ["keep"])

    # This still has T_DAG from injection. Check if it's Clifford
    var_circuit = tsim.Circuit(var_text)
    var_g = var_circuit.get_graph()
    var_g2 = deepcopy(var_g)
    zx.full_reduce(var_g2, paramSafe=True)
    tc = tcount(var_g2)

    if tc > 0:
        print(f"  {name}: T-count={tc}, NOT Clifford, need injection replacement too")
        # Also replace injection
        var_text2 = build_variant(t_lines, cult_blocks, [mode],
                                   inj_blocks, ["identity"])
        var_circuit2 = tsim.Circuit(var_text2)
        var_g2b = deepcopy(var_circuit2.get_graph())
        zx.full_reduce(var_g2b, paramSafe=True)
        tc2 = tcount(var_g2b)
        print(f"    With injection removed: T-count={tc2}")
    else:
        print(f"  {name}: T-count={tc}, Clifford! Can sample with stim")

        # Try to use stim directly
        try:
            stim_circ = stim.Circuit(var_text)
            sampler = stim_circ.compile_sampler()
            samples = sampler.sample(n_samples)
            print(f"    Sampled {n_samples} shots from stim, shape: {samples.shape}")

            # Count unique outcomes
            outcomes = [''.join(str(int(b)) for b in row) for row in samples]
            counter = Counter(outcomes)
            print(f"    Unique outcomes: {len(counter)}")
            print(f"    Top 5 outcomes:")
            for outcome, count in counter.most_common(5):
                print(f"      {outcome}: {count/n_samples:.4f}")
        except Exception as e:
            print(f"    Stim error: {e}")

# ============================================================================
# Sample from original using tsim
# ============================================================================
print(f"\n{'='*60}")
print("Sampling from original d=3 using tsim")
print(f"{'='*60}")

try:
    orig_circuit = tsim.Circuit(t_text)
    # Use tsim's sample method
    samples_tsim = orig_circuit.sample(n_samples)
    print(f"tsim samples shape: {np.array(samples_tsim).shape}")

    outcomes_tsim = [''.join(str(int(b)) for b in row) for row in samples_tsim]
    counter_tsim = Counter(outcomes_tsim)
    print(f"Unique outcomes: {len(counter_tsim)}")
    print(f"Top 5 outcomes:")
    for outcome, count in counter_tsim.most_common(5):
        print(f"  {outcome}: {count/n_samples:.4f}")
except Exception as e:
    print(f"tsim sample error: {e}")
    import traceback
import os
_PRX_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    traceback.print_exc()
