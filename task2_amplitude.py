"""Task 2: Compute stabilizer amplitudes ⟨m|φᵢ⟩ for each Clifford variant.

Uses stim's TableauSimulator to compute amplitudes via sequential projection.
For each Clifford circuit variant, compute the amplitude for any measurement outcome m
by postselecting each measurement qubit to the desired outcome.

Algorithm:
  For each measurement qubit j:
    1. Check if qubit j is deterministic (via peek_observable_expectation or measure_kickback)
    2. If deterministic: amplitude *= 1 if outcome matches, 0 if not
    3. If random: amplitude *= 1/√2, then postselect to desired outcome

Then P(m) = |Σᵢ cᵢ ⟨m|φᵢ⟩|²
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

# ============================================================================
# Load and prepare d=5 circuit
# ============================================================================
CIRCUIT_PATH = Path("/Users/ritesh/Downloads/prx/gidney-circuits/circuits/"
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

# Apply S→T
def replace_s_with_t(s):
    s = re.sub(r'^(\s*)S_DAG(\s)', r'\1T_DAG\2', s, flags=re.MULTILINE)
    return re.sub(r'^(\s*)S(\s)', r'\1T\2', s, flags=re.MULTILINE)

t_text = replace_s_with_t('\n'.join(noiseless))
t_lines = t_text.split('\n')

# Block positions
injection_line = 63
b1_tdag_line, b1_t_line = 105, 127
b1_qubits = [0, 3, 7, 9, 11, 13, 17]
b2_tdag_line, b2_t_line = 238, 272
b2_qubits = [0, 3, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 32, 34, 36, 38, 40]


def build_variant(t_lines, b1_tdag_mode, b1_t_mode, b2_tdag_mode, b2_t_mode, inj_mode="keep"):
    """Build a circuit variant by replacing T/T_DAG with I or Z."""
    result = []
    for i, line in enumerate(t_lines):
        if i == injection_line:
            if inj_mode == "I":
                result.append("TICK")
                continue
            elif inj_mode == "Z":
                result.append("Z 3")
                continue
        if i == b1_tdag_line:
            if b1_tdag_mode == "I":
                result.append("TICK")
                continue
            elif b1_tdag_mode == "Z":
                result.append(f"Z {' '.join(str(q) for q in b1_qubits)}")
                continue
        if i == b1_t_line:
            if b1_t_mode == "I":
                result.append("TICK")
                continue
            elif b1_t_mode == "Z":
                result.append(f"Z {' '.join(str(q) for q in b1_qubits)}")
                continue
        if i == b2_tdag_line:
            if b2_tdag_mode == "I":
                result.append("TICK")
                continue
            elif b2_tdag_mode == "Z":
                result.append(f"Z {' '.join(str(q) for q in b2_qubits)}")
                continue
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
# Spider cutting coefficients
# ============================================================================
# T = (1/√2)(I + e^{iπ/4} Z)  →  T†⊗T splitting:
#   T† = (1/√2)(I + e^{-iπ/4} Z)
#   T = (1/√2)(I + e^{iπ/4} Z)
#
# For n-qubit block: T†^⊗n ⊗ T^⊗n
#   = (1/2)^n Σ_{a,b ∈ {0,1}} e^{-inπa/4} e^{inπb/4} Z^a ⊗ Z^b
#   = (1/2)^n Σ_{a,b} e^{inπ(b-a)/4} Z^a ⊗ Z^b
#
# Where a indexes T† replacement (0=I, 1=Z) and b indexes T replacement (0=I, 1=Z)
#
# For the injection: single qubit T†
#   T† = (1/√2)(I + e^{-iπ/4} Z)
#   → (1/√2) for I, (1/√2)e^{-iπ/4} for Z

def compute_coefficient(b1_tdag, b1_t, b2_tdag, b2_t, inj):
    """Compute the decomposition coefficient for a given variant.

    Each mode is "I" (=0) or "Z" (=1).
    """
    n1 = len(b1_qubits)  # 7
    n2 = len(b2_qubits)  # 19

    # Block 1: T†^⊗n1 replaced by (I or Z), T^⊗n1 replaced by (I or Z)
    a1 = 1 if b1_tdag == "Z" else 0
    b1 = 1 if b1_t == "Z" else 0

    # Block 2: same
    a2 = 1 if b2_tdag == "Z" else 0
    b2 = 1 if b2_t == "Z" else 0

    # Injection: single T†
    a_inj = 1 if inj == "Z" else 0

    # Coefficient from block 1: (1/2)^n1 * e^{i n1 π (b1-a1)/4}
    coeff1 = (0.5)**n1 * np.exp(1j * n1 * np.pi * (b1 - a1) / 4)

    # Coefficient from block 2: (1/2)^n2 * e^{i n2 π (b2-a2)/4}
    coeff2 = (0.5)**n2 * np.exp(1j * n2 * np.pi * (b2 - a2) / 4)

    # Injection: (1/√2) * e^{-i π a_inj / 4}
    coeff_inj = (1/np.sqrt(2)) * np.exp(-1j * np.pi * a_inj / 4)

    return coeff1 * coeff2 * coeff_inj


# ============================================================================
# Build all 32 Clifford variants (2 inj × 4 block1 × 4 block2)
# ============================================================================
print("="*60)
print("Building 32 Clifford variant circuits")
print("="*60)

modes = ["I", "Z"]
variants = []

for inj in modes:
    for b1_tdag in modes:
        for b1_t in modes:
            for b2_tdag in modes:
                for b2_t in modes:
                    coeff = compute_coefficient(b1_tdag, b1_t, b2_tdag, b2_t, inj)
                    text = build_variant(t_lines, b1_tdag, b1_t, b2_tdag, b2_t, inj)

                    label = f"inj={inj},({b1_tdag},{b1_t})×({b2_tdag},{b2_t})"
                    variants.append({
                        'label': label,
                        'text': text,
                        'coeff': coeff,
                        'inj': inj,
                        'b1_tdag': b1_tdag, 'b1_t': b1_t,
                        'b2_tdag': b2_tdag, 'b2_t': b2_t,
                    })
                    print(f"  {label}: |coeff|={abs(coeff):.2e}, "
                          f"phase={np.angle(coeff)/np.pi:.4f}π")

print(f"\nTotal variants: {len(variants)}")

# Check that coefficients sum correctly
total_coeff_sq = sum(abs(v['coeff'])**2 for v in variants)
print(f"Sum of |coeff|²: {total_coeff_sq:.6e}")


# ============================================================================
# Test: load one variant into stim
# ============================================================================
print(f"\n{'='*60}")
print("Testing stim loading of Clifford variant circuits")
print(f"{'='*60}")

# Check if the variant text has any T/T_DAG gates
test_text = variants[0]['text']
has_t = False
for line in test_text.split('\n'):
    s = line.strip()
    if s.startswith('T_DAG') or (s.startswith('T ') and not s.startswith('TICK')):
        has_t = True
        print(f"  WARNING: T gate found: {s}")

if not has_t:
    print("  No T gates in variant — good!")

# Try loading into stim
try:
    stim_circ = stim.Circuit(test_text)
    print(f"  Stim loaded: {stim_circ.num_qubits} qubits, "
          f"{stim_circ.num_measurements} measurements")
except Exception as e:
    print(f"  Stim load error: {e}")
    # The issue might be that S_DAG was converted to T_DAG
    # Check for S gates too
    for line in test_text.split('\n'):
        s = line.strip()
        if 'T_DAG' in s or ('T ' in s and 'TICK' not in s):
            print(f"    Problematic line: {s[:80]}")


# ============================================================================
# Compute amplitudes via stim TableauSimulator
# ============================================================================
print(f"\n{'='*60}")
print("Computing amplitudes via sequential postselection")
print(f"{'='*60}")


def compute_amplitude_stim(circuit_text, measurement_outcome):
    """Compute ⟨m|φ⟩ for a Clifford circuit using stim's TableauSimulator.

    Algorithm: Run the circuit up to the measurements, then for each measurement
    qubit, check if it's deterministic. If deterministic and matches outcome: 1.
    If deterministic and doesn't match: amplitude = 0.
    If random: amplitude *= 1/√2, then postselect to desired outcome.

    Args:
        circuit_text: stim circuit string (must be Clifford)
        measurement_outcome: list of 0/1 values for each measurement

    Returns:
        complex amplitude ⟨m|φ⟩
    """
    sim = stim.TableauSimulator()

    # Parse and execute the circuit instruction by instruction,
    # handling measurements specially
    circ = stim.Circuit(circuit_text)

    amplitude = 1.0 + 0j
    meas_idx = 0

    for instruction in circ.flattened():
        name = instruction.name

        if name in ('M', 'MZ'):
            # Z-basis measurement
            for target in instruction.targets_copy():
                qubit = target.value
                desired = measurement_outcome[meas_idx]

                # Check if deterministic
                kickback = sim.measure_kickback(qubit)
                result = kickback[0]
                pauli = kickback[1]

                if pauli is None:
                    # Deterministic
                    if int(result) != desired:
                        return 0.0 + 0j
                    # amplitude unchanged
                else:
                    # Random: amplitude *= 1/√2
                    amplitude *= 1/np.sqrt(2)
                    # Postselect to desired outcome
                    if int(result) != desired:
                        # Need to flip — apply the kickback Pauli
                        sim.do(stim.Circuit(f"Z {qubit}"))  # rough flip
                        # Actually need to undo the measurement effect
                        # Let's use postselect instead

                meas_idx += 1

        elif name == 'MX':
            # X-basis measurement
            for target in instruction.targets_copy():
                qubit = target.value
                desired = measurement_outcome[meas_idx]

                # Convert to Z-basis: H, measure Z, H
                sim.h(qubit)
                kickback = sim.measure_kickback(qubit)
                result = kickback[0]
                pauli = kickback[1]

                if pauli is None:
                    if int(result) != desired:
                        return 0.0 + 0j
                else:
                    amplitude *= 1/np.sqrt(2)
                    if int(result) != desired:
                        # Need to flip
                        pass

                sim.h(qubit)
                meas_idx += 1

        elif name == 'MR':
            # Measure and reset
            for target in instruction.targets_copy():
                qubit = target.value
                desired = measurement_outcome[meas_idx]

                kickback = sim.measure_kickback(qubit)
                result = kickback[0]
                pauli = kickback[1]

                if pauli is None:
                    if int(result) != desired:
                        return 0.0 + 0j
                else:
                    amplitude *= 1/np.sqrt(2)

                # Reset
                if result:
                    sim.x(qubit)

                meas_idx += 1

        elif name == 'RX':
            # X-basis reset
            for target in instruction.targets_copy():
                sim.h(target.value)
                sim.z(target.value)  # |+⟩ state
                sim.h(target.value)

        else:
            # Execute non-measurement instruction
            sim.do(stim.Circuit(str(instruction)))

    return amplitude


# Alternative simpler approach: use stim's state_vector for small circuits
# For 42 qubits this won't work, but let's test the postselection approach first

# First, let's understand the measurement structure
print("\nAnalyzing measurement structure...")
test_circ = stim.Circuit(variants[0]['text'])
print(f"  Total measurements: {test_circ.num_measurements}")
print(f"  Qubits: {test_circ.num_qubits}")

# Get measurement types from the circuit text
meas_types = []
for line in variants[0]['text'].split('\n'):
    s = line.strip()
    if s.startswith('M ') or s.startswith('MX ') or s.startswith('MR '):
        gate = s.split()[0]
        qubits = [int(x) for x in s.split()[1:]]
        for q in qubits:
            meas_types.append((gate, q))
    elif s == 'M' or s == 'MX' or s == 'MR':
        pass  # no qubits specified, shouldn't happen

print(f"  Measurement breakdown: {Counter(t for t, q in meas_types)}")
print(f"  First 10 measurements: {meas_types[:10]}")
print(f"  Last 10 measurements: {meas_types[-10:]}")


# ============================================================================
# Simpler approach: use stim's sampler + postselection trick
# ============================================================================
print(f"\n{'='*60}")
print("Approach 2: Sample + compute amplitudes via state vector (d=3 first)")
print(f"{'='*60}")

# Let's first validate on d=3 where state_vector() is feasible (15 qubits)
# For d=5 (42 qubits), we need the postselection approach

# Load d=3 circuit
D3_PATH = Path("/Users/ritesh/Downloads/prx/gidney-circuits/circuits/"
               "for_perfectionist_decoding/"
               "c=inject[unitary]+cultivate,p=0.001,noise=uniform,"
               "g=css,q=15,b=Y,r=4,d1=3.stim")

d3_str = D3_PATH.read_text()
d3_lines = d3_str.split('\n')
d3_noiseless = []
for line in d3_lines:
    s = line.strip()
    if any(s.startswith(p) for p in ['X_ERROR', 'Z_ERROR', 'DEPOLARIZE1', 'DEPOLARIZE2']):
        continue
    line = re.sub(r'M\([\d.]+\)', 'M', line)
    line = re.sub(r'MX\([\d.]+\)', 'MX', line)
    d3_noiseless.append(line)

d3_t_text = replace_s_with_t('\n'.join(d3_noiseless))
d3_t_lines = d3_t_text.split('\n')

# Find T gates in d=3
print("\nT/T_DAG lines in d=3:")
d3_t_positions = []
for i, line in enumerate(d3_t_lines):
    s = line.strip()
    if s.startswith('T_DAG') or (s.startswith('T ') and not s.startswith('TICK')):
        d3_t_positions.append((i, s))
        print(f"  Line {i}: {s}")

# For d=3: identify injection and cultivation
# From previous analysis: injection at line ~36 (single qubit), cultivation block with paired T†/T
print(f"\nFound {len(d3_t_positions)} T/T_DAG lines")

# The first T_DAG with single qubit = injection
# T_DAG with multiple qubits = cultivation T†
# T with same qubits = cultivation T
d3_injection_line = None
d3_cult_tdag_line = None
d3_cult_t_line = None
d3_cult_qubits = None

for i, (ln, s) in enumerate(d3_t_positions):
    parts = s.split()
    gate = parts[0]
    qubits = parts[1:]
    if gate == 'T_DAG' and len(qubits) == 1 and d3_injection_line is None:
        d3_injection_line = ln
        print(f"\n  Injection: line {ln}: {s}")
    elif gate == 'T_DAG' and len(qubits) > 1:
        d3_cult_tdag_line = ln
        d3_cult_qubits = [int(q) for q in qubits]
        print(f"  Cultivation T†: line {ln}: {s}")
    elif gate == 'T' and len(qubits) > 1:
        d3_cult_t_line = ln
        print(f"  Cultivation T: line {ln}: {s}")

print(f"\n  d=3 cultivation qubits: {d3_cult_qubits}")
n_cult = len(d3_cult_qubits) if d3_cult_qubits else 0
print(f"  n_cult = {n_cult}")

# Build d=3 variants
def build_d3_variant(lines, inj_mode, cult_tdag_mode, cult_t_mode):
    result = []
    for i, line in enumerate(lines):
        if i == d3_injection_line:
            if inj_mode == "I":
                result.append("TICK")
                continue
            elif inj_mode == "Z":
                parts = line.strip().split()
                q = parts[1]
                result.append(f"Z {q}")
                continue

        if i == d3_cult_tdag_line:
            if cult_tdag_mode == "I":
                result.append("TICK")
                continue
            elif cult_tdag_mode == "Z":
                result.append(f"Z {' '.join(str(q) for q in d3_cult_qubits)}")
                continue

        if i == d3_cult_t_line:
            if cult_t_mode == "I":
                result.append("TICK")
                continue
            elif cult_t_mode == "Z":
                result.append(f"Z {' '.join(str(q) for q in d3_cult_qubits)}")
                continue

        result.append(line)
    return '\n'.join(result)

# Build all 8 d=3 variants (2 injection × 4 cultivation)
print(f"\n{'='*60}")
print("d=3: Building 8 Clifford variants")
print(f"{'='*60}")

d3_variants = []
for inj in modes:
    for cult_tdag in modes:
        for cult_t in modes:
            text = build_d3_variant(d3_t_lines, inj, cult_tdag, cult_t)

            # Check T-count
            circ = tsim.Circuit(text)
            g = circ.get_graph()
            g2 = deepcopy(g)
            zx.full_reduce(g2, paramSafe=True)
            tc = tcount(g2)

            # Coefficient
            n = n_cult
            a_cult = 1 if cult_tdag == "Z" else 0
            b_cult = 1 if cult_t == "Z" else 0
            a_inj = 1 if inj == "Z" else 0

            coeff_cult = (0.5)**n * np.exp(1j * n * np.pi * (b_cult - a_cult) / 4)
            coeff_inj = (1/np.sqrt(2)) * np.exp(-1j * np.pi * a_inj / 4)
            coeff = coeff_cult * coeff_inj

            label = f"inj={inj},cult=({cult_tdag},{cult_t})"

            d3_variants.append({
                'label': label,
                'text': text,
                'coeff': coeff,
                'tc': tc,
            })

            status = "CLIFFORD" if tc == 0 else f"T={tc}"
            print(f"  {label}: {status}, |coeff|={abs(coeff):.4e}, "
                  f"phase={np.angle(coeff)/np.pi:.4f}π")

# Try loading d=3 variants into stim
print(f"\n{'='*60}")
print("d=3: Loading into stim")
print(f"{'='*60}")

for v in d3_variants:
    try:
        stim_circ = stim.Circuit(v['text'])
        print(f"  {v['label']}: {stim_circ.num_qubits} qubits, "
              f"{stim_circ.num_measurements} measurements — OK")
    except Exception as e:
        print(f"  {v['label']}: Error — {e}")
        # Show first T/T_DAG line
        for line in v['text'].split('\n'):
            s = line.strip()
            if s.startswith('T_DAG') or (s.startswith('T ') and not s.startswith('TICK')):
                print(f"    Found T gate: {s}")
                break

# ============================================================================
# d=3: Compute amplitudes via stim state_vector()
# ============================================================================
print(f"\n{'='*60}")
print("d=3: Computing state vectors")
print(f"{'='*60}")

# For d=3 with 15 qubits, state_vector has 2^15 = 32768 entries — easy!
# But we need to handle measurements differently.
# stim's TableauSimulator runs the circuit including measurements.
# After measurements, the state collapses.
# We need the PRE-measurement state or the amplitude for specific outcomes.

# Let's try a different approach: simulate the circuit up to measurements,
# then extract the state vector.

# First, let's identify where measurements happen
print("\nd=3 variant 0: measurement structure")
test_text = d3_variants[0]['text']
meas_info = []
for line in test_text.split('\n'):
    s = line.strip()
    if s.startswith('M ') or s.startswith('MX ') or s.startswith('MR '):
        gate = s.split()[0]
        qubits = [int(x) for x in s.split()[1:]]
        meas_info.append((gate, qubits))
    elif s.startswith('M(') or s.startswith('MX('):
        # Noisy measurement — shouldn't be here after stripping
        pass

for gate, qubits in meas_info:
    print(f"  {gate} {qubits}")

print(f"\nTotal measurement groups: {len(meas_info)}")
total_meas = sum(len(q) for _, q in meas_info)
print(f"Total measurement qubits: {total_meas}")

# The key insight: for the amplitude approach, we need to:
# 1. Run all gates up to final measurements
# 2. For MX (mid-circuit): these are part of the cultivation protocol
#    The outcomes of MX/MR gates define the syndrome
# 3. For final M: these are the data qubit measurements
#
# For a complete amplitude computation, we need to handle mid-circuit measurements.
# Each mid-circuit measurement creates a branch — BUT in the Clifford case,
# the measurement is typically deterministic (forced by the stabilizer structure).
#
# Let's check: run the Clifford variant and see if measurements are deterministic.

print(f"\n{'='*60}")
print("d=3: Testing measurement determinism")
print(f"{'='*60}")

for v in d3_variants[:4]:  # First 4 variants
    try:
        sim = stim.TableauSimulator()
        circ = stim.Circuit(v['text'])

        n_random = 0
        n_deterministic = 0
        meas_idx = 0
        meas_results = []

        for instruction in circ.flattened():
            name = instruction.name
            if name in ('M', 'MZ', 'MX', 'MR', 'MRX'):
                for target in instruction.targets_copy():
                    qubit = target.value

                    if name in ('MX', 'MRX'):
                        sim.h(qubit)

                    result, pauli = sim.measure_kickback(qubit)

                    if pauli is None:
                        n_deterministic += 1
                    else:
                        n_random += 1

                    meas_results.append(int(result))

                    if name in ('MR', 'MRX'):
                        if result:
                            sim.x(qubit)
                    if name in ('MX', 'MRX'):
                        sim.h(qubit)

                    meas_idx += 1
            else:
                sim.do(stim.Circuit(str(instruction)))

        print(f"  {v['label']}: {n_deterministic} deterministic, "
              f"{n_random} random, results[:20]={meas_results[:20]}")

    except Exception as e:
        print(f"  {v['label']}: Error — {e}")
        import traceback
        traceback.print_exc()
