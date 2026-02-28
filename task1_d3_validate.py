"""Task 1: Validate block-replacement decomposition on d=3 circuit.

d=3: 15 qubits, 1 injection + 1 cultivation → 2 Clifford variants
Compare cross-term probabilities with exact simulation.

For 15 qubits, tensor computation is feasible (2^15 = 32768).
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
import os
_PRX_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

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

# Find S/S_DAG blocks (T-gate positions)
print("S/S_DAG lines in d=3 noiseless circuit:")
for i, line in enumerate(noiseless):
    s = line.strip()
    if s.startswith('S_DAG') or s.startswith('S '):
        print(f"  Line {i}: {s}")

# Apply S→T
def replace_s_with_t(s):
    s = re.sub(r'^(\s*)S_DAG(\s)', r'\1T_DAG\2', s, flags=re.MULTILINE)
    return re.sub(r'^(\s*)S(\s)', r'\1T\2', s, flags=re.MULTILINE)

t_text = replace_s_with_t('\n'.join(noiseless))
t_lines = t_text.split('\n')

# Show T gates
print("\nT/T_DAG lines:")
for i, line in enumerate(t_lines):
    s = line.strip()
    if s.startswith('T_DAG') or (s.startswith('T ') and not s.startswith('TICK')):
        print(f"  Line {i}: {s}")

# ZX analysis of original
circuit = tsim.Circuit(t_text)
g = circuit.get_graph()
g2 = deepcopy(g)
zx.full_reduce(g2, paramSafe=True)
print(f"\nOriginal d=3: {circuit.num_qubits} qubits, "
      f"ZX T-count raw={tcount(g)}, reduced={tcount(g2)}")

# Identify block boundaries
print(f"\n{'='*60}")
print("Block identification")
print(f"{'='*60}")

# Find MX lines (measurement hubs)
for i, line in enumerate(t_lines):
    s = line.strip()
    if s.startswith('MX'):
        print(f"  Line {i}: {s}")

# Find cultivation blocks: T_DAG...T pairs around MX
# Look for T_DAG lines followed (eventually) by matching T lines
t_dag_lines = []
t_gate_lines = []
for i, line in enumerate(t_lines):
    s = line.strip()
    if s.startswith('T_DAG'):
        t_dag_lines.append((i, s))
    elif s.startswith('T ') and not s.startswith('TICK'):
        t_gate_lines.append((i, s))

print(f"\nT_DAG lines: {[l for l, s in t_dag_lines]}")
print(f"T lines: {[l for l, s in t_gate_lines]}")

# For d=3: expect 1 injection (single qubit T_DAG) + 1 cultivation (T_DAG + T pair)
# Let's identify the blocks
if len(t_dag_lines) >= 2:
    # First T_DAG = injection, second = cultivation T†, T line = cultivation T
    injection_line = t_dag_lines[0][0]
    cultivation_start = t_dag_lines[1][0]

    # Find matching T line
    cultivation_end = t_gate_lines[-1][0]  # Last T line

    print(f"\nInjection: line {injection_line}: {t_dag_lines[0][1]}")
    print(f"Cultivation start: line {cultivation_start}: {t_dag_lines[1][1]}")
    print(f"Cultivation end: line {cultivation_end}: {t_gate_lines[-1][1]}")

    # Extract cultivation data qubits
    cult_start_parts = t_dag_lines[1][1].split()
    cult_data_qubits = [int(x) for x in cult_start_parts[1:]]
    print(f"Cultivation data qubits: {cult_data_qubits}")

    # Print the cultivation block
    print(f"\nCultivation block (lines {cultivation_start}-{cultivation_end}):")
    for i in range(cultivation_start, cultivation_end + 1):
        print(f"  {i}: {t_lines[i].strip()}")

    # ================================================================
    # Build variants
    # ================================================================
    def make_xsd_lines(data_qubits):
        result = []
        result.append(f"S_DAG {' '.join(str(q) for q in data_qubits)}")
        result.append("TICK")
        result.append(f"X {' '.join(str(q) for q in data_qubits)}")
        result.append("TICK")
        return result

    def build_variant(lines, cult_start, cult_end, mode, inj_line=None, inj_mode="keep"):
        result = []
        for i, line in enumerate(lines):
            if inj_line is not None and i == inj_line:
                if inj_mode == "identity":
                    result.append("TICK")
                    continue
                elif inj_mode == "xsd":
                    inj_parts = line.strip().split()
                    inj_qubits = [int(x) for x in inj_parts[1:]]
                    result.extend(make_xsd_lines(inj_qubits))
                    continue

            if cult_start <= i <= cult_end:
                if i == cult_start:
                    if mode == "identity":
                        result.append("TICK")
                    elif mode == "xsd":
                        result.extend(make_xsd_lines(cult_data_qubits))
                    else:
                        result.append(line)
                elif mode != "keep":
                    continue
                else:
                    result.append(line)
                continue

            result.append(line)
        return '\n'.join(result)

    # ================================================================
    # Compute tensors for all variants and original
    # ================================================================
    print(f"\n{'='*60}")
    print("Computing tensors (15 qubits = 2^15 = 32768 dims)")
    print(f"{'='*60}")

    # Original
    print("\nOriginal circuit...")
    orig_tensor = np.array(circuit.to_tensor())
    n_q = circuit.num_qubits
    idx_zero = tuple([slice(None)] * n_q + [0] * n_q)
    orig_amps = orig_tensor[idx_zero].reshape(-1)
    orig_probs = np.abs(orig_amps)**2
    total_prob = np.sum(orig_probs)
    print(f"  Tensor shape: {orig_tensor.shape}")
    print(f"  Non-zero amplitudes: {np.count_nonzero(np.abs(orig_amps) > 1e-12)}")
    print(f"  Total probability: {total_prob:.6f}")

    # Variant 1: Identity (remove cultivation block)
    print("\nIdentity variant...")
    id_text = build_variant(t_lines, cultivation_start, cultivation_end,
                            "identity", injection_line, "keep")
    id_circuit = tsim.Circuit(id_text)
    id_g = id_circuit.get_graph()
    id_g2 = deepcopy(id_g)
    zx.full_reduce(id_g2, paramSafe=True)
    print(f"  T-count: {tcount(id_g2)}")

    id_tensor = np.array(id_circuit.to_tensor())
    id_amps = id_tensor[idx_zero].reshape(-1)
    print(f"  Non-zero amplitudes: {np.count_nonzero(np.abs(id_amps) > 1e-12)}")

    # Variant 2: XS† (replace cultivation with XS†)
    print("\nXS† variant...")
    xsd_text = build_variant(t_lines, cultivation_start, cultivation_end,
                             "xsd", injection_line, "keep")
    xsd_circuit = tsim.Circuit(xsd_text)
    xsd_g = xsd_circuit.get_graph()
    xsd_g2 = deepcopy(xsd_g)
    zx.full_reduce(xsd_g2, paramSafe=True)
    print(f"  T-count: {tcount(xsd_g2)}")

    xsd_tensor = np.array(xsd_circuit.to_tensor())
    xsd_amps = xsd_tensor[idx_zero].reshape(-1)
    print(f"  Non-zero amplitudes: {np.count_nonzero(np.abs(xsd_amps) > 1e-12)}")

    # ================================================================
    # Test 1: Direct amplitude reconstruction
    # ================================================================
    print(f"\n{'='*60}")
    print("Test 1: Amplitude reconstruction (orig = a·id + b·xsd)")
    print(f"{'='*60}")

    # Only use non-zero entries for fit
    mask = np.abs(orig_amps) > 1e-12
    A = np.column_stack([id_amps[mask], xsd_amps[mask]])
    c, _, _, _ = np.linalg.lstsq(A, orig_amps[mask], rcond=None)
    recon = c[0] * id_amps + c[1] * xsd_amps
    error = np.linalg.norm(orig_amps - recon) / np.linalg.norm(orig_amps)

    print(f"  a = {c[0]:.6f}, b = {c[1]:.6f}")
    print(f"  |a| = {abs(c[0]):.6f}, |b| = {abs(c[1]):.6f}")
    print(f"  Reconstruction error: {error:.4e}")

    # ================================================================
    # Test 2: Probability reconstruction
    # ================================================================
    print(f"\n{'='*60}")
    print("Test 2: Probability reconstruction")
    print(f"{'='*60}")

    # Cross-term: P(m) ∝ |a·⟨m|C_id|0⟩ + b·⟨m|C_xsd|0⟩|²
    cross_probs = np.abs(c[0] * id_amps + c[1] * xsd_amps)**2
    cross_total = np.sum(cross_probs)

    if cross_total > 0:
        cross_probs_norm = cross_probs / cross_total
    else:
        cross_probs_norm = cross_probs

    orig_probs_norm = orig_probs / total_prob if total_prob > 0 else orig_probs

    # Total variation distance
    tvd = 0.5 * np.sum(np.abs(orig_probs_norm - cross_probs_norm))
    print(f"  Total variation distance: {tvd:.6e}")

    # Compare top probabilities
    top_idx = np.argsort(-orig_probs_norm)[:10]
    print(f"\n  Top 10 measurement outcomes:")
    print(f"  {'Outcome':>20} {'P_exact':>10} {'P_cross':>10} {'Ratio':>10}")
    for idx in top_idx:
        bits = format(idx, f'0{n_q}b')
        r = cross_probs_norm[idx] / orig_probs_norm[idx] if orig_probs_norm[idx] > 1e-15 else float('nan')
        print(f"  {bits} {orig_probs_norm[idx]:10.6f} {cross_probs_norm[idx]:10.6f} {r:10.4f}")

    # ================================================================
    # Test 3: Equal coefficients (c=1 for both)
    # ================================================================
    print(f"\n{'='*60}")
    print("Test 3: Equal-weight cross-term: |⟨m|C_id|0⟩ + ⟨m|C_xsd|0⟩|²")
    print(f"{'='*60}")

    equal_probs = np.abs(id_amps + xsd_amps)**2
    equal_total = np.sum(equal_probs)
    if equal_total > 0:
        equal_probs_norm = equal_probs / equal_total
    else:
        equal_probs_norm = equal_probs

    tvd_equal = 0.5 * np.sum(np.abs(orig_probs_norm - equal_probs_norm))
    print(f"  TVD (equal weights): {tvd_equal:.6e}")

    # Test other coefficient combinations
    for ca, cb in [(1, 1j), (1, -1), (1, -1j), (1, np.exp(1j*np.pi/4))]:
        test_probs = np.abs(ca * id_amps + cb * xsd_amps)**2
        test_total = np.sum(test_probs)
        if test_total > 0:
            test_probs_norm = test_probs / test_total
            tvd_test = 0.5 * np.sum(np.abs(orig_probs_norm - test_probs_norm))
            print(f"  TVD (a={ca}, b={cb}): {tvd_test:.6e}")

else:
    print("Not enough T gates found for block identification")
