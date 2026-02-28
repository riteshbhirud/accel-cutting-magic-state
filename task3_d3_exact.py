"""Task 3: Exact d=3 validation using numpy state vector simulation.

Strategy: Don't enumerate all 2^21 outcomes. Instead:
1. Sample from original circuit (random measurement choices) to find reachable outcomes
2. Compute amplitudes only for those specific outcomes
3. Verify: orig_amp(m) = Σ cᵢ × variant_amp_i(m)
"""
import sys, os, re
from pathlib import Path
import numpy as np
from collections import Counter, defaultdict

os.environ["JAX_PLATFORMS"] = "cpu"

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))

# ============================================================================
# Numpy state vector simulator
# ============================================================================
I2 = np.eye(2, dtype=complex)
X_mat = np.array([[0, 1], [1, 0]], dtype=complex)
Y_mat = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z_mat = np.array([[1, 0], [0, -1]], dtype=complex)
H_mat = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
S_mat = np.array([[1, 0], [0, 1j]], dtype=complex)
S_DAG_mat = np.array([[1, 0], [0, -1j]], dtype=complex)
T_mat = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
T_DAG_mat = np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]], dtype=complex)
SQRT_X_mat = np.array([[1+1j, 1-1j], [1-1j, 1+1j]], dtype=complex) / 2
SQRT_X_DAG_mat = np.array([[1-1j, 1+1j], [1+1j, 1-1j]], dtype=complex) / 2


class StateVecSim:
    def __init__(self, n_qubits):
        self.n = n_qubits
        self.state = np.zeros(2**n_qubits, dtype=complex)
        self.state[0] = 1.0

    def _apply_single(self, qubit, mat):
        n = self.n
        state = self.state.reshape([2]*n)
        state = np.moveaxis(state, qubit, 0)
        state = np.einsum('ij,j...->i...', mat, state)
        state = np.moveaxis(state, 0, qubit)
        self.state = state.reshape(-1)

    def _apply_cx_fast(self, control, target):
        n = self.n
        state = self.state.reshape([2]*n)
        state2 = state.copy()
        c1t0 = [slice(None)] * n
        c1t0[control] = 1
        c1t0[target] = 0
        c1t1 = [slice(None)] * n
        c1t1[control] = 1
        c1t1[target] = 1
        state2[tuple(c1t0)] = state[tuple(c1t1)]
        state2[tuple(c1t1)] = state[tuple(c1t0)]
        self.state = state2.reshape(-1)

    def _project_z(self, qubit, outcome):
        """Project onto Z-basis outcome. Returns amplitude factor (= sqrt(probability))."""
        n = self.n
        state = self.state.reshape([2]*n)
        idx_other = [slice(None)] * n
        idx_other[qubit] = 1 - outcome
        state[tuple(idx_other)] = 0
        self.state = state.reshape(-1)
        norm = np.linalg.norm(self.state)
        if norm > 1e-15:
            self.state /= norm
            return norm
        return 0.0

    def _reset_z(self, qubit):
        n = self.n
        state = self.state.reshape([2]*n)
        idx1 = [slice(None)] * n
        idx1[qubit] = 1
        if np.sum(np.abs(state[tuple(idx1)])**2) > 1e-12:
            self._apply_single(qubit, X_mat)


def compute_amplitude(circuit_text, n_qubits, target_outcome):
    """Compute ⟨m|U|0⟩ for a specific measurement outcome m.

    Runs the circuit, forcing each measurement to the desired outcome,
    and returns the product of projection amplitudes.
    """
    sim = StateVecSim(n_qubits)
    total_amp = 1.0 + 0j
    meas_idx = 0

    for line in circuit_text.split('\n'):
        s = line.strip()
        if not s or s.startswith('#') or s.startswith('TICK') or s.startswith('QUBIT_COORDS'):
            continue
        if s.startswith('DETECTOR') or s.startswith('OBSERVABLE_INCLUDE'):
            continue

        parts = s.split()
        gate = parts[0]

        if gate == 'R':
            for q in parts[1:]:
                sim._reset_z(int(q))
        elif gate == 'RX':
            for q in parts[1:]:
                sim._reset_z(int(q))
                sim._apply_single(int(q), H_mat)
        elif gate == 'H':
            for q in parts[1:]:
                sim._apply_single(int(q), H_mat)
        elif gate == 'X':
            for q in parts[1:]:
                sim._apply_single(int(q), X_mat)
        elif gate == 'Y':
            for q in parts[1:]:
                sim._apply_single(int(q), Y_mat)
        elif gate == 'Z':
            for q in parts[1:]:
                sim._apply_single(int(q), Z_mat)
        elif gate == 'S':
            for q in parts[1:]:
                sim._apply_single(int(q), S_mat)
        elif gate == 'S_DAG':
            for q in parts[1:]:
                sim._apply_single(int(q), S_DAG_mat)
        elif gate == 'T':
            for q in parts[1:]:
                sim._apply_single(int(q), T_mat)
        elif gate == 'T_DAG':
            for q in parts[1:]:
                sim._apply_single(int(q), T_DAG_mat)
        elif gate == 'SQRT_X':
            for q in parts[1:]:
                sim._apply_single(int(q), SQRT_X_mat)
        elif gate == 'SQRT_X_DAG':
            for q in parts[1:]:
                sim._apply_single(int(q), SQRT_X_DAG_mat)
        elif gate in ('CX', 'CNOT', 'ZCX'):
            targets = parts[1:]
            for i in range(0, len(targets), 2):
                c, t = int(targets[i]), int(targets[i+1])
                sim._apply_cx_fast(c, t)
        elif gate == 'CZ':
            targets = parts[1:]
            for i in range(0, len(targets), 2):
                c, t = int(targets[i]), int(targets[i+1])
                sim._apply_single(t, H_mat)
                sim._apply_cx_fast(c, t)
                sim._apply_single(t, H_mat)
        elif gate in ('M', 'MZ'):
            for q in parts[1:]:
                desired = target_outcome[meas_idx]
                amp = sim._project_z(int(q), desired)
                total_amp *= amp
                meas_idx += 1
                if abs(total_amp) < 1e-15:
                    return 0.0  # Early exit
        elif gate == 'MX':
            for q in parts[1:]:
                desired = target_outcome[meas_idx]
                sim._apply_single(int(q), H_mat)
                amp = sim._project_z(int(q), desired)
                sim._apply_single(int(q), H_mat)
                total_amp *= amp
                meas_idx += 1
                if abs(total_amp) < 1e-15:
                    return 0.0
        elif gate == 'MR':
            for q in parts[1:]:
                desired = target_outcome[meas_idx]
                amp = sim._project_z(int(q), desired)
                total_amp *= amp
                sim._reset_z(int(q))
                meas_idx += 1
                if abs(total_amp) < 1e-15:
                    return 0.0
        elif gate == 'MRX':
            for q in parts[1:]:
                desired = target_outcome[meas_idx]
                sim._apply_single(int(q), H_mat)
                amp = sim._project_z(int(q), desired)
                sim._apply_single(int(q), H_mat)
                total_amp *= amp
                sim._reset_z(int(q))
                sim._apply_single(int(q), H_mat)
                meas_idx += 1
                if abs(total_amp) < 1e-15:
                    return 0.0
        elif gate in ('I', 'TICK'):
            pass

    return total_amp


def sample_outcomes(circuit_text, n_qubits, n_samples=100):
    """Sample measurement outcomes by running circuit with random measurements.
    Returns list of (outcome_tuple, amplitude) pairs.
    """
    outcomes = set()

    for _ in range(n_samples):
        sim = StateVecSim(n_qubits)
        meas_results = []
        total_amp = 1.0 + 0j

        for line in circuit_text.split('\n'):
            s = line.strip()
            if not s or s.startswith('#') or s.startswith('TICK') or s.startswith('QUBIT_COORDS'):
                continue
            if s.startswith('DETECTOR') or s.startswith('OBSERVABLE_INCLUDE'):
                continue

            parts = s.split()
            gate = parts[0]

            if gate == 'R':
                for q in parts[1:]:
                    sim._reset_z(int(q))
            elif gate == 'RX':
                for q in parts[1:]:
                    sim._reset_z(int(q))
                    sim._apply_single(int(q), H_mat)
            elif gate == 'H':
                for q in parts[1:]:
                    sim._apply_single(int(q), H_mat)
            elif gate == 'X':
                for q in parts[1:]:
                    sim._apply_single(int(q), X_mat)
            elif gate == 'Y':
                for q in parts[1:]:
                    sim._apply_single(int(q), Y_mat)
            elif gate == 'Z':
                for q in parts[1:]:
                    sim._apply_single(int(q), Z_mat)
            elif gate == 'S':
                for q in parts[1:]:
                    sim._apply_single(int(q), S_mat)
            elif gate == 'S_DAG':
                for q in parts[1:]:
                    sim._apply_single(int(q), S_DAG_mat)
            elif gate == 'T':
                for q in parts[1:]:
                    sim._apply_single(int(q), T_mat)
            elif gate == 'T_DAG':
                for q in parts[1:]:
                    sim._apply_single(int(q), T_DAG_mat)
            elif gate == 'SQRT_X':
                for q in parts[1:]:
                    sim._apply_single(int(q), SQRT_X_mat)
            elif gate == 'SQRT_X_DAG':
                for q in parts[1:]:
                    sim._apply_single(int(q), SQRT_X_DAG_mat)
            elif gate in ('CX', 'CNOT', 'ZCX'):
                targets = parts[1:]
                for i in range(0, len(targets), 2):
                    c, t = int(targets[i]), int(targets[i+1])
                    sim._apply_cx_fast(c, t)
            elif gate == 'CZ':
                targets = parts[1:]
                for i in range(0, len(targets), 2):
                    c, t = int(targets[i]), int(targets[i+1])
                    sim._apply_single(t, H_mat)
                    sim._apply_cx_fast(c, t)
                    sim._apply_single(t, H_mat)
            elif gate in ('M', 'MZ'):
                for q in parts[1:]:
                    n = sim.n
                    state = sim.state.reshape([2]*n)
                    idx0 = [slice(None)] * n
                    idx0[int(q)] = 0
                    p0 = np.sum(np.abs(state[tuple(idx0)])**2)
                    outcome = 0 if np.random.random() < p0 else 1
                    amp = sim._project_z(int(q), outcome)
                    total_amp *= amp
                    meas_results.append(outcome)
            elif gate == 'MX':
                for q in parts[1:]:
                    sim._apply_single(int(q), H_mat)
                    n = sim.n
                    state = sim.state.reshape([2]*n)
                    idx0 = [slice(None)] * n
                    idx0[int(q)] = 0
                    p0 = np.sum(np.abs(state[tuple(idx0)])**2)
                    outcome = 0 if np.random.random() < p0 else 1
                    amp = sim._project_z(int(q), outcome)
                    sim._apply_single(int(q), H_mat)
                    total_amp *= amp
                    meas_results.append(outcome)
            elif gate == 'MR':
                for q in parts[1:]:
                    n = sim.n
                    state = sim.state.reshape([2]*n)
                    idx0 = [slice(None)] * n
                    idx0[int(q)] = 0
                    p0 = np.sum(np.abs(state[tuple(idx0)])**2)
                    outcome = 0 if np.random.random() < p0 else 1
                    amp = sim._project_z(int(q), outcome)
                    total_amp *= amp
                    sim._reset_z(int(q))
                    meas_results.append(outcome)
            elif gate in ('I', 'TICK'):
                pass

        outcomes.add(tuple(meas_results))

    return outcomes


# ============================================================================
# Load d=3 circuit
# ============================================================================
D3_PATH = Path("/Users/ritesh/Downloads/prx/gidney-circuits/circuits/"
               "for_perfectionist_decoding/"
               "c=inject[unitary]+cultivate,p=0.001,noise=uniform,"
               "g=css,q=15,b=Y,r=4,d1=3.stim")

circuit_str = D3_PATH.read_text()
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

t_text = replace_s_with_t('\n'.join(noiseless))
t_lines = t_text.split('\n')

d3_injection_line = 36
d3_cult_tdag_line = 78
d3_cult_t_line = 100
d3_cult_qubits = [0, 3, 7, 9, 10, 12, 13]
n_cult = len(d3_cult_qubits)
n_qubits = 15

# Count measurements
n_measurements = 0
for line in t_text.split('\n'):
    s = line.strip()
    if s.startswith(('M ', 'MX ', 'MR ', 'MRX ')):
        n_measurements += len(s.split()) - 1
print(f"d=3: {n_qubits} qubits, {n_measurements} measurements")

# ============================================================================
# Step 1: Find reachable outcomes by sampling
# ============================================================================
print(f"\n{'='*60}")
print("Step 1: Sampling reachable outcomes from original circuit")
print(f"{'='*60}")

print("Sampling 200 shots...")
reachable = sample_outcomes(t_text, n_qubits, n_samples=200)
print(f"Found {len(reachable)} unique outcome patterns")

for outcome in sorted(reachable):
    outcome_str = ''.join(str(b) for b in outcome)
    print(f"  {outcome_str}")

# ============================================================================
# Step 2: Compute exact amplitude for each reachable outcome
# ============================================================================
print(f"\n{'='*60}")
print("Step 2: Exact amplitudes for original circuit")
print(f"{'='*60}")

orig_amps = {}
for outcome_tuple in sorted(reachable):
    outcome_list = list(outcome_tuple)
    amp = compute_amplitude(t_text, n_qubits, outcome_list)
    orig_amps[outcome_tuple] = amp
    outcome_str = ''.join(str(b) for b in outcome_tuple)
    print(f"  {outcome_str}: amp = {amp:.8f}, |amp|² = {abs(amp)**2:.8f}")

total_prob = sum(abs(a)**2 for a in orig_amps.values())
print(f"\nTotal probability: {total_prob:.8f}")

# ============================================================================
# Step 3: Build Clifford variants and compute amplitudes
# ============================================================================
print(f"\n{'='*60}")
print("Step 3: Clifford variant amplitudes")
print(f"{'='*60}")

modes = ["I", "Z"]

def build_variant(lines, inj_mode, cult_tdag_mode, cult_t_mode):
    result = []
    for i, line in enumerate(lines):
        if i == d3_injection_line:
            if inj_mode == "I":
                result.append("TICK")
                continue
            elif inj_mode == "Z":
                result.append("Z 3")
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

def compute_coefficient(inj, cult_tdag, cult_t):
    n = n_cult
    if cult_tdag == "I":
        coeff_tdag = (1/np.sqrt(2))**n
    else:
        coeff_tdag = (1/np.sqrt(2))**n * np.exp(-1j * n * np.pi / 4)
    if cult_t == "I":
        coeff_t = (1/np.sqrt(2))**n
    else:
        coeff_t = (1/np.sqrt(2))**n * np.exp(1j * n * np.pi / 4)
    if inj == "I":
        coeff_inj = 1/np.sqrt(2)
    else:
        coeff_inj = (1/np.sqrt(2)) * np.exp(-1j * np.pi / 4)
    return coeff_tdag * coeff_t * coeff_inj

# Also find reachable outcomes for each variant
all_reachable = set(reachable)
for inj in modes:
    for cult_tdag in modes:
        for cult_t in modes:
            text = build_variant(t_lines, inj, cult_tdag, cult_t)
            variant_reachable = sample_outcomes(text, n_qubits, n_samples=50)
            all_reachable.update(variant_reachable)
            label = f"inj={inj},cult=({cult_tdag},{cult_t})"
            print(f"  {label}: {len(variant_reachable)} reachable outcomes")

print(f"\nTotal unique outcomes across all: {len(all_reachable)}")

# Compute amplitudes for all variants at all reachable outcomes
variant_amps = {}
variant_coeffs = {}

for inj in modes:
    for cult_tdag in modes:
        for cult_t in modes:
            label = f"inj={inj},cult=({cult_tdag},{cult_t})"
            text = build_variant(t_lines, inj, cult_tdag, cult_t)
            coeff = compute_coefficient(inj, cult_tdag, cult_t)
            variant_coeffs[label] = coeff

            amps = {}
            for outcome_tuple in sorted(all_reachable):
                amp = compute_amplitude(text, n_qubits, list(outcome_tuple))
                if abs(amp) > 1e-15:
                    amps[outcome_tuple] = amp

            variant_amps[label] = amps
            print(f"\n  {label}: coeff={coeff:.6f}, non-zero={len(amps)}")
            for ot, amp in sorted(amps.items(), key=lambda x: -abs(x[1])):
                os_str = ''.join(str(b) for b in ot)
                print(f"    {os_str}: amp={amp:.8f}")

# Also compute original amplitudes for any newly discovered outcomes
for ot in all_reachable:
    if ot not in orig_amps:
        amp = compute_amplitude(t_text, n_qubits, list(ot))
        orig_amps[ot] = amp
        if abs(amp) > 1e-15:
            os_str = ''.join(str(b) for b in ot)
            print(f"  NEW original: {os_str}: amp={amp:.8f}")


# ============================================================================
# Step 4: Reconstruct and compare
# ============================================================================
print(f"\n{'='*60}")
print("Step 4: Amplitude reconstruction")
print(f"  orig_amp(m) ≟ Σᵢ cᵢ × variant_amp_i(m)")
print(f"{'='*60}")

print(f"\n  {'Outcome':>25} {'orig amp':>20} {'recon amp':>20} {'error':>10}")
print(f"  {'-'*75}")

total_error_sq = 0.0
total_norm_sq = 0.0

for ot in sorted(all_reachable):
    a_orig = orig_amps.get(ot, 0.0)

    a_recon = 0.0 + 0j
    for label in variant_amps:
        coeff = variant_coeffs[label]
        a_var = variant_amps[label].get(ot, 0.0)
        a_recon += coeff * a_var

    os_str = ''.join(str(b) for b in ot)
    err = abs(a_orig - a_recon)
    total_error_sq += err**2
    total_norm_sq += abs(a_orig)**2

    if abs(a_orig) > 1e-12 or abs(a_recon) > 1e-12:
        print(f"  {os_str:>25} {a_orig:>20.8f} {a_recon:>20.8f} {err:10.2e}")

total_norm = np.sqrt(total_norm_sq)
rel_error = np.sqrt(total_error_sq) / total_norm if total_norm > 0 else float('inf')

print(f"\n  Relative error: {rel_error:.4e}")

if rel_error < 1e-6:
    print(f"\n  *** VALIDATION PASSED: decomposition is exact! ***")
elif rel_error < 0.01:
    print(f"\n  *** VALIDATION CLOSE: error = {rel_error:.4e} ***")
else:
    print(f"\n  *** VALIDATION FAILED: error = {rel_error:.4e} ***")

    # Diagnose: fit coefficients
    print(f"\n  Fitting optimal coefficients...")
    outcomes_list = sorted(all_reachable)
    n_out = len(outcomes_list)
    n_var = len(variant_amps)

    A_mat = np.zeros((n_out, n_var), dtype=complex)
    b_vec = np.zeros(n_out, dtype=complex)

    labels_sorted = sorted(variant_amps.keys())
    for i, ot in enumerate(outcomes_list):
        b_vec[i] = orig_amps.get(ot, 0.0)
        for j, label in enumerate(labels_sorted):
            A_mat[i, j] = variant_amps[label].get(ot, 0.0)

    x, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)

    print(f"\n  {'Label':>35} {'Expected':>15} {'Fitted':>15} {'Ratio':>10}")
    for j, label in enumerate(labels_sorted):
        exp = variant_coeffs[label]
        fit = x[j]
        ratio = fit / exp if abs(exp) > 1e-15 else float('nan')
        print(f"  {label:>35} {exp:>15.6f} {fit:>15.6f} {ratio:>10.4f}")

    recon_fitted = A_mat @ x
    err_fitted = np.linalg.norm(b_vec - recon_fitted) / np.linalg.norm(b_vec)
    print(f"\n  Fitted reconstruction error: {err_fitted:.4e}")

# ============================================================================
# Step 5: Probability comparison
# ============================================================================
print(f"\n{'='*60}")
print("Step 5: Probability comparison")
print(f"{'='*60}")

total_p_orig = sum(abs(orig_amps.get(ot, 0.0))**2 for ot in all_reachable)
total_p_recon = 0.0

recon_amps = {}
for ot in all_reachable:
    a_recon = sum(variant_coeffs[l] * variant_amps[l].get(ot, 0.0) for l in variant_amps)
    recon_amps[ot] = a_recon
    total_p_recon += abs(a_recon)**2

print(f"  Total P_orig = {total_p_orig:.8f}")
print(f"  Total P_recon = {total_p_recon:.8f}")

if total_p_orig > 0 and total_p_recon > 0:
    print(f"\n  {'Outcome':>25} {'P_orig':>10} {'P_recon':>10} {'Ratio':>10}")
    for ot in sorted(all_reachable):
        p_o = abs(orig_amps.get(ot, 0.0))**2 / total_p_orig
        p_r = abs(recon_amps.get(ot, 0.0))**2 / total_p_recon
        os_str = ''.join(str(b) for b in ot)
        if p_o > 1e-8 or p_r > 1e-8:
            ratio = p_r / p_o if p_o > 1e-12 else float('inf')
            print(f"  {os_str:>25} {p_o:10.6f} {p_r:10.6f} {ratio:10.4f}")

    tvd = 0.5 * sum(
        abs(abs(orig_amps.get(ot, 0.0))**2 / total_p_orig -
            abs(recon_amps.get(ot, 0.0))**2 / total_p_recon)
        for ot in all_reachable
    )
    print(f"\n  TVD = {tvd:.6e}")
