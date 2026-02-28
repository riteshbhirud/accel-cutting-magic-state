"""Task 1: Exact state vector simulation of d=3 and Clifford variants.

15 qubits = 32K state vector elements. Brute-force simulation is trivial.
Compare exact probability distributions of original vs cross-term sum.
"""
import sys, os, re
from pathlib import Path
import numpy as np
from collections import defaultdict

os.environ["JAX_PLATFORMS"] = "cpu"
_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))

# Gate matrices
I2 = np.eye(2, dtype=complex)
X_gate = np.array([[0, 1], [1, 0]], dtype=complex)
Z_gate = np.array([[1, 0], [0, -1]], dtype=complex)
H_gate = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
S_gate = np.array([[1, 0], [0, 1j]], dtype=complex)
Sd_gate = np.array([[1, 0], [0, -1j]], dtype=complex)
T_gate = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
Td_gate = np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]], dtype=complex)

class StateVecSim:
    """Simple state vector simulator for n qubits."""
    def __init__(self, n):
        self.n = n
        self.state = np.zeros(2**n, dtype=complex)
        self.state[0] = 1.0  # |0...0⟩
        self.measurements = []  # (bit_index, outcome) pairs

    def apply_1q(self, gate, qubit):
        """Apply single-qubit gate."""
        n = self.n
        state = self.state.reshape([2] * n)
        axes = list(range(n))
        # Move target qubit to last position
        axes.remove(qubit)
        axes.append(qubit)
        state = np.transpose(state, axes)
        # Apply gate
        state = np.tensordot(state, gate, axes=([n-1], [1]))
        # Move back
        inv_axes = [0] * n
        for i, a in enumerate(axes):
            inv_axes[a] = i
        inv_axes[-1] = n  # tensordot puts result last...
        # Actually simpler approach: apply gate via reshape
        self.state = self.state.reshape([2] * n)
        # Contract on qubit axis
        new_state = np.zeros_like(self.state)
        idx_0 = [slice(None)] * n
        idx_1 = [slice(None)] * n
        for out_val in range(2):
            idx_out = [slice(None)] * n
            idx_out[qubit] = out_val
            for in_val in range(2):
                idx_in = [slice(None)] * n
                idx_in[qubit] = in_val
                new_state[tuple(idx_out)] += gate[out_val, in_val] * self.state[tuple(idx_in)]
        self.state = new_state.reshape(-1)

    def apply_cx(self, control, target):
        """Apply CNOT (CX) gate."""
        n = self.n
        state = self.state.reshape([2] * n)
        new_state = state.copy()
        # When control=1, flip target
        idx_c1_t0 = [slice(None)] * n
        idx_c1_t1 = [slice(None)] * n
        idx_c1_t0[control] = 1
        idx_c1_t0[target] = 0
        idx_c1_t1[control] = 1
        idx_c1_t1[target] = 1
        new_state[tuple(idx_c1_t0)] = state[tuple(idx_c1_t1)]
        new_state[tuple(idx_c1_t1)] = state[tuple(idx_c1_t0)]
        self.state = new_state.reshape(-1)

    def measure_z(self, qubit):
        """Measure qubit in Z basis. Collapse state, record outcome."""
        n = self.n
        state = self.state.reshape([2] * n)
        # Probability of |0⟩
        idx_0 = [slice(None)] * n
        idx_0[qubit] = 0
        p0 = np.sum(np.abs(state[tuple(idx_0)])**2)

        # Choose outcome (deterministic if p0=0 or p0=1, else random)
        if p0 > 1 - 1e-12:
            outcome = 0
        elif p0 < 1e-12:
            outcome = 1
        else:
            outcome = 0 if np.random.random() < p0 else 1

        # Project
        idx_other = [slice(None)] * n
        idx_other[qubit] = 1 - outcome
        state[tuple(idx_other)] = 0

        # Renormalize
        norm = np.linalg.norm(state)
        if norm > 1e-15:
            state /= norm
        self.state = state.reshape(-1)
        self.measurements.append(outcome)
        return outcome

    def measure_x(self, qubit):
        """Measure qubit in X basis = H then Z-measure then H."""
        self.apply_1q(H_gate, qubit)
        outcome = self.measure_z(qubit)
        return outcome

    def reset_z(self, qubit):
        """Reset qubit to |0⟩ (R instruction)."""
        n = self.n
        state = self.state.reshape([2] * n)
        # Measure (project to definite outcome)
        idx_0 = [slice(None)] * n
        idx_0[qubit] = 0
        p0 = np.sum(np.abs(state[tuple(idx_0)])**2)
        if p0 < 1e-12:
            # Qubit is in |1⟩, flip to |0⟩
            self.apply_1q(X_gate, qubit)
        elif p0 > 1 - 1e-12:
            pass  # Already |0⟩
        else:
            # Mixed, project to |0⟩ and renormalize
            idx_1 = [slice(None)] * n
            idx_1[qubit] = 1
            state[tuple(idx_1)] = 0
            norm = np.linalg.norm(state)
            if norm > 1e-15:
                state /= norm
            self.state = state.reshape(-1)

    def reset_x(self, qubit):
        """Reset qubit to |+⟩ (RX instruction)."""
        self.reset_z(qubit)
        self.apply_1q(H_gate, qubit)

    def get_measurement_probs(self):
        """Get probability of each computational basis state."""
        return np.abs(self.state)**2


def simulate_circuit(circ_lines, n_qubits, use_t_gates=True):
    """Simulate a circuit given as list of instruction strings.

    Returns: (measurement_outcomes, final_state_probs)
    """
    sim = StateVecSim(n_qubits)

    gate_map = {
        'H': H_gate, 'X': X_gate, 'Z': Z_gate,
        'S': S_gate, 'S_DAG': Sd_gate,
    }
    if use_t_gates:
        gate_map['T'] = T_gate
        gate_map['T_DAG'] = Td_gate
    else:
        gate_map['T'] = S_gate  # Fallback (shouldn't be used)
        gate_map['T_DAG'] = Sd_gate

    for line in circ_lines:
        s = line.strip()
        if not s or s.startswith('TICK') or s.startswith('DETECTOR') or \
           s.startswith('OBSERVABLE') or s.startswith('QUBIT_COORDS') or \
           s.startswith('REPEAT') or s.startswith('}'):
            continue

        parts = s.split()
        gate = parts[0]
        args = [int(x) for x in parts[1:] if x.isdigit() or (x.startswith('-') and x[1:].isdigit())]

        if gate in gate_map:
            for q in args:
                sim.apply_1q(gate_map[gate], q)
        elif gate == 'CX':
            for i in range(0, len(args), 2):
                sim.apply_cx(args[i], args[i+1])
        elif gate == 'M':
            for q in args:
                sim.measure_z(q)
        elif gate == 'MX':
            for q in args:
                sim.measure_x(q)
        elif gate == 'R':
            for q in args:
                sim.reset_z(q)
        elif gate == 'RX':
            for q in args:
                sim.reset_x(q)
        elif gate in ('CZ', 'CY', 'SWAP', 'ISWAP', 'SQRT_X', 'SQRT_X_DAG'):
            raise ValueError(f"Gate {gate} not implemented")
        # Ignore unknown instructions

    return sim.measurements, sim.state


# ============================================================================
# Load d=3 circuit
# ============================================================================
CIRCUIT_PATH = Path("/Users/ritesh/Downloads/prx/gidney-circuits/circuits/"
                    "for_perfectionist_decoding/"
                    "c=inject[unitary]+cultivate,p=0.001,noise=uniform,"
                    "g=css,q=15,b=Y,r=4,d1=3.stim")
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

def replace_s_with_t(s):
    s = re.sub(r'^(\s*)S_DAG(\s)', r'\1T_DAG\2', s, flags=re.MULTILINE)
    return re.sub(r'^(\s*)S(\s)', r'\1T\2', s, flags=re.MULTILINE)

t_text = replace_s_with_t('\n'.join(noiseless))
t_lines = t_text.split('\n')

n_qubits = 15

# Block definitions from previous analysis
injection_line = 36
cult_start, cult_end = 78, 100
cult_qubits = [0, 3, 7, 9, 10, 12, 13]

def make_xsd_lines(data_qubits):
    return [
        f"S_DAG {' '.join(str(q) for q in data_qubits)}",
        "TICK",
        f"X {' '.join(str(q) for q in data_qubits)}",
        "TICK",
    ]

def build_variant_lines(t_lines, cult_mode, inj_mode):
    result = []
    skip = set()
    if inj_mode != "keep":
        skip.add(injection_line)
    if cult_mode != "keep":
        for i in range(cult_start, cult_end + 1):
            skip.add(i)

    for i, line in enumerate(t_lines):
        if i in skip:
            if i == injection_line and inj_mode != "keep":
                if inj_mode == "identity":
                    result.append("TICK")
                elif inj_mode == "xsd":
                    result.extend(make_xsd_lines([3]))
            elif i == cult_start and cult_mode != "keep":
                if cult_mode == "identity":
                    result.append("TICK")
                elif cult_mode == "xsd":
                    result.extend(make_xsd_lines(cult_qubits))
            continue
        result.append(line)
    return result

# ============================================================================
# Run exact simulation: sample many times for ORIGINAL circuit
# (measurements introduce randomness)
# ============================================================================
print("="*60)
print(f"Exact state vector simulation of d=3 ({n_qubits} qubits)")
print("="*60)

n_trials = 10000
orig_counter = defaultdict(int)

print(f"\nSimulating original circuit ({n_trials} trials)...")
for trial in range(n_trials):
    measurements, state = simulate_circuit(t_lines, n_qubits, use_t_gates=True)
    outcome = ''.join(str(m) for m in measurements)
    orig_counter[outcome] += 1

print(f"Unique outcomes: {len(orig_counter)}")
print("Top outcomes:")
for outcome, count in sorted(orig_counter.items(), key=lambda x: -x[1])[:10]:
    print(f"  {outcome}: {count/n_trials:.4f}")

# ============================================================================
# Simulate Clifford variants (these are deterministic except for MX randomness)
# ============================================================================
print(f"\n{'='*60}")
print("Simulating 4 Clifford variants")
print(f"{'='*60}")

variants = [
    ("I⊗I", "identity", "identity"),
    ("I⊗XS†", "identity", "xsd"),
    ("XS†⊗I", "xsd", "identity"),
    ("XS†⊗XS†", "xsd", "xsd"),
]

variant_counters = {}
for name, cult_m, inj_m in variants:
    lines_v = build_variant_lines(t_lines, cult_m, inj_m)
    counter = defaultdict(int)

    for trial in range(1000):  # Fewer trials for Clifford (mostly deterministic)
        measurements, state = simulate_circuit(lines_v, n_qubits, use_t_gates=False)
        outcome = ''.join(str(m) for m in measurements)
        counter[outcome] += 1

    variant_counters[name] = counter
    print(f"\n  {name}:")
    for outcome, count in sorted(counter.items(), key=lambda x: -x[1]):
        print(f"    {outcome}: {count/1000:.4f}")

# ============================================================================
# Compare distributions
# ============================================================================
print(f"\n{'='*60}")
print("Distribution comparison")
print(f"{'='*60}")

# Collect all outcomes
all_outcomes = set(orig_counter.keys())
for c in variant_counters.values():
    all_outcomes.update(c.keys())

print(f"\nAll unique outcomes: {len(all_outcomes)}")
print(f"\n{'Outcome':>25} {'P_orig':>8}", end="")
for name, _, _ in variants:
    print(f" {name:>10}", end="")
print()

for outcome in sorted(all_outcomes):
    p_orig = orig_counter.get(outcome, 0) / n_trials
    print(f"{outcome:>25} {p_orig:>8.4f}", end="")
    for name, _, _ in variants:
        p_var = variant_counters[name].get(outcome, 0) / 1000
        print(f" {p_var:>10.4f}", end="")
    print()
