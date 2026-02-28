import stim
import numpy as np

D3 = "/Users/ritesh/Downloads/prx/gidney-circuits/circuits/for_perfectionist_decoding/c=inject[unitary]+cultivate,p=0.001,noise=uniform,g=css,q=15,b=Y,r=4,d1=3.stim"

circuit = stim.Circuit(open(D3).read())
flat = circuit.flattened()
instructions = list(flat)

mpp_instructions = [(i, inst) for i, inst in enumerate(instructions) if inst.name == "MPP"]
last_mpp_idx = mpp_instructions[-1][0]

prefix_circuit = stim.Circuit()
for inst in instructions[:last_mpp_idx]:
    prefix_circuit.append(inst)

# Run prefix
sim_base = stim.TableauSimulator()
sim_base.do(prefix_circuit)

# Unmeasured qubits for d=3 (from Task 7)
unmeasured_qubits = [0, 3, 9, 10, 12, 13]
n_unmeasured = len(unmeasured_qubits)  # = 6
n_outcomes = 2**n_unmeasured  # = 64

def compute_amplitude(sim_frozen, unmeasured_qubits, target_int):
    s = sim_frozen.copy()
    amplitude = 1.0 + 0j
    n_total = s.num_qubits

    for i, qubit in enumerate(unmeasured_qubits):
        target_bit = (target_int >> i) & 1

        # Build Z Pauli string on this qubit
        pauli = stim.PauliString(n_total)
        pauli[qubit] = 3  # Z

        expectation = s.peek_observable_expectation(pauli)

        if expectation == +1:
            if target_bit == 1:
                return 0.0 + 0j
        elif expectation == -1:
            if target_bit == 0:
                return 0.0 + 0j
        else:
            amplitude *= (1.0 / np.sqrt(2))
            result, kickback = s.measure_kickback(qubit)
            if result != target_bit:
                if kickback is not None:
                    s.do_pauli_string(kickback)

    return amplitude

# Compute all 64 amplitudes for d=3
print("Computing amplitudes for all 64 outcomes...")
amplitudes = np.zeros(n_outcomes, dtype=complex)
for m in range(n_outcomes):
    amplitudes[m] = compute_amplitude(sim_base, unmeasured_qubits, m)

# Verify
print(f"Sum |a|^2 = {np.sum(np.abs(amplitudes)**2):.6f}  (should be ~1.0)")
print(f"Nonzero amplitudes: {np.count_nonzero(np.abs(amplitudes) > 1e-6)} out of {n_outcomes}")
print(f"Max amplitude: {np.max(np.abs(amplitudes)):.6f}")

# Show the nonzero ones
print()
print("Nonzero amplitudes:")
for m in range(n_outcomes):
    a = amplitudes[m]
    if abs(a) > 1e-6:
        bits = format(m, f"0{n_unmeasured}b")
        print(f"  |{bits}> = {a.real:+.6f} {a.imag:+.6f}j  |a|^2 = {abs(a)**2:.6f}")
