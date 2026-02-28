import stim
import numpy as np
import os
_PRX_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

D3 = os.path.join(_PRX_ROOT, "gidney-circuits", "circuits", "for_perfectionist_decoding", "c=inject[unitary]+cultivate,p=0.001,noise=uniform,g=css,q=15,b=Y,r=4,d1=3.stim"

circuit = stim.Circuit(open(D3).read())
flat = circuit.flattened()
instructions = list(flat)

# Split circuit at final MPP
mpp_instructions = [(i, inst) for i, inst in enumerate(instructions) if inst.name == "MPP"]
last_mpp_idx = mpp_instructions[-1][0]
final_mpp = mpp_instructions[-1][1]

prefix_circuit = stim.Circuit()
for inst in instructions[:last_mpp_idx]:
    prefix_circuit.append(inst)

# Unmeasured qubits before final MPP (from Task 7)
unmeasured_qubits = [0, 3, 9, 10, 12, 13]
n_unmeasured = len(unmeasured_qubits)

def compute_amplitudes(sim_state, unmeasured_qubits):
    """Compute amplitude vector over all 2^n_unmeasured outcomes."""
    n = len(unmeasured_qubits)
    n_outcomes = 2**n
    n_total = sim_state.num_qubits
    amplitudes = np.zeros(n_outcomes, dtype=complex)

    for m in range(n_outcomes):
        s = sim_state.copy()
        amp = 1.0 + 0j
        for i, qubit in enumerate(unmeasured_qubits):
            target_bit = (m >> i) & 1
            pauli = stim.PauliString(n_total)
            pauli[qubit] = 3  # Z
            expectation = s.peek_observable_expectation(pauli)
            if expectation == +1:
                if target_bit == 1:
                    amp = 0.0 + 0j
                    break
            elif expectation == -1:
                if target_bit == 0:
                    amp = 0.0 + 0j
                    break
            else:
                amp *= (1.0 / np.sqrt(2))
                result, kickback = s.measure_kickback(qubit)
                if result != target_bit and kickback is not None:
                    s.do_pauli_string(kickback)
        amplitudes[m] = amp
    return amplitudes

# Run N shots and collect statistics
N_SHOTS = 10000
print(f"Running {N_SHOTS} shots...")

totals = []
for shot in range(N_SHOTS):
    sim = stim.TableauSimulator()
    sim.do(prefix_circuit)
    amps = compute_amplitudes(sim, unmeasured_qubits)
    probs = np.abs(amps)**2
    total = probs.sum()
    totals.append(total)
    if abs(total - 1.0) > 0.01:
        print(f"WARNING shot {shot}: sum |a|^2 = {total:.4f}")
    if shot % 1000 == 0:
        print(f"  Shot {shot}: sum|a|^2 = {total:.6f}, nonzero = {np.count_nonzero(probs > 1e-6)}")

totals = np.array(totals)
print(f"\nAll {N_SHOTS} shots completed")
print(f"Min sum|a|^2 = {totals.min():.6f}")
print(f"Max sum|a|^2 = {totals.max():.6f}")
print(f"Mean sum|a|^2 = {totals.mean():.6f}")
print(f"Shots with sum|a|^2 exactly 1.0: {np.sum(totals == 1.0)}")
print(f"Shots with |sum|a|^2 - 1| > 0.01: {np.sum(np.abs(totals - 1.0) > 0.01)}")
