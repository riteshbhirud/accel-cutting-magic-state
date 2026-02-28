"""Task 1: Numerically verify decomposition on the ACTUAL first cultivation block.

Extract the first cultivation block from the Gidney d=5 circuit:
  Qubits involved: {0,1,2,3,6,7,8,9,11,12,13,17,18}
  Data qubits (get T†/T): {0,3,7,9,11,13,17}
  Hub qubit (measured MX): 7

Remap to contiguous indices 0-12 for tractable tensor computation.
Compute operator for each MX outcome and check decomposition.
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

# The first cultivation block (lines 105-127 of noiseless circuit):
# Line 105: S_DAG 0 3 7 9 11 13 17        → T_DAG on data qubits
# Line 107: CX 1 0 2 3 6 7 8 9 12 11 18 17  → fan-in layer 1
# Line 109: CX 3 1 7 8 12 18               → fan-in layer 2
# Line 111: CX 7 3 12 13                    → fan-in layer 3
# Line 113: CX 7 12                         → fan-in layer 4
# Line 115: MX 7                            → measure hub
# Line 117: RX 7                            → reset hub
# Line 119-125: reverse cascade (fan-out)
# Line 127: S 0 3 7 9 11 13 17             → T on data qubits

# Qubit mapping: original → new
orig_qubits = sorted([0, 1, 2, 3, 6, 7, 8, 9, 11, 12, 13, 17, 18])
qmap = {q: i for i, q in enumerate(orig_qubits)}
n_qubits = len(orig_qubits)

print(f"Qubit mapping (original → new):")
for q, i in sorted(qmap.items()):
    role = "DATA+HUB" if q == 7 else "DATA" if q in [0,3,9,11,13,17] else "helper"
    print(f"  {q:2d} → {i:2d}  ({role})")

data_qubits_new = sorted([qmap[q] for q in [0, 3, 7, 9, 11, 13, 17]])
hub_new = qmap[7]
print(f"\nData qubits (new): {data_qubits_new}")
print(f"Hub qubit (new): {hub_new}")
print(f"Total qubits: {n_qubits}")

def remap_cx(cx_str):
    """Remap CX instruction to new qubit indices."""
    parts = cx_str.split()
    assert parts[0] == "CX"
    nums = [int(x) for x in parts[1:]]
    pairs = [(nums[i], nums[i+1]) for i in range(0, len(nums), 2)]
    remapped_pairs = [(qmap[c], qmap[t]) for c, t in pairs]
    args = " ".join(f"{c} {t}" for c, t in remapped_pairs)
    return f"CX {args}"

# Build the cultivation block circuit with remapped qubits
data_str = " ".join(str(qmap[q]) for q in [0, 3, 7, 9, 11, 13, 17])

block_lines = [
    f"T_DAG {data_str}",
    "TICK",
    remap_cx("CX 1 0 2 3 6 7 8 9 12 11 18 17"),
    "TICK",
    remap_cx("CX 3 1 7 8 12 18"),
    "TICK",
    remap_cx("CX 7 3 12 13"),
    "TICK",
    remap_cx("CX 7 12"),
    "TICK",
    f"MX {hub_new}",
    "TICK",
    f"RX {hub_new}",
    "TICK",
    remap_cx("CX 7 12"),
    "TICK",
    remap_cx("CX 7 3 12 13"),
    "TICK",
    remap_cx("CX 3 1 7 8 12 18"),
    "TICK",
    remap_cx("CX 1 0 2 3 6 7 8 9 12 11 18 17"),
    "TICK",
    f"T {data_str}",
]

# Add final measurements
block_lines.append("TICK")
block_lines.append(f"M {' '.join(str(i) for i in range(n_qubits))}")

block_circ_str = "\n".join(block_lines)
print(f"\nCultivation block circuit (remapped):")
print(block_circ_str)

# Check T-count
circuit = tsim.Circuit(block_circ_str)
g = circuit.get_graph()
g2 = deepcopy(g)
zx.full_reduce(g2, paramSafe=True)
print(f"\nZX T-count: raw={tcount(g)}, reduced={tcount(g2)}")

# Compute tensor
print(f"\nComputing tensor ({n_qubits} qubits = 2^{n_qubits} = {2**n_qubits} dims)...")
tensor = np.array(circuit.to_tensor())
print(f"Tensor shape: {tensor.shape}")

# The tensor has shape (2,)*2n for n measurement outputs + n initial states
# For 13 qubits with all measured: shape (2,)*26
# First 13 indices: measurement outcomes
# Last 13 indices: input states

# For all inputs = |0⟩: slice [..., 0, 0, ..., 0]
n = n_qubits
input_zero = tuple([slice(None)] * n + [0] * n)
amp_zero_input = tensor[input_zero]
print(f"Amplitudes for |0⟩^⊗{n} input: shape {amp_zero_input.shape}")

# The hub qubit (index hub_new) is measured by MX (X-basis)
# Other qubits are measured by M (Z-basis)
# For the decomposition, we care about the operator on DATA qubits
# conditioned on MX outcome of hub qubit

# Let's look at which measurement outcomes have non-zero amplitude
amp_flat = amp_zero_input.reshape(-1)
nonzero_outcomes = np.where(np.abs(amp_flat) > 1e-10)[0]
print(f"\nNon-zero outcomes for |0⟩^{n} input: {len(nonzero_outcomes)} out of {2**n}")

for idx in nonzero_outcomes[:20]:
    bits = format(idx, f'0{n}b')
    print(f"  |{bits}⟩: amp = {amp_flat[idx]:.6f}, prob = {abs(amp_flat[idx])**2:.6f}")

# ============================================================================
# Now build identity and XS† variants of just this block
# ============================================================================
print(f"\n{'='*60}")
print("Identity variant")
print(f"{'='*60}")

# Identity: remove entire block, just measure
id_lines = ["TICK"] * (len(block_lines) - 2)  # Skip everything except final M
id_lines.append("TICK")
id_lines.append(f"M {' '.join(str(i) for i in range(n_qubits))}")
id_circ_str = "\n".join(id_lines)

id_circuit = tsim.Circuit(id_circ_str)
id_tensor = np.array(id_circuit.to_tensor())
id_amp = id_tensor[input_zero].reshape(-1)

print(f"Non-zero outcomes: {np.count_nonzero(np.abs(id_amp) > 1e-10)}")

# XS† variant: replace block with S_DAG + X on data qubits
print(f"\n{'='*60}")
print("XS† variant")
print(f"{'='*60}")

xsd_lines = [f"S_DAG {data_str}", "TICK", f"X {data_str}", "TICK"]
xsd_lines.extend(["TICK"] * (len(block_lines) - 4))
xsd_lines.append(f"M {' '.join(str(i) for i in range(n_qubits))}")
xsd_circ_str = "\n".join(xsd_lines)

xsd_circuit = tsim.Circuit(xsd_circ_str)
xsd_tensor = np.array(xsd_circuit.to_tensor())
xsd_amp = xsd_tensor[input_zero].reshape(-1)

print(f"Non-zero outcomes: {np.count_nonzero(np.abs(xsd_amp) > 1e-10)}")

# ============================================================================
# Least squares fit: orig = a·id + b·xsd
# ============================================================================
print(f"\n{'='*60}")
print("Decomposition fit: orig = a·id + b·xsd")
print(f"{'='*60}")

A = np.column_stack([id_amp, xsd_amp])
coeffs, residuals, rank, sv = np.linalg.lstsq(A, amp_flat, rcond=None)
a, b = coeffs

recon = a * id_amp + b * xsd_amp
error = np.linalg.norm(amp_flat - recon) / max(np.linalg.norm(amp_flat), 1e-15)

print(f"a = {a}")
print(f"b = {b}")
print(f"|a| = {abs(a):.8f}, |b| = {abs(b):.8f}")
print(f"Reconstruction error: {error:.2e}")
print(f"Match: {error < 1e-8}")

# ============================================================================
# Try: fit per measurement outcome (condition on hub outcome)
# ============================================================================
print(f"\n{'='*60}")
print("Per-hub-outcome decomposition")
print(f"{'='*60}")

# Hub qubit is at index hub_new in the measurement outcomes
# Group outcomes by hub bit value
hub_bit_idx = hub_new  # bit position in the output

for hub_val in [0, 1]:
    print(f"\n--- Hub MX outcome = {hub_val} ---")
    # Select outcomes where hub bit = hub_val
    mask = np.zeros(2**n, dtype=bool)
    for idx in range(2**n):
        bits = (idx >> (n - 1 - hub_bit_idx)) & 1
        if bits == hub_val:
            mask[idx] = True

    orig_sub = amp_flat[mask]
    id_sub = id_amp[mask]
    xsd_sub = xsd_amp[mask]

    if np.linalg.norm(orig_sub) < 1e-15:
        print("  All zero for this outcome")
        continue

    A_sub = np.column_stack([id_sub, xsd_sub])
    c_sub, _, _, _ = np.linalg.lstsq(A_sub, orig_sub, rcond=None)
    recon_sub = A_sub @ c_sub
    err_sub = np.linalg.norm(orig_sub - recon_sub) / np.linalg.norm(orig_sub)

    print(f"  a = {c_sub[0]:.6f}, b = {c_sub[1]:.6f}")
    print(f"  |a| = {abs(c_sub[0]):.6f}, |b| = {abs(c_sub[1]):.6f}")
    print(f"  Error: {err_sub:.2e}")

# ============================================================================
# Try: more general decomposition with Z^⊗n on data qubits
# ============================================================================
print(f"\n{'='*60}")
print("General decomposition: orig = a·id + b·xsd + c·z + d·other")
print(f"{'='*60}")

# Z variant: replace block with Z on data qubits
z_lines = [f"Z {data_str}", "TICK"]
z_lines.extend(["TICK"] * (len(block_lines) - 3))
z_lines.append(f"M {' '.join(str(i) for i in range(n_qubits))}")
z_circ_str = "\n".join(z_lines)

z_circuit = tsim.Circuit(z_circ_str)
z_tensor = np.array(z_circuit.to_tensor())
z_amp = z_tensor[input_zero].reshape(-1)

# S_DAG variant (just S† on data, no X)
sd_lines = [f"S_DAG {data_str}", "TICK"]
sd_lines.extend(["TICK"] * (len(block_lines) - 3))
sd_lines.append(f"M {' '.join(str(i) for i in range(n_qubits))}")
sd_circ_str = "\n".join(sd_lines)

sd_circuit = tsim.Circuit(sd_circ_str)
sd_tensor = np.array(sd_circuit.to_tensor())
sd_amp = sd_tensor[input_zero].reshape(-1)

# 4-term fit
A4 = np.column_stack([id_amp, xsd_amp, z_amp, sd_amp])
c4, _, _, _ = np.linalg.lstsq(A4, amp_flat, rcond=None)
recon4 = A4 @ c4
err4 = np.linalg.norm(amp_flat - recon4) / max(np.linalg.norm(amp_flat), 1e-15)

print(f"Coefficients: id={c4[0]:.6f}, xsd={c4[1]:.6f}, z={c4[2]:.6f}, sd={c4[3]:.6f}")
print(f"Error: {err4:.2e}")

# What about: id, xsd, x (X on data), s (S on data)?
x_lines = [f"X {data_str}", "TICK"]
x_lines.extend(["TICK"] * (len(block_lines) - 3))
x_lines.append(f"M {' '.join(str(i) for i in range(n_qubits))}")
x_circ_str = "\n".join(x_lines)

x_circuit = tsim.Circuit(x_circ_str)
x_tensor = np.array(x_circuit.to_tensor())
x_amp = x_tensor[input_zero].reshape(-1)

# S variant
s_lines = [f"S {data_str}", "TICK"]
s_lines.extend(["TICK"] * (len(block_lines) - 3))
s_lines.append(f"M {' '.join(str(i) for i in range(n_qubits))}")
s_circ_str = "\n".join(s_lines)

s_circuit = tsim.Circuit(s_circ_str)
s_tensor = np.array(s_circuit.to_tensor())
s_amp = s_tensor[input_zero].reshape(-1)

A6 = np.column_stack([id_amp, xsd_amp, z_amp, sd_amp, x_amp, s_amp])
c6, _, _, _ = np.linalg.lstsq(A6, amp_flat, rcond=None)
recon6 = A6 @ c6
err6 = np.linalg.norm(amp_flat - recon6) / max(np.linalg.norm(amp_flat), 1e-15)

print(f"\n6-term fit:")
labels = ["id", "xsd", "z", "sd", "x", "s"]
for label, coeff in zip(labels, c6):
    print(f"  {label:4s}: {coeff:.6f} (|c|={abs(coeff):.6f})")
print(f"Error: {err6:.2e}")
