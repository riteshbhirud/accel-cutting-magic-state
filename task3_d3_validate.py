"""Task 3: Validate cross-term decomposition on d=3 circuit.

d=3: 15 qubits, 1 injection (T_DAG on qubit 3) + 1 cultivation (T_DAG/T on 7 qubits)
→ 8 Clifford variants (2 inj × 4 cult)

Strategy:
  1. Get deterministic outcomes from each Clifford variant via stim
  2. Compute cross-term probability P(m) = |Σ cᵢ ⟨m|φᵢ⟩|²
  3. Compare with tsim's sampling from original circuit
"""
import sys, os, re
from pathlib import Path
from copy import deepcopy
from collections import Counter, defaultdict
import numpy as np

os.environ["JAX_PLATFORMS"] = "cpu"

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))

import tsim
import pyzx_param as zx
from stab_rank_cut import tcount
import stim

# ============================================================================
# Load d=3 circuit
# ============================================================================
D3_PATH = Path(os.path.join(_PRX_ROOT, "gidney-circuits", "circuits", ""
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

# Block positions
d3_injection_line = 36
d3_cult_tdag_line = 78
d3_cult_t_line = 100
d3_cult_qubits = [0, 3, 7, 9, 10, 12, 13]
n_cult = len(d3_cult_qubits)

print("d=3 circuit structure:")
print(f"  Injection: line {d3_injection_line}: {t_lines[d3_injection_line].strip()}")
print(f"  Cult T†:   line {d3_cult_tdag_line}: {t_lines[d3_cult_tdag_line].strip()}")
print(f"  Cult T:    line {d3_cult_t_line}: {t_lines[d3_cult_t_line].strip()}")


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
    """Spider cutting coefficient.

    T = (1/√2)(I + e^{iπ/4} Z)  →  Z branch gets e^{iπ/4}
    T† = (1/√2)(I + e^{-iπ/4} Z) → Z branch gets e^{-iπ/4}

    For n qubits: T^⊗n = (1/√2)^n (I^⊗n + e^{inπ/4} Z^⊗n)  [only 2 terms for spider]
    For T†^⊗n = (1/√2)^n (I^⊗n + e^{-inπ/4} Z^⊗n)

    Cultivation block: T†^⊗n ⊗ T^⊗n with cascade in between
    Spider cutting of T†^⊗n: (1/√2)^n × (I if mode=I, e^{-inπ/4} Z^⊗n if mode=Z)
    Spider cutting of T^⊗n:  (1/√2)^n × (I if mode=I, e^{inπ/4} Z^⊗n if mode=Z)
    """
    n = n_cult  # 7

    # T†^⊗n spider: coefficient is (1/√2)^n × phase
    if cult_tdag == "I":
        coeff_tdag = (1/np.sqrt(2))**n
    else:  # "Z"
        coeff_tdag = (1/np.sqrt(2))**n * np.exp(-1j * n * np.pi / 4)

    # T^⊗n spider: coefficient is (1/√2)^n × phase
    if cult_t == "I":
        coeff_t = (1/np.sqrt(2))**n
    else:  # "Z"
        coeff_t = (1/np.sqrt(2))**n * np.exp(1j * n * np.pi / 4)

    # Injection T†: single qubit
    if inj == "I":
        coeff_inj = 1/np.sqrt(2)
    else:  # "Z"
        coeff_inj = (1/np.sqrt(2)) * np.exp(-1j * np.pi / 4)

    return coeff_tdag * coeff_t * coeff_inj


modes = ["I", "Z"]

# ============================================================================
# Step 1: Get deterministic outcomes from each Clifford variant
# ============================================================================
print(f"\n{'='*60}")
print("Step 1: Deterministic outcomes from Clifford variants")
print(f"{'='*60}")

variant_data = []

for inj in modes:
    for cult_tdag in modes:
        for cult_t in modes:
            text = build_variant(t_lines, inj, cult_tdag, cult_t)
            coeff = compute_coefficient(inj, cult_tdag, cult_t)
            label = f"inj={inj},cult=({cult_tdag},{cult_t})"

            # Also sample from stim to get the outcome
            try:
                stim_circ = stim.Circuit(text)
                sampler = stim_circ.compile_sampler()
                samples = sampler.sample(10)
                # All samples should be identical (deterministic Clifford)
                outcomes = [''.join(str(int(b)) for b in row) for row in samples]
                unique = set(outcomes)
                outcome = outcomes[0]

                variant_data.append({
                    'label': label,
                    'coeff': coeff,
                    'outcome': outcome,
                    'n_unique': len(unique),
                })

                print(f"  {label}: outcome={outcome}, unique={len(unique)}, "
                      f"|coeff|={abs(coeff):.4e}, phase={np.angle(coeff)/np.pi:.3f}π")

                if len(unique) > 1:
                    print(f"    WARNING: multiple unique outcomes! {unique}")

            except Exception as e:
                print(f"  {label}: Error — {e}")

# ============================================================================
# Step 2: Cross-term probabilities
# ============================================================================
print(f"\n{'='*60}")
print("Step 2: Cross-term probabilities P(m) = |Σ cᵢ δ(m,mᵢ)|²")
print(f"{'='*60}")

outcome_groups = defaultdict(list)
for v in variant_data:
    outcome_groups[v['outcome']].append(v)

print(f"Unique outcomes across all variants: {len(outcome_groups)}")

cross_probs = {}
for outcome in sorted(outcome_groups.keys()):
    group = outcome_groups[outcome]
    total_amp = sum(v['coeff'] for v in group)
    prob = abs(total_amp)**2
    cross_probs[outcome] = prob

    print(f"\n  Outcome: {outcome}")
    for v in group:
        print(f"    {v['label']}: coeff = {v['coeff']:.6f}")
    print(f"    Σ cᵢ = {total_amp:.6f}")
    print(f"    P(m) = |Σ cᵢ|² = {prob:.6e}")

total_cross = sum(cross_probs.values())
print(f"\n  Total probability (unnormalized): {total_cross:.6e}")

if total_cross > 0:
    print("\n  Normalized probabilities:")
    for outcome, prob in sorted(cross_probs.items(), key=lambda x: -x[1]):
        print(f"    {outcome}: {prob/total_cross:.6f}")


# ============================================================================
# Step 3: Compare with tsim sampling
# ============================================================================
print(f"\n{'='*60}")
print("Step 3: tsim sampling from original d=3 circuit")
print(f"{'='*60}")

orig_circuit = tsim.Circuit(t_text)
print(f"  Original: {orig_circuit.num_qubits} qubits")
g = orig_circuit.get_graph()
g2 = deepcopy(g)
zx.full_reduce(g2, paramSafe=True)
print(f"  T-count: raw={tcount(g)}, reduced={tcount(g2)}")

# Sample
print("  Sampling with tsim (10000 shots)...")
try:
    n_shots = 10000
    samples = orig_circuit.sample(n_shots)
    samples_arr = np.array(samples)
    print(f"  Samples shape: {samples_arr.shape}")

    outcomes_tsim = [''.join(str(int(b)) for b in row) for row in samples_arr]
    counter = Counter(outcomes_tsim)
    print(f"  Unique outcomes: {len(counter)}")

    print(f"\n  tsim outcome distribution:")
    for outcome, count in counter.most_common(20):
        p = count / n_shots
        print(f"    {outcome}: {p:.4f} ({count}/{n_shots})")

    # Compare
    print(f"\n{'='*60}")
    print("Step 4: Comparison")
    print(f"{'='*60}")

    all_outcomes = sorted(set(list(counter.keys()) + list(cross_probs.keys())))
    print(f"\n  {'Outcome':>25} {'P_tsim':>10} {'P_cross':>10} {'Match':>8}")
    print(f"  {'-'*53}")

    for outcome in all_outcomes:
        p_tsim = counter.get(outcome, 0) / n_shots
        p_cross = cross_probs.get(outcome, 0) / total_cross if total_cross > 0 else 0
        match = "✓" if abs(p_tsim - p_cross) < 0.05 else "✗"
        print(f"  {outcome:>25} {p_tsim:10.4f} {p_cross:10.4f} {match:>8}")

except Exception as e:
    print(f"  tsim error: {e}")
    import traceback
import os
_PRX_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    traceback.print_exc()


# ============================================================================
# Step 5: Also test with stim sampling from reference (noiseless with S gates)
# ============================================================================
print(f"\n{'='*60}")
print("Step 5: Reference — stim sampling from noiseless circuit (S gates = Clifford)")
print(f"{'='*60}")

# The original circuit with S/S_DAG (not T/T_DAG) is pure Clifford
# This is NOT the same physics, but gives us a baseline
try:
    noiseless_text = '\n'.join(noiseless)
    stim_orig = stim.Circuit(noiseless_text)
    sampler = stim_orig.compile_sampler()
    samples_ref = sampler.sample(10000)

    outcomes_ref = [''.join(str(int(b)) for b in row) for row in samples_ref]
    counter_ref = Counter(outcomes_ref)

    print(f"  S-gate circuit (Clifford baseline):")
    print(f"  Unique outcomes: {len(counter_ref)}")
    for outcome, count in counter_ref.most_common(10):
        print(f"    {outcome}: {count/10000:.4f}")
except Exception as e:
    print(f"  Error: {e}")


# ============================================================================
# Step 6: Coefficient analysis
# ============================================================================
print(f"\n{'='*60}")
print("Step 6: Coefficient structure analysis")
print(f"{'='*60}")

print(f"\n  All 8 coefficients:")
for v in variant_data:
    c = v['coeff']
    print(f"    {v['label']:30s}: {c:+.6f} = {abs(c):.4e} × e^(i{np.angle(c)/np.pi:.3f}π)")

print(f"\n  Note: all |cᵢ| = (1/√2)^(1+7+7) = (1/√2)^15 = {(1/np.sqrt(2))**15:.6e}")
print(f"  Computed |cᵢ| = {abs(variant_data[0]['coeff']):.6e}")

# Check if phases match expected pattern
print(f"\n  Phase structure:")
print(f"  T† spider (n=7): e^(-i7π/4) = e^(-i7π/4) = e^(iπ/4)  [mod 2π]")
print(f"    = {np.exp(-1j*7*np.pi/4):.6f}")
print(f"  T spider (n=7):  e^(i7π/4) = e^(-iπ/4)  [mod 2π]")
print(f"    = {np.exp(1j*7*np.pi/4):.6f}")
print(f"  Inj T† (n=1):    e^(-iπ/4)")
print(f"    = {np.exp(-1j*np.pi/4):.6f}")
