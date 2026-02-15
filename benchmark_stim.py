"""Stim Clifford-proxy throughput reference.

Times stim's compiled detector sampler on the S-gate (Clifford) version
of the d=3 cultivation circuit, for direct comparison with the exact
(non-Clifford) throughput reported in results.jsonl.

Note on batch size: stim samples all shots in a single call, allocating a
(shots x n_detectors) boolean array, so its throughput depends on call size.
At 2^29 (~536M) shots the array exceeds L3 capacity and throughput drops to
~4-5M shots/s (vs ~13-15M at 2^27). The exact sampler, by contrast, loops
over fixed 2M-shot batches, so its throughput is independent of total shot
count. We use 2^29 here as a representative large-scale stim measurement;
this may not be optimal for stim — smaller call sizes would yield higher
throughput.

Usage:
    python benchmark_stim.py
"""
import sys, time, re
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))

import stim
import numpy as np
import gen

from d_3_circuit_definitions import circuit_source_injection_T, circuit_source_projection_proj


def add_noise(*, circuit, noise_strength):
    if noise_strength <= 0:
        return circuit
    noise_model = gen.NoiseModel.uniform_depolarizing(noise_strength)
    return noise_model.noisy_circuit_skipping_mpp_boundaries(circuit)


def replace_t_with_s(s):
    s = re.sub(r'^(\s*)T_DAG(\s)', r'\1S_DAG\2', s, flags=re.MULTILINE)
    return re.sub(r'^(\s*)T(\s)', r'\1S\2', s, flags=re.MULTILINE)


# Build S-gate (Clifford) circuit
clifford_injection = replace_t_with_s(circuit_source_injection_T)
clifford_projection = replace_t_with_s(circuit_source_projection_proj)

# Noiseless sanity check
full_noiseless = stim.Circuit(clifford_injection + clifford_projection)
sampler = full_noiseless.compile_detector_sampler()
det, obs = sampler.sample(shots=10000, separate_observables=True)
assert np.sum(np.any(det, axis=1)) == 0, "Noiseless detectors fired"
noiseless_obs = int(obs[0, 0])
assert np.sum(obs[:, 0].astype(int) != noiseless_obs) == 0, "Noiseless obs inconsistent"
print(f"Noiseless sanity check: PASSED (obs={noiseless_obs})")

# Benchmark config
NOISE_STRENGTHS = [0.0002, 0.0005, 0.001, 0.002, 0.005]
SHOTS = 2**29  # ~536M (matches total shots used by the exact sampler)

print(f"\nStim benchmark: {SHOTS} shots per noise strength")
print(f"{'p':<8} {'Shots/s':>10} {'Time':>8} {'PSR':>8} {'Fidelity':>12}")
print(f"{'-'*8} {'-'*10} {'-'*8} {'-'*8} {'-'*12}")

for p in NOISE_STRENGTHS:
    noisy_inj = add_noise(circuit=stim.Circuit(clifford_injection), noise_strength=p)
    full_noisy = stim.Circuit(str(noisy_inj) + "\n" + clifford_projection)
    sampler = full_noisy.compile_detector_sampler()

    # Warmup
    sampler.sample(shots=10000, separate_observables=True)

    t0 = time.time()
    det, obs = sampler.sample(shots=SHOTS, separate_observables=True)
    t_sample = time.time() - t0

    trivial = np.all(det == 0, axis=1)
    n_kept = int(np.sum(trivial))
    n_errors = int(np.sum(obs[trivial, 0].astype(int) != noiseless_obs))
    psr = n_kept / SHOTS
    fidelity = 1.0 - n_errors / n_kept if n_kept > 0 else 0
    throughput = SHOTS / t_sample

    print(f"{p:<8} {throughput:>10.0f} {t_sample:>7.3f}s {psr:>8.4f} {fidelity:>12.8f}")
