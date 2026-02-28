"""Task 4b: High-statistics d=3 LER vs p sweep.

Uses 4M shots per noise point with small batches to avoid OOM.
This gives enough statistics for reliable LER estimates.
"""
import sys, time, re, json
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))

import os
os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
import tsim
import stim
import gen

from d_3_circuit_definitions import circuit_source_injection_T, circuit_source_projection_proj
from tsim_cutting import compile_detector_sampler_subcomp_enum_general

# ============================================================================
# Setup
# ============================================================================

def replace_t_with_s(s):
    s = re.sub(r'^(\s*)T_DAG(\s)', r'\1S_DAG\2', s, flags=re.MULTILINE)
    return re.sub(r'^(\s*)T(\s)', r'\1S\2', s, flags=re.MULTILINE)

def replace_s_with_t(c):
    p = str(c)
    p = re.sub(r'^(\s*)S_DAG(\s)', r'\1T_DAG\2', p, flags=re.MULTILINE)
    return re.sub(r'^(\s*)S(\s)', r'\1T\2', p, flags=re.MULTILINE)

def add_noise(*, circuit, noise_strength):
    if noise_strength <= 0:
        return circuit
    noise_model = gen.NoiseModel.uniform_depolarizing(noise_strength)
    return noise_model.noisy_circuit_skipping_mpp_boundaries(circuit)

clifford_source = replace_t_with_s(circuit_source_injection_T)
clifford_circuit = stim.Circuit(clifford_source)
c_projection = tsim.Circuit(circuit_source_projection_proj)

TOTAL_SHOTS = 4_194_304  # 2^22 = ~4M
BATCH = 4096
N_BATCHES = TOTAL_SHOTS // BATCH

noiseless_raw = 0

print(f"d=3 LER vs p sweep — {TOTAL_SHOTS} shots per point")
print(f"{'p':>8} {'PSR':>8} {'LER':>12} {'Err':>8} {'Kept':>10} {'Shots/s':>10} {'Compile':>8} {'Graphs':>6}")
print(f"{'-'*8} {'-'*8} {'-'*12} {'-'*8} {'-'*10} {'-'*10} {'-'*8} {'-'*6}")
sys.stdout.flush()

results = []

for p in [0.001, 0.0015, 0.002, 0.0025, 0.003, 0.004, 0.005]:
    noisy_clif = add_noise(circuit=clifford_circuit, noise_strength=p)
    noisy_str = replace_s_with_t(noisy_clif)
    c_inj = tsim.Circuit(noisy_str)
    circ_p = c_inj + c_projection

    t0 = time.time()
    sampler_p = compile_detector_sampler_subcomp_enum_general(
        circ_p, seed=42, max_cut_iterations=10
    )
    t_compile = time.time() - t0

    # Count graphs from repr
    sampler_str = str(sampler_p)

    # Warmup
    sampler_p.sample(shots=BATCH, batch_size=BATCH, separate_observables=True)

    # Sample in small batches
    all_det = []
    all_obs = []
    t0 = time.time()
    for _ in range(N_BATCHES):
        det_b, obs_b = sampler_p.sample(
            shots=BATCH, batch_size=BATCH, separate_observables=True
        )
        all_det.append(det_b)
        all_obs.append(obs_b)
    t_sample = time.time() - t0

    det = np.concatenate(all_det, axis=0)
    obs = np.concatenate(all_obs, axis=0)

    trivial = np.all(det == 0, axis=1)
    kept = int(np.sum(trivial))
    errs = int(np.sum(obs[trivial, 0].astype(int) != noiseless_raw))
    psr = kept / TOTAL_SHOTS
    ler = errs / kept if kept > 0 else 0
    tput = TOTAL_SHOTS / t_sample

    # Extract graph count from sampler repr
    graphs_str = "?"
    import re as re2
    m = re2.search(r'(\d+)\s*graphs', sampler_str)
    if m:
        graphs_str = m.group(1)

    results.append({
        'p': p, 'psr': psr, 'ler': ler, 'errs': errs, 'kept': kept,
        'shots': TOTAL_SHOTS, 'tput': tput, 'compile': t_compile,
    })

    print(f"{p:>8.4f} {psr:>8.4f} {ler:>12.2e} {errs:>8d} {kept:>10d} "
          f"{tput:>10.0f} {t_compile:>7.1f}s {graphs_str:>6}")
    sys.stdout.flush()
    del sampler_p

# Summary
print(f"\n{'=' * 60}")
print(f"d=3 LER vs p ({TOTAL_SHOTS} shots per point)")
print(f"{'=' * 60}")
for r in results:
    print(f"  p={r['p']:.4f}: PSR={r['psr']:.4f}, LER={r['ler']:.2e}, "
          f"errs={r['errs']}/{r['kept']}")

with open(os.path.join(str(_THIS_DIR), "d3_ler_results_highstats.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved to d3_ler_results_highstats.json")
