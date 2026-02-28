"""d=3 LER vs p sweep using the cutting pipeline."""
import sys, time, re, os
os.environ["JAX_PLATFORMS"] = "cpu"

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

import numpy as np
import tsim, stim, gen
from d_3_circuit_definitions import circuit_source_injection_T, circuit_source_projection_proj
from tsim_cutting import compile_detector_sampler_subcomp_enum_general

def add_noise(*, circuit, noise_strength):
    noise_model = gen.NoiseModel.uniform_depolarizing(noise_strength)
    return noise_model.noisy_circuit_skipping_mpp_boundaries(circuit)

def replace_t_with_s(s):
    s = re.sub(r'^(\s*)T_DAG(\s)', r'\1S_DAG\2', s, flags=re.MULTILINE)
    return re.sub(r'^(\s*)T(\s)', r'\1S\2', s, flags=re.MULTILINE)

def replace_s_with_t(c):
    p = str(c)
    p = re.sub(r'^(\s*)S_DAG(\s)', r'\1T_DAG\2', p, flags=re.MULTILINE)
    return re.sub(r'^(\s*)S(\s)', r'\1T\2', p, flags=re.MULTILINE)

clifford_source = replace_t_with_s(circuit_source_injection_T)
clifford_circuit = stim.Circuit(clifford_source)
c_projection = tsim.Circuit(circuit_source_projection_proj)
noiseless_raw = 0

SHOTS = 2**22  # ~4M per noise point
BATCH = 2**18  # 256K

print(f"d=3 LER vs p sweep — {SHOTS} shots per point")
print(f"{'p':>8} {'PSR':>8} {'LER':>12} {'Err':>8} {'Kept':>10} {'Shots/s':>10} {'Compile':>8} {'Graphs':>6}")
print(f"{'-'*8} {'-'*8} {'-'*12} {'-'*8} {'-'*10} {'-'*10} {'-'*8} {'-'*6}")

results = []

for p in [0.001, 0.0015, 0.002, 0.0025, 0.003, 0.004, 0.005]:
    noisy_clif = add_noise(circuit=clifford_circuit, noise_strength=p)
    noisy_str = replace_s_with_t(noisy_clif)
    c_inj = tsim.Circuit(noisy_str)
    circ = c_inj + c_projection

    t0 = time.time()
    sampler = compile_detector_sampler_subcomp_enum_general(circ, seed=42, max_cut_iterations=10)
    t_compile = time.time() - t0

    rep = repr(sampler)
    n_graphs = int(rep.split('(')[1].split(' ')[0])

    # Warmup
    sampler.sample(shots=BATCH, batch_size=BATCH, separate_observables=True)

    t0 = time.time()
    det, obs = sampler.sample(shots=SHOTS, batch_size=BATCH, separate_observables=True)
    t_sample = time.time() - t0

    trivial = np.all(det == 0, axis=1)
    kept = int(np.sum(trivial))
    errs = int(np.sum(obs[trivial, 0].astype(int) != noiseless_raw))
    psr = kept / SHOTS
    ler = errs / kept if kept > 0 else 0
    tput = SHOTS / t_sample

    results.append({'p': p, 'psr': psr, 'ler': ler, 'errs': errs, 'kept': kept,
                    'tput': tput, 'compile': t_compile, 'graphs': n_graphs})

    print(f"{p:>8.4f} {psr:>8.4f} {ler:>12.2e} {errs:>8d} {kept:>10d} {tput:>10.0f} {t_compile:>7.1f}s {n_graphs:>6d}")
    sys.stdout.flush()
    del sampler

# Save results
import json
out_file = os.path.join(_THIS_DIR, "d3_ler_results.json")
with open(out_file, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {out_file}")
