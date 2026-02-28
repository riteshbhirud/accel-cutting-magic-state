"""d=3 LER vs p sweep — small batches to avoid OOM."""
import sys, time, re, os
os.environ["JAX_PLATFORMS"] = "cpu"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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

TOTAL_SHOTS = 100_000
BATCH = 2048  # Small batches to avoid OOM
N_BATCHES = TOTAL_SHOTS // BATCH

results = []

for p in [0.001, 0.0015, 0.002, 0.0025, 0.003, 0.004, 0.005]:
    noisy_clif = add_noise(circuit=clifford_circuit, noise_strength=p)
    noisy_str = replace_s_with_t(noisy_clif)
    c_inj = tsim.Circuit(noisy_str)
    circ = c_inj + c_projection

    t0 = time.time()
    sampler = compile_detector_sampler_subcomp_enum_general(circ, seed=42, max_cut_iterations=10)
    t_compile = time.time() - t0

    # Warmup
    sampler.sample(shots=256, batch_size=256, separate_observables=True)

    # Sample in small batches
    t0 = time.time()
    total_kept = 0
    total_errs = 0
    total_shots = 0
    for _ in range(N_BATCHES):
        det, obs = sampler.sample(shots=BATCH, batch_size=BATCH, separate_observables=True)
        trivial = np.all(det == 0, axis=1)
        total_kept += int(np.sum(trivial))
        total_errs += int(np.sum(obs[trivial, 0].astype(int) != 0))
        total_shots += BATCH
    t_sample = time.time() - t0

    psr = total_kept / total_shots
    ler = total_errs / total_kept if total_kept > 0 else 0
    tput = total_shots / t_sample

    results.append({'p': p, 'psr': psr, 'ler': ler, 'errs': total_errs,
                    'kept': total_kept, 'shots': total_shots, 'tput': tput})

    print(f"p={p:.4f}: PSR={psr:.4f}, LER={ler:.2e}, errs={total_errs}/{total_kept}, "
          f"{tput:.0f} shots/s, compile={t_compile:.1f}s")
    sys.stdout.flush()
    del sampler

# Summary
print(f"\n{'='*60}")
print(f"d=3 LER vs p ({TOTAL_SHOTS} shots per point)")
print(f"{'='*60}")
for r in results:
    print(f"  p={r['p']:.4f}: PSR={r['psr']:.4f}, LER={r['ler']:.2e}, "
          f"errs={r['errs']}/{r['kept']}")

import json
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "d3_ler_results.json"), "w") as f:
    json.dump(results, f, indent=2)
print("Saved to d3_ler_results.json")
