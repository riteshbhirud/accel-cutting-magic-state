"""Dedup + sparse geometric channel sampler benchmark.
Loops over multiple noise strengths."""
import sys, time, json
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))

import tsim, stim, numpy as np, re
import gen


def add_noise(*, circuit, noise_strength, gateset="css"):
    if noise_strength <= 0:
        return circuit
    noise_model = gen.NoiseModel.uniform_depolarizing(noise_strength)
    return noise_model.noisy_circuit_skipping_mpp_boundaries(circuit)


def replace_t_with_s(s):
    s = re.sub(r'^(\s*)T_DAG(\s)', r'\1S_DAG\2', s, flags=re.MULTILINE)
    return re.sub(r'^(\s*)T(\s)', r'\1S\2', s, flags=re.MULTILINE)


def replace_s_with_t(c):
    p = str(c)
    p = re.sub(r'^(\s*)S_DAG(\s)', r'\1T_DAG\2', p, flags=re.MULTILINE)
    return re.sub(r'^(\s*)S(\s)', r'\1T\2', p, flags=re.MULTILINE)


from d_3_circuit_definitions import circuit_source_injection_T, circuit_source_projection_proj

_projection_T_Proj_decode = "\nTICK\nCX 8 3\nCX 11 6\nCX 0 9\nTICK\nCX 8 14\nCX 11 9\nCX 0 3\nTICK\nCX 11 14\nCX 8 9\nCX 0 6\nTICK\nCX 6 14\nTICK\nCX 6 3\nTICK\n"
_projection_T_Proj_measure = "\nTICK\nMX 0 11 8\nM 9 3 14\nMX 6\nDETECTOR(0.625, 0.125, 0, -1, -9) rec[-20] rec[-19] rec[-14] rec[-7]\nDETECTOR(0.875, 0.125, 0, -1, -9) rec[-17] rec[-4]\nDETECTOR(1.25, 1.4375, 0, -1, -9) rec[-20] rec[-14] rec[-6] rec[-5]\nDETECTOR(1.5, 1.4375, 0, -1, -9) rec[-16] rec[-3]\nDETECTOR(2.5, 0.9375, 0, -1, -9) rec[-14] rec[-6]\nDETECTOR(2.75, 0.9375, 0, -1, -9) rec[-15] rec[-2]\n"


def compute_projection_obs_include(injection_clifford_str):
    """Compute projection circuit with OBSERVABLE_INCLUDE via stim flow generators."""
    for gate, gate_clifford in [("T_DAG 6", "S_DAG 6"), ("T 6", "S 6")]:
        proj_clifford = _projection_T_Proj_decode + gate_clifford + "\n" + _projection_T_Proj_measure
        full_clifford = injection_clifford_str + proj_clifford + "OBSERVABLE_INCLUDE(0) rec[-1]\n"
        circ = stim.Circuit(full_clifford)
        total_meas = circ.num_measurements
        last_rec = total_meas - 1
        for flow in circ.flow_generators():
            flow_str = str(flow)
            if f'rec[{last_rec}]' not in flow_str:
                continue
            output_part = flow_str.split(' -> ')[1] if ' -> ' in flow_str else ''
            if any(c in output_part.replace('xor', '') for c in 'XYZ'):
                continue
            recs = sorted([int(x) for x in re.findall(r'rec\[(\d+)\]', flow_str)])
            neg_recs = [f'rec[{r - total_meas}]' for r in recs]
            obs_line = f"OBSERVABLE_INCLUDE(0) {' '.join(neg_recs)}"
            stim_circ = stim.Circuit(injection_clifford_str + proj_clifford + obs_line + "\n")
            raw = stim_circ.compile_sampler().sample(shots=1)
            noiseless_raw = 0
            for r in recs:
                noiseless_raw ^= int(raw[0, r])
            if noiseless_raw == 0:
                return _projection_T_Proj_decode + gate + "\n" + _projection_T_Proj_measure + obs_line + "\n", 0
            break
    raise ValueError


# ============================================================================
NOISE_STRENGTHS = [0.001, 0.0015, 0.002, 0.0025, 0.003, 0.004, 0.005]
SHOTS = 2**30          # ~1B total shots per noise strength
BATCH_SIZE = 2**21     # 2M shots per batch (fits in L3 cache / JAX JIT sweet spot)
SEED = 2               # fixed seed for reproducibility
# ============================================================================

# Replace tsim's default evaluator with optimised complex64 BLAS version
from evaluate_matmul_cfloat import evaluate_batch as evaluate_batch_cfloat
import tsim_cutting as mod_cutting
mod_cutting.evaluate_batch = evaluate_batch_cfloat

from tsim_cutting import compile_detector_sampler_subcomp_enum_general
from sampler_noiseless_cache import add_noiseless_cache
from sampler_dedup import patch_sampler_fast
from channel_sampler_fast import patch_channel_sampler_fast


def main():
    # Noise-independent setup (done once)
    clifford_source = replace_t_with_s(circuit_source_injection_T)
    c_projection = tsim.Circuit(circuit_source_projection_proj)
    noiseless_raw = 0  # T gate on qubit 6: noiseless observable is 0
    clifford_circuit = stim.Circuit(clifford_source)

    all_results = []

    for noise_strength in NOISE_STRENGTHS:
        print(f"\n{'#'*60}")
        print(f"# p = {noise_strength}")
        print(f"{'#'*60}")

        # Build noisy circuit
        noisy_clifford = add_noise(circuit=clifford_circuit, noise_strength=noise_strength)
        noisy_injection_str = replace_s_with_t(noisy_clifford)
        c_injection = tsim.Circuit(noisy_injection_str)
        circ = c_injection + c_projection

        # Setup sampler
        sampler = compile_detector_sampler_subcomp_enum_general(circ, seed=SEED, max_cut_iterations=10)
        add_noiseless_cache(sampler)
        patch_sampler_fast(sampler, top_k=None, max_unique=None, use_dedup=True, verbose=False)
        patch_channel_sampler_fast(sampler, verbose=False)

        # Warmup
        print(f"  Warming up...")
        t_warmup = time.time()
        sampler.sample(shots=BATCH_SIZE, batch_size=BATCH_SIZE, separate_observables=True)
        t_warmup = time.time() - t_warmup
        print(f"  Warmup: {t_warmup:.2f}s")

        # Benchmark
        print(f"  Running {SHOTS} shots...")
        t0 = time.time()
        det, obs = sampler.sample(shots=SHOTS, batch_size=BATCH_SIZE, separate_observables=True)
        t_sample = time.time() - t0

        trivial = np.all(det == 0, axis=1)
        n_kept = int(np.sum(trivial))
        n_errors = int(np.sum(obs[trivial, 0].astype(int) != noiseless_raw))
        psr = n_kept / SHOTS
        fidelity = 1.0 - n_errors / n_kept if n_kept > 0 else 0
        throughput = SHOTS / t_sample

        print(f"  {t_sample:.3f}s  {throughput:.0f} shots/s  PSR={psr:.6f}  "
              f"errors={n_errors}/{n_kept}  fidelity={fidelity:.8f}")

        all_results.append({
            "noise_strength": noise_strength, "shots": SHOTS, "batch_size": BATCH_SIZE,
            "seed": SEED, "warmup_s": round(t_warmup, 3), "time_s": round(t_sample, 3),
            "throughput": round(throughput, 1), "psr": round(psr, 6),
            "n_kept": n_kept, "n_errors": n_errors, "fidelity": round(fidelity, 8),
        })
        del sampler

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY (dedup + sparse geometric channel)")
    print(f"{'='*70}")
    print(f"  batch_size={BATCH_SIZE}, shots={SHOTS}")
    print(f"\n{'p':<8} {'Shots/s':>10} {'Time':>8} {'PSR':>8} {'Errors':>7} {'Fidelity':>12}")
    print(f"{'-'*8} {'-'*10} {'-'*8} {'-'*8} {'-'*7} {'-'*12}")
    for r in all_results:
        print(f"{r['noise_strength']:<8} {r['throughput']:>10.0f} {r['time_s']:>7.3f}s "
              f"{r['psr']:>8.4f} {r['n_errors']:>7d} {r['fidelity']:>12.8f}")

    out_file = str(_THIS_DIR / "results.jsonl")
    with open(out_file, "a") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")
    print(f"\nResults saved to: {out_file}")


if __name__ == "__main__":
    main()
