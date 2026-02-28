"""Task 4: Validate d=3 cutting pipeline end-to-end.

Uses the exact same pipeline as run.py to:
1. Build d=3 circuit (injection + projection)
2. Compile via cutting decomposition
3. Sample and compute PSR + fidelity
4. Compare noiseless and noisy results
"""
import sys, time, re
from pathlib import Path
from copy import deepcopy

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))

import os
os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
import tsim
import stim
import gen

from d_3_circuit_definitions import circuit_source_injection_T, circuit_source_projection_proj
from tsim.core.graph import prepare_graph, connected_components, get_params
from stab_rank_cut import tcount
import pyzx_param as zx

# ============================================================================
# Helper functions (same as run.py)
# ============================================================================

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

# ============================================================================
# Step 1: Build d=3 circuit
# ============================================================================
print("=" * 60)
print("Step 1: Building d=3 circuit")
print("=" * 60)

clifford_source = replace_t_with_s(circuit_source_injection_T)
clifford_circuit = stim.Circuit(clifford_source)
c_projection = tsim.Circuit(circuit_source_projection_proj)

print(f"  Injection circuit: {clifford_circuit.num_qubits} qubits, "
      f"{clifford_circuit.num_measurements} measurements")

# ============================================================================
# Step 2: Noiseless graph analysis
# ============================================================================
print(f"\n{'=' * 60}")
print("Step 2: Noiseless graph analysis")
print(f"{'=' * 60}")

noisy_injection_str = replace_s_with_t(clifford_circuit)
c_injection = tsim.Circuit(noisy_injection_str)
circ_noiseless = c_injection + c_projection

prepared = prepare_graph(circ_noiseless, sample_detectors=True)
components = connected_components(prepared.graph)
components_sorted = sorted(components, key=lambda c: len(c.output_indices))

n_trivial = sum(1 for cc in components_sorted if len(cc.graph.outputs()) <= 1)
n_nontrivial = len(components_sorted) - n_trivial

print(f"  Components: {len(components_sorted)} ({n_trivial} trivial, {n_nontrivial} non-trivial)")
print(f"  Outputs: {prepared.num_outputs}, Detectors: {prepared.num_detectors}")

for i, cc in enumerate(components_sorted):
    if len(cc.graph.outputs()) > 1 or tcount(cc.graph) > 0:
        g2 = deepcopy(cc.graph)
        zx.full_reduce(g2, paramSafe=True)
        tc = tcount(g2)
        print(f"  Component {i}: {len(list(cc.graph.vertices()))} verts, "
              f"{len(cc.graph.outputs())} outputs, T-count={tc}")

# ============================================================================
# Step 3: Compile and sample noiseless
# ============================================================================
print(f"\n{'=' * 60}")
print("Step 3: Noiseless compilation + sampling")
print(f"{'=' * 60}")

from tsim_cutting import compile_detector_sampler_subcomp_enum_general

t0 = time.time()
sampler = compile_detector_sampler_subcomp_enum_general(
    circ_noiseless, seed=42, max_cut_iterations=10
)
t_compile = time.time() - t0
print(f"  Compile: {t_compile:.2f}s")
print(f"  {sampler}")

SHOTS = 2**20  # ~1M
BATCH = 2**18

# Warmup
print(f"  Warmup...")
sampler.sample(shots=BATCH, batch_size=BATCH, separate_observables=True)

t0 = time.time()
det, obs = sampler.sample(shots=SHOTS, batch_size=BATCH, separate_observables=True)
t_sample = time.time() - t0

noiseless_raw = 0  # From run.py
trivial = np.all(det == 0, axis=1)
n_kept = int(np.sum(trivial))
n_errors = int(np.sum(obs[trivial, 0].astype(int) != noiseless_raw))
psr = n_kept / SHOTS
fid = 1.0 - n_errors / n_kept if n_kept > 0 else 0

print(f"\n  Noiseless results ({SHOTS} shots, {SHOTS/t_sample:.0f} shots/s):")
print(f"    PSR: {psr:.6f} (expect 1.0)")
print(f"    Fidelity: {fid:.8f} (expect 1.0)")
print(f"    Errors: {n_errors}/{n_kept}")

del sampler

# ============================================================================
# Step 4: Noisy sweep
# ============================================================================
print(f"\n{'=' * 60}")
print("Step 4: LER vs p sweep (d=3)")
print(f"{'=' * 60}")

NOISE_STRENGTHS = [0.001, 0.002, 0.003, 0.004, 0.005]
SHOTS_SWEEP = 2**20

results = []

for p in NOISE_STRENGTHS:
    noisy_clif = add_noise(circuit=clifford_circuit, noise_strength=p)
    noisy_str = replace_s_with_t(noisy_clif)
    c_inj = tsim.Circuit(noisy_str)
    circ_p = c_inj + c_projection

    t0 = time.time()
    sampler_p = compile_detector_sampler_subcomp_enum_general(
        circ_p, seed=42, max_cut_iterations=10
    )
    t_compile_p = time.time() - t0
    print(f"\n  p={p}: compiled in {t_compile_p:.1f}s — {sampler_p}")

    sampler_p.sample(shots=BATCH, batch_size=BATCH, separate_observables=True)

    t0 = time.time()
    det_p, obs_p = sampler_p.sample(
        shots=SHOTS_SWEEP, batch_size=BATCH, separate_observables=True
    )
    t_p = time.time() - t0

    trivial_p = np.all(det_p == 0, axis=1)
    kept = int(np.sum(trivial_p))
    errs = int(np.sum(obs_p[trivial_p, 0].astype(int) != noiseless_raw))
    psr_p = kept / SHOTS_SWEEP
    fid_p = 1.0 - errs / kept if kept > 0 else 0
    ler_p = 1 - fid_p
    tput = SHOTS_SWEEP / t_p

    results.append({
        'p': p, 'psr': psr_p, 'kept': kept, 'errors': errs,
        'fidelity': fid_p, 'ler': ler_p, 'throughput': tput,
    })

    print(f"    PSR={psr_p:.4f}, LER={ler_p:.2e}, errors={errs}/{kept}, "
          f"{tput:.0f} shots/s")
    del sampler_p

# ============================================================================
# Summary
# ============================================================================
print(f"\n{'=' * 60}")
print("SUMMARY: d=3 cultivation circuit")
print(f"{'=' * 60}")
print(f"\n  {'p':>8} {'PSR':>8} {'LER':>12} {'Errors':>8} {'Kept':>8} {'Shots/s':>10}")
print(f"  {'-'*8} {'-'*8} {'-'*12} {'-'*8} {'-'*8} {'-'*10}")

for r in results:
    print(f"  {r['p']:>8.4f} {r['psr']:>8.4f} {r['ler']:>12.2e} "
          f"{r['errors']:>8d} {r['kept']:>8d} {r['throughput']:>10.0f}")

print(f"\n  Shots per point: {SHOTS_SWEEP}")
