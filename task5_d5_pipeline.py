"""Task 5: d=5 cutting pipeline — compile + sample.

Loads Gidney's d=5 cultivation circuit, strips noise,
adds noise via gen.NoiseModel, replaces S→T, and runs
through the cutting compilation pipeline.

Steps:
1. Load d=5 circuit, strip noise → noiseless Clifford
2. Add noise programmatically
3. Replace S→T for non-Clifford simulation
4. Compile and analyze
5. Sample for LER
"""
import sys, time, re, os
os.environ["JAX_PLATFORMS"] = "cpu"

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

import numpy as np
import tsim
import stim
import gen
from tsim.core.graph import prepare_graph, connected_components, get_params
from stab_rank_cut import tcount
from copy import deepcopy
import pyzx_param as zx

# ============================================================================
# Helper functions
# ============================================================================

def strip_noise(circuit_str):
    """Remove noise from a stim circuit string."""
    lines = circuit_str.split('\n')
    clean = []
    for line in lines:
        s = line.strip()
        if any(s.startswith(p) for p in ['X_ERROR', 'Z_ERROR', 'DEPOLARIZE1', 'DEPOLARIZE2']):
            continue
        line = re.sub(r'M\([\d.]+\)', 'M', line)
        line = re.sub(r'MX\([\d.]+\)', 'MX', line)
        clean.append(line)
    return '\n'.join(clean)


def add_noise(*, circuit, noise_strength):
    if noise_strength <= 0:
        return circuit
    noise_model = gen.NoiseModel.uniform_depolarizing(noise_strength)
    return noise_model.noisy_circuit_skipping_mpp_boundaries(circuit)


def replace_s_with_t(c):
    p = str(c)
    p = re.sub(r'^(\s*)S_DAG(\s)', r'\1T_DAG\2', p, flags=re.MULTILINE)
    return re.sub(r'^(\s*)S(\s)', r'\1T\2', p, flags=re.MULTILINE)


# ============================================================================
# Step 1: Load d=5 circuit
# ============================================================================
print("=" * 60)
print("Step 1: Loading d=5 circuit")
print("=" * 60)

CIRCUIT_PATH = "/Users/ritesh/Downloads/prx/gidney-circuits/circuits/" \
    "for_perfectionist_decoding/" \
    "c=inject[unitary]+cultivate,p=0.001,noise=uniform," \
    "g=css,q=42,b=Y,r=10,d1=5.stim"

raw = open(CIRCUIT_PATH).read()
noiseless_str = strip_noise(raw)
clifford_circuit = stim.Circuit(noiseless_str)

print(f"  Qubits: {clifford_circuit.num_qubits}")
print(f"  Measurements: {clifford_circuit.num_measurements}")
print(f"  Detectors: {clifford_circuit.num_detectors}")
print(f"  Observables: {clifford_circuit.num_observables}")

# Count S/S_DAG (= T/T_DAG after replacement)
n_s = len(re.findall(r'^\s*S\s', noiseless_str, re.MULTILINE))
n_sdag = len(re.findall(r'^\s*S_DAG\s', noiseless_str, re.MULTILINE))
print(f"  S gates: {n_s}, S_DAG gates: {n_sdag}")

# ============================================================================
# Step 2: Noiseless compile test
# ============================================================================
print(f"\n{'=' * 60}")
print("Step 2: Noiseless compilation")
print(f"{'=' * 60}")

t_str = replace_s_with_t(clifford_circuit)
circ = tsim.Circuit(t_str)

t0 = time.time()
prepared = prepare_graph(circ, sample_detectors=True)
t_prep = time.time() - t0
print(f"  prepare_graph: {t_prep:.2f}s")
print(f"  Outputs: {prepared.num_outputs}, Detectors: {prepared.num_detectors}")

components = connected_components(prepared.graph)
components_sorted = sorted(components, key=lambda c: len(c.output_indices))
n_trivial = sum(1 for cc in components_sorted if len(cc.graph.outputs()) <= 1)

print(f"  Components: {len(components_sorted)} ({n_trivial} trivial)")

# Show non-trivial components
for i, cc in enumerate(components_sorted):
    n_out = len(cc.graph.outputs())
    if n_out > 1 or tcount(cc.graph) > 0:
        g2 = deepcopy(cc.graph)
        zx.full_reduce(g2, paramSafe=True)
        tc = tcount(g2)
        print(f"  Component {i}: {len(list(cc.graph.vertices()))} verts, "
              f"{n_out} outputs, T-count={tc}")

sys.stdout.flush()

# Try compilation
print(f"\n  Compiling...")
from tsim_cutting import compile_detector_sampler_subcomp_enum_general

t0 = time.time()
sampler = compile_detector_sampler_subcomp_enum_general(
    circ, seed=42, max_cut_iterations=10
)
t_compile = time.time() - t0
print(f"  Compiled in {t_compile:.1f}s")
print(f"  {sampler}")
sys.stdout.flush()

# Quick noiseless sample
print(f"\n  Sampling 10K noiseless...")
sampler.sample(shots=1024, batch_size=1024, separate_observables=True)  # warmup
det, obs = sampler.sample(shots=10000, batch_size=10000, separate_observables=True)

trivial = np.all(det == 0, axis=1)
kept = int(np.sum(trivial))
errs = int(np.sum(obs[trivial, 0].astype(int) != 0))
print(f"  PSR: {kept/10000:.4f}, Errors: {errs}/{kept}")
sys.stdout.flush()
del sampler

# ============================================================================
# Step 3: Noisy compile test (p=0.001)
# ============================================================================
print(f"\n{'=' * 60}")
print("Step 3: Noisy compilation (p=0.001)")
print(f"{'=' * 60}")

p = 0.001
noisy_clif = add_noise(circuit=clifford_circuit, noise_strength=p)
noisy_str = replace_s_with_t(noisy_clif)
circ_noisy = tsim.Circuit(noisy_str)

t0 = time.time()
sampler_noisy = compile_detector_sampler_subcomp_enum_general(
    circ_noisy, seed=42, max_cut_iterations=10
)
t_compile = time.time() - t0
print(f"  Compiled in {t_compile:.1f}s")
print(f"  {sampler_noisy}")
sys.stdout.flush()

# Sample
print(f"\n  Sampling 100K at p={p}...")
sampler_noisy.sample(shots=1024, batch_size=1024, separate_observables=True)

t0 = time.time()
det_n, obs_n = sampler_noisy.sample(shots=100000, batch_size=100000, separate_observables=True)
t_sample = time.time() - t0

trivial_n = np.all(det_n == 0, axis=1)
kept_n = int(np.sum(trivial_n))
errs_n = int(np.sum(obs_n[trivial_n, 0].astype(int) != 0))
psr_n = kept_n / 100000
ler_n = errs_n / kept_n if kept_n > 0 else 0

print(f"  PSR: {psr_n:.4f}, LER: {ler_n:.2e}, errors: {errs_n}/{kept_n}")
print(f"  Time: {t_sample:.2f}s ({100000/t_sample:.0f} shots/s)")
sys.stdout.flush()

del sampler_noisy

# ============================================================================
# Step 4: LER vs p sweep
# ============================================================================
print(f"\n{'=' * 60}")
print("Step 4: d=5 LER vs p sweep")
print(f"{'=' * 60}")

SHOTS = 2**20  # ~1M
BATCH = 2**18

print(f"Shots per point: {SHOTS}")
print(f"{'p':>8} {'PSR':>8} {'LER':>12} {'Err':>8} {'Kept':>10} {'Shots/s':>10} {'Compile':>8}")
print(f"{'-'*8} {'-'*8} {'-'*12} {'-'*8} {'-'*10} {'-'*10} {'-'*8}")

results = []

for p in [0.001, 0.0015, 0.002, 0.0025, 0.003, 0.004, 0.005]:
    noisy_clif = add_noise(circuit=clifford_circuit, noise_strength=p)
    noisy_str = replace_s_with_t(noisy_clif)
    c = tsim.Circuit(noisy_str)

    t0 = time.time()
    s = compile_detector_sampler_subcomp_enum_general(c, seed=42, max_cut_iterations=10)
    t_compile = time.time() - t0

    s.sample(shots=BATCH, batch_size=BATCH, separate_observables=True)  # warmup

    t0 = time.time()
    det, obs = s.sample(shots=SHOTS, batch_size=BATCH, separate_observables=True)
    t_sample = time.time() - t0

    trivial = np.all(det == 0, axis=1)
    kept = int(np.sum(trivial))
    errs = int(np.sum(obs[trivial, 0].astype(int) != 0))
    psr = kept / SHOTS
    ler = errs / kept if kept > 0 else 0
    tput = SHOTS / t_sample

    results.append({'p': p, 'psr': psr, 'ler': ler, 'errs': errs, 'kept': kept,
                    'tput': tput, 'compile': t_compile})

    print(f"{p:>8.4f} {psr:>8.4f} {ler:>12.2e} {errs:>8d} {kept:>10d} {tput:>10.0f} {t_compile:>7.1f}s")
    sys.stdout.flush()
    del s

import json
with open(os.path.join(_THIS_DIR, "d5_ler_results.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to d5_ler_results.json")
