"""Quick d=3 validation: compilation + small sample."""
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
from stab_rank_cut import tcount
import pyzx_param as zx

def add_noise(*, circuit, noise_strength):
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

# Build circuit
print("Building d=3 circuit...")
clifford_source = replace_t_with_s(circuit_source_injection_T)
clifford_circuit = stim.Circuit(clifford_source)
c_projection = tsim.Circuit(circuit_source_projection_proj)

noise_strength = 0.001
noisy_clif = add_noise(circuit=clifford_circuit, noise_strength=noise_strength)
noisy_str = replace_s_with_t(noisy_clif)
c_inj = tsim.Circuit(noisy_str)
circ = c_inj + c_projection

print(f"  Circuit built")

# Compile
print("\nCompiling...")
from tsim_cutting import compile_detector_sampler_subcomp_enum_general

t0 = time.time()
sampler = compile_detector_sampler_subcomp_enum_general(
    circ, seed=42, max_cut_iterations=10
)
t_compile = time.time() - t0
print(f"  Compiled in {t_compile:.1f}s")
print(f"  {sampler}")

# Small sample to test
print("\nSampling 1024 shots...")
t0 = time.time()
det, obs = sampler.sample(shots=1024, batch_size=1024, separate_observables=True)
t1 = time.time() - t0
print(f"  Done in {t1:.2f}s")
print(f"  det shape: {det.shape}, obs shape: {obs.shape}")

noiseless_raw = 0
trivial = np.all(det == 0, axis=1)
kept = int(np.sum(trivial))
errs = int(np.sum(obs[trivial, 0].astype(int) != noiseless_raw))
print(f"  Kept: {kept}/1024 (PSR={kept/1024:.4f})")
print(f"  Errors: {errs}/{kept}")
if kept > 0:
    print(f"  Fidelity: {1 - errs/kept:.6f}")

# Bigger sample
print("\nSampling 100K shots...")
t0 = time.time()
det2, obs2 = sampler.sample(shots=100000, batch_size=100000, separate_observables=True)
t2 = time.time() - t0
print(f"  Done in {t2:.2f}s ({100000/t2:.0f} shots/s)")

trivial2 = np.all(det2 == 0, axis=1)
kept2 = int(np.sum(trivial2))
errs2 = int(np.sum(obs2[trivial2, 0].astype(int) != noiseless_raw))
psr2 = kept2 / 100000
print(f"  Kept: {kept2}/100000 (PSR={psr2:.4f})")
print(f"  Errors: {errs2}/{kept2}")
if kept2 > 0:
    print(f"  Fidelity: {1 - errs2/kept2:.8f}")
    print(f"  LER: {errs2/kept2:.2e}")

print("\nDone!")
