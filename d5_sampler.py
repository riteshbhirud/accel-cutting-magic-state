"""d5_sampler.py — Full d=5 cultivation sampler with closed Pauli web post-selection.

Architecture:
  1. Compile 69-det working circuit (T=30, N=4, K=512)
  2. Use augmented error_transform (121 rows = 101 f-params + 20 proj det parities)
  3. Post-select on 20 proj det = 0 BEFORE expensive ZX evaluation
  4. Evaluate remaining shots via sample_program_subcomp_enum
  5. Return 69 working det + 1 observable + 20 proj det (all 0 for kept shots)
"""

import os
import re
import time
from math import ceil

os.environ.setdefault("JAX_PLATFORMS", "cuda")

import numpy as np
import jax
import jax.numpy as jnp

import tsim
from tsim.core.graph import prepare_graph
from tsim.noise.channels import ChannelSampler

from d5_circuit_utils import build_d5_working_circuit, get_d5_path
from pauli_web_evaluator import load_web_matrix_Wproj

# Imports from the accel compilation pipeline
from tsim_cutting import (
    compile_program_subcomp_enum_general,
    sample_program_subcomp_enum,
)


class D5DetectorSampler:
    """Full d=5 detector sampler with 89-detector post-selection.

    Uses the 69-det working circuit for ZX evaluation (T=30, N=4) and
    computes 20 projection detector parities from the error configuration
    via the augmented error_transform.

    Args:
        noise_strength: Physical noise level p
        seed: Random seed
        max_cut_iterations: Spider cutting iterations (default 10)
    """

    def __init__(
        self,
        noise_strength: float = 0.001,
        *,
        seed: int = 42,
        max_cut_iterations: int = 10,
    ):
        self.noise_strength = noise_strength

        # Load and build circuit
        with open(get_d5_path(noise_strength)) as f:
            content = f.read()
        working_str = build_d5_working_circuit(content)
        circuit = tsim.Circuit(working_str)

        # Prepare ZX graph
        self._key = jax.random.key(seed)
        prepared = prepare_graph(circuit, sample_detectors=True)
        self._num_f = prepared.error_transform.shape[0]  # 101
        self._num_detectors = prepared.num_detectors  # 69
        self._num_outputs = prepared.num_outputs  # 70

        # Compile ZX program
        self._program = compile_program_subcomp_enum_general(
            prepared,
            max_cut_iterations=max_cut_iterations,
        )

        # Build augmented error_transform: [et69; W_proj]
        W_proj = load_web_matrix_Wproj()
        self._num_proj_det = W_proj.shape[0]  # 20
        augmented_et = np.vstack([
            np.array(prepared.error_transform, dtype=np.uint8),
            W_proj.astype(np.uint8),
        ])

        # Create ChannelSampler with augmented error_transform
        self._key, subkey = jax.random.split(self._key)
        channel_seed = int(jax.random.randint(subkey, (), 0, 2**30))
        self._channel_sampler = ChannelSampler(
            channel_probs=prepared.channel_probs,
            error_transform=augmented_et,
            seed=channel_seed,
        )

    def sample(
        self,
        shots: int,
        *,
        batch_size: int | None = None,
    ) -> dict:
        """Sample detector and observable outcomes with full 89-det post-selection.

        For each shot:
        1. Draw augmented f-params (121 bits)
        2. Post-select on 20 proj det parities = 0
        3. Evaluate 69-det ZX graph for surviving shots
        4. Return combined results

        Args:
            shots: Number of post-selection-surviving shots to produce
            batch_size: Batch size for evaluation (default: min(shots, 1024))

        Returns:
            dict with:
                'det_working': (shots, 69) working detector parities (all 0 if post-selected)
                'det_proj': (shots, 20) projection detector parities (all 0)
                'observable': (shots,) observable bit
                'det_all': (shots, 89) all detector parities [working + proj]
                'total_sampled': total shots drawn before post-selection
                'proj_rejected': shots rejected by projection post-selection
                'survival_rate': fraction of shots surviving projection post-selection
        """
        if batch_size is None:
            batch_size = min(shots, 1024)

        num_f = self._num_f
        num_proj = self._num_proj_det

        # Collect surviving results
        det_working_list = []
        obs_list = []
        total_sampled = 0
        proj_rejected = 0

        while len(det_working_list) < shots:
            # How many more do we need?
            needed = shots - len(det_working_list)
            # Over-sample to account for post-selection rejection
            # Estimate survival rate, with minimum 10% floor
            if total_sampled > 0:
                est_survival = max(0.1, 1.0 - proj_rejected / total_sampled)
            else:
                est_survival = 0.5  # Conservative initial estimate
            this_batch = min(batch_size, max(int(needed / est_survival * 1.2), needed))

            # Step 1: Sample augmented f-params
            f_augmented = self._channel_sampler.sample(this_batch)
            total_sampled += this_batch

            # Step 2: Split into f-params and proj det parities
            f_params = f_augmented[:, :num_f]
            proj_det = f_augmented[:, num_f:]

            # Step 3: Post-select on proj det = 0
            proj_det_np = np.array(proj_det, dtype=np.uint8)
            keep_mask = np.all(proj_det_np == 0, axis=1)
            n_rejected = this_batch - keep_mask.sum()
            proj_rejected += n_rejected

            if keep_mask.sum() == 0:
                continue  # All rejected, try again

            # Filter to surviving shots
            f_kept = f_params[keep_mask]

            # Step 4: Evaluate ZX graph for surviving shots
            self._key, subkey = jax.random.split(self._key)
            samples = sample_program_subcomp_enum(
                self._program, f_kept, subkey
            )
            samples_np = np.array(samples, dtype=np.uint8)

            # Split into detectors and observable
            det_w = samples_np[:, :self._num_detectors]  # (n_kept, 69)
            obs_bits = samples_np[:, self._num_detectors:]  # (n_kept, 1)

            det_working_list.append(det_w)
            obs_list.append(obs_bits[:, 0])

        # Concatenate and trim to requested shots
        det_working = np.concatenate(det_working_list)[:shots]
        observable = np.concatenate(obs_list)[:shots]

        # Projection detectors are all 0 (by construction)
        det_proj = np.zeros((shots, num_proj), dtype=np.uint8)
        det_all = np.hstack([det_working, det_proj])

        survival_rate = 1.0 - proj_rejected / total_sampled if total_sampled > 0 else 0

        return {
            'det_working': det_working,
            'det_proj': det_proj,
            'observable': observable,
            'det_all': det_all,
            'total_sampled': total_sampled,
            'proj_rejected': proj_rejected,
            'survival_rate': survival_rate,
        }

    def sample_simple(
        self,
        shots: int,
        *,
        batch_size: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simple interface: returns (det_89, obs) matching stim's format.

        Post-selects on ALL 89 detectors = 0.

        Returns:
            det_all: (N_surviving, 89) detector parities (all 0 by construction)
            obs: (N_surviving,) observable bits
        """
        result = self.sample(shots, batch_size=batch_size)
        # Further post-select on working detectors = 0
        all_det = result['det_all']
        obs = result['observable']
        working_trivial = np.all(result['det_working'] == 0, axis=1)

        return all_det[working_trivial], obs[working_trivial]

    def estimate_logical_error_rate(
        self,
        shots: int,
        *,
        batch_size: int | None = None,
    ) -> dict:
        """Estimate the logical error rate at this noise strength.

        Post-selects on all 89 detectors = 0 and computes P(obs=1).

        Returns:
            dict with logical_error_rate, num_surviving, total_sampled, etc.
        """
        result = self.sample(shots, batch_size=batch_size)
        det_working = result['det_working']
        obs = result['observable']

        # Full 89-det post-selection: working det = 0 AND proj det = 0
        # (proj det already = 0 by construction)
        working_trivial = np.all(det_working == 0, axis=1)
        num_surviving = working_trivial.sum()
        obs_surviving = obs[working_trivial]

        if num_surviving > 0:
            logical_error_rate = obs_surviving.mean()
        else:
            logical_error_rate = float('nan')

        return {
            'logical_error_rate': logical_error_rate,
            'num_surviving': int(num_surviving),
            'total_sampled': result['total_sampled'],
            'proj_survival_rate': result['survival_rate'],
            'working_survival_rate': float(working_trivial.mean()),
            'overall_survival_rate': float(num_surviving / result['total_sampled']),
            'noise_strength': self.noise_strength,
        }


if __name__ == '__main__':
    print("=" * 70)
    print("D=5 Cultivation Sampler Test")
    print("=" * 70)

    print("\nCompiling (p=0.001)...")
    t0 = time.time()
    sampler = D5DetectorSampler(noise_strength=0.001, seed=42)
    print(f"Compilation: {time.time()-t0:.1f}s")

    # Quick test — small batch to avoid slow JIT on large arrays
    print("\nSampling 50 proj-surviving shots (batch=128)...")
    t0 = time.time()
    result = sampler.sample(50, batch_size=128)
    t_sample = time.time() - t0
    print(f"Time: {t_sample:.1f}s (includes JIT warmup)")
    print(f"Total sampled: {result['total_sampled']}")
    print(f"Proj survival: {result['survival_rate']:.4f}")
    working_trivial = np.all(result['det_working'] == 0, axis=1)
    print(f"Working trivial: {working_trivial.mean():.4f}")
    print(f"Observable rate: {result['observable'].mean():.4f}")
    print(f"Full 89-det surviving: {working_trivial.sum()}")

    # Post-JIT: should be faster
    print("\nSampling 200 more proj-surviving shots (post-JIT)...")
    t0 = time.time()
    result2 = sampler.sample(200, batch_size=128)
    t_sample2 = time.time() - t0
    print(f"Time: {t_sample2:.1f}s")
    print(f"Total sampled: {result2['total_sampled']}")
    print(f"Proj survival: {result2['survival_rate']:.4f}")
    working_trivial2 = np.all(result2['det_working'] == 0, axis=1)
    n_surviving2 = working_trivial2.sum()
    obs_surviving2 = result2['observable'][working_trivial2]
    print(f"Full 89-det surviving: {n_surviving2}")
    if n_surviving2 > 0:
        print(f"Logical error rate (89-det): {obs_surviving2.mean():.6f}")

    # Logical error rate estimate
    print("\nEstimating logical error rate (500 proj-surviving shots)...")
    t0 = time.time()
    ler = sampler.estimate_logical_error_rate(500, batch_size=128)
    t_ler = time.time() - t0
    print(f"Time: {t_ler:.1f}s")
    print(f"Logical error rate: {ler['logical_error_rate']:.6f}")
    print(f"Surviving shots (89-det): {ler['num_surviving']}")
    print(f"Total sampled: {ler['total_sampled']}")
    print(f"Proj survival: {ler['proj_survival_rate']:.4f}")
    print(f"Working survival: {ler['working_survival_rate']:.4f}")
    print(f"Overall survival: {ler['overall_survival_rate']:.4f}")
