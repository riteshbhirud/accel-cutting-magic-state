"""Fast channel sampler — sparse geometric skip for low-noise regimes.

At low noise (p=0.0005), P(non-identity) ~ 0.001 per channel per shot.
Instead of generating 124 x 2M = 248M random numbers (dense per-channel sampling),
use geometric distribution to skip directly to fire events.

At p=0.0005: ~248K fire events instead of 248M draws (~1000x reduction).
Uses numpy for all operations, avoiding 124 JAX dispatch roundtrips.

Usage:
    from channel_sampler_fast import patch_channel_sampler_fast
    patch_channel_sampler_fast(sampler)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def _make_sparse_sample_fn(channels, matrix_jax, seed):
    """Create numpy-based sparse channel sampling using geometric skip.

    For each channel, precomputes:
    - p_fire: probability of any non-identity outcome
    - cond_cdf: conditional CDF over non-identity outcomes
    - xor_patterns: precomputed XOR effect for each non-identity outcome

    At runtime, uses geometric distribution to skip directly to fire events,
    then applies the appropriate XOR pattern. Positions are sorted from
    cumsum, giving sequential (cache-friendly) memory access.
    """
    n_channels = len(channels)
    num_outputs = matrix_jax.shape[1]
    matrix_np = np.asarray(matrix_jax).astype(np.uint8)

    # Precompute per-channel sparse sampling data
    channel_data = []
    for ch in channels:
        probs = np.asarray(jax.nn.softmax(ch.logits), dtype=np.float64)
        p_identity = float(probs[0])
        p_fire = 1.0 - p_identity

        col_ids = np.asarray(ch.unique_col_ids)
        num_bits = len(col_ids)
        n_outcomes = len(probs)

        # At p_fire=1e-15 with batch_size=2M, expected fires < 2e-9; skip entirely
        if p_fire <= 1e-15 or n_outcomes <= 1:
            channel_data.append(None)
            continue

        # Conditional CDF for non-identity outcomes (k=1,...,K-1)
        cond_probs = probs[1:] / p_fire
        cond_cdf = np.cumsum(cond_probs).astype(np.float64)
        cond_cdf /= cond_cdf[-1]  # normalize to handle float rounding

        # Precompute XOR pattern for each non-identity outcome
        xor_patterns = np.zeros((n_outcomes - 1, num_outputs), dtype=np.uint8)
        for k in range(1, n_outcomes):
            for b in range(num_bits):
                if (k >> b) & 1:
                    xor_patterns[k - 1] ^= matrix_np[col_ids[b]]

        channel_data.append({
            'p_fire': p_fire,
            'cond_cdf': cond_cdf,
            'xor_patterns': xor_patterns,
            'n_non_id': n_outcomes - 1,
        })

    rng = np.random.default_rng(seed)

    def sparse_sample(num_samples):
        result = np.zeros((num_samples, num_outputs), dtype=np.uint8)
        noisy_mask = np.zeros(num_samples, dtype=np.bool_)

        for ch_data in channel_data:
            if ch_data is None:
                continue

            p_fire = ch_data['p_fire']
            cond_cdf = ch_data['cond_cdf']
            xor_pats = ch_data['xor_patterns']

            # Geometric skip: generate enough draws that cumsum reaches num_samples.
            # Each Geom(p) draw has mean 1/p, variance (1-p)/p².
            # For n draws, cumsum mean = n/p, std = sqrt(n*(1-p))/p.
            # We need cumsum > num_samples with overwhelming probability:
            #   n/p - 6*sqrt(n*(1-p))/p > num_samples
            # Conservative: n = expected + 6σ + 100 safety margin.
            expected_fires = num_samples * p_fire
            sigma = np.sqrt(expected_fires * (1.0 - p_fire))
            n_gen = max(int(expected_fires + 6 * sigma + 100), 100)
            skips = rng.geometric(p_fire, size=n_gen)
            positions = np.cumsum(skips) - 1  # 0-indexed, sorted
            positions = positions[positions < num_samples]
            n_fires = len(positions)

            if n_fires == 0:
                continue

            if ch_data['n_non_id'] == 1:
                # Binary channel: only one non-identity outcome
                result[positions] ^= xor_pats[0]
            else:
                # Multiple non-identity outcomes: sample which one
                u = rng.uniform(size=n_fires)
                outcome_idx = np.searchsorted(cond_cdf, u)
                result[positions] ^= xor_pats[outcome_idx]

            noisy_mask[positions] = True

        return result, noisy_mask

    return sparse_sample


def patch_channel_sampler_fast(sampler, verbose: bool = True) -> None:
    """Replace channel sampler with sparse geometric skip version."""
    cs = sampler._channel_sampler
    n_channels = len(cs.channels)
    n_outputs = cs.signature_matrix.shape[1]

    # Derive numpy seed from JAX key for reproducibility
    seed = int(jax.random.bits(cs._key, dtype=jnp.uint32))
    sparse_fn = _make_sparse_sample_fn(cs.channels, cs.signature_matrix, seed)

    total_outcomes = sum(len(ch.logits) for ch in cs.channels)
    avg_outcomes = total_outcomes / n_channels

    # Compute average p_fire for stats
    p_fires = []
    for ch in cs.channels:
        probs = np.asarray(jax.nn.softmax(ch.logits), dtype=np.float64)
        p_fires.append(1.0 - float(probs[0]))
    avg_p_fire = float(np.mean(p_fires))

    cs.sample = lambda num_samples=1: sparse_fn(num_samples)

    if verbose:
        print(
            f"Fast channel sampler: sparse geometric ({n_channels} channels, "
            f"avg {avg_outcomes:.1f} outcomes, {n_outputs} f-outputs, "
            f"avg p_fire={avg_p_fire:.4f})"
        )
