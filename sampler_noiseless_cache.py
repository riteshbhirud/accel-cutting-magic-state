"""Per-shot noiseless caching for tsim samplers.

Two-pass approach:
  Phase 1: For each batch, sample f-params + cache noiseless results.
           Collect noisy f-params for later.
  Phase 2: Process ALL noisy shots consolidated into full-size sub-batches
           at batch_size (already JIT-traced, zero retrace overhead).

This eliminates per-batch JIT retracing and reduces total pipeline calls
to ceil(n_noisy_total / batch_size) instead of n_batches.

Usage:
    from sampler_noiseless_cache import add_noiseless_cache
    add_noiseless_cache(sampler_enum)    # patches _sample_batches in-place
    det, obs = sampler_enum.sample(shots=524288, batch_size=1024,
                                    separate_observables=True)
"""

from __future__ import annotations

import types
from dataclasses import dataclass
from math import ceil

import jax
import jax.numpy as jnp
import numpy as np

_LOG_EPS = 1e-30  # prevent log(0) and division by zero in probability computations

# Import evaluate_batch from the modules (respects monkey-patching)
import tsim_cutting as mod_cutting
from tsim_cutting import (
    SubcompEnumComponentData,
    sample_program_subcomp_enum,
    SubcompComponentData,
    sample_program_subcomp,
)


# =============================================================================
# Cache data structure — optimized for batched sampling
# =============================================================================

@dataclass
class NoiselessCache:
    """Pre-computed noiseless distributions, structured for fast batched sampling.

    Binary components (1 output, 2 combos) are batched into a single
    Bernoulli call. Multi-output components use individual categorical calls.

    Stores both numpy (for fast CPU sampling) and JAX (for categorical/GPU)
    versions of the probability arrays.
    """
    # Binary (1-output) components — numpy for fast CPU Bernoulli
    binary_probs_np: np.ndarray | None  # (n_binary,) float64 — P(output=1)
    binary_final_cols: np.ndarray       # (n_binary,) int — column index in final output

    # Multi-output components — JAX for categorical sampling
    cat_log_probs: list[jax.Array]      # list of (n_combos,) float32
    cat_probs_np: list[np.ndarray]      # list of (n_combos,) float64 — normalized
    cat_m_combos_np: list[np.ndarray]   # list of (n_combos, n_outputs) bool
    cat_final_cols: list[np.ndarray]    # list of (n_outputs,) int — column indices

    num_total_outputs: int

    # Diagnostics
    n_binary: int
    n_categorical: int
    n_skipped: int  # components with 0 outputs


# =============================================================================
# Pre-computation
# =============================================================================

def _compute_component_log_probs(data, is_enum: bool) -> tuple[np.ndarray, np.ndarray]:
    """Compute noiseless log-probabilities for a component.

    Returns:
        log_probs: (n_combos,) float64
        m_combos: (n_combos, num_outputs) bool
    """
    if is_enum:
        return _compute_enum_log_probs(data)
    else:
        return _compute_autoregressive_log_probs(data)


def _compute_enum_log_probs(data: SubcompEnumComponentData) -> tuple[np.ndarray, np.ndarray]:
    """Compute noiseless log-probs for an enum component."""
    m_combos = np.asarray(data.m_combos)
    n_combos = m_combos.shape[0]
    n_f = data.f_selection.shape[0]

    f_zero = jnp.zeros((n_combos, n_f), dtype=jnp.bool_)
    full_params = jnp.hstack([f_zero, jnp.array(m_combos, dtype=jnp.bool_)])

    joint_magnitudes = jnp.ones(n_combos)
    for k in range(data.num_subcomps):
        csg = data.subcomp_compiled[k]
        idx_map = data.subcomp_param_index_maps[k]
        sub_params = full_params[:, idx_map]
        vals = mod_cutting.evaluate_batch(csg, sub_params)
        joint_magnitudes = joint_magnitudes * jnp.abs(vals)

    probs = np.asarray(joint_magnitudes, dtype=np.float64)
    return np.log(probs + _LOG_EPS), m_combos


def _compute_autoregressive_log_probs(data: SubcompComponentData) -> tuple[np.ndarray, np.ndarray]:
    """Compute noiseless log-probs for an autoregressive component."""
    num_outputs = len(data.output_indices)
    n_combos = 2 ** num_outputs
    n_f = data.f_selection.shape[0]

    m_combos = np.array(
        [[(j >> i) & 1 for i in range(num_outputs)] for j in range(n_combos)],
        dtype=np.bool_,
    )

    f_zero = jnp.zeros((n_combos, n_f), dtype=jnp.bool_)
    norm = jnp.abs(mod_cutting.evaluate_batch(data.compiled_scalar_graphs[0], f_zero))
    prev = norm
    joint_prob = jnp.ones(n_combos)
    ones = jnp.ones((n_combos, 1), dtype=jnp.bool_)
    m_jnp = jnp.array(m_combos, dtype=jnp.bool_)

    for i, circuit in enumerate(data.compiled_scalar_graphs[1:]):
        params = jnp.hstack([f_zero, m_jnp[:, :i], ones])
        p1 = jnp.abs(mod_cutting.evaluate_batch(circuit, params))
        p_one = jnp.clip(p1 / (prev + _LOG_EPS), 0.0, 1.0)
        bit_i = m_jnp[:, i].astype(jnp.float32)
        cond_prob = bit_i * p_one + (1 - bit_i) * (1 - p_one)
        joint_prob = joint_prob * cond_prob
        prev = jnp.where(m_jnp[:, i], p1, prev - p1)

    if data.has_product_level:
        i_last = num_outputs - 1
        params_full = jnp.hstack([f_zero, m_jnp[:, :i_last], ones])
        val = None
        for csg, idx_map in zip(
            data.product_compiled_subcomps, data.product_param_index_maps
        ):
            sub_params = params_full[:, idx_map]
            sub_val = mod_cutting.evaluate_batch(csg, sub_params)
            val = sub_val if val is None else val * sub_val
        p1 = jnp.abs(val)
        p_one = jnp.clip(p1 / (prev + _LOG_EPS), 0.0, 1.0)
        bit_last = m_jnp[:, i_last].astype(jnp.float32)
        cond_prob = bit_last * p_one + (1 - bit_last) * (1 - p_one)
        joint_prob = joint_prob * cond_prob

    magnitudes = np.asarray(joint_prob * norm, dtype=np.float64)
    return np.log(magnitudes + _LOG_EPS), m_combos


def _build_noiseless_cache(program) -> NoiselessCache:
    """Build a NoiselessCache with batched sampling arrays.

    Separates components into:
    - binary (1 output): batched into a single Bernoulli call
    - categorical (>1 output): individual categorical calls
    - skipped (0 outputs): ignored
    """
    all_output_indices = []
    for data in program.component_data:
        all_output_indices.extend(data.output_indices)
    sorted_globals = np.sort(np.array(all_output_indices))

    binary_probs_list = []
    binary_cols_list = []
    cat_log_probs = []
    cat_m_combos = []
    cat_final_cols = []
    n_skipped = 0

    for data in program.component_data:
        num_outputs = len(data.output_indices)
        if num_outputs == 0:
            n_skipped += 1
            continue

        is_enum = isinstance(data, SubcompEnumComponentData)
        log_probs, m_combos = _compute_component_log_probs(data, is_enum)

        # Map this component's output_indices to final column positions
        cols = np.searchsorted(sorted_globals, np.array(data.output_indices))

        if num_outputs == 1:
            # Binary component: extract P(output=1)
            probs = np.exp(log_probs - np.max(log_probs))  # numerically stable
            p_one = float(probs[1] / (probs[0] + probs[1]))
            binary_probs_list.append(p_one)
            binary_cols_list.append(cols[0])
        else:
            # Multi-output categorical
            cat_log_probs.append(jnp.array(log_probs, dtype=jnp.float32))
            cat_m_combos.append(jnp.array(m_combos))
            cat_final_cols.append(cols)

    n_binary = len(binary_probs_list)
    binary_probs_np = np.array(binary_probs_list, dtype=np.float64) if n_binary > 0 else None
    binary_final_cols = np.array(binary_cols_list, dtype=np.int32) if n_binary > 0 else np.array([], dtype=np.int32)

    # Numpy versions of categorical probabilities for fast CPU sampling
    cat_probs_np = []
    cat_m_combos_np = []
    for lp, mc in zip(cat_log_probs, cat_m_combos):
        lp_np = np.asarray(lp, dtype=np.float64)
        probs = np.exp(lp_np - np.max(lp_np))
        probs /= probs.sum()
        cat_probs_np.append(probs)
        cat_m_combos_np.append(np.asarray(mc))

    return NoiselessCache(
        binary_probs_np=binary_probs_np,
        binary_final_cols=binary_final_cols,
        cat_log_probs=cat_log_probs,
        cat_probs_np=cat_probs_np,
        cat_m_combos_np=cat_m_combos_np,
        cat_final_cols=cat_final_cols,
        num_total_outputs=program.num_outputs,
        n_binary=n_binary,
        n_categorical=len(cat_log_probs),
        n_skipped=n_skipped,
    )


# =============================================================================
# Fast batched noiseless sampling — pure numpy (GPU: swap to JAX trivially)
# =============================================================================

def _make_noiseless_sampler(cache: NoiselessCache, seed: int):
    """Create a fast numpy-based noiseless sampler closure.

    Uses numpy PCG64 for all random generation:
    - Binary: rng.random(n, 28) < probs  (~3B/sec on CPU)
    - Categorical: rng.choice with precomputed probabilities

    For GPU migration: replace rng.random with jax.random.uniform
    and rng.choice with jax.random.categorical (same interface).

    Returns: callable(n_shots) -> (n_shots, num_outputs) bool array
    """
    rng = np.random.default_rng(seed)
    binary_probs = cache.binary_probs_np
    binary_cols = cache.binary_final_cols
    n_binary = cache.n_binary
    cat_probs = cache.cat_probs_np
    cat_m_combos = cache.cat_m_combos_np
    cat_cols = cache.cat_final_cols
    n_out = cache.num_total_outputs

    # Pre-build categorical CDF for fast searchsorted sampling
    cat_cdfs = []
    for probs in cat_probs:
        cdf = np.cumsum(probs)
        cdf[-1] = 1.0  # numerical safety
        cat_cdfs.append(cdf)

    def sample_into(out: np.ndarray) -> None:
        """Write noiseless samples directly into a pre-zeroed output slice.

        Avoids per-batch np.zeros allocation (saves 112MB × 32 = 3.5GB alloc).
        Caller must ensure `out` is zeroed for columns not covered by components.
        """
        n_shots = out.shape[0]

        # Binary: column-by-column to keep each rng.random in L3 (~16MB vs 448MB)
        if binary_probs is not None:
            for i in range(n_binary):
                out[:, binary_cols[i]] = rng.random(n_shots) < binary_probs[i]

        # Categorical: searchsorted on pre-built CDF (faster than rng.choice)
        for cdf, mc, cols in zip(cat_cdfs, cat_m_combos, cat_cols):
            u = rng.random(n_shots)
            chosen = np.searchsorted(cdf, u)
            out[:, cols] = mc[chosen]

    return sample_into


# =============================================================================
# Two-pass cached _sample_batches
# =============================================================================

def _sample_batches_cached(
    self,
    shots: int,
    batch_size: int | None = None,
) -> np.ndarray:
    """Two-pass replacement for _sample_batches with noiseless caching."""
    if batch_size is None:
        batch_size = shots

    n_batches = ceil(shots / batch_size)
    # Single zero-init: noiseless sampler writes only its columns into this.
    # Noisy Phase 2 overwrites entire rows for noisy shots.
    all_results = np.zeros((n_batches * batch_size, self._program.num_outputs), dtype=np.bool_)

    noisy_global_indices: list[np.ndarray] = []
    noisy_f_params: list[np.ndarray] = []

    for b in range(n_batches):
        offset = b * batch_size

        ch_result = self._channel_sampler.sample(batch_size)
        self._noiseless_sampler(all_results[offset:offset + batch_size])

        if isinstance(ch_result, tuple):
            f_np, is_noisy = ch_result
        else:
            f_np = np.asarray(ch_result)
            is_noisy = ~np.all(f_np == 0, axis=1)

        if int(np.sum(is_noisy)) > 0:
            noisy_in_batch = np.where(is_noisy)[0]
            noisy_global_indices.append(offset + noisy_in_batch)
            noisy_f_params.append(f_np[noisy_in_batch])

    if noisy_global_indices:
        all_noisy_idx = np.concatenate(noisy_global_indices)
        all_noisy_f = np.concatenate(noisy_f_params, axis=0)
        n_noisy_total = len(all_noisy_idx)

        for sub_start in range(0, n_noisy_total, batch_size):
            sub_end = min(sub_start + batch_size, n_noisy_total)
            sub_f = all_noisy_f[sub_start:sub_end]
            sub_n = sub_end - sub_start

            if sub_n < batch_size:
                pad = np.tile(sub_f[-1:], (batch_size - sub_n, 1))
                sub_f = np.concatenate([sub_f, pad], axis=0)

            self._key, subkey = jax.random.split(self._key)
            sub_out = self._sample_fn(self._program, jnp.array(sub_f), subkey)
            all_results[all_noisy_idx[sub_start:sub_end]] = np.asarray(sub_out[:sub_n])

    return all_results[:shots]


# =============================================================================
# Public API
# =============================================================================

def add_noiseless_cache(sampler, verbose: bool = True) -> None:
    """Add noiseless caching to an existing compiled sampler.

    Monkey-patches _sample_batches to use cached noiseless distributions
    for shots where all f-params are zero.

    Args:
        sampler: A SubcompEnumDetectorSampler or SubcompDetectorSampler.
        verbose: Print cache statistics.
    """
    from tsim_cutting import _SubcompEnumSamplerBase, _SubcompSamplerBase

    # Determine which sample function to use for noisy shots
    if isinstance(sampler, _SubcompEnumSamplerBase):
        sample_fn = sample_program_subcomp_enum
    elif isinstance(sampler, _SubcompSamplerBase):
        sample_fn = sample_program_subcomp
    else:
        raise TypeError(f"Unsupported sampler type: {type(sampler)}")

    # Pre-compute cache
    cache = _build_noiseless_cache(sampler._program)

    # Measure noiseless fraction empirically
    test_result = sampler._channel_sampler.sample(4096)
    if isinstance(test_result, tuple):
        test_f, test_noisy = test_result
        noiseless_frac = 1.0 - float(np.mean(test_noisy))
    else:
        test_f = np.asarray(test_result)
        noiseless_frac = float(np.mean(np.all(test_f == 0, axis=1)))

    # Create numpy-based noiseless sampler (fast on CPU, trivially portable to GPU)
    seed = int(jax.random.bits(sampler._key, dtype=jnp.uint32))
    noiseless_sampler = _make_noiseless_sampler(cache, seed)

    # Attach to sampler
    sampler._cache = cache
    sampler._sample_fn = sample_fn
    sampler._noiseless_sampler = noiseless_sampler

    # Monkey-patch _sample_batches
    sampler._sample_batches = types.MethodType(_sample_batches_cached, sampler)

    if verbose:
        print(
            f"Noiseless cache: {cache.n_binary} binary + "
            f"{cache.n_categorical} categorical + {cache.n_skipped} skipped "
            f"components | noiseless fraction: {noiseless_frac:.1%} "
            f"(max theoretical speedup: {1/(1-noiseless_frac + noiseless_frac*0.01):.2f}x)"
        )
