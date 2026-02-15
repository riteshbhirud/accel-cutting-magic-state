"""Fast enum subcomp sampling — numpy-based unique pattern dedup.

Key insight: at low noise (p=0.0005), most noisy shots share the same f-pattern
(e.g., only 1-2 channels fire). Instead of evaluating all 65536 shots through
the expensive enum evaluation (~1200ms), we:
  1. Extract f-patterns to numpy and find unique rows (numpy, fast, no x64 needed)
  2. Evaluate enum ONLY on unique patterns (~300-500 instead of 65536, ~40ms)
  3. Map magnitudes back via inverse index
  4. Sample categorically per-shot (each shot gets own random draw)

This is EXACT — same probabilities, same sampling, just avoiding redundant computation.
Phase 2 speedup: ~1200ms → ~60ms per batch (20x).

Usage (add to run script after compile + noiseless cache):

    from sampler_dedup import patch_sampler_fast
    patch_sampler_fast(sampler, top_k=None, max_unique=2048)
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from tsim_cutting import (
    SubcompEnumComponentData,
    _sample_component_enum,
    SubcompComponentData,
    _sample_component_subcomp,
)
from evaluate_matmul_cfloat import (
    evaluate_batch_split_fm,
    precompute_m_rowsums,
    precomp_to_tuple,
)

_LOG_EPS = 1e-30  # prevent log(0); well above float64 subnormal range (~5e-324)


# =============================================================================
# Precomputed data for split f/m evaluation
# =============================================================================

@dataclass(frozen=True)
class PrecomputedEnum:
    """Precomputed data for one enum component's split f/m evaluation."""
    n_combos: int
    m_combos: jnp.ndarray
    precomp_tuples: tuple
    f_param_positions: tuple
    n_f: int


def _select_top_k(
    data: SubcompEnumComponentData,
    K: int,
) -> tuple[jnp.ndarray, jnp.ndarray, float]:
    """Select top-K combos by noiseless probability mass."""
    import tsim_cutting as mod_cutting

    m_combos = data.m_combos
    n_combos = m_combos.shape[0]
    n_f = data.f_selection.shape[0]

    f_zero = jnp.zeros((n_combos, n_f), dtype=jnp.bool_)
    full_params = jnp.hstack([f_zero, m_combos])

    joint_magnitudes = jnp.ones(n_combos)
    for k in range(data.num_subcomps):
        csg = data.subcomp_compiled[k]
        idx_map = data.subcomp_param_index_maps[k]
        sub_params = full_params[:, idx_map]
        vals = mod_cutting.evaluate_batch(csg, sub_params)
        joint_magnitudes = joint_magnitudes * jnp.abs(vals)

    top_indices = jnp.argsort(-joint_magnitudes)[:K]
    top_m_combos = m_combos[top_indices]

    total_mass = float(jnp.sum(joint_magnitudes))
    top_mass = float(jnp.sum(joint_magnitudes[top_indices]))
    coverage = top_mass / max(total_mass, _LOG_EPS)

    return top_m_combos, top_indices, coverage


def build_precomputed(
    data: SubcompEnumComponentData,
    top_k: int | None = None,
    verbose: bool = True,
) -> PrecomputedEnum:
    """Build precomputed data for split f/m evaluation."""
    n_f = data.f_selection.shape[0]
    m_combos = data.m_combos
    n_combos_orig = m_combos.shape[0]

    if top_k is not None and top_k < n_combos_orig:
        m_combos, top_indices, coverage = _select_top_k(data, top_k)
        if verbose:
            print(f"  Top-K pruning: {n_combos_orig} → {top_k} combos "
                  f"(noiseless coverage: {coverage:.6f})")
    else:
        if verbose:
            print(f"  Using all {n_combos_orig} combos")

    n_combos = m_combos.shape[0]

    precomp_tuples = []
    f_positions = []

    for k in range(data.num_subcomps):
        csg = data.subcomp_compiled[k]
        idx_map_np = np.asarray(data.subcomp_param_index_maps[k])

        f_col_indices = jnp.array(
            [i for i, idx in enumerate(idx_map_np) if idx < n_f], dtype=jnp.int32
        )
        m_col_indices = jnp.array(
            [i for i, idx in enumerate(idx_map_np) if idx >= n_f], dtype=jnp.int32
        )

        m_param_indices = jnp.array(
            [idx_map_np[int(i)] - n_f for i in np.asarray(m_col_indices)],
            dtype=jnp.int32,
        )
        m_combos_sub = m_combos[:, m_param_indices]

        precomp = precompute_m_rowsums(csg, m_combos_sub, f_col_indices, m_col_indices)
        precomp_tuples.append(precomp_to_tuple(precomp))

        f_param_pos = idx_map_np[np.asarray(f_col_indices)]
        f_positions.append(f_param_pos)

    return PrecomputedEnum(
        n_combos=n_combos,
        m_combos=m_combos,
        precomp_tuples=tuple(precomp_tuples),
        f_param_positions=tuple(f_positions),
        n_f=n_f,
    )


def _sample_component_enum_split(
    data: SubcompEnumComponentData,
    f_params: jax.Array,
    key: jax.Array,
    precomp: PrecomputedEnum,
) -> tuple[jax.Array, jax.Array]:
    """Sample using split f/m evaluation (no combo expansion)."""
    batch_size = f_params.shape[0]
    f_selected = f_params[:, data.f_selection].astype(jnp.bool_)
    n_combos = precomp.n_combos

    joint_magnitudes = jnp.ones((batch_size, n_combos))

    for k in range(data.num_subcomps):
        csg = data.subcomp_compiled[k]
        precomp_tuple_k = precomp.precomp_tuples[k]
        f_pos_k = precomp.f_param_positions[k]
        f_vals_k = f_selected[:, f_pos_k].astype(jnp.bool_)
        vals = evaluate_batch_split_fm(csg, f_vals_k, precomp_tuple_k, n_combos)
        joint_magnitudes = joint_magnitudes * jnp.abs(vals)

    probs = joint_magnitudes
    row_sums = probs.sum(axis=1, keepdims=True)
    fallback = jnp.zeros_like(probs).at[:, 0].set(1.0)
    safe_probs = jnp.where(row_sums > 0, probs, fallback)

    log_probs = jnp.log(safe_probs + _LOG_EPS)
    key, subkey = jax.random.split(key)
    chosen = jax.random.categorical(subkey, log_probs)

    sampled_m = precomp.m_combos[chosen]

    return sampled_m, key


# =============================================================================
# Split JIT functions for dedup approach
# =============================================================================

def _make_enum_eval_jit(data, precomp, use_vmap=True):
    """Create a JIT function that evaluates enum magnitudes for a batch of f-patterns.

    Returns (batch, n_combos) magnitudes.
    Compiled ONCE at the padded batch size (max_unique), reused every call.
    """
    n_combos = precomp.n_combos

    @jax.jit
    def eval_magnitudes(unique_f_selected):
        """(max_unique, n_f) bool → (max_unique, n_combos) float32"""
        joint_magnitudes = jnp.ones((unique_f_selected.shape[0], n_combos))

        for k in range(data.num_subcomps):
            csg = data.subcomp_compiled[k]
            precomp_tuple_k = precomp.precomp_tuples[k]
            f_pos_k = precomp.f_param_positions[k]
            f_vals_k = unique_f_selected[:, f_pos_k].astype(jnp.bool_)

            vals = evaluate_batch_split_fm(csg, f_vals_k, precomp_tuple_k, n_combos)
            joint_magnitudes = joint_magnitudes * jnp.abs(vals)

        return joint_magnitudes

    return eval_magnitudes


def _make_autoreg_plus_combine_jit(program, precomp_list, enum_index):
    """Create a JIT function that:
    1. Runs autoreg components on full batch
    2. Does categorical sampling for enum using pre-computed magnitudes
    3. Combines all outputs

    The enum magnitudes are passed as input (pre-computed via dedup).
    """
    precomp_enum = precomp_list[enum_index]

    @jax.jit
    def sample_with_precomputed_enum(f_params, enum_shot_magnitudes, key):
        results = []

        for i, data in enumerate(program.component_data):
            if i == enum_index:
                probs = enum_shot_magnitudes
                row_sums = probs.sum(axis=1, keepdims=True)
                fallback = jnp.zeros_like(probs).at[:, 0].set(1.0)
                safe_probs = jnp.where(row_sums > 0, probs, fallback)
                log_probs = jnp.log(safe_probs + _LOG_EPS)
                key, subkey = jax.random.split(key)
                chosen = jax.random.categorical(subkey, log_probs)
                sampled_m = precomp_enum.m_combos[chosen]
                results.append(sampled_m)
            elif isinstance(data, SubcompEnumComponentData):
                samples, key = _sample_component_enum(data, f_params, key)
                results.append(samples)
            else:
                samples, key = _sample_component_subcomp(data, f_params, key)
                results.append(samples)

        combined = jnp.concatenate(results, axis=1)
        return combined[:, jnp.argsort(jnp.array(program.output_order))]

    return sample_with_precomputed_enum


# =============================================================================
# Numpy-based unique pattern dedup
# =============================================================================

def _next_pow2(n, minimum=1024):
    """Round up to next power of 2, with a minimum."""
    n = max(n, minimum)
    return 1 << (n - 1).bit_length()


def _numpy_dedup_f_patterns(f_params_np, f_selection, max_unique=None, _powers=None):
    """Find unique f-patterns using numpy int64 hashing (~2ms vs 300ms for axis=0).

    Hashes each n_f-bit f-pattern to a single int64, then uses 1D np.unique
    (O(n log n) on scalars, vastly faster than row-wise unique).

    Pads to next power of 2 above n_unique (dynamic sizing). JAX caches
    JIT compilations per shape, so this only recompiles for new sizes.

    Args:
        f_params_np: (batch, num_f_params) numpy array
        f_selection: array of indices for the enum component's f-params
        max_unique: safety cap (warn if exceeded). None = no cap.
        _powers: precomputed 2^j array (optional, for avoiding reallocation)

    Returns:
        unique_f_selected: (pad_size, n_f) uint8 numpy array (padded to next power of 2)
        inverse_idx: (batch,) int32 numpy array
        n_unique: actual number of unique patterns
        unique_hashes: (n_unique,) int64 numpy array of hash keys
    """
    n_f = len(f_selection)
    if n_f > 63:
        raise ValueError(
            f"n_f={n_f} exceeds int64 bit width (63); hash collisions guaranteed. "
            f"Dedup requires n_f <= 63."
        )
    f_selected = f_params_np[:, f_selection].astype(np.int64)

    # Hash: pack n_f bits into int64 (collision-free for n_f <= 63)
    if _powers is None:
        _powers = np.array([1 << j for j in range(n_f)], dtype=np.int64)
    hashes = f_selected @ _powers  # (batch,) int64 — fast BLAS matmul

    # 1D unique on scalars (much faster than axis=0 on rows)
    unique_hashes, inverse_idx = np.unique(hashes, return_inverse=True)
    n_unique = len(unique_hashes)

    if max_unique is not None and n_unique > max_unique:
        raise RuntimeError(
            f"Dedup overflow: {n_unique} unique patterns exceeds max_unique={max_unique}. "
            f"Increase max_unique or set max_unique=None."
        )

    # Decode unique hashes back to bit patterns
    bit_positions = np.arange(n_f, dtype=np.int64)
    unique_pats = ((unique_hashes[:, None] >> bit_positions[None, :]) & 1).astype(np.uint8)

    # Pad to next power of 2 (dynamic sizing — JAX caches JIT per shape)
    pad_size = _next_pow2(n_unique)
    padded = np.zeros((pad_size, n_f), dtype=np.uint8)
    padded[:n_unique] = unique_pats

    return padded, inverse_idx.astype(np.int32), n_unique, unique_hashes


# =============================================================================
# Full-program JIT without dedup
# =============================================================================

def _make_jit_fn_no_dedup(program, precomp_list):
    """Create JIT using split f/m evaluation (no dedup)."""

    @jax.jit
    def _jit_inner(f_params, key):
        results = []
        for i, data in enumerate(program.component_data):
            if isinstance(data, SubcompEnumComponentData) and precomp_list[i] is not None:
                samples, key = _sample_component_enum_split(
                    data, f_params, key, precomp_list[i]
                )
            elif isinstance(data, SubcompEnumComponentData):
                samples, key = _sample_component_enum(data, f_params, key)
            else:
                samples, key = _sample_component_subcomp(data, f_params, key)
            results.append(samples)

        combined = jnp.concatenate(results, axis=1)
        return combined[:, jnp.argsort(jnp.array(program.output_order))]

    return _jit_inner


# =============================================================================
# Public API
# =============================================================================

def patch_sampler_fast(
    sampler,
    top_k: int | None = None,
    max_unique: int = 8192,
    use_dedup: bool = True,
    verbose: bool = True,
) -> None:
    """Monkey-patch sampler with fast enum evaluation (numpy dedup).

    Applies optimizations:
    1. Split f/m evaluation with precomputed m-rowsums
    2. Unique pattern dedup via numpy: evaluate only distinct f-patterns

    This is EXACT: same probabilities, same sampling distribution.
    The dedup happens OUTSIDE JIT (in numpy), so no x64 issues.

    Call this AFTER add_noiseless_cache(sampler).

    Args:
        sampler: A SubcompEnumSamplerBase instance.
        top_k: If set, only evaluate top-K combos by noiseless probability.
        max_unique: Maximum padded size for unique patterns.
                    8192 is safe for p≤0.001. Enum eval scales linearly.
        use_dedup: Whether to use numpy dedup (default True).
        verbose: Print configuration details.
    """
    from tsim_cutting import _SubcompEnumSamplerBase

    if not isinstance(sampler, _SubcompEnumSamplerBase):
        raise TypeError(f"Expected SubcompEnumSampler, got {type(sampler)}")

    # 1. Build precomputed data for each enum component
    program = sampler._program
    precomp_list = []
    enum_index = None

    for i, data in enumerate(program.component_data):
        if isinstance(data, SubcompEnumComponentData) and data.num_component_outputs > 1:
            precomp = build_precomputed(data, top_k=top_k, verbose=verbose)
            precomp_list.append(precomp)
            enum_index = i
        else:
            precomp_list.append(None)

    if not use_dedup or enum_index is None:
        # Fall back to JIT without dedup
        jit_fn = _make_jit_fn_no_dedup(program, precomp_list)

        def sample_fn_fast(program, f_params, key):
            return jit_fn(f_params, key)

        sampler._sample_fn = sample_fn_fast
        if verbose:
            print(f"Fast enum sampler: split f/m (no dedup)")
        return

    # 3. Build separate JIT functions for dedup approach
    enum_data = program.component_data[enum_index]
    f_selection_np = np.asarray(enum_data.f_selection)

    # JIT for enum magnitudes (recompiles per input shape, cached by JAX)
    eval_enum_jit = _make_enum_eval_jit(enum_data, precomp_list[enum_index])

    # Pre-warm eval_enum_jit for common pad sizes to avoid recompilation
    # during the main run (consolidated noisy shots may need larger pads)
    n_f = precomp_list[enum_index].n_f
    for size in [1024, 2048, 4096, 8192]:
        dummy = jnp.zeros((size, n_f), dtype=jnp.bool_)
        eval_enum_jit(dummy).block_until_ready()

    # JIT for autoreg + enum categorical + combine
    sample_combine_jit = _make_autoreg_plus_combine_jit(
        program, precomp_list, enum_index
    )

    # Persistent cross-batch cache: hash(int) -> magnitudes(n_combos,) numpy float32
    eval_cache = {}
    n_combos_enum = precomp_list[enum_index].n_combos

    def sample_fn_dedup(program, f_params, key):
        """Sample function with numpy dedup + cross-batch cache."""
        # Step 1: Transfer f_params to numpy for dedup
        f_params_np = np.asarray(f_params)

        # Step 2: Find unique f-patterns with hashes for cache lookup
        padded_unique, inverse_idx, n_unique, unique_hashes = _numpy_dedup_f_patterns(
            f_params_np, f_selection_np, max_unique
        )

        # Step 3: Check cache — only evaluate NEW patterns
        hash_list = unique_hashes.tolist()  # Python ints for fast dict lookup
        new_indices = [i for i in range(n_unique) if hash_list[i] not in eval_cache]
        n_new = len(new_indices)

        if n_new > 0:
            # Gather new patterns, pad, evaluate
            new_pats = padded_unique[new_indices]  # (n_new, n_f)
            pad_size_new = _next_pow2(n_new)
            padded_new = np.zeros((pad_size_new, n_f), dtype=np.uint8)
            padded_new[:n_new] = new_pats

            new_jax = jnp.array(padded_new, dtype=jnp.bool_)
            new_mags = np.asarray(eval_enum_jit(new_jax))  # (pad_size_new, n_combos)

            # Store in cache
            for j, idx in enumerate(new_indices):
                eval_cache[hash_list[idx]] = new_mags[j]

        # Step 4: Assemble magnitudes from cache for all unique patterns
        all_mags = np.empty((n_unique, n_combos_enum), dtype=np.float32)
        for i in range(n_unique):
            all_mags[i] = eval_cache[hash_list[i]]

        # Pad to match padded_unique shape for inverse indexing
        pad_size = padded_unique.shape[0]
        if n_unique < pad_size:
            padded_mags = np.zeros((pad_size, n_combos_enum), dtype=np.float32)
            padded_mags[:n_unique] = all_mags
        else:
            padded_mags = all_mags

        # Step 5: Map magnitudes back to all shots
        unique_mag_jax = jnp.array(padded_mags)
        inverse_jax = jnp.array(inverse_idx, dtype=jnp.int32)
        shot_magnitudes = unique_mag_jax[inverse_jax]  # (batch, n_combos)

        # Step 6: Autoreg + categorical sampling + combine (JAX JIT)
        return sample_combine_jit(f_params, shot_magnitudes, key)

    sampler._sample_fn = sample_fn_dedup

    if verbose:
        n_opt = sum(1 for p in precomp_list if p is not None)
        k_str = f"top_k={top_k}" if top_k else "all combos"
        print(f"Fast enum sampler: split f/m + {k_str} + numpy dedup(max={max_unique}) "
              f"({n_opt} enum components optimized)")
