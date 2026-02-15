"""Complex-float evaluate_batch + split f/m variant.

Drop-in replacement for tsim.compile.evaluate.evaluate_batch.
All arithmetic is done in complex64 (float32 real/imag) — no ExactScalarArray,
no associative_scan of _scalar_mul, no 4-component int32 coefficient tracking.

Precision: float32 matmul for binary rowsums is exact for P < 2^24 parameters
(sum of 0/1 values stays representable). Unit-circle lookups (e^{i*pi*k/4})
have ~2e-7 relative error at k=1,3,5,7; accumulated product error across ~32
terms is ~3e-7, well below the O(1) scale of the amplitudes being computed.

Also contains the split f/m variant that separates f-params from m-params
for the dedup optimization (precompute m-rowsums once, reuse per batch).

Usage (monkey-patch):
    from evaluate_matmul_cfloat import evaluate_batch as evaluate_batch_cfloat
    import tsim_cutting as mod_cutting
    mod_cutting.evaluate_batch = evaluate_batch_cfloat
"""

import functools
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array

from tsim.compile.compile import CompiledScalarGraphs


# =============================================================================
# Shared primitives
# =============================================================================

def _batched_rowsum(param_bits: Array, param_vals: Array) -> Array:
    """Compute binary row sums via float32 matmul (activates BLAS).

    Args:
        param_bits: (G, T, P) uint8 array.
        param_vals: (batch, P) array.

    Returns:
        (batch, G, T) int32 array of row sums mod 2.
    """
    G, T, P = param_bits.shape
    batch = param_vals.shape[0]
    if T == 0:
        return jnp.zeros((batch, G, 0), dtype=jnp.int32)
    bits_2d = param_bits.reshape(G * T, P).astype(jnp.float32)
    vals_t = param_vals.T.astype(jnp.float32)
    raw = bits_2d @ vals_t
    return (raw.reshape(G, T, batch).transpose(2, 0, 1).astype(jnp.int32)) % 2


# Complex64 lookup tables for omega^k = e^(i*pi*k/4), k=0..7
_UNIT_C = jnp.array(
    [jnp.exp(1j * jnp.pi * k / 4) for k in range(8)], dtype=jnp.complex64
)

# (1 + omega^k) as complex64
_ONE_PLUS_C = 1.0 + _UNIT_C

# For converting ExactScalar int32 coefficients [a, b, c, d] to complex:
# value = a + b * e^(i*pi/4) + c * i + d * e^(-i*pi/4)
_E4 = jnp.exp(1j * jnp.pi / 4).astype(jnp.complex64)
_E4D = jnp.exp(-1j * jnp.pi / 4).astype(jnp.complex64)


# =============================================================================
# Full evaluate_batch (monkey-patch replacement)
# =============================================================================

@functools.partial(jax.jit, static_argnums=(2,))
def _evaluate_batch_cfloat(
    circuit: CompiledScalarGraphs,
    param_vals: Array,
    has_approximate_floatfactor: bool,
) -> Array:
    """Batch evaluate compiled circuit using complex64 arithmetic throughout."""
    batch = param_vals.shape[0]

    # TYPE A: Node Terms (1 + e^(i*alpha))
    rowsum_a = _batched_rowsum(circuit.a_param_bits, param_vals)
    phase_idx_a = (4 * rowsum_a + circuit.a_const_phases) % 8
    vals_a = _ONE_PLUS_C[phase_idx_a]
    a_mask = (
        jnp.arange(circuit.a_const_phases.shape[1])[None, :]
        < circuit.a_num_terms[:, None]
    )
    vals_a = jnp.where(a_mask, vals_a, 1.0 + 0j)
    prod_a = jnp.prod(vals_a, axis=2)

    # TYPE B: Half-Pi Terms (e^(i*beta))
    rowsum_b = _batched_rowsum(circuit.b_param_bits, param_vals)
    phase_idx_b = (rowsum_b * circuit.b_term_types) % 8
    sum_phases_b = jnp.sum(phase_idx_b, axis=2) % 8
    val_b = _UNIT_C[sum_phases_b]

    # TYPE C: Pi-Pair Terms, (-1)^(Psi*Phi)
    rowsum_a_c = (
        _batched_rowsum(circuit.c_param_bits_a, param_vals) + circuit.c_const_bits_a
    ) % 2
    rowsum_b_c = (
        _batched_rowsum(circuit.c_param_bits_b, param_vals) + circuit.c_const_bits_b
    ) % 2
    exponent_c = (rowsum_a_c * rowsum_b_c) % 2
    sum_exp_c = jnp.sum(exponent_c, axis=2) % 2
    val_c = 1.0 - 2.0 * sum_exp_c

    # TYPE D: Phase Pairs (1 + e^a + e^b - e^(a+b))
    rowsum_a_d = _batched_rowsum(circuit.d_param_bits_a, param_vals)
    rowsum_b_d = _batched_rowsum(circuit.d_param_bits_b, param_vals)
    alpha = (circuit.d_const_alpha + rowsum_a_d * 4) % 8
    beta = (circuit.d_const_beta + rowsum_b_d * 4) % 8
    gamma = (alpha + beta) % 8
    vals_d = 1.0 + _UNIT_C[alpha] + _UNIT_C[beta] - _UNIT_C[gamma]
    d_mask = (
        jnp.arange(circuit.d_const_alpha.shape[1])[None, :]
        < circuit.d_num_terms[:, None]
    )
    vals_d = jnp.where(d_mask, vals_d, 1.0 + 0j)
    prod_d = jnp.prod(vals_d, axis=2)

    # Static per-graph factors
    static_c = _UNIT_C[circuit.phase_indices]
    ff = circuit.floatfactor.astype(jnp.float32)
    float_c = ff[:, 0] + ff[:, 1] * _E4 + ff[:, 2] * 1j + ff[:, 3] * _E4D
    scale = jnp.pow(2.0, circuit.power2.astype(jnp.float32))

    # Final combination
    total = prod_a * val_b * val_c * prod_d * static_c * float_c

    if not has_approximate_floatfactor:
        return jnp.sum(total * scale, axis=-1)
    else:
        return jnp.sum(
            total * scale * circuit.approximate_floatfactors, axis=-1
        )


def evaluate_batch(circuit: CompiledScalarGraphs, param_vals: Array) -> Array:
    """Drop-in replacement for tsim.compile.evaluate.evaluate_batch."""
    return _evaluate_batch_cfloat(
        circuit, param_vals, circuit.has_approximate_floatfactors
    )


# =============================================================================
# Split f/m precomputation (called once at compile time)
# =============================================================================

@dataclass(frozen=True)
class PrecomputedMRowsums:
    """Precomputed m-contribution rowsums and f-only param bit slices."""
    m_rs_d_a: jnp.ndarray
    m_rs_d_b: jnp.ndarray
    m_rs_a: jnp.ndarray
    m_rs_b: jnp.ndarray
    m_rs_c_a: jnp.ndarray
    m_rs_c_b: jnp.ndarray
    d_bits_a_f: jnp.ndarray
    d_bits_b_f: jnp.ndarray
    a_bits_f: jnp.ndarray
    b_bits_f: jnp.ndarray
    c_bits_a_f: jnp.ndarray
    c_bits_b_f: jnp.ndarray
    n_combos: int


def precompute_m_rowsums(
    csg: CompiledScalarGraphs,
    m_combos_sub: jnp.ndarray,
    f_col_indices: jnp.ndarray,
    m_col_indices: jnp.ndarray,
) -> PrecomputedMRowsums:
    """Precompute m-contribution rowsums for all term types."""
    m_bool = m_combos_sub.astype(jnp.bool_)

    m_rs_d_a = _batched_rowsum(csg.d_param_bits_a[:, :, m_col_indices], m_bool)
    m_rs_d_b = _batched_rowsum(csg.d_param_bits_b[:, :, m_col_indices], m_bool)
    m_rs_a = _batched_rowsum(csg.a_param_bits[:, :, m_col_indices], m_bool)
    m_rs_b = _batched_rowsum(csg.b_param_bits[:, :, m_col_indices], m_bool)
    m_rs_c_a = _batched_rowsum(csg.c_param_bits_a[:, :, m_col_indices], m_bool)
    m_rs_c_b = _batched_rowsum(csg.c_param_bits_b[:, :, m_col_indices], m_bool)

    return PrecomputedMRowsums(
        m_rs_d_a=m_rs_d_a, m_rs_d_b=m_rs_d_b,
        m_rs_a=m_rs_a, m_rs_b=m_rs_b,
        m_rs_c_a=m_rs_c_a, m_rs_c_b=m_rs_c_b,
        d_bits_a_f=csg.d_param_bits_a[:, :, f_col_indices],
        d_bits_b_f=csg.d_param_bits_b[:, :, f_col_indices],
        a_bits_f=csg.a_param_bits[:, :, f_col_indices],
        b_bits_f=csg.b_param_bits[:, :, f_col_indices],
        c_bits_a_f=csg.c_param_bits_a[:, :, f_col_indices],
        c_bits_b_f=csg.c_param_bits_b[:, :, f_col_indices],
        n_combos=int(m_combos_sub.shape[0]),
    )


def precomp_to_tuple(precomp: PrecomputedMRowsums) -> tuple:
    """Convert PrecomputedMRowsums to a flat tuple for JAX JIT compatibility."""
    return (
        precomp.m_rs_d_a, precomp.m_rs_d_b,
        precomp.m_rs_a, precomp.m_rs_b,
        precomp.m_rs_c_a, precomp.m_rs_c_b,
        precomp.d_bits_a_f, precomp.d_bits_b_f,
        precomp.a_bits_f, precomp.b_bits_f,
        precomp.c_bits_a_f, precomp.c_bits_b_f,
    )


# =============================================================================
# Split f/m evaluate_batch
# =============================================================================

def evaluate_batch_split_fm(
    csg: CompiledScalarGraphs,
    f_vals: Array,
    precomp_tuple: tuple,
    n_combos: int,
) -> Array:
    """Evaluate all (batch, combo) pairs with split f/m computation.

    Uses jax.lax.scan over combos instead of Python for-loop unrolling.
    Compiles a single loop body (vs 32 copies), reducing i-cache pressure.

    Precomputes D-term lookup table (4 values per position) and
    combo-independent f-contribution rowsums.

    Returns (batch, n_combos) complex64 amplitudes.
    """
    (m_rs_d_a, m_rs_d_b, m_rs_a, m_rs_b, m_rs_c_a, m_rs_c_b,
     d_bits_a_f, d_bits_b_f, a_bits_f, b_bits_f, c_bits_a_f, c_bits_b_f) = precomp_tuple

    # f-contribution rowsums (batch-dependent, combo-independent)
    f_rs_d_a = _batched_rowsum(d_bits_a_f, f_vals)
    f_rs_d_b = _batched_rowsum(d_bits_b_f, f_vals)
    f_rs_c_a = _batched_rowsum(c_bits_a_f, f_vals)
    f_rs_c_b = _batched_rowsum(c_bits_b_f, f_vals)
    f_rs_a = _batched_rowsum(a_bits_f, f_vals)
    f_rs_b = _batched_rowsum(b_bits_f, f_vals)

    # Static per-graph factors
    static_c = _UNIT_C[csg.phase_indices]
    ff = csg.floatfactor.astype(jnp.float32)
    float_c = ff[:, 0] + ff[:, 1] * _E4 + ff[:, 2] * 1j + ff[:, 3] * _E4D
    scale = jnp.pow(2.0, csg.power2.astype(jnp.float32))
    has_approx = csg.has_approximate_floatfactors
    approx_ff = csg.approximate_floatfactors if has_approx else jnp.ones_like(scale)

    # ---- D-term precomputation: 4 possible values per (G, T_d) ----
    d_mask = (
        jnp.arange(csg.d_const_alpha.shape[1])[None, :]
        < csg.d_num_terms[:, None]
    )

    def _d_val(ca, cb):
        g = (ca + cb) % 8
        return jnp.where(d_mask, 1.0 + _UNIT_C[ca] + _UNIT_C[cb] - _UNIT_C[g], 1.0 + 0j)

    ca0, ca1 = csg.d_const_alpha, (csg.d_const_alpha + 4) % 8
    cb0, cb1 = csg.d_const_beta, (csg.d_const_beta + 4) % 8
    d_00 = _d_val(ca0, cb0)
    d_01 = _d_val(ca0, cb1)
    d_10 = _d_val(ca1, cb0)
    d_11 = _d_val(ca1, cb1)

    # A-term mask
    a_mask = (
        jnp.arange(csg.a_const_phases.shape[1])[None, :]
        < csg.a_num_terms[:, None]
    )

    # Closured constants for scan body
    c_const_a = csg.c_const_bits_a
    c_const_b = csg.c_const_bits_b
    a_const_ph = csg.a_const_phases
    b_types = csg.b_term_types

    def combo_body(carry, combo_slice):
        m_d_a_c, m_d_b_c, m_c_a_c, m_c_b_c, m_a_c, m_b_c = combo_slice

        # A-terms (vary per combo via m_rs_a)
        rs_a = (f_rs_a + m_a_c) % 2
        phase_idx_a = (4 * rs_a + a_const_ph) % 8
        vals_a = jnp.where(a_mask, _ONE_PLUS_C[phase_idx_a], 1.0 + 0j)
        prod_a = jnp.prod(vals_a, axis=2)

        # B-terms (vary per combo via m_rs_b)
        rs_b = (f_rs_b + m_b_c) % 2
        phase_idx_b = (rs_b * b_types) % 8
        sum_phases_b = jnp.sum(phase_idx_b, axis=2) % 8
        val_b = _UNIT_C[sum_phases_b]

        # D-terms: select from precomputed table
        rs_d_a = (f_rs_d_a + m_d_a_c) % 2
        rs_d_b = (f_rs_d_b + m_d_b_c) % 2
        vals_d = jnp.where(
            rs_d_a,
            jnp.where(rs_d_b, d_11, d_10),
            jnp.where(rs_d_b, d_01, d_00),
        )
        prod_d = jnp.prod(vals_d, axis=2)

        # C-terms (vary per combo)
        rs_c_a = (f_rs_c_a + m_c_a_c + c_const_a) % 2
        rs_c_b = (f_rs_c_b + m_c_b_c + c_const_b) % 2
        sum_exp_c = jnp.sum((rs_c_a * rs_c_b) % 2, axis=2) % 2
        val_c = 1.0 - 2.0 * sum_exp_c

        total = prod_a * val_b * val_c * prod_d * static_c * float_c
        val = jnp.sum(total * scale * approx_ff, axis=-1)

        return carry, val

    # Scan over combos: combo_data has leading axis n_combos
    combo_data = (m_rs_d_a, m_rs_d_b, m_rs_c_a, m_rs_c_b, m_rs_a, m_rs_b)
    _, all_vals = jax.lax.scan(combo_body, None, combo_data)
    # all_vals: (n_combos, batch) → (batch, n_combos)
    return all_vals.T
