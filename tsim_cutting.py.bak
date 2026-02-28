"""Stabiliser-rank cutting, subcomp compilation, and enum sampling.

Merges the cutting decomposition, subcomponent compilation, enumeration-based
sampling, and the general (non-disconnecting) variant into a single module.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from math import ceil
from typing import TYPE_CHECKING, Literal, overload

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pyzx_param as zx
from pyzx_param.graph.base import BaseGraph
from pyzx_param.graph.scalar import Scalar

from tsim.circuit import Circuit
from tsim.compile.compile import CompiledScalarGraphs, compile_scalar_graphs
from tsim.compile.evaluate import evaluate_batch
from tsim.compile.stabrank import find_stab
from tsim.core.graph import (
    ConnectedComponent, connected_components, get_params, prepare_graph)
from tsim.core.types import CompiledComponent, CompiledProgram, SamplingGraph
from tsim.noise.channels import ChannelSampler
from tsim.sampler import sample_program
from stab_rank_cut import decompose as stab_rank_decompose

_LOG_EPS = 1e-30  # prevent log(0)

# =============================================================================
# CUTTING DECOMPOSITION
# =============================================================================

if TYPE_CHECKING:
    from jax import Array as PRNGKey
    from tsim.circuit import Circuit


# =============================================================================
# Cutting-based stabiliser decomposition
# =============================================================================

def find_stab_cutting(
    graph: BaseGraph,
    max_cut_iterations: int = 10,
    debug: bool = False,
    cut_strategy: str = "fewest_neighbors",
    use_tsim_bss: bool = True,
    skip_initial_reduce: bool = False,
) -> list[BaseGraph]:
    """Decompose a ZX-graph using cutting first, then BSS.

    This is a drop-in replacement for tsim's find_stab that uses cutting
    to reduce T-count before applying BSS.

    Args:
        graph: The ZX graph to decompose.
        max_cut_iterations: Maximum cutting iterations.
        debug: If True, print decomposition progress.
        cut_strategy: Strategy for selecting vertices to cut.
        use_tsim_bss: If True, use tsim's find_stab for final BSS.
        skip_initial_reduce: If True, skip initial full_reduce before cutting.

    Returns:
        A list of Clifford graphs (T-count=0) whose sum equals the original.
    """
    # Step 1: Apply cutting decomposition
    cut_terms = stab_rank_decompose(
        graph,
        debug=debug,
        use_bss_fallback=False,  # Don't use internal BSS, we'll apply it ourselves
        max_iterations=max_cut_iterations,
        param_safe=True,
        cut_strategy=cut_strategy,
        use_tsim_bss=False,  # Don't use tsim BSS internally
    )

    if debug:
        print(f"Cutting produced {len(cut_terms)} terms")

    # Step 2: For each cut term, full_reduce, skip zero scalars, then find_stab (BSS)
    all_clifford_terms: list[BaseGraph] = []

    for i, term in enumerate(cut_terms):
        # Full reduce with param safety
        zx.full_reduce(term, paramSafe=True)

        # Skip terms with zero scalar (like tsim's _decompose does)
        if term.scalar.is_zero:
            if debug:
                print(f"  Cut term {i}: scalar is zero, skipping")
            if len(all_clifford_terms) > 0:
                continue
            # Keep at least one term to avoid empty results

        tc = zx.simplify.tcount(term)

        if tc == 0:
            # Already Clifford
            all_clifford_terms.append(term)
            if debug:
                print(f"  Cut term {i}: already Clifford (T-count=0)")
        else:
            # Apply BSS to finish decomposition
            if use_tsim_bss:
                clifford_terms = find_stab(term)
            else:
                clifford_terms = zx.simulate.find_stabilizer_decomp(term)
            all_clifford_terms.extend(clifford_terms)

            if debug:
                print(f"  Cut term {i}: T-count={tc} -> {len(clifford_terms)} Clifford terms")

    if debug:
        print(f"Total Clifford terms: {len(all_clifford_terms)}")

    return all_clifford_terms


def _graph_signature(graph: BaseGraph) -> tuple:
    """Create a hashable signature for a ZX graph structure.

    Two graphs with the same signature have identical structure (vertices, edges,
    types, phases) and can have their scalars combined.

    Returns:
        A tuple that uniquely identifies the graph structure.
    """
    # Get sorted vertex list for consistent ordering
    vertices = sorted(graph.vertices())

    # Create vertex mapping to canonical indices
    v_to_idx = {v: i for i, v in enumerate(vertices)}

    # Collect vertex info: (canonical_idx, type, phase, is_input, is_output)
    inputs = set(graph.inputs())
    outputs = set(graph.outputs())

    vertex_info = []
    for v in vertices:
        phase = graph.phase(v)
        # Convert phase to a hashable form
        if hasattr(phase, 'limit_denominator'):
            phase_key = (phase.numerator, phase.denominator)
        else:
            phase_key = float(phase)

        vertex_info.append((
            v_to_idx[v],
            graph.type(v),
            phase_key,
            v in inputs,
            v in outputs,
        ))

    # Collect edge info: (canonical_idx1, canonical_idx2, edge_type)
    edge_info = []
    for e in graph.edges():
        v1, v2 = graph.edge_st(e)
        idx1, idx2 = v_to_idx[v1], v_to_idx[v2]
        # Normalize edge direction for consistent hashing
        if idx1 > idx2:
            idx1, idx2 = idx2, idx1
        edge_info.append((idx1, idx2, graph.edge_type(e)))

    edge_info.sort()

    return (tuple(vertex_info), tuple(edge_info))


def _combine_scalars(base_graph: BaseGraph, scalars: list) -> BaseGraph:
    """Combine multiple scalars into a single graph.

    Creates a new graph with the combined scalar value.
    """
    from pyzx_param.graph.scalar import Scalar

    result = deepcopy(base_graph)

    # Sum all the scalar values
    # For exact arithmetic, we need to handle the scalar components
    combined = Scalar()

    for s in scalars:
        # Add each scalar to the combined total
        # This is approximate - for exact combination we'd need more complex logic
        combined = combined + s

    result.scalar = combined
    return result


def find_stab_cutting_optimised(
    graph: BaseGraph,
    max_cut_iterations: int = 10,
    debug: bool = False,
    cut_strategy: str = "fewest_neighbors",
    use_tsim_bss: bool = True,
) -> list[BaseGraph]:
    """Decompose a ZX-graph using cutting first, then BSS, with like-term collection.

    This is an optimised version of find_stab_cutting that:
    1. Groups identical graph structures together
    2. Combines their scalars to reduce the total number of terms

    Args:
        graph: The ZX graph to decompose.
        max_cut_iterations: Maximum cutting iterations.
        debug: If True, print decomposition progress.
        cut_strategy: Strategy for selecting vertices to cut.
        use_tsim_bss: If True, use tsim's find_stab for final BSS.

    Returns:
        A list of Clifford graphs (T-count=0) whose sum equals the original.
    """
    # Step 1: Apply cutting decomposition
    cut_terms = stab_rank_decompose(
        graph,
        debug=debug,
        use_bss_fallback=False,  # Don't use internal BSS
        max_iterations=max_cut_iterations,
        param_safe=True,
        cut_strategy=cut_strategy,
        use_tsim_bss=False,
    )

    if debug:
        print(f"Cutting produced {len(cut_terms)} terms")

    # Step 2: For each cut term, full_reduce, skip zero scalars, then find_stab (BSS)
    all_clifford_terms: list[BaseGraph] = []

    for i, term in enumerate(cut_terms):
        # Full reduce with param safety
        zx.full_reduce(term, paramSafe=True)

        # Skip terms with zero scalar (like tsim's _decompose does)
        if term.scalar.is_zero:
            if debug:
                print(f"  Cut term {i}: scalar is zero, skipping")
            if len(all_clifford_terms) > 0:
                continue
            # Keep at least one term to avoid empty results

        tc = zx.simplify.tcount(term)

        if tc == 0:
            # Already Clifford
            all_clifford_terms.append(term)
            if debug:
                print(f"  Cut term {i}: already Clifford (T-count=0)")
        else:
            # Apply BSS to finish decomposition
            if use_tsim_bss:
                clifford_terms = find_stab(term)
            else:
                clifford_terms = zx.simulate.find_stabilizer_decomp(term)
            all_clifford_terms.extend(clifford_terms)

            if debug:
                print(f"  Cut term {i}: T-count={tc} -> {len(clifford_terms)} Clifford terms")

    if debug:
        print(f"Total Clifford terms before dedup: {len(all_clifford_terms)}")

    # Step 3: Group like terms by graph structure
    from collections import defaultdict

    term_groups: dict[tuple, list[BaseGraph]] = defaultdict(list)

    for term in all_clifford_terms:
        try:
            sig = _graph_signature(term)
            term_groups[sig].append(term)
        except Exception as e:
            # If signature fails, keep term as-is
            if debug:
                print(f"  Warning: Could not compute signature: {e}")
            # Use id as unique key
            term_groups[(id(term),)].append(term)

    if debug:
        print(f"Grouped into {len(term_groups)} unique structures")
        group_sizes = [len(g) for g in term_groups.values()]
        if max(group_sizes) > 1:
            print(f"  Group sizes: min={min(group_sizes)}, max={max(group_sizes)}, "
                  f"avg={sum(group_sizes)/len(group_sizes):.1f}")

    # Step 4: Combine terms with same structure
    combined_terms: list[BaseGraph] = []

    for sig, terms in term_groups.items():
        if len(terms) == 1:
            # Single term, no combination needed
            combined_terms.append(terms[0])
        else:
            # Multiple terms with same structure - combine scalars
            try:
                base = deepcopy(terms[0])
                scalars = [t.scalar for t in terms]

                # Sum the scalars
                combined_scalar = scalars[0]
                for s in scalars[1:]:
                    combined_scalar = combined_scalar + s

                # Check if combined scalar is zero
                if combined_scalar.is_zero:
                    if debug:
                        print(f"  Combined {len(terms)} terms -> zero scalar, skipping")
                    continue

                base.scalar = combined_scalar
                combined_terms.append(base)

                if debug:
                    print(f"  Combined {len(terms)} like terms into 1")
            except Exception as e:
                # If combination fails, keep all original terms
                if debug:
                    print(f"  Warning: Could not combine terms: {e}")
                combined_terms.extend(terms)

    # Ensure we have at least one term
    if len(combined_terms) == 0 and len(all_clifford_terms) > 0:
        combined_terms.append(all_clifford_terms[0])

    if debug:
        print(f"Total Clifford terms after dedup: {len(combined_terms)}")
        reduction = len(all_clifford_terms) - len(combined_terms)
        if reduction > 0:
            print(f"  Reduced by {reduction} terms ({100*reduction/len(all_clifford_terms):.1f}%)")

    return combined_terms


# =============================================================================
# Modified compilation pipeline using cutting
# =============================================================================

def _get_f_indices(graph: BaseGraph) -> list[int]:
    """Extract numerically sorted list of f-parameter indices from the graph."""
    all_params = get_params(graph)
    f_indices = sorted([int(p[1:]) for p in all_params if p.startswith("f")])
    return f_indices


def _remove_phase_terms(graph: BaseGraph) -> None:
    """Remove phase terms from the graph."""
    graph.scalar.phasevars_halfpi = dict()
    graph.scalar.phasevars_pi_pair = []


def _plug_outputs(
    graph: BaseGraph,
    m_chars: list[str],
    outputs_to_plug: list[int],
) -> list[BaseGraph]:
    """Create graphs with specified numbers of outputs plugged."""
    graphs: list[BaseGraph] = []
    num_outputs = len(graph.outputs())

    for num_plugged in outputs_to_plug:
        g = deepcopy(graph)
        output_vertices = list(g.outputs())

        effect = "0" * num_plugged + "+" * (num_outputs - num_plugged)
        g.apply_effect(effect)
        for i, v in enumerate(output_vertices[:num_plugged]):
            g.set_phase(v, m_chars[i])

        g.scalar.add_power(num_outputs - num_plugged)
        graphs.append(g)

    return graphs


def _compile_component_cutting(
    component: ConnectedComponent,
    f_indices_global: list[int],
    mode: Literal["sequential", "joint"],
    max_cut_iterations: int = 10,
    debug: bool = False,
    optimise_like_terms: bool = False,
    cut_strategy: str = "fewest_neighbors",
    use_tsim_bss: bool = True,
) -> CompiledComponent:
    """Compile a single connected component using cutting-based decomposition.

    Args:
        component: The connected component to compile.
        f_indices_global: Global list of f-parameter indices.
        mode: Compilation mode ("sequential" or "joint").
        max_cut_iterations: Maximum cutting iterations.
        debug: If True, print debug info.
        optimise_like_terms: If True, use optimised cutting that combines like terms.
        cut_strategy: Strategy for selecting vertices to cut.
        use_tsim_bss: If True, use tsim's find_stab for final BSS.
    """
    graph = component.graph
    output_indices = component.output_indices
    num_component_outputs = len(graph.outputs())

    component_f_set = set(_get_f_indices(graph))
    f_selection = [i for i in f_indices_global if i in component_f_set]

    if mode == "sequential":
        outputs_to_plug = list(range(num_component_outputs + 1))
    else:
        outputs_to_plug = [0, num_component_outputs]

    compiled_graphs: list[CompiledScalarGraphs] = []
    component_m_chars = [f"m{i}" for i in output_indices]
    plugged_graphs = _plug_outputs(graph, component_m_chars, outputs_to_plug)

    power2_base: int | None = None

    # Select decomposition function
    decomp_func = find_stab_cutting_optimised if optimise_like_terms else find_stab_cutting

    for num_m_plugged, plugged_graph in zip(outputs_to_plug, plugged_graphs):
        g_copy = deepcopy(plugged_graph)
        zx.full_reduce(g_copy, paramSafe=True)
        g_copy.normalize()

        if power2_base is None:
            power2_base = g_copy.scalar.power2
        g_copy.scalar.add_power(-power2_base)

        _remove_phase_terms(g_copy)

        param_names = [f"f{i}" for i in f_selection]
        param_names += [f"m{output_indices[j]}" for j in range(num_m_plugged)]

        # Use cutting-based decomposition instead of find_stab
        g_list = decomp_func(
            g_copy,
            max_cut_iterations=max_cut_iterations,
            debug=debug,
            cut_strategy=cut_strategy,
            use_tsim_bss=use_tsim_bss,
        )

        if len(g_list) == 1:
            _remove_phase_terms(g_list[0])

        # Convert phasevars_pi → phasevars_pi_pair before compilation.
        # Cutting creates phasevars_pi terms that compile_scalar_graphs ignores.
        for t in g_list:
            if t.scalar.phasevars_pi:
                t.scalar.phasevars_pi_pair.append(
                    [set(t.scalar.phasevars_pi), {"1"}]
                )
                t.scalar.phasevars_pi = set()

        compiled = compile_scalar_graphs(g_list, param_names)
        compiled_graphs.append(compiled)

    return CompiledComponent(
        output_indices=tuple(output_indices),
        f_selection=jnp.array(f_selection, dtype=jnp.int32),
        compiled_scalar_graphs=tuple(compiled_graphs),
    )


def compile_program_cutting(
    prepared: SamplingGraph,
    *,
    mode: Literal["sequential", "joint"],
    max_cut_iterations: int = 10,
    debug: bool = False,
    optimise_like_terms: bool = False,
    cut_strategy: str = "fewest_neighbors",
    use_tsim_bss: bool = True,
) -> CompiledProgram:
    """Compile a prepared graph using cutting-based decomposition.

    This is a drop-in replacement for tsim's compile_program that uses
    cutting before BSS.

    Args:
        prepared: The prepared sampling graph.
        mode: Compilation mode ("sequential" or "joint").
        max_cut_iterations: Maximum cutting iterations.
        debug: If True, print debug info.
        optimise_like_terms: If True, combine like terms after cutting to reduce
            the total number of stabiliser terms.
        cut_strategy: Strategy for selecting vertices to cut.
        use_tsim_bss: If True, use tsim's find_stab for final BSS.
    """
    components = connected_components(prepared.graph)
    f_indices_global = _get_f_indices(prepared.graph)
    num_outputs = prepared.num_outputs

    compiled_components: list[CompiledComponent] = []
    output_order: list[int] = []

    sorted_components = sorted(components, key=lambda c: len(c.output_indices))

    for component in sorted_components:
        compiled = _compile_component_cutting(
            component=component,
            f_indices_global=f_indices_global,
            mode=mode,
            max_cut_iterations=max_cut_iterations,
            debug=debug,
            optimise_like_terms=optimise_like_terms,
            cut_strategy=cut_strategy,
            use_tsim_bss=use_tsim_bss,
        )
        compiled_components.append(compiled)
        output_order.extend(component.output_indices)

    return CompiledProgram(
        components=tuple(compiled_components),
        output_order=jnp.array(output_order, dtype=jnp.int32),
        num_outputs=num_outputs,
        num_f_params=len(f_indices_global),
        num_detectors=prepared.num_detectors,
    )


# =============================================================================
# Cutting-based samplers
# =============================================================================

class _CuttingSamplerBase:
    """Base class for cutting-based samplers."""

    def __init__(
        self,
        circuit: Circuit,
        *,
        sample_detectors: bool,
        mode: Literal["sequential", "joint"],
        max_cut_iterations: int = 10,
        debug: bool = False,
        seed: int | None = None,
        optimise_like_terms: bool = False,
        cut_strategy: str = "fewest_neighbors",
        use_tsim_bss: bool = True,
    ):
        if seed is None:
            seed = int(np.random.default_rng().integers(0, 2**30))

        self._key = jax.random.key(seed)

        prepared = prepare_graph(circuit, sample_detectors=sample_detectors)

        # Use cutting-based compilation
        self._program = compile_program_cutting(
            prepared,
            mode=mode,
            max_cut_iterations=max_cut_iterations,
            debug=debug,
            optimise_like_terms=optimise_like_terms,
            cut_strategy=cut_strategy,
            use_tsim_bss=use_tsim_bss,
        )

        self._key, subkey = jax.random.split(self._key)
        channel_seed = int(jax.random.randint(subkey, (), 0, 2**30))
        self._channel_sampler = ChannelSampler(
            channel_probs=prepared.channel_probs,
            error_transform=prepared.error_transform,
            seed=channel_seed,
        )

        self.circuit = circuit
        self._num_detectors = prepared.num_detectors

    def _sample_batches(self, shots: int, batch_size: int | None = None) -> np.ndarray:
        """Sample in batches and concatenate results."""
        if batch_size is None:
            batch_size = shots

        batches: list[jax.Array] = []
        for _ in range(ceil(shots / batch_size)):
            f_params = self._channel_sampler.sample(batch_size)
            self._key, subkey = jax.random.split(self._key)
            samples = sample_program(self._program, f_params, subkey)
            batches.append(samples)

        return np.concatenate(batches)[:shots]

    def __repr__(self) -> str:
        """Return a string representation with compilation statistics."""
        c_graphs = []
        c_params = []
        c_a_terms = []
        c_b_terms = []
        c_c_terms = []
        c_d_terms = []
        num_circuits = 0
        total_memory_bytes = 0
        num_outputs = []

        for component in self._program.components:
            for circuit in component.compiled_scalar_graphs:
                num_outputs.append(len(component.output_indices))
                c_graphs.append(circuit.num_graphs)
                c_params.append(circuit.n_params)
                c_a_terms.append(circuit.a_const_phases.size)
                c_b_terms.append(circuit.b_term_types.size)
                c_c_terms.append(circuit.c_const_bits_a.size)
                c_d_terms.append(circuit.d_const_alpha.size + circuit.d_const_beta.size)
                num_circuits += 1

                total_memory_bytes += sum(
                    v.nbytes
                    for v in jax.tree_util.tree_leaves(circuit)
                    if isinstance(v, jax.Array)
                )

        def _format_bytes(n: int) -> str:
            if n < 1024:
                return f"{n} B"
            if n < 1024**2:
                return f"{n / 1024:.1f} kB"
            return f"{n / (1024**2):.1f} MB"

        total_memory_str = _format_bytes(total_memory_bytes)
        error_channel_bits = sum(
            channel.num_bits for channel in self._channel_sampler.channels
        )

        return (
            f"{type(self).__name__}[SIMPLE_CUT]({np.sum(c_graphs)} graphs, "
            f"{error_channel_bits} error channel bits, "
            f"{np.max(num_outputs) if num_outputs else 0} outputs for largest cc, "
            f"≤ {np.max(c_params) if c_params else 0} parameters, "
            f"{np.sum(c_a_terms)} A terms, {np.sum(c_b_terms)} B terms, "
            f"{np.sum(c_c_terms)} C terms, {np.sum(c_d_terms)} D terms, "
            f"{total_memory_str})"
        )


class CuttingMeasurementSampler(_CuttingSamplerBase):
    """Samples measurement outcomes using cutting-based decomposition.

    Drop-in replacement for tsim's CompiledMeasurementSampler.
    """

    def __init__(
        self,
        circuit: Circuit,
        *,
        max_cut_iterations: int = 10,
        debug: bool = False,
        seed: int | None = None,
        optimise_like_terms: bool = False,
        cut_strategy: str = "fewest_neighbors",
        use_tsim_bss: bool = True,
    ):
        super().__init__(
            circuit,
            sample_detectors=False,
            mode="sequential",
            max_cut_iterations=max_cut_iterations,
            debug=debug,
            seed=seed,
            optimise_like_terms=optimise_like_terms,
            cut_strategy=cut_strategy,
            use_tsim_bss=use_tsim_bss,
        )

    def sample(self, shots: int, *, batch_size: int = 1024) -> np.ndarray:
        """Sample measurement outcomes from the circuit."""
        return self._sample_batches(shots, batch_size)


def _maybe_bit_pack(array: np.ndarray, *, bit_packed: bool) -> np.ndarray:
    """Optionally bit-pack a boolean array."""
    if not bit_packed:
        return array
    return np.packbits(array.astype(np.bool_), axis=1, bitorder="little")


class CuttingDetectorSampler(_CuttingSamplerBase):
    """Samples detector and observable outcomes using cutting-based decomposition.

    Drop-in replacement for tsim's CompiledDetectorSampler.
    """

    def __init__(
        self,
        circuit: Circuit,
        *,
        max_cut_iterations: int = 10,
        debug: bool = False,
        seed: int | None = None,
        optimise_like_terms: bool = False,
        cut_strategy: str = "fewest_neighbors",
        use_tsim_bss: bool = True,
    ):
        super().__init__(
            circuit,
            sample_detectors=True,
            mode="sequential",
            max_cut_iterations=max_cut_iterations,
            debug=debug,
            seed=seed,
            optimise_like_terms=optimise_like_terms,
            cut_strategy=cut_strategy,
            use_tsim_bss=use_tsim_bss,
        )

    @overload
    def sample(
        self,
        shots: int,
        *,
        batch_size: int | None = None,
        prepend_observables: bool = False,
        append_observables: bool = False,
        separate_observables: Literal[True],
        bit_packed: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]: ...

    @overload
    def sample(
        self,
        shots: int,
        *,
        batch_size: int | None = None,
        prepend_observables: bool = False,
        append_observables: bool = False,
        separate_observables: Literal[False] = False,
        bit_packed: bool = False,
    ) -> np.ndarray: ...

    def sample(
        self,
        shots: int,
        *,
        batch_size: int | None = None,
        prepend_observables: bool = False,
        append_observables: bool = False,
        separate_observables: bool = False,
        bit_packed: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Return detector samples from the circuit."""
        samples = self._sample_batches(shots, batch_size)

        if append_observables:
            return _maybe_bit_pack(samples, bit_packed=bit_packed)

        num_detectors = self._num_detectors
        det_samples = samples[:, :num_detectors]
        obs_samples = samples[:, num_detectors:]

        if prepend_observables:
            combined = np.concatenate([obs_samples, det_samples], axis=1)
            return _maybe_bit_pack(combined, bit_packed=bit_packed)
        if separate_observables:
            return (
                _maybe_bit_pack(det_samples, bit_packed=bit_packed),
                _maybe_bit_pack(obs_samples, bit_packed=bit_packed),
            )

        return _maybe_bit_pack(det_samples, bit_packed=bit_packed)


# =============================================================================
# Convenience functions for Circuit
# =============================================================================

def compile_sampler_cutting(
    circuit: Circuit,
    *,
    max_cut_iterations: int = 10,
    debug: bool = False,
    seed: int | None = None,
    optimise_like_terms: bool = False,
    cut_strategy: str = "fewest_neighbors",
    use_tsim_bss: bool = True,
) -> CuttingMeasurementSampler:
    """Compile circuit into a cutting-based measurement sampler.

    Drop-in replacement for circuit.compile_sampler().

    Args:
        circuit: The quantum circuit to compile.
        max_cut_iterations: Maximum cutting iterations.
        debug: If True, print debug info.
        seed: Random seed for sampling.
        optimise_like_terms: If True, combine like terms after cutting to reduce
            the total number of stabiliser terms.
        cut_strategy: Strategy for selecting vertices to cut.
            Options: "fewest_neighbors", "first", "z_only_first"
        use_tsim_bss: If True, use tsim's find_stab for final BSS.
    """
    return CuttingMeasurementSampler(
        circuit,
        max_cut_iterations=max_cut_iterations,
        debug=debug,
        seed=seed,
        optimise_like_terms=optimise_like_terms,
        cut_strategy=cut_strategy,
        use_tsim_bss=use_tsim_bss,
    )


def compile_detector_sampler_cutting(
    circuit: Circuit,
    *,
    max_cut_iterations: int = 10,
    debug: bool = False,
    seed: int | None = None,
    optimise_like_terms: bool = False,
    cut_strategy: str = "fewest_neighbors",
    use_tsim_bss: bool = True,
) -> CuttingDetectorSampler:
    """Compile circuit into a cutting-based detector sampler.

    Drop-in replacement for circuit.compile_detector_sampler().

    Args:
        circuit: The quantum circuit to compile.
        max_cut_iterations: Maximum cutting iterations.
        debug: If True, print debug info.
        seed: Random seed for sampling.
        optimise_like_terms: If True, combine like terms after cutting to reduce
            the total number of stabiliser terms.
        cut_strategy: Strategy for selecting vertices to cut.
            Options: "fewest_neighbors", "first", "z_only_first"
        use_tsim_bss: If True, use tsim's find_stab for final BSS.
    """
    return CuttingDetectorSampler(
        circuit,
        max_cut_iterations=max_cut_iterations,
        debug=debug,
        seed=seed,
        optimise_like_terms=optimise_like_terms,
        cut_strategy=cut_strategy,
        use_tsim_bss=use_tsim_bss,
    )


# =============================================================================
# SUBCOMPONENT COMPILATION
# =============================================================================

# =============================================================================
# Helper functions
# =============================================================================

def _find_zx_components(g: BaseGraph) -> list[set[int]]:
    """Find connected components of a ZX graph via BFS."""
    vertices = list(g.vertices())
    if not vertices:
        return []
    visited: set[int] = set()
    components: list[set[int]] = []
    for start in vertices:
        if start in visited:
            continue
        comp: set[int] = set()
        queue = [start]
        while queue:
            v = queue.pop()
            if v in visited:
                continue
            visited.add(v)
            comp.add(v)
            for n in g.neighbors(v):
                if n not in visited:
                    queue.append(n)
        components.append(comp)
    return components


def _extract_subgraph(
    g: BaseGraph, keep_vertices: set[int], reset_scalar: bool = False
) -> BaseGraph:
    """Extract subgraph with only the given vertices.

    Args:
        g: Source graph.
        keep_vertices: Set of vertex IDs to keep.
        reset_scalar: If True, reset the scalar to unity (for secondary sub-components).

    Note: Uses deepcopy instead of g.copy() because pyzx copy() can renumber vertices.
    """
    from copy import deepcopy
    from pyzx_param.graph.scalar import Scalar

    h = deepcopy(g)
    to_remove = [v for v in list(h.vertices()) if v not in keep_vertices]
    for v in to_remove:
        h.remove_vertex(v)
    if reset_scalar:
        h.scalar = Scalar()
    return h


def _get_f_indices(graph: BaseGraph) -> list[int]:
    """Extract sorted f-parameter indices from graph."""
    all_params = get_params(graph)
    return sorted([int(p[1:]) for p in all_params if p.startswith("f")])


def _remove_phase_terms(graph: BaseGraph) -> None:
    """Remove complex-valued phase variable terms from scalar.

    Only removes phasevars_halfpi (exp(iπj/2 * parity) terms) which should
    be zero for the doubled graph and would introduce complex values.

    Preserves phasevars_pi_pair ((-1)^(ψ*φ) terms) and phasevars_pi
    ((-1)^parity terms) which encode real ±1 sign factors. These are needed
    for the doubled graph to evaluate to non-negative probabilities; removing
    them breaks sign coherence and corrupts autoregressive sampling.
    """
    graph.scalar.phasevars_halfpi = dict()


def _plug_outputs(
    graph: BaseGraph,
    m_chars: list[str],
    outputs_to_plug: list[int],
) -> list[BaseGraph]:
    """Create graphs with specified numbers of outputs plugged."""
    graphs: list[BaseGraph] = []
    num_outputs = len(graph.outputs())

    for num_plugged in outputs_to_plug:
        g = deepcopy(graph)
        output_vertices = list(g.outputs())

        effect = "0" * num_plugged + "+" * (num_outputs - num_plugged)
        g.apply_effect(effect)
        for i, v in enumerate(output_vertices[:num_plugged]):
            g.set_phase(v, m_chars[i])

        g.scalar.add_power(num_outputs - num_plugged)
        graphs.append(g)

    return graphs


# =============================================================================
# Data structures
# =============================================================================

class SubcompComponentData(eqx.Module):
    """Component data supporting product evaluation at disconnected levels.

    For levels where the graph remains connected, standard autoregressive
    evaluation is used (via compiled_scalar_graphs). For the level where
    the graph disconnects, product evaluation multiplies independent
    sub-component evaluations.
    """

    output_indices: tuple[int, ...] = eqx.field(static=True)
    f_selection: jax.Array
    # Standard levels: compiled_scalar_graphs[0] = normalization,
    # compiled_scalar_graphs[1..N-1] = levels 1 to N-1 (or 1 to N if no product)
    compiled_scalar_graphs: tuple[CompiledScalarGraphs, ...]
    # Product-level data (for the level where the graph disconnects)
    has_product_level: bool = eqx.field(static=True)
    product_compiled_subcomps: tuple[CompiledScalarGraphs, ...]
    product_param_index_maps: tuple[jax.Array, ...]


@dataclass(frozen=True)
class SubcompCompiledProgram:
    """Compiled program with sub-component product optimization."""

    component_data: tuple[SubcompComponentData, ...]
    output_order: np.ndarray
    num_outputs: int
    num_f_params: int
    num_detectors: int


# =============================================================================
# Compilation
# =============================================================================

def _compile_component_subcomp(
    component: ConnectedComponent,
    f_indices_global: list[int],
    max_cut_iterations: int = 10,
    debug: bool = False,
    cut_strategy: str = "fewest_neighbors",
    use_tsim_bss: bool = True,
) -> SubcompComponentData:
    """Compile a component with product optimization for disconnected levels.

    At each plugging level, checks if the reduced graph disconnects into
    independent sub-components. If so, compiles each sub-component separately
    and stores product-level data for efficient evaluation.
    """
    graph = component.graph
    output_indices = component.output_indices
    num_component_outputs = len(graph.outputs())

    component_f_set = set(_get_f_indices(graph))
    f_selection = [i for i in f_indices_global if i in component_f_set]

    outputs_to_plug = list(range(num_component_outputs + 1))
    component_m_chars = [f"m{i}" for i in output_indices]
    plugged_graphs = _plug_outputs(graph, component_m_chars, outputs_to_plug)

    power2_base: int | None = None
    compiled_graphs: list[CompiledScalarGraphs] = []
    product_compiled: tuple[CompiledScalarGraphs, ...] = ()
    product_idx_maps: tuple[jax.Array, ...] = ()
    has_product = False

    for level_idx, (num_m_plugged, plugged_graph) in enumerate(
        zip(outputs_to_plug, plugged_graphs)
    ):
        g_copy = deepcopy(plugged_graph)
        zx.full_reduce(g_copy, paramSafe=True)
        g_copy.normalize()

        if power2_base is None:
            power2_base = g_copy.scalar.power2
        g_copy.scalar.add_power(-power2_base)

        _remove_phase_terms(g_copy)

        param_names = [f"f{i}" for i in f_selection]
        param_names += [f"m{output_indices[j]}" for j in range(num_m_plugged)]

        is_last_level = level_idx == len(outputs_to_plug) - 1

        # Check for disconnection at the last level
        if is_last_level and num_component_outputs > 0:
            zx_comps = _find_zx_components(g_copy)

            if len(zx_comps) >= 2:
                if debug:
                    print(
                        f"  Level {level_idx}: graph disconnects into "
                        f"{len(zx_comps)} sub-components"
                    )

                _product_compiled: list[CompiledScalarGraphs] = []
                _product_idx_maps: list[jax.Array] = []

                for ci, comp_verts in enumerate(zx_comps):
                    # First sub-component keeps the original scalar;
                    # subsequent ones get unit scalar
                    sub_g = _extract_subgraph(
                        g_copy, comp_verts, reset_scalar=(ci > 0)
                    )

                    # Find which params this sub-component uses
                    sub_params_set = set(get_params(sub_g))
                    sub_param_names = [p for p in param_names if p in sub_params_set]

                    # Decompose independently
                    sub_terms = find_stab_cutting(
                        sub_g,
                        max_cut_iterations=max_cut_iterations,
                        debug=debug,
                        cut_strategy=cut_strategy,
                        use_tsim_bss=use_tsim_bss,
                    )

                    if len(sub_terms) == 1:
                        _remove_phase_terms(sub_terms[0])

                    sub_compiled = compile_scalar_graphs(sub_terms, sub_param_names)
                    _product_compiled.append(sub_compiled)

                    # Build index map: sub_param_names -> full param_names
                    idx_map = np.array(
                        [param_names.index(p) for p in sub_param_names],
                        dtype=np.int32,
                    )
                    _product_idx_maps.append(jnp.array(idx_map, dtype=jnp.int32))

                    if debug:
                        n_verts = len(comp_verts)
                        n_terms = len(sub_terms)
                        n_d = (
                            sub_compiled.d_const_alpha.shape[1]
                            if sub_compiled.d_const_alpha.ndim >= 2
                            else 0
                        )
                        print(
                            f"    Sub-comp {ci}: {n_verts} verts, "
                            f"{n_terms} terms, {len(sub_param_names)} params, "
                            f"{n_d} D-terms"
                        )

                product_compiled = tuple(_product_compiled)
                product_idx_maps = tuple(_product_idx_maps)
                has_product = True
                continue  # Don't add to compiled_graphs for this level

        # Standard decomposition for this level
        g_list = find_stab_cutting(
            g_copy,
            max_cut_iterations=max_cut_iterations,
            debug=debug,
            cut_strategy=cut_strategy,
            use_tsim_bss=use_tsim_bss,
        )

        if len(g_list) == 1:
            _remove_phase_terms(g_list[0])

        compiled = compile_scalar_graphs(g_list, param_names)
        compiled_graphs.append(compiled)

    return SubcompComponentData(
        output_indices=tuple(output_indices),
        f_selection=jnp.array(f_selection, dtype=jnp.int32),
        compiled_scalar_graphs=tuple(compiled_graphs),
        has_product_level=has_product,
        product_compiled_subcomps=product_compiled,
        product_param_index_maps=product_idx_maps,
    )


def compile_program_subcomp(
    prepared: SamplingGraph,
    *,
    max_cut_iterations: int = 10,
    debug: bool = False,
    cut_strategy: str = "fewest_neighbors",
    use_tsim_bss: bool = True,
) -> SubcompCompiledProgram:
    """Compile a prepared graph using sub-component product optimization."""
    components = connected_components(prepared.graph)
    sorted_components = sorted(components, key=lambda c: len(c.output_indices))

    f_indices_global = _get_f_indices(prepared.graph)

    compiled_data: list[SubcompComponentData] = []
    for cc in sorted_components:
        data = _compile_component_subcomp(
            cc,
            f_indices_global,
            max_cut_iterations=max_cut_iterations,
            debug=debug,
            cut_strategy=cut_strategy,
            use_tsim_bss=use_tsim_bss,
        )
        compiled_data.append(data)

    output_order = np.concatenate(
        [np.array(d.output_indices) for d in compiled_data]
    )

    return SubcompCompiledProgram(
        component_data=tuple(compiled_data),
        output_order=output_order,
        num_outputs=prepared.num_outputs,
        num_f_params=len(f_indices_global),
        num_detectors=prepared.num_detectors,
    )


# =============================================================================
# Sampling
# =============================================================================

def _sample_component_subcomp(
    data: SubcompComponentData,
    f_params: jax.Array,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Sample from component with product evaluation support.

    For standard levels, uses the same autoregressive scheme as tsim.
    For the product level (last level when graph disconnects), evaluates
    each sub-component independently and multiplies the results.
    """
    batch_size = f_params.shape[0]

    num_standard = len(data.compiled_scalar_graphs)
    num_outputs = num_standard - 1 + (1 if data.has_product_level else 0)

    f_selected = f_params[:, data.f_selection].astype(jnp.bool_)
    m_accumulated = jnp.zeros((batch_size, num_outputs), dtype=jnp.bool_)

    # Normalization: level 0 (no outputs plugged)
    prev = jnp.abs(evaluate_batch(data.compiled_scalar_graphs[0], f_selected))
    ones = jnp.ones((batch_size, 1), dtype=jnp.bool_)

    # Standard autoregressive for levels in compiled_scalar_graphs[1:]
    for i, circuit in enumerate(data.compiled_scalar_graphs[1:]):
        params = jnp.hstack([f_selected, m_accumulated[:, :i], ones])
        p1 = jnp.abs(evaluate_batch(circuit, params))
        key, subkey = jax.random.split(key)
        bits = jax.random.bernoulli(subkey, p=p1 / prev)
        m_accumulated = m_accumulated.at[:, i].set(bits)
        prev = jnp.where(bits, p1, prev - p1)

    # Product level (last output bit, if graph disconnects)
    if data.has_product_level:
        i_last = num_outputs - 1
        params_full = jnp.hstack(
            [f_selected, m_accumulated[:, :i_last], ones]
        )

        # Multiply evaluations from independent sub-components
        val = None
        for csg, idx_map in zip(
            data.product_compiled_subcomps, data.product_param_index_maps
        ):
            sub_params = params_full[:, idx_map]
            sub_val = evaluate_batch(csg, sub_params)
            val = sub_val if val is None else val * sub_val

        p1 = jnp.abs(val)
        key, subkey = jax.random.split(key)
        bits = jax.random.bernoulli(subkey, p=p1 / prev)
        m_accumulated = m_accumulated.at[:, i_last].set(bits)

    return m_accumulated, key


@jax.jit
def _sample_component_subcomp_jit(
    data: SubcompComponentData,
    f_params: jax.Array,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """JIT-compiled version of _sample_component_subcomp."""
    return _sample_component_subcomp(data, f_params, key)


def sample_component_subcomp(
    data: SubcompComponentData,
    f_params: jax.Array,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Sample from component, using JIT for components with >1 output."""
    if len(data.output_indices) <= 1:
        return _sample_component_subcomp(data, f_params, key)
    return _sample_component_subcomp_jit(data, f_params, key)


def sample_program_subcomp(
    program: SubcompCompiledProgram,
    f_params: jax.Array,
    key: jax.Array,
) -> jax.Array:
    """Sample all outputs from a compiled program."""
    results: list[jax.Array] = []

    for data in program.component_data:
        samples, key = sample_component_subcomp(data, f_params, key)
        results.append(samples)

    combined = jnp.concatenate(results, axis=1)
    return combined[:, jnp.argsort(jnp.array(program.output_order))]


# =============================================================================
# Sampler classes
# =============================================================================

class _SubcompSamplerBase:
    """Base class for sub-component product samplers."""

    def __init__(
        self,
        circuit: Circuit,
        *,
        sample_detectors: bool,
        max_cut_iterations: int = 10,
        debug: bool = False,
        seed: int | None = None,
        cut_strategy: str = "fewest_neighbors",
        use_tsim_bss: bool = True,
    ):
        if seed is None:
            seed = int(np.random.default_rng().integers(0, 2**30))

        self._key = jax.random.key(seed)

        prepared = prepare_graph(circuit, sample_detectors=sample_detectors)

        self._program = compile_program_subcomp(
            prepared,
            max_cut_iterations=max_cut_iterations,
            debug=debug,
            cut_strategy=cut_strategy,
            use_tsim_bss=use_tsim_bss,
        )

        self._key, subkey = jax.random.split(self._key)
        channel_seed = int(jax.random.randint(subkey, (), 0, 2**30))
        self._channel_sampler = ChannelSampler(
            channel_probs=prepared.channel_probs,
            error_transform=prepared.error_transform,
            seed=channel_seed,
        )

        self.circuit = circuit
        self._num_detectors = prepared.num_detectors

    def _sample_batches(
        self, shots: int, batch_size: int | None = None
    ) -> np.ndarray:
        """Sample in batches and concatenate results."""
        if batch_size is None:
            batch_size = shots

        batches: list[jax.Array] = []
        for _ in range(ceil(shots / batch_size)):
            f_params = self._channel_sampler.sample(batch_size)
            self._key, subkey = jax.random.split(self._key)
            samples = sample_program_subcomp(self._program, f_params, subkey)
            batches.append(samples)

        return np.concatenate(batches)[:shots]

    def __repr__(self) -> str:
        """Return string representation with compilation statistics."""
        total_graphs = 0
        total_a_terms = 0
        total_b_terms = 0
        total_c_terms = 0
        total_d_terms = 0
        total_memory_bytes = 0
        max_params = 0
        max_outputs = 0
        product_info_strs: list[str] = []

        for data in self._program.component_data:
            max_outputs = max(max_outputs, len(data.output_indices))

            for csg in data.compiled_scalar_graphs:
                total_graphs += csg.num_graphs
                max_params = max(max_params, csg.n_params)
                total_a_terms += csg.a_const_phases.size
                total_b_terms += csg.b_term_types.size
                total_c_terms += csg.c_const_bits_a.size
                total_d_terms += csg.d_const_alpha.size + csg.d_const_beta.size
                total_memory_bytes += sum(
                    v.nbytes
                    for v in jax.tree_util.tree_leaves(csg)
                    if isinstance(v, jax.Array)
                )

            if data.has_product_level:
                for ci, pcsg in enumerate(data.product_compiled_subcomps):
                    total_graphs += pcsg.num_graphs
                    max_params = max(max_params, pcsg.n_params)
                    d = pcsg.d_const_alpha.size + pcsg.d_const_beta.size
                    total_d_terms += d
                    total_a_terms += pcsg.a_const_phases.size
                    total_b_terms += pcsg.b_term_types.size
                    total_c_terms += pcsg.c_const_bits_a.size
                    total_memory_bytes += sum(
                        v.nbytes
                        for v in jax.tree_util.tree_leaves(pcsg)
                        if isinstance(v, jax.Array)
                    )
                    product_info_strs.append(
                        f"sub{ci}:{pcsg.num_graphs}g/{d}D"
                    )

        def _format_bytes(n: int) -> str:
            if n < 1024:
                return f"{n} B"
            if n < 1024**2:
                return f"{n / 1024:.1f} kB"
            return f"{n / (1024**2):.1f} MB"

        error_channel_bits = sum(
            channel.num_bits for channel in self._channel_sampler.channels
        )
        product_str = (
            f", product=[{', '.join(product_info_strs)}]"
            if product_info_strs
            else ""
        )

        return (
            f"{type(self).__name__}[SUBCOMP]("
            f"{total_graphs} graphs, "
            f"{error_channel_bits} error channel bits, "
            f"{max_outputs} outputs for largest cc, "
            f"≤ {max_params} parameters, "
            f"{total_a_terms} A terms, {total_b_terms} B terms, "
            f"{total_c_terms} C terms, {total_d_terms} D terms, "
            f"{_format_bytes(total_memory_bytes)}{product_str})"
        )


def _maybe_bit_pack(array: np.ndarray, *, bit_packed: bool) -> np.ndarray:
    """Optionally bit-pack a boolean array."""
    if not bit_packed:
        return array
    return np.packbits(array.astype(np.bool_), axis=1, bitorder="little")


class SubcompDetectorSampler(_SubcompSamplerBase):
    """Detector sampler with sub-component product optimization.

    Drop-in replacement for CuttingDetectorSampler that exploits natural
    graph disconnections at the fully-plugged level to reduce evaluation cost.
    """

    def __init__(
        self,
        circuit: Circuit,
        *,
        max_cut_iterations: int = 10,
        debug: bool = False,
        seed: int | None = None,
        cut_strategy: str = "fewest_neighbors",
        use_tsim_bss: bool = True,
    ):
        super().__init__(
            circuit,
            sample_detectors=True,
            max_cut_iterations=max_cut_iterations,
            debug=debug,
            seed=seed,
            cut_strategy=cut_strategy,
            use_tsim_bss=use_tsim_bss,
        )

    @overload
    def sample(
        self,
        shots: int,
        *,
        batch_size: int | None = None,
        prepend_observables: bool = False,
        append_observables: bool = False,
        separate_observables: Literal[True],
        bit_packed: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]: ...

    @overload
    def sample(
        self,
        shots: int,
        *,
        batch_size: int | None = None,
        prepend_observables: bool = False,
        append_observables: bool = False,
        separate_observables: Literal[False] = False,
        bit_packed: bool = False,
    ) -> np.ndarray: ...

    def sample(
        self,
        shots: int,
        *,
        batch_size: int | None = None,
        prepend_observables: bool = False,
        append_observables: bool = False,
        separate_observables: bool = False,
        bit_packed: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Sample detector and observable outcomes.

        Args:
            shots: Number of samples.
            batch_size: Batch size for sampling.
            prepend_observables: Prepend observable columns.
            append_observables: Append observable columns.
            separate_observables: Return (detectors, observables) tuple.
            bit_packed: Bit-pack the output arrays.
        """
        raw = self._sample_batches(shots, batch_size)

        d = self._num_detectors
        det = raw[:, :d]
        obs = raw[:, d:]

        det_out = _maybe_bit_pack(det, bit_packed=bit_packed)
        obs_out = _maybe_bit_pack(obs, bit_packed=bit_packed)

        if separate_observables:
            return det_out, obs_out
        if prepend_observables:
            return np.hstack([obs_out, det_out])
        if append_observables:
            return np.hstack([det_out, obs_out])
        return det_out


# =============================================================================
# Convenience function
# =============================================================================

def compile_detector_sampler_subcomp(
    circuit: Circuit,
    *,
    seed: int | None = None,
    max_cut_iterations: int = 10,
    debug: bool = False,
    cut_strategy: str = "fewest_neighbors",
    use_tsim_bss: bool = True,
) -> SubcompDetectorSampler:
    """Create a detector sampler with sub-component product optimization.

    This is a drop-in replacement for compile_detector_sampler_cutting that
    exploits natural graph disconnections to reduce the evaluation cost of
    the most expensive autoregressive level.
    """
    return SubcompDetectorSampler(
        circuit,
        max_cut_iterations=max_cut_iterations,
        debug=debug,
        seed=seed,
        cut_strategy=cut_strategy,
        use_tsim_bss=use_tsim_bss,
    )


# =============================================================================
# ENUMERATION-BASED SAMPLING
# =============================================================================


# =============================================================================
# Data structures
# =============================================================================

class SubcompEnumComponentData(eqx.Module):
    """Component compiled for joint enumeration with product evaluation.

    The fully-plugged ZX graph disconnects into sub-components. We enumerate
    all 2^N output combos, evaluate each sub-component independently, and
    multiply magnitudes to get the joint probability distribution.

    This eliminates the autoregressive chain entirely — only the fully-plugged
    level is compiled. No intermediate levels needed.
    """

    output_indices: tuple[int, ...] = eqx.field(static=True)
    f_selection: jax.Array  # indices into global f-params

    # Per sub-component: compiled graphs + param index maps
    subcomp_compiled: tuple[CompiledScalarGraphs, ...]
    subcomp_param_index_maps: tuple[jax.Array, ...]

    # Single combo table for ALL outputs: shape (2^num_outputs, num_outputs)
    m_combos: jax.Array

    # Number of sub-components and outputs (static for JIT)
    num_subcomps: int = eqx.field(static=True)
    num_component_outputs: int = eqx.field(static=True)


@dataclass(frozen=True)
class SubcompEnumCompiledProgram:
    """Compiled program with enum-based product optimization."""

    component_data: tuple  # tuple of SubcompEnumComponentData | SubcompComponentData
    output_order: np.ndarray
    num_outputs: int
    num_f_params: int
    num_detectors: int


# =============================================================================
# Compilation
# =============================================================================

MAX_ENUM_OUTPUTS = 8  # Max outputs for enumeration (2^8 = 256 combos)


def _compile_component_enum(
    component: ConnectedComponent,
    f_indices_global: list[int],
    max_cut_iterations: int = 10,
    debug: bool = False,
    cut_strategy: str = "fewest_neighbors",
    use_tsim_bss: bool = True,
) -> SubcompEnumComponentData | None:
    """Try to compile a component in enum mode.

    Returns None if not eligible (graph doesn't disconnect or too many outputs).
    """
    graph = component.graph
    output_indices = component.output_indices
    num_outputs = len(graph.outputs())

    if num_outputs == 0:
        return None  # Nothing to enumerate

    if num_outputs > MAX_ENUM_OUTPUTS:
        if debug:
            print(f"  Enum: too many outputs ({num_outputs} > {MAX_ENUM_OUTPUTS}), "
                  f"falling back")
        return None

    # Determine which f-params this component uses
    component_f_set = set(_get_f_indices(graph))
    f_selection = [i for i in f_indices_global if i in component_f_set]

    component_m_chars = [f"m{i}" for i in output_indices]

    # Plug level-0 (no outputs plugged) AND fully-plugged level
    plugged_graphs = _plug_outputs(graph, component_m_chars, [0, num_outputs])

    # Level-0: compute power2_base for normalization (matches subcomp sampler)
    g_level0 = deepcopy(plugged_graphs[0])
    zx.full_reduce(g_level0, paramSafe=True)
    g_level0.normalize()
    power2_base = g_level0.scalar.power2

    # Fully-plugged level with power2_base normalization
    g_plugged = plugged_graphs[1]
    # deepcopy is critical: g.copy() renumbers vertex IDs in pyzx_param!
    g_copy = deepcopy(g_plugged)
    zx.full_reduce(g_copy, paramSafe=True)
    g_copy.normalize()
    g_copy.scalar.add_power(-power2_base)
    _remove_phase_terms(g_copy)
    # phasevars_pi_pair and phasevars_pi are now preserved by _remove_phase_terms.
    # They encode (-1)^(f*m) cross-product signs needed for correct evaluation.
    # phasenodevars (legless spiders) contribute |1+e^(iπα)| which varies
    # with params — must NOT be cleared or it changes the probability distribution.

    # Check for disconnection (required for product evaluation)
    zx_comps = _find_zx_components(g_copy)
    if len(zx_comps) < 2:
        if debug:
            print("  Enum: graph does not disconnect, falling back")
        return None

    # Full param names for the fully-plugged level
    param_names = [f"f{i}" for i in f_selection]
    param_names += [f"m{output_indices[j]}" for j in range(num_outputs)]

    if debug:
        print(f"  Enum: graph disconnects into {len(zx_comps)} sub-components, "
              f"enumerating {2**num_outputs} combos over {num_outputs} outputs")

    # Compile each sub-component
    _compiled: list[CompiledScalarGraphs] = []
    _idx_maps: list[jax.Array] = []

    for ci, comp_verts in enumerate(zx_comps):
        # First sub-component keeps original scalar; others get unit scalar
        sub_g = _extract_subgraph(g_copy, comp_verts, reset_scalar=(ci > 0))

        # Get ALL params (vertex + scalar) for correct evaluation.
        # The scalar may reference cross-component m-params (e.g. phasenodevars),
        # but that's fine — each combo provides all m-values via the full param array.
        sub_params_set = set(get_params(sub_g))
        sub_param_names = [p for p in param_names if p in sub_params_set]

        sub_terms = find_stab_cutting(
            sub_g,
            max_cut_iterations=max_cut_iterations,
            debug=debug,
            cut_strategy=cut_strategy,
            use_tsim_bss=use_tsim_bss,
        )
        if len(sub_terms) == 1:
            _remove_phase_terms(sub_terms[0])

        # Convert phasevars_pi → phasevars_pi_pair before compilation.
        # Cutting creates phasevars_pi terms (via pivot rules after cut_spider),
        # but compile_scalar_graphs does NOT compile phasevars_pi — only
        # phasevars_pi_pair (Type C). Convert: (-1)^{XOR(S)} = (-1)^{XOR(S)·1}
        for t in sub_terms:
            if t.scalar.phasevars_pi:
                t.scalar.phasevars_pi_pair.append(
                    [set(t.scalar.phasevars_pi), {"1"}]
                )
                t.scalar.phasevars_pi = set()

        sub_compiled = compile_scalar_graphs(sub_terms, sub_param_names)
        _compiled.append(sub_compiled)

        # Index map: sub_param_names indices → full param_names indices
        idx_map = np.array(
            [param_names.index(p) for p in sub_param_names], dtype=np.int32
        )
        _idx_maps.append(jnp.array(idx_map, dtype=jnp.int32))

        if debug:
            n_d = sub_compiled.d_const_alpha.size + sub_compiled.d_const_beta.size
            print(f"    Sub-comp {ci}: {len(comp_verts)} verts, "
                  f"{len(sub_terms)} terms, {len(sub_param_names)} params, "
                  f"{n_d} D-terms")

    # Pre-compute ALL m-combos: shape (2^num_outputs, num_outputs)
    n_combos = 2 ** num_outputs
    m_combos = jnp.array(
        [[bool((i >> bit) & 1) for bit in range(num_outputs)]
         for i in range(n_combos)],
        dtype=jnp.bool_,
    )

    return SubcompEnumComponentData(
        output_indices=tuple(output_indices),
        f_selection=jnp.array(f_selection, dtype=jnp.int32),
        subcomp_compiled=tuple(_compiled),
        subcomp_param_index_maps=tuple(_idx_maps),
        m_combos=m_combos,
        num_subcomps=len(zx_comps),
        num_component_outputs=num_outputs,
    )


def compile_program_subcomp_enum(
    prepared: SamplingGraph,
    *,
    max_cut_iterations: int = 10,
    debug: bool = False,
    cut_strategy: str = "fewest_neighbors",
    use_tsim_bss: bool = True,
) -> SubcompEnumCompiledProgram:
    """Compile a prepared graph using enum-based product optimization.

    For components whose fully-plugged graph disconnects, uses joint enumeration
    with product evaluation. For all others, falls back to autoregressive subcomp.
    """
    components = connected_components(prepared.graph)
    sorted_components = sorted(components, key=lambda c: len(c.output_indices))

    f_indices_global = _get_f_indices(prepared.graph)

    compiled_data = []
    for cc in sorted_components:
        # Try enum mode first
        data = _compile_component_enum(
            cc,
            f_indices_global,
            max_cut_iterations=max_cut_iterations,
            debug=debug,
            cut_strategy=cut_strategy,
            use_tsim_bss=use_tsim_bss,
        )

        if data is None:
            # Fall back to autoregressive subcomp
            data = _compile_component_subcomp(
                cc,
                f_indices_global,
                max_cut_iterations=max_cut_iterations,
                debug=debug,
                cut_strategy=cut_strategy,
                use_tsim_bss=use_tsim_bss,
            )

        compiled_data.append(data)

    output_order = np.concatenate(
        [np.array(d.output_indices) for d in compiled_data]
    )

    return SubcompEnumCompiledProgram(
        component_data=tuple(compiled_data),
        output_order=output_order,
        num_outputs=prepared.num_outputs,
        num_f_params=len(f_indices_global),
        num_detectors=prepared.num_detectors,
    )


# =============================================================================
# Sampling
# =============================================================================

def _sample_component_enum(
    data: SubcompEnumComponentData,
    f_params: jax.Array,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Sample from component using joint enumeration with product evaluation.

    For each of 2^N output combos:
      1. Build param array with f-params + m-combo bits
      2. Evaluate each sub-component independently via evaluate_batch
      3. Multiply magnitudes to get joint |scalar| = |LEFT| x |RIGHT| x ...
      4. Sample from the joint categorical distribution
      5. Decode sampled index back to m-param bits
    """
    batch_size = f_params.shape[0]
    num_outputs = data.num_component_outputs
    m_combos = data.m_combos  # (n_combos, num_outputs)
    n_combos = m_combos.shape[0]  # 2^num_outputs

    f_selected = f_params[:, data.f_selection].astype(jnp.bool_)
    n_f = f_selected.shape[1]

    # Build full param array for all (shot, combo) pairs
    # Template: [f_selected, zeros_for_m_params] — shape (batch, n_f + num_outputs)
    full_template = jnp.hstack([
        f_selected,
        jnp.zeros((batch_size, num_outputs), dtype=jnp.bool_),
    ])

    # Expand for all combos: (batch * n_combos, n_f + num_outputs)
    full_expanded = jnp.repeat(full_template, n_combos, axis=0)

    # Tile m-combos for each shot: (batch * n_combos, num_outputs)
    m_tiled = jnp.tile(m_combos, (batch_size, 1))

    # Fill ALL m-positions at once
    full_expanded = full_expanded.at[:, n_f:].set(m_tiled)

    # Evaluate each sub-component and accumulate magnitudes via product
    # Initialize joint magnitudes to 1.0
    joint_magnitudes = jnp.ones(batch_size * n_combos)

    for k in range(data.num_subcomps):
        csg = data.subcomp_compiled[k]
        idx_map = data.subcomp_param_index_maps[k]

        # Select sub-component params and evaluate
        sub_params = full_expanded[:, idx_map]
        vals = evaluate_batch(csg, sub_params)  # (batch * n_combos,) complex

        # Multiply magnitudes (product of |LEFT| * |RIGHT| * ...)
        joint_magnitudes = joint_magnitudes * jnp.abs(vals)

    # Reshape to (batch, n_combos) for categorical sampling
    probs = joint_magnitudes.reshape(batch_size, n_combos)

    # Safety net: when all magnitudes are 0 (e.g. the contribution cancels for
    # every combo), default to combo 0 (all-zeros output) instead of sampling
    # uniformly at random.  This matches the subcomp sampler's autoregressive
    # behaviour where 0/0 → bernoulli(NaN) → False → all-zero outputs.
    row_sums = probs.sum(axis=1, keepdims=True)
    fallback = jnp.zeros_like(probs).at[:, 0].set(1.0)
    safe_probs = jnp.where(row_sums > 0, probs, fallback)

    # Categorical sample from joint distribution
    log_probs = jnp.log(safe_probs + _LOG_EPS)
    key, subkey = jax.random.split(key)
    chosen = jax.random.categorical(subkey, log_probs)  # (batch,)

    # Decode chosen index → output m-param bits
    sampled_m = m_combos[chosen]  # (batch, num_outputs)

    return sampled_m, key


@jax.jit
def _sample_component_enum_jit(
    data: SubcompEnumComponentData,
    f_params: jax.Array,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """JIT-compiled version of _sample_component_enum."""
    return _sample_component_enum(data, f_params, key)


def sample_component_enum(
    data: SubcompEnumComponentData,
    f_params: jax.Array,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Sample from enum component, using JIT for components with >1 output."""
    if len(data.output_indices) <= 1:
        return _sample_component_enum(data, f_params, key)
    return _sample_component_enum_jit(data, f_params, key)


def sample_program_subcomp_enum(
    program: SubcompEnumCompiledProgram,
    f_params: jax.Array,
    key: jax.Array,
) -> jax.Array:
    """Sample all outputs from a compiled program.

    Dispatches to enum sampling for SubcompEnumComponentData and
    autoregressive sampling for SubcompComponentData.
    """
    results: list[jax.Array] = []

    for data in program.component_data:
        if isinstance(data, SubcompEnumComponentData):
            samples, key = sample_component_enum(data, f_params, key)
        else:
            samples, key = sample_component_subcomp(data, f_params, key)
        results.append(samples)

    combined = jnp.concatenate(results, axis=1)
    return combined[:, jnp.argsort(jnp.array(program.output_order))]


# =============================================================================
# Sampler classes
# =============================================================================

class _SubcompEnumSamplerBase:
    """Base class for enum-based sub-component product samplers."""

    def __init__(
        self,
        circuit: Circuit,
        *,
        sample_detectors: bool,
        max_cut_iterations: int = 10,
        debug: bool = False,
        seed: int | None = None,
        cut_strategy: str = "fewest_neighbors",
        use_tsim_bss: bool = True,
    ):
        if seed is None:
            seed = int(np.random.default_rng().integers(0, 2**30))

        self._key = jax.random.key(seed)

        prepared = prepare_graph(circuit, sample_detectors=sample_detectors)

        self._program = compile_program_subcomp_enum(
            prepared,
            max_cut_iterations=max_cut_iterations,
            debug=debug,
            cut_strategy=cut_strategy,
            use_tsim_bss=use_tsim_bss,
        )

        self._key, subkey = jax.random.split(self._key)
        channel_seed = int(jax.random.randint(subkey, (), 0, 2**30))
        self._channel_sampler = ChannelSampler(
            channel_probs=prepared.channel_probs,
            error_transform=prepared.error_transform,
            seed=channel_seed,
        )

        self.circuit = circuit
        self._num_detectors = prepared.num_detectors

    def _sample_batches(
        self, shots: int, batch_size: int | None = None
    ) -> np.ndarray:
        """Sample in batches and concatenate results."""
        if batch_size is None:
            batch_size = shots

        batches: list[jax.Array] = []
        for _ in range(ceil(shots / batch_size)):
            f_params = self._channel_sampler.sample(batch_size)
            self._key, subkey = jax.random.split(self._key)
            samples = sample_program_subcomp_enum(self._program, f_params, subkey)
            batches.append(samples)

        return np.concatenate(batches)[:shots]

    def __repr__(self) -> str:
        """Return string representation with compilation statistics."""
        total_graphs = 0
        total_a_terms = 0
        total_b_terms = 0
        total_c_terms = 0
        total_d_terms = 0
        total_memory_bytes = 0
        max_params = 0
        max_outputs = 0
        enum_info_strs: list[str] = []
        product_info_strs: list[str] = []
        n_enum_components = 0
        n_autoregressive_components = 0

        for data in self._program.component_data:
            max_outputs = max(max_outputs, len(data.output_indices))

            if isinstance(data, SubcompEnumComponentData):
                n_enum_components += 1
                n_combos = data.m_combos.shape[0]
                for ci, pcsg in enumerate(data.subcomp_compiled):
                    total_graphs += pcsg.num_graphs
                    max_params = max(max_params, pcsg.n_params)
                    d = pcsg.d_const_alpha.size + pcsg.d_const_beta.size
                    total_d_terms += d
                    total_a_terms += pcsg.a_const_phases.size
                    total_b_terms += pcsg.b_term_types.size
                    total_c_terms += pcsg.c_const_bits_a.size
                    total_memory_bytes += sum(
                        v.nbytes
                        for v in jax.tree_util.tree_leaves(pcsg)
                        if isinstance(v, jax.Array)
                    )
                    enum_info_strs.append(
                        f"sub{ci}:{pcsg.num_graphs}g/{d}D"
                    )
                # Add combo count info
                enum_info_strs.append(f"{n_combos}combos")
            else:
                # SubcompComponentData (autoregressive fallback)
                n_autoregressive_components += 1
                for csg in data.compiled_scalar_graphs:
                    total_graphs += csg.num_graphs
                    max_params = max(max_params, csg.n_params)
                    total_a_terms += csg.a_const_phases.size
                    total_b_terms += csg.b_term_types.size
                    total_c_terms += csg.c_const_bits_a.size
                    total_d_terms += csg.d_const_alpha.size + csg.d_const_beta.size
                    total_memory_bytes += sum(
                        v.nbytes
                        for v in jax.tree_util.tree_leaves(csg)
                        if isinstance(v, jax.Array)
                    )

                if data.has_product_level:
                    for ci, pcsg in enumerate(data.product_compiled_subcomps):
                        total_graphs += pcsg.num_graphs
                        max_params = max(max_params, pcsg.n_params)
                        d = pcsg.d_const_alpha.size + pcsg.d_const_beta.size
                        total_d_terms += d
                        total_a_terms += pcsg.a_const_phases.size
                        total_b_terms += pcsg.b_term_types.size
                        total_c_terms += pcsg.c_const_bits_a.size
                        total_memory_bytes += sum(
                            v.nbytes
                            for v in jax.tree_util.tree_leaves(pcsg)
                            if isinstance(v, jax.Array)
                        )
                        product_info_strs.append(
                            f"sub{ci}:{pcsg.num_graphs}g/{d}D"
                        )

        def _format_bytes(n: int) -> str:
            if n < 1024:
                return f"{n} B"
            if n < 1024**2:
                return f"{n / 1024:.1f} kB"
            return f"{n / (1024**2):.1f} MB"

        error_channel_bits = sum(
            channel.num_bits for channel in self._channel_sampler.channels
        )

        enum_str = (
            f", enum=[{', '.join(enum_info_strs)}]"
            if enum_info_strs else ""
        )
        product_str = (
            f", product=[{', '.join(product_info_strs)}]"
            if product_info_strs else ""
        )

        return (
            f"{type(self).__name__}[SUBCOMP_ENUM]("
            f"{total_graphs} graphs, "
            f"{error_channel_bits} error channel bits, "
            f"{max_outputs} outputs for largest cc, "
            f"≤ {max_params} parameters, "
            f"{total_a_terms} A terms, {total_b_terms} B terms, "
            f"{total_c_terms} C terms, {total_d_terms} D terms, "
            f"{_format_bytes(total_memory_bytes)}{enum_str}{product_str})"
        )


def _maybe_bit_pack(array: np.ndarray, *, bit_packed: bool) -> np.ndarray:
    """Optionally bit-pack a boolean array."""
    if not bit_packed:
        return array
    return np.packbits(array.astype(np.bool_), axis=1, bitorder="little")


class SubcompEnumDetectorSampler(_SubcompEnumSamplerBase):
    """Detector sampler with enumeration-based product optimization.

    Eliminates autoregressive sampling by enumerating all output combos
    and evaluating via the product decomposition (LEFT x RIGHT).
    Only compiles the fully-plugged level — no intermediate levels needed.
    """

    def __init__(
        self,
        circuit: Circuit,
        *,
        max_cut_iterations: int = 10,
        debug: bool = False,
        seed: int | None = None,
        cut_strategy: str = "fewest_neighbors",
        use_tsim_bss: bool = True,
    ):
        super().__init__(
            circuit,
            sample_detectors=True,
            max_cut_iterations=max_cut_iterations,
            debug=debug,
            seed=seed,
            cut_strategy=cut_strategy,
            use_tsim_bss=use_tsim_bss,
        )

    @overload
    def sample(
        self,
        shots: int,
        *,
        batch_size: int | None = None,
        prepend_observables: bool = False,
        append_observables: bool = False,
        separate_observables: Literal[True],
        bit_packed: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]: ...

    @overload
    def sample(
        self,
        shots: int,
        *,
        batch_size: int | None = None,
        prepend_observables: bool = False,
        append_observables: bool = False,
        separate_observables: Literal[False] = False,
        bit_packed: bool = False,
    ) -> np.ndarray: ...

    def sample(
        self,
        shots: int,
        *,
        batch_size: int | None = None,
        prepend_observables: bool = False,
        append_observables: bool = False,
        separate_observables: bool = False,
        bit_packed: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Sample detector and observable outcomes."""
        raw = self._sample_batches(shots, batch_size)

        d = self._num_detectors
        det = raw[:, :d]
        obs = raw[:, d:]

        det_out = _maybe_bit_pack(det, bit_packed=bit_packed)
        obs_out = _maybe_bit_pack(obs, bit_packed=bit_packed)

        if separate_observables:
            return det_out, obs_out
        if prepend_observables:
            return np.hstack([obs_out, det_out])
        if append_observables:
            return np.hstack([det_out, obs_out])
        return det_out


# =============================================================================
# Convenience function
# =============================================================================

def compile_detector_sampler_subcomp_enum(
    circuit: Circuit,
    *,
    seed: int | None = None,
    max_cut_iterations: int = 10,
    debug: bool = False,
    cut_strategy: str = "fewest_neighbors",
    use_tsim_bss: bool = True,
) -> SubcompEnumDetectorSampler:
    """Create a detector sampler with enum-based product optimization.

    This sampler eliminates autoregressive sampling by enumerating all 2^N
    output combos and evaluating via the product decomposition. Only the
    fully-plugged level is compiled — no intermediate autoregressive levels.

    Compiled D-terms: ~1.4K (vs 40K subcomp, 83K baseline).
    Falls back to autoregressive for components that don't disconnect.
    """
    return SubcompEnumDetectorSampler(
        circuit,
        max_cut_iterations=max_cut_iterations,
        debug=debug,
        seed=seed,
        cut_strategy=cut_strategy,
        use_tsim_bss=use_tsim_bss,
    )


# =============================================================================
# GENERAL ENUM COMPILATION
# =============================================================================


# =============================================================================
# Compilation
# =============================================================================

MAX_ENUM_OUTPUTS_GENERAL = 12  # 2^12 = 4096 combos


def _compile_component_enum_general(
    component: ConnectedComponent,
    f_indices_global: list[int],
    max_cut_iterations: int = 10,
    debug: bool = False,
    cut_strategy: str = "fewest_neighbors",
    use_tsim_bss: bool = True,
    max_enum_outputs: int = MAX_ENUM_OUTPUTS_GENERAL,
) -> SubcompEnumComponentData | None:
    """Compile a multi-output component using enum — never falls back to autoregressive.

    Returns None for 0-1 output components (autoregressive is correct for these
    since the |a+b| != |a|+|b| bug only manifests with 2+ outputs).

    For non-disconnecting graphs: compiles the full graph as a single monolithic
    sub-component (num_subcomps=1). The existing _sample_component_enum handles
    this correctly — its product loop just runs once.

    Raises ValueError for components with more outputs than max_enum_outputs.
    """
    graph = component.graph
    output_indices = component.output_indices
    num_outputs = len(graph.outputs())

    if num_outputs <= 1:
        return None  # Safe for autoregressive — bug only triggers with 2+ outputs

    if num_outputs > max_enum_outputs:
        raise ValueError(
            f"Component has {num_outputs} outputs, exceeding max_enum_outputs="
            f"{max_enum_outputs}. Increase max_enum_outputs or reduce circuit size."
        )

    # Determine which f-params this component uses
    component_f_set = set(_get_f_indices(graph))
    f_selection = [i for i in f_indices_global if i in component_f_set]

    component_m_chars = [f"m{i}" for i in output_indices]

    # Plug level-0 (normalization) AND fully-plugged level
    plugged_graphs = _plug_outputs(graph, component_m_chars, [0, num_outputs])

    # Level-0: compute power2_base for normalization
    g_level0 = deepcopy(plugged_graphs[0])
    zx.full_reduce(g_level0, paramSafe=True)
    g_level0.normalize()
    power2_base = g_level0.scalar.power2

    # Fully-plugged level with normalization
    g_plugged = plugged_graphs[1]
    g_copy = deepcopy(g_plugged)
    zx.full_reduce(g_copy, paramSafe=True)
    g_copy.normalize()
    g_copy.scalar.add_power(-power2_base)
    _remove_phase_terms(g_copy)

    # Full param names
    param_names = [f"f{i}" for i in f_selection]
    param_names += [f"m{output_indices[j]}" for j in range(num_outputs)]

    # Check for disconnection
    zx_comps = _find_zx_components(g_copy)

    if len(zx_comps) >= 2:
        # === DISCONNECTING: product decomposition (same as old enum) ===
        if debug:
            print(f"  EnumGeneral: graph disconnects into {len(zx_comps)} "
                  f"sub-components, enumerating {2**num_outputs} combos "
                  f"over {num_outputs} outputs")

        _compiled = []
        _idx_maps = []

        for ci, comp_verts in enumerate(zx_comps):
            sub_g = _extract_subgraph(g_copy, comp_verts, reset_scalar=(ci > 0))
            sub_params_set = set(get_params(sub_g))
            sub_param_names = [p for p in param_names if p in sub_params_set]

            sub_terms = find_stab_cutting(
                sub_g,
                max_cut_iterations=max_cut_iterations,
                debug=debug,
                cut_strategy=cut_strategy,
                use_tsim_bss=use_tsim_bss,
            )
            if len(sub_terms) == 1:
                _remove_phase_terms(sub_terms[0])

            # Convert phasevars_pi → phasevars_pi_pair
            for t in sub_terms:
                if t.scalar.phasevars_pi:
                    t.scalar.phasevars_pi_pair.append(
                        [set(t.scalar.phasevars_pi), {"1"}]
                    )
                    t.scalar.phasevars_pi = set()

            sub_compiled = compile_scalar_graphs(sub_terms, sub_param_names)
            _compiled.append(sub_compiled)

            idx_map = np.array(
                [param_names.index(p) for p in sub_param_names], dtype=np.int32
            )
            _idx_maps.append(jnp.array(idx_map, dtype=jnp.int32))

            if debug:
                n_d = sub_compiled.d_const_alpha.size + sub_compiled.d_const_beta.size
                print(f"    Sub-comp {ci}: {len(comp_verts)} verts, "
                      f"{len(sub_terms)} terms, {len(sub_param_names)} params, "
                      f"{n_d} D-terms")
    else:
        # === NON-DISCONNECTING: monolithic evaluation ===
        if debug:
            print(f"  EnumGeneral: graph does NOT disconnect, using monolithic "
                  f"evaluation with {2**num_outputs} combos over {num_outputs} "
                  f"outputs")

        # Compile the full graph as a single "sub-component"
        sub_terms = find_stab_cutting(
            g_copy,
            max_cut_iterations=max_cut_iterations,
            debug=debug,
            cut_strategy=cut_strategy,
            use_tsim_bss=use_tsim_bss,
        )
        if len(sub_terms) == 1:
            _remove_phase_terms(sub_terms[0])

        # Convert phasevars_pi → phasevars_pi_pair
        for t in sub_terms:
            if t.scalar.phasevars_pi:
                t.scalar.phasevars_pi_pair.append(
                    [set(t.scalar.phasevars_pi), {"1"}]
                )
                t.scalar.phasevars_pi = set()

        full_compiled = compile_scalar_graphs(sub_terms, param_names)

        # Identity index map: all params map to themselves
        idx_map = jnp.arange(len(param_names), dtype=jnp.int32)

        _compiled = [full_compiled]
        _idx_maps = [idx_map]

        if debug:
            n_d = full_compiled.d_const_alpha.size + full_compiled.d_const_beta.size
            print(f"    Monolithic: {len(sub_terms)} terms, "
                  f"{len(param_names)} params, {n_d} D-terms")

    # Pre-compute ALL m-combos: shape (2^num_outputs, num_outputs)
    n_combos = 2 ** num_outputs
    m_combos = jnp.array(
        [[bool((i >> bit) & 1) for bit in range(num_outputs)]
         for i in range(n_combos)],
        dtype=jnp.bool_,
    )

    return SubcompEnumComponentData(
        output_indices=tuple(output_indices),
        f_selection=jnp.array(f_selection, dtype=jnp.int32),
        subcomp_compiled=tuple(_compiled),
        subcomp_param_index_maps=tuple(_idx_maps),
        m_combos=m_combos,
        num_subcomps=len(_compiled),
        num_component_outputs=num_outputs,
    )


def compile_program_subcomp_enum_general(
    prepared: SamplingGraph,
    *,
    max_cut_iterations: int = 10,
    debug: bool = False,
    cut_strategy: str = "fewest_neighbors",
    use_tsim_bss: bool = True,
    max_enum_outputs: int = MAX_ENUM_OUTPUTS_GENERAL,
) -> SubcompEnumCompiledProgram:
    """Compile a prepared graph using general enum — no autoregressive for multi-output.

    Falls back to autoregressive for 0-1 output components (safe — bug only with 2+).
    """
    components = connected_components(prepared.graph)
    sorted_components = sorted(components, key=lambda c: len(c.output_indices))

    f_indices_global = _get_f_indices(prepared.graph)

    compiled_data = []
    for cc in sorted_components:
        # Try general enum (handles disconnecting AND non-disconnecting)
        data = _compile_component_enum_general(
            cc,
            f_indices_global,
            max_cut_iterations=max_cut_iterations,
            debug=debug,
            cut_strategy=cut_strategy,
            use_tsim_bss=use_tsim_bss,
            max_enum_outputs=max_enum_outputs,
        )

        if data is None:
            # 0-1 output component — autoregressive is safe here
            data = _compile_component_subcomp(
                cc,
                f_indices_global,
                max_cut_iterations=max_cut_iterations,
                debug=debug,
                cut_strategy=cut_strategy,
                use_tsim_bss=use_tsim_bss,
            )

        compiled_data.append(data)

    output_order = np.concatenate(
        [np.array(d.output_indices) for d in compiled_data]
    )

    return SubcompEnumCompiledProgram(
        component_data=tuple(compiled_data),
        output_order=output_order,
        num_outputs=prepared.num_outputs,
        num_f_params=len(f_indices_global),
        num_detectors=prepared.num_detectors,
    )


# =============================================================================
# Sampler classes
# =============================================================================

class SubcompEnumGeneralDetectorSampler(_SubcompEnumSamplerBase):
    """Detector sampler with general enum — no autoregressive fallback.

    Uses enum categorical for ALL multi-output components:
    - Disconnecting graphs: product decomposition (same as old enum)
    - Non-disconnecting graphs: monolithic evaluation (num_subcomps=1)

    Falls back to autoregressive ONLY for 0-output components (safe).
    """

    def __init__(
        self,
        circuit: Circuit,
        *,
        max_cut_iterations: int = 10,
        debug: bool = False,
        seed: int | None = None,
        cut_strategy: str = "fewest_neighbors",
        use_tsim_bss: bool = True,
        max_enum_outputs: int = MAX_ENUM_OUTPUTS_GENERAL,
    ):
        if seed is None:
            seed = int(np.random.default_rng().integers(0, 2**30))

        self._key = jax.random.key(seed)

        prepared = prepare_graph(circuit, sample_detectors=True)

        self._program = compile_program_subcomp_enum_general(
            prepared,
            max_cut_iterations=max_cut_iterations,
            debug=debug,
            cut_strategy=cut_strategy,
            use_tsim_bss=use_tsim_bss,
            max_enum_outputs=max_enum_outputs,
        )

        self._key, subkey = jax.random.split(self._key)
        from tsim.noise.channels import ChannelSampler
        channel_seed = int(jax.random.randint(subkey, (), 0, 2**30))
        self._channel_sampler = ChannelSampler(
            channel_probs=prepared.channel_probs,
            error_transform=prepared.error_transform,
            seed=channel_seed,
        )

        self.circuit = circuit
        self._num_detectors = prepared.num_detectors

    def sample(
        self,
        shots: int,
        *,
        batch_size: int | None = None,
        prepend_observables: bool = False,
        append_observables: bool = False,
        separate_observables: bool = False,
        bit_packed: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Sample detector and observable outcomes."""
        raw = self._sample_batches(shots, batch_size)

        d = self._num_detectors
        det = raw[:, :d]
        obs = raw[:, d:]

        if bit_packed:
            det = np.packbits(det.astype(np.bool_), axis=1, bitorder="little")
            obs = np.packbits(obs.astype(np.bool_), axis=1, bitorder="little")

        if separate_observables:
            return det, obs
        if prepend_observables:
            return np.hstack([obs, det])
        if append_observables:
            return np.hstack([det, obs])
        return det


def compile_detector_sampler_subcomp_enum_general(
    circuit: Circuit,
    *,
    seed: int | None = None,
    max_cut_iterations: int = 10,
    debug: bool = False,
    cut_strategy: str = "fewest_neighbors",
    use_tsim_bss: bool = True,
    max_enum_outputs: int = MAX_ENUM_OUTPUTS_GENERAL,
) -> SubcompEnumGeneralDetectorSampler:
    """Create a detector sampler with general enum — no autoregressive fallback.

    Uses enum categorical for ALL multi-output components. Falls back to
    autoregressive ONLY for 0-output components (safe — no Bernoulli calls).
    """
    return SubcompEnumGeneralDetectorSampler(
        circuit,
        max_cut_iterations=max_cut_iterations,
        debug=debug,
        seed=seed,
        cut_strategy=cut_strategy,
        use_tsim_bss=use_tsim_bss,
        max_enum_outputs=max_enum_outputs,
    )
