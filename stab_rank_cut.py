"""
Stabiliser Rank Cutting

"""

import pyzx_param as zx
from pyzx_param.graph.base import BaseGraph
import numpy as np
from fractions import Fraction
from typing import List, Tuple, Optional
from copy import deepcopy  # Use deepcopy instead of .clone() to avoid shallow copy bugs with params

VertexId = int


# ============================================================================
# Basic utilities
# ============================================================================

def is_boundary(g: BaseGraph, v: VertexId) -> bool:
    """Check if vertex is a boundary (input/output)."""
    return g.type(v) == 0


def is_z_spider(g: BaseGraph, v: VertexId) -> bool:
    """Check if vertex is a Z-spider."""
    return g.type(v) == 1


def is_x_spider(g: BaseGraph, v: VertexId) -> bool:
    """Check if vertex is an X-spider."""
    return g.type(v) == 2


def is_t_like(phase) -> bool:
    """Check if phase is T-like (non-Clifford)."""
    if hasattr(phase, 'numerator'):
        return phase in (Fraction(1,4), Fraction(3,4), Fraction(5,4), Fraction(7,4))
    elif isinstance(phase, (int, float)):
        p = phase % 2
        return p in (0.25, 0.75, 1.25, 1.75)
    return False


def tcount(g: BaseGraph) -> int:
    """Count T-gates (non-Clifford phases) in graph."""
    count = 0
    for v in g.vertices():
        if is_t_like(g.phase(v)):
            count += 1
    return count


def can_cut(g: BaseGraph, v: VertexId) -> bool:
    """
    Check if a vertex can be safely cut.

    A vertex can be cut if:
    - It is not a boundary vertex
    - None of its neighbors are boundary vertices
    """
    if is_boundary(g, v):
        return False

    for n in g.neighbors(v):
        if is_boundary(g, n):
            return False

    return True


def safe_set_io(g: BaseGraph, inputs: List[VertexId], outputs: List[VertexId]):
    """Safely set I/O, filtering out vertices that no longer exist."""
    valid_verts = set(g.vertices())
    valid_inputs = [v for v in inputs if v in valid_verts]
    valid_outputs = [v for v in outputs if v in valid_verts]
    if valid_inputs:
        g.set_inputs(valid_inputs)
    if valid_outputs:
        g.set_outputs(valid_outputs)


# ============================================================================
# The correct cutting function
# ============================================================================

def cut_spider(g: BaseGraph, v: VertexId) -> Tuple[BaseGraph, BaseGraph]:
    """
    Apply the cutting rule to ANY Z or X spider.

    The cutting identity from ZX-calculus:
    Spider_α with n legs = (1/√2)^n × (left + e^{iπα} × right)

    Where:
    - Left: Remove spider, create n opposite-color spiders (phase 0)
    - Right: Remove spider, create n opposite-color spiders (phase π)

    Hadamard edge handling:
    - Simple edge → create opposite-color spider with simple edge
    - Hadamard edge → create SAME-color spider with simple edge
      (because X =H= Y is equivalent to Z -- Y)

    Args:
        g: The ZX-graph
        v: Vertex to cut (must be Z or X spider, not adjacent to boundary)

    Returns:
        Tuple of (left_graph, right_graph) whose sum equals the original
    """
    v_type = g.type(v)
    if v_type not in [1, 2]:
        raise ValueError(f"Vertex {v} is not a Z or X spider (type={v_type})")

    # Check not boundary-connected
    for n in g.neighbors(v):
        if g.type(n) == 0:
            raise ValueError(f"Vertex {v} is connected to boundary")

    v_phase = g.phase(v)
    v_params = g.get_params(v)
    neighbors = list(g.neighbors(v))
    n_neighbors = len(neighbors)

    # Record edge types BEFORE modification
    edge_types = {}
    for nb in neighbors:
        e = g.edge(v, nb)
        edge_types[nb] = g.edge_type(e)

    # Determine base spider type to create (opposite color)
    base_create_type = 2 if v_type == 1 else 1

    # Use deepcopy to preserve vertex IDs and avoid shallow copy bugs with params
    g_left = deepcopy(g)
    g_right = deepcopy(g)

    # Remove the cut vertex from both branches
    g_left.remove_vertex(v)
    g_right.remove_vertex(v)

    # Create new spiders connected to each neighbor
    for nb in neighbors:
        et = edge_types[nb]

        # Hadamard edge flips the created spider type
        if et == 2:  # Hadamard edge
            spider_type = v_type  # Same color as cut vertex
        else:  # Simple edge
            spider_type = base_create_type  # Opposite color

        new_edge_type = 1  # Always simple edge

        # Left branch: spider with phase 0
        v_left = g_left.add_vertex(
            spider_type,
            g.qubit(v),
            (g.row(v) + g.row(nb)) / 2,
            0
        )
        g_left.add_edge((v_left, nb), new_edge_type)

        # Right branch: spider with phase π
        v_right = g_right.add_vertex(
            spider_type,
            g.qubit(v),
            (g.row(v) + g.row(nb)) / 2,
            1
        )
        g_right.add_edge((v_right, nb), new_edge_type)

    # Scalar: (1/√2)^n for each branch
    g_left.scalar.add_power(-n_neighbors)
    g_right.scalar.add_power(-n_neighbors)

    # Right branch: additional factor e^{iπα} × (-1)^{XOR(params)}
    if isinstance(v_phase, Fraction):
        g_right.scalar.add_phase(v_phase)
    else:
        g_right.scalar.add_phase(Fraction(v_phase).limit_denominator(1000))
    if v_params:
        g_right.scalar.add_phase_vars_pi(set(v_params))

    return g_left, g_right


# ============================================================================
# Finding vertices to cut
# ============================================================================

def find_cuttable_t(g: BaseGraph) -> Optional[VertexId]:
    """Find a T-gate that can be cut (not boundary-connected)."""
    for v in g.vertices():
        if g.type(v) not in [1, 2]:
            continue
        if not is_t_like(g.phase(v)):
            continue
        if can_cut(g, v):
            return v
    return None


def find_best_cut(g: BaseGraph, strategy: str = "fewest_neighbors") -> Optional[VertexId]:
    """
    Find a vertex to cut based on the specified strategy.

    Args:
        g: The graph to search
        strategy: Selection strategy:
            - "fewest_neighbors": Prefer vertices with fewer neighbors (default)
            - "first": Return the first cuttable T-gate found
            - "z_only_first": Only Z-spiders, first found (matches investigation notebook)

    Returns:
        Vertex ID to cut, or None if no cuttable T-gate found
    """
    if strategy == "first":
        # Return first cuttable T-gate (Z or X)
        for v in g.vertices():
            if g.type(v) not in [1, 2]:
                continue
            if not is_t_like(g.phase(v)):
                continue
            if can_cut(g, v):
                return v
        return None

    elif strategy == "z_only_first":
        # Only Z-spiders, first found (matches investigation notebook exactly)
        for v in g.vertices():
            if g.type(v) != 1:  # Z-spider only
                continue
            if not is_t_like(g.phase(v)):
                continue
            if can_cut(g, v):
                return v
        return None

    else:  # "fewest_neighbors" (default)
        best_v = None
        best_score = float('inf')

        for v in g.vertices():
            if g.type(v) not in [1, 2]:
                continue
            if not is_t_like(g.phase(v)):
                continue
            if not can_cut(g, v):
                continue

            n_neighbors = len(list(g.neighbors(v)))
            if n_neighbors < best_score:
                best_score = n_neighbors
                best_v = v

        return best_v


# ============================================================================
# Main decomposition function
# ============================================================================

def decompose(
    g: BaseGraph,
    max_iterations: int = 50,
    use_bss_fallback: bool = True,
    param_safe: bool = True,
    debug: bool = False,
    reduce_intermediate: bool = True,
    cut_strategy: str = "fewest_neighbors",
    use_tsim_bss: bool = False,
    skip_initial_reduce: bool = False,
) -> List[BaseGraph]:
    """
    Decompose a ZX graph into stabiliser (Clifford) terms using cutting.

    Args:
        g: Input ZX graph
        max_iterations: Maximum cutting iterations
        use_bss_fallback: If True, use BSS when no cuttable T-gate found
        param_safe: If True, use paramSafe=True for full_reduce
        debug: If True, print debug info
        reduce_intermediate: If True, reduce after each cut (default).
                           If False, only reduce at the end.
        cut_strategy: How to select which T-gate to cut:
            - "fewest_neighbors": Prefer vertices with fewer neighbors (default)
            - "first": Return the first cuttable T-gate found
            - "z_only_first": Only Z-spiders, first found (matches investigation notebook)
        use_tsim_bss: If True, use BSS for remaining terms when max_iterations is reached
        skip_initial_reduce: If True, skip the initial full_reduce before cutting

    Returns:
        List of Clifford graphs whose sum equals the original
    """
    orig_inputs = list(g.inputs())
    orig_outputs = list(g.outputs())

    # Initial reduction of input graph (use deepcopy to avoid shallow copy bugs with params)
    g_init = deepcopy(g)
    if not skip_initial_reduce:
        zx.full_reduce(g_init, paramSafe=param_safe)

    if debug:
        print(f"Initial graph: T-count={tcount(g_init)}, scalar.is_zero={g_init.scalar.is_zero}")

    terms = [g_init]
    clifford_terms = []

    for iteration in range(max_iterations):
        new_terms = []

        for term in terms:
            # Optionally reduce the term (can cause scalar issues with disconnected components)
            if reduce_intermediate:
                zx.full_reduce(term, paramSafe=param_safe)

            # Check if scalar became zero (disconnected π-spider issue)
            if term.scalar.is_zero:
                if debug:
                    print(f"  Term scalar became zero after reduction, skipping")
                continue

            tc = tcount(term)

            if tc == 0:
                # Clifford - done with this term
                clifford_terms.append(term)
                continue

            # Find a T-gate to cut
            cut_v = find_best_cut(term, strategy=cut_strategy)

            if cut_v is None:
                # No cuttable T-gate
                if use_bss_fallback:
                    if debug:
                        print(f"  No cuttable T-gate (T-count={tc}), using BSS")
                    try:
                        bss = zx.simulate.find_stabilizer_decomp(term)
                        for b in bss:
                            if not b.scalar.is_zero:
                                clifford_terms.append(b)
                    except (ValueError, RuntimeError) as e:
                        if debug:
                            print(f"  BSS failed: {e}")
                        # Keep term as-is
                        clifford_terms.append(term)
                else:
                    if debug:
                        print(f"  No cuttable T-gate, keeping as-is")
                    clifford_terms.append(term)
                continue

            if debug:
                nb_types = [term.type(n) for n in term.neighbors(cut_v)]
                print(f"  Cutting vertex {cut_v}, T-count={tc}, neighbors={len(nb_types)}")

            # Apply cut
            g_left, g_right = cut_spider(term, cut_v)
            new_terms.extend([g_left, g_right])

        if not new_terms:
            break

        terms = new_terms

        if debug:
            print(f"Iteration {iteration+1}: {len(terms)} non-Clifford, {len(clifford_terms)} Clifford")
    else:
        # for-else: only executes if loop completed WITHOUT break (hit max_iterations)
        # Add remaining unfinished terms
        if use_tsim_bss:
            for term in terms:
                if not term.scalar.is_zero:
                    tc = tcount(term)
                    if debug:
                        print(f"  Using BSS for unfinished term (hit max_iterations), T-count={tc}")
                    try:
                        bss_terms = zx.simulate.find_stabilizer_decomp(term)
                        for bt in bss_terms:
                            if not bt.scalar.is_zero:
                                clifford_terms.append(bt)
                    except (ValueError, RuntimeError) as e:
                        if debug:
                            print(f"  BSS failed: {e}, keeping term as-is")
                        clifford_terms.append(term)
        else:
            for term in terms:
                if not term.scalar.is_zero:
                    clifford_terms.append(term)
                    if debug:
                        print(f"  Adding unfinished term (hit max_iterations), T-count={tcount(term)}")

    # Prepare final terms - reduce and set I/O
    final_terms = []
    for term in clifford_terms:
        safe_set_io(term, orig_inputs, orig_outputs)
        # Only reduce if not already reduced in loop
        if not reduce_intermediate:
            zx.full_reduce(term, paramSafe=param_safe)
        safe_set_io(term, orig_inputs, orig_outputs)
        if not term.scalar.is_zero:
            final_terms.append(term)

    if debug:
        print(f"Decomposition complete: {len(final_terms)} terms")

    return final_terms


def decompose_no_reduce(
    g: BaseGraph,
    max_iterations: int = 50,
    debug: bool = False,
    cut_strategy: str = "fewest_neighbors"
) -> List[BaseGraph]:
    """
    Decompose without intermediate reductions.

    This avoids the scalar.is_zero issue from disconnected components
    by only reducing the input graph once, then cutting without reducing.
    Final reduction happens only after all cuts are complete.

    Args:
        g: Input ZX graph
        max_iterations: Maximum cutting iterations
        debug: If True, print debug info
        cut_strategy: How to select which T-gate to cut:
            - "fewest_neighbors": Prefer vertices with fewer neighbors (default)
            - "first": Return the first cuttable T-gate found
            - "z_only_first": Only Z-spiders, first found (matches investigation notebook)

    Returns:
        List of Clifford graphs whose sum equals the original
    """
    orig_inputs = list(g.inputs())
    orig_outputs = list(g.outputs())

    # Initial reduction (use deepcopy to avoid shallow copy bugs with params)
    g_init = deepcopy(g)
    zx.full_reduce(g_init)
    initial_tc = tcount(g_init)

    if debug:
        print(f"Initial T-count: {initial_tc}")

    if initial_tc == 0:
        return [g_init]

    terms = [g_init]

    # Cut all T-gates without intermediate reduction
    for iteration in range(max_iterations):
        new_terms = []
        all_clifford = True

        for term in terms:
            tc = tcount(term)

            if tc == 0:
                new_terms.append(term)
                continue

            all_clifford = False
            cut_v = find_best_cut(term, strategy=cut_strategy)

            if cut_v is None:
                # Can't cut - keep as is
                new_terms.append(term)
                continue

            if debug:
                print(f"  Iter {iteration+1}: cutting vertex {cut_v}, T-count={tc}")

            g_left, g_right = cut_spider(term, cut_v)
            new_terms.extend([g_left, g_right])

        terms = new_terms

        if all_clifford:
            break

        if debug:
            print(f"Iteration {iteration+1}: {len(terms)} terms")

    # Final reduction
    final_terms = []
    for term in terms:
        safe_set_io(term, orig_inputs, orig_outputs)
        zx.full_reduce(term)
        safe_set_io(term, orig_inputs, orig_outputs)
        if not term.scalar.is_zero:
            final_terms.append(term)

    if debug:
        print(f"Final: {len(final_terms)} non-zero terms")

    return final_terms


# ============================================================================
# Verification utilities
# ============================================================================

def matrices_close(m1, m2, rtol=1e-5, atol=1e-8):
    """Check if two matrices are equal up to global phase."""
    if m1.shape != m2.shape:
        return False, f"Shape mismatch: {m1.shape} vs {m2.shape}"

    if np.allclose(m1, 0) and np.allclose(m2, 0):
        return True, "Both zero"

    if np.allclose(m1, m2, rtol=rtol, atol=atol):
        return True, "Direct match"

    nz1 = np.abs(m1) > atol
    nz2 = np.abs(m2) > atol

    if not np.array_equal(nz1, nz2):
        return False, "Different sparsity pattern"

    if not np.any(nz1):
        return True, "Both effectively zero"

    idx = np.argmax(nz1.flatten())
    phase = m2.flatten()[idx] / m1.flatten()[idx]

    if np.allclose(m1 * phase, m2, rtol=rtol, atol=atol):
        return True, f"Match with global phase (mag={np.abs(phase):.6f}, arg={np.angle(phase):.6f})"

    return False, f"No match"


def verify_decomposition(g: BaseGraph, terms: List[BaseGraph], param_safe: bool = True) -> Tuple[bool, str]:
    """
    Verify decomposition by comparing sum of terms with original.

    Args:
        g: Original graph
        terms: Decomposed terms
        param_safe: Use paramSafe for reductions

    Returns:
        (match, message)
    """
    # Get original matrix (use deepcopy to avoid shallow copy bugs with params)
    g_orig = deepcopy(g)
    g_orig.auto_detect_io()
    orig_matrix = g_orig.to_matrix()

    # Sum term matrices
    term_matrices = []
    for t in terms:
        t_copy = deepcopy(t)  # Use deepcopy to avoid shallow copy bugs with params
        zx.full_reduce(t_copy, paramSafe=param_safe)
        t_copy.auto_detect_io()
        if not t_copy.scalar.is_zero:
            term_matrices.append(t_copy.to_matrix())

    if not term_matrices:
        return False, "No non-zero terms"

    term_sum = sum(term_matrices)

    return matrices_close(orig_matrix, term_sum)


def verify_against_bss(g: BaseGraph, terms: List[BaseGraph], param_safe: bool = True) -> Tuple[bool, str, int, int]:
    """
    Verify decomposition by comparing with BSS.

    Args:
        g: Original graph
        terms: Decomposed terms
        param_safe: Use paramSafe for reductions

    Returns:
        (match, message, n_our_terms, n_bss_terms)
    """
    # Get BSS decomposition (use deepcopy to avoid shallow copy bugs with params)
    g_bss = deepcopy(g)
    zx.full_reduce(g_bss, paramSafe=param_safe)
    g_bss.auto_detect_io()

    tc = tcount(g_bss)
    if tc > 0:
        bss_terms = zx.simulate.find_stabilizer_decomp(g_bss)
    else:
        bss_terms = [g_bss]

    # Compute BSS sum
    bss_valid = [t for t in bss_terms if not t.scalar.is_zero]
    for t in bss_valid:
        t.auto_detect_io()
    bss_sum = sum(t.to_matrix() for t in bss_valid)

    # Compute our terms sum
    our_valid = [t for t in terms if not t.scalar.is_zero]
    our_matrices = []
    for t in our_valid:
        t_copy = deepcopy(t)  # Use deepcopy to avoid shallow copy bugs with params
        zx.full_reduce(t_copy, paramSafe=param_safe)
        t_copy.auto_detect_io()
        if not t_copy.scalar.is_zero:
            our_matrices.append(t_copy.to_matrix())

    if not our_matrices:
        return False, "No non-zero terms", 0, len(bss_valid)

    our_sum = sum(our_matrices)

    match, msg = matrices_close(bss_sum, our_sum)
    return match, msg, len(our_matrices), len(bss_valid)
