"""
d5_circuit_utils.py — Permanent utility module for d=5 cultivation circuit construction.
All functions validated across Tasks 38-92. Do not modify without re-validating.
"""

import re
import os
import json
import stim
import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────

CIRCUIT_DIR = "/Users/ritesh/Downloads/prx/gidney-circuits/circuits/for_perfectionist_decoding"

def get_d5_path(p):
    """Return path to Gidney d=5 circuit at noise level p."""
    import glob
    matches = glob.glob(f"{CIRCUIT_DIR}/c=inject*cultivate*p={p}*d1=5.stim")
    if not matches:
        raise FileNotFoundError(f"No d=5 circuit found for p={p}")
    return matches[0]

# Measurement indices for the monster detector (26 measurements across 4 rounds)
# Validated in Task 88: perfect accuracy against stim detector sampler
MONSTER_MEAS_INDICES = [8,9,12,13,19,20,21,23,24,25,26,37,38,39,41,42,43,44,
                         55,56,57,59,60,61,62,73]

# ── Circuit transformation ────────────────────────────────────────────────────

def replace_s_with_t_safe(s: str) -> str:
    """
    Replace S/S_DAG gates with T/T_DAG for ZX non-Clifford simulation.
    Leaves MPP instructions unchanged (they use S in a different context).
    Validated: preserves all CX, M, MX, DETECTOR, OBSERVABLE_INCLUDE lines.
    """
    out = []
    for l in s.split('\n'):
        ls = l.strip()
        if ls.startswith('MPP'):
            out.append(l)
        elif re.match(r'^\s*S_DAG\b', ls):
            out.append(re.sub(r'\bS_DAG\b', 'T_DAG', l))
        elif re.match(r'^\s*S\b', ls):
            out.append(re.sub(r'\bS\b', 'T', l))
        else:
            out.append(l)
    return '\n'.join(out)

# ── Circuit parsing ───────────────────────────────────────────────────────────

def _find_cut_indices(lines_noisy: list) -> tuple:
    """
    Find the line indices that define the cultivation/projection boundary.
    Returns: (t2_idx, mx_idx, mpp_idx)
      t2_idx: line index of the second T gate block (T 0 3 7...)
      mx_idx: line index of the final MX (after the second T block)
      mpp_idx: line index of the MPP instruction
    Validated: T=30 confirmed for lines[:mx_idx+1] circuit.
    """
    t2_idx = next(
        i for i in range(200, len(lines_noisy))
        if re.search(r'^T\s+0\s+3\s+7', lines_noisy[i].strip())
    )
    mx_idx = next(
        i for i in range(t2_idx, len(lines_noisy))
        if lines_noisy[i].strip().startswith('MX')
    )
    mpp_idx = next(
        i for i, l in enumerate(lines_noisy)
        if l.strip().startswith('MPP')
    )
    return t2_idx, mx_idx, mpp_idx

def _get_obs_line(content: str) -> str:
    """Extract the real OBSERVABLE_INCLUDE line (XOR of last 10 pre-MPP measurements)."""
    noiseless = '\n'.join(
        l for l in content.split('\n')
        if not any(x in l for x in ['DEPOLARIZE','PAULI_CHANNEL','X_ERROR','Z_ERROR'])
    )
    noiseless = re.sub(r'M\([\d.]+\)', 'M', re.sub(r'MX\([\d.]+\)', 'MX', noiseless))
    lines = noiseless.strip().split('\n')
    mpp_idx = next(i for i,l in enumerate(lines) if l.strip().startswith('MPP'))
    return next(
        (l.strip() for l in lines[42:mpp_idx]
         if l.strip().startswith('OBSERVABLE_INCLUDE')),
        'OBSERVABLE_INCLUDE(0) rec[-1]'
    )

# ── Circuit builders ──────────────────────────────────────────────────────────

def build_d5_working_circuit(content: str) -> str:
    """
    Build the working 69-detector T=30 circuit (cultivation only, no projection).
    This is Circuit A: the circuit that compiles cleanly with the existing pipeline.

    Args:
        content: raw .stim file content at any noise level p

    Returns:
        Circuit string suitable for tsim.Circuit()
        - S gates replaced with T gates
        - QUBIT_COORDS removed
        - Real observable included
        - 69 detectors, T=30 in sampling graph
    """
    lines_noisy = replace_s_with_t_safe(content).strip().split('\n')
    _, mx_idx, _ = _find_cut_indices(lines_noisy)
    obs_line = _get_obs_line(content)

    base = '\n'.join(
        l for l in lines_noisy[:mx_idx+1]
        if l.strip()
        and not l.strip().startswith('QUBIT_COORDS')
        and not l.strip().startswith('OBSERVABLE_INCLUDE')
    )
    return base + '\n' + obs_line


def build_d5_full_circuit(content: str) -> str:
    """
    Build the full 89-detector circuit with noiseless projection block.
    This is Circuit B: cultivation (noisy) + projection (noiseless detectors only).

    Architecture (Wan & Zhong Section 9.1):
    - Cultivation block: keep all noise channels
    - Projection block: strip all noise channels, keep DETECTOR lines only
    - Observable: real XOR of last 10 pre-MPP measurements

    Args:
        content: raw .stim file content at any noise level p

    Returns:
        Circuit string suitable for tsim.Circuit() or stim.Circuit() (with T->S)
        - 89 detectors total (69 working + 20 projection)
        - Noiseless projection block
    """
    lines_noisy = replace_s_with_t_safe(content).strip().split('\n')
    _, mx_idx, mpp_idx = _find_cut_indices(lines_noisy)
    obs_line = _get_obs_line(content)

    base = '\n'.join(
        l for l in lines_noisy[:mx_idx+1]
        if l.strip()
        and not l.strip().startswith('QUBIT_COORDS')
        and not l.strip().startswith('OBSERVABLE_INCLUDE')
    )
    proj_block = '\n'.join(
        l for l in lines_noisy[mx_idx+1:mpp_idx]
        if l.strip()
        and not any(l.strip().startswith(x) for x in
                    ['QUBIT_COORDS','SHIFT_COORDS','OBSERVABLE_INCLUDE'])
        and not any(x in l for x in
                    ['DEPOLARIZE','PAULI_CHANNEL','X_ERROR','Z_ERROR'])
    )
    return base + '\n' + proj_block + '\n' + obs_line


def build_d5_combined_circuit(content: str) -> str:
    """
    Build the full d=5 circuit for ZX simulation: cultivation + double-checking
    concatenated as ONE circuit, following the d=3 architecture in run.py.

    This is the circuit passed to compile_detector_sampler_subcomp_enum_general.
    The double-checking block is noiseless (Wan & Zhong Section 9.1).

    Args:
        content: raw .stim file content at any noise level p

    Returns:
        Combined circuit string with:
        - Noisy cultivation block (T gates)
        - Noiseless projection/double-checking block
        - All 89 detectors
        - Real observable
    """
    return build_d5_full_circuit(content)


# ── Detector index accessors ──────────────────────────────────────────────────

def get_d5_proj_det_indices() -> list:
    """
    Return the list of 20 projection detector measurement index lists.
    Each entry is a list of absolute measurement indices (0-indexed) into
    the 93-measurement pre-MPP circuit.

    Source: Task 88, validated with stim detector sampler at 100K shots,
    overall match = 1.000000.

    Returns:
        List of 20 lists, each containing absolute measurement indices.
        proj_det[i] fires iff XOR of raw_measurements[indices] == 1.
    """
    this_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(this_dir, 'd5_proj_det_indices.json')
    if os.path.exists(json_path):
        with open(json_path) as f:
            return json.load(f)
    # Fallback: hardcoded from Task 88 (verified correct)
    return [
        [8,9,12,13,19,20,21,23,24,25,26,37,38,39,41,42,43,44,55,56,57,59,60,61,62,73],
        [74],[75],[76],[77],[78],[79],[80,73],[81],[82],[83],[84],[85],
        [86],[87],[88],[89],[90],[91],[92]
    ]


def get_d5_obs_indices() -> list:
    """
    Return absolute measurement indices for the d=5 observable.
    Observable = XOR of measurements 83-92 (last 10 pre-MPP measurements).

    Source: Task 88, validated.
    """
    return list(range(83, 93))


def compute_detector_parities_from_raw(raw_93: np.ndarray,
                                        proj_indices: list = None) -> np.ndarray:
    """
    Compute all 20 projection detector parities from raw measurements.

    Args:
        raw_93: shape (N, 93) binary array of raw measurement outcomes
        proj_indices: list of 20 index lists (default: get_d5_proj_det_indices())

    Returns:
        shape (N, 20) binary array of detector parities
    """
    if proj_indices is None:
        proj_indices = get_d5_proj_det_indices()
    N = raw_93.shape[0]
    parities = np.zeros((N, len(proj_indices)), dtype=np.uint8)
    for d, indices in enumerate(proj_indices):
        if indices:
            parities[:, d] = raw_93[:, indices].sum(axis=1) % 2
    return parities


# ── Validation ────────────────────────────────────────────────────────────────

def validate_circuit_builders(content: str, n_shots: int = 10000) -> dict:
    """
    Validate that build_d5_working_circuit and build_d5_full_circuit
    produce circuits with the expected detector counts and T-counts.

    Returns dict with validation results.
    """
    import sys
    sys.path.insert(0, '/Users/ritesh/Downloads/prx/tsim/src')
    import tsim
    from tsim.core.graph import prepare_graph, connected_components
    from stab_rank_cut import tcount

    results = {}

    # Working circuit
    working = build_d5_working_circuit(content)
    stim_w = stim.Circuit(re.sub(r'\bT\b','S', working.replace('T_DAG','S_DAG')))
    results['working_detectors'] = stim_w.num_detectors
    results['working_measurements'] = stim_w.num_measurements

    c_w = tsim.Circuit(working)
    sg_w = prepare_graph(c_w, sample_detectors=True)
    comps_w = connected_components(sg_w.graph)
    results['working_T'] = sum(
        tcount(c.graph if hasattr(c,'graph') else c) for c in comps_w)

    # Full circuit
    full = build_d5_full_circuit(content)
    stim_f = stim.Circuit(re.sub(r'\bT\b','S', full.replace('T_DAG','S_DAG')))
    results['full_detectors'] = stim_f.num_detectors
    results['full_measurements'] = stim_f.num_measurements

    # Verify projection detector computation
    raw = stim_f.compile_sampler(seed=42).sample(n_shots).astype(np.uint8)
    det, obs = stim_f.compile_detector_sampler(seed=42).sample(
        n_shots, separate_observables=True)
    proj_computed = compute_detector_parities_from_raw(raw[:, :93])
    proj_stim = det[:, 69:89].astype(np.uint8)
    results['proj_det_match'] = (proj_computed == proj_stim).all(axis=1).mean()

    return results


if __name__ == '__main__':
    import glob
    paths = glob.glob(f"{CIRCUIT_DIR}/c=inject*cultivate*p=0.001*d1=5.stim")
    if paths:
        with open(paths[0]) as f:
            content = f.read()
        results = validate_circuit_builders(content)
        print("Validation results:")
        for k, v in results.items():
            print(f"  {k}: {v}")
        assert results['working_detectors'] == 69, "Expected 69 working detectors"
        assert results['working_T'] == 30, "Expected T=30 for working circuit"
        assert results['full_detectors'] == 89, "Expected 89 full detectors"
        assert results['proj_det_match'] > 0.999, "Projection detector match must be >0.999"
        print("ALL ASSERTIONS PASSED")
    else:
        print("No d=5 circuit found")
