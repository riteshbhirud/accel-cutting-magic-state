"""
pauli_web_evaluator.py — Closed Pauli web post-selection for d=5 cultivation.

Computes the 20 projection detector parities classically from the error
configuration (e-parameters), enabling full 89-detector post-selection
without inflating T from 30 to 68.

Architecture (Wan & Zhong Section 8.2):
  - Web matrix W_proj (20 × 7787) maps e-parameters to projection detector parities
  - Augmented error_transform = vstack(et69, W_proj) for ChannelSampler
  - Output splits: first num_f bits = f-parameters, last 20 = proj det parities
  - Post-select on proj det = 0 BEFORE ZX evaluation (avoids wasted compute)

Validated:
  - 200/200 multi-error injection tests (1-5 simultaneous errors, S-gate circuit)
  - Single-error probing: all 7787 e-parameters, deterministic across seeds
  - Post-selection survival rate matches stim: 18.11% at p=0.001

Construction method:
  - For each e-parameter bit j (0..7786):
    1. Inject single Pauli error at j's circuit position into noiseless stim circuit
    2. Compare measurement outcomes with noiseless baseline
    3. Compute detector parities from measurement flips
    4. Record as column j of W_e
  - W_proj = W_e[69:89, :] (the 20 projection detector rows)
"""

import os
import re
import numpy as np

# ── Constants ────────────────────────────────────────────────────────────────

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def load_web_matrix_We():
    """Load the full 89×7787 web matrix W_e."""
    path = os.path.join(_THIS_DIR, 'd5_web_matrix_We.npy')
    return np.load(path)


def load_web_matrix_Wproj():
    """Load the 20×7787 projection detector web matrix W_proj."""
    path = os.path.join(_THIS_DIR, 'd5_web_matrix_Wproj.npy')
    return np.load(path)


# ── Core functions ───────────────────────────────────────────────────────────

def build_augmented_error_transform(error_transform_69, W_proj=None):
    """Build augmented error_transform = vstack(et69, W_proj).

    The augmented matrix is used as the error_transform in the ChannelSampler.
    Output shape: (num_f + 20, num_e) where num_f = error_transform_69.shape[0].

    The ChannelSampler output splits as:
      - [:num_f]  → f-parameters for the 69-det ZX evaluation
      - [num_f:]  → 20 projection detector parities (post-select on all = 0)

    Args:
        error_transform_69: shape (num_f, num_e) from the 69-det SamplingGraph
        W_proj: shape (20, num_e) projection web matrix (default: load from file)

    Returns:
        augmented: shape (num_f + 20, num_e) binary matrix
        num_f: number of f-parameters (for splitting the output)
    """
    if W_proj is None:
        W_proj = load_web_matrix_Wproj()

    num_f = error_transform_69.shape[0]
    assert error_transform_69.shape[1] == W_proj.shape[1], \
        f"Column mismatch: et69 has {error_transform_69.shape[1]}, W_proj has {W_proj.shape[1]}"

    augmented = np.vstack([
        error_transform_69.astype(np.uint8),
        W_proj.astype(np.uint8)
    ])
    return augmented, num_f


def postselect_projection_detectors(sampler_output, num_f):
    """Post-select on projection detector parities = 0.

    Args:
        sampler_output: shape (N, num_f + 20) from augmented ChannelSampler
        num_f: number of f-parameters (split point)

    Returns:
        f_params: shape (N_kept, num_f) f-parameters for kept shots
        keep_mask: shape (N,) boolean mask of kept shots
    """
    proj_det = sampler_output[:, num_f:]  # (N, 20)
    keep_mask = np.all(proj_det == 0, axis=1)
    f_params = sampler_output[keep_mask, :num_f]
    return f_params, keep_mask


def compute_all_detector_parities(e_params, W_e=None):
    """Compute all 89 detector parities from e-parameters.

    Args:
        e_params: shape (N, num_e) binary e-parameter vectors
        W_e: shape (89, num_e) web matrix (default: load from file)

    Returns:
        det_parities: shape (N, 89) binary detector parity vectors
    """
    if W_e is None:
        W_e = load_web_matrix_We()
    return (e_params @ W_e.T) % 2


def compute_projection_detector_parities(e_params, W_proj=None):
    """Compute 20 projection detector parities from e-parameters.

    Args:
        e_params: shape (N, num_e) binary e-parameter vectors
        W_proj: shape (20, num_e) web matrix (default: load from file)

    Returns:
        proj_parities: shape (N, 20) binary projection detector parity vectors
    """
    if W_proj is None:
        W_proj = load_web_matrix_Wproj()
    return (e_params @ W_proj.T) % 2


# ── Construction ─────────────────────────────────────────────────────────────

def build_web_matrix(stim_circuit, verbose=True):
    """Build the full 89×num_e web matrix W_e by probing each e-parameter.

    This constructs W_e from scratch by injecting single Pauli errors into
    the noiseless version of the circuit and computing which detectors fire.

    Args:
        stim_circuit: stim.Circuit (T→S substituted, 89-detector version)
        verbose: print progress

    Returns:
        W_e: shape (num_det, num_e) binary web matrix
        e_map: list of (pauli_char, qubit, inst_idx) for each e-parameter
    """
    import stim as _stim

    NOISE_1Q = {'DEPOLARIZE1', 'PAULI_CHANNEL_1'}
    NOISE_2Q = {'DEPOLARIZE2', 'PAULI_CHANNEL_2'}

    # Parse e-parameter map
    e_map = []
    for inst_idx, inst in enumerate(stim_circuit.flattened()):
        name = inst.name
        targets = [t.value for t in inst.targets_copy()]
        args = inst.gate_args_copy()
        if name in NOISE_1Q:
            for q in targets:
                e_map.append(('Z', q, inst_idx))
                e_map.append(('X', q, inst_idx))
        elif name in NOISE_2Q:
            for i in range(0, len(targets), 2):
                qi, qj = targets[i], targets[i + 1]
                e_map.append(('Z', qi, inst_idx))
                e_map.append(('X', qi, inst_idx))
                e_map.append(('Z', qj, inst_idx))
                e_map.append(('X', qj, inst_idx))
        elif name == 'X_ERROR':
            for q in targets:
                e_map.append(('X', q, inst_idx))
        elif name == 'Z_ERROR':
            for q in targets:
                e_map.append(('Z', q, inst_idx))
        elif name == 'Y_ERROR':
            for q in targets:
                e_map.append(('Y', q, inst_idx))
        elif name in ('M', 'MZ', 'MR', 'MRZ', 'MX', 'MRX', 'MY', 'MRY'):
            if args and args[0] > 0:
                for q in targets:
                    e_map.append(('X', q, inst_idx))

    num_e = len(e_map)

    # Parse detector indices
    flat = stim_circuit.flattened()
    meas_count = 0
    det_indices = []
    for inst in flat:
        name = inst.name
        if name in ('M', 'MX', 'MZ', 'MY', 'MR', 'MRX', 'MRZ', 'MRY'):
            meas_count += len(inst.targets_copy())
        elif name == 'DETECTOR':
            abs_indices = []
            for t in inst.targets_copy():
                if t.is_measurement_record_target:
                    abs_indices.append(meas_count + t.value)
            det_indices.append(abs_indices)

    num_det = len(det_indices)
    flat_noisy = list(stim_circuit.flattened())

    # Build noiseless circuit
    noiseless_lines = []
    for inst in flat_noisy:
        name = inst.name
        if name in NOISE_1Q | NOISE_2Q | {'X_ERROR', 'Z_ERROR', 'Y_ERROR'}:
            continue
        if name in ('M', 'MZ', 'MR', 'MRZ', 'MX', 'MRX', 'MY', 'MRY'):
            args = inst.gate_args_copy()
            if args and args[0] > 0:
                targets_str = ' '.join(str(t.value) for t in inst.targets_copy())
                noiseless_lines.append(f"{name} {targets_str}")
                continue
        noiseless_lines.append(str(inst))
    noiseless = _stim.Circuit('\n'.join(noiseless_lines))

    baseline = noiseless.compile_sampler(seed=42).sample(1).astype(np.uint8)[0]

    # Probe each e-parameter bit
    W_e = np.zeros((num_det, num_e), dtype=np.uint8)
    for j in range(num_e):
        pauli_char, qubit, target_inst_idx = e_map[j]

        lines = []
        for inst_idx, inst in enumerate(flat_noisy):
            name = inst.name
            if name in NOISE_1Q | NOISE_2Q | {'X_ERROR', 'Z_ERROR', 'Y_ERROR'}:
                if inst_idx == target_inst_idx:
                    if pauli_char == 'X':
                        lines.append(f"X {qubit}")
                    elif pauli_char == 'Z':
                        lines.append(f"Z {qubit}")
                    elif pauli_char == 'Y':
                        lines.append(f"Y {qubit}")
                continue
            if name in ('M', 'MZ', 'MR', 'MRZ', 'MX', 'MRX', 'MY', 'MRY'):
                args = inst.gate_args_copy()
                if args and args[0] > 0:
                    if inst_idx == target_inst_idx:
                        lines.append(f"X {qubit}")
                    targets_str = ' '.join(str(t.value) for t in inst.targets_copy())
                    lines.append(f"{name} {targets_str}")
                    continue
            lines.append(str(inst))

        error_circ = _stim.Circuit('\n'.join(lines))
        probe = error_circ.compile_sampler(seed=42).sample(1).astype(np.uint8)[0]
        flips = probe ^ baseline

        for d, indices in enumerate(det_indices):
            if indices:
                W_e[d, j] = flips[indices].sum() % 2

        if verbose and (j + 1) % 1000 == 0:
            print(f"  Probed {j + 1}/{num_e} e-parameter bits")

    return W_e, e_map


# ── Validation ───────────────────────────────────────────────────────────────

def validate_web_matrix(stim_circuit, W_e=None, n_tests=200, seed=123):
    """Validate W_e with random multi-error injection tests.

    Args:
        stim_circuit: stim.Circuit (T→S substituted, 89-detector version)
        W_e: web matrix to validate (default: load from file)
        n_tests: number of random tests
        seed: random seed

    Returns:
        dict with validation results
    """
    import stim as _stim

    if W_e is None:
        W_e = load_web_matrix_We()

    NOISE_1Q = {'DEPOLARIZE1', 'PAULI_CHANNEL_1'}
    NOISE_2Q = {'DEPOLARIZE2', 'PAULI_CHANNEL_2'}

    # Parse e-param map
    e_map = []
    for inst_idx, inst in enumerate(stim_circuit.flattened()):
        name = inst.name
        targets = [t.value for t in inst.targets_copy()]
        args = inst.gate_args_copy()
        if name in NOISE_1Q:
            for q in targets:
                e_map.append(('Z', q, inst_idx))
                e_map.append(('X', q, inst_idx))
        elif name in NOISE_2Q:
            for i in range(0, len(targets), 2):
                e_map.append(('Z', targets[i], inst_idx))
                e_map.append(('X', targets[i], inst_idx))
                e_map.append(('Z', targets[i + 1], inst_idx))
                e_map.append(('X', targets[i + 1], inst_idx))
        elif name == 'X_ERROR':
            for q in targets:
                e_map.append(('X', q, inst_idx))
        elif name == 'Z_ERROR':
            for q in targets:
                e_map.append(('Z', q, inst_idx))
        elif name == 'Y_ERROR':
            for q in targets:
                e_map.append(('Y', q, inst_idx))
        elif name in ('M', 'MZ', 'MR', 'MRZ', 'MX', 'MRX', 'MY', 'MRY'):
            if args and args[0] > 0:
                for q in targets:
                    e_map.append(('X', q, inst_idx))

    # Parse detector indices
    meas_count = 0
    det_indices = []
    for inst in stim_circuit.flattened():
        name = inst.name
        if name in ('M', 'MX', 'MZ', 'MY', 'MR', 'MRX', 'MRZ', 'MRY'):
            meas_count += len(inst.targets_copy())
        elif name == 'DETECTOR':
            abs_indices = []
            for t in inst.targets_copy():
                if t.is_measurement_record_target:
                    abs_indices.append(meas_count + t.value)
            det_indices.append(abs_indices)

    flat_noisy = list(stim_circuit.flattened())

    # Noiseless baseline
    noiseless_lines = []
    for inst in flat_noisy:
        name = inst.name
        if name in NOISE_1Q | NOISE_2Q | {'X_ERROR', 'Z_ERROR', 'Y_ERROR'}:
            continue
        if name in ('M', 'MZ', 'MR', 'MRZ', 'MX', 'MRX', 'MY', 'MRY'):
            args = inst.gate_args_copy()
            if args and args[0] > 0:
                targets_str = ' '.join(str(t.value) for t in inst.targets_copy())
                noiseless_lines.append(f"{name} {targets_str}")
                continue
        noiseless_lines.append(str(inst))
    noiseless = _stim.Circuit('\n'.join(noiseless_lines))
    baseline = noiseless.compile_sampler(seed=42).sample(1).astype(np.uint8)[0]

    active_e = np.where(W_e.sum(axis=0) > 0)[0]
    rng = np.random.RandomState(seed)
    match_count = 0

    for _ in range(n_tests):
        n_errors = rng.randint(1, 6)
        error_bits = rng.choice(active_e, size=n_errors, replace=False).tolist()

        e_vec = np.zeros(len(e_map), dtype=np.uint8)
        e_vec[error_bits] = 1
        predicted = (W_e @ e_vec) % 2

        # Build error circuit
        errors_by_inst = {}
        for j in error_bits:
            pc, q, idx = e_map[j]
            errors_by_inst.setdefault(idx, []).append((pc, q))

        lines = []
        for inst_idx, inst in enumerate(flat_noisy):
            name = inst.name
            if name in NOISE_1Q | NOISE_2Q | {'X_ERROR', 'Z_ERROR', 'Y_ERROR'}:
                if inst_idx in errors_by_inst:
                    for pc, q in errors_by_inst[inst_idx]:
                        lines.append(f"{'X' if pc == 'X' else 'Z' if pc == 'Z' else 'Y'} {q}")
                continue
            if name in ('M', 'MZ', 'MR', 'MRZ', 'MX', 'MRX', 'MY', 'MRY'):
                args = inst.gate_args_copy()
                if args and args[0] > 0:
                    if inst_idx in errors_by_inst:
                        for pc, q in errors_by_inst[inst_idx]:
                            lines.append(f"X {q}")
                    targets_str = ' '.join(str(t.value) for t in inst.targets_copy())
                    lines.append(f"{name} {targets_str}")
                    continue
            lines.append(str(inst))

        ec = _stim.Circuit('\n'.join(lines))
        probe = ec.compile_sampler(seed=42).sample(1).astype(np.uint8)[0]
        flips = probe ^ baseline
        actual = np.zeros(len(det_indices), dtype=np.uint8)
        for d, indices in enumerate(det_indices):
            if indices:
                actual[d] = flips[indices].sum() % 2

        if np.array_equal(predicted, actual):
            match_count += 1

    return {
        'n_tests': n_tests,
        'match_count': match_count,
        'match_rate': match_count / n_tests,
    }


if __name__ == '__main__':
    import stim
    from d5_circuit_utils import build_d5_full_circuit, get_d5_path

    with open(get_d5_path(0.001)) as f:
        content = f.read()

    full_str = build_d5_full_circuit(content)
    stim_str = re.sub(r'\bT\b', 'S', full_str.replace('T_DAG', 'S_DAG'))
    circ = stim.Circuit(stim_str)

    # Load or rebuild
    try:
        W_e = load_web_matrix_We()
        print(f"Loaded W_e: {W_e.shape}, {W_e.sum()} nonzeros")
    except FileNotFoundError:
        print("Building W_e from scratch...")
        W_e, _ = build_web_matrix(circ)
        np.save(os.path.join(_THIS_DIR, 'd5_web_matrix_We.npy'), W_e)
        np.save(os.path.join(_THIS_DIR, 'd5_web_matrix_Wproj.npy'), W_e[69:])
        print(f"Built and saved W_e: {W_e.shape}")

    # Validate
    print("\nValidating W_e (200 multi-error tests)...")
    results = validate_web_matrix(circ, W_e)
    print(f"Match: {results['match_count']}/{results['n_tests']} = {results['match_rate']:.4f}")
    assert results['match_rate'] == 1.0, "W_e validation failed!"
    print("VALIDATION PASSED")
