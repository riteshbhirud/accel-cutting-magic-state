"""run_d5_paper.py — Self-contained d=5 cultivation sampler for cluster runs.

Usage:
    python run_d5_paper.py --p 0.001 --shots 500000
    python run_d5_paper.py --p 0.001 --shots 500000 --resume  # continue from existing jsonl

Output:
    d5_results_{p}.jsonl — one JSON line per batch, append-mode (survives interruption)

Each line contains:
    {p, batch_shots, n_proj_pass, n_89_pass, obs_sum, psr_89, ler,
     elapsed, cumul_shots, cumul_89_pass, cumul_obs_sum, cumul_ler}
"""

import os
import sys
import re
import time
import json
import argparse
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cuda")

import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'tsim', 'src'))

import tsim
from tsim.core.graph import prepare_graph
from tsim.noise.channels import ChannelSampler

from d5_circuit_utils import get_d5_path, build_d5_working_circuit
from pauli_web_evaluator import load_web_matrix_Wproj

# Monkey-patch for cfloat performance (if available)
try:
    from evaluate_matmul_cfloat import evaluate_batch as evaluate_batch_cfloat
    import tsim_cutting as _mod
    _mod.evaluate_batch = evaluate_batch_cfloat
except ImportError:
    pass

from tsim_cutting import (compile_program_subcomp_enum_general,
                           sample_program_subcomp_enum)


def make_p0002_circuit_if_needed():
    """Create p=0.0002 circuit from p=0.001 template if it doesn't exist."""
    try:
        get_d5_path(0.0002)
        return  # already exists
    except FileNotFoundError:
        pass
    template_path = get_d5_path(0.001)
    with open(template_path) as f:
        template = f.read()
    circuit_0002 = re.sub(r'(?<=\()0\.001(?=\))', '0.0002', template)
    out_path = template_path.replace('p=0.001', 'p=0.0002')
    with open(out_path, 'w') as f:
        f.write(circuit_0002)
    print(f"Created p=0.0002 circuit: {out_path}")


def load_existing_results(jsonl_path):
    """Load cumulative state from existing jsonl file."""
    cumul_shots = 0
    cumul_89_pass = 0
    cumul_obs_sum = 0
    if os.path.exists(jsonl_path):
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                cumul_shots = rec.get('cumul_shots', cumul_shots)
                cumul_89_pass = rec.get('cumul_89_pass', cumul_89_pass)
                cumul_obs_sum = rec.get('cumul_obs_sum', cumul_obs_sum)
    return cumul_shots, cumul_89_pass, cumul_obs_sum


def main():
    parser = argparse.ArgumentParser(description='D=5 cultivation paper runs')
    parser.add_argument('--p', type=float, required=True,
                        help='Physical noise level')
    parser.add_argument('--shots', type=int, required=True,
                        help='Total shots to draw (pre-postselection)')
    parser.add_argument('--batch-size', type=int, default=10000,
                        help='Shots per batch (default: 10000)')
    parser.add_argument('--eval-batch', type=int, default=1024,
                        help='ZX evaluation batch size (default: 1024)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', action='store_true',
                        help='Resume from existing jsonl file')
    parser.add_argument('--output-dir', type=str, default='.',
                        help='Output directory')
    args = parser.parse_args()

    p = args.p

    # Create p=0.0002 if needed
    if p == 0.0002:
        make_p0002_circuit_if_needed()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / f"d5_results_{p}.jsonl"

    # ── Load existing state if resuming ───────────────────────────────────────
    if args.resume:
        cumul_shots, cumul_89_pass, cumul_obs_sum = load_existing_results(jsonl_path)
        if cumul_shots > 0:
            print(f"Resuming from {jsonl_path}: "
                  f"{cumul_shots} shots, {cumul_89_pass} passes, "
                  f"LER={cumul_obs_sum/cumul_89_pass:.4e}" if cumul_89_pass > 0 else "")
    else:
        cumul_shots = 0
        cumul_89_pass = 0
        cumul_obs_sum = 0

    remaining = args.shots - cumul_shots
    if remaining <= 0:
        print(f"Already have {cumul_shots} >= {args.shots} shots. Done.")
        return

    # ── Compile ───────────────────────────────────────────────────────────────
    print(f"{'='*60}")
    print(f"D=5 Cultivation Sampler — p={p}")
    print(f"{'='*60}")
    print(f"Target: {args.shots} total shots, {remaining} remaining")
    print(f"Batch size: {args.batch_size}, eval batch: {args.eval_batch}")

    with open(get_d5_path(p)) as f:
        content = f.read()
    working_str = build_d5_working_circuit(content)
    circ = tsim.Circuit(working_str)

    t0 = time.time()
    prepared = prepare_graph(circ, sample_detectors=True)
    num_f = prepared.error_transform.shape[0]
    num_det = prepared.num_detectors

    program = compile_program_subcomp_enum_general(
        prepared, max_cut_iterations=30)
    compile_time = time.time() - t0
    print(f"Compiled in {compile_time:.1f}s (num_f={num_f}, num_det={num_det})")

    # Build augmented error_transform
    W_proj = load_web_matrix_Wproj()
    et = np.array(prepared.error_transform, dtype=np.uint8)
    augmented_et = np.vstack([et, W_proj])

    key = jax.random.key(args.seed + cumul_shots)  # offset seed if resuming
    key, subkey = jax.random.split(key)
    channel_seed = int(jax.random.randint(subkey, (), 0, 2**30))
    cs = ChannelSampler(
        channel_probs=prepared.channel_probs,
        error_transform=augmented_et,
        seed=channel_seed,
    )

    # ── JIT warmup ────────────────────────────────────────────────────────────
    print("JIT warmup...")
    warmup_f = jnp.zeros((1, num_f), dtype=jnp.uint8)
    key, subkey = jax.random.split(key)
    _ = sample_program_subcomp_enum(program, warmup_f, subkey)
    print("JIT warmup done.")

    # ── Main sampling loop ────────────────────────────────────────────────────
    print(f"\nSampling (output: {jsonl_path})...")
    t_start = time.time()
    shots_done = 0

    while shots_done < remaining:
        batch = min(args.batch_size, remaining - shots_done)
        t_batch = time.time()

        # Sample augmented f-params
        f_aug = np.array(cs.sample(batch), dtype=np.uint8)
        f_params = f_aug[:, :num_f]
        proj_det = f_aug[:, num_f:]

        # Post-select on projection detectors
        proj_pass = np.all(proj_det == 0, axis=1)
        n_proj_pass = int(proj_pass.sum())

        n_89_pass = 0
        obs_sum = 0

        if n_proj_pass > 0:
            f_kept = f_params[proj_pass]

            # Evaluate in sub-batches for memory
            det_chunks = []
            obs_chunks = []
            for i in range(0, n_proj_pass, args.eval_batch):
                chunk = jnp.array(f_kept[i:i + args.eval_batch])
                key, subkey = jax.random.split(key)
                samples = sample_program_subcomp_enum(program, chunk, subkey)
                samples_np = np.array(samples, dtype=np.uint8)
                det_chunks.append(samples_np[:, :num_det])
                obs_chunks.append(samples_np[:, num_det:])

            det_all = np.concatenate(det_chunks)
            obs_all = np.concatenate(obs_chunks)

            # Full 89-det post-selection
            working_pass = np.all(det_all == 0, axis=1)
            n_89_pass = int(working_pass.sum())
            obs_surviving = obs_all[working_pass]
            if obs_surviving.ndim > 1:
                obs_surviving = obs_surviving[:, 0]
            obs_sum = int(obs_surviving.sum())

        # Update cumulative state
        shots_done += batch
        cumul_shots += batch
        cumul_89_pass += n_89_pass
        cumul_obs_sum += obs_sum
        cumul_ler = cumul_obs_sum / cumul_89_pass if cumul_89_pass > 0 else float('nan')

        elapsed_batch = time.time() - t_batch
        psr_batch = n_89_pass / batch if batch > 0 else 0
        ler_batch = obs_sum / n_89_pass if n_89_pass > 0 else float('nan')

        # Write result line (append mode — survives interruption)
        record = {
            'p': p,
            'batch_shots': batch,
            'n_proj_pass': n_proj_pass,
            'n_89_pass': n_89_pass,
            'obs_sum': obs_sum,
            'psr_89': psr_batch,
            'ler': ler_batch if not (isinstance(ler_batch, float) and
                                     ler_batch != ler_batch) else None,
            'elapsed': round(elapsed_batch, 2),
            'cumul_shots': cumul_shots,
            'cumul_89_pass': cumul_89_pass,
            'cumul_obs_sum': cumul_obs_sum,
            'cumul_ler': cumul_ler if not (isinstance(cumul_ler, float) and
                                           cumul_ler != cumul_ler) else None,
        }
        with open(jsonl_path, 'a') as f:
            f.write(json.dumps(record) + '\n')

        # Progress reporting
        elapsed_total = time.time() - t_start
        rate = cumul_shots / elapsed_total if elapsed_total > 0 else 0
        if shots_done % (args.batch_size * 10) == 0 or shots_done >= remaining:
            print(f"  [{cumul_shots:>10d} shots] "
                  f"89-pass={cumul_89_pass:>8d}  "
                  f"LER={cumul_ler:.4e}  "
                  f"PSR={cumul_89_pass/cumul_shots:.4f}  "
                  f"rate={rate:.0f} shots/s  "
                  f"elapsed={elapsed_total:.0f}s")

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed_total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"COMPLETE — p={p}")
    print(f"{'='*60}")
    print(f"Total shots:     {cumul_shots}")
    print(f"89-det passing:  {cumul_89_pass}")
    print(f"Observable sum:  {cumul_obs_sum}")
    print(f"PSR (89-det):    {cumul_89_pass/cumul_shots:.6f}")
    print(f"LER:             {cumul_ler:.6e}")
    print(f"Elapsed:         {elapsed_total:.1f}s")
    print(f"Rate:            {cumul_shots/elapsed_total:.0f} shots/s")

    # Wilson score CI
    if cumul_89_pass > 0:
        p_hat = cumul_ler
        n = cumul_89_pass
        z = 1.96
        denom = 1 + z**2 / n
        center = (p_hat + z**2 / (2 * n)) / denom
        spread = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom
        ci_lo, ci_hi = max(0, center - spread), min(1, center + spread)
        print(f"95% CI:          [{ci_lo:.6e}, {ci_hi:.6e}]")
        print(f"Relative width:  {(ci_hi - ci_lo) / p_hat:.2f}" if p_hat > 0 else "")

    print(f"\nResults saved to {jsonl_path}")


if __name__ == '__main__':
    main()
