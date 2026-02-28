"""d5_run_paper.py — Produce d=5 cultivation paper results.

Runs the D5DetectorSampler at multiple noise levels and computes
logical error rates with error bars.

Noise levels (from paper): p = 0.0002, 0.0005, 0.001, 0.002

Usage:
    python d5_run_paper.py [--shots N] [--batch-size B] [--noise-levels p1,p2,...]
"""

import os
import sys
import time
import argparse
import json

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from d5_sampler import D5DetectorSampler


def wilson_score_interval(p_hat, n, z=1.96):
    """Wilson score confidence interval for binomial proportion."""
    if n == 0:
        return (0, 1)
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    spread = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom
    return (max(0, center - spread), min(1, center + spread))


def run_noise_level(noise_strength, shots, batch_size, seed=42):
    """Run d=5 sampling at a single noise level."""
    print(f"\n{'='*60}")
    print(f"  p = {noise_strength}")
    print(f"{'='*60}")

    t0 = time.time()
    sampler = D5DetectorSampler(
        noise_strength=noise_strength,
        seed=seed,
    )
    t_compile = time.time() - t0
    print(f"  Compile: {t_compile:.1f}s")

    t0 = time.time()
    ler_result = sampler.estimate_logical_error_rate(
        shots=shots,
        batch_size=batch_size,
    )
    t_sample = time.time() - t0

    n_surv = ler_result['num_surviving']
    p_hat = ler_result['logical_error_rate']
    ci_lo, ci_hi = wilson_score_interval(p_hat, n_surv)

    print(f"  Sample time: {t_sample:.1f}s")
    print(f"  Total shots drawn: {ler_result['total_sampled']}")
    print(f"  Proj survival: {ler_result['proj_survival_rate']:.4f}")
    print(f"  Working survival: {ler_result['working_survival_rate']:.4f}")
    print(f"  89-det surviving: {n_surv}")
    print(f"  Logical error rate: {p_hat:.6f}")
    print(f"  95% CI: [{ci_lo:.6f}, {ci_hi:.6f}]")
    print(f"  Throughput: {ler_result['total_sampled']/t_sample:.0f} total shots/s")

    return {
        'noise_strength': noise_strength,
        'logical_error_rate': float(p_hat),
        'ci_lower': float(ci_lo),
        'ci_upper': float(ci_hi),
        'num_surviving': n_surv,
        'total_sampled': ler_result['total_sampled'],
        'proj_survival_rate': float(ler_result['proj_survival_rate']),
        'working_survival_rate': float(ler_result['working_survival_rate']),
        'overall_survival_rate': float(ler_result['overall_survival_rate']),
        'compile_time': t_compile,
        'sample_time': t_sample,
    }


def main():
    parser = argparse.ArgumentParser(description='D=5 cultivation paper results')
    parser.add_argument('--shots', type=int, default=500,
                        help='Proj-surviving shots per noise level (default: 500)')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for evaluation (default: 128)')
    parser.add_argument('--noise-levels', type=str, default='0.001',
                        help='Comma-separated noise levels (default: 0.001)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default='d5_results.json',
                        help='Output JSON file')
    args = parser.parse_args()

    noise_levels = [float(p) for p in args.noise_levels.split(',')]

    print("=" * 60)
    print("D=5 Cultivation: Paper Results")
    print("=" * 60)
    print(f"Noise levels: {noise_levels}")
    print(f"Shots (proj-surviving): {args.shots}")
    print(f"Batch size: {args.batch_size}")
    print(f"Seed: {args.seed}")

    results = []
    for p in noise_levels:
        r = run_noise_level(p, args.shots, args.batch_size, seed=args.seed)
        results.append(r)

    # Summary table
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'p':>10s}  {'LER':>10s}  {'CI_lo':>10s}  {'CI_hi':>10s}  {'N_surv':>8s}  {'N_total':>8s}")
    for r in results:
        print(f"{r['noise_strength']:>10.4f}  {r['logical_error_rate']:>10.6f}  "
              f"{r['ci_lower']:>10.6f}  {r['ci_upper']:>10.6f}  "
              f"{r['num_surviving']:>8d}  {r['total_sampled']:>8d}")

    # Save results
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.output)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
