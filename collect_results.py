"""collect_results.py — Aggregate d=5 cultivation JSONL results into a summary table.

Reads all d5_results_*.jsonl files in the current directory (or --dir),
prints a summary table with PSR, LER, and 95% Wilson score confidence intervals.

Usage:
    python collect_results.py
    python collect_results.py --dir /path/to/results
"""

import argparse
import json
import math
from pathlib import Path


def wilson_ci(p_hat, n, z=1.96):
    """Wilson score 95% CI for binomial proportion."""
    if n == 0 or p_hat != p_hat:  # nan check
        return (float('nan'), float('nan'))
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    spread = z * math.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom
    return (max(0, center - spread), min(1, center + spread))


def load_jsonl(path):
    """Load last cumulative state from a JSONL file."""
    cumul_shots = 0
    cumul_89_pass = 0
    cumul_obs_sum = 0
    p = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            p = rec.get('p', p)
            cumul_shots = rec.get('cumul_shots', cumul_shots)
            cumul_89_pass = rec.get('cumul_89_pass', cumul_89_pass)
            cumul_obs_sum = rec.get('cumul_obs_sum', cumul_obs_sum)
    return {
        'p': p,
        'cumul_shots': cumul_shots,
        'cumul_89_pass': cumul_89_pass,
        'cumul_obs_sum': cumul_obs_sum,
    }


def main():
    parser = argparse.ArgumentParser(description='Collect d=5 cultivation results')
    parser.add_argument('--dir', type=str, default='.',
                        help='Directory containing d5_results_*.jsonl files')
    args = parser.parse_args()

    results_dir = Path(args.dir)
    jsonl_files = sorted(results_dir.glob('d5_results_*.jsonl'))

    if not jsonl_files:
        print(f"No d5_results_*.jsonl files found in {results_dir}")
        return

    results = []
    for path in jsonl_files:
        r = load_jsonl(path)
        if r['p'] is None:
            continue

        n = r['cumul_89_pass']
        ler = r['cumul_obs_sum'] / n if n > 0 else float('nan')
        psr = n / r['cumul_shots'] if r['cumul_shots'] > 0 else 0
        ci_lo, ci_hi = wilson_ci(ler, n)
        rel_width = (ci_hi - ci_lo) / ler if ler > 0 and not math.isnan(ler) else float('nan')

        results.append({
            'p': r['p'],
            'shots': r['cumul_shots'],
            'n_pass': n,
            'psr': psr,
            'ler': ler,
            'ci_lo': ci_lo,
            'ci_hi': ci_hi,
            'rel_width': rel_width,
        })

    results.sort(key=lambda x: x['p'])

    # Print table
    print(f"\n{'='*80}")
    print("D=5 Cultivation Results Summary")
    print(f"{'='*80}")
    print(f"{'p':>8}  {'Shots':>10}  {'N_pass':>8}  {'PSR':>8}  "
          f"{'LER':>10}  {'95% CI':>23}  {'Rel':>6}")
    print("-" * 80)
    for r in results:
        ci_str = f"[{r['ci_lo']:.4e}, {r['ci_hi']:.4e}]" if not math.isnan(r['ci_lo']) else "N/A"
        rel_str = f"{r['rel_width']:.1%}" if not math.isnan(r['rel_width']) else "N/A"
        print(f"  {r['p']:.4f}  {r['shots']:>10,}  {r['n_pass']:>8,}  "
              f"{r['psr']:>8.4f}  {r['ler']:>10.4e}  {ci_str:>23}  {rel_str:>6}")

    print(f"{'='*80}")

    # Check LER monotonicity
    lers = [r['ler'] for r in results if not math.isnan(r['ler'])]
    if len(lers) >= 2:
        monotonic = all(lers[i] <= lers[i+1] for i in range(len(lers)-1))
        print(f"LER monotonically increasing with p: {'YES' if monotonic else 'NO'}")

    # Check all relative widths < 20%
    all_good = all(r['rel_width'] < 0.20 for r in results if not math.isnan(r['rel_width']))
    print(f"All relative CI widths < 20%: {'YES' if all_good else 'NO'}")
    print()


if __name__ == '__main__':
    main()
