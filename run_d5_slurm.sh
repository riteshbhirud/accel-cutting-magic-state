#!/bin/bash
#
# run_d5_slurm.sh — Submit d=5 cultivation runs at 4 noise levels.
#
# Usage:
#   sbatch run_d5_slurm.sh            # submit all 4 jobs
#   bash run_d5_slurm.sh              # run locally (sequential, for testing)
#
# Each noise level runs as a separate SLURM array task.
# Results saved to d5_results_{p}.jsonl (append-mode, resumable).
#

# ── SLURM configuration ──────────────────────────────────────────────────────
#SBATCH --job-name=d5-cultivation
#SBATCH --array=0-3
#SBATCH --partition=gpu          # EDIT: your GPU partition name
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=d5_%A_%a.out
#SBATCH --error=d5_%A_%a.err

# ── Module loads (EDIT for your cluster) ──────────────────────────────────────
# module load cuda/12.x
# module load python/3.12
# module load conda
# conda activate tsim  # or your environment name

# ── Environment ───────────────────────────────────────────────────────────────
export JAX_PLATFORMS=cuda          # Use GPU
# export JAX_PLATFORMS=cpu         # Uncomment for CPU-only testing

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Noise levels and shot counts ─────────────────────────────────────────────
# Determined from validation + shot requirement analysis (20% relative CI):
#   p=0.0002: PSR~0.72, LER~2.8e-3  → min 192K shots → plan 500K (12% bars)
#   p=0.0005: PSR~0.44, LER~5.9e-3  → min 147K shots → plan 500K (11% bars)
#   p=0.001:  PSR~0.19, LER~7.4e-3  → min 273K shots → plan 600K (14% bars)
#   p=0.002:  PSR~0.035, LER~2.6e-2 → min 417K shots → plan 2M   (9% bars)
NOISE_LEVELS=(0.0002 0.0005 0.001 0.002)
SHOT_COUNTS=(500000 500000 600000 2000000)

# Select this task's noise level
if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    IDX=$SLURM_ARRAY_TASK_ID
else
    # Running locally — process all sequentially
    for IDX in 0 1 2 3; do
        P=${NOISE_LEVELS[$IDX]}
        N=${SHOT_COUNTS[$IDX]}
        echo "=== p=$P, shots=$N ==="
        python run_d5_paper.py --p "$P" --shots "$N" --resume \
            --batch-size 10000 --eval-batch 1024
    done
    exit 0
fi

P=${NOISE_LEVELS[$IDX]}
N=${SHOT_COUNTS[$IDX]}

echo "============================================================"
echo "D=5 Cultivation — p=$P, shots=$N"
echo "Node: $(hostname), GPU: $CUDA_VISIBLE_DEVICES"
echo "Date: $(date)"
echo "============================================================"

python run_d5_paper.py --p "$P" --shots "$N" --resume \
    --batch-size 10000 --eval-batch 1024

echo "Done. $(date)"
