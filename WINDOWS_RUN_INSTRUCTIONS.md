# D=5 Cultivation — Windows GPU Run Instructions

## Prerequisites

- Python 3.12 with conda/miniconda
- NVIDIA GPU with CUDA 12.x
- JAX with CUDA support (`pip install jax[cuda12]`)
- stim, numpy, pyzx_param

## Directory Layout

```
C:\prx\
├── tsim\                          # bloqade-tsim repo
├── accel-cutting-magic-state\     # this repo
└── gidney-circuits\               # Gidney circuit files
    └── circuits\
        └── for_perfectionist_decoding\
            └── c=inject[unitary]+cultivate,p=*.stim
```

## Install tsim

```powershell
cd C:\prx\tsim
pip install -e .
```

## Environment Variables

Set `D5_CIRCUIT_DIR` if circuits are NOT at the default sibling location:

```powershell
# Only needed if gidney-circuits is not at C:\prx\gidney-circuits
$env:D5_CIRCUIT_DIR = "C:\path\to\gidney-circuits\circuits\for_perfectionist_decoding"
```

JAX is configured to use CUDA by default in `run_d5_paper.py`. To override:

```powershell
$env:JAX_PLATFORMS = "cpu"   # Force CPU mode
```

## Quick Smoke Test (CPU)

```powershell
cd C:\prx\accel-cutting-magic-state
$env:JAX_PLATFORMS = "cpu"
python run_d5_paper.py --p 0.001 --shots 5000 --batch-size 1000
```

Expected: ~900 surviving shots, LER ~ 0.008, completes in ~10s.

## GPU Run — Single Noise Level

```powershell
cd C:\prx\accel-cutting-magic-state
python run_d5_paper.py --p 0.001 --shots 600000 --batch-size 10000 --eval-batch 1024 --resume
```

## GPU Run — All 4 Noise Levels

Run each in a separate terminal or sequentially:

```powershell
python run_d5_paper.py --p 0.0002 --shots 500000  --batch-size 10000 --eval-batch 1024 --resume
python run_d5_paper.py --p 0.0005 --shots 500000  --batch-size 10000 --eval-batch 1024 --resume
python run_d5_paper.py --p 0.001  --shots 600000  --batch-size 10000 --eval-batch 1024 --resume
python run_d5_paper.py --p 0.002  --shots 2000000 --batch-size 10000 --eval-batch 1024 --resume
```

## Output

Each run produces `d5_results_{p}.jsonl` — one JSON line per batch, append-mode.

Use `--resume` to continue from where you left off after any interruption.

## Collect Results

After all runs complete:

```powershell
python collect_results.py
```

This reads all `d5_results_*.jsonl` files and produces a summary table with
PSR, LER, and 95% Wilson score confidence intervals.

## Shot Plan

| p      | Planned Shots | Expected PSR | Expected LER | Expected Rel Width |
|--------|--------------|-------------|-------------|-------------------|
| 0.0002 | 500,000      | 0.72        | 2.8e-3      | ~12%              |
| 0.0005 | 500,000      | 0.44        | 5.9e-3      | ~11%              |
| 0.001  | 600,000      | 0.19        | 7.4e-3      | ~14%              |
| 0.002  | 2,000,000    | 0.035       | 2.6e-2      | ~9%               |

## Troubleshooting

- **JAX not finding GPU**: Ensure `jax[cuda12]` is installed and CUDA is visible.
  Check with `python -c "import jax; print(jax.devices())"`.
- **Circuit not found**: Set `D5_CIRCUIT_DIR` to the correct path.
- **p=0.0002 circuit missing**: It will be auto-created from the p=0.001 template
  on first run.
