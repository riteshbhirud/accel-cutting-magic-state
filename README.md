# Accelerated Exact Sampling for Magic State Cultivation

Companion code for "Simulating magic state cultivation with few Clifford terms (arXiv:2509.08658)".

An optimised sampling pipeline for the $d=3$ T-gate magic state cultivation circuit, built on top of [`bloqade-tsim==0.1.0`](https://github.com/QuEraComputing/tsim). Achieves >4 million exact (non-Clifford) shots per second at low noise ($p \leq 5 \times 10^{-4}$) on an Apple M4 Macbook Pro laptop, within ~1.1x of a fully Clifford proxy via `stim`.

## Optimisations

1. **Sparse geometric channel sampling** (`channel_sampler_fast.py`): At low noise, most channels produce the identity outcome. Instead of sampling every channel per shot, uses `Geom(p_fire)` to skip directly to non-identity events via cumulative sums. Reduces random draws from ~248M to ~500K per batch at $p = 5 \times 10^{-4}$.

2. **Unique-pattern deduplication with persistent cross-batch cache** (`sampler_dedup.py`): Packs noise-channel f-parameters into `int64` hashes via BLAS matmul, deduplicates with `np.unique`, and evaluates only unique patterns. A persistent dictionary caches results across all batches, reducing total evaluations from ~16K to ~1K across an entire run.

3. **Scan-based combo evaluation** (`evaluate_matmul_cfloat.py`): Replaces the JIT-unrolled Python for-loop over 32 measurement-outcome combos with `jax.lax.scan`, compiling a single loop body instead of 32 copies. Includes D-term lookup table precomputation and split f/m rowsums.

4. **Noiseless outcome caching** (`sampler_noiseless_cache.py`): Two-pass architecture that separates noiseless sampling (NumPy PCG64, column-by-column) from noisy evaluation (JAX/XLA). Noiseless samples are written directly into a pre-allocated array, avoiding redundant allocation.

## File Structure

| File | Description |
|------|-------------|
| `run.py` | Entry point and benchmark runner |
| `d_3_circuit_definitions.py` | Distance-3 cultivation circuit definitions |
| `tsim_cutting.py` | Cutting decomposition, subcomponent compilation, enumeration-based sampling |
| `evaluate_matmul_cfloat.py` | Complex64 BLAS evaluation with `jax.lax.scan` combo loop |
| `sampler_dedup.py` | Hash-based deduplication with persistent cross-batch cache |
| `channel_sampler_fast.py` | Sparse geometric-skip channel sampler |
| `sampler_noiseless_cache.py` | Two-pass noiseless/noisy sampling with NumPy caching |
| `stab_rank_cut.py` | ZX-graph spider cutting primitives |
| `gen/` | Noise model from [magic-state-cultivation](https://github.com/Strilanc/magic-state-cultivation) (Apache 2.0) |
| `pipeline_stages.ipynb` | Notebook illustrating pipeline stages |
| `plot_results.ipynb` | Plotting benchmark results |
| `benchmark_stim.py` | Stim Clifford-proxy throughput reference |
| `results.jsonl` | Pre-computed benchmark data |

## Requirements

```
pip install -r requirements.txt
```

Dependencies: `bloqade-tsim==0.1.0`, `stim`, `jax`, `jaxlib`, `numpy`, `equinox`, `pyzx_param==0.9.2`

## Usage

```bash
python run.py
```

Runs the sampling benchmark across noise strengths and appends results to `results.jsonl`.

## Results

All benchmarks on Apple M4 Pro, batch size 2M. Throughput is total shots / wall-clock time (excluding JIT warmup). Results were collected across multiple experimental runs with different seeds (recorded per entry in `results.jsonl`). See `benchmark_stim.py` for the stim Clifford-proxy reference.

| $p$ | Shots/s | PSR (post-selection rate) | Fidelity |
|-----|---------|-----|----------|
| 0.0002 | 5,261,675 | 0.918 | 0.999999982 |
| 0.0005 | 4,102,367 | 0.807 | 0.999999821 |
| 0.001 | 2,777,295 | 0.651 | 0.999999 |
| 0.002 | 1,580,499 | 0.424 | 0.999993 |
| 0.005 | 592,605 | 0.118 | 0.999902 |

## Attribution

If you wish to cite this in an academic work, please cite the [accompanying paper](https://arxiv.org/abs/2509.08658):
<pre>
  @misc{wan2026simulatingmagicstatecultivation,
      title={Simulating magic state cultivation with few Clifford terms}, 
      author={Kwok Ho Wan and Zhenghao Zhong},
      year={2026},
      eprint={2509.08658},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      url={https://arxiv.org/abs/2509.08658}, 
}
</pre>
and or this repository:
<pre>
@software{Wan2026accel,
  title={Accelerated Exact Sampling for Magic State Cultivation},
  author={Wan, Kwok Ho and Zhong, Zhenghao and ., Ainhoa},
  url={https://github.com/kh428/accel-cutting-magic-state},
  year={2026},
  note = {Apache-2.0 License},
}
</pre>
  
## Acknowledgements

Parts of this codebase were developed with the assistance of LLMs.
