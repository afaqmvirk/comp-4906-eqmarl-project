# COMP 4906: eQMARL

Communication-noise experiments extending
[DeRieux and Saad's eQMARL](https://github.com/news-vt/eqmarl).

## Structure

| Path | Contents |
|---|---|
| `upstream/eqmarl/` | Original implementation used by the noise study |
| `upstream/eqmarl-paper-iclr2025/` | Original paper-tagged implementation |
| `src/eqmarl_network_study/learning_noise/` | Critics, training, and noise analysis |
| `src/eqmarl_network_study/network/`, `accounting/` | Network timing and resource model |
| `src/eqmarl_network_study/analysis/`, `experiments/` | Learning-curve utilities and network workflows |
| `scripts/` | Experiment runners and analysis commands |
| `configs/` | Parameters and seed grids |
| `tests/` | Numerical and training checks |
| `results/study2_exact/` | Exact-noise results, weights, and provenance |
| `results/study2/raw/`, `results/feasibility/` | Reused baselines and precision controls |
| `results/paper_reference/` | Fresh official-paper reproduction |
| `results/review/` | Reward-20/25 comparisons and learning curves |
| `results/raw/`, `results/summaries/`, `results/figures/` | Network-study outputs |
| `proposal/` | Preproposal source and PDF |

Upstream code is pinned in unmodified Git submodules under the authors'
[CC BY 4.0 license](upstream/eqmarl/LICENSE.md). Our additions are in `src/`
and `scripts/`, including adaptations of the original MAA2C update.

## Running

Clone with `--recurse-submodules`. In a Python 3.12 virtual environment:

```sh
python -m pip install -e ".[dev]"
python scripts/analyze_results.py --check
python -m pytest -q
```

Use `analyze_results.py --write` to rebuild the summary and plot.
`run_study2_exact.py` runs the noise grid; `run_paper_reference.py` runs the
paper reproduction. Training requires Linux/WSL and Python 3.9; use
`scripts/setup_study2_wsl.sh` or `scripts/setup_paper_reference_wsl.sh`.
