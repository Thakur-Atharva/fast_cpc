# Fast CPC & Causal Discovery CLI

A high-performance, command-line-driven causal discovery suite tailored for high-dimensional manufacturing routing datasets. This repository provides a unified interface to run multiple causal discovery algorithms—**C-PC, k-PC, FCI, and GES**—while enforcing temporal physical constraints and performing network complexity reduction.

This project is a fork of the original [kenneth-lee-ch/cpc](https://github.com/kenneth-lee-ch/cpc) repository (by Kenneth Lee, Bruno Ribeiro, and Murat Kocaoglu). We have optimized the core execution pathways, introduced multi-core CPU parallelization, and built a standardized CLI pipeline.

---

## Key Features

- **Unified CLI (`discover.py`)**: Run and compare CPC, k-PC, FCI, and GES using standard command-line flags.
- **CPU Parallelization**: The C-PC algorithm uses a chunked task-parallel strategy via `joblib` and integer index pre-mapping to speed up conditional independence (CI) tests on high-core systems.
- **Automated Hub Analysis**: Dynamically identifies routing sequence "hubs" based on presence rates and entropy to form conditionally closed sets.
- **Chronological Temporal Filtering**: Discards causal edges that violate physical time order (i.e., downstream processes causing upstream processes).
- **Transitive Complexity Reduction**: Simplifies discovered graphs by removing transitive bypass connections using NetworkX transitive reduction.
- **Standardized Outputs**: Decodes PAG/CPDAG edge markings (directed `-->`, bidirected `<->`, possibly causal `o->`, undirected `---`, etc.) and formats them into a clean terminal table.
- **Visual Graph Generation**: Generates high-quality chronological layout DAG plots for each run.

---

## Installation

Ensure you are in the project root directory. We recommend using `uv` or standard `virtualenv`:

```bash
# Create and activate virtual environment
uv venv cpc
source cpc/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Quick Start Pipeline

### 1. Generate Fake/Synthetic Data
To generate a synthetic manufacturing dataset (`fake_data.csv`) with 6,000 lots, 800 stages, and 5 known ground-truth dependency chains, run:

```bash
python data_generate.py
```

### 2. Run C-PC (Conditional PC)
C-PC is optimized for high-dimensional structures and uses automated hub analysis.
- **Note**: The Bonferroni-corrected significance threshold (`--alpha 1e-10`) is recommended for the multiple-testing problem with 800 variables.

```bash
python discover.py --cpc --data fake_data.csv --alpha 1e-10 --max-hubs 30 --n-jobs -1
```

### 3. Run k-PC (PC restricted to depth k)
k-PC runs skeleton learning restricted to conditioning sets of size at most `k`.
- **Note**: Using `--k 1` with `--fast-adj` is recommended for high dimensions to keep the CI search space tractable.

```bash
python discover.py --kpc --data fake_data.csv --k 1 --alpha 1e-10
```

### 4. Run FCI (Fast Causal Inference)
FCI allows for latent (unobserved) confounders and outputs a Partial Ancestral Graph (PAG).

```bash
python discover.py --fci --data fake_data.csv --depth 1 --alpha 1e-10
```

### 5. Run GES (Greedy Equivalence Search)
GES is a score-based algorithm that optimizes a likelihood score.
- **Note**: For discrete/binary manufacturing routing data, the Bayesian Dirichlet equivalent uniform (`local_score_BDeu`) score is used.
- *Caution*: Scoring 800 columns sequentially can be computationally expensive; you may want to limit the search space or run it on a subset of variables.

```bash
python discover.py --ges --data fake_data.csv --score-func local_score_BDeu
```

---

## CLI Argument Reference

| Flag | Type | Description | Default |
| :--- | :--- | :--- | :--- |
| **Algorithm Selectors** | | *(Select exactly one)* | |
| `--cpc` | Flag | Run C-PC structural learning loops | Required |
| `--kpc` | Flag | Run k-PC structural learning | Required |
| `--fci` | Flag | Run FCI algorithm | Required |
| `--ges` | Flag | Run Greedy Equivalence Search | Required |
| **Shared Parameters** | | | |
| `--data` | Path | Path to CSV data file | `fake_data.csv` |
| `--alpha` | Float | CI test significance level (p-value threshold) | `1e-10` (CPC), `0.05` (Others) |
| `--tester` | Str | Independence test type (`chisq`, `fisherz`, `gsq`) | `chisq` |
| `--no-temporal` | Flag | Disable chronological filtering | `False` |
| `--no-transitive-reduction` | Flag | Disable transitive reduction post-processing | `False` |
| `--output` | Path | Filename to save visualization PNG | `<algo>_manufacturing_dag.png` |
| **Algorithm Specifics** | | | |
| `--max-hubs` | Int | [CPC] Max number of conditioning hubs to use | `30` |
| `--k` | Int | [k-PC] Max size of separating sets | `1` |
| `--fast-adj` | Flag | [k-PC] Enable fast adjacency search | `True` |
| `--depth` | Int | [FCI] Max conditioning set depth (-1 for unlimited) | `1` |
| `--score-func` | Str | [GES] Scoring function (`local_score_BDeu`, `local_score_BIC`) | `local_score_BDeu` |

---

## Parallel Performance & Optimization

**CPC, k-PC, and FCI** all feature heavy computational optimization and parallel scaling for multi-core systems:
1. **Pre-Mapped Memory**: String-based conditioning sets and columns are mapped once to integer arrays, avoiding millions of redundant lookup/allocation operations.
2. **Chunked Task Allocation**: Groups CI statement pairs or edge tests into chunks and evaluates them concurrently in parallel blocks (`n_jobs`), minimizing schedule/IPC overhead.
3. **Parallel Stable Skeleton Search**: Under stable skeleton discovery (default), CI tests at each depth are independent and are evaluated fully in parallel using multi-core worker processes.

### Example Performance (800 variables, 6000 rows):
- **C-PC** (30 hubs):
  - Sequential: `41.2 seconds`
  - Parallel (48 cores): `7.2 seconds` *(~5.7x Speedup)*
- **k-PC** (k=1):
  - Sequential: `22.0 seconds`
  - Parallel (48 cores): `7.2 seconds` *(~3.0x Speedup)*

---

## Citation & Credits

This code contains wrappers and modified versions of the algorithms from **causal-learn**. 

If you use CPC in your research, please cite:
```bibtex
@inproceedings{lee2025cpc,
  title={Constraint-based Causal Discovery from a Collection of Conditioning Sets},
  author={Lee, Kenneth and Ribeiro, Bruno and Kocaoglu, Murat},
  booktitle={Proceedings of the 41st Conference on Uncertainty in Artificial Intelligence (UAI)},
  year={2025}
}
```