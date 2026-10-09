# TSP Evolutionary Algorithm — TARI29 Group 9

Genetic Algorithm (GA) for the **Travelling Salesman Problem (TSP)**,
implemented in pure Python (stdlib + matplotlib).

Target benchmark: **berlin52** (52 cities, known optimum **7542**).
Example result with `--init-method nn`: **7542.0** (gap 0.00%, TSPLIB metric).

Group members: 
- Christos Zampounis
- Mehmet Cihan Yalçın
- Péter Dávid Hajdu
- Pierre Kasparian
- Prajval Parthiban

## Algorithm

Mitchell (1997) GA template adapted to TSP permutation representation:

| Component       | Choice                        | Source            |
| --------------- | ----------------------------- | ----------------- |
| Representation  | Permutation of city indices   | Standard          |
| Initialization  | NN + 2-opt AND angle-based    | Liu (2014), Liao et al. (2012) |
| Fitness         | F(π) = 1 / C(π)               | Report Eq. (5)    |
| Selection       | Tournament selection (k=5)    | Report Section 3  |
| Crossover       | Order Crossover (OX)          | Hussain et al. (2017) |
| Mutation        | Inversion mutation            | Report Section 3  |
| Survivor policy | Tournament selection (k=5)    | Algorithm 1       |

Default parameters: `p=100, r=0.7, m=0.1, G=500, k=5, mutation='inversion'`, init `random`.

## Repository structure

```
.
├── main.py                 ← entry point: python main.py
├── src/
│   ├── utils.py            ← TSPLIB loading, distance, cost, fitness
│   ├── initialization.py   ← random, NN+2-opt, angle-based, smart init
│   ├── operators.py        ← tournament selection, OX, swap/inversion mutation
│   ├── ga.py               ← main GA loop (Algorithm 1) + random-search baseline
│   ├── experiments.py      ← all 5 experiments, writes results/results.txt
│   └── plot.py             ← convergence + tour + scaling figures
├── data/                   ← the 6 .tsp instances, committed
├── scripts/
│   └── download_data.py      ← fetches the 6 .tsp files into data/ (stdlib only)
├── results/                ← results.txt + .png figures (generated)
├── tests/ + src/tests/     ← pytest suite (44 tests)
└── README.md
```

## Requirements

- Python 3.12+
- `matplotlib` (only external dependency)

```bash
pip install matplotlib
# or with uv:
uv sync
```

## Setup: datasets

The `.tsp` files are committed, so a fresh clone runs as-is. Only needed if one goes missing:

```bash
python scripts/download_data.py              # fetch all 6 datasets into data/
python scripts/download_data.py --force      # re-download even if files exist
python scripts/download_data.py berlin52     # fetch a single dataset
```

The script tries multiple mirrors (GitHub raw mirrors first, official
Heidelberg/ZIB sources as fallback) and validates each file
(`DIMENSION` + `NODE_COORD_SECTION`), so a dead mirror doesn't break setup.
Existing valid files are skipped unless `--force` is given.

## Usage

```bash
python main.py                          # sanity run on berlin52 + full experiment suite
python main.py --quick                  # smoke test (G=20, runs=2)
python main.py --init-only --runs 30    # initialization comparison only; outputs in results/init_comparison_only/
python main.py --skip-experiments       # only the berlin52 sanity run
python main.py --seed 123               # repeatable run with a different master seed
```

| Argument             | Default | Description                              |
| -------------------- | ------- | ---------------------------------------- |
| `--seed`             | 42      | Master seed; every trial derives `Random(seed + offset)` from it |
| `--population-size`  | 100     | Population size `p`                      |
| `--generations`      | 500     | Generations `G`                          |
| `--replacement-rate` | 0.7     | Fraction replaced by crossover `r`       |
| `--mutation-rate`    | 0.1     | Mutation rate `m`                        |
| `--init-method`      | random  | `random` \| `nn` \| `angle` \| `smart`   |
| `--runs`             | 10      | Repetitions per experiment configuration |
| `--skip-experiments` | off     | Skip the full suite                      |
| `--init-only`        | off     | Run only initialization comparisons and save results/plots to a separate folder |
| `--quick`            | off     | Shortcut for `G=20, runs=2`              |

`main.py` loads berlin52, runs the GA, prints result vs. optimum, saves
`convergence_berlin52.png` + `tour_berlin52_best.png`, then runs all
experiments into `results/results.txt` (+ comparison/scaling plots).
A reproducibility self-check re-runs the sanity configuration and asserts
identical output.

## Experiments

| # | Varies | Values | Dataset(s) | Runs |
|---|--------|--------|------------|------|
| 1 | init method | random, nn, angle, smart | berlin52 | 10 |
| 2 | problem size | burma14, berlin52, kroA100, kroA200, pcb442, pr1002 | all, random init | 10 |
| 3 | population `p` | 50, 100, 120, 150 | berlin52 | 10 |
| 4 | mutation `m` | 0.05, 0.1, 0.2, 0.4 | berlin52 | 10 |
| 5 | replacement `r` | 0.3, 0.5, 0.7, 0.9 | berlin52 | 10 |

Known optimals used for gap computation: burma14 3323, berlin52 7542,
kroA100 21282, kroA200 29368, pcb442 50778, pr1002 259045. `results/results.txt` contains one CSV-style
section per experiment (`dataset,...,run,seed,best_cost,...,time_seconds`).

## Tests

```bash
python -m pytest tests src/tests -q
```

Covers data loading, cost/fitness, all initializers, OX validity
(1000 random children), mutation invariants, GA convergence/shape, and
seed reproducibility.

## Distances (TSPLIB-compliant)

- `EUC_2D` (berlin52, kroA100/200): `int(sqrt(dx²+dy²)+0.5)` per edge.
- `GEO` (burma14): great-circle with `RRR=6378.388` per TSPLIB spec.
- `load_instance` reads `EDGE_WEIGHT_TYPE` from the `.tsp` header;
  `calculate_cost(tour, cities, weight_type)` and all init heuristics
  follow it, so gaps vs. `KNOWN_OPTIMALS` (Heidelberg STSP list:
  berlin52 7542, kroA100 21282, kroA200 29368, burma14 3323) are exact.
  `distance()` (raw Euclidean) is kept only for backward compatibility.

## Performance notes (CPU vs GPU)

2-opt uses O(1) delta evaluation against a precomputed `n × n` matrix,
so it runs to convergence with no iteration cap
(`experiments._two_opt_limit` removed). Measured (berlin52, p=100):

- Sanity run `G=20`, smart init: **~0.2 s** (was ~20 s).
- Full `pytest` suite: **~2 s** (was ~20 s).

The remaining bottleneck is the GA loop itself (50k cost evaluations per
`G=500` run), which is sequential, branchy logic — exactly what GPUs
handle worst at these sizes (kernel-launch overhead exceeds the math).
A GPU port (CuPy/Numba CUDA) would mean rewriting `operators.py`, `ga.py`
and `initialization.py`, adding GB-sized CUDA dependencies (breaking the
stdlib-only constraint), and invalidating the injected-`rng`/`cost_fn`
interfaces the test suite relies on — for no measurable gain here.

Cheap next step if the full 10-runs × 5-experiments suite is still slow:
parallelize repetitions with `concurrent.futures` (stdlib) — trials are
embarrassingly parallel, near-linear speed-up on multi-core CPUs.
