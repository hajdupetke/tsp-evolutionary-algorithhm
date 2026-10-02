"""experiments.py — Full experiment suite.

Runs the 5 required experiments and writes ``results/results.txt``.

Interface notes (important):
  * ``src/ga.py`` (Coder 2) exposes ``run_ga(D, params, rng, init_fn, cost_fn)``
    where ``rng`` is a ``random.Random`` instance, ``init_fn(n, p, rng)``
    builds the initial population and ``cost_fn(tour, D)`` evaluates a tour.
  * ``src/initialization.py`` (Coder 1) exposes the heuristics plus
    ``rng``-aware population builders: ``random_population(p, n, rng)`` and
    ``smart_population(cities, p, rng, max_iter, weight_type)``. The 'nn' and 'angle'
    variants below reuse the same deterministic single-tour heuristics
    (NN + 2-opt, angle sort) with all randomness drawn from ``rng``.

Experiments (default datasets; override via run_all_experiments(dataset=..., scaling_datasets=...)):
  1. Initialization comparison: random / nn / angle / smart (default berlin52)
  2. Problem size scaling: burma14 / berlin52 / kroA100 / kroA200 (smart)
  3. Population size: 50 / 100 / 150 / 200 (default berlin52, smart)
  4. Mutation rate: 0.05 / 0.1 / 0.2 / 0.4 (default berlin52, smart)
  5. Replacement rate: 0.3 / 0.5 / 0.7 / 0.9 (default berlin52, smart)

Each configuration is repeated ``runs`` times with seeds
``base_seed + run_index`` (plus a per-experiment offset) so results are
fully reproducible via ``main.py --seed``.
"""

import random
import time
from pathlib import Path

from src.utils import load_cities, load_instance, calculate_cost
from src.initialization import (
    nearest_neighbour_tour,
    two_opt_improvement,
    angle_based_tour,
    random_population,
    smart_population,
    DIVERSITY_SWAPS,
)
from src.ga import run_ga

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_FILE = RESULTS_DIR / "results.txt"

KNOWN_OPTIMALS = {
    # TSPLIB symmetric-TSP optima — verified 2026-10-01, values unchanged
    # (previous values were already correct).
    # Primary source: Reinelt (Heidelberg), "Optimal solutions for symmetric TSPs":
    #   original URL http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/STSP.html
    #   (now 404; mirror: https://www.iwr.uni-heidelberg.de/groups/comopt/software/TSPLIB95/STSP.html)
    #   lists: burma14: 3323, berlin52: 7542, kroA100: 21282, kroA200: 29368.
    # Cross-checks:
    #   - berlin52 = 7542 (e.g. Clever Algorithms ILS page; Engelbrecht/ACO gist
    #     citing the Heidelberg STSP list; teeline `solve berlin52` -> tour 7542 gap 0.0%).
    #   - kroA100 = 21282, kroA200 = 29368 (Heidelberg STSP list excerpt;
    #     kroA100=21282 also in ResearchGate BBO table "KroA100 optimal = 21,282").
    #   - burma14 = 3323 (Heidelberg STSP list; arxiv:2606.09529 "burma14 ... known
    #     optimum 3,323"; arxiv:2106.05948 "optimal tour length of 3323 [1]"; GEO distances).
    # Comparable only with TSPLIB edge weights (utils.calculate_cost):
    # EUC_2D for berlin52/kroA100/kroA200, GEO for burma14.
    "burma14": 3323,
    "berlin52": 7542,
    "kroA100": 21282,
    "kroA200": 29368,
}

DATASETS = {
    "burma14": "burma14.tsp",
    "berlin52": "berlin52.tsp",
    "kroA100": "kroA100.tsp",
    "kroA200": "kroA200.tsp",
}

INIT_METHODS = ("random", "nn", "angle", "smart")

DEFAULT_PARAMS = {
    "p": 100,       # population size
    "r": 0.7,       # replacement rate (fraction replaced by crossover)
    "m": 0.1,       # mutation rate
    "G": 500,       # generations
    "k": 5,         # tournament size
    "mutation": "swap",
}


# ---------------------------------------------------------------------------
# Init-function adapters
# ---------------------------------------------------------------------------

def make_init_fn(cities, method, weight_type="EUC_2D"):
    """Build an ``init_fn(n, p, rng)`` closure for ``run_ga``.

    :param cities: list of (x, y) coordinates (captured by the closure)
    :param method: 'random' | 'nn' | 'angle' | 'smart'
      - 'random': p random permutations (via rng.shuffle)
      - 'nn': NN + 2-opt from rotating start cities (Liu, 2014)
      - 'angle': angle-based base tour + a few rng swaps (Liao et al., 2012)
      - 'smart': half NN+2-opt, half angle-based + swaps (report Sec. 2.3)
    :param weight_type: TSPLIB EDGE_WEIGHT_TYPE of `cities`.
    2-opt always runs to convergence (delta evaluation makes the old
    iteration cap unnecessary).
    """
    if method not in INIT_METHODS:
        raise ValueError(f"Unknown init_method={method!r}, expected one of {INIT_METHODS}")

    n = len(cities)

    def _diversify(base, rng):
        tour = base[:]
        for _ in range(DIVERSITY_SWAPS):
            i, j = rng.sample(range(len(tour)), 2)
            tour[i], tour[j] = tour[j], tour[i]
        return tour

    def init_fn(n_arg, p, rng):
        del n_arg  # population is always built for the captured cities
        if method == "random":
            return random_population(p, n, rng)
        if method == "smart":
            return smart_population(cities, p, rng=rng, weight_type=weight_type)
        population = []
        if method == "nn":
            for i in range(p):
                tour = nearest_neighbour_tour(cities, start_city=i % n, weight_type=weight_type)
                tour = two_opt_improvement(tour, cities, weight_type=weight_type)
                population.append(tour)
        elif method == "angle":
            base = angle_based_tour(cities)
            for _ in range(p):
                population.append(_diversify(base, rng))
        return population

    return init_fn


# ---------------------------------------------------------------------------
# Single-run helper
# ---------------------------------------------------------------------------

def _log(msg):
    print(msg, flush=True)


def run_single(cities, init_method, params, seed, verbose=False, log_interval=100,
               weight_type="EUC_2D"):
    """Run one GA trial with a dedicated seed.

    :return: dict with best_tour, best_cost, history_best, history_gen,
             seconds, seed, init_method
    """
    rng = random.Random(seed)
    init_fn = make_init_fn(cities, init_method, weight_type)

    def cost_fn(tour, D):
        return calculate_cost(tour, D, weight_type)

    t0 = time.time()
    if verbose:
        _log(f"    trial start: init={init_method} seed={seed} p={params['p']} G={params['G']} ({weight_type})")
    result = run_ga(cities, dict(params), rng, init_fn, cost_fn,
                    verbose=verbose, log_interval=log_interval)
    # run_ga already measures seconds internally; keep wall time as fallback
    result.setdefault("seconds", time.time() - t0)
    result["seed"] = seed
    result["init_method"] = init_method
    result["params"] = dict(params)
    return result


# ---------------------------------------------------------------------------
# The 5 experiments
# ---------------------------------------------------------------------------

def experiment_init_comparison(cities, dataset_name="berlin52",
                               params=None, runs=10, base_seed=42, verbose=True,
                               weight_type="EUC_2D"):
    """Experiment 1 — compare init methods on one dataset."""
    params = dict(DEFAULT_PARAMS if params is None else params)
    rows = []
    histories = {}
    total = len(INIT_METHODS) * runs
    done = 0
    t0 = time.time()
    for method in INIT_METHODS:
        histories[method] = []
        for run in range(runs):
            seed = base_seed + run  # same seeds across methods => fair pairing
            done += 1
            if verbose:
                _log(f"  [exp1 {done}/{total}] init={method} run={run+1}/{runs} seed={seed} ...")
            res = run_single(cities, method, params, seed, weight_type=weight_type)
            rows.append({
                "dataset": dataset_name,
                "init_method": method,
                "run": run + 1,
                "seed": seed,
                "best_cost": res["best_cost"],
                "time_seconds": res["seconds"],
            })
            if verbose:
                _log(f"  [exp1 {done}/{total}] done: best={res['best_cost']:.1f} {res['seconds']:.1f}s elapsed={time.time()-t0:.0f}s")
            histories[method].append(res["history_best"])
    return rows, histories


def experiment_scaling(datasets=None, params=None, runs=10, base_seed=1000, verbose=True):
    """Experiment 2 (MANDATORY) — scaling across problem sizes, init='smart'."""
    params = dict(DEFAULT_PARAMS if params is None else params)
    datasets = datasets or list(DATASETS.keys())
    rows = []
    histories = {}
    total = len(datasets) * runs
    done = 0
    t0 = time.time()
    for idx, name in enumerate(datasets):
        cities, weight_type = load_instance(str(DATA_DIR / DATASETS[name]))
        histories[name] = []
        optimal = KNOWN_OPTIMALS.get(name, float("nan"))
        for run in range(runs):
            seed = base_seed + idx * 1000 + run
            done += 1
            if verbose:
                _log(f"  [exp2 {done}/{total}] dataset={name} run={run+1}/{runs} ...")
            res = run_single(cities, "smart", params, seed, weight_type=weight_type)
            gap = (res["best_cost"] - optimal) / optimal * 100 if optimal else float("nan")
            rows.append({
                "dataset": name,
                "num_cities": len(cities),
                "run": run + 1,
                "seed": seed,
                "best_cost": res["best_cost"],
                "known_optimal": optimal,
                "gap_percent": gap,
                "time_seconds": res["seconds"],
            })
            if verbose:
                _log(f"  [exp2 {done}/{total}] done: best={res['best_cost']:.1f} {res['seconds']:.1f}s elapsed={time.time()-t0:.0f}s")
            histories[name].append(res["history_best"])
    return rows, histories


def experiment_population_size(cities, dataset_name="berlin52",
                               values=(50, 100, 150, 200),
                               params=None, runs=10, base_seed=2000, verbose=True,
                               weight_type="EUC_2D"):
    """Experiment 3 — vary population size p."""
    params = dict(DEFAULT_PARAMS if params is None else params)
    rows = []
    histories = {}
    total = len(values) * runs
    done = 0
    t0 = time.time()
    for vi, p in enumerate(values):
        cfg = dict(params, p=p)
        histories[p] = []
        for run in range(runs):
            seed = base_seed + vi * 1000 + run
            done += 1
            if verbose:
                _log(f"  [exp3 {done}/{total}] p={p} run={run+1}/{runs} ...")
            res = run_single(cities, "smart", cfg, seed, weight_type=weight_type)
            rows.append({
                "dataset": dataset_name, "p": p, "run": run + 1,
                "seed": seed, "best_cost": res["best_cost"],
                "time_seconds": res["seconds"],
            })
            if verbose:
                _log(f"  [exp3 {done}/{total}] done: best={res['best_cost']:.1f} {res['seconds']:.1f}s elapsed={time.time()-t0:.0f}s")
            histories[p].append(res["history_best"])
    return rows, histories


def experiment_mutation_rate(cities, dataset_name="berlin52",
                             values=(0.05, 0.1, 0.2, 0.4),
                             params=None, runs=10, base_seed=3000, verbose=True,
                             weight_type="EUC_2D"):
    """Experiment 4 — vary mutation rate m."""
    params = dict(DEFAULT_PARAMS if params is None else params)
    rows = []
    histories = {}
    total = len(values) * runs
    done = 0
    t0 = time.time()
    for vi, m in enumerate(values):
        cfg = dict(params, m=m)
        histories[m] = []
        for run in range(runs):
            seed = base_seed + vi * 1000 + run
            done += 1
            if verbose:
                _log(f"  [exp4 {done}/{total}] m={m} run={run+1}/{runs} ...")
            res = run_single(cities, "smart", cfg, seed, weight_type=weight_type)
            rows.append({
                "dataset": dataset_name, "m": m, "run": run + 1,
                "seed": seed, "best_cost": res["best_cost"],
                "time_seconds": res["seconds"],
            })
            if verbose:
                _log(f"  [exp4 {done}/{total}] done: best={res['best_cost']:.1f} {res['seconds']:.1f}s elapsed={time.time()-t0:.0f}s")
            histories[m].append(res["history_best"])
    return rows, histories


def experiment_replacement_rate(cities, dataset_name="berlin52",
                                values=(0.3, 0.5, 0.7, 0.9),
                                params=None, runs=10, base_seed=4000, verbose=True,
                                weight_type="EUC_2D"):
    """Experiment 5 — vary replacement rate r."""
    params = dict(DEFAULT_PARAMS if params is None else params)
    rows = []
    histories = {}
    total = len(values) * runs
    done = 0
    t0 = time.time()
    for vi, r in enumerate(values):
        cfg = dict(params, r=r)
        histories[r] = []
        for run in range(runs):
            seed = base_seed + vi * 1000 + run
            done += 1
            if verbose:
                _log(f"  [exp5 {done}/{total}] r={r} run={run+1}/{runs} ...")
            res = run_single(cities, "smart", cfg, seed, weight_type=weight_type)
            rows.append({
                "dataset": dataset_name, "r": r, "run": run + 1,
                "seed": seed, "best_cost": res["best_cost"],
                "time_seconds": res["seconds"],
            })
            if verbose:
                _log(f"  [exp5 {done}/{total}] done: best={res['best_cost']:.1f} {res['seconds']:.1f}s elapsed={time.time()-t0:.0f}s")
            histories[r].append(res["history_best"])
    return rows, histories


# ---------------------------------------------------------------------------
# Results file
# ---------------------------------------------------------------------------

def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def _write_results(path, sections):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for title, header, lines in sections:
            f.write(title + "\n")
            f.write(header + "\n")
            for line in lines:
                f.write(line + "\n")
            f.write("\n")


def run_all_experiments(base_seed=42, runs=10, generations=500,
                        population_size=100, results_path=None,
                        dataset="berlin52", scaling_datasets=None):
    """Run all 5 experiments and write ``results/results.txt``.

    :param base_seed: master seed; every trial derives its own seed from it.
    :param runs: repetitions per configuration (README requires 10).
    :param generations: override G for all experiments (README: 500).
    :param population_size: override default p (README: 100).
    :param dataset: dataset key from DATASETS used for exp1/3/4/5
        (single-dataset experiments). Default "berlin52" (eski davranış).
    :param scaling_datasets: dataset key list for exp2 (scaling).
        None => all datasets in DATASETS (eski davranış).
    :return: dict with row-lists and histories per experiment (for plotting).
    """
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset={dataset!r}, expected one of {sorted(DATASETS)}")
    if scaling_datasets is None:
        scaling_datasets = list(DATASETS.keys())
    else:
        scaling_datasets = list(scaling_datasets)
        unknown = [d for d in scaling_datasets if d not in DATASETS]
        if unknown:
            raise ValueError(f"Unknown scaling_datasets={unknown!r}, expected subset of {sorted(DATASETS)}")
        if not scaling_datasets:
            raise ValueError("scaling_datasets must not be empty")
    results_path = Path(results_path) if results_path else RESULTS_FILE
    base_params = dict(DEFAULT_PARAMS, p=population_size, G=generations)

    cities, weight_type = load_instance(str(DATA_DIR / DATASETS[dataset]))
    n_trials = (len(INIT_METHODS) + len(scaling_datasets) + 4 + 4 + 4) * runs
    _log(f"[suite] {n_trials} GA trials total (runs={runs}, G={generations}, p={population_size}, "
         f"dataset={dataset}, scaling={scaling_datasets}). This takes a while with smart init — progress below.")
    tall0 = time.time()

    _log(f"[exp1] init comparison on {dataset} (runs={runs}, G={generations}) ...")
    exp1_rows, exp1_hist = experiment_init_comparison(
        cities, dataset_name=dataset, params=base_params, runs=runs, base_seed=base_seed,
        weight_type=weight_type)
    _log(f"[exp1] done in {time.time()-tall0:.0f}s")

    _log(f"[exp2] problem size scaling on {scaling_datasets} ...")
    exp2_rows, exp2_hist = experiment_scaling(
        datasets=scaling_datasets, params=base_params, runs=runs, base_seed=base_seed + 1000)
    _log(f"[exp2] done, suite elapsed={time.time()-tall0:.0f}s")

    _log(f"[exp3] population size on {dataset} ...")
    exp3_rows, exp3_hist = experiment_population_size(
        cities, dataset_name=dataset, params=base_params, runs=runs, base_seed=base_seed + 2000,
        weight_type=weight_type)
    _log(f"[exp3] done, suite elapsed={time.time()-tall0:.0f}s")

    _log(f"[exp4] mutation rate on {dataset} ...")
    exp4_rows, exp4_hist = experiment_mutation_rate(
        cities, dataset_name=dataset, params=base_params, runs=runs, base_seed=base_seed + 3000,
        weight_type=weight_type)
    _log(f"[exp4] done, suite elapsed={time.time()-tall0:.0f}s")

    _log(f"[exp5] replacement rate on {dataset} ...")
    exp5_rows, exp5_hist = experiment_replacement_rate(
        cities, dataset_name=dataset, params=base_params, runs=runs, base_seed=base_seed + 4000,
        weight_type=weight_type)
    _log(f"[exp5] done, suite elapsed={time.time()-tall0:.0f}s")

    sections = [
        ("EXPERIMENT 1: Initialization Comparison",
         "dataset,init_method,run,seed,best_cost,time_seconds",
         [f"{r['dataset']},{r['init_method']},{r['run']},{r['seed']},"
          f"{r['best_cost']:.2f},{r['time_seconds']:.2f}" for r in exp1_rows]),
        ("EXPERIMENT 2: Problem Size Scaling",
         "dataset,num_cities,run,seed,best_cost,known_optimal,gap_percent,time_seconds",
         [f"{r['dataset']},{r['num_cities']},{r['run']},{r['seed']},"
          f"{r['best_cost']:.2f},{r['known_optimal']},"
          f"{r['gap_percent']:.2f},{r['time_seconds']:.2f}" for r in exp2_rows]),
        ("EXPERIMENT 3: Population Size",
         "dataset,p,run,seed,best_cost,time_seconds",
         [f"{r['dataset']},{r['p']},{r['run']},{r['seed']},"
          f"{r['best_cost']:.2f},{r['time_seconds']:.2f}" for r in exp3_rows]),
        ("EXPERIMENT 4: Mutation Rate",
         "dataset,m,run,seed,best_cost,time_seconds",
         [f"{r['dataset']},{r['m']},{r['run']},{r['seed']},"
          f"{r['best_cost']:.2f},{r['time_seconds']:.2f}" for r in exp4_rows]),
        ("EXPERIMENT 5: Replacement Rate",
         "dataset,r,run,seed,best_cost,time_seconds",
         [f"{r['dataset']},{r['r']},{r['run']},{r['seed']},"
          f"{r['best_cost']:.2f},{r['time_seconds']:.2f}" for r in exp5_rows]),
    ]
    _write_results(results_path, sections)
    print(f"Results written to {results_path}")

    # Console summary (mean best cost per config)
    def summarize(rows, key):
        cfgs = sorted({r[key] for r in rows})
        for c in cfgs:
            vals = [r["best_cost"] for r in rows if r[key] == c]
            print(f"  {key}={c}: mean={_mean(vals):.2f} best={min(vals):.2f}")

    print("Summary exp1 (init):"); summarize(exp1_rows, "init_method")
    print("Summary exp2 (dataset):"); summarize(exp2_rows, "dataset")
    print("Summary exp3 (p):"); summarize(exp3_rows, "p")
    print("Summary exp4 (m):"); summarize(exp4_rows, "m")
    print("Summary exp5 (r):"); summarize(exp5_rows, "r")

    return {
        "exp1": (exp1_rows, exp1_hist),
        "exp2": (exp2_rows, exp2_hist),
        "exp3": (exp3_rows, exp3_hist),
        "exp4": (exp4_rows, exp4_hist),
        "exp5": (exp5_rows, exp5_hist),
        "results_path": str(results_path),
    }
