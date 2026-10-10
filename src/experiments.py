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
    "pcb442": 50778,
    "pr1002": 259045,
}

DATASETS = {
    "burma14": "burma14.tsp",
    "berlin52": "berlin52.tsp",
    "kroA100": "kroA100.tsp",
    "kroA200": "kroA200.tsp",
    "pcb442": "pcb442.tsp",
    "pr1002": "pr1002.tsp",
}

# Datasets for the per-init parameter sweeps; the larger ones are only used for scaling.
SWEEP_DATASETS = ("burma14", "berlin52", "kroA100", "kroA200")

INIT_METHODS = ("random", "nn", "angle", "smart")

DEFAULT_PARAMS = {
    "p": 100,       # population size
    "r": 0.7,       # replacement rate (fraction replaced by crossover)
    "m": 0.1,       # mutation rate
    "G": 500,       # generations
    "k": 5,         # tournament size
    "mutation": "inversion",  # random-init: ~8.4% vs ~24.8% gap on berlin52
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


def _metrics(dataset_name, res):
    """Result columns shared by every experiment row."""
    optimal = KNOWN_OPTIMALS[dataset_name]
    return {
        "init_method": res["init_method"],
        "best_cost": res["best_cost"],
        "init_best": res["init_best"],
        "gap_percent": (res["best_cost"] - optimal) / optimal * 100,
        "time_seconds": res["seconds"],
    }


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
                "run": run + 1,
                "seed": seed,
                **_metrics(dataset_name, res),
            })
            if verbose:
                _log(f"  [exp1 {done}/{total}] done: best={res['best_cost']:.1f} {res['seconds']:.1f}s elapsed={time.time()-t0:.0f}s")
            histories[method].append(res["history_best"])
    return rows, histories


def experiment_scaling(datasets=None, params=None, runs=10, base_seed=1000, verbose=True,
                       init_method="random"):
    """Experiment 2 (MANDATORY) — scaling across problem sizes (default random init,
    so the GA itself is measured; smart would flatline at init_best)."""
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
        optimal = KNOWN_OPTIMALS[name]
        for run in range(runs):
            seed = base_seed + idx * 1000 + run
            done += 1
            if verbose:
                _log(f"  [exp2 {done}/{total}] dataset={name} run={run+1}/{runs} ...")
            res = run_single(cities, init_method, params, seed, weight_type=weight_type)
            rows.append({
                "dataset": name,
                "num_cities": len(cities),
                "run": run + 1,
                "seed": seed,
                "known_optimal": optimal,
                **_metrics(name, res),
            })
            if verbose:
                _log(f"  [exp2 {done}/{total}] done: best={res['best_cost']:.1f} {res['seconds']:.1f}s elapsed={time.time()-t0:.0f}s")
            histories[name].append(res["history_best"])
    return rows, histories


def experiment_population_size(cities, dataset_name="berlin52",
                               values=(50, 100, 150, 200),
                               params=None, runs=10, base_seed=2000, verbose=True,
                               weight_type="EUC_2D", init_method="random"):
    """Experiment 3 — vary population size p (random init to avoid smart ceiling)."""
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
            res = run_single(cities, init_method, cfg, seed, weight_type=weight_type)
            rows.append({
                "dataset": dataset_name, "p": p, "run": run + 1,
                "seed": seed, **_metrics(dataset_name, res),
            })
            if verbose:
                _log(f"  [exp3 {done}/{total}] done: best={res['best_cost']:.1f} {res['seconds']:.1f}s elapsed={time.time()-t0:.0f}s")
            histories[p].append(res["history_best"])
    return rows, histories


def experiment_mutation_rate(cities, dataset_name="berlin52",
                             values=(0.05, 0.1, 0.2, 0.4),
                             params=None, runs=10, base_seed=3000, verbose=True,
                             weight_type="EUC_2D", init_method="random"):
    """Experiment 4 — vary mutation rate m (random init to avoid smart ceiling)."""
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
            res = run_single(cities, init_method, cfg, seed, weight_type=weight_type)
            rows.append({
                "dataset": dataset_name, "m": m, "run": run + 1,
                "seed": seed, **_metrics(dataset_name, res),
            })
            if verbose:
                _log(f"  [exp4 {done}/{total}] done: best={res['best_cost']:.1f} {res['seconds']:.1f}s elapsed={time.time()-t0:.0f}s")
            histories[m].append(res["history_best"])
    return rows, histories


def experiment_replacement_rate(cities, dataset_name="berlin52",
                                values=(0.3, 0.5, 0.7, 0.9),
                                params=None, runs=10, base_seed=4000, verbose=True,
                                weight_type="EUC_2D", init_method="random"):
    """Experiment 5 — vary replacement rate r (random init to avoid smart ceiling)."""
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
            res = run_single(cities, init_method, cfg, seed, weight_type=weight_type)
            rows.append({
                "dataset": dataset_name, "r": r, "run": run + 1,
                "seed": seed, **_metrics(dataset_name, res),
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


FLOAT_COLS = {"best_cost", "init_best", "gap_percent", "time_seconds"}


def _section(title, rows, cols):
    """Build one (title, header, lines) block for the results file."""
    def fmt(col, value):
        return f"{value:.2f}" if col in FLOAT_COLS else str(value)
    lines = [",".join(fmt(c, r[c]) for c in cols) for r in rows]
    return title, ",".join(cols), lines


def run_initialization_comparison(datasets=None, runs=30, base_seed=42,
                                  params=None, results_dir=None):
    """Run only the initialization comparison and save its rows and plots."""
    from src.plot import plot_convergence_comparison

    datasets = list(datasets) if datasets is not None else list(SWEEP_DATASETS)
    unknown = [name for name in datasets if name not in DATASETS]
    if unknown:
        raise ValueError(f"Unknown datasets={unknown!r}, expected subset of {sorted(DATASETS)}")
    if not datasets:
        raise ValueError("datasets must not be empty")

    params = dict(DEFAULT_PARAMS if params is None else params)
    output_dir = Path(results_dir) if results_dir else RESULTS_DIR / "init_comparison_only"
    tail = ["run", "seed", "best_cost", "init_best", "gap_percent", "time_seconds"]
    sections = []
    output = {}

    for dataset_index, name in enumerate(datasets):
        cities, weight_type = load_instance(str(DATA_DIR / DATASETS[name]))
        seed = base_seed + dataset_index * 100000
        _log(f"[init-only] dataset={name} runs={runs} G={params['G']} p={params['p']} ...")
        rows, histories = experiment_init_comparison(
            cities, dataset_name=name, params=params, runs=runs,
            base_seed=seed, weight_type=weight_type)

        dataset_dir = output_dir / name
        plot_convergence_comparison(
            histories,
            title=f"{name}: initialization comparison (mean over {runs} runs)",
            save_path=str(dataset_dir / "convergence_init.png"),
            settings_text=(
                f"dataset={name} seed={seed} runs={runs} "
                f"p={params['p']} G={params['G']} r={params['r']} "
                f"m={params['m']} k={params['k']}"
            ),
        )
        sections.append(_section(
            f"EXPERIMENT 1: Initialization Comparison ({name})",
            rows, ["dataset", "init_method"] + tail))
        output[name] = {"rows": rows, "histories": histories}

    results_path = output_dir / "results.txt"
    _write_results(results_path, sections)
    _log(f"[init-only] results and plots saved under {output_dir}")
    return {"datasets": output, "results_path": str(results_path)}


def run_all_experiments(base_seed=42, runs=10, generations=500,
                        population_size=100, results_path=None,
                        dataset="berlin52", scaling_datasets=None,
                        suite_init_method="random"):
    """Run all 5 experiments and write ``results/results.txt``.

    :param base_seed: master seed; every trial derives its own seed from it.
    :param runs: repetitions per configuration (README requires 10).
    :param generations: override G for all experiments (README: 500).
    :param population_size: override default p (README: 100).
    :param dataset: dataset key from DATASETS used for exp1/3/4/5
        (single-dataset experiments). Default "berlin52" (eski davranış).
    :param scaling_datasets: dataset key list for exp2 (scaling).
        None => all datasets in DATASETS (eski davranış).
    :param suite_init_method: kept for backward compatibility; per-init sweeps
        now cover every method in ``init_methods`` so this is only a fallback
        when ``init_methods`` is empty (defaults to ``suite_init_method``).
    :return: {dataset: {exp1 rows/histories, per_init: {init: {...}},
        results_path, plot_dir, cities, weight_type}}
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
          f"dataset={dataset}, scaling={scaling_datasets}, suite_init={suite_init_method}). Progress below.")
    tall0 = time.time()

    _log(f"[exp1] init comparison on {dataset} (runs={runs}, G={generations}) ...")
    exp1_rows, exp1_hist = experiment_init_comparison(
        cities, dataset_name=dataset, params=base_params, runs=runs, base_seed=base_seed,
        weight_type=weight_type)
    _log(f"[exp1] done in {time.time()-tall0:.0f}s")

    _log(f"[exp2] problem size scaling on {scaling_datasets} (init={suite_init_method}) ...")
    exp2_rows, exp2_hist = experiment_scaling(
        datasets=scaling_datasets, params=base_params, runs=runs, base_seed=base_seed + 1000,
        init_method=suite_init_method)
    _log(f"[exp2] done, suite elapsed={time.time()-tall0:.0f}s")

    _log(f"[exp3] population size on {dataset} (init={suite_init_method}) ...")
    exp3_rows, exp3_hist = experiment_population_size(
        cities, dataset_name=dataset, params=base_params, runs=runs, base_seed=base_seed + 2000,
        weight_type=weight_type, init_method=suite_init_method)
    _log(f"[exp3] done, suite elapsed={time.time()-tall0:.0f}s")

    _log(f"[exp4] mutation rate on {dataset} (init={suite_init_method}) ...")
    exp4_rows, exp4_hist = experiment_mutation_rate(
        cities, dataset_name=dataset, params=base_params, runs=runs, base_seed=base_seed + 3000,
        weight_type=weight_type, init_method=suite_init_method)
    _log(f"[exp4] done, suite elapsed={time.time()-tall0:.0f}s")

    _log(f"[exp5] replacement rate on {dataset} (init={suite_init_method}) ...")
    exp5_rows, exp5_hist = experiment_replacement_rate(
        cities, dataset_name=dataset, params=base_params, runs=runs, base_seed=base_seed + 4000,
        weight_type=weight_type, init_method=suite_init_method)
    _log(f"[exp5] done, suite elapsed={time.time()-tall0:.0f}s")

    tail = ["run", "seed", "best_cost", "init_best", "gap_percent", "time_seconds"]
    sections = [
        _section("EXPERIMENT 1: Initialization Comparison", exp1_rows,
                 ["dataset", "init_method"] + tail),
        _section("EXPERIMENT 2: Problem Size Scaling", exp2_rows,
                 ["dataset", "num_cities", "init_method", "known_optimal"] + tail),
        _section("EXPERIMENT 3: Population Size", exp3_rows,
                 ["dataset", "init_method", "p"] + tail),
        _section("EXPERIMENT 4: Mutation Rate", exp4_rows,
                 ["dataset", "init_method", "m"] + tail),
        _section("EXPERIMENT 5: Replacement Rate", exp5_rows,
                 ["dataset", "init_method", "r"] + tail),
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


# ---------------------------------------------------------------------------
# Full sweep (one flag in main.py runs everything per dataset)
# ---------------------------------------------------------------------------

SWEEP_P_VALUES = (50, 100, 120, 150)
SWEEP_G_VALUES = (100, 300, 500, 800)
SWEEP_M_VALUES = (0.05, 0.1, 0.2, 0.4)
SWEEP_R_VALUES = (0.3, 0.5, 0.7, 0.9)


def experiment_generations(cities, dataset_name="berlin52",
                           values=SWEEP_G_VALUES,
                           params=None, runs=10, base_seed=5000, verbose=True,
                           weight_type="EUC_2D", init_method="random"):
    """Experiment 6 — vary generations G (random init to avoid smart ceiling)."""
    params = dict(DEFAULT_PARAMS if params is None else params)
    rows = []
    histories = {}
    total = len(values) * runs
    done = 0
    t0 = time.time()
    for vi, G in enumerate(values):
        cfg = dict(params, G=G)
        histories[G] = []
        for run in range(runs):
            seed = base_seed + vi * 1000 + run
            done += 1
            if verbose:
                _log(f"  [expG {done}/{total}] G={G} run={run+1}/{runs} ...")
            res = run_single(cities, init_method, cfg, seed, weight_type=weight_type)
            rows.append({
                "dataset": dataset_name, "G": G, "run": run + 1,
                "seed": seed, **_metrics(dataset_name, res),
            })
            if verbose:
                _log(f"  [expG {done}/{total}] done: best={res['best_cost']:.1f} {res['seconds']:.1f}s elapsed={time.time()-t0:.0f}s")
            histories[G].append(res["history_best"])
    return rows, histories


def run_full_sweep(base_seed=42, runs=10,
                   p_values=SWEEP_P_VALUES, g_values=SWEEP_G_VALUES,
                   m_values=SWEEP_M_VALUES, r_values=SWEEP_R_VALUES,
                   init_methods=INIT_METHODS, datasets=None,
                   suite_init_method="random", results_dir=None):
    """Run the full grid: every dataset x (init/p/G/m/r sweeps).

    - exp1 per dataset tries all init_methods one by one (random/nn/angle/smart).
    - exp3/4/5/G per dataset sweep p/G/m/r with suite_init_method.
    - exp2 scaling runs once over every dataset in DATASETS (suite_init_method).
    - One results.txt per dataset is written to <results_dir>/<dataset>/results.txt;
      everything is also collected in <results_dir>/results.txt.

    :return: {dataset: {exp1..expG rows/histories, results_path, cities, weight_type}}
    """
    from src.plot import (plot_convergence_comparison, plot_tour)  # local import: avoid cycle
    from src.utils import calculate_cost as _cost

    datasets = list(datasets) if datasets else list(SWEEP_DATASETS)
    unknown = [d for d in datasets if d not in DATASETS]
    if unknown:
        raise ValueError(f"Unknown datasets={unknown!r}")
    base_dir = Path(results_dir) if results_dir else RESULTS_DIR
    out_all = {}
    all_sections = []
    tall0 = time.time()

    _log("[sweep] problem size scaling ...")
    exp2_rows, exp2_hist = experiment_scaling(
        params=dict(DEFAULT_PARAMS), runs=runs, base_seed=base_seed + 1000,
        init_method=suite_init_method)
    from src.plot import plot_scaling
    plot_scaling(exp2_rows, save_path=str(base_dir / "scaling_results.png"))
    tail = ["run", "seed", "best_cost", "init_best", "gap_percent", "time_seconds"]
    all_sections.append(_section(
        "EXPERIMENT 2: Problem Size Scaling", exp2_rows,
        ["dataset", "num_cities", "init_method", "known_optimal"] + tail))

    for di, name in enumerate(datasets):
        cities, weight_type = load_instance(str(DATA_DIR / DATASETS[name]))
        ddir = base_dir / name
        ddir.mkdir(parents=True, exist_ok=True)
        base_params = dict(DEFAULT_PARAMS)
        _log(f"[sweep {di+1}/{len(datasets)}] dataset={name} "
             f"p={list(p_values)} G={list(g_values)} m={list(m_values)} r={list(r_values)} "
             f"inits={list(init_methods)} ...")

        exp1_rows, exp1_hist = experiment_init_comparison(
            cities, dataset_name=name, params=base_params, runs=runs,
            base_seed=base_seed + di * 100000, weight_type=weight_type)

        all_sections.append(_section(
            f"EXPERIMENT 1: Initialization Comparison ({name})", exp1_rows,
            ["dataset", "init_method"] + tail))

        # Per-dataset plot comparing inits, with its own settings footer.
        def _sweep_cfg(extra, init=None):
            tag = f"init={init} " if init else ""
            return (f"dataset={name} {tag}seed={base_seed} runs={runs} {extra} "
                    f"defaults p={DEFAULT_PARAMS['p']} G={DEFAULT_PARAMS['G']} "
                    f"r={DEFAULT_PARAMS['r']} m={DEFAULT_PARAMS['m']} k={DEFAULT_PARAMS['k']}")

        plot_convergence_comparison(
            exp1_hist, title=f"{name}: init comparison (mean over {runs} runs)",
            save_path=str(ddir / "convergence_init.png"),
            settings_text=_sweep_cfg(f"inits={list(init_methods)}"))

        # Every sweep repeated for every init -> results/<dataset>/<init>/*.png
        per_init = {}
        dataset_all_rows = list(exp1_rows)
        for ii, init in enumerate(init_methods):
            idir = ddir / init
            idir.mkdir(parents=True, exist_ok=True)
            seed_base = base_seed + di * 100000 + ii * 10000
            _log(f"[sweep {di+1}/{len(datasets)}] dataset={name} init={init} ...")
            exp3_rows, exp3_hist = experiment_population_size(
                cities, dataset_name=name, values=tuple(p_values), params=base_params,
                runs=runs, base_seed=seed_base + 2000,
                weight_type=weight_type, init_method=init)
            expG_rows, expG_hist = experiment_generations(
                cities, dataset_name=name, values=tuple(g_values), params=base_params,
                runs=runs, base_seed=seed_base + 5000,
                weight_type=weight_type, init_method=init)
            exp4_rows, exp4_hist = experiment_mutation_rate(
                cities, dataset_name=name, values=tuple(m_values), params=base_params,
                runs=runs, base_seed=seed_base + 3000,
                weight_type=weight_type, init_method=init)
            exp5_rows, exp5_hist = experiment_replacement_rate(
                cities, dataset_name=name, values=tuple(r_values), params=base_params,
                runs=runs, base_seed=seed_base + 4000,
                weight_type=weight_type, init_method=init)

            plot_convergence_comparison(
                exp3_hist, title=f"{name}/{init}: population size sweep",
                save_path=str(idir / "convergence_popsize.png"),
                settings_text=_sweep_cfg(f"p_values={list(p_values)}", init=init))
            plot_convergence_comparison(
                expG_hist, title=f"{name}/{init}: generations sweep",
                save_path=str(idir / "convergence_generations.png"),
                settings_text=_sweep_cfg(f"G_values={list(g_values)}", init=init))
            plot_convergence_comparison(
                exp4_hist, title=f"{name}/{init}: mutation rate sweep",
                save_path=str(idir / "convergence_mutation.png"),
                settings_text=_sweep_cfg(f"m_values={list(m_values)}", init=init))
            plot_convergence_comparison(
                exp5_hist, title=f"{name}/{init}: replacement rate sweep",
                save_path=str(idir / "convergence_replacement.png"),
                settings_text=_sweep_cfg(f"r_values={list(r_values)}", init=init))

            rep = run_single(cities, init, dict(DEFAULT_PARAMS),
                             seed_base + 999, weight_type=weight_type)
            plot_tour(rep["best_tour"], cities,
                      title=f"{name}/{init}: best tour (cost={rep['best_cost']:.1f})",
                      save_path=str(idir / "tour_best.png"),
                      settings_text=_sweep_cfg(
                          f"p={DEFAULT_PARAMS['p']} G={DEFAULT_PARAMS['G']} "
                          f"r={DEFAULT_PARAMS['r']} m={DEFAULT_PARAMS['m']} "
                          f"k={DEFAULT_PARAMS['k']} seed={rep['seed']}", init=init))
            from src.plot import plot_convergence as _pc
            _pc(rep["history_best"],
                title=f"{name}/{init}: single-run convergence",
                save_path=str(idir / "convergence_single.png"),
                settings_text=_sweep_cfg(
                    f"p={DEFAULT_PARAMS['p']} G={DEFAULT_PARAMS['G']} "
                    f"seed={rep['seed']}", init=init))

            sections = [
                _section(f"EXPERIMENT 3 ({init}): Population Size", exp3_rows,
                         ["dataset", "init_method", "p"] + tail),
                _section(f"EXPERIMENT 6 ({init}): Generations", expG_rows,
                         ["dataset", "init_method", "G"] + tail),
                _section(f"EXPERIMENT 4 ({init}): Mutation Rate", exp4_rows,
                         ["dataset", "init_method", "m"] + tail),
                _section(f"EXPERIMENT 5 ({init}): Replacement Rate", exp5_rows,
                         ["dataset", "init_method", "r"] + tail),
            ]
            all_sections += sections
            _write_results(idir / "results.txt", sections)
            per_init[init] = {
                "exp3": (exp3_rows, exp3_hist),
                "expG": (expG_rows, expG_hist),
                "exp4": (exp4_rows, exp4_hist),
                "exp5": (exp5_rows, exp5_hist),
                "results_path": str(idir / "results.txt"),
                "plot_dir": str(idir),
            }
            dataset_all_rows += exp3_rows + expG_rows + exp4_rows + exp5_rows
            _log(f"[sweep {di+1}/{len(datasets)}] {name}/{init} done, "
                 f"elapsed={time.time()-tall0:.0f}s")

        best_row = min(dataset_all_rows, key=lambda r: r["best_cost"])
        _log(f"[sweep {di+1}/{len(datasets)}] {name} best over sweep: "
             f"{best_row['best_cost']:.1f} (seed={best_row.get('seed')})")
        _write_results(ddir / "results.txt", [
            ("SWEEP SUMMARY: best trial per init",
             "dataset,init,best_cost",
             [f"{name},{init},{min(r['best_cost'] for r in per_init[init]['exp3'][0]):.2f}"
              for init in init_methods]),
        ])

        out_all[name] = {
            "exp1": (exp1_rows, exp1_hist),
            "per_init": per_init,
            "results_path": str(ddir / "results.txt"),
            "plot_dir": str(ddir),
            "cities": cities,
            "weight_type": weight_type,
        }
        _log(f"[sweep {di+1}/{len(datasets)}] {name} done, elapsed={time.time()-tall0:.0f}s")
    _write_results(base_dir / "results.txt", all_sections)
    _log(f"[sweep] consolidated results: {base_dir / 'results.txt'}")
    return out_all
