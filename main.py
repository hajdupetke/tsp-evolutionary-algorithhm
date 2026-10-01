# TARI29 Group 9 — Christos Zampounis, Mehmet Cihan Yalçın, Péter Dávid Hajdu, Pierre Kasparian, Prajval Parthiban
# TSP Genetic Algorithm — entry point.
#
# Run:  python main.py [--seed 42] [--quick] [--skip-experiments] ...
# All randomness flows from the --seed argument so experiments are repeatable.

import argparse

from src.utils import load_cities, load_instance, calculate_cost
from src.experiments import (
    run_single,
    run_all_experiments,
    DEFAULT_PARAMS,
    KNOWN_OPTIMALS,
    DATASETS,
    DATA_DIR,
    RESULTS_DIR,
)
from src.plot import (
    plot_convergence,
    plot_convergence_comparison,
    plot_tour,
    plot_scaling,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="TSP Genetic Algorithm (TARI29 Group 9) — run GA + experiments.")
    p.add_argument("--seed", type=int, default=42,
                   help="Master random seed for reproducibility (default: 42).")
    p.add_argument("--population-size", type=int, default=100,
                   help="Population size p (default: 100).")
    p.add_argument("--generations", type=int, default=500,
                   help="Generations G for the sanity run and experiments (default: 500).")
    p.add_argument("--replacement-rate", type=float, default=0.7,
                   help="Replacement rate r (default: 0.7).")
    p.add_argument("--mutation-rate", type=float, default=0.1,
                   help="Mutation rate m (default: 0.1).")
    p.add_argument("--init-method", default="smart",
                   choices=["random", "nn", "angle", "smart"],
                   help="Init method for the sanity run (default: smart).")
    p.add_argument("--dataset", default="berlin52",
                   choices=sorted(DATASETS.keys()),
                   help="Dataset for the sanity run (default: berlin52).")
    p.add_argument("--suite-dataset", default="berlin52",
                   choices=sorted(DATASETS.keys()),
                   help="Dataset for exp1/3/4/5 in the full suite (default: berlin52).")
    p.add_argument("--scaling-datasets", nargs="*", default=None,
                   choices=sorted(DATASETS.keys()),
                   help="Datasets for exp2 scaling (default: all datasets).")
    p.add_argument("--runs", type=int, default=10,
                   help="Repetitions per experiment configuration (README: 10).")
    p.add_argument("--skip-experiments", action="store_true",
                   help="Only do the sanity run, skip the full experiment suite.")
    p.add_argument("--quick", action="store_true",
                   help="Smoke test: G=20, runs=2 (overrides --generations/--runs).")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.generations = 20
        args.runs = 2

    print("=== TARI29 Group 9 — TSP Genetic Algorithm ===\n", flush=True)
    print(f"Master seed: {args.seed} (all trials derive their own seed from it)", flush=True)

    # ---- Quick sanity run (dataset selectable via --dataset) ------------------
    print(f"Loading {args.dataset} ...", flush=True)
    cities, weight_type = load_instance(str(DATA_DIR / DATASETS[args.dataset]))
    print(f"Loaded {args.dataset}: {len(cities)} cities ({weight_type})", flush=True)

    params = dict(DEFAULT_PARAMS,
                  p=args.population_size,
                  G=args.generations,
                  r=args.replacement_rate,
                  m=args.mutation_rate)
    print(f"Running GA on {args.dataset} (init={args.init_method}, "
          f"p={params['p']}, r={params['r']}, m={params['m']}, "
          f"G={params['G']}) ...", flush=True)
    log_every = 50 if params['G'] > 50 else 5
    res = run_single(cities, args.init_method, params, seed=args.seed,
                     verbose=True, log_interval=log_every, weight_type=weight_type)
    best_tour, best_cost, history = res["best_tour"], res["best_cost"], res["history_best"]

    # Sanity check: recompute cost independently (same TSPLIB metric)
    assert abs(calculate_cost(best_tour, cities, weight_type) - best_cost) < 1e-6
    assert sorted(best_tour) == list(range(len(cities))), "Invalid tour!"

    optimal = KNOWN_OPTIMALS[args.dataset]
    print(f"\nResult:        {best_cost:.4f}", flush=True)
    print(f"Known optimal: {optimal:.4f}", flush=True)
    print(f"Gap:           {(best_cost - optimal) / optimal * 100:.4f}%", flush=True)
    print(f"Time:          {res['seconds']:.2f}s", flush=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_convergence(history, title=f"{args.dataset} convergence",
                     save_path=str(RESULTS_DIR / f"convergence_{args.dataset}.png"))
    plot_tour(best_tour, cities, title=f"{args.dataset} best tour",
              save_path=str(RESULTS_DIR / f"tour_{args.dataset}_best.png"))
    print("Saved convergence + tour plots.", flush=True)

    # ---- Full experiment suite --------------------------------------------
    if args.skip_experiments:
        print("\nSkipping full experiments (--skip-experiments).", flush=True)
        return

    print("\nRunning full experiment suite ...", flush=True)
    out = run_all_experiments(base_seed=args.seed,
                              runs=args.runs,
                              generations=args.generations,
                              population_size=args.population_size,
                              dataset=args.suite_dataset,
                              scaling_datasets=args.scaling_datasets)
    # Figures from experiment histories
    plot_convergence_comparison(
        out["exp1"][1], title=f"Init comparison ({args.suite_dataset} mean over runs)",
        save_path=str(RESULTS_DIR / "convergence_comparison_init.png"))
    plot_scaling(out["exp2"][0],
                 save_path=str(RESULTS_DIR / "scaling_results.png"))
    # Reseed demo: same master seed must reproduce the sanity run
    check = run_single(cities, args.init_method, params, seed=args.seed,
                       weight_type=weight_type)
    assert check["best_cost"] == best_cost, "Seed reproducibility broken!"
    print(f"\nReproducibility check passed (seed={args.seed}).")
    print(f"All done. Results: {out['results_path']}")


if __name__ == "__main__":
    main()
