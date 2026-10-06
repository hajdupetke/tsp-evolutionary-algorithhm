"""Chunk: pcb442/angle sweep (160 runs) + pr1002/random sweep (160 runs)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pathlib import Path
from src.experiments import (
    experiment_population_size, experiment_generations,
    experiment_mutation_rate, experiment_replacement_rate,
    _write_results, DEFAULT_PARAMS,
    SWEEP_P_VALUES, SWEEP_G_VALUES, SWEEP_M_VALUES, SWEEP_R_VALUES,
)
from src.utils import load_instance


def run_init_sweep(dataset, init, seed_base):
    cities, wt = load_instance(f"data/{dataset}.tsp")
    base = dict(DEFAULT_PARAMS)
    runs = 10
    e3, _ = experiment_population_size(cities, dataset, tuple(SWEEP_P_VALUES), base, runs, seed_base + 2000, True, wt, init)
    eG, _ = experiment_generations(cities, dataset, tuple(SWEEP_G_VALUES), base, runs, seed_base + 5000, True, wt, init)
    e4, _ = experiment_mutation_rate(cities, dataset, tuple(SWEEP_M_VALUES), base, runs, seed_base + 3000, True, wt, init)
    e5, _ = experiment_replacement_rate(cities, dataset, tuple(SWEEP_R_VALUES), base, runs, seed_base + 4000, True, wt, init)
    idir = Path(f"results/{dataset}/{init}")
    idir.mkdir(parents=True, exist_ok=True)
    sections = []
    for title, rows, key in (
        (f"EXPERIMENT 3 ({init}): Population Size", e3, "p"),
        (f"EXPERIMENT 6 ({init}): Generations", eG, "G"),
        (f"EXPERIMENT 4 ({init}): Mutation Rate", e4, "m"),
        (f"EXPERIMENT 5 ({init}): Replacement Rate", e5, "r"),
    ):
        header = f"dataset,init,{key},run,seed,init_best,best_cost,known_optimal,gap_percent,time_seconds"
        lines = [
            f"{r['dataset']},{init},{r[key]},{r['run']},{r['seed']},"
            f"{r['init_best']:.2f},{r['best_cost']:.2f},{r['known_optimal']},"
            f"{r['gap_percent']:.2f},{r['time_seconds']:.2f}"
            for r in rows
        ]
        sections.append((title, header, lines))
    _write_results(idir / "results.txt", sections)
    print(f"{dataset}/{init} DONE", flush=True)


if __name__ == "__main__":
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else "pcb442-angle"
    if which == "pcb442-angle":
        run_init_sweep("pcb442", "angle", 42 + 4 * 100000 + 2 * 10000)
    elif which == "pr1002-random":
        run_init_sweep("pr1002", "random", 42 + 5 * 100000 + 0 * 10000)
