"""plot.py — Figures for the report.

Generates (deliverables checklist:
  - convergence_berlin52.png        single-run convergence curve
  - convergence_comparison_init.png mean convergence per init method (exp1)
  - tour_berlin52_best.png          best tour drawn on the city map
  - scaling_results.png             gap-from-optimal vs problem size (exp2)

All functions take an explicit ``save_path``; parent folders are created
automatically. ``matplotlib`` is the only external dependency.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _ensure_parent(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _mean_histories(histories):
    """Average a list of equal-length history curves element-wise."""
    if not histories:
        return []
    g = min(len(h) for h in histories)
    return [sum(h[i] for h in histories) / len(histories) for i in range(g)]


def _annotate_settings(settings_text):
    """Footer line on the figure so each PNG records its own settings."""
    if settings_text:
        plt.figtext(0.5, 0.01, str(settings_text), ha="center",
                    fontsize=7, wrap=True)
        plt.subplots_adjust(bottom=0.18)


def plot_convergence(history, title="Convergence",
                     save_path="results/convergence_berlin52.png",
                     ylabel="Best tour cost", settings_text=None):
    """Plot best-cost-so-far over generations for a single GA run."""
    _ensure_parent(save_path)
    plt.figure()
    plt.plot(range(1, len(history) + 1), history)
    plt.xlabel("Generation")
    plt.ylabel(ylabel)
    plt.title(title)
    _annotate_settings(settings_text)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return str(save_path)


def plot_convergence_comparison(histories_dict,
                                title="Initialization comparison (berlin52)",
                                save_path="results/convergence_comparison_init.png",
                                settings_text=None):
    """Plot mean convergence curve per configuration.

    :param histories_dict: {label: [history_run1, history_run2, ...]}
    """
    _ensure_parent(save_path)
    plt.figure()
    for label in sorted(histories_dict.keys(), key=str):
        mean_curve = _mean_histories(histories_dict[label])
        if mean_curve:
            plt.plot(range(1, len(mean_curve) + 1), mean_curve, label=str(label))
    plt.xlabel("Generation")
    plt.ylabel("Mean best tour cost")
    plt.title(title)
    plt.legend()
    _annotate_settings(settings_text)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return str(save_path)


def plot_tour(tour, cities, title="Best tour",
              save_path="results/tour_berlin52_best.png",
              settings_text=None):
    """Draw the tour path on top of the city coordinates."""
    _ensure_parent(save_path)
    xs = [cities[i][0] for i in tour] + [cities[tour[0]][0]]
    ys = [cities[i][1] for i in tour] + [cities[tour[0]][1]]
    plt.figure()
    plt.plot(xs, ys, "-o", markersize=3)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.axis("equal")
    _annotate_settings(settings_text)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return str(save_path)


def plot_scaling(exp2_rows, title="Scaling: gap from optimal",
                 save_path="results/scaling_results.png",
                 settings_text=None):
    """Bar chart of mean %-gap from known optimal per dataset (exp2).

    :param exp2_rows: row dicts from ``experiment_scaling`` with keys
                      dataset / best_cost / known_optimal / gap_percent.
    """
    _ensure_parent(save_path)
    order = [d for d in ("burma14", "berlin52", "kroA100", "kroA200", "pcb442", "pr1002")
             if any(r["dataset"] == d for r in exp2_rows)]
    if not order:  # fall back to whatever datasets are present
        order = sorted({r["dataset"] for r in exp2_rows})
    means = []
    for d in order:
        gaps = [r["gap_percent"] for r in exp2_rows if r["dataset"] == d]
        means.append(sum(gaps) / len(gaps) if gaps else 0.0)
    plt.figure()
    plt.bar(order, means)
    plt.xlabel("Dataset")
    plt.ylabel("Mean gap from optimal (%)")
    plt.title(title)
    for x, y in zip(order, means):
        plt.text(x, y, f"{y:.2f}%", ha="center", va="bottom", fontsize=8)
    _annotate_settings(settings_text)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return str(save_path)
