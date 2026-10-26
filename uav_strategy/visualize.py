"""Publication-quality visualisations for simulation and experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle

from .environment import Environment
from .fitness import Evaluation, FitnessEvaluator
from .genome import Genome

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 170,
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "legend.fontsize": 8,
        "figure.autolayout": False,
    }
)

DRONE_COLORS = ["#00d4ff", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]


def _ensure(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def draw_environment(ax: plt.Axes, env: Environment, show_targets: bool = True) -> None:
    """Draw hazards, no-fly zones, priority hotspots and the base."""

    n = env.size
    extent = (-0.5, n - 0.5, -0.5, n - 0.5)
    im = ax.imshow(
        env.hazard,
        origin="lower",
        extent=extent,
        cmap="inferno",
        vmin=0,
        vmax=1,
        alpha=0.72,
        zorder=1,
    )
    if env.no_fly.any():
        ax.contourf(
            np.arange(n),
            np.arange(n),
            env.no_fly.astype(float),
            levels=[0.5, 1.5],
            colors=["#3a3a3a"],
            alpha=0.92,
            zorder=3,
        )
    if env.priority_mask.max() > 0:
        ax.contour(
            np.arange(n),
            np.arange(n),
            env.priority_mask,
            levels=[1.0, 2.0, 3.0],
            colors=["#ffe066", "#ffb703", "#ff8800"],
            linewidths=0.8,
            alpha=0.85,
            zorder=4,
        )
    for zone_index, zone in enumerate(env.zones, start=1):
        ax.add_patch(
            Circle(
                (zone.cx, zone.cy),
                zone.radius,
                fill=False,
                ec="#ffe066",
                lw=0.8,
                ls=":",
                alpha=0.7,
                zorder=4,
            )
        )
        ax.text(
            zone.cx,
            zone.cy,
            f"Z{zone_index} · P{zone.priority}",
            color="#1a1a1a",
            ha="center",
            va="center",
            fontsize=6.5,
            zorder=6,
            bbox=dict(boxstyle="round,pad=0.18", fc="#ffe066", ec="none", alpha=0.85),
        )
    if show_targets:
        ax.scatter(
            env.targets[:, 0],
            env.targets[:, 1],
            marker="x",
            s=16,
            c="#ff4d6d",
            linewidths=0.9,
            zorder=5,
            label="Priority targets",
        )
    ax.scatter(
        env.config.base[0],
        env.config.base[1],
        marker="s",
        s=90,
        c="white",
        edgecolors="black",
        linewidths=1.0,
        zorder=8,
        label="Base",
    )
    # Wind direction arrow (energy model).
    ax.annotate(
        "",
        xy=(
            n - 6 + 5 * env.wind_direction[0],
            n - 6 + 5 * env.wind_direction[1],
        ),
        xytext=(n - 6, n - 6),
        arrowprops=dict(arrowstyle="->", color="white", lw=1.2),
        zorder=9,
    )
    ax.text(n - 6, n - 4.5, "wind", color="white", fontsize=7, ha="center", zorder=9)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_xlabel("x (cells)")
    ax.set_ylabel("y (cells)")
    ax.set_aspect("equal")
    return im


def plot_mission(env: Environment, path: str | Path) -> Path:
    """Static mission map (hazard field + zones + targets)."""

    out = _ensure(path)
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    im = draw_environment(ax, env)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("Hazard / radiation intensity")
    handles = [
        Line2D([0], [0], color="#ffe066", lw=1, label="Priority zone"),
        Line2D([0], [0], marker="x", color="#ff4d6d", lw=0, label="Target"),
        Line2D([0], [0], marker="s", color="white", markeredgecolor="black", lw=0, label="Base"),
        Line2D([0], [0], color="#3a3a3a", lw=7, label="No-fly zone"),
    ]
    ax.legend(handles=handles, loc="lower right", framealpha=0.82)
    ax.set_title("Mission environment: hazard plumes, disaster hotspots and restricted airspace")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_paths(env: Environment, genome: Genome, evaluation: Evaluation, path: str | Path) -> Path:
    """Visualise the evolved trajectories of every active UAV."""

    out = _ensure(path)
    fig, ax = plt.subplots(figsize=(7.6, 6.6))
    im = draw_environment(ax, env)
    for i in range(genome.n_drones):
        if not genome.active[i]:
            continue
        route = genome.path(i, env.config.base)
        color = DRONE_COLORS[i % len(DRONE_COLORS)]
        ax.plot(route[:, 0], route[:, 1], "-", color=color, lw=1.8, zorder=7, label=f"UAV {i + 1}")
        ax.scatter(
            genome.deploy[i, 0],
            genome.deploy[i, 1],
            marker="*",
            s=130,
            c=color,
            edgecolors="white",
            linewidths=0.5,
            zorder=9,
        )
        sensor_on = genome.sensor[i] > 0.1
        nodes = np.vstack([genome.deploy[i][None, :], genome.patrol[i]])
        ax.scatter(
            nodes[sensor_on, 0],
            nodes[sensor_on, 1],
            s=14 + 28 * genome.sensor[i][sensor_on],
            facecolors="none",
            edgecolors=color,
            linewidths=0.7,
            alpha=0.8,
            zorder=8,
        )
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("Hazard intensity")
    ax.set_title(
        f"Evolved UAV trajectories  |  coverage {evaluation.coverage:.2f}, "
        f"detection {evaluation.detection:.2f}, risk {evaluation.risk:.2f}"
    )
    ax.legend(loc="lower right", ncol=2, framealpha=0.82)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_coverage(
    env: Environment,
    evaluation: Evaluation,
    genome: Genome,
    path: str | Path,
) -> Path:
    """Fused sensor-coverage heatmap with the final routes overlaid."""

    out = _ensure(path)
    assert evaluation.coverage_map is not None
    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    n = env.size
    extent = (-0.5, n - 0.5, -0.5, n - 0.5)
    im = ax.imshow(
        evaluation.coverage_map,
        origin="lower",
        extent=extent,
        cmap="viridis",
        vmin=0,
        vmax=1,
        zorder=1,
    )
    if env.no_fly.any():
        ax.contourf(
            np.arange(n),
            np.arange(n),
            env.no_fly.astype(float),
            levels=[0.5, 1.5],
            colors=["#555555"],
            alpha=0.9,
            zorder=2,
        )
    if env.priority_mask.max() > 0:
        ax.contour(
            np.arange(n),
            np.arange(n),
            env.priority_mask,
            levels=[0.5, 1.5, 2.5],
            colors="#ffd166",
            linewidths=0.7,
            zorder=3,
        )
    for i in range(genome.n_drones):
        if genome.active[i]:
            route = genome.path(i, env.config.base)
            ax.plot(route[:, 0], route[:, 1], color="white", lw=1.0, ls="--", alpha=0.75, zorder=4)
    ax.scatter(
        env.targets[:, 0],
        env.targets[:, 1],
        marker="x",
        c="#ff4d6d",
        s=20,
        zorder=5,
        label="Targets",
    )
    ax.scatter(
        env.config.base[0],
        env.config.base[1],
        marker="s",
        c="white",
        edgecolors="black",
        s=90,
        zorder=6,
        label="Base",
    )
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("Detection probability")
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("x (cells)")
    ax.set_ylabel("y (cells)")
    ax.set_title(
        f"Coverage heatmap  |  priority coverage {evaluation.coverage:.2f}, "
        f"target detection {evaluation.detection_rate:.2f}"
    )
    ax.legend(loc="lower right", framealpha=0.85)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_coverage_evolution(
    env: Environment,
    evaluator: FitnessEvaluator,
    snapshots: list[tuple[int, Genome]],
    path: str | Path,
    panels: int = 4,
) -> Path:
    """Coverage heatmaps at evenly spaced generations."""

    out = _ensure(path)
    if not snapshots:
        raise ValueError("no snapshots available")
    if len(snapshots) > panels:
        idx = np.linspace(0, len(snapshots) - 1, panels).round().astype(int)
        chosen = [snapshots[i] for i in sorted(set(idx))]
    else:
        chosen = snapshots
    cols = 2
    rows = int(np.ceil(len(chosen) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(9.4, 4.3 * rows), squeeze=False)
    n = env.size
    extent = (-0.5, n - 0.5, -0.5, n - 0.5)
    for ax, (generation, genome) in zip(axes.ravel(), chosen):
        ev = evaluator.evaluate(genome, keep_map=True)
        assert ev.coverage_map is not None
        im = ax.imshow(
            ev.coverage_map,
            origin="lower",
            extent=extent,
            cmap="viridis",
            vmin=0,
            vmax=1,
        )
        if env.priority_mask.max() > 0:
            ax.contour(
                np.arange(n),
                np.arange(n),
                env.priority_mask,
                levels=[0.5, 1.5, 2.5],
                colors="#ffd166",
                linewidths=0.5,
            )
        if env.no_fly.any():
            ax.contourf(
                np.arange(n),
                np.arange(n),
                env.no_fly.astype(float),
                levels=[0.5, 1.5],
                colors=["#555555"],
                alpha=0.85,
            )
        ax.set_title(f"Generation {generation}  |  coverage {ev.coverage:.2f}  |  energy {ev.energy_frac:.0%}")
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes.ravel()[len(chosen) :]:
        ax.axis("off")
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02, label="Detection probability")
    fig.suptitle("Best-genome coverage footprint and energy use across generations", y=0.995)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_convergence(history: list[dict], path: str | Path) -> Path:
    """Best vs average fitness and population diversity over generations."""

    out = _ensure(path)
    gens = [h["generation"] for h in history]
    best = [h["best"] for h in history]
    avg = [h["average"] for h in history]
    div = [h["diversity"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 3.9))
    ax1.plot(gens, best, color="#00b4d8", lw=2.0, label="Best fitness")
    ax1.plot(gens, avg, color="#ff7f0e", lw=1.4, label="Average fitness")
    ax1.fill_between(gens, avg, best, color="#00b4d8", alpha=0.10)
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Scalarised fitness")
    ax1.set_title("Fitness convergence: best vs population average")
    ax1.legend(loc="lower right")
    ax1.grid(alpha=0.25)

    ax2.plot(gens, div, color="#2ca02c", lw=1.8, label="Genotypic diversity")
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Mean normalised gene distance")
    ax2.set_title("Population diversity maintained by mutation / immigrants")
    ax2.set_ylim(bottom=0)
    ax2.legend(loc="upper right")
    ax2.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_pareto(
    pareto_metrics: list[dict],
    best_evaluation: Evaluation,
    path: str | Path,
) -> Path:
    """Final non-dominated trade-off surface (coverage vs energy vs safety)."""

    out = _ensure(path)
    if not pareto_metrics:
        return out
    cov = np.array([m["coverage"] for m in pareto_metrics])
    energy = np.array([m["energy"] for m in pareto_metrics])
    risk = np.array([m["risk"] for m in pareto_metrics])
    det = np.array([m["detection"] for m in pareto_metrics])

    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    scatter = ax.scatter(
        energy,
        cov,
        c=risk,
        s=28 + 45 * det,
        cmap="plasma_r",
        edgecolors="white",
        linewidths=0.4,
        alpha=0.9,
    )
    ax.scatter(
        best_evaluation.energy,
        best_evaluation.coverage,
        marker="*",
        s=260,
        c="#00ffc8",
        edgecolors="black",
        linewidths=1.0,
        label="Selected best compromise",
        zorder=5,
    )
    cb = fig.colorbar(scatter, ax=ax)
    cb.set_label("Hazard exposure risk (lower is better)")
    ax.set_xlabel("Energy expenditure (fitness units)")
    ax.set_ylabel("Priority-weighted coverage")
    ax.set_title("Pareto trade-off surface: coverage vs energy vs risk")
    ax.legend(loc="lower right", framealpha=0.85)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_sensitivity(
    values: list[float],
    runs: list[dict],
    parameter: str,
    path: str | Path,
) -> Path:
    """One-at-a-time hyper-parameter sensitivity with seed variability."""

    out = _ensure(path)
    fitness = np.array([r["fitness"] for r in runs])
    coverage = np.array([r["coverage"] for r in runs])
    risk = np.array([r["risk"] for r in runs])
    # group by repeated seed runs
    f_mean = np.array([fitness[np.array([r["value"] for r in runs]) == v].mean() for v in values])
    f_std = np.array([fitness[np.array([r["value"] for r in runs]) == v].std() for v in values])
    c_mean = np.array([coverage[np.array([r["value"] for r in runs]) == v].mean() for v in values])
    c_std = np.array([coverage[np.array([r["value"] for r in runs]) == v].std() for v in values])
    r_mean = np.array([risk[np.array([r["value"] for r in runs]) == v].mean() for v in values])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 3.8))
    ax1.errorbar(values, f_mean, yerr=f_std, marker="o", color="#0077b6", capsize=3, lw=1.8)
    ax1.set_xlabel(parameter)
    ax1.set_ylabel("Final best fitness")
    ax1.set_title(f"Sensitivity of fitness to {parameter}")
    ax1.grid(alpha=0.25)

    ax2.errorbar(values, c_mean, yerr=c_std, marker="s", color="#2ca02c", capsize=3, lw=1.8, label="Coverage")
    ax2.plot(values, r_mean, marker="^", color="#d62728", lw=1.4, label="Risk")
    ax2.set_xlabel(parameter)
    ax2.set_ylabel("Score")
    ax2.set_title(f"Coverage / risk response to {parameter}")
    ax2.set_ylim(0, 1)
    ax2.legend(loc="best")
    ax2.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_sensitivity_summary(
    summary: dict,
    path: str | Path,
) -> Path:
    """Compact three-panel sensitivity figure used in the technical report."""

    out = _ensure(path)
    specs = [
        ("population_size", "Population size"),
        ("mutation_rate", "Mutation rate"),
        ("tournament_size", "Tournament size"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.1))
    for ax, (key, label) in zip(axes, specs):
        raw_values = sorted(summary[key].keys(), key=float)
        means = [summary[key][raw]["fitness"]["mean"] for raw in raw_values]
        stds = [summary[key][raw]["fitness"]["std"] for raw in raw_values]
        xs = np.arange(len(raw_values))
        ax.errorbar(xs, means, yerr=stds, marker="o", color="#0077b6", capsize=3, lw=1.8)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{float(v):g}" for v in raw_values])
        ax.set_xlabel(label)
        ax.set_ylabel("Final fitness" if key == "population_size" else "")
        ax.set_title(label)
        ax.grid(alpha=0.25)
    fig.suptitle("Hyper-parameter sensitivity (mean ± s.d. over independent GA seeds)", y=1.02)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_baseline_comparison(
    ga_histories: list[list[dict]],
    baseline_histories: list[list[dict]],
    ga_final: list[dict],
    baseline_final: list[dict],
    path: str | Path,
) -> Path:
    """Equal-budget convergence and final-metric comparison vs random search."""

    out = _ensure(path)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 3.9))

    # Align traces on the number of chromosome evaluations (equal budget).
    def _trace(histories: list[list[dict]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        max_eval = max(h[-1]["evaluations"] for h in histories)
        grid = np.linspace(0, max_eval, 250)
        traces = []
        for h in histories:
            xs = np.array([row["evaluations"] for row in h], dtype=float)
            ys = np.array([row["best"] for row in h], dtype=float)
            traces.append(np.interp(grid, xs, ys))
        arr = np.vstack(traces)
        return grid, arr.mean(axis=0), arr.min(axis=0), arr.max(axis=0)

    for histories, color, label in [
        (ga_histories, "#0077b6", "GA (mean of seeds)"),
        (baseline_histories, "#d62728", "Random search (mean of seeds)"),
    ]:
        xs, mean, lo, hi = _trace(histories)
        ax1.plot(xs, mean, color=color, lw=1.9, label=label)
        ax1.fill_between(xs, lo, hi, color=color, alpha=0.15)
    ax1.set_xlabel("Chromosome evaluations")
    ax1.set_ylabel("Best-so-far fitness")
    ax1.set_title("Best-so-far fitness at equal evaluation budget")
    ax1.legend(loc="lower right")
    ax1.grid(alpha=0.25)

    metrics = ["fitness", "coverage", "detection", "energy_frac", "risk"]
    labels = ["Fitness", "Coverage", "Detection", "Energy*", "Risk"]
    ga_means = [np.mean([r[m] for r in ga_final]) for m in metrics]
    base_means = [np.mean([r[m] for r in baseline_final]) for m in metrics]
    x = np.arange(len(metrics))
    width = 0.36
    ax2.bar(x - width / 2, ga_means, width, color="#0077b6", label="GA")
    ax2.bar(x + width / 2, base_means, width, color="#d62728", label="Random")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_title("Final solution quality (mean of seeds)")
    ax2.legend(loc="best")
    ax2.grid(axis="y", alpha=0.25)
    ax2.text(
        0.99,
        0.02,
        "* lower is better",
        transform=ax2.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
        color="#555555",
    )
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out
