"""High-level simulation runner that produces the required deliverables."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Callable

from .config import FitnessWeights, GAConfig, MissionConfig
from .ga import GAResult, GeneticStrategyEngine
from .visualize import (
    plot_convergence,
    plot_coverage,
    plot_coverage_evolution,
    plot_mission,
    plot_pareto,
    plot_paths,
)


def run_simulation(
    mission: MissionConfig | None = None,
    ga: GAConfig | None = None,
    weights: FitnessWeights | None = None,
    output_dir: str | Path = "outputs/run",
    seed: int | None = None,
    make_plots: bool = True,
    verbose: bool = True,
    progress_callback: Callable[[dict], None] | None = None,
) -> tuple[GAResult, dict[str, str]]:
    """Run the GA, persist artefacts and render every required figure.

    Returns
    -------
    result, figures
        The :class:`GAResult` and a mapping of figure names to file paths.
    """

    mission = mission or MissionConfig()
    ga = ga or GAConfig()
    weights = weights or FitnessWeights()
    if seed is not None:
        ga = replace(ga, seed=seed)

    engine = GeneticStrategyEngine(mission, ga, weights)
    result = engine.run(verbose=verbose, progress_callback=progress_callback)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    result.save(out)

    figures: dict[str, str] = {}
    if make_plots:
        figures["mission"] = str(plot_mission(engine.env, out / "figures" / "mission_map.png"))
        figures["convergence"] = str(
            plot_convergence(result.history, out / "figures" / "fitness_convergence.png")
        )
        figures["paths"] = str(
            plot_paths(
                engine.env,
                result.best_genome,
                result.best_evaluation,
                out / "figures" / "evolved_paths.png",
            )
        )
        figures["coverage"] = str(
            plot_coverage(
                engine.env,
                result.best_evaluation,
                result.best_genome,
                out / "figures" / "coverage_heatmap.png",
            )
        )
        figures["coverage_evolution"] = str(
            plot_coverage_evolution(
                engine.env,
                engine.evaluator,
                result.snapshots,
                out / "figures" / "coverage_evolution.png",
            )
        )
        figures["pareto"] = str(
            plot_pareto(
                result.pareto_metrics,
                result.best_evaluation,
                out / "figures" / "pareto_front.png",
            )
        )
        (out / "figures.json").write_text(json.dumps(figures, indent=2), encoding="utf-8")

    if verbose:
        ev = result.best_evaluation
        lines = [
            "=== Evolved strategy ===",
            f"best fitness       : {ev.fitness:.4f}",
            f"priority coverage  : {ev.coverage:.3f}",
            f"target detection   : {ev.detection:.3f} (rate >= 0.5: {ev.detection_rate:.3f})",
            f"energy / budget    : {ev.energy:.2f} ({ev.energy_frac:.0%})",
            f"mission time       : {ev.time:.1f} ({ev.time_frac:.0%} of budget)",
            f"hazard exposure    : {ev.risk:.3f}",
            f"no-fly violations  : {ev.no_fly_fraction:.3f}",
            f"Pareto archive     : {len(result.pareto_genomes)} non-dominated strategies",
            f"evaluations / time : {result.evaluations} / {result.elapsed_seconds:.1f}s",
            f"artefacts          : {out}",
        ]
        print("\n" + "\n".join(lines))
        (out / "console_output.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result, figures
