"""Sensitivity analysis and equal-budget baseline experiment."""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from .baseline import run_random_search
from .config import ExperimentConfig, GAConfig
from .ga import GeneticStrategyEngine
from .visualize import plot_baseline_comparison, plot_sensitivity, plot_sensitivity_summary

METRIC_KEYS = (
    "fitness",
    "coverage",
    "detection",
    "detection_rate",
    "energy_frac",
    "time_frac",
    "risk",
    "penalty",
)


def _summarise(runs: list[dict]) -> dict:
    summary = {}
    for key in METRIC_KEYS:
        values = np.array([r[key] for r in runs], dtype=float)
        summary[key] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return summary


def run_sensitivity(exp: ExperimentConfig | None = None, verbose: bool = True) -> dict:
    """One-at-a-time study of population size, mutation rate and selection
    pressure on the *same* environment."""

    exp = exp or ExperimentConfig()
    out = Path(exp.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    base_ga = replace(exp.ga, generations=exp.generations)
    studies = {
        "population_size": (
            list(exp.population_sizes),
            lambda v: {"population_size": int(v)},
        ),
        "mutation_rate": (
            list(exp.mutation_rates),
            lambda v: {"mutation_rate": float(v)},
        ),
        "tournament_size": (
            list(exp.tournament_sizes),
            lambda v: {"tournament_size": int(v)},
        ),
    }

    all_runs: dict[str, list[dict]] = {}
    summaries: dict[str, dict] = {}
    for name, (values, make_kwargs) in studies.items():
        runs: list[dict] = []
        for value in values:
            for seed in exp.seeds:
                ga = replace(base_ga, seed=seed, **make_kwargs(value))
                engine = GeneticStrategyEngine(exp.mission, ga, exp.weights)
                result = engine.run(verbose=False)
                ev = result.best_evaluation
                runs.append(
                    {
                        "parameter": name,
                        "value": float(value),
                        "seed": seed,
                        "fitness": ev.fitness,
                        "coverage": ev.coverage,
                        "detection": ev.detection,
                        "detection_rate": ev.detection_rate,
                        "energy_frac": ev.energy_frac,
                        "time_frac": ev.time_frac,
                        "risk": ev.risk,
                        "penalty": ev.penalty,
                        "generations_run": result.generations_run,
                        "evaluations": result.evaluations,
                        "elapsed_seconds": result.elapsed_seconds,
                    }
                )
                if verbose:
                    print(
                        f"[{name}={value}] seed {seed}: fitness {ev.fitness:.3f}, "
                        f"coverage {ev.coverage:.3f}, risk {ev.risk:.3f}"
                    )
        all_runs[name] = runs
        summaries[name] = {
            str(value): _summarise([r for r in runs if r["value"] == float(value)])
            for value in values
        }

    (out / "sensitivity_runs.json").write_text(
        json.dumps(all_runs, indent=2), encoding="utf-8"
    )
    (out / "sensitivity_summary.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )
    for name, (values, _) in studies.items():
        plot_sensitivity(
            values,
            all_runs[name],
            {
                "population_size": "Population size",
                "mutation_rate": "Mutation rate",
                "tournament_size": "Tournament size (selection pressure)",
            }[name],
            out / f"sensitivity_{name}.png",
        )
    plot_sensitivity_summary(summaries, out / "sensitivity_combined.png")
    return {"runs": all_runs, "summary": summaries}


def run_baseline_study(exp: ExperimentConfig | None = None, verbose: bool = True) -> dict:
    """Compare the GA against blind random search at an equal budget."""

    exp = exp or ExperimentConfig()
    out = Path(exp.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    base_ga = replace(exp.ga, generations=exp.generations)
    n_eval = base_ga.population_size * base_ga.generations

    ga_final: list[dict] = []
    ga_histories: list[list[dict]] = []
    base_final: list[dict] = []
    base_histories: list[list[dict]] = []
    detail: dict[str, list] = {"ga": [], "baseline": []}

    for seed in exp.seeds:
        ga = replace(base_ga, seed=seed)
        engine = GeneticStrategyEngine(exp.mission, ga, exp.weights)
        result = engine.run(verbose=False)
        ev = result.best_evaluation
        ga_final.append(
            {
                "seed": seed,
                "fitness": ev.fitness,
                "coverage": ev.coverage,
                "detection": ev.detection,
                "detection_rate": ev.detection_rate,
                "energy_frac": ev.energy_frac,
                "time_frac": ev.time_frac,
                "risk": ev.risk,
                "penalty": ev.penalty,
                "evaluations": result.evaluations,
                "generations": result.generations_run,
                "elapsed_seconds": result.elapsed_seconds,
            }
        )
        ga_histories.append(result.history)
        detail["ga"].append(
            {
                "seed": seed,
                "history": result.history,
                "best_metrics": ev.scalar_summary(),
                "termination": result.termination_reason,
            }
        )

        baseline = run_random_search(
            engine.env,
            exp.mission,
            exp.weights,
            n_evaluations=n_eval,
            seed=seed,
        )
        bev = baseline.best_evaluation
        base_final.append(
            {
                "seed": seed,
                "fitness": bev.fitness,
                "coverage": bev.coverage,
                "detection": bev.detection,
                "detection_rate": bev.detection_rate,
                "energy_frac": bev.energy_frac,
                "time_frac": bev.time_frac,
                "risk": bev.risk,
                "penalty": bev.penalty,
                "evaluations": baseline.evaluations,
                "elapsed_seconds": baseline.elapsed_seconds,
            }
        )
        base_histories.append(baseline.history)
        detail["baseline"].append(
            {"seed": seed, "history": baseline.history, "best_metrics": bev.scalar_summary()}
        )
        if verbose:
            print(
                f"seed {seed}: GA {ev.fitness:.3f} (cov {ev.coverage:.3f}) vs "
                f"random {bev.fitness:.3f} (cov {bev.coverage:.3f})"
            )

    summary = {
        "budget_evaluations": n_eval,
        "generations": exp.generations,
        "seeds": list(exp.seeds),
        "ga": _summarise(ga_final),
        "baseline": _summarise(base_final),
        "comparison": {
            key: {
                "ga": float(np.mean([r[key] for r in ga_final])),
                "baseline": float(np.mean([r[key] for r in base_final])),
                "relative_gain": float(
                    (
                        np.mean([r[key] for r in ga_final])
                        - np.mean([r[key] for r in base_final])
                    )
                    / max(abs(np.mean([r[key] for r in base_final])), 1e-9)
                ),
            }
            for key in METRIC_KEYS
        },
    }
    (out / "baseline_comparison.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (out / "baseline_detail.json").write_text(
        json.dumps(detail, indent=2), encoding="utf-8"
    )
    plot_baseline_comparison(
        ga_histories,
        base_histories,
        ga_final,
        base_final,
        out / "baseline_vs_ga.png",
    )
    return summary


def run_all_experiments(exp: ExperimentConfig | None = None, verbose: bool = True) -> dict:
    """Run the complete study and write a compact summary bundle."""

    exp = exp or ExperimentConfig()
    start = time.perf_counter()
    sensitivity = run_sensitivity(exp, verbose=verbose)
    baseline = run_baseline_study(exp, verbose=verbose)
    payload = {
        "config": exp.to_dict(),
        "sensitivity": sensitivity["summary"],
        "baseline": baseline,
        "elapsed_seconds": time.perf_counter() - start,
    }
    out = Path(exp.output_dir)
    (out / "experiment_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    if verbose:
        print(f"\nExperiments finished in {payload['elapsed_seconds']:.1f}s -> {out}")
    return payload
