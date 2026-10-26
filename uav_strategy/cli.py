"""Command-line interface for the UAV strategy engine."""

from __future__ import annotations

import argparse
import json

from .config import ExperimentConfig, FitnessWeights, GAConfig, MissionConfig
from .experiments import run_all_experiments
from .simulation import run_simulation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="uav-strategy",
        description="Genetic-algorithm strategy engine for disaster-surveillance UAV fleets.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run one GA simulation and render all figures.")
    run.add_argument("--output", default="outputs/run", help="Output directory.")
    run.add_argument("--population", type=int, default=60)
    run.add_argument("--generations", type=int, default=80)
    run.add_argument("--mutation-rate", type=float, default=0.20)
    run.add_argument("--tournament-size", type=int, default=4)
    run.add_argument("--mode", choices=["weighted", "pareto"], default="weighted")
    run.add_argument("--seed", type=int, default=7)
    run.add_argument("--grid-size", type=int, default=50)
    run.add_argument("--drones", type=int, default=5)
    run.add_argument("--no-plots", action="store_true")

    exp = sub.add_parser("experiments", help="Run sensitivity analysis and baseline comparison.")
    exp.add_argument("--output", default="outputs/experiments")
    exp.add_argument("--generations", type=int, default=40)
    exp.add_argument("--seeds", type=int, nargs="+", default=[11, 23])
    exp.add_argument("--quick", action="store_true", help="Single seed, fewer generations.")

    sub.add_parser("report", help="Explain how to build the PDF report.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "run":
        mission = MissionConfig(grid_size=args.grid_size, n_drones=args.drones)
        ga = GAConfig(
            population_size=args.population,
            generations=args.generations,
            mutation_rate=args.mutation_rate,
            tournament_size=args.tournament_size,
            mode=args.mode,
            seed=args.seed,
        )
        result, figures = run_simulation(
            mission=mission,
            ga=ga,
            weights=FitnessWeights(),
            output_dir=args.output,
            make_plots=not args.no_plots,
            verbose=True,
        )
        print(json.dumps({"figures": figures}, indent=2))
        return 0

    if args.command == "experiments":
        if args.quick:
            exp = ExperimentConfig(
                seeds=(11,),
                generations=25,
                population_sizes=(24, 40),
                mutation_rates=(0.05, 0.20),
                tournament_sizes=(2, 4),
                output_dir=args.output,
            )
        else:
            exp = ExperimentConfig(
                seeds=tuple(args.seeds),
                generations=args.generations,
                output_dir=args.output,
            )
        run_all_experiments(exp, verbose=True)
        return 0

    if args.command == "report":
        print(
            "Run the experiments first, then build the report:\n"
            "  python -m uav_strategy experiments\n"
            "  python scripts/build_report.py\n"
            "This produces REPORT.md, docs/TECHNICAL_REPORT.docx and docs/TECHNICAL_REPORT.pdf."
        )
        return 0
    return 1
