"""Genetic algorithm that evolves the UAV deployment and patrol strategy."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .config import FitnessWeights, GAConfig, MissionConfig
from .environment import Environment
from .fitness import Evaluation, FitnessEvaluator
from .genome import Genome, random_genome
from .operators import crossover, mutate, repair
from .pareto import environmental_selection, pareto_front
from .selection import population_diversity, tournament_select


@dataclass
class GAResult:
    """Container for everything a simulation run produced."""

    best_genome: Genome
    best_evaluation: Evaluation
    history: list[dict]
    pareto_genomes: list[Genome]
    pareto_objectives: np.ndarray
    pareto_metrics: list[dict]
    mission: MissionConfig
    ga: GAConfig
    weights: FitnessWeights
    population: list[Genome]
    snapshots: list[tuple[int, Genome]]
    evaluations: int
    generations_run: int
    termination_reason: str
    elapsed_seconds: float
    config: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "best_fitness": self.best_evaluation.fitness,
            "best_metrics": self.best_evaluation.scalar_summary(),
            "evaluations": self.evaluations,
            "generations_run": self.generations_run,
            "termination_reason": self.termination_reason,
            "elapsed_seconds": self.elapsed_seconds,
            "mission": self.mission.to_dict(),
            "ga": self.ga.to_dict(),
            "weights": self.weights.to_dict(),
        }

    def save(self, directory: str | Path) -> Path:
        """Persist the best genome, score history, archetype and config."""

        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        (out / "best_genome.json").write_text(
            json.dumps(self.best_genome.to_dict(), indent=2), encoding="utf-8"
        )
        (out / "run_summary.json").write_text(
            json.dumps(self.to_dict(), indent=2), encoding="utf-8"
        )
        (out / "history.json").write_text(
            json.dumps(self.history, indent=2), encoding="utf-8"
        )
        pareto = [
            {
                "objectives": [float(x) for x in obj],
                "metrics": metrics,
                "genome": genome.to_dict(),
            }
            for genome, obj, metrics in zip(
                self.pareto_genomes, self.pareto_objectives, self.pareto_metrics
            )
        ]
        (out / "pareto_archive.json").write_text(
            json.dumps(pareto, indent=2), encoding="utf-8"
        )
        (out / "config.json").write_text(
            json.dumps(self.config, indent=2), encoding="utf-8"
        )
        return out


class GeneticStrategyEngine:
    """Multi-objective GA with hybrid chromosomes and spatial operators.

    Parameters
    ----------
    mission, ga, weights
        Immutable configuration objects.
    environment
        Optional pre-built environment (useful for fair A/B experiments).

    Notes
    -----
    ``mode='weighted'`` uses a steady-state generational GA with elite
    preservation and random immigrants.  ``mode='pareto'`` replaces survival
    with NSGA-II non-dominated sorting and crowding distance.  In both modes a
    Pareto archive is maintained so the final trade-off surface can be
    inspected.
    """

    def __init__(
        self,
        mission: MissionConfig | None = None,
        ga: GAConfig | None = None,
        weights: FitnessWeights | None = None,
        environment: Environment | None = None,
    ) -> None:
        self.mission = mission or MissionConfig()
        self.ga = ga or GAConfig()
        self.weights = weights or FitnessWeights()
        self.env = environment or Environment(self.mission)
        self.evaluator = FitnessEvaluator(self.env, self.mission, self.weights)
        seed = (self.ga.seed * 1_000_003 + self.mission.seed) % (2**32 - 1)
        self.rng = np.random.default_rng(seed)
        self.evaluations = 0

        self._archive_genomes: list[Genome] = []
        self._archive_objectives: list[tuple[float, ...]] = []
        self._archive_metrics: list[dict] = []

    # ------------------------------------------------------------- evaluation
    def _evaluate(self, genome: Genome, progress: float) -> Evaluation:
        self.evaluations += 1
        return self.evaluator.evaluate(genome, progress=progress)

    # ------------------------------------------------------------- population
    def _initial_population(self) -> tuple[list[Genome], list[Evaluation]]:
        n = self.ga.population_size
        pop: list[Genome] = []
        # One hotspot-aware seed gives the search a useful starting point;
        # the rest are random to preserve exploration.
        pop.append(repair(random_genome(self.env, self.mission, self.rng, heuristic=True), self.env, self.mission))
        while len(pop) < n:
            g = random_genome(self.env, self.mission, self.rng, heuristic=False)
            pop.append(repair(g, self.env, self.mission))
        evaluations = [self._evaluate(g, 0.0) for g in pop]
        return pop, evaluations

    def _offspring(
        self,
        pop: list[Genome],
        fitness: np.ndarray,
        count: int,
        progress: float,
        intensity: float,
    ) -> list[tuple[Genome, Evaluation]]:
        children: list[tuple[Genome, Evaluation]] = []
        n = len(pop)
        while len(children) < count:
            p1 = pop[tournament_select(fitness, self.rng, self.ga.tournament_size)]
            p2 = pop[tournament_select(fitness, self.rng, self.ga.tournament_size)]
            if self.rng.random() < self.ga.crossover_rate:
                child = crossover(p1, p2, self.rng, self.mission)
            else:
                child = p1.copy()
            child = mutate(
                child,
                self.rng,
                self.mission,
                rate=self.ga.mutation_rate,
                sigma=self.ga.mutation_sigma,
                intensity=intensity,
            )
            child = repair(child, self.env, self.mission)
            children.append((child, self._evaluate(child, progress)))
        return children

    def _update_archive(
        self, genomes: list[Genome], evaluations: list[Evaluation]
    ) -> None:
        """Maintain the non-dominated set seen so far."""

        # Drop archived solutions dominated by any newcomer.
        keep = [True] * len(self._archive_objectives)
        for ev in evaluations:
            for k, old in enumerate(self._archive_objectives):
                if not keep[k]:
                    continue
                if _dominates(ev.objectives, old):
                    keep[k] = False
        self._archive_genomes = [g for g, k in zip(self._archive_genomes, keep) if k]
        self._archive_objectives = [o for o, k in zip(self._archive_objectives, keep) if k]
        self._archive_metrics = [m for m, k in zip(self._archive_metrics, keep) if k]

        for genome, ev in zip(genomes, evaluations):
            if any(_dominates(old, ev.objectives) for old in self._archive_objectives):
                continue
            if any(np.allclose(old, ev.objectives) for old in self._archive_objectives):
                continue
            self._archive_genomes.append(genome.copy())
            self._archive_objectives.append(ev.objectives)
            self._archive_metrics.append(
                {
                    "fitness": ev.fitness,
                    "coverage": ev.coverage,
                    "detection": ev.detection,
                    "energy": ev.energy,
                    "time": ev.time,
                    "risk": ev.risk,
                    "penalty": ev.penalty,
                }
            )
        if len(self._archive_objectives) > 150:
            # Crowding-based thinning keeps the extremes and the sparse middle.
            from .pareto import crowding_distance

            obj = np.asarray(self._archive_objectives)
            dist = crowding_distance(list(range(len(obj))), obj)
            keep_idx = np.argsort(-dist)[:150]
            self._archive_genomes = [self._archive_genomes[i] for i in keep_idx]
            self._archive_objectives = [self._archive_objectives[i] for i in keep_idx]
            self._archive_metrics = [self._archive_metrics[i] for i in keep_idx]

    # -------------------------------------------------------------------- run
    def run(
        self,
        verbose: bool = False,
        progress_callback: Callable[[dict], None] | None = None,
    ) -> GAResult:
        start = time.perf_counter()
        self.evaluations = 0
        self._archive_genomes, self._archive_objectives, self._archive_metrics = [], [], []
        cfg = self.ga

        pop, evaluations = self._initial_population()
        fitness = np.array([e.fitness for e in evaluations])
        self._update_archive(pop, evaluations)

        history: list[dict] = []
        best_idx = int(np.argmax(fitness))
        best_genome = pop[best_idx].copy()
        best_eval = evaluations[best_idx]
        best_fitness = float(best_eval.fitness)
        if progress_callback is not None:
            progress_callback(
                {
                    "generation": 0,
                    "generations": cfg.generations,
                    "best": best_fitness,
                    "average": float(np.mean(fitness)),
                    "coverage": best_eval.coverage,
                    "detection": best_eval.detection,
                    "energy_frac": best_eval.energy_frac,
                    "risk": best_eval.risk,
                    "diversity": 0.0,
                    "progress": 0.0,
                }
            )
        stall = 0
        termination = "max_generations"
        generation = 0
        snapshots: list[tuple[int, Genome]] = [(0, best_genome.copy())]
        checkpoints = {
            max(1, cfg.generations // 4),
            max(1, cfg.generations // 2),
            max(1, 3 * cfg.generations // 4),
            cfg.generations,
        }

        for generation in range(1, cfg.generations + 1):
            progress = (generation - 1) / max(cfg.generations - 1, 1)
            intensity = 1.0 + 2.5 * min(stall / max(cfg.stall_limit, 1), 1.0)

            if cfg.mode == "weighted":
                n_immigrants = max(1, int(cfg.population_size * cfg.immigrant_fraction))
                n_children = max(1, cfg.population_size - cfg.elitism - n_immigrants)
                children = self._offspring(
                    pop, fitness, n_children, progress, intensity
                )
                elite_idx = np.argsort(-fitness)[: cfg.elitism]
                next_pop = [pop[i].copy() for i in elite_idx]
                next_eval = [evaluations[i] for i in elite_idx]
                for g, e in children:
                    next_pop.append(g)
                    next_eval.append(e)
                for _ in range(n_immigrants):
                    g = repair(
                        random_genome(self.env, self.mission, self.rng, heuristic=False),
                        self.env,
                        self.mission,
                    )
                    next_pop.append(g)
                    next_eval.append(self._evaluate(g, progress))
                pop, evaluations = next_pop, next_eval
            else:
                children = self._offspring(
                    pop, fitness, cfg.population_size, progress, intensity
                )
                combined_pop = pop + [g for g, _ in children]
                combined_eval = evaluations + [e for _, e in children]
                objectives = np.array([e.objectives for e in combined_eval])
                survivors = environmental_selection(objectives, cfg.population_size)
                pop = [combined_pop[i] for i in survivors]
                evaluations = [combined_eval[i] for i in survivors]

            # Diversity injection when progress stalls: mutate copies of the
            # best solution rather than replacing members with pure randomness.
            if stall >= cfg.stall_limit and stall % cfg.stall_limit == 0:
                n_reset = max(2, int(0.08 * cfg.population_size))
                worst = np.argsort([e.fitness for e in evaluations])[:n_reset]
                for i in worst:
                    g = mutate(
                        best_genome.copy(),
                        self.rng,
                        self.mission,
                        rate=min(0.9, cfg.mutation_rate * 2.5),
                        sigma=cfg.mutation_sigma * 1.5,
                        intensity=2.0,
                    )
                    g = repair(g, self.env, self.mission)
                    pop[int(i)] = g
                    evaluations[int(i)] = self._evaluate(g, progress)

            fitness = np.array([e.fitness for e in evaluations])
            current_best = int(np.argmax(fitness))
            if fitness[current_best] > best_fitness + 1e-6:
                best_fitness = float(fitness[current_best])
                best_genome = pop[current_best].copy()
                best_eval = evaluations[current_best]
                stall = 0
            else:
                stall += 1

            self._update_archive(pop, evaluations)
            diversity = population_diversity(pop, self.mission.grid_size)
            record = {
                "generation": generation,
                "best": float(np.max(fitness)),
                "average": float(np.mean(fitness)),
                "worst": float(np.min(fitness)),
                "diversity": diversity,
                "coverage": best_eval.coverage,
                "detection": best_eval.detection,
                "energy_frac": best_eval.energy_frac,
                "time_frac": best_eval.time_frac,
                "risk": best_eval.risk,
                "penalty": best_eval.penalty,
                "n_active": best_eval.n_active,
                "stall": stall,
                "evaluations": self.evaluations,
            }
            history.append(record)
            if generation in checkpoints:
                snapshots.append((generation, best_genome.copy()))
            if progress_callback is not None:
                progress_callback(
                    {
                        **record,
                        "generations": cfg.generations,
                        "progress": generation / max(cfg.generations, 1),
                    }
                )

            if verbose and (generation % 10 == 0 or generation == 1):
                print(
                    f"gen {generation:3d} | best {record['best']:.4f} "
                    f"| avg {record['average']:.4f} | cov {record['coverage']:.3f} "
                    f"| energy {record['energy_frac']:.2f} | risk {record['risk']:.3f} "
                    f"| div {diversity:.4f}"
                )
            if stall >= cfg.stall_limit:
                termination = "stall"
                break

        elapsed = time.perf_counter() - start
        # Always report the best-ever chromosome, not merely the last population.
        final_eval = self.evaluator.evaluate(best_genome, progress=1.0, keep_map=True)
        self.evaluations += 1

        # Final Pareto archive (for plotting and the report).
        archive_obj = np.array(self._archive_objectives) if self._archive_objectives else np.empty((0, 5))
        if len(archive_obj) > 0:
            front = pareto_front(archive_obj)
            pareto_genomes = [self._archive_genomes[i] for i in front]
            pareto_objectives = archive_obj[front]
            pareto_metrics = [self._archive_metrics[i] for i in front]
        else:
            pareto_genomes, pareto_objectives, pareto_metrics = [], np.empty((0, 5)), []

        return GAResult(
            best_genome=best_genome,
            best_evaluation=final_eval,
            history=history,
            pareto_genomes=pareto_genomes,
            pareto_objectives=pareto_objectives,
            pareto_metrics=pareto_metrics,
            mission=self.mission,
            ga=self.ga,
            weights=self.weights,
            population=pop,
            snapshots=snapshots,
            evaluations=self.evaluations,
            generations_run=generation,
            termination_reason=termination,
            elapsed_seconds=elapsed,
            config={
                "mission": self.mission.to_dict(),
                "ga": self.ga.to_dict(),
                "weights": self.weights.to_dict(),
                "seed": self.ga.seed,
            },
        )


def _dominates(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
    a_arr = np.asarray(a)
    b_arr = np.asarray(b)
    return bool(np.all(a_arr <= b_arr) and np.any(a_arr < b_arr))
