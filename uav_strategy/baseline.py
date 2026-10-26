"""Random-search baseline for a fair comparison against the evolved strategy."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .config import FitnessWeights, MissionConfig
from .environment import Environment
from .fitness import Evaluation, FitnessEvaluator
from .genome import Genome, random_genome
from .operators import repair


@dataclass
class BaselineResult:
    """Best solution and convergence trace of the random-search baseline."""

    best_genome: Genome
    best_evaluation: Evaluation
    history: list[dict]
    evaluations: int
    elapsed_seconds: float
    fitness_samples: list[float] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "best_fitness": self.best_evaluation.fitness,
            "best_metrics": self.best_evaluation.scalar_summary(),
            "evaluations": self.evaluations,
            "elapsed_seconds": self.elapsed_seconds,
            "history": self.history,
            "fitness_samples": self.fitness_samples,
        }


def run_random_search(
    env: Environment,
    mission: MissionConfig,
    weights: FitnessWeights,
    n_evaluations: int,
    seed: int = 0,
    record_every: int | None = None,
) -> BaselineResult:
    """Blind sampling of the same chromosome space.

    Every candidate receives the same repair operator and evaluator as the GA
    so that the only difference is the search strategy (random draw versus
    selection + recombination + mutation).
    """

    rng = np.random.default_rng(seed + 999)
    evaluator = FitnessEvaluator(env, mission, weights)
    if record_every is None:
        record_every = max(1, n_evaluations // 60)

    best_genome: Genome | None = None
    best_eval: Evaluation | None = None
    best_fitness = -np.inf
    history: list[dict] = []
    samples: list[float] = []
    start = time.perf_counter()

    for k in range(1, n_evaluations + 1):
        genome = repair(
            random_genome(env, mission, rng, heuristic=False), env, mission
        )
        ev = evaluator.evaluate(genome)
        samples.append(ev.fitness)
        if ev.fitness > best_fitness:
            best_fitness = ev.fitness
            best_genome = genome
            best_eval = ev
        if k % record_every == 0 or k == n_evaluations:
            history.append(
                {
                    "evaluations": k,
                    "best": float(best_fitness),
                    "average": float(np.mean(samples[-record_every:])),
                }
            )

    assert best_genome is not None and best_eval is not None
    return BaselineResult(
        best_genome=best_genome,
        best_evaluation=evaluator.evaluate(best_genome, keep_map=True),
        history=history,
        evaluations=n_evaluations,
        elapsed_seconds=time.perf_counter() - start,
        fitness_samples=samples,
    )
