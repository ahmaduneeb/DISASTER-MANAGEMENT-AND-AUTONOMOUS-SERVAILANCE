"""Parent-selection mechanisms and diversity helpers."""

from __future__ import annotations

import numpy as np

from .genome import Genome


def tournament_select(
    fitness: np.ndarray, rng: np.random.Generator, size: int = 4
) -> int:
    """Return the index of the fittest individual in a random tournament."""

    contenders = rng.integers(0, len(fitness), size=size)
    return int(contenders[np.argmax(fitness[contenders])])


def rank_select(fitness: np.ndarray, rng: np.random.Generator) -> int:
    """Linear-rank selection: weakest still has a non-zero chance."""

    order = np.argsort(fitness)
    ranks = np.empty(len(fitness), dtype=float)
    ranks[order] = np.arange(1, len(fitness) + 1)
    # P(rank r) proportional to r, so the best is n times likelier than worst.
    probs = ranks / ranks.sum()
    return int(rng.choice(len(fitness), p=probs))


def roulette_select(fitness: np.ndarray, rng: np.random.Generator) -> int:
    """Fitness-proportional selection with a non-negative shift."""

    shifted = fitness - fitness.min() + 1e-6
    probs = shifted / shifted.sum()
    return int(rng.choice(len(fitness), p=probs))


def population_diversity(population: list[Genome], grid_size: int) -> float:
    """Mean pairwise genotypic distance normalised to ``[0, 1]``.

    Spatial genes are divided by the grid size, sensor power by 1 and speed by
    its range, so all segments contribute comparably.
    """

    if len(population) < 2:
        return 0.0
    vectors = []
    for g in population:
        v = np.concatenate(
            [
                g.deploy.ravel() / grid_size,
                g.patrol.ravel() / grid_size,
                g.sensor.ravel(),
                (g.speed.ravel() - 0.55) / 1.45,
                g.active.astype(float),
            ]
        )
        vectors.append(v)
    data = np.vstack(vectors)
    # Unbiased sample of pairs keeps the metric cheap for large populations.
    n = len(data)
    pairs = min(400, n * (n - 1) // 2)
    i = np.random.default_rng(0).integers(0, n, size=pairs)
    j = np.random.default_rng(1).integers(0, n, size=pairs)
    keep = i != j
    return float(np.mean(np.abs(data[i[keep]] - data[j[keep]])))
