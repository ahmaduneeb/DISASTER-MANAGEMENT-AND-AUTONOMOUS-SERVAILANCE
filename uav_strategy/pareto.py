"""NSGA-II helpers: non-dominated sorting, crowding distance and survival."""

from __future__ import annotations

import numpy as np


def dominates(a: tuple[float, ...] | np.ndarray, b: tuple[float, ...] | np.ndarray) -> bool:
    """True if ``a`` Pareto-dominates ``b`` (all objectives minimised)."""

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return bool(np.all(a <= b) and np.any(a < b))


def fast_non_dominated_sort(objectives: np.ndarray) -> list[list[int]]:
    """Return Pareto fronts, best first. Complexity ``O(M N^2)``."""

    n = len(objectives)
    domination_count = np.zeros(n, dtype=int)
    dominated: list[list[int]] = [[] for _ in range(n)]
    fronts: list[list[int]] = [[]]

    for i in range(n):
        for j in range(i + 1, n):
            if dominates(objectives[i], objectives[j]):
                dominated[i].append(j)
                domination_count[j] += 1
            elif dominates(objectives[j], objectives[i]):
                dominated[j].append(i)
                domination_count[i] += 1
    for i in range(n):
        if domination_count[i] == 0:
            fronts[0].append(i)

    current = 0
    while fronts[current]:
        nxt: list[int] = []
        for i in fronts[current]:
            for j in dominated[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    nxt.append(j)
        current += 1
        fronts.append(nxt)
    return fronts[:-1]


def crowding_distance(front: list[int], objectives: np.ndarray) -> np.ndarray:
    """Deb et al.'s crowding distance for one front."""

    if not front:
        return np.array([])
    length = len(front)
    if length <= 2:
        return np.full(length, np.inf)
    obj = objectives[front]
    distances = np.zeros(length)
    for m in range(obj.shape[1]):
        order = np.argsort(obj[:, m])
        distances[order[0]] = np.inf
        distances[order[-1]] = np.inf
        span = obj[order[-1], m] - obj[order[0], m]
        if span <= 1e-12:
            continue
        for k in range(1, length - 1):
            distances[order[k]] += (
                obj[order[k + 1], m] - obj[order[k - 1], m]
            ) / span
    return distances


def environmental_selection(
    objectives: np.ndarray, n_survivors: int
) -> list[int]:
    """NSGA-II truncation: keep full fronts, then crowding-distance fill."""

    fronts = fast_non_dominated_sort(objectives)
    selected: list[int] = []
    for front in fronts:
        if len(selected) + len(front) <= n_survivors:
            selected.extend(front)
        else:
            remaining = n_survivors - len(selected)
            distances = crowding_distance(front, objectives)
            keep = np.argsort(-distances)[:remaining]
            selected.extend(front[int(i)] for i in keep)
            break
    return selected


def pareto_front(objectives: np.ndarray) -> list[int]:
    """Indices of the first Pareto front."""

    return fast_non_dominated_sort(objectives)[0]
