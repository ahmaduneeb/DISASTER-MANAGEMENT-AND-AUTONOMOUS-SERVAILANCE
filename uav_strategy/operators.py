"""Spatial crossover, mutation and constraint-repair operators."""

from __future__ import annotations

import numpy as np

from .config import MissionConfig
from .environment import Environment
from .genome import SPEED_MAX, SPEED_MIN, Genome, random_genome


# --------------------------------------------------------------------- repair
def repair(genome: Genome, env: Environment, cfg: MissionConfig) -> Genome:
    """Repair hard-constraint violations while preserving as much genotype as
    possible.

    Steps
    -----
    1. Clip coordinates to the map and speed/sensor genes to their limits.
    2. Project waypoints that fall inside no-fly zones to the nearest free cell.
    3. Contract the patrol path toward the base until the closed flight range
       is within ``max_range``.
    4. Re-project any waypoint pushed into an obstacle by the contraction.
    """

    g = genome.copy()
    lo, hi = env.bounds
    g.active = (g.active > 0).astype(np.int8)
    g.deploy = np.clip(g.deploy, lo, hi)
    g.patrol = np.clip(g.patrol, lo, hi)
    g.sensor = np.clip(g.sensor, 0.0, 1.0)
    g.speed = np.clip(g.speed, SPEED_MIN, SPEED_MAX)

    for i in range(g.n_drones):
        g.deploy[i] = env.nearest_free(g.deploy[i])
        for j in range(g.n_patrol):
            g.patrol[i, j] = env.nearest_free(g.patrol[i, j])

    base = np.asarray(cfg.base, dtype=float)

    def drone_distance(gen: Genome, i: int) -> float:
        if not gen.active[i]:
            return 0.0
        return float(np.linalg.norm(np.diff(gen.path(i, cfg.base), axis=0), axis=1).sum())

    # Contract only the offending drone; a route violation on one UAV must not
    # shrink the entire fleet's coverage.
    for _ in range(14):
        violations = [
            i
            for i in range(g.n_drones)
            if g.active[i] and drone_distance(g, i) > cfg.max_range
        ]
        if not violations:
            break
        for i in violations:
            distance = drone_distance(g, i)
            if distance < 1e-9:
                continue
            factor = 0.97 * cfg.max_range / distance
            g.deploy[i] = base + (g.deploy[i] - base) * factor
            g.patrol[i] = base + (g.patrol[i] - base) * factor
            g.deploy[i] = np.clip(g.deploy[i], lo, hi)
            g.patrol[i] = np.clip(g.patrol[i], lo, hi)
            g.deploy[i] = env.nearest_free(g.deploy[i])
            for j in range(g.n_patrol):
                g.patrol[i, j] = env.nearest_free(g.patrol[i, j])

    # Extreme fallback: bring stranded drones home.
    for i in range(g.n_drones):
        if g.active[i] and drone_distance(g, i) > cfg.max_range * 1.01:
            g.deploy[i] = base
            g.patrol[i] = base
    return g


# ------------------------------------------------------------------ crossover
def geometric_crossover(
    p1: Genome, p2: Genome, rng: np.random.Generator, cfg: MissionConfig
) -> Genome:
    """Blend coordinates of spatially related genes.

    A per-drone blend weight keeps each route's shape coherent (rather than
    turning every waypoint into an unrelated average), which is important for
    path planning where adjacent waypoints are correlated.
    """

    child = p1.copy()
    n = p1.n_drones

    # Deployment: per-drone convex combination.
    alpha = rng.uniform(0.0, 1.0, size=(n, 1))
    child.deploy = alpha * p1.deploy + (1 - alpha) * p2.deploy
    # Patrol: BLX-alpha around each pair of waypoints.
    for i in range(n):
        w = rng.uniform(-0.2, 1.2)
        child.patrol[i] = np.clip(
            w * p1.patrol[i] + (1 - w) * p2.patrol[i], 0.0, cfg.grid_size - 1
        )

    # Sensor and speed genes use uniform crossover to avoid averaging away
    # extreme (high-power / low-speed) strategies.
    mask = rng.random(size=p1.sensor.shape) < 0.5
    child.sensor = np.where(mask, p1.sensor, p2.sensor)
    mask = rng.random(size=p1.speed.shape) < 0.5
    child.speed = np.where(mask, p1.speed, p2.speed)
    mask = rng.random(size=n) < 0.5
    child.active = np.where(mask, p1.active, p2.active)
    return child


def segment_crossover(
    p1: Genome, p2: Genome, rng: np.random.Generator, cfg: MissionConfig
) -> Genome:
    """Swap entire drone mission segments (deploy + patrol + schedule)."""

    child = p1.copy()
    swapped = False
    for i in range(child.n_drones):
        if rng.random() < 0.5:
            swapped = True
            child.active[i] = p2.active[i]
            child.deploy[i] = p2.deploy[i]
            child.patrol[i] = p2.patrol[i]
            child.sensor[i] = p2.sensor[i]
            child.speed[i] = p2.speed[i]
    if not swapped:
        i = int(rng.integers(0, child.n_drones))
        child.deploy[i] = p2.deploy[i]
        child.patrol[i] = p2.patrol[i]
        child.sensor[i] = p2.sensor[i]
        child.speed[i] = p2.speed[i]
    return child


def route_splice_crossover(
    p1: Genome, p2: Genome, rng: np.random.Generator, cfg: MissionConfig
) -> Genome:
    """Cut-and-splice patrol routes per drone (geometric order crossover)."""

    child = p1.copy()
    for i in range(child.n_drones):
        cut = int(rng.integers(0, p1.n_patrol + 1))
        child.patrol[i] = np.vstack([p1.patrol[i, :cut], p2.patrol[i, cut:]])
        child.deploy[i] = p1.deploy[i] if rng.random() < 0.5 else p2.deploy[i]
        mask = rng.random(size=p1.n_patrol + 1) < 0.5
        child.sensor[i] = np.where(mask, p1.sensor[i], p2.sensor[i])
        mask = rng.random(size=p1.n_patrol + 2) < 0.5
        child.speed[i] = np.where(mask, p1.speed[i], p2.speed[i])
        child.active[i] = p1.active[i] if rng.random() < 0.5 else p2.active[i]
    return child


def crossover(
    p1: Genome,
    p2: Genome,
    rng: np.random.Generator,
    cfg: MissionConfig,
    style: str | None = None,
) -> Genome:
    """Dispatch to one of the three custom crossover operators."""

    if style is None:
        style = str(rng.choice(["geometric", "segment", "splice"], p=[0.4, 0.3, 0.3]))
    if style == "geometric":
        return geometric_crossover(p1, p2, rng, cfg)
    if style == "segment":
        return segment_crossover(p1, p2, rng, cfg)
    if style == "splice":
        return route_splice_crossover(p1, p2, rng, cfg)
    raise ValueError(f"unknown crossover style: {style}")


# ------------------------------------------------------------------- mutation
def _jitter(values: np.ndarray, mask: np.ndarray, sigma: float, rng, lo, hi) -> np.ndarray:
    noise = rng.normal(0.0, sigma, size=values.shape)
    return np.clip(values + noise * mask, lo, hi)


def mutate(
    genome: Genome,
    rng: np.random.Generator,
    cfg: MissionConfig,
    rate: float,
    sigma: float = 1.6,
    intensity: float = 1.0,
) -> Genome:
    """Spatially aware mutation with three complementary moves.

    * Gaussian waypoint jitter (fine local search).
    * Random relocation of a waypoint to a free cell (long-range exploration).
    * Route reversal / rotation and sensor-speed schedule changes (structural
      diversity without breaking the chromosome layout).

    ``intensity`` allows the GA to boost exploration when progress stalls.
    """

    g = genome.copy()
    rate = float(np.clip(rate * intensity, 0.0, 1.0))
    n, k = g.n_drones, g.n_patrol

    # The public mutation rate is an individual-level control.  Damping it
    # per gene prevents ~20% of a 200+ gene chromosome being destroyed at once.
    point_rate = rate * 0.30
    schedule_rate = rate * 0.40

    # Deployment jitter.
    mask = rng.random((n, 2)) < point_rate
    g.deploy = _jitter(g.deploy, mask, sigma, rng, 0, cfg.grid_size - 1)

    # Per-waypoint local jitter and occasional teleport.
    mask = rng.random((n, k, 2)) < point_rate
    g.patrol = _jitter(g.patrol, mask, sigma, rng, 0, cfg.grid_size - 1)
    teleport = rng.random((n, k)) < point_rate * 0.15
    for i, j in zip(*np.nonzero(teleport)):
        g.patrol[i, j] = rng.uniform(0, cfg.grid_size - 1, size=2)

    # Structural route moves.
    for i in range(n):
        if rng.random() < rate * 0.5:
            g.patrol[i] = g.patrol[i][::-1]
        elif rng.random() < rate * 0.35:
            shift = int(rng.integers(1, k)) if k > 1 else 0
            g.patrol[i] = np.roll(g.patrol[i], shift, axis=0)

    # Sensor schedule: gaussian tuning, occasional shutdown / full power.
    mask = rng.random(g.sensor.shape) < schedule_rate
    g.sensor = np.clip(g.sensor + rng.normal(0, 0.18, size=g.sensor.shape) * mask, 0, 1)
    off = rng.random(g.sensor.shape) < schedule_rate * 0.12
    g.sensor[off] = 0.0
    full = rng.random(g.sensor.shape) < schedule_rate * 0.08
    g.sensor[full] = 1.0

    # Speed genes.
    mask = rng.random(g.speed.shape) < schedule_rate
    g.speed = np.clip(
        g.speed + rng.normal(0, 0.14, size=g.speed.shape) * mask, SPEED_MIN, SPEED_MAX
    )

    # Rare activation toggles.
    toggle = rng.random(n) < rate * 0.05
    g.active[toggle] = 1 - g.active[toggle]

    # Rare full-drone re-initialisation (immigrant gene segment).
    if rng.random() < rate * 0.06:
        i = int(rng.integers(0, n))
        fresh = random_genome(env=_NoopEnv(cfg), cfg=cfg, rng=rng)
        g.deploy[i] = fresh.deploy[i]
        g.patrol[i] = fresh.patrol[i]
        g.sensor[i] = fresh.sensor[i]
        g.speed[i] = fresh.speed[i]

    if not np.any(mask) and not np.any(teleport):
        i, j = int(rng.integers(0, n)), int(rng.integers(0, k))
        g.patrol[i, j] = rng.uniform(0, cfg.grid_size - 1, size=2)
    return g


class _NoopEnv:
    """Minimal env shim for random re-initialisation inside mutation.

    The full :class:`Environment` is not needed to draw a random gene; this
    avoids threading the environment through every operator call.
    """

    def __init__(self, cfg: MissionConfig) -> None:
        self.zones = []
        self.config = cfg

    def is_free(self, x: float, y: float) -> bool:  # pragma: no cover - trivial
        return True

    def clip_point(self, point) -> np.ndarray:  # pragma: no cover - trivial
        return np.clip(np.asarray(point, dtype=float), 0.0, self.config.grid_size - 1)

    def nearest_free(self, point) -> np.ndarray:  # pragma: no cover - trivial
        return np.asarray(point, dtype=float)
