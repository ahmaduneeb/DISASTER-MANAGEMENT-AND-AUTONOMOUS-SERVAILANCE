"""Hybrid chromosome representation for the UAV deployment problem."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import MissionConfig
from .environment import Environment


@dataclass
class Genome:
    """Multi-segment hybrid chromosome.

    Segment layout (``N`` drones, ``K`` patrol waypoints)::

        active   (N,)      binary   -- drone switched on/off
        deploy   (N, 2)    real     -- initial deployment position
        patrol   (N, K, 2) real     -- ordered patrol waypoints
        sensor   (N, K+1)  real     -- sensor power at deploy + waypoints
        speed    (N, K+2)  real     -- speed gene for each traversed leg

    The path of drone ``i`` is ``base -> deploy[i] -> patrol[i, 0..K-1] -> base``.
    The first sensor gene is active while the UAV loiters over the deployment
    cell; the remaining genes are the activation schedule along the patrol.
    The path has ``K+2`` legs: base -> deployment, deployment -> first patrol
    point, ``K-1`` inter-waypoint legs and the return leg.
    """

    active: np.ndarray
    deploy: np.ndarray
    patrol: np.ndarray
    sensor: np.ndarray
    speed: np.ndarray

    def __post_init__(self) -> None:
        self.active = np.asarray(self.active, dtype=np.int8).reshape(-1)
        self.deploy = np.asarray(self.deploy, dtype=float)
        self.patrol = np.asarray(self.patrol, dtype=float)
        self.sensor = np.asarray(self.sensor, dtype=float)
        self.speed = np.asarray(self.speed, dtype=float)
        n = len(self.active)
        if self.deploy.shape != (n, 2):
            raise ValueError("deploy must have shape (n_drones, 2)")
        if self.patrol.ndim != 3 or self.patrol.shape[0] != n or self.patrol.shape[2] != 2:
            raise ValueError("patrol must have shape (n_drones, n_patrol, 2)")
        k = self.patrol.shape[1]
        if self.sensor.shape != (n, k + 1):
            raise ValueError("sensor must have shape (n_drones, n_patrol + 1)")
        if self.speed.shape != (n, k + 2):
            raise ValueError("speed must have shape (n_drones, n_patrol + 2)")

    # ------------------------------------------------------------- utilities
    @property
    def n_drones(self) -> int:
        return int(self.active.shape[0])

    @property
    def n_patrol(self) -> int:
        return int(self.patrol.shape[1])

    def copy(self) -> "Genome":
        return Genome(
            self.active.copy(),
            self.deploy.copy(),
            self.patrol.copy(),
            self.sensor.copy(),
            self.speed.copy(),
        )

    def path(self, drone: int, base: tuple[int, int]) -> np.ndarray:
        """Return the closed path of one drone as ``(K+3, 2)`` coordinates."""

        b = np.asarray(base, dtype=float)[None, :]
        return np.vstack([b, self.deploy[drone][None, :], self.patrol[drone], b])

    def flat(self) -> np.ndarray:
        """Concatenate all real-valued genes (used for diversity metrics)."""

        return np.concatenate(
            [
                self.active.astype(float),
                self.deploy.ravel(),
                self.patrol.ravel(),
                self.sensor.ravel(),
                self.speed.ravel(),
            ]
        )

    def active_positions(self) -> np.ndarray:
        mask = self.active.astype(bool)
        if not mask.any():
            return np.empty((0, 2), dtype=float)
        nodes = np.concatenate(
            [self.deploy[mask, None, :], self.patrol[mask]], axis=1
        )
        return nodes.reshape(-1, 2)

    def to_dict(self) -> dict:
        return {
            "active": self.active.tolist(),
            "deploy": self.deploy.tolist(),
            "patrol": self.patrol.tolist(),
            "sensor": self.sensor.tolist(),
            "speed": self.speed.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Genome":
        return cls(
            np.asarray(data["active"]),
            np.asarray(data["deploy"]),
            np.asarray(data["patrol"]),
            np.asarray(data["sensor"]),
            np.asarray(data["speed"]),
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"Genome(n_drones={self.n_drones}, n_patrol={self.n_patrol}, active={int(self.active.sum())})"


# gene limits shared with the genetic operators
SENSOR_MIN, SENSOR_MAX = 0.0, 1.0
SPEED_MIN, SPEED_MAX = 0.55, 2.0


def random_genome(
    env: Environment,
    cfg: MissionConfig,
    rng: np.random.Generator,
    heuristic: bool = False,
) -> Genome:
    """Create a random (optionally hotspot-seeded) chromosome."""

    n, k = cfg.n_drones, cfg.n_patrol
    active = np.ones(n, dtype=np.int8)
    deploy = np.zeros((n, 2), dtype=float)
    patrol = np.zeros((n, k, 2), dtype=float)
    sensor = rng.uniform(0.25, 1.0, size=(n, k + 1))
    speed = rng.uniform(0.8, 1.6, size=(n, k + 2))

    zones = env.zones
    if heuristic and zones:
        for i in range(n):
            zone = zones[i % len(zones)]
            theta = rng.uniform(0, 2 * np.pi)
            radius = rng.uniform(1.0, max(1.5, zone.radius * 0.55))
            deploy[i] = env.clip_point(
                [zone.cx + radius * np.cos(theta), zone.cy + radius * np.sin(theta)]
            )
            # A connected orbit around the hotspot improves the initial mean.
            orbit = rng.uniform(2.0, min(6.0, zone.radius))
            phase = rng.uniform(0, 2 * np.pi)
            for j in range(k):
                ang = phase + 2 * np.pi * j / k
                patrol[i, j] = env.clip_point(
                    [
                        zone.cx + orbit * np.cos(ang) + rng.normal(0, 0.7),
                        zone.cy + orbit * np.sin(ang) + rng.normal(0, 0.7),
                    ]
                )
            sensor[i] = rng.uniform(0.55, 1.0, size=k + 1)
            speed[i] = rng.uniform(0.8, 1.4, size=k + 2)
    else:
        for i in range(n):
            free = False
            for _ in range(100):
                p = rng.uniform(0, cfg.grid_size - 1, size=2)
                if env.is_free(*p):
                    deploy[i] = p
                    free = True
                    break
            if not free:
                deploy[i] = env.nearest_free(rng.uniform(0, cfg.grid_size - 1, size=2))
            heading = rng.uniform(0, 2 * np.pi)
            current = deploy[i].copy()
            for j in range(k):
                heading += rng.normal(0, 0.55)
                step = rng.uniform(3.0, 10.0)
                current = env.clip_point(current + step * np.array([np.cos(heading), np.sin(heading)]))
                patrol[i, j] = current

    return Genome(active, deploy, patrol, sensor, speed)
