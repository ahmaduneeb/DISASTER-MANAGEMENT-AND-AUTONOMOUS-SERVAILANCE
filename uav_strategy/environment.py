"""Procedural discrete environment used by the UAV strategy engine.

The map is a 2-D grid.  Each layer has a concrete meaning:

``priority``
    Weighted importance of every cell (disaster hotspots).
``hazard``
    Normalised radiation / toxic-plume intensity in ``[0, 1]``.
``no_fly``
    Boolean hard obstacle layer (restricted airspace and buildings).
``targets``
    Discrete high-value points used to estimate the detection rate.
``wind``
    A uniform wind vector that modulates propulsion energy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import MissionConfig


@dataclass(frozen=True)
class PriorityZone:
    """A circular disaster hotspot on the grid."""

    cx: float
    cy: float
    radius: float
    priority: int


class Environment:
    """Mission map generated deterministically from a seed."""

    def __init__(self, config: MissionConfig) -> None:
        self.config = config
        self.size = config.grid_size
        self.rng = np.random.default_rng(config.seed)
        self.priority = np.zeros((self.size, self.size), dtype=float)
        self.hazard = np.zeros((self.size, self.size), dtype=float)
        self.no_fly = np.zeros((self.size, self.size), dtype=bool)
        self.zones: list[PriorityZone] = []
        self.hazard_pockets: list[dict[str, float]] = []
        self.no_fly_shapes: list[dict[str, float | str]] = []

        self._build_priority()
        self._build_nofly()
        self._build_hazard()
        self._build_targets()
        self._build_wind()

    # ------------------------------------------------------------------ build
    def _build_priority(self) -> None:
        """Scatter disaster zones, keeping them away from the launch base."""

        cfg = self.config
        bx, by = cfg.base
        attempts = 0
        while len(self.zones) < cfg.n_priority_zones and attempts < 10_000:
            attempts += 1
            cx = float(self.rng.uniform(10, self.size - 8))
            cy = float(self.rng.uniform(8, self.size - 8))
            if np.hypot(cx - bx, cy - by) < 14:
                continue
            if any(np.hypot(cx - z.cx, cy - z.cy) < z.radius + 7 for z in self.zones):
                continue
            radius = float(self.rng.uniform(5.0, 8.5))
            priority = int(self.rng.integers(1, 4))
            self.zones.append(PriorityZone(cx, cy, radius, priority))

        yy, xx = np.mgrid[0 : self.size, 0 : self.size]
        for zone in self.zones:
            d2 = (xx - zone.cx) ** 2 + (yy - zone.cy) ** 2
            # Smooth wedge: highest importance at the epicentre.
            weight = zone.priority * np.exp(-d2 / (2.0 * (zone.radius * 0.62) ** 2))
            self.priority += weight
        mask = 0.0

        # Keep a crisp footprint for plotting / constraints.
        for zone in self.zones:
            d2 = (xx - zone.cx) ** 2 + (yy - zone.cy) ** 2
            mask = np.maximum(mask, (d2 <= zone.radius**2).astype(float) * zone.priority)
        self.priority_mask = mask
        self.priority = np.where(mask > 0, self.priority, 0.0)

    def _build_nofly(self) -> None:
        """Add no-fly circles and rectangular restricted blocks."""

        cfg = self.config
        bx, by = cfg.base
        yy, xx = np.mgrid[0 : self.size, 0 : self.size]
        attempts = 0
        while len(self.no_fly_shapes) < cfg.n_no_fly and attempts < 5_000:
            attempts += 1
            shape = "circle" if self.rng.random() < 0.55 else "rect"
            cx = float(self.rng.uniform(8, self.size - 8))
            cy = float(self.rng.uniform(8, self.size - 8))
            if np.hypot(cx - bx, cy - by) < 10:
                continue
            if shape == "circle":
                r = float(self.rng.uniform(2.0, 4.5))
                self.no_fly_shapes.append({"kind": "circle", "cx": cx, "cy": cy, "r": r})
                self.no_fly |= (xx - cx) ** 2 + (yy - cy) ** 2 <= r**2
            else:
                w = float(self.rng.uniform(4, 9))
                h = float(self.rng.uniform(4, 9))
                self.no_fly_shapes.append(
                    {"kind": "rect", "x": cx - w / 2, "y": cy - h / 2, "w": w, "h": h}
                )
                self.no_fly |= (xx >= cx - w / 2) & (xx <= cx + w / 2) & (
                    yy >= cy - h / 2
                ) & (yy <= cy + h / 2)

        # Guarantee a safe launch pad.
        self.no_fly &= (xx - bx) ** 2 + (yy - by) ** 2 > 4.0**2

    def _build_hazard(self) -> None:
        """Sum Gaussian plumes into a smooth radiation / toxic field."""

        cfg = self.config
        yy, xx = np.mgrid[0 : self.size, 0 : self.size]
        for _ in range(cfg.n_hazards):
            cx = float(self.rng.uniform(5, self.size - 5))
            cy = float(self.rng.uniform(5, self.size - 5))
            sigma = float(self.rng.uniform(2.5, 6.0))
            amp = float(self.rng.uniform(0.45, 1.0))
            self.hazard_pockets.append(
                {"cx": cx, "cy": cy, "sigma": sigma, "amp": amp}
            )
            self.hazard += amp * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2))
        # Light terrain-wide background radiation.
        self.hazard += 0.04
        self.hazard = np.clip(self.hazard / max(self.hazard.max(), 1e-9), 0.0, 1.0)

    def _build_targets(self) -> None:
        """Sample discrete targets from the priority distribution."""

        cfg = self.config
        weights = self.priority.ravel().copy()
        weights[self.no_fly.ravel()] = 0.0
        if weights.sum() <= 0:
            weights = np.ones_like(weights)
        probs = weights / weights.sum()
        chosen = self.rng.choice(
            self.size * self.size,
            size=min(cfg.n_targets, int(np.count_nonzero(probs))),
            replace=False,
            p=probs,
        )
        ys, xs = np.divmod(chosen, self.size)
        self.targets = np.column_stack([xs, ys]).astype(float)
        self.target_weights = np.array(
            [self.priority[int(y), int(x)] for x, y in self.targets], dtype=float
        )
        self.target_weights /= max(self.target_weights.sum(), 1e-9)

    def _build_wind(self) -> None:
        angle = float(self.rng.uniform(0, 2 * np.pi))
        self.wind_direction = np.array([np.cos(angle), np.sin(angle)])
        self.wind_magnitude = float(self.rng.uniform(0.25, 0.75))

    # ------------------------------------------------------------- geometry
    @property
    def bounds(self) -> tuple[float, float]:
        return (0.0, float(self.size - 1))

    def is_free(self, x: float, y: float) -> bool:
        xi = int(np.clip(round(x), 0, self.size - 1))
        yi = int(np.clip(round(y), 0, self.size - 1))
        return not bool(self.no_fly[yi, xi])

    def nearest_free(self, point: np.ndarray | tuple[float, float]) -> np.ndarray:
        """Project a point to the closest non-blocked cell (repair operator)."""

        x, y = float(point[0]), float(point[1])
        xi = int(np.clip(round(x), 0, self.size - 1))
        yi = int(np.clip(round(y), 0, self.size - 1))
        if not self.no_fly[yi, xi]:
            return np.array([xi, yi], dtype=float)
        best: tuple[float, int, int] | None = None
        for radius in range(1, self.size):
            for dx in range(-radius, radius + 1):
                for dy in (-radius, radius):
                    nx, ny = xi + dx, yi + dy
                    if 0 <= nx < self.size and 0 <= ny < self.size and not self.no_fly[ny, nx]:
                        d = dx * dx + dy * dy
                        if best is None or d < best[0]:
                            best = (d, nx, ny)
            for dy in range(-radius + 1, radius):
                for dx in (-radius, radius):
                    nx, ny = xi + dx, yi + dy
                    if 0 <= nx < self.size and 0 <= ny < self.size and not self.no_fly[ny, nx]:
                        d = dx * dx + dy * dy
                        if best is None or d < best[0]:
                            best = (d, nx, ny)
            if best is not None:
                return np.array([best[1], best[2]], dtype=float)
        return np.array(self.config.base, dtype=float)

    def clip_point(self, point: np.ndarray | tuple[float, float]) -> np.ndarray:
        lo, hi = self.bounds
        return np.clip(np.asarray(point, dtype=float), lo, hi)

    def sample_polyline(
        self, path: np.ndarray, step: float = 0.45
    ) -> tuple[np.ndarray, np.ndarray]:
        """Densely sample a polyline.

        Returns
        -------
        points : (M, 2) float array
        segment_ids : (M,) int array with the index of the traversed leg.
        """

        chunks: list[np.ndarray] = []
        ids: list[np.ndarray] = []
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            length = float(np.hypot(*(b - a)))
            n = max(int(np.ceil(length / step)), 1)
            t = np.linspace(0.0, 1.0, n, endpoint=False)
            chunks.append(a[None, :] + t[:, None] * (b - a)[None, :])
            ids.append(np.full(n, i, dtype=int))
        chunks.append(path[-1][None, :])
        ids.append(np.array([len(path) - 2], dtype=int))
        return np.vstack(chunks), np.concatenate(ids)

    def hazard_at(self, points: np.ndarray) -> np.ndarray:
        """Bilinear-free nearest-cell hazard lookup."""

        xs = np.clip(np.round(points[:, 0]).astype(int), 0, self.size - 1)
        ys = np.clip(np.round(points[:, 1]).astype(int), 0, self.size - 1)
        return self.hazard[ys, xs]

    def blocked_at(self, points: np.ndarray) -> np.ndarray:
        xs = np.clip(np.round(points[:, 0]).astype(int), 0, self.size - 1)
        ys = np.clip(np.round(points[:, 1]).astype(int), 0, self.size - 1)
        return self.no_fly[ys, xs]

    def coverage_from_sensors(
        self, nodes: np.ndarray, powers: np.ndarray, radius: float | None = None
    ) -> np.ndarray:
        """Fuse sensor footprints into a detection-probability grid.

        Each sensor contributes ``p * exp(-2.5 (d / r_eff)^2)`` where ``p`` is
        its power gene and ``r_eff`` scales with power.  Independent sensors
        are fused with the noisy-OR rule.
        """

        if radius is None:
            radius = self.config.sensor_radius
        covered = np.zeros((self.size, self.size), dtype=float)
        for (x, y), power in zip(nodes, powers):
            if power <= 0.02:
                continue
            r_eff = radius * (0.45 + 0.55 * float(power))
            reach = int(np.ceil(2.0 * r_eff)) + 1
            xi = int(np.clip(round(x), 0, self.size - 1))
            yi = int(np.clip(round(y), 0, self.size - 1))
            x0, x1 = max(0, xi - reach), min(self.size, xi + reach + 1)
            y0, y1 = max(0, yi - reach), min(self.size, yi + reach + 1)
            yy, xx = np.mgrid[y0:y1, x0:x1]
            d2 = (xx - x) ** 2 + (yy - y) ** 2
            prob = float(power) * np.exp(-2.5 * d2 / (r_eff**2))
            local = 1.0 - (1.0 - covered[y0:y1, x0:x1]) * (1.0 - prob)
            covered[y0:y1, x0:x1] = local
        return covered

    def coverage_score(self, covered: np.ndarray) -> float:
        denom = float(self.priority.sum())
        if denom <= 0:
            return 0.0
        return float((covered * self.priority).sum() / denom)

    def detection_score(self, covered: np.ndarray) -> float:
        if len(self.targets) == 0:
            return 0.0
        xs = np.clip(self.targets[:, 0].astype(int), 0, self.size - 1)
        ys = np.clip(self.targets[:, 1].astype(int), 0, self.size - 1)
        return float((covered[ys, xs] * self.target_weights).sum())

    def detection_rate(self, covered: np.ndarray, threshold: float = 0.5) -> float:
        """Fraction of priority-weighted targets confidently detected."""

        if len(self.targets) == 0:
            return 0.0
        xs = np.clip(self.targets[:, 0].astype(int), 0, self.size - 1)
        ys = np.clip(self.targets[:, 1].astype(int), 0, self.size - 1)
        hits = covered[ys, xs] >= threshold
        return float(self.target_weights[hits].sum())

    def wind_factor(self, direction: np.ndarray) -> float:
        """Headwind/crosswind multiplier applied to propulsion energy."""

        norm = float(np.linalg.norm(direction))
        if norm < 1e-9:
            return 1.0
        unit = direction / norm
        head = float(np.dot(unit, self.wind_direction))
        return float(1.0 + 0.12 * self.wind_magnitude * max(head, -0.5))
