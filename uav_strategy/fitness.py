"""Multi-objective fitness evaluation with dynamic constraint handling."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .config import FitnessWeights, MissionConfig
from .environment import Environment
from .genome import Genome


@dataclass
class Evaluation:
    """All raw and normalised scores produced for one chromosome."""

    fitness: float
    coverage: float
    detection: float
    detection_rate: float
    energy: float
    energy_frac: float
    time: float
    time_frac: float
    risk: float
    distance: float
    range_frac: float
    no_fly_fraction: float
    penalty: float
    n_active: int
    objectives: tuple[float, ...]
    coverage_map: np.ndarray | None = field(default=None, repr=False, compare=False)

    def scalar_summary(self) -> dict:
        """Serialisable summary without the large coverage matrix."""

        data = asdict(self)
        data.pop("coverage_map", None)
        data["objectives"] = list(self.objectives)
        return data


class FitnessEvaluator:
    """Evaluates chromosomes against the five competing objectives.

    Objectives
    ----------
    * maximise weighted disaster-zone coverage,
    * maximise detection probability over priority-weighted targets,
    * minimise normalised energy expenditure,
    * minimise normalised mission time,
    * minimise integrated hazard exposure.

    Hard constraints (flight range, no-fly airspace, time budget, minimum
    coverage) are enforced with repair operators plus a constraint-aware
    penalty function.
    """

    def __init__(
        self,
        env: Environment,
        cfg: MissionConfig,
        weights: FitnessWeights | None = None,
    ) -> None:
        self.env = env
        self.cfg = cfg
        self.weights = weights or FitnessWeights()

    def evaluate(
        self, genome: Genome, progress: float = 0.0, keep_map: bool = False
    ) -> Evaluation:
        cfg = self.cfg
        base = np.asarray(cfg.base, dtype=float)
        n = genome.n_drones
        active = genome.active.astype(bool)

        all_nodes: list[np.ndarray] = []
        all_powers: list[np.ndarray] = []
        total_distance = 0.0
        max_drone_distance = 0.0
        total_energy = 0.0
        total_time = 0.0
        hazard_sum = 0.0
        samples = 0
        blocked = 0

        for i in range(n):
            if not active[i]:
                continue
            path = genome.path(i, cfg.base)
            nodes = path[1:-1]  # deployment + patrol; base is not a sensor node
            powers = genome.sensor[i]
            all_nodes.append(nodes)
            all_powers.append(powers)

            dists = np.linalg.norm(np.diff(path, axis=0), axis=1)
            directions = np.diff(path, axis=0)
            speeds = np.clip(genome.speed[i], 0.2, 3.0)
            wind = np.array([self.env.wind_factor(d) for d in directions])
            energy = np.sum(dists * (0.06 + 0.05 * speeds**2) * wind)
            energy += 0.12 * float(np.sum(powers)) + 0.01 * len(nodes)
            time = float(np.sum(dists / speeds)) + 0.15 * float(np.sum(powers > 0.1))
            total_distance += float(dists.sum())
            max_drone_distance = max(max_drone_distance, float(dists.sum()))
            total_energy += float(energy)
            total_time += time

            pts, _ = self.env.sample_polyline(path)
            hazard_sum += float(self.env.hazard_at(pts).sum())
            samples += len(pts)
            blocked += int(self.env.blocked_at(pts).sum())

        if all_nodes:
            nodes = np.vstack(all_nodes)
            powers = np.concatenate(all_powers)
            coverage_map = self.env.coverage_from_sensors(
                nodes, powers, cfg.sensor_radius
            )
        else:
            coverage_map = np.zeros((self.env.size, self.env.size), dtype=float)

        coverage = self.env.coverage_score(coverage_map)
        detection = self.env.detection_score(coverage_map)
        detection_rate = self.env.detection_rate(coverage_map)
        risk = hazard_sum / max(samples, 1)
        no_fly_fraction = blocked / max(samples, 1)

        energy_frac = total_energy / max(cfg.energy_budget, 1e-9)
        time_frac = total_time / max(cfg.max_time, 1e-9)
        range_frac = max_drone_distance / max(cfg.max_range, 1e-9)

        # Constraint-aware penalty.  The violation magnitudes make the pressure
        # self-adjusting: gross violations are punished heavily while a nearly
        # feasible route incurs only a small gradient toward feasibility.
        scale = 1.0
        deficit = max(0.0, cfg.min_coverage - coverage)
        range_excess = max(0.0, range_frac - 1.0)
        time_excess = max(0.0, time_frac - 1.0)
        penalty = scale * (
            self.weights.penalty_nofly * no_fly_fraction
            + self.weights.penalty_coverage * deficit
            + self.weights.penalty_range * range_excess
            + self.weights.penalty_time * time_excess
        )

        w = self.weights
        fitness = (
            w.coverage * coverage
            + w.detection * detection
            + w.energy * (1.0 - energy_frac)
            + w.time * (1.0 - time_frac)
            + w.safety * (1.0 - risk)
            - penalty
        )

        # Minimisation form used by the NSGA-II archive.  The penalty is added
        # to every objective so that feasible solutions strictly dominate
        # infeasible ones.
        objectives = (
            -coverage + penalty,
            -detection + penalty,
            energy_frac + penalty,
            time_frac + penalty,
            risk + penalty,
        )

        return Evaluation(
            fitness=float(fitness),
            coverage=float(coverage),
            detection=float(detection),
            detection_rate=float(detection_rate),
            energy=float(total_energy),
            energy_frac=float(energy_frac),
            time=float(total_time),
            time_frac=float(time_frac),
            risk=float(risk),
            distance=float(total_distance),
            range_frac=float(range_frac),
            no_fly_fraction=float(no_fly_fraction),
            penalty=float(penalty),
            n_active=int(active.sum()),
            objectives=tuple(float(x) for x in objectives),
            coverage_map=coverage_map if keep_map else None,
        )
