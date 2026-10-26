"""Configuration objects for the mission, the GA and the fitness function."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class MissionConfig:
    """Static description of the disaster-surveillance mission.

    All spatial quantities are expressed in grid-cell units.  A cell is a
    square patch of the real world; the default 50 x 50 grid keeps the
    simulation fast while still producing non-trivial combinatorial choices.
    """

    grid_size: int = 50
    base: tuple[int, int] = (2, 2)
    n_drones: int = 5
    n_patrol: int = 8
    n_priority_zones: int = 4
    n_hazards: int = 5
    n_no_fly: int = 4
    n_targets: int = 24
    sensor_radius: float = 5.0
    max_range: float = 175.0
    max_time: float = 240.0
    energy_budget: float = 36.0
    min_coverage: float = 0.55
    seed: int = 42

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class FitnessWeights:
    """Weights of the scalarised multi-objective fitness function.

    The five weights sum to one.  Maximisation of coverage and detection is
    rewarded directly; energy, time and hazard exposure enter through their
    complements ``(1 - normalised_cost)``.
    """

    coverage: float = 0.45
    detection: float = 0.15
    energy: float = 0.10
    time: float = 0.10
    safety: float = 0.20

    # Constraint handling
    penalty_nofly: float = 4.0
    penalty_coverage: float = 3.0
    penalty_range: float = 2.0
    penalty_time: float = 2.0

    def __post_init__(self) -> None:
        total = self.coverage + self.detection + self.energy + self.time + self.safety
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"fitness weights must sum to 1.0 (got {total:.3f})")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class GAConfig:
    """Hyper-parameters of the evolutionary strategy engine."""

    population_size: int = 60
    generations: int = 80
    crossover_rate: float = 0.90
    mutation_rate: float = 0.20
    tournament_size: int = 4
    elitism: int = 2
    immigrant_fraction: float = 0.04
    mutation_sigma: float = 1.6
    stall_limit: int = 35
    mode: str = "weighted"  # "weighted" or "pareto" (NSGA-II)
    seed: int = 7

    def __post_init__(self) -> None:
        if self.mode not in {"weighted", "pareto"}:
            raise ValueError("mode must be 'weighted' or 'pareto'")
        if self.population_size < 4:
            raise ValueError("population_size must be at least 4")
        if self.elitism >= self.population_size:
            raise ValueError("elitism must be smaller than the population")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ExperimentConfig:
    """Settings for the reproducible sensitivity / baseline study."""

    seeds: tuple[int, ...] = (11, 23)
    population_sizes: tuple[int, ...] = (24, 40, 60)
    mutation_rates: tuple[float, ...] = (0.05, 0.15, 0.30)
    tournament_sizes: tuple[int, ...] = (2, 4, 8)
    generations: int = 40
    output_dir: str = "outputs/experiments"
    mission: MissionConfig = field(default_factory=MissionConfig)
    ga: GAConfig = field(default_factory=GAConfig)
    weights: FitnessWeights = field(default_factory=FitnessWeights)

    def to_dict(self) -> dict:
        return {
            "seeds": list(self.seeds),
            "population_sizes": list(self.population_sizes),
            "mutation_rates": list(self.mutation_rates),
            "tournament_sizes": list(self.tournament_sizes),
            "generations": self.generations,
            "output_dir": self.output_dir,
            "mission": self.mission.to_dict(),
            "ga": self.ga.to_dict(),
            "weights": self.weights.to_dict(),
        }
