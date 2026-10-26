"""Adaptive UAV strategy engine for disaster management and autonomous surveillance.

The package implements a hybrid-encoded Genetic Algorithm (GA) that jointly
optimises UAV deployment, patrol waypoints, sensor power schedules and leg
speeds on a discretised hazard grid.

Public entry points
-------------------
``MissionConfig`` / ``GAConfig`` / ``FitnessWeights``
    Immutable configuration objects.
``Environment``
    Procedural hazard grid with priority zones, no-fly zones and targets.
``GeneticStrategyEngine``
    Runs the evolutionary optimisation and returns a :class:`GAResult`.
"""

from .config import FitnessWeights, GAConfig, MissionConfig
from .environment import Environment
from .ga import GAResult, GeneticStrategyEngine
from .genome import Genome

__all__ = [
    "FitnessWeights",
    "GAConfig",
    "MissionConfig",
    "Environment",
    "Genome",
    "GeneticStrategyEngine",
    "GAResult",
]
