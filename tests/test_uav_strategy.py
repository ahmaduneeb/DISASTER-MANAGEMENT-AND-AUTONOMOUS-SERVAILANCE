"""Unit tests for the UAV strategy engine (run with ``python -m unittest``)."""

from __future__ import annotations

import unittest

import numpy as np

from uav_strategy.config import FitnessWeights, GAConfig, MissionConfig
from uav_strategy.environment import Environment
from uav_strategy.fitness import FitnessEvaluator
from uav_strategy.ga import GeneticStrategyEngine
from uav_strategy.genome import Genome, random_genome
from uav_strategy.operators import crossover, mutate, repair
from uav_strategy.pareto import dominates, fast_non_dominated_sort, pareto_front


def make_genome(cfg: MissionConfig, active: list[int] | None = None) -> Genome:
    n, k = cfg.n_drones, cfg.n_patrol
    if active is None:
        active = [1] * n
    deploy = np.tile(np.array([[10.0, 10.0]]), (n, 1))
    patrol = np.tile(np.linspace(8, 18, k)[:, None], (1, 2))[None, :, :]
    patrol = np.repeat(patrol, n, axis=0)
    sensor = np.ones((n, k + 1))
    speed = np.ones((n, k + 2)) * 1.2
    return Genome(np.array(active), deploy, patrol, sensor, speed)


class TestEnvironment(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = MissionConfig()
        cls.env = Environment(cls.cfg)

    def test_deterministic(self) -> None:
        other = Environment(self.cfg)
        np.testing.assert_allclose(self.env.hazard, other.hazard)
        np.testing.assert_array_equal(self.env.no_fly, other.no_fly)
        np.testing.assert_array_equal(self.env.targets, other.targets)

    def test_targets_not_in_no_fly(self) -> None:
        for x, y in self.env.targets:
            self.assertTrue(self.env.is_free(x, y))

    def test_priority_and_hazard_ranges(self) -> None:
        self.assertGreaterEqual(self.env.priority.min(), 0.0)
        self.assertGreater(self.env.priority.max(), 0.0)
        self.assertGreaterEqual(self.env.hazard.min(), 0.0)
        self.assertLessEqual(self.env.hazard.max(), 1.0)

    def test_nearest_free_returns_free_cell(self) -> None:
        blocked = np.argwhere(self.env.no_fly)
        self.assertGreater(len(blocked), 0)
        y, x = blocked[0]
        fixed = self.env.nearest_free(np.array([x, y]))
        self.assertTrue(self.env.is_free(*fixed))

    def test_coverage_fusion(self) -> None:
        nodes = np.array([[25.0, 25.0]])
        low = self.env.coverage_from_sensors(nodes, np.array([0.3]))
        high = self.env.coverage_from_sensors(nodes, np.array([1.0]))
        self.assertGreaterEqual(high.sum(), low.sum())
        self.assertLessEqual(high.max(), 1.0 + 1e-9)


class TestGenome(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = MissionConfig(grid_size=30, n_drones=3, n_patrol=4)
        self.env = Environment(self.cfg)

    def test_shapes_and_path(self) -> None:
        g = random_genome(self.env, self.cfg, np.random.default_rng(1))
        self.assertEqual(g.deploy.shape, (3, 2))
        self.assertEqual(g.patrol.shape, (3, 4, 2))
        self.assertEqual(g.sensor.shape, (3, 5))
        self.assertEqual(g.speed.shape, (3, 6))
        path = g.path(0, self.cfg.base)
        self.assertEqual(path.shape, (7, 2))
        np.testing.assert_allclose(path[0], self.cfg.base)
        np.testing.assert_allclose(path[-1], self.cfg.base)

    def test_round_trip(self) -> None:
        g = random_genome(self.env, self.cfg, np.random.default_rng(2))
        restored = Genome.from_dict(g.to_dict())
        for field in ("active", "deploy", "patrol", "sensor", "speed"):
            np.testing.assert_allclose(getattr(g, field), getattr(restored, field))

    def test_active_positions(self) -> None:
        g = make_genome(self.cfg, active=[1, 0, 1])
        self.assertEqual(g.active_positions().shape[0], 2 * (self.cfg.n_patrol + 1))


class TestFitnessAndRepair(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = MissionConfig(grid_size=30, n_drones=3, n_patrol=4)
        self.env = Environment(self.cfg)
        self.weights = FitnessWeights()
        self.evaluator = FitnessEvaluator(self.env, self.cfg, self.weights)

    def test_metrics_bounds(self) -> None:
        g = random_genome(self.env, self.cfg, np.random.default_rng(3))
        g = repair(g, self.env, self.cfg)
        ev = self.evaluator.evaluate(g, keep_map=True)
        self.assertGreaterEqual(ev.coverage, 0.0)
        self.assertLessEqual(ev.coverage, 1.0)
        self.assertGreaterEqual(ev.risk, 0.0)
        self.assertLessEqual(ev.risk, 1.0)
        self.assertIsNotNone(ev.coverage_map)
        self.assertEqual(ev.coverage_map.shape, (30, 30))
        self.assertEqual(len(ev.objectives), 5)

    def test_inactive_drone_reduces_cost(self) -> None:
        g = make_genome(self.cfg, active=[1, 0, 0])
        ev = self.evaluator.evaluate(g)
        self.assertEqual(ev.n_active, 1)
        self.assertLess(ev.distance, 30 * 4)

    def test_repair_range_is_per_drone(self) -> None:
        g = make_genome(self.cfg)
        # Push one drone far away so its closed route exceeds the limit.
        g.deploy[0] = [29.0, 29.0]
        g.patrol[0] = [[29.0, 29.0]] * self.cfg.n_patrol
        repaired = repair(g, self.env, self.cfg)
        for i in range(self.cfg.n_drones):
            path = repaired.path(i, self.cfg.base)
            distance = np.linalg.norm(np.diff(path, axis=0), axis=1).sum()
            self.assertLessEqual(distance, self.cfg.max_range * 1.02)

    def test_repair_moves_waypoints_out_of_no_fly(self) -> None:
        g = make_genome(self.cfg)
        blocked = np.argwhere(self.env.no_fly)[0]
        g.deploy[0] = [blocked[1], blocked[0]]
        repaired = repair(g, self.env, self.cfg)
        self.assertTrue(self.env.is_free(*repaired.deploy[0]))

    def test_nofly_penalty(self) -> None:
        g = make_genome(self.cfg, active=[1, 0, 0])
        blocked = np.argwhere(self.env.no_fly)
        # Force a crossing through a blocked cell without repair.
        target = blocked[len(blocked) // 2]
        g.deploy[0] = [target[1], target[0]]
        g.patrol[0] = [[target[1], target[0]]] * self.cfg.n_patrol
        ev = self.evaluator.evaluate(g)
        self.assertGreater(ev.no_fly_fraction, 0.0)
        self.assertGreater(ev.penalty, 0.0)


class TestOperators(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = MissionConfig(grid_size=30, n_drones=3, n_patrol=4)
        self.env = Environment(self.cfg)
        self.rng = np.random.default_rng(5)
        self.p1 = repair(random_genome(self.env, self.cfg, self.rng), self.env, self.cfg)
        self.p2 = repair(random_genome(self.env, self.cfg, self.rng), self.env, self.cfg)

    def test_crossover_styles(self) -> None:
        for style in ("geometric", "segment", "splice"):
            child = crossover(self.p1, self.p2, self.rng, self.cfg, style=style)
            self.assertEqual(child.deploy.shape, self.p1.deploy.shape)
            self.assertEqual(child.patrol.shape, self.p1.patrol.shape)
            self.assertEqual(child.sensor.shape, self.p1.sensor.shape)
            self.assertEqual(child.speed.shape, self.p1.speed.shape)

    def test_mutation_bounds(self) -> None:
        g = mutate(self.p1, self.rng, self.cfg, rate=0.5, sigma=3.0)
        self.assertTrue(np.all(g.sensor >= 0) and np.all(g.sensor <= 1))
        self.assertTrue(np.all(g.speed >= 0.55) and np.all(g.speed <= 2.0))
        self.assertTrue(np.all(g.deploy >= 0) and np.all(g.deploy <= 29))


class TestPareto(unittest.TestCase):
    def test_dominance(self) -> None:
        self.assertTrue(dominates((1, 1), (2, 2)))
        self.assertFalse(dominates((1, 2), (2, 1)))
        self.assertFalse(dominates((1, 1), (1, 1)))

    def test_fronts(self) -> None:
        objectives = np.array([[1, 3], [2, 2], [3, 1], [3, 3], [4, 4]])
        fronts = fast_non_dominated_sort(objectives)
        self.assertEqual(set(fronts[0]), {0, 1, 2})
        self.assertIn(3, fronts[1])
        self.assertEqual(pareto_front(objectives), [0, 1, 2])


class TestGA(unittest.TestCase):
    def test_short_run_and_determinism(self) -> None:
        mission = MissionConfig(grid_size=30, n_drones=3, n_patrol=4)
        ga = GAConfig(population_size=14, generations=6, seed=3, stall_limit=20)
        r1 = GeneticStrategyEngine(mission, ga).run()
        r2 = GeneticStrategyEngine(mission, ga).run()
        self.assertAlmostEqual(r1.best_evaluation.fitness, r2.best_evaluation.fitness, places=6)
        self.assertEqual(len(r1.history), r1.generations_run)
        self.assertGreaterEqual(r1.best_evaluation.fitness, r1.history[0]["best"] - 1e-9)
        self.assertGreaterEqual(r1.best_evaluation.coverage, 0.0)
        self.assertLessEqual(r1.best_evaluation.no_fly_fraction, 1.0)
        self.assertTrue(r1.snapshots)

    def test_pareto_mode(self) -> None:
        mission = MissionConfig(grid_size=30, n_drones=3, n_patrol=4)
        ga = GAConfig(population_size=12, generations=4, mode="pareto", seed=2)
        result = GeneticStrategyEngine(mission, ga).run()
        self.assertIsNotNone(result.best_evaluation)
        self.assertGreaterEqual(len(result.pareto_objectives), 1)


if __name__ == "__main__":
    unittest.main()
