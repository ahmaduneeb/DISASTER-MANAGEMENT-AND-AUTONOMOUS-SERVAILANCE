"""Tests for the browser dashboard's configuration and JSON endpoints."""

from __future__ import annotations

import unittest

from uav_strategy.webapp import build_configs, create_app


class TestBuildConfigs(unittest.TestCase):
    def test_defaults_are_valid(self) -> None:
        mission, ga, weights = build_configs({})
        self.assertEqual(mission.grid_size, 50)
        self.assertEqual(ga.population_size, 40)
        self.assertAlmostEqual(
            weights.coverage
            + weights.detection
            + weights.energy
            + weights.time
            + weights.safety,
            1.0,
            places=9,
        )

    def test_weights_are_normalised(self) -> None:
        payload = {
            "w_coverage": 2,
            "w_detection": 1,
            "w_energy": 1,
            "w_time": 0,
            "w_safety": 0,
        }
        _mission, _ga, weights = build_configs(payload)
        data = weights.to_dict()
        objective = ["coverage", "detection", "energy", "time", "safety"]
        self.assertAlmostEqual(data["coverage"], 0.5, places=6)
        self.assertAlmostEqual(data["detection"], 0.25, places=6)
        self.assertAlmostEqual(sum(data[key] for key in objective), 1.0, places=9)

    def test_zero_weights_fall_back_to_balanced(self) -> None:
        _mission, _ga, weights = build_configs(
            {key: 0 for key in ("w_coverage", "w_detection", "w_energy", "w_time", "w_safety")}
        )
        self.assertAlmostEqual(weights.coverage, 0.45, places=6)

    def test_budgets_scale_with_grid(self) -> None:
        mission, _ga, _weights = build_configs({"grid_size": 60})
        self.assertAlmostEqual(mission.sensor_radius, 6.0, places=6)
        self.assertAlmostEqual(mission.max_range, 210.0, places=6)
        self.assertAlmostEqual(mission.energy_budget, 43.2, places=6)

    def test_out_of_range_values_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_configs({"grid_size": 5})
        with self.assertRaises(ValueError):
            build_configs({"n_drones": 12})
        with self.assertRaises(ValueError):
            build_configs({"mode": "rnd"})


class TestDashboardEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = create_app().test_client()

    def test_index_renders(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Mission control", response.data)
        self.assertIn(b"APP_DEFAULTS", response.data)

    def test_health(self) -> None:
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "ok")

    def test_preview_returns_url(self) -> None:
        response = self.client.post(
            "/api/preview",
            json={"grid_size": 30, "seed": 2, "n_drones": 2, "n_patrol": 4},
        )
        self.assertEqual(response.status_code, 200)
        url = response.get_json()["url"]
        self.assertTrue(url.startswith("/previews/mission_"))
        image = self.client.get(url)
        self.assertEqual(image.status_code, 200)
        self.assertEqual(image.content_type, "image/png")
        image.close()

    def test_bad_payload_is_rejected(self) -> None:
        response = self.client.post("/api/run", json={"population": 1})
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.get_json())


if __name__ == "__main__":
    unittest.main()
