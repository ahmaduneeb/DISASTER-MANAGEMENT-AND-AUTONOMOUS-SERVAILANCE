"""Local web dashboard for the UAV strategy engine.

Run with ``python app.py`` or ``python -m uav_strategy ui`` and open
http://127.0.0.1:5000.  Simulations execute in a background thread, so the
browser can poll live GA progress and display the generated figures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import threading
import time
import traceback
import uuid
import webbrowser
from pathlib import Path
from typing import Any

from flask import Flask, abort, jsonify, render_template, request, send_from_directory

from .config import FitnessWeights, GAConfig, MissionConfig
from .environment import Environment
from .simulation import run_simulation
from .visualize import plot_mission

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "outputs" / "ui_runs"
PREVIEW_DIR = ROOT / "outputs" / "ui_previews"
RUNS_DIR.mkdir(parents=True, exist_ok=True)
PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

WEIGHT_PRESETS = {
    "balanced": {"coverage": 0.45, "detection": 0.15, "energy": 0.10, "time": 0.10, "safety": 0.20},
    "coverage": {"coverage": 0.55, "detection": 0.20, "energy": 0.08, "time": 0.07, "safety": 0.10},
    "safety": {"coverage": 0.35, "detection": 0.15, "energy": 0.08, "time": 0.07, "safety": 0.35},
    "efficiency": {"coverage": 0.40, "detection": 0.15, "energy": 0.20, "time": 0.10, "safety": 0.15},
}


def _clamp(value: Any, low: float, high: float, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not (low <= number <= high):
        raise ValueError(f"{name} must be between {low:g} and {high:g}")
    return number


def _int(value: Any, low: int, high: int, name: str) -> int:
    number = int(round(_clamp(value, low, high, name)))
    return number


def build_configs(payload: dict) -> tuple[MissionConfig, GAConfig, FitnessWeights]:
    """Validate browser input and scale mission budgets to the grid size."""

    grid = _int(payload.get("grid_size", 50), 30, 70, "Grid size")
    drones = _int(payload.get("n_drones", 5), 2, 8, "UAV count")
    patrol = _int(payload.get("n_patrol", 8), 4, 12, "Patrol waypoints")
    population = _int(payload.get("population", 40), 8, 120, "Population size")
    generations = _int(payload.get("generations", 40), 3, 150, "Generations")
    tournament = _int(payload.get("tournament_size", 4), 2, 12, "Tournament size")
    mutation = _clamp(payload.get("mutation_rate", 0.20), 0.01, 0.60, "Mutation rate")
    seed = _int(payload.get("seed", 7), 0, 999_999, "Seed")
    mode = str(payload.get("mode", "weighted"))
    if mode not in {"weighted", "pareto"}:
        raise ValueError("Mode must be weighted or pareto")

    raw_weights = {
        key: max(0.0, float(payload.get(f"w_{key}", default)))
        for key, default in WEIGHT_PRESETS["balanced"].items()
    }
    total = sum(raw_weights.values())
    if total <= 1e-9:
        raw_weights = dict(WEIGHT_PRESETS["balanced"])
        total = 1.0
    weights = {key: value / total for key, value in raw_weights.items()}
    # Force the floating-point residual into the last weight so the dataclass
    # sum-to-one validation cannot fail on rounding noise.
    weights["safety"] += 1.0 - sum(weights.values())

    mission = MissionConfig(
        grid_size=grid,
        n_drones=drones,
        n_patrol=patrol,
        n_priority_zones=4,
        n_hazards=5,
        n_no_fly=4,
        n_targets=24,
        sensor_radius=round(0.10 * grid, 2),
        max_range=round(3.5 * grid, 1),
        max_time=round(4.8 * grid, 1),
        energy_budget=round(0.72 * grid, 1),
        min_coverage=0.55,
        seed=seed,
    )
    ga = GAConfig(
        population_size=population,
        generations=generations,
        mutation_rate=mutation,
        tournament_size=tournament,
        mode=mode,
        seed=seed + 101,
        stall_limit=max(20, generations // 2),
    )
    return mission, ga, FitnessWeights(**weights)


class JobStore:
    """Thread-safe in-memory status store for background simulations."""

    def __init__(self) -> None:
        self._jobs: dict[str, dict] = {}
        self._lock = threading.Lock()

    def create(self) -> tuple[str, dict]:
        job_id = uuid.uuid4().hex[:12]
        job = {
            "id": job_id,
            "status": "queued",
            "progress": 0.0,
            "generation": 0,
            "generations": 0,
            "created": time.time(),
            "started": None,
            "finished": None,
            "history": [],
            "best": None,
            "average": None,
            "coverage": None,
            "detection": None,
            "energy_frac": None,
            "risk": None,
            "result": None,
            "error": None,
        }
        with self._lock:
            # Keep the store bounded during long browser sessions.
            if len(self._jobs) > 12:
                oldest = sorted(self._jobs.values(), key=lambda j: j["created"])[:2]
                for old in oldest:
                    self._jobs.pop(old["id"], None)
            self._jobs[job_id] = job
        return job_id, job

    def update(self, job_id: str, **fields: Any) -> None:
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].update(fields)

    def append_history(self, job_id: str, record: dict) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            if not job["history"] or job["history"][-1].get("generation") != record.get("generation"):
                job["history"].append(
                    {
                        "generation": record.get("generation", 0),
                        "best": record.get("best", 0.0),
                        "average": record.get("average", 0.0),
                        "coverage": record.get("coverage", 0.0),
                    }
                )
            job.update(
                {
                    "progress": float(record.get("progress", 0.0)),
                    "generation": int(record.get("generation", 0)),
                    "generations": int(record.get("generations", 0)),
                    "best": record.get("best"),
                    "average": record.get("average"),
                    "coverage": record.get("coverage"),
                    "detection": record.get("detection"),
                    "energy_frac": record.get("energy_frac"),
                    "risk": record.get("risk"),
                }
            )

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None


    def count(self) -> int:
        with self._lock:
            return len(self._jobs)


JOBS = JobStore()


def _run_job(
    job_id: str,
    mission: MissionConfig,
    ga: GAConfig,
    weights: FitnessWeights,
    output_dir: Path,
) -> None:
    started = time.time()
    JOBS.update(job_id, status="running", started=started)

    def on_progress(record: dict) -> None:
        JOBS.append_history(job_id, record)
        JOBS.update(job_id, status="running")

    try:
        result, figures = run_simulation(
            mission=mission,
            ga=ga,
            weights=weights,
            output_dir=output_dir,
            make_plots=True,
            verbose=False,
            progress_callback=on_progress,
        )
        metrics = result.best_evaluation.scalar_summary()
        figure_urls = {
            key: f"/runs/{job_id}/{Path(value).relative_to(output_dir).as_posix()}"
            for key, value in figures.items()
        }
        if "coverage_evolution" in figure_urls:
            figure_urls["evolution"] = figure_urls["coverage_evolution"]
        downloads = {
            "best_genome.json": f"/runs/{job_id}/best_genome.json",
            "run_summary.json": f"/runs/{job_id}/run_summary.json",
            "history.json": f"/runs/{job_id}/history.json",
            "pareto_archive.json": f"/runs/{job_id}/pareto_archive.json",
        }
        JOBS.update(
            job_id,
            status="complete",
            progress=1.0,
            finished=time.time(),
            result={
                "metrics": metrics,
                "figures": figure_urls,
                "downloads": downloads,
                "generations_run": result.generations_run,
                "termination": result.termination_reason,
                "evaluations": result.evaluations,
                "elapsed": result.elapsed_seconds,
                "pareto_size": len(result.pareto_genomes),
                "n_active": metrics.get("n_active", 0),
                "config": {
                    "mission": mission.to_dict(),
                    "ga": ga.to_dict(),
                    "weights": weights.to_dict(),
                },
            },
        )
    except Exception as exc:  # pragma: no cover - exercised by API errors
        traceback.print_exc()
        JOBS.update(
            job_id,
            status="error",
            finished=time.time(),
            error=f"{type(exc).__name__}: {exc}",
        )


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["JSON_SORT_KEYS"] = False

    @app.get("/")
    def index():
        defaults = {
            "grid_size": 50,
            "n_drones": 5,
            "n_patrol": 8,
            "population": 40,
            "generations": 40,
            "mutation_rate": 0.20,
            "tournament_size": 4,
            "seed": 7,
            "mode": "weighted",
            "weights": WEIGHT_PRESETS["balanced"],
        }
        return render_template(
            "index.html",
            defaults=defaults,
            presets=WEIGHT_PRESETS,
        )

    @app.post("/api/run")
    def api_run():
        try:
            payload = request.get_json(force=True) or {}
            mission, ga, weights = build_configs(payload)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        job_id, _ = JOBS.create()
        output_dir = RUNS_DIR / job_id
        output_dir.mkdir(parents=True, exist_ok=True)
        thread = threading.Thread(
            target=_run_job,
            args=(job_id, mission, ga, weights, output_dir),
            daemon=True,
            name=f"uav-job-{job_id}",
        )
        thread.start()
        return jsonify({"job_id": job_id, "status": "queued"}), 202

    @app.get("/api/jobs/<job_id>")
    def api_job(job_id: str):
        job = JOBS.get(job_id)
        if job is None:
            return jsonify({"error": "job not found"}), 404
        return jsonify(job)

    @app.post("/api/preview")
    def api_preview():
        try:
            payload = request.get_json(force=True) or {}
            mission, _ga, _weights = build_configs(payload)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        key = hashlib.sha1(
            json.dumps(
                {
                    "grid": mission.grid_size,
                    "seed": mission.seed,
                    "drones": mission.n_drones,
                    "patrol": mission.n_patrol,
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:12]
        preview = PREVIEW_DIR / f"mission_{key}.png"
        if not preview.exists():
            env = Environment(mission)
            plot_mission(env, preview)
        return jsonify({"url": f"/previews/{preview.name}"})

    @app.get("/runs/<job_id>/<path:filename>")
    def run_file(job_id: str, filename: str):
        directory = RUNS_DIR / job_id
        if not directory.is_dir() or not filename or ".." in Path(filename).parts:
            abort(404)
        return send_from_directory(directory, filename)

    @app.get("/previews/<path:filename>")
    def preview_file(filename: str):
        return send_from_directory(PREVIEW_DIR, filename)

    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok", "active_jobs": JOBS.count()})

    return app


def run_ui(
    host: str = "127.0.0.1",
    port: int = 5000,
    debug: bool = False,
    open_browser: bool = True,
) -> None:
    app = create_app()
    url = f"http://{host}:{port}"
    if open_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    print(f"\n  UAV Strategy Engine UI  ->  {url}\n")
    app.run(host=host, port=port, debug=debug, threaded=True, use_reloader=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Launch the UAV strategy web dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)
    run_ui(args.host, args.port, args.debug, not args.no_browser)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
