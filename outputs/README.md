# Generated artefacts

Everything in this directory is produced automatically and can be regenerated from the
documented seeds.

## `outputs/run/` — reference simulation

| File | Description |
|---|---|
| `run_summary.json` | Configuration, best fitness and full metric summary |
| `best_genome.json` | Best evolved chromosome (all five gene segments) |
| `history.json` | Per-generation best/average fitness, diversity, coverage, energy, risk |
| `pareto_archive.json` | Non-dominated strategies with metrics and genomes |
| `config.json` | Exact mission/GA/fitness configuration used |
| `figures/mission_map.png` | Hazard plumes, priority zones, no-fly zones, targets and base |
| `figures/evolved_paths.png` | Final UAV deployment and patrol trajectories |
| `figures/coverage_heatmap.png` | Fused sensor coverage over the final paths |
| `figures/coverage_evolution.png` | Coverage footprints at generations 0, 20, 60 and 80 |
| `figures/fitness_convergence.png` | Best vs average fitness and diversity |
| `figures/pareto_front.png` | Coverage–energy–risk trade-off surface |

## `outputs/experiments/` — study

| File | Description |
|---|---|
| `sensitivity_combined.png` | Population / mutation / selection-pressure summary |
| `sensitivity_population_size.png` | Population-size sweep |
| `sensitivity_mutation_rate.png` | Mutation-rate sweep |
| `sensitivity_tournament_size.png` | Tournament-size sweep |
| `baseline_vs_ga.png` | Equal-budget GA vs random-search convergence and final metrics |
| `sensitivity_runs.json` | Raw per-seed sensitivity runs |
| `sensitivity_summary.json` | Mean/std summary per configuration |
| `baseline_comparison.json` | Aggregate GA vs baseline comparison |
| `baseline_detail.json` | Per-seed histories and metrics |
| `experiment_summary.json` | Combined bundle consumed by the report builder |

The technical report is written to `docs/TECHNICAL_REPORT.docx` and
`docs/TECHNICAL_REPORT.pdf`; the Markdown version lives at the repository root as
`REPORT.md`.
