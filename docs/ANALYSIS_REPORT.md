# Design Rationale, Hyperparameter Sensitivity and Uniqueness of the Evolved UAV Strategy

*Companion analysis to the technical report — Multi-Objective Genetic Strategy Engine for Autonomous UAV Disaster Surveillance*

*Artificial Intelligence — BSCS 5, Department of Computer Science, Bahria University Lahore Campus*  
*Assignment 2 (CLO4) — Analysis report*  

## Executive summary

This companion report explains why the strategy is encoded and weighted as it is, quantifies the sensitivity of the genetic search to its three main hyperparameters, and tests whether the evolved plan is genuinely different from blind sampling. Against an equal-budget random search, the GA raises mean fitness by 1.797 (from -1.187 to 0.609), triples priority coverage (3.47×) and raises confident detection 5.60×, while removing the constraint violations retained by every random best solution.

- Encoding: 190 genes in five semantic blocks are 13× smaller than a dense 50 × 50 occupancy chromosome, keep standard operators applicable, and let evolution retire two of five UAVs.
- Weighting: coverage contributes 49.8% and safety 26.9% of the reference fitness. Equal weights would select a different plan (0.680 coverage, 84.3% energy) instead of the 0.716-coverage compromise.
- Tuning: population 24–40 is a plateau; mutation 0.05 wins a short run but 0.20 is retained for diversity; tournament 4 is clearly best (0.609) versus 2 (0.449) and unstable at 8 (CV 11.7%).
- Uniqueness: the GA beats the best of 2,400 random draws after 176 evaluations and returns a 149-member Pareto archive of coordinated, feasible plans.

## 1. Chromosome Design Rationale

A strategy is a coupled decision over fleet size, deployment points, patrol order, sensor scheduling and leg speed. Changing a deployment point changes reachable targets, which changes the useful patrol order, which changes time and energy, which changes whether a UAV should fly at all. The genome therefore keeps these decisions in separate semantic blocks but evaluates them as one plan. Its fixed length makes every individual a rectangular NumPy array, so crossover, repair, elitism and JSON archiving are uniform operations.

**Table 1. Hybrid chromosome layout for five UAVs and eight patrol points.**

| Segment | Genes | Type / domain | Role | Why this representation |
|---|---|---|---|---|
| active | 5 | binary | UAV deployed or reserved | Fleet sizing; the reference retires two UAVs. |
| deploy | 5 × 2 = 10 | real, [0, 49]² | Initial loiter point | Separates loitering from patrolling. |
| patrol | 5 × 8 × 2 = 80 | real, [0, 49]² | Ordered waypoints | Order-aware segment/splice crossover. |
| sensor | 5 × 9 = 45 | real, [0, 1] | Power at nine nodes | Continuous power; values below 0.10 switch off. |
| speed | 5 × 10 = 50 | real, [0.55, 2.0] | Ten closed-route legs | Explicit time–energy trade-off. |
| Total | 190 | mixed | Complete strategy | One plan, one archive entry. |

The alternatives were considered and rejected for concrete reasons. A dense occupancy mask needs 2,500 bits, has no route order or sensor schedule, and must reconstruct tours ambiguously. A permutation of cells suits one travelling salesman but cannot express five cooperating UAVs, loiter points or per-node power. Separate variable-length routes per UAV are flexible but break uniform crossover, elitism and archiving. The hybrid is not merely smaller; it makes the repair rules auditable.

The design's effect is visible in the reference genome: 3 of 5 UAVs are active, deployed roughly 19.6 cells apart, with patrol centroids only 3.5–6.1 cells from their loiter points. 26 of 27 sensor slots on active aircraft are switched on at mean power 0.89; total flown distance is 239.0 cells and the longest route is 107.1 cells (61% of the 175-cell per-drone limit). At generation 1, five aircraft consumed 1.77 of the energy budget with a penalty of 1.78. The activation gene is what allows the final plan to reach 0.716 coverage with three aircraft at 99.6% energy and zero penalty.

![Figure 1. Evolved deployment and patrol trajectories (left) and the fused noisy-OR coverage heatmap (right). Stars mark deployment points; polygons show no-fly airspace and priority zones. Coverage is concentrated on the three P3 zones rather than spread uniformly.](outputs/run/figures/evolved_paths.png)

![Figure 1. Evolved deployment and patrol trajectories (left) and the fused noisy-OR coverage heatmap (right). Stars mark deployment points; polygons show no-fly airspace and priority zones. Coverage is concentrated on the three P3 zones rather than spread uniformly.](outputs/run/figures/coverage_heatmap.png)

*Figure 1. Evolved deployment and patrol trajectories (left) and the fused noisy-OR coverage heatmap (right). Stars mark deployment points; polygons show no-fly airspace and priority zones. Coverage is concentrated on the three P3 zones rather than spread uniformly.*

## 2. Fitness Weighting Rationale

Sensors are fused with a noisy-OR model, so overlapping footprints raise confidence smoothly. The five normalised objectives are priority coverage C, weighted detection D, energy E, time T and hazard exposure R. The scalar fitness is

```text
f = 0.45 C + 0.15 D + 0.10 (1 - E/36) + 0.10 (1 - T/240) + 0.20 (1 - R) - P
P = 4 f_nofly + 3 max(0, 0.55 - C) + 2 max(0, R_range - 1) + 2 max(0, T/240 - 1)
```

**Table 2. Default weights and their realised contribution to the reference fitness of 0.6467.**

| Objective | Weight | Score at reference | Contribution to f | Share of f |
|---|---|---|---|---|
| Coverage C | 0.45 | 0.716 | 0.322 | 49.8% |
| Detection D | 0.15 | 0.908 | 0.136 | 21.1% |
| Energy (1 - E/36) | 0.10 | 0.004 | 0.000 | 0.1% |
| Time (1 - T/240) | 0.10 | 0.140 | 0.014 | 2.2% |
| Safety (1 - R) | 0.20 | 0.870 | 0.174 | 26.9% |
| Total | 1.00 | — | 0.647 | 100.0% |

The allocation is intentionally unequal. Coverage receives 0.45 because a route that misses the P3 zones has failed the mission. Safety receives 0.20: because (1 − R) is near 0.87 for an ordinary route, this is a substantial always-present term that penalises excursions into hazard plumes, not a rare tie-breaker. Detection receives 0.15 to separate strategies with similar area coverage but different target confidence. Energy and time each receive 0.10 as soft efficiency terms; hard limits are already enforced by range, time and coverage penalties. At the reference point energy contributes only 0.0004 because the strategy spends 99.6% of its budget to buy coverage. Penalty weights (4/3/2/2) are an order of magnitude larger than any objective gain, so feasible strategies dominate infeasible ones and the five weights decide only within the feasible region.

The weights are not interchangeable. Rescoring the 149-member archive with equal weights selects a different plan: coverage 0.680, detection 0.886, energy 84.3% and risk 0.121, instead of the default choice at 0.716, 0.908, 99.6% and 0.130. The default therefore buys roughly 3.6 coverage points with 15 energy points. The archive also shows why scalarisation is not the whole answer: its extremes run from 0.525 to 0.718 coverage, 62.3% to 116.0% energy and 0.075 to 0.157 risk, so a human can inspect alternatives that no single weight vector selects.

## 3. Hyperparameter Sensitivity Analysis

A one-at-a-time study varied population size P ∈ {24, 40, 60}, mutation rate r_m ∈ {0.05, 0.15, 0.30} and tournament size k_t ∈ {2, 4, 8} over 40 generations with two seeds (11, 23), holding the mission map (seed 42), 90% crossover, two elites and 4% immigrants fixed. Two seeds expose large effects but not narrow confidence intervals, so the discussion separates real gaps from noise.

**Table 3. Sensitivity results, mean ± s.d. over two seeds. Energy and time are fractions of their budgets.**

| Parameter | Value | Fitness | Coverage | Detection | Energy | Time | Risk | Penalty |
|---|---|---|---|---|---|---|---|---|
| Population size | 24 | 0.614 ± 0.001 | 0.622 | 0.889 | 95.1% | 81.7% | 0.115 | 0.000 |
| Population size | 40 | 0.619 ± 0.019 | 0.646 | 0.872 | 89.7% | 88.4% | 0.122 | 0.000 |
| Population size | 60 | 0.609 ± 0.025 | 0.603 | 0.839 | 84.5% | 75.3% | 0.140 | 0.000 |
| Mutation rate | 0.05 | 0.636 ± 0.027 | 0.674 | 0.883 | 96.1% | 82.4% | 0.109 | 0.000 |
| Mutation rate | 0.15 | 0.621 ± 0.031 | 0.683 | 0.898 | 108.3% | 90.9% | 0.108 | 0.000 |
| Mutation rate | 0.3 | 0.612 ± 0.002 | 0.640 | 0.864 | 96.9% | 78.2% | 0.152 | 0.000 |
| Tournament size | 2 | 0.449 ± 0.017 | 0.546 | 0.741 | 128.9% | 101.5% | 0.186 | 0.041 |
| Tournament size | 4 | 0.609 ± 0.025 | 0.603 | 0.839 | 84.5% | 75.3% | 0.140 | 0.000 |
| Tournament size | 8 | 0.572 ± 0.067 | 0.623 | 0.801 | 105.5% | 95.6% | 0.131 | 0.001 |

Population size is a plateau between 24 and 40: 0.619 ± 0.019 at 40 versus 0.614 ± 0.001 at 24, a 0.005 gap well inside the spread. Size 60 costs 2.5× the evaluations of 24 at the same generation count yet returns 0.609 ± 0.025; 40 generations end before the larger population converts diversity into a better plan. Use 40 for short runs and 60 for the 80-generation reference, where it reaches 0.6467.

Mutation shows the expected short-horizon bias. Rate 0.05 gives the best fitness (0.636) and detection (0.883) because it polishes the hotspot-aware seed; 0.15 gives the best coverage (0.683) but spends 108.3% of the energy budget; 0.30 is weakest on fitness and raises risk to 0.152. The 0.05 advantage (0.014) is smaller than the seed spread (~0.03), and a long run needs variation to escape stalls. The defensible range is 0.15–0.20; 0.05 is useful only for short polishing runs.

Selection pressure has the largest and cleanest effect. Tournament 2 under-selects (0.449), leaves a penalty of 0.041, consumes 128.9% of energy and 101.5% of time, and reaches only 0.546 coverage. Tournament 4 is best (0.609) with zero penalty. Tournament 8 lowers mean fitness to 0.572 and triples the coefficient of variation to 11.7% (one seed 0.505, the other 0.639): a few early winners dominate before a robust route is assembled. Keep k_t = 4; if population grows, 5–6 is reasonable, but weak pressure (2) is the single most damaging setting.

Two caveats bound these conclusions: standard deviations from two seeds are indicative, and the one-at-a-time design cannot reveal interactions (for example mutation × selection pressure). A factorial design with five or more seeds is the natural next experiment.

![Figure 2. Fitness sensitivity to population size, mutation rate and tournament size. Error bars are one standard deviation over two seeds.](outputs/experiments/sensitivity_combined.png)

*Figure 2. Fitness sensitivity to population size, mutation rate and tournament size. Error bars are one standard deviation over two seeds.*

## 4. Evolved Strategy versus Random Search

Random search shares everything except the evolutionary machinery: the same chromosome distribution, repair operator, noisy-OR evaluator and mission, with the best of 2,400 draws per seed (40 generations × 60 individuals = 2,400 evaluations), matching the GA budget. This isolates the contribution of selection, recombination and mutation.

**Table 4. Equal-budget comparison. Positive change means the GA is higher; negative means it is lower. For energy, time, risk and penalty, lower is better.**

| Metric | GA (mean ± s.d.) | Random search (mean ± s.d.) | Change |
|---|---|---|---|
| Fitness (↑) | 0.609 ± 0.025 | -1.187 ± 0.056 | +151.3% |
| Priority coverage (↑) | 0.603 ± 0.009 | 0.174 ± 0.045 | +246.8% |
| Weighted detection (↑) | 0.839 ± 0.021 | 0.218 ± 0.016 | +284.7% |
| Confident detection rate (↑) | 0.920 ± 0.006 | 0.164 ± 0.025 | +460.4% |
| Energy / budget (↓) | 0.85 ± 0.14 | 1.39 ± 0.22 | -39.3% |
| Time / budget (↓) | 0.75 ± 0.19 | 1.06 ± 0.14 | -28.6% |
| Hazard risk (↓) | 0.140 ± 0.000 | 0.065 ± 0.004 | +115.0% |
| Constraint penalty (↓) | 0.000 ± 0.000 | 1.440 ± 0.044 | -100.0% |

The GA improves fitness by 1.797 (+151.3%), roughly 41 pooled standard deviations. The distributions do not overlap: the GA minimum 0.584 exceeds the random maximum −1.131. Coverage rises 3.47× and detection 3.85×; confident detection rises from 0.164 to 0.920 (5.60×). The GA beats the best of all 2,400 random draws after only 176 evaluations, so the advantage is present almost immediately rather than being a late artefact.

The stronger difference is qualitative. Every random best solution activates all five UAVs and still violates the mission: 139.3% of the energy budget, 105.6% of the time budget, a penalty of 1.440 and 2.4–3.4% no-fly crossings. The GA uses two or three UAVs, 84.5% energy and 75.3% time with zero penalty and zero no-fly violations. It also spends 1.40 energy-budget units per unit coverage versus 8.00 for random search, a 5.7× efficiency advantage.

Hazard risk is the one raw metric that favours random search (0.065 versus 0.140), because random routes often stay in low-hazard outskirts and never reach the priority zones. Normalised by coverage the ordering reverses (0.232 versus 0.375 risk per unit coverage, a 38% reduction). A decision-maker who truly prioritised safety could pick the archive's minimum-risk member (risk 0.075) at the cost of 0.525 coverage and 116.0% energy.

Structurally, the reference run returns 149 non-dominated strategies with 91 distinct active deployment sets; 8 use two UAVs and 141 use three, none use four or five. Random search produces one best draw and a noisy best-so-far curve. It cannot preserve a non-dominated alternative, recombine a good deployment block with another parent's sensor schedule, or hold a UAV in reserve while improving the active routes. The GA's advantage is therefore not just a higher score: it is a coordinated, feasible, multi-alternative strategy rather than an isolated lucky sample.

![Figure 3. Best-genome coverage footprints at generations 0, 25%, 50%, 75% and 100% (left), and the non-dominated coverage–energy–risk archive with the scalarised best compromise marked (right).](outputs/run/figures/coverage_evolution.png)

![Figure 3. Best-genome coverage footprints at generations 0, 25%, 50%, 75% and 100% (left), and the non-dominated coverage–energy–risk archive with the scalarised best compromise marked (right).](outputs/run/figures/pareto_front.png)

*Figure 3. Best-genome coverage footprints at generations 0, 25%, 50%, 75% and 100% (left), and the non-dominated coverage–energy–risk archive with the scalarised best compromise marked (right).*

## 5. Conclusions and Recommendations

- Keep the 190-gene hybrid encoding and the three spatial crossovers; it is compact, auditable and directly supports deployment, patrol, sensor and speed decisions.
- Keep the 0.45/0.15/0.10/0.10/0.20 weights for the reported mission, but expose presets: equal weights would choose a materially more economical 0.680-coverage plan.
- Use population 40 for 40-generation runs, population 60 for the 80-generation reference, mutation 0.15–0.20 for long runs, and tournament size 4. Treat population 24–40 as a plateau until more seeds are available.
- Present the equal-budget comparison as the main evidence of usefulness: +1.797 fitness, 3.47× coverage, 5.60× confident detection, zero penalties and a 149-member archive. Do not claim random search is useless; claim it cannot produce a coordinated, feasible, multi-alternative plan at this budget.

## Reproducibility

```bash
python -m uav_strategy run --generations 80 --population 60
python -m uav_strategy experiments --generations 40
python scripts/build_analysis_report.py
python scripts/build_report.py    # main technical report
```
