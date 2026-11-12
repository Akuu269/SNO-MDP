# Experiment protocol

## 1. Configurations

| File | Purpose |
|---|---|
| `configs/smoke.json` | Small generated-map integration check; not benchmark evidence |
| `configs/released.json` | Pinned released 20 × 20 map and nominal source settings |
| `configs/generated.json` | Independent GP maps for future validation/evaluation |
| `configs/study_pilot.json` | One released map, one observation seed, three noise factors; development only |
| `configs/study_validation.json` | New generated maps with matched noise; validation, not final test data |

The released map does not change with `map_seed`. Suites reject multiple released-map seeds masquerading as independent maps. Generated maps use separate altitude and reward GP draws; reward is shifted by its minimum as in the source, not normalized to one. An invalid initial region is a setup error, never repaired by altering fields or selecting for reward.

## 2. Controlled comparison

Methods: `safemdp`, `sno_basic`, `sno_es2`, `oracle`. Give learners the same map, initial certificates, sensor access, actual noise, total horizon, and model settings. The oracle's information advantage is explicit.

The nominal actual altitude-noise STD is 0.001. Assumed STDs are 0.0005, 0.001, and 0.002 for factors 0.5, 1, and 2. Reward-noise STD stays correctly specified at 0.001. These are altitude-coordinate units, not Mars meters. Do not combine an alternative noise scale with this study without labeling it as another experimental condition.

The paper uses discount 0.99 and a 20 × 20 synthetic grid. Our released-map demonstration has a 120-action horizon; the first comparison pilot uses 40. Both have a common total physical-action budget and no forced phase switch. State these differences in every comparison.

## 3. Split and freeze

1. Treat released-map and smoke results as development evidence.
2. Use map seeds 31, 37, 41 and observation seeds 2100, 2101 in the validation suite for calibration/stopping choices.
3. After validation, reserve fresh generated-map seeds, for example 100–109 with observation seeds 3000–3004. These final conditions are proposed, not a completed evaluation.
4. Freeze source revision, kernels, confidence settings, tolerance, horizon, methods, metrics, and exclusion/failure rules before viewing held-out outcomes.
5. Run every planned condition. Report setup errors separately and fix defects consistently across affected matched conditions.

Changing settings after examining final outcomes requires fresh final evidence. A missing ES² benefit is an acceptable finding; it is not a reason to select favorable held-out maps.

## 4. Measurements

- Raw cumulative reward and discounted return, including exploration/travel costs.
- Unsafe directed actions and largest threshold shortfall.
- Incorrect certified actions, even if not executed.
- Executable-state/action growth, exploration effort, transition step/reason, and ES² outside successors.
- Model/planning failures and horizon exhaustion during exploration, reported separately.
- Inference and decision computation from `timing.csv`; do not include figure rendering in policy cost.
- Map-level variability and ES²/basic paired differences on matched map/observation seeds.

No hidden truth-based action block or safety-based termination is used. Early model failures retain their executed return prefix and receive zero extra credit for the unused scheduled budget. Actual action counts remain visible; no steps are fabricated. The primary analysis includes failures.

Average observation-seed replicates within each map, then aggregate across maps. Reported SD is across map means, not a confidence interval. The reference is an infinite-horizon discounted policy rolled out for the same finite horizon; ratios are descriptive, not universal upper bounds. Repeating one map cannot establish variability across environments.

## 5. Artifacts

| Artifact | Use |
|---|---|
| `config.json`, `manifest.json`, `map.json` | Settings, exact source/data hashes, evaluator fields |
| `observations.json.gz` | Learner-accessible noisy data; no hidden terrain values |
| `trajectory.json.gz`, `progress.csv` | Actions, phases, certificates, returns, and actual safety outcomes |
| `es2_checks.json.gz` | Auxiliary action rewards, policy, successors, outside sources, residual |
| `timing.csv` | Nondeterministic computation measurements, separated from repeatable algorithm artifacts |
| `index.html` | Saved replay, explicitly labeled headless |
| `overview.png`, `overview.svg` | Optional exported report visualization |

A study additionally saves `runs.csv`, `aggregates.csv`, `paired_es2_vs_basic.csv`, configuration, curves, summary, and standalone figures. Full traces use lossless gzip compression. Read them with `sno_mdp.artifacts.read_json` or Python’s `gzip.open`. The standalone replay embeds its compressed data and needs a current browser with `DecompressionStream` support; it requires no server or network.

Run `python3 -m sno_mdp diagnose --run <run-directory>` or `--study <study-directory>` to add a separate `safety/` analysis:

| Diagnostic artifact | Meaning |
|---|---|
| `index.html`, `REPORT.md`, `diagnostics.png/.svg` | Local report and optional exportable figures |
| `coverage_by_step.csv`, `coverage_by_multiplier.csv` | Moving-action coverage; alternative multipliers use fixed trajectories |
| `executed_actions.csv`, `unsafe_actions.csv` | Pre-action bounds, first-acceptance bounds, and current/retained support |
| `certificate_history.json.gz` | First acceptance, first executable step, and first execution |
| `summary.json`, `manifest.json` | Counts, coverage denominators, verified inputs, and analysis source hashes |

Study diagnostics save run-level metrics, map-level aggregates, and links to every learner's report; per-run figures can be generated with `--run`. Coverage pools edge-frame counts within a run, averages measured observation replicates within each map, then averages measured maps. Missing-measurement denominators and failure statuses remain explicit. Original experiment settings and results are not rewritten. These diagnostics do not replace policy reruns for calibration.

## 6. Professor presentation

Present: problem → source profile → GP covariance → directed safe exploration → ES² → controlled comparison → findings and limitations. Show one worked example, a saved benchmark replay, and a failure/incorrect-certificate case. Identify designed illustrations separately from aggregate experiments.

Required figures: reward against actions, executable-region growth, certificate/violation counts by noise assumption, ES²/basic paired differences, and computation cost. Label axes, units, horizon, sample counts, normalization, and method deviations.

Native legacy rendering, exact published curves, theorem guarantees, P-ES², and the measured-Mars application remain separate work. See [the roadmap](../05_Project_Plan.md).

## 7. Completed-project study design

The controlled `research` runner executes complete policies and saves every planned case, setup error, failed prefix, current-bound check, and computation measurement. It is distinct from the diagnostic multiplier comparison, which holds trajectories fixed.

- **Calibration:** maps 31/37/41, observations 2100/2101, matched noise, horizon 120; retained/current-LCB modes × multipliers 2/3/4, plus the oracle (42 cases).
- **Selection:** among current-LCB ES² candidates, minimize failure rate, then unsafe-run rate, then maximize mean reference ratio, then choose the smaller multiplier. The rule and final seeds/budget are saved before calibration outcomes.
- **Stopping:** maps 31/37/41, observation 2100, horizons 120/240; source c=2 basic/ES² and selected-current-LCB basic/ES², plus the oracle (30 cases). The final horizon remains the predeclared 160.
- **Final:** fresh maps 100–105, observations 3000/3001, noise factors 0.5/1/2, horizon 160. Compare source c=2 SafeMDP/basic/ES², selected-confidence retained ES² when distinct, selected-current-LCB ES², and an oracle shared across noise assumptions.

The selected configuration is saved in `configs/research_final.json`. `report/frozen_protocol.json` records exact source hashes, full settings, every planned case, and accounting rules before final evaluation. No final-outcome tuning is permitted. The final sample size is six maps: findings are descriptive and limited to this simulation profile.

Final research artifacts include `planned_cases.json`, `runs.json/.csv`, `aggregates.csv`, `curves.json.gz`, paired differences, figures, selected replays, and `verification.json`. All runs are included in raw return means. Ratios with a failed/missing oracle remain missing rather than dividing by zero. Failures stop physical execution; padded curves show no extra reward, and any carried region size describes the last belief, not continued exploration.

Inference timing includes GP fitting and safe-set updates. Decision timing includes planning, not visualization. The oracle’s one-time privileged policy construction is not included in the per-decision timing comparison. Map SD and paired-map SD are variability summaries, not significance tests or safety bounds.
