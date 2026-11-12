# SNO-MDP: reproduction, noise robustness, and certificate revalidation

**Status:** Automated work completed; human team review and names remain to be supplied.

## Question and contribution

How do noise assumptions and retention of old safety certificates affect safe exploration, reward, and ES²? We provide a source-audited headless reimplementation and a current-LCB revalidation extension. This course project claims empirical implementation and analysis, not a new theorem, exact published curves, or patentability.

## Method

Independent 20 × 20 RBF-GP altitude/reward maps, spacing 0.5, discount 0.99, and threshold −0.25. Directed safety is g(s,a)=z(s)−z(next). Separate GPs learn from noisy local readings; transition uncertainty includes joint covariance. Exploration uses reachable/returnable certified actions; ES² checks auxiliary-policy successors; reward learning continues online. [Paper/source deviations](../docs/method.md) are part of the method specification.

The original profile retains accepted certificates. The extension requires supplied initial certificates or a passing current LCB, then recomputes reachability/returnability. If the current state leaves that region, motion halts and the run counts as failure. No evaluator truth selects or vetoes learner actions. Repeated readings use exact Gaussian count/mean sufficient statistics, including jitter divided by count.

## Development selection and stopping

Calibration reran both modes at multipliers 2/3/4 on maps 31/37/41 and two observation seeds. The predeclared failure-rate → unsafe-run-rate → negative reference-ratio → multiplier ranking selected **4**. [Selection record](../results/research_calibration/selection.json). No final outcome was used for selection.

Stopping validation completed 30 cases at 120/240 actions. Accuracy stops, ES² stops, region loss, and exploration-budget exhaustion are separate outcomes. [Stopping results](../results/research_stopping/RESULTS.md).

## Frozen final evaluation

192/192 planned cases completed across 6 fresh maps, two observation seeds, noise factors 0.5/1/2, and a predeclared 160-action horizon. The original c=2 methods, selected-confidence retained ES² control when distinct, revalidated ES² extension, and privileged oracle use matched conditions.

Every setup/model/planning/region-loss failure remains in the inventory. Unexecuted actions earn zero extra reward. Average observation replicates within each map, then average maps equally. SD describes map variability, not confidence intervals. The oracle is an infinite-horizon policy rolled out for the same finite budget; reference ratios are descriptive.

Matched-noise results:

| Profile:method | Runs/maps | Mean return | Map SD | Unsafe-run fraction | Failure fraction | ES² fraction |
|---|---:|---:|---:|---:|---:|---:|
| oracle:oracle | 12/6 | 268.96 | 74.68 | 0.00% | 0.00% | 0.00% |
| source:safemdp | 12/6 | 158.93 | 69.51 | 83.33% | 0.00% | 0.00% |
| source:sno_basic | 12/6 | 158.93 | 69.51 | 83.33% | 0.00% | 0.00% |
| source:sno_es2 | 12/6 | 177.56 | 80.04 | 83.33% | 0.00% | 91.67% |
| confidence:sno_es2 | 12/6 | 157.18 | 55.97 | 0.00% | 0.00% | 16.67% |
| revalidated:sno_es2 | 12/6 | 155.31 | 55.05 | 0.00% | 0.00% | 16.67% |

Across all final conditions: **0 failed runs and 384 unsafe actions**. Matched policies and repeated observations are correlated, so these are not independent-trial risk estimates.

[Every noise condition and result](../results/final/RESULTS.md) · [Dashboard and replays](../results/final/index.html) · [Frozen protocol](../results/final/protocol.json) · [Every planned case](../results/final/runs.csv)

## Figures

- [Reward against actions](../results/final/figures/returns.png)
- [Executable-region growth](../results/final/figures/region_growth.png)
- [Incorrect certificates](../results/final/figures/certificate_errors.png)
- [Safety, failure, exploration and planning cost](../results/final/figures/outcomes.png)
- [Paired map-level differences](../results/final/figures/paired_returns.png)

## Verification and reproduction

Verified all 192 final cases and 30720 physical actions against directed dynamics, execution certificates, return accounting, map/source hashes, and the frozen protocol. Numerical, graph, revalidation, fallback, failure, protocol, and integration tests accompany the implementation.

From the project directory:

```bash
python3 -m unittest discover -s tests
python3 -m sno_mdp verify --study results/final
python3 -m sno_mdp research --suite configs/research_final.json --output results/final_reproduced --protocol report/frozen_protocol.json
```

## Limitations

Finite maps and practical confidence bands establish no uniform safety guarantee. Revalidation can sacrifice reward or completion. The extension remains distinct from the original method. Kernel/noise assumptions, initial certificates, and observation rules limit transfer. Native legacy MuJoCo rendering, P-ES², and Mars experiments are outside this submission scope.

## Presentation, team, and references

[Presentation PDF](presentation.pdf) · [Browser slides](presentation.html) · [Speaker notes](presentation.md) · [Human team review](team_review.md).

- [Wachi and Sui, ICML 2020](https://proceedings.mlr.press/v119/wachi20a.html).
- [Local paper](../SNO_MDP_Wachi_Sui_2020_arxiv_v1.pdf).
- [Source revisions, licenses, and checksums](../third_party/README.md).
- [Method audit](../docs/method.md) and [experimental protocol](../docs/experiments.md).
