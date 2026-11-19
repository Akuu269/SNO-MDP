# Presenting the SNO-MDP project through the interactive UI

Members 1–3 deliver the PPT. Member 4 presents only the dashboard, following the shortened sequence in the [team presentation plan](team_presentation_plan.md): 3:30 target, four-minute hard maximum, and no main PPT slides. This longer document is preparation material, not a script to deliver in full during that slot.

## Your main message

> We built a reproducible, discrete safe-exploration simulation based on the authors’ released SNO-MDP benchmark. We examine how sensor-noise assumptions, confidence settings, and certificate retention affect safety and reward. Our extension revalidates certificates against current GP bounds.

The replay explains a mechanism. The multi-map study supports the empirical conclusions.

## Prepare

From `#34_Project`, run `python3 -m analysis_ui` and open **http://127.0.0.1:8765**. Use the existing instance if it already owns the queue. Refresh the browser after UI updates.

Open `demo_source` from **Saved demos & reproductions**. Check that the sidebar says the scientific source matches the frozen protocol. Keep the [report](report.md), [slides](presentation.html), [PDF](presentation.pdf), and [saved final study](../results/final/index.html) available.

## 1. Problem, environment, and observations

**Show:** `demo_source → Initial state`.

**Explain:** each cell is a possible rover position. A 20 × 20 grid has 400 states, with up/right/down/left/stay actions. The goal is useful reward while learning which directed actions satisfy a safety constraint.

| Information | Access |
|---|---|
| Noisy altitude/reward at the current cell and neighbours | Learner |
| Initial certificates and known movement dynamics | Learner |
| Complete true altitude/reward maps | Evaluator and oracle only |

State the environment accurately: this is a **headless port of the GP-Safety-Gym discrete benchmark**, using synthetic generated GP fields or the released fixture. The UI is not Mars data, hardware, or native legacy MuJoCo execution.

## 2. GP learning and the safety rule

**Show:** **Field inspector → GP altitude uncertainty**, with **Colour scale → Fixed for this run**. Move the slider; then switch to GP reward mean and true reward.

**Explain:** separate GPs learn altitude and reward from authorized noisy observations. Mean is an estimate; latent standard deviation describes uncertainty. Both phases continue learning online.

On the left map, blue is the model-certified reachable/returnable region; gold is optimistic only. Blue does not establish ground-truth safety. The path shows executed actions; red segments record violations. Read each colour scale’s limits; colours are not automatically comparable across different runs or fields.

Explain these equations:

```text
g(s,a) = altitude(s) − altitude(next_state)
safe when g(s,a) ≥ h; demo threshold h = −0.25
LCB = posterior mean of g − c × posterior STD of g
Var[g] = Var[z(s)] + Var[z(next)] − 2 Cov[z(s), z(next)]
```

The covariance term matters. A good individual bound is followed by graph reachability/returnability checks. Safety attaches to a direction: the reverse move can behave differently. This benchmark limits uphill altitude increase; it is not a complete physical rover safety model.

## 3. One concrete failure and why it happens

**Show:** **Before first violation** in `demo_source`.

The UI moves to **step 96**, position **(8, 8)**. Examine **up → (8, 9)**:

| Quantity | Value |
|---|---:|
| Threshold h | −0.25 |
| Current pre-action LCB | approximately −0.280632 |
| True action margin, evaluator only | approximately −0.279888 |
| Executable? | Yes |
| Support | Retained certificate |

**Say:** “The current bound fails the threshold, but this action was accepted earlier. The source profile retains the certificate, and the executable graph still permits the move.”

Press **First violation**: at step **97**, the rover arrives at **(8, 9)** and the cumulative unsafe-action count becomes **1**.

Distinguish an **incorrect certificate** from an **executed violation**. Revalidation requires a current passing bound or a supplied initial certificate, followed by graph checks. This example explains retention; it does not prove that all source violations have that cause, or that current bounds are always correct.

## 4. Exploration, ES2, and reward

**Show:** **Reward switch**, then **Final result**.

Explain that safety exploration gathers information. ES2 examines an auxiliary optimistic policy: when its relevant successors stay inside the executable region, exploration can stop and reward optimization can begin.

In this particular source demo, the switch is at **step 159**, leaving **one reward-optimization action** in the 160-action budget. Do not present it as a large early-stopping benefit. The revalidated demo has no switch within 160 actions; its disabled switch button reflects an absent event.

The phase badge describes the next decision. Reward and violation metrics describe actions already executed.

```text
cumulative reward = sum of true arrival rewards after actions
discounted return = r1 + γ r2 + γ² r3 + ...
```

Both metrics include exploration and travel costs. Horizon completion during exploration is distinct from model/planning failure.

## 5. Controlled comparisons and the final study

**Show:** **Compare & analyse → Load matched example**.

The four trajectories share map 100, observation seed 3000, horizon 160, and actual noise:

1. ES2 retained, **c=2**: source-profile baseline.
2. ES2 retained, **c=4**: confidence control.
3. ES2 revalidated, **c=4**: project extension.
4. Oracle: privileged true-field reference.

Compare the two c=4 conditions to examine certificate handling at fixed confidence. Comparing c=2 retained directly to c=4 revalidated changes two controls. Switch between return, unsafe actions, and false certificates. With two selected runs, show their configuration differences.

Next show **Aggregate analysis → Frozen final study → factor 1 → Update analysis**:

| ES2 condition | Runs / maps | Mean return | Mean unsafe actions / run |
|---|---:|---:|---:|
| Retained c=2 | 12 / 6 | 177.56 | 2.500 |
| Retained c=4 | 12 / 6 | 157.18 | 0.000 |
| Revalidated c=4 | 12 / 6 | 155.31 | 0.000 |

These are the **matched-noise subset**, not all-noise totals. Average observation repeats within each map, then average the six map means. SD is across map means, not a confidence interval. The full final study has **192 runs: 180 learner runs and 12 oracle references**.

**Defensible interpretation:** confidence tuning changes the observed safety/reward trade-off. Both c=4 ES2 conditions have zero violations at matched noise; that subset does not establish an additional violation reduction from revalidation.

Choose **All factors → Update analysis** to discuss robustness. Across the three factors, saved ES2 totals are **98 unsafe actions for retained c=2**, **2 for retained c=4**, and **0 for revalidated c=4**, with 36 runs per learner condition. Discuss these alongside reward, variability, stopping, and failures. Finite zero violations is not a proof.

Read comparison warnings. Different horizons, kernels, actual noise, source revisions, or map/observation cohorts prevent a one-change causal interpretation.

## 6. Run a small live experiment

**Show:** **New experiment → Quick 6 × 6**, then choose **Matched method comparison**.

Select retained c=4 and revalidated c=4. Explain:

- Map seed changes generated terrain; observation seed changes reproducible sensor randomness.
- Actual altitude noise changes readings.
- Assumed-noise factor changes the altitude GP’s noise assumption, not the readings.
- c changes confidence width and is separate from sensor noise.
- Horizon counts total physical actions.

Press **Preview cases**, inspect shared settings and the two controlled conditions, then **Run experiment**. Open a completed replay. Progress updates between cases; the app does not stream live steps.

These are exploratory outputs under `results/interactive/`. The 6 × 6 run checks the execution pipeline; it is not new final-study evidence.

In aggregate analysis, choose **All interactive runs**, select the specific batch, and update. Review planned/finished/setup-failure counts before interpreting trajectory means. Setup failures with no trajectory are disclosed separately; those means are conditional on available trajectories.

## 7. Reproducibility and limitations

**Show:** evidence downloads, exact settings/cohorts, **Export analysis JSON**, and **Export graph SVG**.

Explain that cases save configuration, maps, observations, trajectories, summaries, and source hashes. Analysis exports retain filters, run paths, settings, cohorts, batch context, and source identities; SVG metadata retains the corresponding context. Duplicate demo copies are excluded from aggregates. The source-match badge alone does not verify every result; the scientific runner has a separate protocol verifier.

Close with the actual scope: a documented source profile, robustness study, and labelled current-LCB extension. Do not claim complete theorem reproduction, P-ES2, exact published curves, native MuJoCo execution, hardware safety, or universal guarantees.

## Questions to prepare for

| Question | Point to explain |
|---|---|
| Why does safe RL record unsafe moves? | Practical confidence intervals can be wrong; historical permission can survive a later failing bound. Fixed multipliers do not establish the paper’s uniform confidence assumptions. |
| Why not show only the two demos? | Their c and mode both differ. One map is not multi-environment evidence. Use the c=4 control and the full study. |
| Does no switch mean failure? | Not by itself; distinguish exploration-budget exhaustion from model/planning failure. |
| Is oracle a universal return upper bound? | No. It uses privileged fields and an infinite-horizon policy rolled out for the finite budget. |
| What is the contribution? | Source audit, reproducible robustness experiments, certificate diagnostics, the revalidation extension, and the analysis interface. |

Every team member should explain one complete observe → GP → certificate → graph → action → evaluation loop and identify their actual contribution.
