# Review of the supplied presentation outline

The outline adds useful teaching structure, but it mixes the paper, another proposed experiment, and this implemented project. The updated slides preserve those distinctions.

## Evidence used

- [Published paper](https://proceedings.mlr.press/v119/wachi20a.html) and [official PDF](https://proceedings.mlr.press/v119/wachi20a/wachi20a.pdf), especially Sections 2–6 and Table 1.
- [Local paper with appendices](../SNO_MDP_Wachi_Sui_2020_arxiv_v1.pdf).
- [Method/source audit](../docs/method.md), [configuration](../configs/demo_source.json), [frozen suite](../configs/research_final.json), and [recorded final results](../results/final/summary.json).
- Current `sno_mdp/` implementation and `analysis_ui/` controls.

## Important corrections

| Supplied statement | Verified interpretation | Presentation treatment |
|---|---|---|
| Safety is strictly guaranteed with zero violations at every step | The paper’s safety theorem is a simultaneous high-probability statement under its hypotheses. It is not unconditional zero risk under arbitrary misspecification. | Main slides 1 and 3 state the qualification; project results remain empirical. |
| Both functions are completely unknown | Their values are unknown, but dynamics, a safe seed, kernels/regularity, and noise assumptions provide essential structure. | Slides 2–3 show what is known and unknown. |
| SafeMDP stays put or wanders aimlessly | It deliberately explores the safely certifiable region to reduce uncertainty; reward maximization is not its objective. | Slide 4 describes the objective fairly. |
| Safe-ExpOpt-MDP lacks near-optimality guarantees | The 2020 paper identifies this gap for its comparator; that is a literature comparison, not a measured baseline in our frozen suite. | Explain in viva; do not invent a local result for this method. |
| GP bands are simply μ ± σ | A confidence multiplier matters. The paper uses a confidence schedule and historical intersections; our released-source port uses direct c multipliers and retained action acceptance. | Slides 5 and 7 distinguish notation and behavior. |
| Pessimistic and optimistic sets need only lower/upper bounds | Reachability, returnability, and the paper’s Lipschitz construction also matter. Optimistic membership is not physical execution permission. | Slide 6 defines the regions and expanders. |
| Expanders are boundary cells | They are candidates whose optimistic information could certify further territory. Our source port tracks state-action pairs. | Slide 6 corrects this shortcut. |
| Basic SNO maps every corner, then freezes all learning | The paper uses an accuracy criterion for safely reachable expansion; this need not reveal every true-safe state. Our port continues GP/certificate updates in both phases and checks executable-pair widths. | Slides 6–8 and notes explain the actual behavior. |
| Reward multiplier 3 makes ES2 rarely stop early | The paper motivates P-ES2 through optimism over possibly safe states. No single multiplier establishes a universal stopping claim. Our matched-noise c=2 ES2 triggers in 11/12 runs and averages 117.42 exploration actions, versus 160 for basic SNO. | Do not claim ES2 is inherently ineffective. Use actual stopping records. |
| P-ES2 is our implemented winner with a fivefold speedup | P-ES2 appears in the paper; it is absent from our implemented method list and final inventory. The paper reports faster empirical convergence without the same reward near-optimality guarantee. | Main slide 8 and backup 16 label it paper-only; no project performance score or universal speed ratio. |
| We ran 5 seeds × 600 steps and 5 Mars sites × 800 steps | Our frozen evaluation is 6 generated maps × 2 observation seeds, 160 actions, three assumed-noise factors, 180 learner cases plus 12 oracle references. | Main slide 9 and all result graphs retain the actual protocol. |
| Our data are a named HiRISE DTM with 40 training crops and held-out sites A–E | Those data/crops/sites are not in the current implementation or frozen records. The reviewed paper describes a HiRISE Mars crop, not this claimed project protocol. | Literature context only; no claimed Mars implementation or results. |
| Mars normalized reward is 0.87 for P-ES2 and 0.66 for SNO | The paper’s Table 1 reports 0.81 for P-ES2, 0.78 for ES2, and 0.49 for SNO-MDP, with zero recorded unsafe actions for those author experiments. These are not our measurements. | Backup 15 shows the author numbers and labels their origin. |
| Project nominal sensor noise is 0.02 | Our altitude/reward sensor STD is 0.001. Safety c is 2 by source default and 4 by development selection; reward c is 3; shared RBF length scale is 2. | Slide 5 and backup 17 use actual configuration values. |
| β=2 implies a run-level 97.7% guarantee | Direct c=2 gives about 97.7% one-sided mass for one correctly specified Gaussian posterior. It is not simultaneous coverage over adaptive actions/time. Paper β and code c are not interchangeable. | Backup 17 and notes explain pointwise versus uniform claims. |
| Larger/smaller sliders guarantee stalling, violations, or local optima | At fixed posteriors, multipliers predictably affect bounds/bonuses. Entire reruns can follow different paths and observation histories, so outcomes need not change monotonically. | Parameter notes use tendencies, not guaranteed outcomes. |
| The current dashboard is explorer.html with hillshade, slope degrees, probability tabs, and orange expander diamonds | Our dashboard is launched with `python3 -m analysis_ui`. It shows blue executable/gold optimistic regions, GP fields, trajectories, action-bound inspection, experiments and comparisons. It does not implement those Mars/slope/probability views. | Member 4 follows the implemented interface; the extra views belong to a different or proposed interface. |
| GPs guarantee safety while neural networks cannot quantify uncertainty | GPs fit the paper’s kernel/RKHS confidence analysis. Neural uncertainty methods also exist; either model class requires appropriate assumptions and calibration. | Backup 18 gives a precise viva answer. |

## Probability and parameter notation

For a Gaussian posterior with positive STD:

```text
p(s) = Φ((μg(s) − h) / σg(s))
     = 1 − 0.5 erfc((μg(s) − h) / (sqrt(2) σg(s)))
```

This is the GP model’s pointwise probability. P-ES2 uses it in a virtual planning model with an absorbing zero-reward state; it does not authorize real risk-taking moves outside the certified region. Its safety remains conditional, and its practical reward behavior should not be presented as the ES2 near-optimality theorem.

Products of these virtual transition weights describe the auxiliary planning model. They are not automatically calibrated joint safety probabilities for paths through GP-correlated terrain.

For our project:

Notation matters: the paper writes safety intervals with `sqrt(beta_t)` and reward optimism with `sqrt(alpha_(t+1))`. Our code uses direct multipliers `c_s` and `c_r`. Matching the printed numbers 2 or 3 does not make the formulas or theoretical schedules identical.

```text
safety LCB = μg − c_s σg          c_s: 2 or selected 4
reward UCB = μr + c_r σr         c_r: 3
actual altitude / reward STD     0.001 / 0.001
assumed altitude STD             actual altitude STD × factor
factors                         0.5, 1, 2
shared RBF length scale          2
exploration accuracy             0.005
discount / action threshold      0.99 / −0.25
final physical-action horizon    160
```

The paper’s Mars configuration is different: 40 × 30 crop, 25° slope criterion, Matérn-5/2 elevation GP, 15 m length scale, 100 m² elevation variance, 0.075 m elevation noise STD, and discount 0.9. Those units and slope threshold must not be substituted into our dimensionless directed-altitude benchmark.

The current source expander setting `L=0` is not a claim that the varying terrain is globally zero-Lipschitz. We therefore do not assert that the fixed-c source port satisfies the paper’s theorem hypotheses.

## What changed in the deck

- Main slides 1–4 now establish the objective, unknowns, assumptions, prior-method objectives, and implementation boundary.
- Main slides 5–8 define bounds, X−/X+/expanders, the source/extension distinction, and ES2 versus paper-only P-ES2.
- Main slides 9–12 retain the real 192-case protocol and its reward, safety and robustness graphs.
- Backups 15–18 cover original-paper experiments, P-ES2, parameters, and viva preparation. Backup 19 retains reproducibility and limits.
- Members 1–3 still have four main slides each. Member 4 still presents the dashboard only. Extra theory belongs to Q&A, not a longer main talk.

The scientific implementation and frozen results were not changed. Implementing P-ES2 or validating Mars terrain would require separate code, protocol, and measured evidence.
