# Professor demonstration walkthrough

For the interactive dashboard, use the [UI presentation guide](ui_presentation_guide.md), which gives the exact screens, controls, worked failure example, and interpretation of the measured results.

## Start with the question

We reproduce a documented SNO-MDP source profile, investigate sensor-noise assumptions, and compare an explicitly labeled certificate-revalidation extension. A finite experiment is not a safety proof.

## Explain one decision

1. A 20 × 20 grid has 400 states. Coordinates are `(x index, y index)`; each action moves to a neighboring state or stays. The small 6 × 6 configuration is an integration check.
2. The rover receives noisy altitude and reward readings at its current state and neighboring states.
3. The altitude GP predicts directed differences, including covariance. A transition passes the model threshold when its lower bound is at least −0.25.
4. The graph keeps reachable and returnable actions. Safety exploration samples uncertainty; ES² can switch to reward optimization.
5. Both phases continue learning online. True fields score the chosen action after execution; they never steer a learner.

## Open the saved evidence

- [Final comparison dashboard](../results/final/index.html): explain the six maps, two observation seeds, noise factors, fixed budget, and all failures.
- [Baseline replay](../results/final/runs/map_100/obs_3000/h_160/source/sno_es2/noise_1p0/index.html).
- [Revalidated replay](../results/final/runs/map_100/obs_3000/h_160/revalidated/sno_es2/noise_1p0/index.html).

These are the first map/observation at matched noise, chosen by a fixed demonstration rule rather than by favorable outcomes. Blue/yellow regions represent model certificates and optimistic possibilities; a cell’s height/color is not a cell-level safety label. Safety belongs to directed actions.

Move the slider and show the chosen action, executable directions, exploration/reward phase, reward, incorrect certificates, and unsafe-action count. The right-hand true field is evaluator information. Clarify that the browser replay is the project’s saved visualization, not the legacy native MuJoCo renderer.

## Explain the diagnostic mechanism

A certificate can be incorrectly admitted under practical confidence bands. The original profile retains it even if later evidence changes its lower bound. Revalidation requires a passing current bound or a supplied initial certificate. Losing the current reachable/returnable region halts motion and counts as failure; it does not fabricate a return path.

## Show controlled results

Use the selected-confidence retained ES² control to distinguish confidence tuning from certificate handling. Show paired map differences, unsafe actions, completion failures, exploration effort, and decision/inference cost. Read actual numbers from the report; do not infer general guarantees from zero observed violations.

## Reproduce and close

```bash
python3 -m sno_mdp verify --study results/final
```

Keep the presentation PDF as an offline backup. Each member should explain their actual contribution, one equation, one controlled comparison, and one limitation. See [team review](team_review.md) for human checks still requiring confirmation.

## Questions to be ready for

- **Why can safe RL record unsafe actions?** The theoretical assumptions and uniform confidence event are not established by fixed practical multipliers. One false certificate can persist in the released-code profile.
- **Why does a larger multiplier delay basic stopping?** Its interval width is `2 × multiplier × posterior STD`; the same accuracy tolerance therefore requires more observations.
- **Does the rover use the true map?** Only the evaluator/oracle do. The learner uses noisy observations, dynamics, and the supplied initial certificates.
- **What does revalidation guarantee?** It enforces current model bounds and graph checks. Those bounds can be wrong, so finite zero-violation evidence is not a safety proof.
- **Why include a c=4 retained control?** It separates confidence tuning from the change in certificate handling.
- **What is new here?** A documented empirical robustness study and clearly labeled project extension; algorithmic novelty and patentability are not claimed.
