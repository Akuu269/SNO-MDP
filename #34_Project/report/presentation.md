# Presentation and speaker notes

## Slide 1: SNO-MDP: safe exploration and reward

- Four-person course project; source-audited headless simulation.
- Question: how do noise assumptions and certificate retention affect behavior?
- Wachi and Sui, ICML 2020; current-LCB revalidation is our project extension.

## Slide 2: What the rover knows

- Known dynamics and supplied initial safe actions.
- Unknown altitude and reward learned from noisy local observations.
- True fields belong to the evaluator/oracle only.
- Safe action: altitude(source) minus altitude(successor) >= -0.25.

## Slide 3: GP inference and SNO-MDP

- Separate altitude and reward RBF GPs update online.
- Difference variance: Var(z_s)+Var(z_t)-2 Cov(z_s,z_t).
- Explore uncertain expandable actions through certified travel.
- ES² checks auxiliary-policy successors from every executable state.

## Slide 4: Diagnostic finding and extension

- Development violations relied on old certificates after current bounds failed.
- Baseline retains certificates, following the released source profile.
- Extension rechecks current LCBs and rebuilds reachable/returnable paths.
- Region-loss halts count as failures; no truth-based action guard.

## Slide 5: Development and frozen evaluation

- Selected multiplier 4 from calibration data only.
- Stopping tested at 120/240 actions; final horizon fixed at 160.
- 6 fresh maps; two observation seeds; three noise assumptions.
- Equal budgets, all failures retained, paired comparisons and map-level variability.

## Slide 6: Held-out reward curves


Explain axes, map/seed counts, failures, and variability in returns.png.

## Slide 7: Executable certificates and errors


Explain axes, map/seed counts, failures, and variability in certificate_errors.png.

## Slide 8: Safety, completion, exploration and computation


Explain axes, map/seed counts, failures, and variability in outcomes.png.

## Slide 9: Paired return differences


Explain axes, map/seed counts, failures, and variability in paired_returns.png.

## Slide 10: Findings and practical limits

- Final inventory: 192 cases; 0 failures; 384 unsafe actions.
- Read per-noise results to assess reward/safety/completion tradeoffs.
- Finite experiments do not prove safety or reproduce exact published curves.
- Source, protocol, raw records, tests, replays and reproduction commands supplied.
