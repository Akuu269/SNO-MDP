# Method and source audit

## 1. Sources and claim

We implement a **headless, source-audited port of the released GP-Safety-Gym synthetic benchmark**. It executes the discrete model used by the authors' drivers without their legacy renderer. It is not a claim to have run native `Engine_GP`, reproduce the published curves, or establish the paper's guarantees.

- [Paper and appendices](../SNO_MDP_Wachi_Sui_2020_arxiv_v1.pdf): Sections 2–4 for the method, Section 5 for assumptions/guarantees, Section 6 for experiments.
- [SNO-MDP source](https://github.com/akifumi-wachi-4/safe_near_optimal_mdp), pinned at `e88370ef87a4f8eff8b731339a7721508d1bd3b0`.
- [SafeMDP dependency](https://github.com/befelix/SafeMDP), pinned at `725eaaeb445f88cffa0c695d59be0a30e84cd2c3`.
- [Machine-readable provenance](../third_party/upstream.json), [reviewed reference archive](../third_party/reference_sources.zip), and preserved MIT notices in `third_party/`.

The code and the paper do not make identical operational choices. This document separates them so the report can make an accurate claim.

## 2. Environment and information

A state is `(x_index, y_index)`. Source action order is stay, up (+y), right (+x), down (-y), left (-x). Out-of-grid actions are unavailable. Coordinates supplied to GPs equal indices multiplied by grid spacing.

The released fixture has 20 × 20 states, spacing 0.5, reward and altitude fields. We preserve both arrays exactly, including zero rewards and rewards greater than one. The stored field named `safety` upstream is altitude, not a per-cell safety probability.

The source's directed transition safety feature is

$$
g(s,a)=z(s)-z(f(s,a)),\qquad g(s,a)\ge h,\quad h=-0.25.
$$

Stay is always safe. A cell can have safe and unsafe outgoing directions. This is different from interpreting negative height as an unsafe cell or applying an absolute slope limit to a physical rover.

The initial certificate is a star around the declared start. A neighboring pair is initially supplied only when

$$
-|z(s)-z(s')|/\mathrm{spacing}\ge h,
$$

with at least one connection in each coordinate direction. This reproduces the reviewed `compute_S_hat0` rule; its initialization rule is deliberately stricter than the later directed-difference feature. Our released-map run centers this star on saved `start_pos=148` for reproducibility; the released driver independently chooses a random initial star.

The learner sees only dynamics, prior settings, initial certificates, and noisy altitude/reward readings at its current state and valid neighbors. The evaluator and oracle can inspect true fields. Neither can select learner targets, repair confidence problems, or veto actions using truth. Noise streams are paired by map, cell, sensor type, and observation occurrence.

## 3. GP inference and transition uncertainty

Separate exact zero-mean RBF GPs model altitude and reward. Nominal source settings are length scale 2, prior variance 1, altitude/reward noise STD 0.001, safety multiplier 2, and reward multiplier 3. Noise is squared when added as covariance variance. We report latent posterior STD, not the noise of a future measurement.

For altitude posterior mean `mu` and covariance `Sigma`, a directed difference has

$$
\mu_g(s,a)=\mu(s)-\mu(s'),\qquad
\sigma_g^2(s,a)=\Sigma_{ss}+\Sigma_{s's'}-2\Sigma_{ss'},\quad s'=f(s,a).
$$

Ignoring covariance would give incorrect uncertainty. The test suite compares this calculation against an independent full-covariance solution.

Current edge bounds are `mu_g ± 2 sigma_g`. Following the released `GridWorld`, current bounds are recomputed and accepted safe actions are retained monotonically. This is not the historical state-function interval intersection described in the paper text. An initial certificate contradicted by an upper bound raises an explicit failure.

A fixed multiplier is not a uniform confidence guarantee across all states, actions, and decisions. Retaining an erroneous certificate can matter later; the evaluator records false action certificates even when the rover never executes them.

## 4. Safe exploration and comparators

Accepted directed actions form a graph. Retain states reachable from the initial region and able to return to it; executable actions connect those states through accepted edges. The tests include one-way traps, disconnected islands, and excluded directions.

Expanders are executable state-action pairs whose upper bounds could certify unknown pairs for the same action. The released default `L=0` is used only for this expander score. We do not claim it is a valid global Lipschitz bound for terrain. Select the widest expander; if none exists, select the widest executable pair, as the source does. Travel through certified actions to its source state, observing along the route. Sampling at the target can use a stay action because the sensor also sees neighbors.

- `safemdp`: continues uncertainty sampling throughout the common total horizon; no reward-seeking phase.
- `sno_basic`: switches when the maximum width over executable pairs is at most the declared tolerance, nominally 0.005.
- `sno_es2`: can also switch when the auxiliary successor test passes.
- `oracle`: true-field planning on the reachable/returnable true-safe action graph, with a documented information advantage.

All travel and stay actions consume budget. Exhausting the horizon is distinct from convergence or a model error.

## 5. Reward optimization and ES²

The optimistic reward is

$$
U(s)=\mu_r(s)+c_r\sigma_r(s).
$$

Value iteration uses only admissible successors and certified directed actions:

$$
V(s)=\max_{a\in A_{\rm cert}(s)}[U(f(s,a))+\gamma V(f(s,a))].
$$

We preserve the released reward-bonus cap of 100. GPs and certificates update from authorized noisy observations in both phases; the policy is recomputed. The previous prototype froze them for the whole reward phase. The source audit showed that the authors' driver continues updating, so that prototype behavior was retired. Beliefs remain fixed within each individual Bellman solve.

ES² plans over optimistic executable actions. Its immediate reward is the lower reward bound at the successor of a certified action and the upper reward bound at the successor of an optimistic-only action. For its optimal policy `pi_y`, form

$$
Y=\{f(s,\pi_y(s)):s\text{ is currently executable}\}.
$$

Stop safety exploration when every such successor remains in the executable state set. Save the policy, successor indices, outside-directed source cells, and solver residual. The test considers every executable source, not just the rover's current state.

After a passing test, our released-code profile optimizes on certified actions. We do not impose the paper's separate fixed `Y` reward domain: in a directed-edge formulation, state membership alone does not ensure each connecting action is certified. This difference is explicit and precludes claiming the paper's exact Y-domain theorem.

## 6. Source profile and deviations

| Item | Project choice / limitation |
|---|---|
| Maps | Exact released fixture or RBF-GP draws with source covariance; generated random streams differ from legacy NumPy's streams |
| Initialization | Deterministic valid star at the released start; generated maps use a seeded validity search; source bootstrap observations are not copied |
| Boundaries | Invalid actions excluded, instead of a heavily penalized virtual sink |
| Safety uncertainty | Latent difference covariance; upstream GPy's default predictive-likelihood variance is not assumed identical |
| Confidence history | Current bounds plus monotonically retained accepted edges, following released code; not paper interval intersection |
| Observations | Explicit controlled Gaussian noise and observations during travel; the source drivers insert many true values directly |
| Reward phase | Online updates as in the released driver; physical actions use hard certificates instead of probability-weighted virtual transitions |
| Stopping | Common finite physical-action horizon; no artificial forced reward-phase cap |
| Rendering | Self-contained replay and exported plots; legacy Engine_GP not executed |
| Guarantees | Source-audited empirical port, not an exact numerical/theorem reproduction |

## 7. Read the code in this order

1. [config.py](../sno_mdp/config.py): public settings.
2. [gp_safety_gym.py](../sno_mdp/gp_safety_gym.py) and [environment.py](../sno_mdp/environment.py): source fixture, dynamics, authorized sensing, evaluator.
3. [gp.py](../sno_mdp/gp.py): posterior and difference covariance.
4. [graph.py](../sno_mdp/graph.py) and [transition_sets.py](../sno_mdp/transition_sets.py): executable action certificates.
5. [agent.py](../sno_mdp/agent.py): observation-driven target and path selection.
6. [reward_planning.py](../sno_mdp/reward_planning.py): reward and auxiliary planning.
7. [experiment.py](../sno_mdp/experiment.py): phases, execution, diagnostics, and outputs.
8. [benchmark.py](../sno_mdp/benchmark.py): matched comparisons and aggregation.
9. [diagnostics.py](../sno_mdp/diagnostics.py): evaluator-only confidence coverage and certificate provenance.

For preparation, revise Sutton–Barto Sections 3.1, 3.3, 3.5–3.6, 4.1 and 4.4, and Rasmussen–Williams Section 2.2 ([GP regression chapter](https://gaussianprocess.org/gpml/chapters/RW2.pdf)). The original paper remains the main reading source.

## 8. Safety diagnostics

`diagnose` reconstructs the joint difference posterior from saved noisy observations. It verifies the map/configuration manifest, the recorded GP/certificate source files, reconstructed masks, altitude predictions, and executed outcomes. Original run files remain unchanged; separate analysis manifests record input hashes and the current analysis source.

Coverage is measured over (a) all valid moving actions and (b) executable moving actions. Stay and invalid boundary actions are excluded. Count a miss below the LCB when `g < mu_g − c sigma_g`; an upper miss has `g > mu_g + c sigma_g`. Empty subsets have missing coverage, not 100%. A failed GP update receives no fabricated fresh posterior; its earlier executions remain included.

The audit records the first acceptance of each directed certificate, its bounds at acceptance, when it first becomes executable, and its first execution. Every physical action has a pre-action trace. Unsafe executions are distinguished by whether the current LCB passes or an older certificate is retained despite a failing current LCB. The latter follows the unchanged source-profile union rule; diagnostics do not revoke edges or insert a truth-based action guard.

Multiplier comparisons hold the saved trajectory and posteriors fixed. They measure counterfactual interval coverage and instantaneous LCB passes, **not** reward or safety performance of rerun policies. Correlated edges and repeated frames are not independent trials, and high average coverage is not simultaneous coverage or a safety guarantee. Selecting settings and testing certificate-retention changes require separate development experiments before final evaluation.

## 9. Current-LCB project extension

`certificate_mode="retained"` preserves the source-profile baseline. `current_lcb` keeps historical acceptance for provenance but permits execution only through initial certificates or actions whose current LCB passes. Reachability and returnability are recomputed after every observation. Withdrawn actions can be recertified by later evidence; expanders include currently unsupported directions.

If the rover is outside the recomputed executable region, it halts with `certificate_region_lost`. This is an explicit failed run, not successful safe completion. The executed return prefix remains in all comparisons; the unused budget earns zero extra reward. No true-map action block, automatic reset, or fabricated return route is used. The extension is not claimed as the original paper’s algorithm or guarantee.

Repeated Gaussian readings at an identical coordinate are compressed to their count and mean. The diagonal observation variance becomes `(noise_variance + per_reading_jitter)/count`, giving the same latent Gaussian conditioning as all individual readings. Independent full-matrix tests check this identity. This implementation revision can differ numerically at floating-point ties from historical pilot traces; final cases share one frozen revision.
