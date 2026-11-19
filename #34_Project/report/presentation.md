# Final professor presentation and speaker notes

Members 1 and 2 retain slides 1–4 and 5–8. Member 3 presents experimental design (9), results (10–13), saved dashboard evidence (14), and Thank You (15). Srinivas then presents the live dashboard.
15 slides total. Superseded results and Q&A backups have been removed. Member 3 follows the five-minute rehearsal, including design and dashboard evidence.
Presenter and timing information is in these notes, not on the audience slides. Graphs use existing frozen results.
See [claim corrections and sources](presentation_claim_review.md) and the [integrated Member 3 rehearsal](member3_results_rehearsal.md). P-ES2 remains literature background; Mars results are not claimed as our experiments.

[PowerPoint](presentation.pptx) · [PDF](presentation.pdf) · [Browser slides](presentation.html) · [Team timing and demo plan](team_presentation_plan.md)

## Slide 1 — SNO-MDP: learning reward while remaining safe

**Member 1 · slide allocation 40s.**



The paper is Safe Reinforcement Learning in Constrained Markov Decision Processes by Akifumi Wachi and Yanan Sui, ICML 2020. It studies unknown reward and safety functions, with safety required during learning, not just after convergence. Its guarantees are high-probability statements under explicit assumptions, not unconditional zero-risk claims. Our project is a source-audited empirical reproduction with a certificate-revalidation extension. We separate literature, implementation, and measured results.

## Slide 2 — The challenge: learn two unknown functions safely

**Member 1 · slide allocation 60s.**



In the paper, reward r of a state and safety g of a state are unknown, while transition dynamics are known. The state constraint must hold at each visited time, unlike merely bounding an expected accumulated cost. Our implemented discrete benchmark instead checks a directed altitude difference. A move uphill can fail while the reverse move passes. The learner observes noisy local values and supplied initial certificates; complete truth is evaluator information. The final project uses twenty-by-twenty synthetic fields. This six-by-six grid is an illustration, not a Mars experiment.

## Slide 3 — What makes the paper’s guarantees possible?

**Member 1 · slide allocation 45s.**



Safe learning from nothing is not generally possible. The paper assumes a nonempty known-safe seed set, bounded RKHS norms for reward and safety, Lipschitz regularity of safety, and controlled observation noise. Its confidence schedules support a simultaneous high-probability confidence event. On that event, the safe-region construction supports safety at every step and the stated near-optimality results. Our fixed practical multipliers do not verify those hypotheses, and finite zero-violation runs cannot replace that proof.

## Slide 4 — From safe exploration to near-optimal reward

**Member 1 · slide allocation 55s.**



SafeMDP has a purposeful information-gathering objective: it expands the safely certifiable region without optimizing reward. Calling it aimless is inaccurate. SNO-MDP separates safety exploration from optimistic reward planning; ES2 can stop safety exploration early while retaining the paper’s conditional near-optimality result. P-ES2 is the paper’s probability-weighted practical alternative. It can converge faster empirically, but the paper does not supply the same reward near-optimality guarantee for it. Our frozen project implements basic SNO-MDP, ES2, and current-LCB revalidation, not P-ES2 or Mars evaluation. Member 2 explains the mechanics.

## Slide 5 — Two GPs: uncertainty for safety and optimism for reward

**Member 2 · slide allocation 55s.**



Separate GPs model reward and altitude. In our port, safety concerns a correlated difference between endpoint altitudes, so its variance includes the covariance term. The code uses direct standard-deviation multipliers c_s and c_r; the paper’s intervals use square-root beta for safety and square-root alpha for reward. These numerical settings are not interchangeable. The reward upper bound uses c_r equal to three. Actual sensor noise defaults to point zero zero one, and a separate factor controls assumed altitude noise. These values do not by themselves establish the paper’s confidence event.

## Slide 6 — Pessimistic, optimistic, and expansion candidates

**Member 2 · slide allocation 50s.**



The pessimistic region contains model-certified states with reachability and returnability. The optimistic region contains possibilities consistent with upper bounds and graph constraints; it is not permission to move. Expanders are permitted candidates whose optimistic information could certify more territory. They are not merely every boundary cell. The paper defines state regions with Lipschitz propagation. Our source port uses directed state-action certificates and an expander score. It selects large uncertainty widths and travels through executable actions. The project’s L equal to zero is an expander setting, not a valid global flat-terrain assumption.

## Slide 7 — Our extension: recheck historical safety permissions

**Member 2 · slide allocation 55s.**



Our baseline recomputes bounds and retains previously accepted actions, following the released source profile. This differs from the paper’s historical interval intersection. Our extension requires current support or an initial certificate and rebuilds the reachable and returnable graph. If the current state loses this region, the run halts and counts as a failure. The saved step-ninety-six up action has LCB minus zero point two eight zero six, below minus zero point two five, yet retained permission allows it. Member 4 shows the violation. Both GPs keep updating in our reward phase; our implementation does not freeze learning.

## Slide 8 — ES2 and P-ES2: different auxiliary planning models

**Member 2 · slide allocation 40s.**



ES2 checks an auxiliary optimistic MDP and stops when all relevant policy successors from executable sources remain inside the pessimistic region. Our project implements that source-profile test. The paper’s P-ES2 weights transitions using GP safety probabilities and a virtual zero-reward absorbing state. It is a practical alternative without the same near-optimality proof. Early stopping changes when exploration ends, not permission to take unsafe moves. Our project does not implement P-ES2. Member 3 now shows our actual empirical results.

## Slide 9 — Experimental design: freeze first, evaluate fresh cases

**Member 3 · team slide 11 · allocation 40 seconds.**

*Do not speak: Point to six maps, two repeats, three noise factors, and the common 160-action budget.*

Before showing the graphs, here is our evaluation setup. We froze the settings before testing six new 20-by-20 maps, with two sensor-noise repeats and three assumed-noise settings.

We completed 192 runs, including reference runs. Every run used 160 actions, including travel and exploration. This makes the comparison fair. To test revalidation separately from confidence tuning, we have compare methods with the same safety confidence multiplier.

## Slide 10 — Reward across the same action budget

**Member 3 · team slide 12 · allocation 45 seconds.**

*Do not speak: Point to the axes, then compare the grey and teal curves. Identify the dashed reference.*

The horizontal axis shows actions; the vertical axis shows discounted return, meaning collected reward with later rewards weighted less.

Compare grey SNO-MDP (Safe Near-Optimal Markov Decision Process, the basic method) with teal ES² (Early Stopping of Exploration of Safety).

With the confidence multiplier at two, early stopping earns greater than the basic method. The dashed oracle knows the true environment and is a privileged reference. Higher reward alone does not establish safer performance.

## Slide 11 — Safety and reward must be read together

**Member 3 · team slide 13 · allocation 75 seconds.**

*Do not speak: Explain the error bars, define the two permission rules, then compare the two conditions at four.*

Here in the bar-chart, read reward and safety together. The error bars show SD (standard deviation: variation across six map averages), not confidence intervals.

The label c (safety confidence multiplier) controls the width of the model's safety margin. Four requires stronger evidence than two.

Retained means an earlier learned safety permission can remain active even when the current bound fails.
Revalidated means learned permissions are checked against current bounds; unsupported permissions are excluded before planning.

At multiplier two, the average is 2.5 unsafe actions per run. Both versions at four record zero here, with lower mean reward. Comparing these two at the same multiplier isolates the permission rule. Since both have zero violations here, this slide alone does not show an additional safety benefit from revalidation.

## Slide 12 — Robustness when sensor noise is misspecified

**Member 3 · team slide 14 · allocation 50 seconds.**

*Do not speak: Explain the three assumptions, then point to the side-panel totals: 98, 2, and 0.*

Here the actual altitude-sensor noise stays fixed; we change only the assumed noise standard deviation.

A factor of 0.5 underestimates noise, one matches it, and two overestimates it. The graph shows unsafe actions per run; the side panel gives totals across 36 runs per method.

Retained at two records 98 unsafe actions, retained at four records two, and revalidated at four records zero. The two retained-at-four violations occur in one underestimated-noise run. This is limited experimental evidence, not a universal guarantee.

## Slide 13 — ES2 shifts actions from exploration to reward

**Member 3 · team slide 15 · allocation 40 seconds.**

*Do not speak: Compare the first two bars and point to the 42.58-action difference.*

This graph shows how early stopping changes the fixed budget. Compare the first two bars, both using the confidence multiplier of two.

The basic method spends all 160 actions exploring safety. Early stopping averages 117.42, reallocating 42.58 actions—26.6 percent of the budget—to reward optimization.

Every run still uses 160 actions and collects reward during exploration. This shows a change in priorities, not faster computation.

## Slide 14 — Saved dashboard evidence

**Member 3 · team slide 16 · allocation 40 seconds.**

*Do not speak: Identify the saved replay. Point to the failed current bound and the violation count at step 97.*

This is saved dashboard evidence. Before the move at step 96, the LCB (lower confidence bound: the model's conservative safety estimate) is about minus 0.281, below the threshold of minus 0.25.

The current bound therefore cannot certify the move, but an earlier retained permission allows it. At step 97, one actual violation is recorded.

This single trajectory illustrates the mechanism; the earlier graphs summarize multiple runs. Srinivas will show the interactive replay.

## Slide 15 — Thank You

**Member 3 · team slide 17 · allocation 10 seconds.**

*Do not speak: Hand over to Srinivas without introducing another result.*

Thank you. Srinivas will now demonstrate the dashboard and compare individual experiments.
