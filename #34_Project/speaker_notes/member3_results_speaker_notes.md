# Member : Akash — speaker notes for slides 11–15


## Slide 11 — Experimental design: freeze first, evaluate fresh cases

Don't speak : *0:00–0:40. Point to six maps, two repeats, three noise factors, and the common 160-action budget.*

Before showing the graphs, here is our evaluation setup. We froze the settings before testing six new 20-by-20 maps, with two sensor-noise repeats and three assumed-noise settings.

We completed 192 runs, including reference runs. Every run used 160 actions, including travel and exploration. This makes the comparison fair. To test revalidation separately from confidence tuning, we have compare methods with the same safety confidence multiplier.

## Slide 12 — Reward across the same action budget

Don't speak : *0:40–1:25. Point to the axes, then compare the grey and teal curves. Identify the dashed reference.*

In the graph-chart, The horizontal axis shows the number of actions taken. The vertical axis shows discounted return—the total reward accumulated so far, where a reward earned earlier counts more than the same reward earned later.

Compare grey SNO-MDP (Safe Near-Optimal Markov Decision Process, the basic method) with teal ES² (Early Stopping of Exploration of Safety).

With the confidence multiplier at c = 2, early stopping earns greater than the basic method. The dashed oracle is a true environment and is a main reference. Higher reward alone does not establish safer performance.

## Slide 13 — Safety and reward must be read together

Don't speak : *1:25–2:40. This Explain the error bars, define the two permission rules, then compare the two conditions at four.*

Here in the bar-chart, showing reward and safety together. The error bars show SD (standard deviation: variation across six map averages), it's not confidence intervals.

The label c (safety confidence multiplier) controls the width of the model's safety margin. c=4 requires stronger evidence than c = 2.

**Retained means here** : an earlier learned safety permission can remain active even when the current bound fails. 
and **Revalidated means here**  : learned permissions are checked against current bounds; unsupported permissions are excluded before planning.

so like, left side bar-chart is Avg return with confidence multiplier. means “How much reward did the rover collect?” here Higher is better. The black error bars show SD (standard deviation)—how much the average returns differed between maps. Longer error bars mean more variation.
Right side bar chart is Avg unsafe actions per run with confidence multiplier. means “How many unsafe moves did the rover actually execute?” here Lower is better.

At sefety multiplier c = 2, the average is 2.5 unsafe actions per run. Both retained & revalidate  at c = 4 record zero here, with lower mean reward. Comparing these two at the same multiplier isolates the permission rule.

Retained c = 2 to retained c = 4: changing only the confidence setting reduced observed violations, while also reducing reward.
Retained c = 4 to Revalidated c = 4: keeping confidence fixed and examine the permission rule. Both recorded zero violations on this slide, so these results alone do not show an additional safety benefit from revalidation.

so here it's confirming clear : Higher reward alone does not mean better safe reinforcement learning. We must examine reward and unsafe actions together.

## Slide 14 — Robustness when sensor noise is misspecified

Don't speak : *2:40–3:30. Explain the three assumptions, then point to the side-panel totals: 98, 2, and 0.*

In this graph "sensor noise is misspecified" shows the actual altitude-sensor noise stays fixed; we change only the assumed noise standard deviation.

A factor of 0.5 underestimates noise, one matches it, and two overestimates it. The graph shows unsafe actions per run; the side panel gives totals across 36 runs per method.

Retained at c = 2 records 98 unsafe actions, retained at c = 4 records 2 only unsafe actions, and revalidated at c = 4 records zero.

## Slide 15 — ES2 shifts actions from exploration to reward

Don't speak : *3:30–4:10. Compare the first two bars and point to the 42.58-action difference.*

In this graph bar-chart, shows how early stopping changes the fixed budget. Compare the first two bars, both using the confidence multiplier of 2 ( c = 2)

The basic method spends all 160 actions exploring safety. Early stopping averages 117.42, reallocating 42.58 actions and 26.6 % of the budget—to reward optimization.

Every run still uses 160 actions and collects reward during exploration. This shows a change in priorities, not faster computation.


Thank you. Srinivas will now demonstrate the dashboard.

---

## Preparation only — do not add to the five-minute speech

**Retained**: “Keep an earlier approval that a movement is safe, even if newer observations no longer support that approval.”
**Revalidated**: “Check earlier learned safety approvals again using the latest observations. Exclude unsupported approvals before planning the next move.”

main difference : Retained keeps earlier approvals with data; Revalidated checks them again.

For example, an action could pass yesterday's bound and fail today's updated bound. Retained can preserve that learned permission. Revalidated excludes it until it is supported again, unless it belongs to the supplied initial certificates.

| Label on a slide | Full name and meaning |
|---|---|
| SNO-MDP | Safe Near-Optimal Markov Decision Process: the base exploration-then-reward method |
| ES² / ES2 | Early Stopping of Exploration of Safety: stops the safety-exploration phase when its auxiliary test passes |
| SD | Standard deviation across six map averages: the plotted return variability |
| STD | Standard deviation of sensor noise: the quantity used in the assumed/actual noise ratio |
| c | Safety confidence multiplier: scales uncertainty in the safety bound; this is not the reward multiplier |
| γ | Gamma, the discount factor: weights later rewards; 0.99 in this study |
| GP | Gaussian process: the model used to predict a function and its uncertainty |
| LCB | Lower confidence bound: the conservative estimate used to check an action's safety margin |
| UCB | Upper confidence bound: predicted reward plus an uncertainty bonus |
| Map 100 / obs 3000 | Map seed 100 and observation seed 3000; the demo is 20×20 with 160 actions |

At matched noise, 
ES² mean returns are 177.56 / 157.18 / 155.31 
and mean unsafe-action counts are 2.5 / 0 / 0 for retained at c = 2 / retained at c = 4 / revalidated at c = 4. Across all noise factors, unsafe-action totals are 98 / 2 / 0, with 36 runs per condition. The two violations in the retained at c= 4 condition occur in one run.

## Questions the professor may ask

**Preparation only — these questions are not part of the five-minute speech.**

### Slide 11 — Experimental design

1. What does “freeze the settings before evaluation” mean, and why is it important?

   **Answer:** We selected the settings using development data and kept them unchanged during final testing to avoid tuning them to the final results.

2. How did you obtain 192 runs? How many runs belong to each method and noise setting, and how are the oracle references counted?

   **Answer:** 192 = five learner conditions × six maps × two observation seeds × three noise settings + twelve oracle runs, with twelve runs per learner condition per noise setting.

   **Five learner condition : ( SafeMDP — explores to learn safety , Basic SNO-MDP — safety exploration followed by reward optimization , SNO-MDP with early stopping ES² (c=2) and c = 4 , SNO-MDP with early stopping ES²( c = 4 -- revalidate) )**

   **6 maps** : Six different simulated environments, each containing a 20×20 grid with different terrain and reward values

3. Why use six maps with two observation seeds each? Are twelve runs equivalent to twelve independent maps?

   **Answer:** The maps test terrain variation and the observation seeds test sensor-noise variation, so there are six independent map units, each repeated twice.

4. Where do the terrain and reward values come from? Are these synthetic maps or real Mars data, and what information can the learner actually observe?

   **Answer:** These are synthetic terrain and reward fields generated using Gaussian processes; the learner receives noisy local readings instead of the full true maps.


### Slide 12 — Reward comparison

5. What is discounted return, and how does it differ from cumulative reward? Are the plotted values raw returns or normalized scores?

   **Answer:** We plot raw discounted return, where the same reward counts more when earned earlier, while cumulative reward adds rewards without discounting.

6. What information does the oracle have that the learner does not? Is it a fair competitor or a privileged reference?

   **Answer:** The oracle knows the true reward and safety fields, making it a privileged planning reference with more information than the learner.

7. What does SD (standard deviation) across map averages represent? Can overlapping error bars establish that two methods perform equally well?

   **Answer:** SD (standard deviation) shows variation between the six map averages; overlapping error bars alone do not prove equal performance.

8. Do safety exploration, travel, and staying in place consume the action budget and contribute reward?

   **Answer:** Yes; exploration, travel, and staying all consume the action budget and contribute the reward at the resulting state.


### Slide 13 — Safety and reward together

9. What exactly is retained: observations, model predictions, or permissions to execute actions? What does revalidation check again?

   **Answer:** Retained keeps earlier learned movement permissions; Revalidated rechecks their current safety bounds, while preserving the supplied initial safe permissions.

10. What does c (safety confidence multiplier) control? Why compare Retained at four against Revalidated at four when evaluating the extension?

   **Answer:** The multiplier scales uncertainty in the safety bound; using four in both methods separates permission rechecking from confidence tuning.

11. What counts as an unsafe action in this environment? How can the average be 2.5 when individual actions are counted in whole numbers?

   **Answer:** An unsafe action has altitude(current) − altitude(next) < −0.25; the average of 2.5 comes from thirty unsafe actions across twelve runs.

12. If both methods at a multiplier of four record zero violations here, what additional benefit of revalidation has this particular experiment demonstrated?

   **Answer:** This matched-noise subset shows no additional violation reduction; the difference appears under underestimated noise, where the retained control has two violations and revalidation has zero.

13. Why does the more conservative setting earn less reward? How would you choose between higher reward and fewer unsafe actions?

   **Answer:** Conservative bounds can restrict rewarding routes; we prioritize the safety requirement and then compare reward among settings that satisfy it in the evaluation.

14. When revalidation withdraws a permission, how are reachability and the ability to return to the initial safe region checked again?

   **Answer:** We recompute reachability and permitted return paths, and stop the run if the rover is outside the remaining executable region.


### Slide 14 — Noise robustness

15. What do noise factors 0.5, one, and two mean? Are you changing the actual sensor noise or only the model's assumption about its standard deviation?

   **Answer:** Actual noise stays fixed, while the assumed altitude-noise standard deviation is set to half, equal to, or twice its actual value.

16. How were the totals 98, two, and zero calculated? Did the two violations in the retained-at-four condition occur in one run or two different runs?

   **Answer:** Each total sums thirty-six runs for one condition; both violations with retained permissions at confidence multiplier four occurred in one underestimated-noise run.

17. Does observing zero violations in these runs prove that the method is always safe? How much can you conclude from six maps?

   **Answer:** No; zero observed violations is limited evidence from these tested cases and does not establish safety in every environment.

18. What could happen if the Gaussian process model, its kernel, or its assumed measurement noise is incorrect?

   **Answer:** The model can become too confident and approve unsafe moves, so rechecking alone cannot correct an incorrect safety model.


### Slide 15 — Early-stopping benefit

19. What condition makes ES² (Early Stopping of Exploration of Safety) stop safety exploration?

   **Answer:** It stops when every next state chosen by its optimistic auxiliary policy from the executable region is also inside that region.

20. How do the mean exploration counts of 160 and 117.42 produce the 42.58-action difference and 26.6 percent figure?

   **Answer:** 160 − 117.42 = 42.58 actions, and 42.58 ÷ 160 × 100 ≈ 26.6 percent of the fixed budget.

21. Does reallocating exploration actions mean the algorithm executes fewer total actions or requires less computation time?

   **Answer:** No; every final run still executes 160 actions, so the result shows a different budget allocation, without establishing a computation-time saving.

22. Why does the basic method spend the whole 160-action budget exploring? Would a longer horizon change the comparison?

   **Answer:** Here, the basic method never meets its exploration accuracy threshold within 160 actions; a longer horizon could change the comparison.


### Overall contribution and limitations

23. Which methods did your team actually implement and evaluate? Which results belong to the published paper, including its practical early-stopping variant and Mars experiments?

   **Answer:** We evaluated SafeMDP, basic SNO-MDP, three ES² settings, and the oracle; P-ES² and Mars experiment results belong to the published paper.

24. The paper gives safety guarantees under assumptions. Why can your implemented source profile still record violations, and what guarantee can you claim for your extension?

   **Answer:** Our practical fixed confidence bounds do not establish the paper’s assumptions, so we claim measured outcomes on tested cases, without a general safety guarantee.

25. What is the main limitation of your study, and which additional experiment would best strengthen the result or support transfer to a real rover?

   **Answer:** Our evidence covers only six synthetic maps; we would test more maps and incorrect noise or kernel assumptions before validation on realistic terrain.

