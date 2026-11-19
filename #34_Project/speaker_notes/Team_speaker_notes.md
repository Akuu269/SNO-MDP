## Speaker Notes — Slides 1 to 5
Safe RL in Constrained MDPs with Unknown Safety & Reward (Wachi & Sui)

## Slide 1 — Introduction

Good afternoon, Professor and team. We are a team of four members: myself, Srinivas, Akash and Sahana.
We have experimented with and analysed Safe RL in Constrained MDPs with Unknown Safety and Unknown Reward (Wachi & Sui), comparing it with previously existing approaches: SafeMDP, SafeMDP Exp-Opt, and Safe – Reward Known (Oracle).
Using ES² and P-ES², we extended the algorithm with safety re-checks, implementing graphs and interactive dashboards.
## Slide 2 — Main Motto

Enable an RL agent to maximize cumulative reward in an unknown environment with strictly guaranteed safety (zero constraint violations) at every single step of training, rather than only after convergence.
Constrained MDP & Baseline Comparison
Safety-constrained MDP tuple:
$$M = (S, A, f, r, g, \gamma)$$
All models give guaranteed safety, but the reward is either none, heuristic, or optimal. In ours, it is near-optimal, with our ES² speed-up and stepwise decoupling.

## Slide 3 — Key Assumptions

The paper relies on four key assumptions:
Known safe starting zone (S₀) — provides a home base to launch exploration from, and to retreat back to if nearby paths look risky.
Smoothness & continuity (Lipschitz) — $|g(s) - g(s')| \le L \cdot d(s, s')$
Gaussian Process (GP) — both safety and reward are smooth, with a GP similarity kernel.
Known dynamics & bounded noise — state transitions $s_{t+1} = f(s_t, a_t)$ are deterministic and known, with bounded, zero-mean noise.

## Slide 4 — Solution

SNO-MDP · ES² · P-ES²
SNO-MDP: Explore boundary "expanders" until the entire reachable safe region is certified.
ES²: Stop expanding safety early if extra exploration cannot improve the optimal reward.
P-ES²: Replaces the overly optimistic assumption with realistic risk probabilities.

## Slide 5 — Dual Gaussian & Confidence Bounds

By using confidence multipliers α and β, we control the safety and reward of the agent, i.e.:
c_s : safety multiplier
c_r : reward multiplier
The code uses standard-deviation multipliers c_s and c_r, since these values do not by themselves establish the paper's confidence [guarantees].

## Slide 6 — Pessimistic/Optimistic Safe Spaces & Expander Selection (~45s)

Using the lower and upper confidence bounds from our Safety GP, the algorithm divides the entire grid S into three regions:
First, the inner region X_t- is the pessimistic safe space. A state is only added here if its safety lower bound minus the slope margin L .d(s, s') stays above the threshold h, and there is a verified path to reach it and safely return back to our starting zone. During Step 1, the agent is strictly forbidden from leaving X_t-.
Second, the yellow ring X_t+ \X_t- is the optimistic frontier—states that might turn out to be safe based on their upper confidence bound u_t(s), while everything outside in red is unsafe with high probability.
To expand X_t- we have the boundary expanders G_t—certified safe border cells that can verify new neighbors—and pick the single state 'eta' defined by the widest uncertainty interval w_t(s) = u_t(s) - l_t(s) as our next exploration target.

## Slide 7 — How ES2 and P-ES2 Decide When to Stop Exploring Safety (~45s)
 
If we wait until uncertainty shrinks below a certain level across the entire boundary, Step 1 wastes hundreds of actions mapping low-reward regions. To solve this, the paper introduces two early-stopping checks:
On the left, ES2 creates an optimistic virtual MDP M_y over X_t+ where cells outside our certified safe set get an optimistic reward bonus u^r + alpha^{1/2}.sigma^r, while cells inside get a pessimistic reward u^r - alpha^{1/2}.sigma^r. If even this optimistic planner chooses to stay inside X_t- (Y_t subset of X_t-), no boundary expansion can improve our reward, so we stop Step 1 immediately.
To counter the limitation of ES2, in P-ES2 instead of assuming every cell in X_t+ is 100% safe, it uses our Safety GP's normal distribution to compute the actual probability p(s) = Pr[g(s) >= h] of each cell being safe, and adds a failure probability 1 - p(s) of landing in a zero-reward absorbing state z. Penalizing risky border cells this way triggers early stopping much sooner

## Slide 8 — SNO-MDP Algorithm Flow & Reward Exploitation (~40s)
Putting everything together, Algorithm 1 runs in two clean stages:
In Step 1, starting from our safe base S_0, the agent repeatedly updates X_t- and X_t+, moves safely toward the most uncertain expander eta, updates both Gaussian Processes, and checks whether ES2 or P-ES2 triggers an early break.
Once Step 1 terminates, we enter Step 2: we freeze the certified safe set Omega and run standard Value Iteration using the optimistic reward upper confidence bound U_t(s) = u_t^r(s) + alpha_{t+1}^{1/2}.sigma_t^r(s). Because every state inside Omega is already certified safe, Step 2 doesn't have to trade off safety against reward—it simply plans the highest-reward path.

## Slide 9 — Paper's Experimental Results: Synthetic & Mars DEM (~45s)

Wachi and Sui validate this approach across two benchmarks:
First is the 20×20 synthetic GP-SAFETY-GYM environment, original SNO-MDP takes roughly 740 actions just to finish Step 1. Adding ES2 cuts that down to 180 actions, while P-ES2 switches at just 60 actions, reaching the full 1.00 normalized optimal reward while SafeExpOpt-MDP plateaus around 0.71.
Next, on real Mars HiRISE terrain data with a strict 25-degree slope limit, original SNO-MDP only scores 0.49 because it spends almost its entire episode mapping slopes. With ES2 it jumps to 0.78 and P-ES2 reaches 0.81, outperforming every baseline without a single slope violation.

## Slide 10 — Our Team's Extension: Dynamic Recheck of Historical Safety Permissions (~45s)
 
While auditing the paper's implementation on larger, noisy maps, we uncovered a subtle vulnerability: standard SNO-MDP treats the certified safe set X_t- as append-only.
In practice, early optimistic GP estimates can certify a border cell as safe, giving it a 'retained certificate.' Later, when the agent gathers nearby observations showing a steep drop, the updated GP's lower confidence bound can fall below the safety threshold h—yet the agent still trusts the old certificate. 
Our extension fixes this with a three-step dynamic audit after every GP update: we re-evaluate LCBs across all previously certified cells, revoke any cell that falls below h, and re-verify reachability and returnability before choosing the next action. Next, Akash will show our implementation and experimental results proving how this eliminates noisy violations.

## Slide 11 — Experimental design: freeze first, evaluate fresh cases
 
Don't speak : *0:00–0:40. Point to six maps, two repeats, three noise factors, and the common 160-action budget.*
 
Before showing the graphs, here is our evaluation setup. We froze the settings before testing six new 20-by-20 maps, with two sensor-noise repeats and three assumed-noise settings.
 
We completed 192 runs, including reference runs. Every run used 160 actions, including travel and exploration. This makes the comparison fair. To test revalidation separately from confidence tuning, we have compare methods with the same safety confidence multiplier.
 
## Slide 12 — Reward across the same action budget
 
Don't speak : *0:40–1:25. Point to the axes, then compare the grey and teal curves. Identify the dashed reference.*
 
The horizontal axis shows the number of actions taken. The vertical axis shows discounted return—the total reward accumulated so far, where a reward earned earlier counts more than the same reward earned later.
 
Compare grey SNO-MDP (Safe Near-Optimal Markov Decision Process, the basic method) with teal ES² (Early Stopping of Exploration of Safety).
 
With the confidence multiplier at c = 2, early stopping earns greater than the basic method. The dashed oracle knows the true environment and is a privileged reference. Higher reward alone does not establish safer performance.
 
## Slide 13 — Safety and reward must be read together
 
Don't speak : *1:25–2:40. This Explain the error bars, define the two permission rules, then compare the two conditions at four.*
 
Here in the bar-chart, showing reward and safety together. The error bars show SD (standard deviation: variation across six map averages), not confidence intervals.
 
The label c (safety confidence multiplier) controls the width of the model's safety margin. c=4 requires stronger evidence than c = 2.
 
Retained means here: an earlier learned safety permission can remain active even when the current bound fails.
and Revalidated means here : learned permissions are checked against current bounds; unsupported permissions are excluded before planning.
 
so like, left side bar-chart is Avg return with confidence multiplier. means “How much reward did the rover collect?” Higher is better. The black error bars show SD (standard deviation)—how much the average returns differed between maps. Longer error bars mean more variation.
Right side bar chart is Avg unsafe actions per run with confidence multiplier. means “How many unsafe moves did the rover actually execute?” Lower is better.
 
At sefety multiplier c = 2, the average is 2.5 unsafe actions per run. Both versions ( retained & revalidate) at c = 4 record zero here, with lower mean reward. Comparing these two at the same multiplier isolates the permission rule.
 
Retained c = 2 to retained c = 4: changing only the confidence setting reduced observed violations, while also reducing reward.
Retained c = 4 to Revalidated c = 4: keeping confidence fixed and examine the permission rule. Both recorded zero violations on this slide, so these results alone do not show an additional safety benefit from revalidation.
 
so here it's confirming clearly : Higher reward alone does not mean better safe reinforcement learning. We must examine reward and unsafe actions together.
 
## Slide 14 — Robustness when sensor noise is misspecified
 
Don't speak : *2:40–3:30. Explain the three assumptions, then point to the side-panel totals: 98, 2, and 0.*
 
In this graph "Robustness when sensor noise is misspecified" shows the actual altitude-sensor noise stays fixed; we change only the assumed noise standard deviation.
 
A factor of 0.5 underestimates noise, one matches it, and two overestimates it. The graph shows unsafe actions per run; the side panel gives totals across 36 runs per method.
 
Retained at c = 2 records 98 unsafe actions, retained at c = 4 records 2 only unsafe actions, and revalidated at c = 4 records zero. This is limited experimental evidence only.
 
## Slide 15 — ES2 shifts actions from exploration to reward
 
Don't speak : *3:30–4:10. Compare the first two bars and point to the 42.58-action difference.*
 
In this graph bar-chart, shows how early stopping changes the fixed budget. Compare the first two bars, both using the confidence multiplier of 2 ( c = 2)
 
The basic method spends all 160 actions exploring safety. Early stopping averages 117.42, reallocating 42.58 actions and 26.6 % of the budget—to reward optimization.
 
Every run still uses 160 actions and collects reward during exploration. This shows a change in priorities, not faster computation.
 
 
Thank you. Srinivas will now demonstrate the dashboard.
 
---