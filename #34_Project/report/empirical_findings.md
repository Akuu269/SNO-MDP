# Empirical findings and interpretation

All settings were frozen before the held-out evaluation. These are descriptive simulation findings, not safety guarantees or significance claims.

- Completed 192 final cases on six maps with two observation seeds and three noise assumptions.
- Source c=2 ES²: 98 unsafe actions across 36 cases.
- Selected-confidence retained ES²: 2 across 36 cases.
- Revalidated ES²: 0 across 36 cases; 0 failures.

Selected-confidence retained ES² recorded 2 unsafe actions; revalidated ES² recorded 0. This is the matched-confidence comparison needed to separate certificate handling from confidence tuning.

At matched noise, source ES² mean return is 177.555; selected-confidence retained ES² is 157.184; revalidated ES² is 155.306. Compare map SD and paired differences rather than treating these means as universal improvements.

The development-selected confidence setting does not maximize held-out nominal reward. The source c=2 ES² policy achieves higher nominal mean reward here, with more unsafe runs; the conservative variants give up reward while reducing observed violations. This unfavorable reward result is retained rather than retuning on held-out outcomes.

The earlier diagnostic mechanism remains informative: a practical confidence error can admit an action, and monotone retention can allow later execution despite corrected beliefs. That observation does not refute the paper’s theorem under its stated assumptions; this implementation does not establish a uniform confidence event.

[Noise robustness figure](../results/final/figures/noise_robustness.png) · [Inference and decision cost](../results/final/figures/computation.png) · [Every final case](../results/final/runs.csv) · [Full report](report.md)
