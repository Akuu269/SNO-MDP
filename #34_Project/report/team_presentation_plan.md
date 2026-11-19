# Final presentation plan: three PPT presenters and one dashboard presenter

**Members 1–3 present the integrated 15-slide deck. Srinivas presents the live dashboard afterward.** Members 1 and 2 retain their existing slides and speaking slots. Member 3 uses the five-minute rehearsal, including experimental design and saved dashboard evidence.

Names remain placeholders. Speaking responsibility is separate from actual implementation contributions.

## Final division

| Presenter | Slides | Responsibility | Target / maximum |
|---|---|---|---|
| Member 1 | 1–4 | Paper objective, unknown functions, assumptions, prior-method objectives and project scope | 3:30 / 4:00 |
| Member 2 | 5–8 | GP bounds, X−/X+/expanders, revalidation, ES2 versus paper-only P-ES2 | 3:30 / 4:00 |
| Member 3 | 9–15 | Experimental design, four results graphs, saved dashboard evidence, Thank You and handoff | 5:00 including handoff |
| Member 4 | No main PPT slides | Live dashboard, matched experiment, evidence and closing | 3:30 / 4:00 |

Use [PowerPoint](presentation.pptx), [PDF](presentation.pdf), or [browser slides](presentation.html). Detailed [speaker notes](presentation.md) are embedded in PowerPoint Presenter View. Audience slides contain no rehearsal timings or member instructions.

## Member 1 — problem and scope

| Time | Slide | Explain |
|---|---|---|
| 0:00–0:40 | 1 | Paper, motivation, research question |
| 0:40–1:40 | 2 | States/actions, local observations, reward, directed safety |
| 1:40–2:25 | 3 | Safe seed, kernel/RKHS regularity, Lipschitz safety, calibrated confidence |
| 2:25–3:20 | 4 | SafeMDP objective; SNO-MDP/ES2; paper-only P-ES2 and project scope |
| 3:20–3:30 | Handoff | “Member 2 will explain how observations become safety bounds and executable actions.” |

Show the grid as an illustration, not a measured result. Say that our study uses synthetic GP-Safety-Gym fields in a headless discrete port. Evaluator/oracle truth is unavailable to the learner. Native MuJoCo and measured Mars data are not claimed.

## Member 2 — method and implementation

| Time | Slide | Explain |
|---|---|---|
| 0:00–0:55 | 5 | Two GPs; covariance in transition uncertainty; LCB |
| 0:55–1:45 | 6 | Pessimistic/optimistic regions, expanders, graph checks and uncertainty widths |
| 1:45–2:40 | 7 | Retained certificates versus our current-LCB extension |
| 2:40–3:20 | 8 | ES2 successor test versus paper-only probability-weighted P-ES2 |
| 3:20–3:30 | Handoff | “Member 3 will show the controlled study and the actual reward and safety graphs.” |

Explain the covariance equation rather than deriving the full GP posterior. Distinguish actual sensor noise, assumed-noise factor, and direct code multiplier c. The paper's beta notation is not interchangeable with c. P-ES2 is literature background, not a method we ran. Do not repeat the full dashboard failure walkthrough; Member 4 will show steps 96/97.

## Member 3 — experiments, graphs, and conclusions

The [Member 3 rehearsal](member3_results_rehearsal.md) takes five minutes across final team slides **11–17**, matching the [prepared replacement section](member3_shared_deck_replacement.pptx). In the local 15-slide [preview](presentation.pptx), the same section occupies slides 9–15.

| Final team slide | Local preview | Focus | Seconds |
|---|---|---|---|
| 11 | 9 | Experimental design | 40 |
| 12 | 10 | Reward comparison | 45 |
| 13 | 11 | Safety–reward comparison; Retained and Revalidated definitions | 75 |
| 14 | 12 | Noise robustness | 50 |
| 15 | 13 | Early-stopping benefit | 40 |
| 16 | 14 | Saved dashboard evidence | 40 |
| 17 | 15 | Thank You and Srinivas handoff | 10 |

The total is 300 seconds. Follow the spoken paragraphs; do not read the italic cues or preparation glossary aloud.

The four result graphs come from [member3_results.pptx](member3_results.pptx). The former duplicate results and unused backup slides have been removed from the combined deck; saved dashboard evidence is now included in the main sequence. Numerical preparation and viva material remain in the [detailed results notes](member3_results_speaker_notes.md) and [claim review](presentation_claim_review.md).

Hold the safety confidence multiplier at four when interpreting the certificate extension. Zero observed violations is not a safety proof. Identify the dashboard screenshot as saved evidence, distinct from the live demonstration.

## Member 4 — dashboard only

Switch from the Thank You slide (15) to the already-open dashboard. Saved evidence is included on slide 14 immediately before Thank You.

| Time | Screen | Show and explain |
|---|---|---|
| 0:00–0:20 | `demo_source` | Blue/gold regions, GP uncertainty, fixed colour scale |
| 0:20–1:05 | Replay | **Before first violation**: step 96, up-action LCB −0.280632 < −0.25 with retained support. **First violation**: step 97, count becomes 1 |
| 1:05–1:45 | Compare & analyse | Load matched example; point to the c=4 control and the cohort used for the PPT graphs |
| 1:45–2:35 | New experiment | Preview and run the prepared two-case 6 × 6 comparison below |
| 2:35–3:05 | Completed case | Open replay; show return, violations, and saved configuration/evidence |
| 3:05–3:30 | Evidence and closing | Raw records/source hashes, analysis export, exploratory versus frozen results, one limitation |
| 3:30–4:00 | Buffer | Covers window switches or a slow queue; no new material |

### Prepare before presenting

1. Run `python3 -m analysis_ui` from `#34_Project` and open **http://127.0.0.1:8765**. Use the existing running instance if present.
2. Tab A: `demo_source` at Initial state, **GP altitude uncertainty**, **Fixed for this run**.
3. Tab B: **Compare & analyse → Load matched example**, **Frozen final study / factor 1 → Update analysis**.
4. Tab C: **Quick 6 × 6 → Matched method comparison**, horizon **8**, only **ES2 retained c=4** and **ES2 revalidated c=4** selected. Keep map 3, observation 2026, discount 0.95, and nominal noise.
5. Run the same small comparison during rehearsal and retain its labelled saved batch as fallback. Prepare the same values for the live run.
6. Test the projector and use one computer for quick transitions.

A preset resets the experiment type, so select **Matched method comparison after Quick 6 × 6**. New runs are exploratory outputs, not replacements for the frozen study.

**If a run takes more than 15 seconds:** show the saved identical rehearsal batch and clearly identify it as saved evidence while the new run continues. Do not wait for a 20 × 20 study or multi-map sweep.

**If the UI fails:** return to slide 14's saved step-97 screenshot. Explain the mechanism and limitations, then finish within the same four-minute cap.

## Evidence and questions

The final deck has no hidden backup slides. Saved dashboard evidence is slide 14, immediately before Thank You. Use the [claim review](presentation_claim_review.md) and detailed notes to prepare for questions about paper assumptions, parameter choices, and the limits of the study.

## Final checks

- Members 1, 2, and 4 rehearse their existing slots. Member 3 rehearses the complete seven-slide route within five minutes.
- Do not repeat Member 3’s graph explanation during the dashboard demo.
- Add actual names; confirm implementation contributions in [team review](team_review.md).
- Use [the longer UI guide](ui_presentation_guide.md) for preparation only.
- Scientific source and recorded final outcomes remain unchanged.

To rebuild the deck and graph assets: `python3 report/build_team_presentation.py`. Matplotlib/Pillow and `python-pptx` are build dependencies; no new UI dependency is needed. Graph CSVs and provenance are under `report/presentation_assets/`.

The original `sno_mdp package` command still contains its old slide template. Run this standalone builder afterward if necessary. Existing submission archives are older snapshots until explicitly refreshed.

