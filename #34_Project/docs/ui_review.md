# UI critique and corrections

Review date: 16 November 2026. Scope: the local UI, experiment wrapper, and presentation of existing evidence. Scientific files under `sno_mdp/` and the frozen evaluation were preserved.

| Gap | Why it mattered | Correction |
|---|---|---|
| Heatmaps rescaled every step | Shrinking uncertainty could retain equally intense colours | Fixed per-field scale across a run; labelled optional per-frame scaling |
| Different settings produced similarly named rows | A horizon/kernel/source difference could look like a method effect | Condition IDs, exact settings, source identity, and cohorts |
| Old aggregates survived changed filters/selections | Displayed controls and exported results could disagree | Clear stale graphs/exports, require recomputation, ignore late responses |
| Interactive batches were pooled | Experiments and missing results were hard to isolate | Batch filter plus planned/finished/setup-failure/missing-result counts |
| Pair matching omitted map seed | The released fixture can stay fixed while its seed changes sensor randomness | Include map seed; disclose unequal cohorts and source revisions |
| Multi-control comparisons were easy to misread | c=2 retained versus c=4 revalidated changes two controls | Explicit warning plus left/right identities and differences |
| Presets retained old method/type | A source preset after oracle selection could run oracle | Reset presets to single ES2; reset copied-config experiment type |
| Overridden controls stayed editable | Displayed values could differ from effective case settings | Disable overridden fields and explain the active controls |
| Preview survived form edits | Reviewed and launched configurations could differ unnoticed | Clear previews on edits; validate before preview |
| Key replay events required manual searching | Hard to connect pre-action evidence with outcomes | Event bookmarks, coordinate inspection, and step explanation |
| Later travel could cover red segments | Violations could visually disappear | Draw violation segments last |
| Exports lacked sufficient context | Graphs could become detached from their experiment | Analysis JSON and SVG metadata retain filters, settings, paths, hashes, cohorts |
| Oracle inspection required matching GP code | Oracle does not use a GP | Exempt oracle; check GP and agent revisions for learner reconstruction |
| Source could change after server startup | Imported code and recorded disk hashes could disagree | Reject new cases until restart after a scientific source change |
| Worker started before binding the port | Failed server creation could retain an unused queue owner | Bind first, clean up initialization failures, concise startup errors |

Recorded early-termination prefixes remain in aggregate means. Setup failures without trajectories remain outside those means; batch-wide counts expose that denominator. Statistics are descriptive means and map SD, without significance tests or confidence intervals. The scientific runner/verifier remains authoritative for the complete frozen protocol.

Validation includes Python/API regressions, exact bounds, source preservation, matching/cohort logic, source-change rejection, actual browser interaction, export provenance, presets, preview invalidation, a new two-case matched run, and mobile layout.

Optional repeatable browser check, with the dashboard running:

```bash
node tests/browser_dashboard.cjs
```

Node and Chrome are needed for this developer check only. It creates a small exploratory matched batch. Set `CHROME_BIN` or `LAB_URL` if needed.

Follow the [UI presentation guide](../report/ui_presentation_guide.md) for the professor demonstration, or the [dashboard guide](interactive_ui.md) for ordinary operation.
