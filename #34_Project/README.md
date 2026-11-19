# SNO-MDP: safe exploration, reward, and certificate revalidation

A reproducible course project based on Wachi and Sui (ICML 2020). We study how sensor-noise assumptions, confidence settings, and old safety certificates affect exploration and reward.

The environment is a **headless port of the authors’ GP-Safety-Gym discrete benchmark**. Native legacy MuJoCo execution and exact published curves are not claimed. The current-LCB alternative is explicitly our project extension.

## Read in this order

Start with the [complete paper-to-presentation guide](06_Final_Complete_Project_Guide.md) for one step-by-step learning document covering fundamentals, paper theory, implementation, experiments, reproduction, and the professor demonstration.

1. [Project roadmap](05_Project_Plan.md): deliverables and remaining human work.
2. [Method](docs/method.md): equations, source differences, and certificate handling.
3. [Experiment protocol](docs/experiments.md): development selection, stopping, and frozen evaluation.
4. [Report](report/report.md) and [empirical interpretation](report/empirical_findings.md): measured findings and limitations.

## Install and check

From the course directory:

```bash
cd '#34_Project'
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m unittest discover -s tests
```

The supplied dependencies were checked with Python 3.9.6. Use an existing environment if they are already installed.

## Demonstration

For the interactive experiment dashboard, run from this project folder:

```bash
python3 -m analysis_ui
```

Open **http://127.0.0.1:8765**. Replay saved runs, inspect GP safety bounds, compare methods and final-study graphs, or launch new runs and parameter sweeps. Begin with **New experiment → Quick 6 × 6**. New outputs go under `results/interactive/`; recorded final-study files are read-only in the dashboard. No additional dependencies or frontend build are needed. See the [dashboard guide](docs/interactive_ui.md) for a walkthrough and analysis rules.

For the professor demonstration, follow the [UI presentation guide](report/ui_presentation_guide.md): it includes the exact step-96/97 failure example, matched controls, measured results, and what the study does and does not establish. The [UI review](docs/ui_review.md) records the critique and corrections.

```bash
python3 -m sno_mdp run --config configs/smoke.json --method sno_es2 --output results/smoke_current/sno_es2
python3 -m sno_mdp run --config configs/demo_source.json --method sno_es2 --output results/source_reproduced
python3 -m sno_mdp run --config configs/demo_revalidated.json --method sno_es2 --output results/revalidated_reproduced
python3 -m sno_mdp plot --run results/revalidated_reproduced
python3 -m sno_mdp diagnose --run results/revalidated_reproduced
```

Open the supplied [source demo](results/demo_source/index.html), [revalidated demo](results/demo_revalidated/index.html), or [professor walkthrough](report/demo_walkthrough.md). These demo folders are labeled copies of the first held-out map/observation at matched noise, not additional trials. A new run saves its replay as `index.html`. Use a current browser for its compressed, self-contained data. No server or network is required. The oracle and evaluator can read true fields; learners receive noisy observations and supplied certificates only.

## Controlled studies

```bash
python3 -m sno_mdp research --suite configs/research_calibration.json --output results/calibration_reproduced
python3 -m sno_mdp research --suite configs/research_stopping.json --output results/stopping_reproduced
python3 -m sno_mdp research --suite configs/research_final.json --output results/final_reproduced --protocol report/frozen_protocol.json
python3 -m sno_mdp verify --study results/final_reproduced
```

The final runner requires the frozen protocol and rejects changed source/configuration. Use a fresh output directory. The calibrated multiplier and selection scores are saved in `report/calibration_selection.json`; the selection rule was specified in `configs/selection_rule.json` before calibration outcomes. Research studies retain full compressed raw traces and create selected replays rather than duplicating a browser page for every case.

The completed study’s entry point is `results/final/index.html`; its inventory, plots, raw logs, and verification are alongside it. `results/research_calibration/` and `results/research_stopping/` contain the development evidence. Earlier pilot, validation, and audit outputs remain historical evidence; their source hashes describe the earlier revision and current replay diagnostics deliberately reject changed model code.

## Report and presentation

The [final PowerPoint](report/presentation.pptx), [PDF](report/presentation.pdf), [browser slides](report/presentation.html), and [speaker notes](report/presentation.md) contain **15 integrated slides**: Members 1–2 on 1–8, experimental design on 9, Member 3's four results graphs on 10–13, saved dashboard evidence on 14, and Thank You on 15. Srinivas then presents the live dashboard. The [Member 3 rehearsal](report/member3_results_rehearsal.md) is a five-minute script for final team slides 11–17; these correspond to slides 9–15 in this local preview. Follow the [team plan](report/team_presentation_plan.md). The [claim review](report/presentation_claim_review.md) separates verified paper facts from project evidence and unsupported claims.

The complete [17-slide v4 deck](report/SNO-MDP_Team_Presentation_v4.pptx) preserves the saved v3 slides 1–10 and uses the five-minute Member 3 section on slides 11–17.

**Member 3:** use the separate [results PowerPoint](report/member3_results.pptx) or [PDF](report/member3_results.pdf), with **four results slides—reward, safety–reward comparison, noise robustness, and ES2 exploration budget—plus a separate Thank You slide (five total)**. Read the [detailed results notes](report/member3_results_speaker_notes.md) for graph/table interpretation and viva answers; the [five-minute rehearsal](report/member3_results_rehearsal.md) covers the complete seven-slide section, including design and saved dashboard evidence. The [prepared section](report/member3_shared_deck_replacement.pptx) includes the same shortened speaker notes and definitions of Retained and Revalidated. Rebuild with `python3 report/build_member3_results.py`; the full team deck is preserved.

The report, empirical interpretation, walkthrough, frozen protocol, and team-review form are also under `report/`. The complete local submission archive is generated with:

```bash
python3 -m sno_mdp package
```

That original package generator recreates its older ten-slide template and team-review form. Rebuild this team version afterward with `python3 report/build_team_presentation.py`; an existing submission ZIP does not automatically include the revised deck.

Run `python3 -m scripts.export_noise_findings` to regenerate the posthoc noise/computation figures and discussion from the frozen results. Packaging writes `dist/SNO_MDP_Project.zip`, including source, data provenance, licenses, configurations, documentation, and saved results. Team names, actual contributions, independent human reproduction, professor formatting, and rehearsal must be confirmed by the team.

## Folder map

| Path | Purpose |
|---|---|
| `sno_mdp/` | Environment, GP, certificates, planning, experiments, diagnostics, reports, CLI |
| `configs/` | Individual runs, development studies, selection rule, final suite |
| `tests/` | Independent numerical, graph, revalidation, failure, and protocol checks |
| `data/`, `third_party/` | Released fixture, pinned reference source, hashes, MIT notices |
| `docs/` | Method and experimental protocol |
| `report/` | Report, slides, frozen protocol, and human review |
| `results/` | Scientific records, figures, and selected replays |
| `extensions/terrain_data/` | Retained Mars measurements/provenance; outside current study |
| `dist/` | Regenerable submission archive |

[Original paper](SNO_MDP_Wachi_Sui_2020_arxiv_v1.pdf). Finite experiments and practical confidence bands do not establish a safety guarantee. Failures and unfavorable results remain part of the study.
