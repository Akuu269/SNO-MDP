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

The report, presentation PDF/browser slides, speaker notes, empirical interpretation, demo walkthrough, frozen protocol, and team-review form are under `report/`. The complete local submission archive is generated with:

```bash
python3 -m sno_mdp package
```

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
