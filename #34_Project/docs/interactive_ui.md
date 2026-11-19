# Interactive experiment and analysis dashboard

## Start

From `#34_Project`, with the project’s Python environment active:

```bash
python3 -m analysis_ui
```

Open **http://127.0.0.1:8765** in a browser. Stop the server with Ctrl+C. To use a different port: `python3 -m analysis_ui --port 8766`.

The dashboard uses the existing NumPy-based simulation engine. It needs no extra packages, frontend build, online fonts, or external services. It binds to this computer’s loopback interface. Keep the Python server running while using the browser; opening its HTML through `file://` cannot run experiments. One dashboard worker owns this project’s interactive queue; use that running instance or stop it before starting another. The worker lock uses the standard Unix interface available on macOS/Linux.

## 1. Understand a recorded run

Select **Saved demos & reproductions** in the library and open `demo_source`.

1. Press **Play**, drag the step slider, or use the previous/next buttons.
2. Watch the rover’s travelled path and executable region. Blue cells belong to the model-certified reachable/returnable region; gold cells are optimistic only. They are not necessarily truly safe. Red path segments mark executed violations.
3. Use **Field inspector** to switch between true altitude/reward and GP mean/uncertainty. True fields are evaluator information; policies never receive them. Each field defaults to a fixed colour scale across the run so changes remain comparable over steps. Optional per-frame rescaling is explicitly labelled. Read the limits when comparing different runs or fields.
4. Click a cell in either map. The action table reconstructs the joint transition GP from saved observations up to that step and shows mean, uncertainty, lower bound, true margin, executable status, and certificate support.
5. Change the trajectory graph metric. Click the graph to move the replay to that step. Download recorded configurations, progress CSV, observations, and source manifests for independent checking.

Coordinates are `(x, y)` with y increasing upwards. The outlined initial cell is the start; a purple outline is the exploration target. A selected inspection cell has an orange border. The rover marker shows its current position. **Certified directions** adds short direction marks for executable actions.

Safety is attached to directed movements:

```text
g(s,a) = altitude(s) − altitude(next_state)
LCB(s,a) = GP_mean_of_difference − c × GP_std_of_difference
true violation: g(s,a) < h
```

The difference uncertainty includes covariance between the two endpoint altitudes. A reversed move can have a different safety outcome. Stay is structurally safe. Supplied initial certificates remain accepted in both modes. The table displays the raw GP difference bound. The certificate tracker uses `h` as its special always-safe stay convention; the trajectory’s raw next-action GP bound for stay can be zero.

**Retained** mode can execute a previously accepted edge whose current LCB has fallen below `h`. **Revalidated** mode requires a current bound or initial certificate, followed by reachability/returnability checks. A good bound on an isolated edge alone does not make it executable. Revalidation is a project extension and is not an unconditional safety guarantee.

Use **Before first violation** in `demo_source` to reach step **96**, cell **(8, 8)**. The up action has LCB approximately −0.280632, below the −0.25 threshold, but remains executable through a retained certificate. **First violation** moves to step **97**, where the count becomes 1. **Reward switch** and **Final result** reach other important events; absent events are disabled. The step explanation connects pre-action bounds, actual outcomes, and ES2. Use the coordinate fields as a keyboard-accessible alternative to clicking a map.

Historical runs remain replayable using their saved fields. Learner bound reconstruction is refused when the recorded GP or agent revision differs from the loaded one. Oracle truth inspection does not require a GP.

## 2. Compare methods and the frozen study

Open **Compare & analyse → Load matched example**. This selects four conditions from map 100 / observation 3000 / matched noise:

| Condition | Interpretation |
|---|---|
| ES2 retained, c=2 | Source-profile baseline |
| ES2 retained, c=4 | Confidence control |
| ES2 revalidated, c=4 | Project extension at the same confidence |
| Oracle | True-field reference with an explicit information advantage |

Compare the two c=4 conditions to isolate certificate handling. Comparing c=2 retained directly against c=4 revalidated changes both confidence and certificate mode. Tick other runs in the library to choose up to 12 trajectories; selection remains active when you switch library collections. With exactly two distinct runs, the dashboard shows configuration differences and a right-minus-left result when map, observation stream, horizon, and scoring checks match. A single pair is descriptive, not a statistical conclusion.

The aggregate section defaults to the **frozen final study at noise factor 1**. Change the factor, map seed, or observation seed to inspect subsets. The final study contains 192 planned runs: 180 learners and 12 oracle references across six generated maps and two observation seeds. Filtering to factor 1 includes 60 learner runs and all 12 oracle references.

Read the aggregate columns together:

- **Return mean ± map SD:** includes the reward earned during exploration and travel. Average observations within each true map first, then average maps equally. SD is across map means, not a confidence interval; a single map has no reported SD.
- **Mean unsafe actions:** physical constraint violations, rather than incorrect beliefs alone.
- **Runs with violation:** average per-map fraction of recorded runs with at least one violation.
- **Early termination:** fraction of recorded runs that stopped before normal horizon completion; their executed return prefixes remain included.
- **ES2 triggered:** whether the ES2 stopping condition occurred; this alone does not establish superior reward or safety.

The scatter plot shows the return/safety trade-off. Hover over points and trajectory graphs for values. Export the aggregate table as CSV or a displayed graph as SVG for presentation.

Condition IDs connect scatter points to table rows. Expand **Exact settings & cohort** to inspect model parameters, source identity, and the map/observation combinations used. Warnings identify differing background settings, source revisions, unmatched cohorts, or multiple changes in a two-run comparison. Changing aggregate filters or the selected cohort clears old results and disables exports until **Update analysis** finishes.

Duplicate copies with identical method/configuration/map/model hashes count once. Different model settings or scientific source revisions stay in separate groups. Map hashes, rather than folder names, determine independent map groups. Recorded setup failures have no trajectory and are listed explicitly in experiment history, outside trajectory aggregates. Review those failures before drawing conclusions about a new sweep. This dashboard does not replace the frozen runner’s full protocol verification or supply confidence intervals/significance tests.

## 3. Run your first experiment

1. Open **New experiment → Quick 6 × 6**.
2. Keep **Single run** and the default SNO-MDP + ES2 method. The quick preset uses map 3, observation 2026, horizon 20, and discount 0.95; it is an integration check, not a final-study trial.
3. Press **Preview cases** and inspect the exact configuration.
4. Press **Run experiment**. The local engine executes it in the background. Queue progress updates between cases; no live per-step progress is fabricated.
5. When the case finishes, press **Open replay**. Its summary, maps, GP fields, and action table use actual saved engine outputs.
6. Use **Use this configuration** on a recorded run to prepare a related experiment. You can also import an existing `config.json`.

The source and revalidated demo buttons load their original 20 × 20 / 160-action settings. Larger GP problems and sweeps take longer. Maximum interactive limits are 25 × 25 cells, 400 actions, 48 cases per batch, and five active/queued batches. Cases execute sequentially. **Cancel remaining** finishes the current case and skips those not yet started. A server interruption is recorded when the app restarts; completed cases remain available, but partial cases are not automatically resumed.

## 4. Make a controlled experiment

Choose **Matched method comparison** and tick the desired presets. The map, observation seed, horizon, actual sensor noise, and other base settings stay shared. The preset labels explicitly set the method, certificate mode, and safety multiplier.

To vary several conditions, choose **Parameter sweep**:

| Control | What changes |
|---|---|
| Map seeds | Independent generated GP terrains; the released fixture remains one fixed map |
| Observation seeds | Sensor-noise realizations on a terrain |
| Actual altitude noise | Noise physically added to altitude observations |
| Actual reward noise | Noise physically added to reward observations |
| Assumed-noise factor | Altitude GP noise assumption = actual altitude noise × factor; reward GP uses actual reward noise |
| Safety multiplier c | Width of the altitude-difference confidence interval |
| Certificate mode | Historical retention or current-LCB revalidation |
| Method | SafeMDP, basic SNO-MDP, SNO-MDP + ES2, or privileged oracle |

A manageable first sweep uses the quick preset, maps `3,4`, observations `2026`, c values `2,4`, factor `1`, both modes, ES2 and oracle: **10 cases**. Oracle runs once per map/observation pair, rather than once per confidence setting. Preview the full product before submitting. A 6 × 6 sweep demonstrates behaviour; it is not a substitute for the existing frozen 20 × 20 evaluation.

After completion, choose **All interactive runs**, select the specific **Interactive batch**, and filter the intended seeds/factors. Batch-wide planned/finished/setup-failure counts are shown before map/noise filtering; trajectory means do not include cases lacking a trajectory. Separate experimental configurations remain separate rows. Inspect failures and individual trajectories before interpreting aggregate changes. Dashboard runs are exploratory evidence; adjusting settings after looking at held-out outcomes does not create a new unbiased final evaluation.

Preset buttons reset to a single ES2 run. Choose the experiment type after selecting a preset. Fields overridden by matched presets or sweep values are disabled and explained. Editing a configuration clears its old preview.

## 5. Saved files and reproducibility

```text
analysis_ui/                    # UI/server code, separate from frozen algorithm
results/interactive/<batch>/
  request.json                  # Requested experiment
  plan.json                     # Fully expanded case configurations
  lab_manifest.json             # UI hashes and frozen-source status
  job.json                      # Queue state, case results, explicit failures
  cases/001/
    config.json, manifest.json, map.json, summary.json
    observations.json.gz, trajectory.json.gz, es2_checks.json.gz
    progress.csv, timing.csv, index.html, ui_case.json
```

Each run uses the original `sno_mdp.experiment.run_experiment` function. The UI never edits scientific package files, recorded final results, or the frozen protocol. For independent CLI reproduction, download the chosen configuration and run:

```bash
python3 -m sno_mdp run --config path/to/config.json \
  --method sno_es2 --output results/my_ui_reproduction
```

Use the method recorded in the selected run; use a fresh output folder. Compare map/config/source hashes, action sequence, termination, unsafe counts, and returns. Computation timings depend on machine and local load. The dashboard reuses recorded timing evidence when available; it is not a controlled runtime benchmark. Existing submission ZIPs are snapshots made before this dashboard and must be regenerated separately if the UI should be included.

**Export analysis JSON** preserves the current filters, computed summary, run paths/configurations, map/source identities, cohorts, and batch context. SVG exports include corresponding context in metadata. Restart the dashboard if scientific source files change; it refuses new cases while its imported code differs from the files on disk.

For a step-by-step professor demonstration, follow the [UI presentation guide](../report/ui_presentation_guide.md). The [UI critique](ui_review.md) documents the review and fixes.

The [complete project guide](../06_Final_Complete_Project_Guide.md), [method](method.md), and [experiment protocol](experiments.md) explain the theory and study design. The UI is an additional way to explore their recorded evidence.
