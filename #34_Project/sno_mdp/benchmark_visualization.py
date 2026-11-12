"""Standalone SVG figures and a local HTML comparison dashboard."""

import json
import io
import os
import tempfile

import numpy as np


COLORS = {
    "oracle": "#283e52",
    "safemdp": "#aa7a24",
    "sno_basic": "#2c80b6",
    "sno_es2": "#9a4dae",
}


def make_return_figure(curves, grid, factor, horizon):
    selected = [
        c
        for c in curves
        if c["grid"] == grid
        and (c["method"] == "oracle" or c["noise_factor"] == factor)
    ]
    series = {
        method: np.mean(
            [c["discounted_return"] for c in selected if c["method"] == method], axis=0
        )
        for method in COLORS
    }
    # Keep font/config caches in a writable temporary directory. Matplotlib is
    # imported lazily so exploration and planning still require only NumPy.
    previous = {k: os.environ.get(k) for k in ("MPLCONFIGDIR", "XDG_CACHE_HOME")}
    with tempfile.TemporaryDirectory(prefix="sno-matplotlib-") as cache:
        for name in previous:
            if previous[name] is None:
                os.environ[name] = cache
        try:
            import matplotlib as mpl
            from matplotlib.figure import Figure

            with mpl.rc_context(
                {"svg.hashsalt": "sno-mdp-comparison", "font.family": "DejaVu Sans"}
            ):
                figure = Figure(figsize=(10, 4.7), layout="constrained")
                axes = figure.subplots()
                for method, values in series.items():
                    axes.plot(
                        np.arange(horizon + 1),
                        values,
                        label=method,
                        color=COLORS[method],
                        linewidth=2,
                    )
                axes.set(
                    xlabel="Scheduled action budget",
                    ylabel="Mean discounted return",
                    title="%s · assumed / actual safety noise = %g ×" % (grid, factor),
                    xlim=(0, horizon),
                    ylim=(0, None),
                )
                axes.grid(alpha=0.2)
                axes.legend(loc="lower right")
                buffer = io.StringIO()
                figure.savefig(buffer, format="svg", metadata={"Date": None})
                return buffer.getvalue()
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


def write_benchmark_visualization(output, summary, rows, curves):
    from .benchmark import noise_slug

    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    grids = ["%dx%d" % tuple(grid) for grid in summary["suite"]["grids"]]
    figure_paths = {}
    for grid in grids:
        for factor in summary["suite"]["noise_factors"]:
            relative = "figures/returns_%s_noise_%s.svg" % (grid, noise_slug(factor))
            (output / relative).write_text(
                make_return_figure(curves, grid, factor, summary["suite"]["horizon"]),
                encoding="utf-8",
            )
            figure_paths[grid + ":" + str(float(factor))] = relative
    payload = json.dumps(
        {"summary": summary, "rows": rows, "figures": figure_paths}, allow_nan=False
    ).replace("</", "<\\/")
    grid_options = "".join('<option value="%s">%s</option>' % (g, g) for g in grids)
    factors = summary["suite"]["noise_factors"]
    default_factor = 1 if 1 in factors else factors[0]
    factor_options = "".join(
        '<option value="%g"%s>%g × actual</option>'
        % (f, " selected" if f == default_factor else "", f)
        for f in factors
    )
    page = (
        HTML.replace("__GRIDS__", grid_options)
        .replace("__FACTORS__", factor_options)
        .replace("__DATA__", payload)
    )
    (output / "index.html").write_text(page, encoding="utf-8")
    (output / "RESULTS.md").write_text(_results_markdown(summary), encoding="utf-8")


def _results_markdown(summary):
    lines = [
        "# Benchmark comparison results",
        "",
        "Generated from saved run records; all attempted runs are included.",
        "",
        "Runs: %d. Grid/map combinations: %d. Failed runs: %d. Runs with unsafe actions: %d. Runs with false certificates: %d. ES²-triggered runs: %d."
        % tuple(
            summary[k]
            for k in (
                "run_count",
                "map_count",
                "failed_runs",
                "runs_with_unsafe_actions",
                "runs_with_false_certificates",
                "es2_triggered_runs",
            )
        ),
        "",
        "| Grid | Method | Noise factor | Mean discounted return | SD across map means | Failure rate | ES² trigger rate |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for a in summary["aggregates"]:
        lines.append(
            "| %s | %s | %s | %.4f | %s | %.1f%% | %.1f%% |"
            % (
                a["grid"],
                a["method"],
                "—" if a["noise_factor"] is None else str(a["noise_factor"]),
                a["discounted_return_mean"],
                "—"
                if a["discounted_return_map_sd"] is None
                else "%.4f" % a["discounted_return_map_sd"],
                100 * a["failure_rate"],
                100 * a["es2_trigger_rate"],
            )
        )
    lines += [
        "",
        summary["aggregation"] + ".",
        "",
        summary["failure_accounting"] + ".",
        "",
        summary["reference"] + ".",
        "",
        summary["baseline"] + ".",
        "",
        summary["scope"] + ".",
        "",
        "ES² success means the stored auxiliary-policy successor test passed. All methods use natural stopping and matched total horizons. These source-ported development results do not establish a theoretical guarantee.",
        "",
        "Stopping policy: %s. Runs exhausting the horizon during exploration: %d; these are distinct from model failures."
        % ("natural", summary.get("budget_exhausted_exploration_runs", 0)),
        "",
        "[Interactive dashboard](index.html) · [Every run](runs.csv) · [Aggregates](aggregates.csv) · [Paired ES² versus basic differences](paired_es2_vs_basic.csv)",
        "",
    ]
    return "\n".join(lines)


HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Safe rover multi-map comparison</title><style>
body{font:16px/1.55 system-ui,sans-serif;color:#193448;background:#f3f6fa;max-width:1250px;margin:auto;padding:24px}h1{line-height:1.15}h2{font-size:1.15rem}.panel{background:white;border:1px solid #d9e2eb;border-radius:12px;padding:20px;margin:18px 0}.controls{display:flex;gap:24px;flex-wrap:wrap}select{font:inherit;padding:6px}table{border-collapse:collapse;width:100%;font-size:.87rem}th,td{border-bottom:1px solid #dce5ed;padding:8px;text-align:right}th:first-child,td:first-child{text-align:left}.scroll{overflow:auto}img{display:block;width:100%;height:auto}.note{border-left:4px solid #ba8439;padding-left:15px}a{color:#2368a1}.muted{color:#516777}.failed{color:#9a3636}code{overflow-wrap:anywhere}button{font:inherit;padding:5px}label{white-space:nowrap}
</style></head><body><h1>GP-Safety-Gym: reward, ES², and sensor-noise robustness</h1>
<p id="overview"></p><p><a href="RESULTS.md">Results notes</a> · <a href="runs.csv">Every run CSV</a> · <a href="aggregates.csv">Aggregates CSV</a> · <a href="paired_es2_vs_basic.csv">Paired ES² minus basic CSV</a> · <a href="summary.json">Full summary</a> · <a href="suite.json">Experiment suite</a></p>
<div class="panel"><h2>Project pipeline</h2><p>Known grid and initial certificates → noisy observations → reward / safety GPs → confidence history and safe-region checks → expanders or ES² successor test → online reward planning on certified actions → evaluate across matched maps, seeds, and noise assumptions.</p><p class="muted">Methods: <strong>safemdp</strong> samples transition uncertainty throughout; <strong>sno_basic</strong> switches after the declared accuracy criterion; <strong>sno_es2</strong> can also switch when its auxiliary-policy test passes; <strong>oracle</strong> knows the true maps.</p></div>
<div class="panel controls"><label>Grid <select id="grid">__GRIDS__</select></label><label>Assumed safety noise <select id="noise">__FACTORS__</select></label></div>
<div class="panel"><h2>Mean discounted return</h2><img id="curve" alt="Mean discounted return by method over the scheduled action budget"><p class="muted">Each condition uses the same maps, observation seeds, actual sensor noise, and total action budget. Policies can collect different observations. Failed prefixes remain flat for the unused budget, corresponding to zero additional reward credit.</p></div>
<div class="panel"><h2>Results for this condition</h2><div class="scroll"><table><thead><tr><th>Method</th><th>Runs / maps</th><th>Mean return</th><th>Map SD</th><th>Reference ratio</th><th>Failures</th><th>Unsafe runs</th><th>False certificates</th><th>ES² triggers</th><th>Exploration horizon limits</th><th>Mean exploration actions</th></tr></thead><tbody id="aggregates"></tbody></table></div><p class="muted">Observation replicates are averaged within each map before averaging maps. SD is across map means, not a confidence interval. All failures are included in mean return and ratio. The reference is an infinite-horizon discounted policy evaluated over the same finite rollout; its ratio is descriptive.</p></div>
<div class="panel"><h2>Individual runs and replays</h2><div class="scroll"><table><thead><tr><th>Method</th><th>Map seed</th><th>Observation seed</th><th>Return</th><th>Actions</th><th>Transition</th><th>Stop</th><th>Unsafe actions</th><th>Replay</th></tr></thead><tbody id="runs"></tbody></table></div></div>
<p class="note">This is a headless port of the released synthetic benchmark, not native legacy MuJoCo execution. Safety is attached to directed altitude differences; GPs update online. ES² checks auxiliary policy successors from every executable state. Physical execution is restricted to certified actions. The source profile and paper pseudocode differ; see docs/method.md. Pilot/validation runs are development evidence, not a frozen final evaluation.</p>
<script id="benchmark-data" type="application/json">__DATA__</script><script>
const data=JSON.parse(document.getElementById('benchmark-data').textContent),$=id=>document.getElementById(id);
const fmt=v=>v===null?'—':Number(v).toFixed(3),pct=v=>(100*v).toFixed(1)+'%';
function row(body,values,failed=false){const tr=document.createElement('tr');if(failed)tr.className='failed';for(const value of values){const td=document.createElement('td');if(typeof value==='object'&&value!==null){const a=document.createElement('a');a.href=value.href;a.textContent=value.text;td.append(a);}else td.textContent=value;tr.append(td);}body.append(tr);}
function update(){const grid=$('grid').value,factor=Number($('noise').value);const key=Object.keys(data.figures).find(k=>k.split(':')[0]===grid&&Number(k.split(':')[1])===factor);$('curve').src=data.figures[key];$('aggregates').replaceChildren();$('runs').replaceChildren();
for(const a of data.summary.aggregates.filter(a=>a.grid===grid&&(a.noise_factor===null||a.noise_factor===factor)))row($('aggregates'),[a.method,`${a.runs} / ${a.maps}`,fmt(a.discounted_return_mean),fmt(a.discounted_return_map_sd),fmt(a.reward_ratio_mean),pct(a.failure_rate),pct(a.unsafe_run_rate),pct(a.false_certificate_run_rate),pct(a.es2_trigger_rate),pct(a.exploration_budget_exhaustion_rate||0),fmt(a.exploration_actions_mean)]);
for(const r of data.rows.filter(r=>r.grid===grid&&(r.noise_factor===null||r.noise_factor===factor)))row($('runs'),[r.method,r.map_seed,r.observation_seed,fmt(r.discounted_return),`${r.physical_actions} / ${r.scheduled_horizon}`,r.transition_reason||'none',r.stop_reason,r.unsafe_actions,{href:r.replay,text:'Open'}],r.failed);}
$('overview').textContent=`${data.summary.run_count} runs across ${data.summary.map_count} grid/map combinations. Failed runs: ${data.summary.failed_runs}; runs with unsafe actions: ${data.summary.runs_with_unsafe_actions}; runs with false certificates: ${data.summary.runs_with_false_certificates}; ES²-triggered runs: ${data.summary.es2_triggered_runs}. Each scheduled rollout has ${data.summary.suite.horizon} actions.`;
$('overview').textContent+=` Stopping policy: natural; horizons exhausted while exploring: ${data.summary.budget_exhausted_exploration_runs||0}.`;
$('grid').addEventListener('change',update);$('noise').addEventListener('change',update);update();
</script></body></html>"""
