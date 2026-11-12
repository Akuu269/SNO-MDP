"""Exportable diagnostic figures and local reports; no controller imports."""

from contextlib import contextmanager
from html import escape
import os
import tempfile

import numpy as np


@contextmanager
def _plot_style():
    previous = {k: os.environ.get(k) for k in ("MPLCONFIGDIR", "XDG_CACHE_HOME")}
    with tempfile.TemporaryDirectory(prefix="snomdp-diagnostics-") as cache:
        for key, value in previous.items():
            if value is None:
                os.environ[key] = cache
        try:
            import matplotlib as mpl

            with mpl.rc_context(
                {"svg.hashsalt": "snomdp-safety", "font.family": "DejaVu Sans"}
            ):
                yield
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def _save(figure, output):
    figure.savefig(output / "diagnostics.png", dpi=150)
    figure.savefig(output / "diagnostics.svg", metadata={"Date": None})


def _run_plot(output, summary, coverage, calibration, unsafe):
    with _plot_style():
        from matplotlib.figure import Figure

        fig = Figure(figsize=(12, 8), layout="constrained")
        axes = fig.subplots(2, 2)
        for subset, label, color in (
            ("available_nonstay", "All valid moving actions", "#2c80b6"),
            ("executable_nonstay", "Executable moving actions", "#9353af"),
        ):
            rows = [r for r in coverage if r["subset"] == subset]
            axes[0, 0].plot(
                [r["step"] for r in rows],
                [r["coverage"] for r in rows],
                label=label,
                color=color,
            )
            rows = [r for r in calibration if r["subset"] == subset]
            axes[0, 1].plot(
                [r["multiplier"] for r in rows],
                [r["coverage"] for r in rows],
                marker="o",
                label=label,
                color=color,
            )
        axes[0, 0].set(
            xlabel="Completed physical actions",
            ylabel="Pointwise coverage",
            ylim=(0, 1.02),
            title="Current transition confidence bands",
        )
        axes[0, 1].set(
            xlabel="Counterfactual confidence multiplier",
            ylabel="Pooled edge-frame coverage",
            ylim=(0, 1.02),
            title="Fixed trajectory / posterior — no policy reruns",
        )
        axes[0, 1].axvline(
            summary["safety_multiplier"], color="#667788", linestyle="--", linewidth=1
        )
        steps = [r for r in coverage if r["subset"] == "available_nonstay"]
        for key, label, color in (
            ("false_current_lcb_supported", "Incorrect, current LCB passes", "#c55435"),
            ("false_retained", "Incorrect, retained older certificate", "#9353af"),
        ):
            axes[1, 0].plot(
                [r["step"] for r in steps],
                [r[key] for r in steps],
                label=label,
                color=color,
            )
        axes[1, 0].set(
            xlabel="Completed physical actions",
            ylabel="Incorrect executable actions",
            title="Two sources of incorrect certificates",
            ylim=(0, None),
        )
        ax = axes[1, 1]
        if unsafe:
            x = np.arange(len(unsafe))
            mean = np.array([r["mean"] for r in unsafe])
            radius = summary["safety_multiplier"] * np.array([r["std"] for r in unsafe])
            ax.errorbar(
                x,
                mean,
                yerr=radius,
                fmt="o",
                capsize=4,
                label="Pre-action GP mean ± multiplier × STD",
                color="#2c80b6",
            )
            ax.scatter(
                x,
                [r["true_g"] for r in unsafe],
                marker="x",
                color="#bd382b",
                label="True directed difference",
                zorder=3,
            )
            ax.axhline(
                unsafe[0]["threshold"],
                linestyle="--",
                color="#555555",
                label="Safety threshold",
            )
            ax.set_xticks(x, [str(r["execution_step"]) for r in unsafe], rotation=45)
            ax.set(
                xlabel="Unsafe execution step",
                ylabel="Directed altitude difference",
                title="Bounds recorded before each unsafe action",
            )
        else:
            ax.text(
                0.5,
                0.5,
                "No unsafe actions in this run.\nThis is not a safety guarantee.",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Unsafe execution traces")
        for ax in axes.ravel():
            ax.grid(alpha=0.2)
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=8)
        fig.suptitle(f"{summary['method']} · evaluator-only safety diagnostics")
        _save(fig, output)


def _table(headers, rows):
    return (
        "<div class='scroll'><table><thead><tr>"
        + "".join("<th>" + escape(str(v)) + "</th>" for v in headers)
        + "</tr></thead><tbody>"
        + "".join(
            "<tr>" + "".join("<td>" + escape(str(v)) + "</td>" for v in row) + "</tr>"
            for row in rows
        )
        + "</tbody></table></div>"
    )


def _page(title, body):
    return (
        """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>"""
        + escape(title)
        + """</title><style>
body{font:16px/1.6 system-ui,sans-serif;max-width:1200px;margin:auto;padding:24px;background:#f3f6fa;color:#193448}h1{line-height:1.2}section{padding:20px;margin:18px 0;border:1px solid #d9e2eb;background:white;border-radius:10px}table{border-collapse:collapse;width:100%;font-size:.9rem}td,th{text-align:left;padding:8px;border-bottom:1px solid #d9e2eb;white-space:nowrap}.scroll{overflow:auto}img{width:100%;height:auto}a{color:#2368a1}.note{border-left:4px solid #bd843b;padding-left:16px}
</style><body><h1>"""
        + escape(title)
        + "</h1>"
        + body
        + "</body></html>"
    )


def _percent(value):
    return "—" if value is None else f"{100 * value:.2f}%"


def write_run_report(output, summary, coverage, calibration, unsafe, plots):
    if plots:
        _run_plot(output, summary, coverage, calibration, unsafe)
    else:
        for ext in ("png", "svg"):
            (output / ("diagnostics." + ext)).unlink(missing_ok=True)
    title = "Safety diagnostics: " + summary["method"]
    coverage_rows = [
        [
            name.replace("_", " "),
            row["count"],
            _percent(row["coverage"]),
            row["below_lower"],
            row["above_upper"],
        ]
        for name, row in summary["coverage"].items()
    ]
    unsafe_rows = [
        [
            r["execution_step"],
            str(r["source"]),
            r["action"],
            str(r["successor"]),
            f"{r['true_g']:.6f}",
            f"{r['lower']:.6f}",
            r["first_accepted_step"],
            "—" if r["acceptance_lower"] is None else f"{r['acceptance_lower']:.6f}",
            r["support"],
        ]
        for r in unsafe
    ]
    headers = [
        "Execution",
        "Source",
        "Action",
        "Successor",
        "True g",
        "Current LCB",
        "Accepted at",
        "Acceptance LCB",
        "Execution support",
    ]
    overview = (
        f"{summary['analyzed_frames']} posterior frames; {summary['executed_actions']} physical actions; "
        f"{summary['unsafe_actions']} unsafe actions. Unsafe actions with a passing current LCB: "
        f"{summary['unsafe_current_lcb_supported']}; relying on an older certificate: {summary['unsafe_retained_certificate']}."
    )
    note = summary["interpretation"]
    body = "<p>" + escape(overview) + "</p><p class='note'>" + escape(note) + "</p>"
    body += "<p>Stay and invalid boundary actions are excluded from coverage. Lower misses mean the true difference is below the model's lower bound. Certificates accepted elsewhere can become executable later.</p>"
    if summary["skipped_failed_update_steps"]:
        body += (
            "<p>Failed posterior updates excluded: "
            + escape(str(summary["skipped_failed_update_steps"]))
            + ". Earlier executed actions remain included.</p>"
        )
    body += (
        "<section><h2>Coverage at the original multiplier</h2>"
        + _table(
            ["Subset", "Edge-frame count", "Coverage", "Below LCB", "Above UCB"],
            coverage_rows,
        )
        + "</section>"
    )
    if plots:
        body += '<section><h2>Diagnostic figures</h2><img src="diagnostics.png" alt="Coverage, counterfactual intervals, incorrect certificates, and unsafe action bounds"><a href="diagnostics.svg">Export SVG</a></section>'
    body += (
        "<section><h2>Every unsafe action</h2>"
        + (
            _table(headers, unsafe_rows)
            if unsafe
            else "<p>No unsafe actions were executed.</p>"
        )
        + "</section>"
    )
    body += """<section><h2>Inspect the records</h2><p><a href="coverage_by_step.csv">Coverage by step</a> · <a href="coverage_by_multiplier.csv">Fixed-trajectory multiplier comparison</a> · <a href="executed_actions.csv">All execution traces</a> · <a href="unsafe_actions.csv">Unsafe traces</a> · <a href="certificate_history.json.gz">First-acceptance history</a> · <a href="manifest.json">Verified inputs and analysis provenance</a> · <a href="../index.html">Original replay</a></p></section>"""
    (output / "index.html").write_text(_page(title, body), encoding="utf-8")
    lines = [
        "# " + title,
        "",
        overview,
        "",
        note,
        "",
        "Stay and unavailable boundary actions are excluded. Initial certificates are distinguished from posterior acceptance. The initial observation is step 0; execution n uses the posterior at decision n−1.",
        "",
        "| Subset | Edge-frame count | Coverage | Below LCB | Above UCB |",
        "|---|---:|---:|---:|---:|",
    ]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in coverage_rows)
    lines += [
        "",
        "## Unsafe actions",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in unsafe_rows)
    if not unsafe:
        lines += [
            "",
            "No unsafe actions were executed; this does not establish a guarantee.",
        ]
    lines += [
        "",
        "[Local report](index.html) · [Original replay](../index.html) · [Analysis provenance](manifest.json)",
        "",
    ]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def write_study_report(output, summary, runs, plots):
    aggregates = summary["aggregates"]
    if plots:
        with _plot_style():
            from matplotlib.figure import Figure

            fig = Figure(figsize=(12, 5), layout="constrained")
            left, right = fig.subplots(1, 2)
            x = np.arange(len(aggregates))
            labels = [
                f"{a['method']}\n{a['noise_factor']:g}× noise" for a in aggregates
            ]
            left.bar(
                x - 0.18,
                [
                    np.nan
                    if a["available_coverage_mean"] is None
                    else a["available_coverage_mean"]
                    for a in aggregates
                ],
                width=0.36,
                label="All valid moving actions",
                color="#2c80b6",
            )
            left.bar(
                x + 0.18,
                [
                    np.nan
                    if a["executable_coverage_mean"] is None
                    else a["executable_coverage_mean"]
                    for a in aggregates
                ],
                width=0.36,
                label="Executable moving actions",
                color="#9353af",
            )
            left.set(
                ylabel="Mean pointwise coverage",
                ylim=(0, 1.02),
                title="Replicates averaged within maps, then across maps",
            )
            current = [a["unsafe_current_lcb_supported"] for a in aggregates]
            retained = [a["unsafe_retained_certificate"] for a in aggregates]
            right.bar(x, current, label="Current LCB passes", color="#c55435")
            right.bar(
                x,
                retained,
                bottom=current,
                label="Retained certificate",
                color="#9353af",
            )
            right.set(
                ylabel="Unsafe actions, summed across runs",
                title="Certificate support before unsafe execution",
            )
            for ax in (left, right):
                ax.set_xticks(x, labels, rotation=45, ha="right")
                ax.grid(axis="y", alpha=0.2)
                ax.legend(fontsize=8)
            _save(fig, output)
    else:
        for ext in ("png", "svg"):
            (output / ("diagnostics." + ext)).unlink(missing_ok=True)
    headers = [
        "Grid",
        "Method",
        "Noise",
        "Runs/maps",
        "All-action coverage",
        "Executable coverage",
        "Unsafe actions",
        "Current LCB / retained",
    ]
    table = [
        [
            a["grid"],
            a["method"],
            a["noise_factor"],
            f"{a['runs']}/{a['maps']}",
            _percent(a["available_coverage_mean"]),
            _percent(a["executable_coverage_mean"]),
            a["unsafe_actions"],
            f"{a['unsafe_current_lcb_supported']}/{a['unsafe_retained_certificate']}",
        ]
        for a in aggregates
    ]
    title = "Study safety diagnostics"
    overview = f"{summary['learner_runs']} learner runs; {summary['unsafe_actions']} unsafe actions. Oracle runs have no learned bands and are excluded."
    body = (
        "<p>"
        + escape(overview)
        + "</p><p class='note'>"
        + escape(summary["aggregation"] + " " + summary["scope"])
        + "</p>"
    )
    body += "<section>" + _table(headers, table) + "</section>"
    if plots:
        body += '<section><img src="diagnostics.png" alt="Coverage and unsafe certificate support by method and noise assumption"><a href="diagnostics.svg">Export SVG</a></section>'
    body += (
        "<section><h2>Individual diagnostic reports</h2><ul>"
        + "".join(
            '<li><a href="'
            + escape("../" + row["report"], quote=True)
            + '">'
            + escape(
                f"{row['method']} · {row['grid']} · map {row['map_seed']} · observation {row['observation_seed']} · noise {row['noise_factor']}"
            )
            + "</a></li>"
            for row in runs
        )
        + '</ul><p><a href="runs.csv">Run metrics</a> · <a href="aggregates.csv">Aggregates</a> · <a href="manifest.json">Provenance</a> · <a href="../index.html">Original study</a></p></section>'
    )
    (output / "index.html").write_text(_page(title, body), encoding="utf-8")
    lines = [
        "# " + title,
        "",
        overview,
        "",
        summary["aggregation"],
        "",
        summary["scope"],
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in table)
    lines += [
        "",
        "[Local report](index.html) · [Run metrics](runs.csv) · [Original study](../index.html)",
        "",
    ]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
