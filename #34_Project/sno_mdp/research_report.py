"""Scientific figures and reports from completed controlled policy reruns."""

from collections import defaultdict
from html import escape
from pathlib import Path

import numpy as np

from .artifacts import read_json, save_json
from .diagnostic_report import _page, _plot_style, _table
from .research import csv_write


def paired_rows(rows):
    lookup = {
        (
            r["map_seed"],
            r["observation_seed"],
            r["horizon"],
            r["noise_factor"],
            r["profile"],
            r["method"],
        ): r
        for r in rows
    }
    pairs = []
    for row in rows:
        if row["method"] == "oracle":
            continue
        reference = (
            ("source", "sno_basic")
            if row["profile"] == "source" and row["method"] == "sno_es2"
            else ("source", "sno_es2")
        )
        if (row["profile"], row["method"]) == reference:
            continue
        key = (
            row["map_seed"],
            row["observation_seed"],
            row["horizon"],
            row["noise_factor"],
            *reference,
        )
        if key not in lookup:
            continue
        baseline = lookup[key]
        pairs.append(
            {
                "map_seed": row["map_seed"],
                "observation_seed": row["observation_seed"],
                "horizon": row["horizon"],
                "noise_factor": row["noise_factor"],
                "comparison": row["profile"]
                + ":"
                + row["method"]
                + " minus "
                + ":".join(reference),
                "return_difference": row["discounted_return"]
                - baseline["discounted_return"],
                "unsafe_action_difference": row["unsafe_actions"]
                - baseline["unsafe_actions"],
                "failed": row["failed"],
                "baseline_failed": baseline["failed"],
            }
        )
    return pairs


def export_research(study):
    study = Path(study)
    summary, rows = read_json(study / "summary.json"), read_json(study / "runs.json")
    curves = read_json(study / "curves.json.gz")
    spec = read_json(study / "suite.json")
    figures = study / "figures"
    figures.mkdir(exist_ok=True)
    pairs = paired_rows(rows)
    if pairs:
        csv_write(study / "paired_differences.csv", pairs)
    with _plot_style():
        from matplotlib.figure import Figure

        for metric, ylabel, filename in (
            ("return", "Mean discounted return", "returns"),
            ("cells", "Mean executable cells", "region_growth"),
            (
                "false_certificates",
                "Mean incorrect executable certificates",
                "certificate_errors",
            ),
        ):
            h = max(spec["horizons"])
            factors = spec["noise_factors"]
            fig = Figure(figsize=(5 * len(factors), 4.5), layout="constrained")
            axes = np.atleast_1d(fig.subplots(1, len(factors)))
            for ax, factor in zip(axes, factors):
                groups = defaultdict(list)
                for curve in curves:
                    if curve["horizon"] == h and curve["noise_factor"] in (
                        None,
                        factor,
                    ):
                        groups[curve["profile"] + ":" + curve["method"]].append(
                            curve[metric]
                        )
                for label, values in groups.items():
                    ax.plot(
                        np.arange(h + 1),
                        np.mean(values, axis=0),
                        label=label,
                        linewidth=1.7,
                    )
                ax.set(
                    xlabel="Scheduled physical action budget",
                    ylabel=ylabel,
                    title=f"Assumed/actual noise {factor:g}× · H={h}",
                )
                ax.grid(alpha=0.2)
                ax.legend(fontsize=7)
            fig.savefig(figures / (filename + ".png"), dpi=140)
            fig.savefig(figures / (filename + ".svg"), metadata={"Date": None})
        h = max(spec["horizons"])
        selected = [
            a
            for a in summary["aggregates"]
            if a["horizon"] == h and a["noise_factor"] in (1, None)
        ]
        fig = Figure(figsize=(13, 7), layout="constrained")
        axes = fig.subplots(2, 2)
        labels = [a["profile"] + "\n" + a["method"] for a in selected]
        for ax, key, title, scale in (
            (
                axes[0, 0],
                "unsafe_run_fraction",
                "Fraction of runs with unsafe actions",
                1,
            ),
            (axes[0, 1], "failed_mean", "Failure fraction (including region loss)", 1),
            (axes[1, 0], "exploration_actions_mean", "Mean exploration actions", 1),
            (
                axes[1, 1],
                "decision_seconds_mean",
                "Mean planning time per decision (ms)",
                1000,
            ),
        ):
            ax.bar(
                np.arange(len(selected)),
                [a[key] * scale for a in selected],
                color="#397e9e",
            )
            ax.set_xticks(
                np.arange(len(selected)), labels, rotation=45, ha="right", fontsize=7
            )
            ax.set_title(title)
            ax.grid(axis="y", alpha=0.2)
        fig.suptitle(
            f"Matched noise · H={h}; inference measured separately in aggregates.csv"
        )
        fig.savefig(figures / "outcomes.png", dpi=140)
        fig.savefig(figures / "outcomes.svg", metadata={"Date": None})
        if pairs:
            groups = defaultdict(lambda: defaultdict(list))
            for pair in pairs:
                if pair["noise_factor"] == 1 and pair["horizon"] == h:
                    groups[pair["comparison"]][pair["map_seed"]].append(
                        pair["return_difference"]
                    )
            if groups:
                fig = Figure(figsize=(10, 5), layout="constrained")
                ax = fig.subplots()
                means, errors = [], []
                for values in groups.values():
                    v = [np.mean(replicates) for replicates in values.values()]
                    means.append(np.mean(v))
                    errors.append(np.std(v, ddof=1) if len(v) > 1 else 0)
                ax.bar(
                    np.arange(len(means)),
                    means,
                    yerr=errors,
                    capsize=4,
                    color="#9353af",
                )
                ax.axhline(0, color="#555555", linewidth=1)
                ax.set_xticks(
                    np.arange(len(means)),
                    list(groups),
                    rotation=25,
                    ha="right",
                    fontsize=8,
                )
                ax.set(
                    ylabel="Paired return difference: map mean ± map SD",
                    title="Matched maps and observations · descriptive, no significance claim",
                )
                fig.savefig(figures / "paired_returns.png", dpi=140)
                fig.savefig(figures / "paired_returns.svg", metadata={"Date": None})
        # Multiple budgets are compared as distinct experimental conditions.
        if len(spec["horizons"]) > 1:
            fig = Figure(figsize=(10, 5), layout="constrained")
            ax = fig.subplots()
            groups = defaultdict(list)
            for a in summary["aggregates"]:
                if a["noise_factor"] == 1:
                    groups[a["profile"] + ":" + a["method"]].append(a)
            for name, values in groups.items():
                values.sort(key=lambda a: a["horizon"])
                ax.plot(
                    [a["horizon"] for a in values],
                    [a["exploration_exhausted_mean"] for a in values],
                    marker="o",
                    label=name,
                )
            ax.set(
                xlabel="Total horizon",
                ylabel="Fraction exhausting budget in exploration",
                ylim=(-0.02, 1.02),
                title="Stopping behavior across predeclared budgets",
            )
            ax.legend(fontsize=8)
            fig.savefig(figures / "stopping.png", dpi=140)
            fig.savefig(figures / "stopping.svg", metadata={"Date": None})
    # Save a few complete replays; every other case retains its full raw log.
    from .visualization import write_visualization

    selected_replays = []
    for row in rows:
        if (
            row["map_seed"] == spec["map_seeds"][0]
            and row["observation_seed"] == spec["observation_seeds"][0]
            and row["horizon"] == max(spec["horizons"])
            and row["noise_factor"] in (None, 1)
            and row["profile"] in ("source", "revalidated", "oracle")
            and row["method"] in ("sno_es2", "oracle")
            and row["stop_reason"] != "setup_failure"
        ):
            directory = study / row["directory"]
            payload = {
                **read_json(directory / "map.json"),
                "frames": read_json(directory / "trajectory.json.gz"),
                "summary": read_json(directory / "summary.json"),
            }
            write_visualization(directory / "index.html", payload)
            selected_replays.append(
                {
                    "name": row["profile"] + ":" + row["method"],
                    "path": row["directory"] + "/index.html",
                }
            )
    save_json(study / "selected_replays.json", selected_replays)
    headers = [
        "Profile",
        "Method",
        "Noise",
        "H",
        "Runs/maps",
        "Return mean",
        "Map SD",
        "Unsafe fraction",
        "Failure fraction",
        "ES² fraction",
    ]
    table = []
    for a in summary["aggregates"]:
        sd = a["discounted_return_map_sd"]
        table.append(
            [
                a["profile"],
                a["method"],
                a["noise_factor"],
                a["horizon"],
                f"{a['runs']}/{a['maps']}",
                f"{a['discounted_return_mean']:.3f}",
                "—" if sd is None else f"{sd:.3f}",
                f"{a['unsafe_run_fraction']:.3f}",
                f"{a['failed_mean']:.3f}",
                f"{a['es2_triggered_mean']:.3f}",
            ]
        )
    title = summary["stage"].capitalize() + " controlled experiment"
    note = f"{summary['completed_runs']}/{summary['planned_runs']} planned runs; {summary['maps']} independent maps; {summary['failed_runs']} failures; {summary['unsafe_actions']} unsafe actions. All failed prefixes included with zero additional reward credit. Error bars/SD describe map variability, not confidence intervals. No safety guarantee or exact published-figure reproduction is claimed."
    lines = [
        "# " + title,
        "",
        note,
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in table)
    lines += [
        "",
        "[Interactive report](index.html) · [Every case](runs.csv) · [Aggregates](aggregates.csv)",
        "",
    ]
    (study / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    body = "<p class='note'>" + escape(note) + "</p>" + _table(headers, table)
    body += (
        "<p>"
        + " · ".join(
            '<a href="'
            + escape(r["path"], quote=True)
            + '">'
            + escape(r["name"])
            + "</a>"
            for r in selected_replays
        )
        + "</p>"
    )
    for image in sorted(figures.glob("*.png")):
        body += (
            "<section><h2>"
            + escape(image.stem.replace("_", " ").title())
            + '</h2><img src="figures/'
            + image.name
            + '" alt="'
            + image.stem
            + '"></section>'
        )
    body += '<p><a href="runs.csv">Every planned case</a> · <a href="aggregates.csv">Map-level aggregates</a> · <a href="RESULTS.md">Results Markdown</a></p>'
    (study / "index.html").write_text(_page(title, body), encoding="utf-8")
