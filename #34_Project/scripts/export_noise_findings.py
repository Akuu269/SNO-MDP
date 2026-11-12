"""Posthoc figures/discussion only; never changes frozen policies or settings."""

import hashlib
from html import escape
from pathlib import Path
import re

from sno_mdp.artifacts import read_json, save_json
from sno_mdp.diagnostic_report import _plot_style


ROOT = Path(__file__).resolve().parent.parent


def main():
    study = ROOT / "results/final"
    summary = read_json(study / "summary.json")
    rows = read_json(study / "runs.json")
    aggregates = summary["aggregates"]
    groups = {}
    for a in aggregates:
        if a["method"] != "oracle":
            groups.setdefault(a["profile"] + ":" + a["method"], []).append(a)
    with _plot_style():
        from matplotlib.figure import Figure

        fig = Figure(figsize=(13, 8), layout="constrained")
        axes = fig.subplots(2, 2)
        for label, values in groups.items():
            values.sort(key=lambda a: a["noise_factor"])
            for ax, key, title in (
                (
                    axes[0, 0],
                    "reward_ratio_mean",
                    "Mean finite-rollout reference ratio (descriptive)",
                ),
                (axes[0, 1], "unsafe_actions_mean", "Mean unsafe physical actions"),
                (axes[1, 0], "failed_mean", "Fraction of failed runs"),
                (axes[1, 1], "exploration_actions_mean", "Mean exploration actions"),
            ):
                ax.plot(
                    [a["noise_factor"] for a in values],
                    [a[key] for a in values],
                    marker="o",
                    label=label,
                )
                ax.set(title=title, xlabel="Assumed / actual altitude-sensor noise STD")
                ax.set_xticks([0.5, 1, 2])
                ax.grid(alpha=0.2)
        axes[0, 0].legend(fontsize=7)
        fig.suptitle(
            "Frozen held-out noise study: six maps, two observation seeds per map"
        )
        for ext in ("png", "svg"):
            fig.savefig(
                study / "figures" / ("noise_robustness." + ext),
                **({"dpi": 140} if ext == "png" else {"metadata": {"Date": None}}),
            )
    current = [r for r in rows if r["profile"] == "revalidated"]
    control = [r for r in rows if r["profile"] == "confidence"]
    source = [r for r in rows if r["profile"] == "source" and r["method"] == "sno_es2"]

    def unsafe(members):
        return sum(r["unsafe_actions"] for r in members)

    if unsafe(control) == unsafe(current) == 0:
        safety = "Both selected-confidence retained ES² and revalidated ES² recorded zero unsafe actions. This evaluation does not establish an additional safety benefit from revalidation beyond the confidence change."
    else:
        safety = (
            f"Selected-confidence retained ES² recorded {unsafe(control)} unsafe actions; "
            f"revalidated ES² recorded {unsafe(current)}. This is the matched-confidence comparison needed to separate certificate handling from confidence tuning."
        )
    nominal = [a for a in aggregates if a["noise_factor"] == 1]
    with _plot_style():
        fig = Figure(figsize=(13, 5), layout="constrained")
        axes = fig.subplots(1, 2)
        for ax, key, title in (
            (axes[0], "inference_seconds", "GP inference and certificate update"),
            (axes[1], "decision_seconds", "Policy decision and planning"),
        ):
            ax.bar(
                range(len(nominal)),
                [1000 * a[key + "_mean"] for a in nominal],
                yerr=[1000 * a[key + "_map_sd"] for a in nominal],
                capsize=4,
                color="#397e9e",
            )
            ax.set_xticks(
                range(len(nominal)),
                [a["profile"] + ":" + a["method"] for a in nominal],
                rotation=30,
                ha="right",
                fontsize=8,
            )
            ax.set(title=title, ylabel="Mean milliseconds per frame ± map SD")
            ax.grid(axis="y", alpha=0.2)
        fig.suptitle(
            "Matched-noise computation; oracle one-time policy construction excluded"
        )
        fig.savefig(study / "figures/computation.png", dpi=140)
        fig.savefig(study / "figures/computation.svg", metadata={"Date": None})
    values = {a["profile"] + ":" + a["method"]: a for a in nominal}
    baseline = values["source:sno_es2"]
    variant = values["revalidated:sno_es2"]
    control_value = values["confidence:sno_es2"]
    discussion = [
        "# Empirical findings and interpretation",
        "",
        "All settings were frozen before the held-out evaluation. These are descriptive simulation findings, not safety guarantees or significance claims.",
        "",
        f"- Completed {summary['completed_runs']} final cases on six maps with two observation seeds and three noise assumptions.",
        f"- Source c=2 ES²: {unsafe(source)} unsafe actions across {len(source)} cases.",
        f"- Selected-confidence retained ES²: {unsafe(control)} across {len(control)} cases.",
        f"- Revalidated ES²: {unsafe(current)} across {len(current)} cases; {sum(r['failed'] for r in current)} failures.",
        "",
        safety,
        "",
        f"At matched noise, source ES² mean return is {baseline['discounted_return_mean']:.3f}; "
        f"selected-confidence retained ES² is {control_value['discounted_return_mean']:.3f}; "
        f"revalidated ES² is {variant['discounted_return_mean']:.3f}. "
        "Compare map SD and paired differences rather than treating these means as universal improvements.",
        "",
        "The development-selected confidence setting does not maximize held-out nominal reward. "
        "The source c=2 ES² policy achieves higher nominal mean reward here, with more unsafe runs; "
        "the conservative variants give up reward while reducing observed violations. This unfavorable reward result is retained rather than retuning on held-out outcomes.",
        "",
        "The earlier diagnostic mechanism remains informative: a practical confidence error can admit an action, and monotone retention can allow later execution despite corrected beliefs. "
        "That observation does not refute the paper’s theorem under its stated assumptions; this implementation does not establish a uniform confidence event.",
        "",
        "[Noise robustness figure](../results/final/figures/noise_robustness.png) · "
        "[Inference and decision cost](../results/final/figures/computation.png) · "
        "[Every final case](../results/final/runs.csv) · [Full report](report.md)",
        "",
    ]
    (ROOT / "report/empirical_findings.md").write_text(
        "\n".join(discussion), encoding="utf-8"
    )
    for replay in read_json(study / "selected_replays.json"):
        path = study / replay["path"]
        case = read_json(path.parent / "summary.json")
        label = (
            "Privileged true-field oracle; no learned confidence bands."
            if case["method"] == "oracle"
            else f"Certificate mode: {case['certificate_mode']}; "
            f"safety multiplier: {case['safety_confidence_multiplier']:g}."
        )
        page = re.sub(r'<p id="certificate-profile">.*?</p>', "", path.read_text())
        label_html = '<p id="certificate-profile">' + escape(label)
        if (path.parent / "safety/index.html").exists():
            label_html += (
                ' · <a href="safety/index.html">Certificate and coverage audit</a>'
            )
        label_html += "</p>"
        page = page.replace('<p id="scope"></p>', '<p id="scope"></p>' + label_html)
        path.write_text(page, encoding="utf-8")
    save_json(
        study / "report_artifacts.json",
        {
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "runs_sha256": hashlib.sha256(
                (study / "runs.json").read_bytes()
            ).hexdigest(),
            "summary_sha256": hashlib.sha256(
                (study / "summary.json").read_bytes()
            ).hexdigest(),
            "scope": "Posthoc reporting; no source, settings, case inventory, or policy changed.",
        },
    )


if __name__ == "__main__":
    main()
