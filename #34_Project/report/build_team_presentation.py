"""Build the professor deck for three slide presenters; Member 4 uses the UI.

Run from #34_Project: python3 report/build_team_presentation.py
Build dependencies: the existing matplotlib/Pillow environment and python-pptx.
"""

from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from functools import lru_cache
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import re
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "snomdp-deck-mpl")
)
os.environ.setdefault(
    "XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "snomdp-deck-cache")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, patches
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image, ImageFont
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE
from pptx.util import Inches, Pt

from member3_talk import TALK, presenter_note, read_speeches, write_rehearsal


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report"
ASSETS = REPORT / "presentation_assets"
WIDTH, HEIGHT = 16, 9
INK, TEAL, MUTED = "#17343d", "#087e83", "#627c85"
PALE, GOLD, RED = "#eaf4f3", "#fbefda", "#bb4b46"
LINE = "#dce7e9"
plt.rcParams["font.family"] = "Arial"


def read(path):
    return json.loads(Path(path).read_text())


@dataclass
class Slide:
    title: str
    member: int
    seconds: int
    notes: str
    elements: list = field(default_factory=list)
    backup: bool = False
    cue: str = ""


@lru_cache(None)
def font(size, bold):
    path = font_manager.findfont(
        font_manager.FontProperties(family="Arial", weight="bold" if bold else "normal")
    )
    return ImageFont.truetype(path, round(size * 4))


def wrap(text, width, size, bold=False):
    result = []
    for paragraph in text.split("\n"):
        words, line = paragraph.split(), ""
        for word in words:
            trial = (line + " " + word).strip()
            if line and font(size, bold).getlength(trial) / 4 > width * 72:
                result.append(line)
                line = word
            else:
                line = trial
        result.append(line)
    return result


def text(slide, value, x, y, w, size=22, color=INK, bold=False, link=None):
    lines = wrap(value, w, size, bold)
    line_height = size * 1.25 / 72
    element = dict(
        kind="text",
        text="\n".join(lines),
        x=x,
        y=y,
        w=w,
        h=line_height * len(lines) + 0.06,
        size=size,
        color=color,
        bold=bold,
        link=link,
        line_height=line_height,
    )
    slide.elements.append(element)
    return element["h"]


def rect(slide, x, y, w, h, color=PALE, edge=None, circle=False):
    slide.elements.append(
        dict(kind="rect", x=x, y=y, w=w, h=h, color=color, edge=edge, circle=circle)
    )


def picture(slide, path, x, y, w, h):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    iw, ih = Image.open(path).size
    scale = min(w / iw, h / ih)
    slide.elements.append(
        dict(
            kind="image",
            path=path,
            x=x + (w - iw * scale) / 2,
            y=y + (h - ih * scale) / 2,
            w=iw * scale,
            h=ih * scale,
        )
    )


def card(slide, title, body, x, y, w, h=3.5, color=PALE):
    rect(slide, x, y, w, h, color)
    text(slide, title, x + 0.25, y + 0.22, w - 0.5, 22, TEAL, True)
    text(slide, body, x + 0.25, y + 1.0, w - 0.5, 21)


def callout(slide, value, y=7.15, color=PALE, size=20):
    rect(slide, 0.7, y, 14.6, 0.86, color)
    text(slide, value, 0.95, y + 0.16, 14.1, size)


def base(title, member, seconds, notes, cue="", backup=False):
    s = Slide(title, member, seconds, notes, backup=backup, cue=cue)
    rect(s, 0, 0, WIDTH, HEIGHT, "#ffffff")
    rect(s, 0, 0, 0.13, HEIGHT, TEAL)
    text(s, "PAPER #34  /  SNO-MDP", 0.7, 0.32, 7, 12, TEAL, True)
    badge = (
        "SUPPORTING EVIDENCE"
        if backup
        else {
            1: "01 / PROBLEM & SCOPE",
            2: "02 / METHOD",
            3: "03 / EXPERIMENTAL RESULTS",
        }[member]
    )
    text(s, badge, 11.1, 0.32, 4.2, 12, MUTED, True)
    text(s, title, 0.7, 0.94, 14.6, 33, INK, True)
    rect(s, 0.7, 8.34, 14.6, 0.012, LINE)
    text(
        s,
        "Our frozen study: results/final  |  Six maps, two observation seeds"
        if member == 3 and not backup
        else "Wachi & Sui · ICML 2020  |  Source-audited SNO-MDP study",
        0.7,
        8.53,
        12.9,
        11,
        MUTED,
    )
    return s


PROFILES = ("source", "confidence", "revalidated")
COLORS = {
    "source": "#087e83",
    "confidence": "#b17c24",
    "revalidated": "#7156a5",
    "basic": "#81919a",
    "oracle": "#283d47",
}
LABELS = {
    "source": "ES2 retained · c=2",
    "confidence": "ES2 retained · c=4",
    "revalidated": "ES2 revalidated · c=4",
    "basic": "Basic SNO-MDP · c=2",
    "oracle": "Oracle reference",
}


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def chart_style(ax):
    ax.set_facecolor("#ffffff")
    ax.spines[["top", "right"]].set_visible(False)
    for name in ("bottom", "left"):
        ax.spines[name].set_color("#bdcdd1")
    ax.grid(axis="y", color="#e4ecee", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=16, colors=INK, length=3)


def result_charts(summary, runs, assets=None):
    """Presentation graphs from recorded cases; repeats first, independent maps second."""
    import numpy as np

    assets = Path(assets) if assets is not None else ASSETS
    assets.mkdir(parents=True, exist_ok=True)

    aggregates = {
        (a["profile"], a["method"], a["noise_factor"]): a for a in summary["aggregates"]
    }
    conditions = [
        ("basic", "source", "sno_basic", 1),
        ("source", "source", "sno_es2", 1),
        ("confidence", "confidence", "sno_es2", 1),
        ("revalidated", "revalidated", "sno_es2", 1),
        ("oracle", "oracle", "oracle", None),
    ]
    series, curve_rows, inputs = {}, [], []
    for key, profile, method, factor in conditions:
        selected = [
            r
            for r in runs
            if (r["profile"], r["method"], r["noise_factor"])
            == (profile, method, factor)
        ]
        by_map = defaultdict(list)
        assert len(selected) == 12
        for r in selected:
            path = ROOT / "results/final" / r["directory"] / "progress.csv"
            with path.open() as stream:
                rows = list(csv.DictReader(stream))
            assert [int(row["step"]) for row in rows] == list(range(161))
            assert not r["failed"] and r["physical_actions"] == 160
            values = [float(row["discounted_return"]) for row in rows]
            assert np.isclose(values[-1], r["discounted_return"], rtol=1e-12)
            by_map[r["map_seed"]].append(values)
            inputs.append(
                {
                    "path": path.relative_to(ROOT).as_posix(),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
        assert len(by_map) == 6 and all(len(v) == 2 for v in by_map.values())
        map_means = np.array([np.mean(values, axis=0) for values in by_map.values()])
        mean, sd = map_means.mean(axis=0), map_means.std(axis=0, ddof=1)
        a = aggregates[profile, method, factor]
        assert np.isclose(mean[-1], a["discounted_return_mean"], rtol=1e-12)
        assert np.isclose(sd[-1], a["discounted_return_map_sd"], rtol=1e-12)
        series[key] = mean
        curve_rows.extend(
            {
                "condition": LABELS[key],
                "step": step,
                "mean_discounted_return": float(value),
                "map_sd": float(sd[step]),
                "maps": 6,
                "runs": 12,
            }
            for step, value in enumerate(mean)
        )
    write_csv(assets / "reward_curve_data.csv", curve_rows)
    with plt.rc_context({"font.size": 17, "axes.labelsize": 18, "legend.fontsize": 14}):
        fig, ax = plt.subplots(figsize=(11.3, 5.55))
        fig.subplots_adjust(left=0.105, right=0.985, top=0.79, bottom=0.16)
        chart_style(ax)
        for key, *_ in conditions:
            ax.plot(
                range(161),
                series[key],
                color=COLORS[key],
                label=LABELS[key],
                linewidth=2.6,
                linestyle="--" if key == "oracle" else "-",
            )
        ax.set(
            xlim=(0, 160),
            ylim=(0, 290),
            xlabel="Physical actions (shared budget)",
            ylabel="Mean discounted return",
        )
        ax.set_xticks(range(0, 161, 40))
        ax.legend(
            loc="lower left",
            bbox_to_anchor=(-0.02, 1.025),
            ncol=2,
            frameon=False,
            borderaxespad=0,
            columnspacing=1.3,
            handlelength=2.1,
        )
        fig.savefig(assets / "result_reward_curves.png", dpi=180, facecolor="white")
        fig.savefig(assets / "result_reward_curves.svg")
        plt.close(fig)

    matched = [aggregates[p, "sno_es2", 1] for p in PROFILES]
    matched_rows = [
        {
            "condition": LABELS[p],
            "mean_return": a["discounted_return_mean"],
            "return_map_sd": a["discounted_return_map_sd"],
            "mean_unsafe_actions": a["unsafe_actions_mean"],
            "runs": a["runs"],
            "maps": a["maps"],
        }
        for p, a in zip(PROFILES, matched)
    ]
    write_csv(assets / "matched_noise_results.csv", matched_rows)
    labels = ["Retained\nc=2", "Retained\nc=4", "Revalidated\nc=4"]
    with plt.rc_context({"font.size": 16, "axes.labelsize": 17}):
        fig, axes = plt.subplots(1, 2, figsize=(11.3, 5.55))
        fig.subplots_adjust(left=0.085, right=0.98, bottom=0.18, top=0.87, wspace=0.35)
        for ax in axes:
            chart_style(ax)
            ax.set_xticks(range(3), labels, fontsize=16)
        colors = [COLORS[p] for p in PROFILES]
        means = [a["discounted_return_mean"] for a in matched]
        sd = [a["discounted_return_map_sd"] for a in matched]
        axes[0].bar(
            range(3),
            means,
            color=colors,
            width=0.58,
            yerr=sd,
            capsize=6,
            error_kw={"elinewidth": 1.8},
        )
        axes[0].set_ylim(0, 300)
        axes[0].set_ylabel("Discounted return")
        axes[0].set_title("Return: mean ± map SD", fontsize=18, pad=14)
        for i, (value, variation) in enumerate(zip(means, sd)):
            axes[0].text(
                i,
                value + variation + 8,
                f"{value:.2f}",
                ha="center",
                fontsize=16,
                color=INK,
            )
        unsafe = [a["unsafe_actions_mean"] for a in matched]
        axes[1].bar(range(3), unsafe, color=colors, width=0.58)
        axes[1].set_ylim(0, 3.15)
        axes[1].set_ylabel("Mean unsafe actions / run")
        axes[1].set_title("Executed constraint violations", fontsize=18, pad=14)
        for i, value in enumerate(unsafe):
            axes[1].text(
                i,
                value + 0.12,
                f"{value:.3f}",
                ha="center",
                fontsize=17,
                color=INK,
                weight="bold",
            )
        fig.savefig(assets / "result_matched_noise.png", dpi=180, facecolor="white")
        fig.savefig(assets / "result_matched_noise.svg")
        plt.close(fig)

    factors = [0.5, 1, 2]
    robustness = []
    with plt.rc_context({"font.size": 17, "axes.labelsize": 18, "legend.fontsize": 15}):
        fig, ax = plt.subplots(figsize=(11.3, 5.55))
        fig.subplots_adjust(left=0.105, right=0.985, top=0.8, bottom=0.17)
        chart_style(ax)
        for p, marker in zip(PROFILES, ("o", "s", "D")):
            values = [
                aggregates[p, "sno_es2", f]["unsafe_actions_mean"] for f in factors
            ]
            ax.plot(
                factors,
                values,
                color=COLORS[p],
                marker=marker,
                markersize=9,
                markerfacecolor="white" if p == "revalidated" else COLORS[p],
                linewidth=2.5,
                label=LABELS[p],
                linestyle="--" if p == "revalidated" else "-",
            )
            robustness.extend(
                {
                    "condition": LABELS[p],
                    "assumed_noise_factor": f,
                    "mean_unsafe_actions": a["unsafe_actions_mean"],
                    "mean_return": a["discounted_return_mean"],
                    "return_map_sd": a["discounted_return_map_sd"],
                    "runs": 12,
                    "maps": 6,
                }
                for f in factors
                for a in [aggregates[p, "sno_es2", f]]
            )
        ax.set(
            xlim=(0.38, 2.12),
            ylim=(-0.2, 5.5),
            xlabel="Assumed / actual altitude-sensor noise STD",
            ylabel="Mean unsafe actions / run",
        )
        ax.set_xticks(
            factors,
            ["0.5 ×\nunderestimated", "1 ×\ncorrectly specified", "2 ×\noverestimated"],
        )
        ax.legend(
            loc="lower left",
            bbox_to_anchor=(-0.01, 1.025),
            ncol=2,
            frameon=False,
            borderaxespad=0,
        )
        ax.annotate(
            "c=4 retained: 2 unsafe actions in 12 runs",
            xy=(0.5, 1 / 6),
            xytext=(0.78, 1.35),
            fontsize=14,
            color=COLORS["confidence"],
            arrowprops={"arrowstyle": "->", "color": COLORS["confidence"]},
        )
        fig.savefig(assets / "result_noise_robustness.png", dpi=180, facecolor="white")
        fig.savefig(assets / "result_noise_robustness.svg")
        plt.close(fig)
    write_csv(assets / "noise_robustness_results.csv", robustness)
    metadata = {
        "source": "Frozen results/final; no new experiments or altered results",
        "aggregation": "Mean observation repeats within each map, then equal-weight map means",
        "return_error_bars": "Sample SD across six map means, not confidence intervals",
        "reward_curve_uncertainty": "Means only; variability is shown on the matched-noise slide",
        "nominal_noise_factor": 1,
        "actual_altitude_noise_std": 0.001,
        "inputs": inputs,
        "summary_sha256": hashlib.sha256(
            (ROOT / "results/final/summary.json").read_bytes()
        ).hexdigest(),
        "runs_sha256": hashlib.sha256(
            (ROOT / "results/final/runs.json").read_bytes()
        ).hexdigest(),
    }
    (assets / "graph_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return aggregates


def foundation_slides():
    """Four context and four method slides, with literature separated from our port."""
    deck = []
    s = base(
        "SNO-MDP: learning reward while remaining safe",
        1,
        40,
        "The paper is Safe Reinforcement Learning in Constrained Markov Decision Processes by Akifumi Wachi and Yanan Sui, ICML 2020. "
        "It studies unknown reward and safety functions, with safety required during learning, not just after convergence. "
        "Its guarantees are high-probability statements under explicit assumptions, not unconditional zero-risk claims. "
        "Our project is a source-audited empirical reproduction with a certificate-revalidation extension. We separate literature, implementation, and measured results.",
    )
    text(
        s,
        "Unknown reward.\nUnknown safety.\nSafe learning throughout.",
        0.85,
        2.1,
        8.8,
        35,
        TEAL,
        True,
    )
    text(
        s,
        "Akifumi Wachi & Yanan Sui · ICML 2020\nSafe Reinforcement Learning in\nConstrained Markov Decision Processes",
        0.9,
        4.75,
        8.6,
        22,
    )
    card(
        s,
        "Paper objective",
        "Maximize cumulative reward while satisfying the safety constraint at every learning step.",
        10.1,
        2.3,
        5.1,
        4.2,
    )
    callout(
        s,
        "Safety and near-optimality are conditional, high-probability results—not automatic properties of a GP.",
        size=19,
    )
    deck.append(s)

    s = base(
        "The challenge: learn two unknown functions safely",
        1,
        60,
        "In the paper, reward r of a state and safety g of a state are unknown, while transition dynamics are known. "
        "The state constraint must hold at each visited time, unlike merely bounding an expected accumulated cost. "
        "Our implemented discrete benchmark instead checks a directed altitude difference. A move uphill can fail while the reverse move passes. "
        "The learner observes noisy local values and supplied initial certificates; complete truth is evaluator information. "
        "The final project uses twenty-by-twenty synthetic fields. This six-by-six grid is an illustration, not a Mars experiment.",
    )
    text(s, "Paper: g(s_t) ≥ h at every visited step", 0.85, 2.1, 8.5, 25, TEAL, True)
    text(
        s,
        "Unknown reward r(s) and safety g(s)\nKnown dynamics + a safe starting region\nNoisy observations at current / nearby cells",
        0.85,
        3.05,
        8.1,
        23,
    )
    rect(s, 0.85, 5.4, 8.1, 1.25, PALE)
    text(
        s, "Our port: g(s,a) = z(s) − z(next) ≥ −0.25", 1.05, 5.73, 7.75, 23, TEAL, True
    )
    cell, gx, gy = 0.65, 10.3, 2.2
    for x in range(6):
        for y in range(6):
            rect(
                s,
                gx + x * cell,
                gy + (5 - y) * cell,
                cell - 0.035,
                cell - 0.035,
                "#4b9fa9" if (x, y) in ((1, 1), (1, 2), (2, 1), (2, 2)) else "#f1dfb5",
            )
    rect(s, gx + 1.18 * cell, gy + 4.18 * cell, 0.25, 0.25, INK, circle=True)
    text(s, "6 × 6 illustration; our final study is 20 × 20", 9.3, 6.5, 6.0, 15, MUTED)
    callout(
        s,
        "The complete true map is available to the evaluator/oracle, never to the learning policy.",
        size=19,
    )
    deck.append(s)

    s = base(
        "What makes the paper’s guarantees possible?",
        1,
        45,
        "Safe learning from nothing is not generally possible. The paper assumes a nonempty known-safe seed set, bounded RKHS norms for reward and safety, Lipschitz regularity of safety, and controlled observation noise. "
        "Its confidence schedules support a simultaneous high-probability confidence event. On that event, the safe-region construction supports safety at every step and the stated near-optimality results. "
        "Our fixed practical multipliers do not verify those hypotheses, and finite zero-violation runs cannot replace that proof.",
    )
    card(
        s,
        "Safe seed S0",
        "A nonempty, known-safe starting set supplies initial permission to explore.",
        0.7,
        2.4,
        4.7,
        3.5,
    )
    card(
        s,
        "Function regularity",
        "Bounded RKHS norms; safety is L-Lipschitz under a chosen distance.",
        5.65,
        2.4,
        4.7,
        3.5,
    )
    card(
        s,
        "Valid confidence",
        "Bounded / sub-Gaussian noise and suitable confidence schedules.",
        10.6,
        2.4,
        4.7,
        3.5,
    )
    callout(
        s,
        "High-probability safety under assumptions ≠ deterministic safety under arbitrary model mismatch.",
        color=GOLD,
        size=19,
    )
    deck.append(s)

    s = base(
        "From safe exploration to near-optimal reward",
        1,
        55,
        "SafeMDP has a purposeful information-gathering objective: it expands the safely certifiable region without optimizing reward. Calling it aimless is inaccurate. "
        "SNO-MDP separates safety exploration from optimistic reward planning; ES2 can stop safety exploration early while retaining the paper’s conditional near-optimality result. "
        "P-ES2 is the paper’s probability-weighted practical alternative. It can converge faster empirically, but the paper does not supply the same reward near-optimality guarantee for it. "
        "Our frozen project implements basic SNO-MDP, ES2, and current-LCB revalidation, not P-ES2 or Mars evaluation. Member 2 explains the mechanics.",
    )
    card(
        s,
        "SafeMDP",
        "Explore safely to reduce uncertainty; no reward-optimization phase.",
        0.7,
        2.35,
        4.7,
        3.65,
    )
    card(
        s,
        "SNO-MDP + ES2",
        "Safety exploration → optimistic reward. Conditional near-optimality analysis.",
        5.65,
        2.35,
        4.7,
        3.65,
    )
    card(
        s,
        "P-ES2: paper only",
        "Probability-weighted auxiliary planning. No project implementation or results.",
        10.6,
        2.35,
        4.7,
        3.65,
    )
    callout(
        s,
        "Our contribution: source audit + robustness study + current-LCB extension + reproducible analysis.",
        size=19,
    )
    deck.append(s)

    s = base(
        "Two GPs: uncertainty for safety and optimism for reward",
        2,
        55,
        "Separate GPs model reward and altitude. In our port, safety concerns a correlated difference between endpoint altitudes, so its variance includes the covariance term. "
        "The code uses direct standard-deviation multipliers c_s and c_r; the paper’s intervals use square-root beta for safety and square-root alpha for reward. These numerical settings are not interchangeable. "
        "The reward upper bound uses c_r equal to three. Actual sensor noise defaults to point zero zero one, and a separate factor controls assumed altitude noise. "
        "These values do not by themselves establish the paper’s confidence event.",
    )
    card(
        s,
        "Altitude GP",
        "Infer transition margins using joint endpoint covariance.",
        0.7,
        2.2,
        7.1,
        2.5,
    )
    card(
        s,
        "Reward GP",
        "Infer rewards and add an uncertainty bonus for planning.",
        8.2,
        2.2,
        7.1,
        2.5,
    )
    text(
        s,
        "Var[g] = Var[z(s)] + Var[z(next)] − 2 Cov[z(s), z(next)]",
        0.95,
        5.05,
        14,
        25,
        TEAL,
        True,
    )
    text(s, "Safety: LCB = μg − c_s σg", 0.95, 6.02, 7.0, 23, TEAL, True)
    text(s, "Reward: U = μr + c_r σr", 8.2, 6.02, 7.0, 23, TEAL, True)
    callout(
        s,
        "Project settings: c_s = 2 or 4 · c_r = 3 · actual sensor STD = 0.001",
        size=19,
    )
    deck.append(s)

    s = base(
        "Pessimistic, optimistic, and expansion candidates",
        2,
        50,
        "The pessimistic region contains model-certified states with reachability and returnability. The optimistic region contains possibilities consistent with upper bounds and graph constraints; it is not permission to move. "
        "Expanders are permitted candidates whose optimistic information could certify more territory. They are not merely every boundary cell. "
        "The paper defines state regions with Lipschitz propagation. Our source port uses directed state-action certificates and an expander score. It selects large uncertainty widths and travels through executable actions. "
        "The project’s L equal to zero is an expander setting, not a valid global flat-terrain assumption.",
    )
    card(
        s,
        "X− · pessimistic",
        "Model-certified, reachable and returnable. Physical execution uses this region.",
        0.7,
        2.35,
        4.7,
        3.6,
        PALE,
    )
    card(
        s,
        "X+ · optimistic",
        "Could be safe according to upper bounds and graph checks.",
        5.65,
        2.35,
        4.7,
        3.6,
        GOLD,
    )
    card(
        s,
        "G_t · expanders",
        "Permitted candidates with potential to certify new states / actions.",
        10.6,
        2.35,
        4.7,
        3.6,
        "#f9e8da",
    )
    text(s, "Explore wide uncertainty: w_t = u_t − l_t", 0.95, 6.05, 14, 25, TEAL, True)
    text(
        s,
        "Basic in our port: max executable width ≤ ε_g = 0.005",
        0.95,
        6.65,
        14,
        18,
        MUTED,
    )
    callout(
        s,
        "Paper: state sets + Lipschitz reasoning. Our source port: directed action certificates.",
        size=19,
    )
    deck.append(s)

    s = base(
        "Our extension: recheck historical safety permissions",
        2,
        55,
        "Our baseline recomputes bounds and retains previously accepted actions, following the released source profile. This differs from the paper’s historical interval intersection. "
        "Our extension requires current support or an initial certificate and rebuilds the reachable and returnable graph. If the current state loses this region, the run halts and counts as a failure. "
        "The saved step-ninety-six up action has LCB minus zero point two eight zero six, below minus zero point two five, yet retained permission allows it. Member 4 shows the violation. "
        "Both GPs keep updating in our reward phase; our implementation does not freeze learning.",
    )
    card(
        s,
        "Retained source profile",
        "An accepted permission can survive a later failing bound.",
        0.7,
        2.25,
        7.1,
        2.9,
        GOLD,
    )
    card(
        s,
        "Current-LCB extension",
        "Require current support or an initial certificate; rebuild the graph.",
        8.2,
        2.25,
        7.1,
        2.9,
    )
    text(
        s,
        "Saved example: LCB −0.280632 < threshold −0.25",
        0.95,
        5.75,
        14,
        26,
        TEAL,
        True,
    )
    callout(
        s,
        "Our GPs and certificates update in both phases. Incorrect permission ≠ executed violation.",
        size=19,
    )
    deck.append(s)

    s = base(
        "ES2 and P-ES2: different auxiliary planning models",
        2,
        40,
        "ES2 checks an auxiliary optimistic MDP and stops when all relevant policy successors from executable sources remain inside the pessimistic region. Our project implements that source-profile test. "
        "The paper’s P-ES2 weights transitions using GP safety probabilities and a virtual zero-reward absorbing state. It is a practical alternative without the same near-optimality proof. "
        "Early stopping changes when exploration ends, not permission to take unsafe moves. Our project does not implement P-ES2. Member 3 now shows our actual empirical results.",
    )
    card(
        s,
        "ES2 · implemented",
        "Auxiliary optimistic policy. Stop when its relevant successors stay in X−.",
        0.7,
        2.35,
        7.1,
        3.25,
    )
    card(
        s,
        "P-ES2 · paper only",
        "Probability-weighted transitions and a virtual absorbing state. No same reward near-optimality guarantee.",
        8.2,
        2.35,
        7.1,
        3.25,
    )
    text(s, "ES2 stops when Y_t remains inside X−_t", 0.95, 6.05, 14, 27, TEAL, True)
    callout(
        s,
        "Stopping changes when exploration ends. Actual movement still needs valid certificates.",
        size=19,
    )
    deck.append(s)
    return deck


def paper_backup_slides():
    result = []
    s = base(
        "Paper experiments and our study are different",
        3,
        0,
        "Paper background only. Section 6.1 uses a twenty-by-twenty synthetic benchmark with RBF models and discount point nine nine. "
        "Section 6.2 uses a forty-by-thirty HiRISE Mars DEM crop, a twenty-five-degree slope rule, Matern-five-halves elevation kernel with fifteen-metre length scale, elevation variance one hundred square metres, noise standard deviation point zero seven five metres, and discount point nine. "
        "Table 1 reports normalized Mars rewards point eight one for P-ES2, point seven eight for ES2, and point four nine for basic SNO-MDP, with zero unsafe actions in those reported conditions. "
        "These are author results, not our measurements, and their normalized reward metric must not be mixed with our raw discounted returns.",
        backup=True,
    )
    card(
        s,
        "Paper · synthetic",
        "20 × 20 GP fields\nRBF reward / safety\nγ = 0.99\nNormalized reward curves",
        0.7,
        2.2,
        4.55,
        4.2,
    )
    card(
        s,
        "Paper · Mars",
        "40 × 30 HiRISE crop\n25° slope constraint\nMatérn-5/2 elevation\nγ = 0.9",
        5.72,
        2.2,
        4.55,
        4.2,
        GOLD,
    )
    card(
        s,
        "Paper · Table 1",
        "Mars normalized reward:\nP-ES2   0.81\nES2       0.78\nSNO       0.49",
        10.74,
        2.2,
        4.55,
        4.2,
    )
    callout(
        s,
        "Authors’ results only. Our 192-run synthetic study has no Mars data or P-ES2 measurements.",
        size=19,
    )
    result.append(s)

    s = base(
        "P-ES2: probability-weighted virtual planning",
        3,
        0,
        "Literature only, Section 4.4. With positive posterior STD the model probability is Phi of mean minus threshold over STD, equivalently one minus one-half erfc of mean minus threshold over square-root-two STD. "
        "The auxiliary model sends probability p to the successor and one minus p to a virtual absorbing zero-reward state. Multi-step speculative paths are discounted by these survival weights. "
        "This is virtual planning, not permission for the rover to gamble with real unsafe actions. Safety remains conditional on valid certificates; the paper explicitly does not provide the same reward near-optimality guarantee for P-ES2. "
        "There is no universal five-times speedup, and we did not implement or measure P-ES2.",
        backup=True,
    )
    text(
        s,
        "p(s) = P(g(s) ≥ h | data) = Φ((μg − h) / σg)",
        0.95,
        2.15,
        14,
        27,
        TEAL,
        True,
    )
    text(
        s,
        "Positive posterior STD; a model probability, not a simultaneous safety certificate.",
        0.95,
        2.9,
        14,
        19,
        MUTED,
    )
    card(
        s,
        "Virtual transition",
        "With probability p:\ncontinue to successor",
        0.7,
        3.75,
        6.8,
        2.3,
    )
    card(
        s,
        "Absorbing state z_fail",
        "With probability 1 − p:\nzero future reward",
        8.3,
        3.75,
        7.0,
        2.3,
        GOLD,
    )
    text(
        s,
        "Physical execution remains inside the certified region.",
        0.95,
        6.4,
        14,
        24,
        TEAL,
        True,
    )
    callout(
        s,
        "Paper-only practical variant; no project implementation, benchmark score, or near-optimality claim.",
        size=19,
    )
    result.append(s)

    s = base(
        "Hyperparameters: meaning and actual project values",
        3,
        0,
        "These are project values, not a universal tuning guide. At a fixed posterior, larger safety c widens intervals and permits fewer instantaneous LCB passes. It can delay expansion, but rerun outcomes need not change monotonically because observations and paths change. "
        "Larger reward c increases the uncertainty bonus, not guaranteed final return. Actual sensor noise changes readings; assumed-noise factor changes only the altitude GP assumption. "
        "Length scale two is shared by both RBF models. Our zero expander Lipschitz setting is not the true global Lipschitz constant in the paper’s theorem. "
        "For a correctly specified pointwise Gaussian posterior, c equal to two corresponds to about ninety-seven point seven percent one-sided mass. This is not a uniform guarantee across adaptive states, actions and time.",
        backup=True,
    )
    cols, widths = [0.7, 4.7, 7.1], [4.0, 2.4, 8.2]
    for x, w, t in zip(cols, widths, ("Control", "Project value", "What changes")):
        rect(s, x, 2.1, w, 0.65, TEAL)
        text(s, t, x + 0.16, 2.28, w - 0.3, 18, "#ffffff", True)
    rows = [
        (
            "Safety multiplier c_s",
            "2; selected 4",
            "Larger: wider intervals, fewer current LCB passes",
        ),
        (
            "Reward multiplier c_r",
            "3",
            "Larger: more reward optimism at fixed posterior",
        ),
        (
            "Actual noise STD",
            ".001",
            "Changes readings; reward and altitude separately",
        ),
        (
            "Assumed altitude factor",
            ".5 / 1 / 2",
            "GP assumption = actual altitude STD × factor",
        ),
        (
            "Shared RBF length scale",
            "2",
            "Controls correlation distance; depends on map units",
        ),
        (
            "Exploration accuracy ε_g",
            ".005",
            "Smaller: stricter uncertainty stopping criterion",
        ),
        (
            "Discount γ / threshold h",
            ".99 / −.25",
            "Reward horizon weighting / directed safety margin",
        ),
    ]
    for i, row in enumerate(rows):
        y = 2.75 + i * 0.54
        for x, w, t in zip(cols, widths, row):
            rect(s, x, y, w, 0.54, PALE if i % 2 == 0 else "#f5f8f8")
            text(s, t, x + 0.16, y + 0.13, w - 0.3, 17)
    callout(
        s,
        "Fixed-posterior trends do not guarantee monotonic changes in complete policy reruns.",
        size=19,
    )
    result.append(s)

    s = base(
        "Viva: precise answers without overstating safety",
        3,
        0,
        "Why GPs rather than neural networks? Their kernel-based uncertainty fits the paper’s RKHS confidence analysis and is convenient in this small-data setting. Neural methods can also quantify uncertainty; neither model family is automatically calibrated. "
        "How can early stopping remain safe? It changes the exploration stopping decision while execution remains restricted to a certified subset on the valid confidence event. "
        "Does zero observed violations prove safety? No. The theorem assumes a safe seed, regularity, noise control and suitable confidence schedules; finite empirical data do not establish that event. "
        "ES2 has the paper’s conditional near-optimality theorem. P-ES2 is practical and lacks that same reward guarantee. Our fixed-c port and extension do not inherit the theorem automatically.",
        backup=True,
    )
    card(
        s,
        "Why GPs?",
        "Kernel uncertainty matches the paper’s analysis. Calibration and regularity still matter.",
        0.7,
        2.35,
        4.7,
        3.7,
    )
    card(
        s,
        "Why can stopping be safe?",
        "It changes when to stop exploring, not where movement is permitted.",
        5.65,
        2.35,
        4.7,
        3.7,
    )
    card(
        s,
        "Do zero violations prove it?",
        "No. Finite outcomes do not establish a uniform high-probability confidence event.",
        10.6,
        2.35,
        4.7,
        3.7,
    )
    callout(
        s,
        "Neither GPs nor neural networks supply an automatic safety guarantee.",
        size=19,
    )
    result.append(s)
    return result


def user_rehearsal_sections(value):
    """Keep current user wording while mapping team labels to the five result scenes."""
    headings = list(re.finditer(r"(?m)^## (.+)\n", value))
    found = {}
    for i, heading in enumerate(headings):
        match = re.match(r"Slide (\d+) — ", heading.group(1))
        if match:
            end = headings[i + 1].start() if i + 1 < len(headings) else len(value)
            found[int(match.group(1))] = value[heading.start() : end].rstrip("\n")
    if set(range(11, 18)) <= found.keys():
        return {i: found[number] for i, number in enumerate((12, 13, 14, 15, 17), 1)}
    if set(range(1, 6)) <= found.keys():
        return {i: found[i] for i in range(1, 6)}
    raise ValueError("Expected the current team rehearsal numbered 11–17.")


def update_member3_rehearsal():
    """Create the five-minute script only if missing; preserve later user edits."""
    path = REPORT / "member3_results_rehearsal.md"
    write_rehearsal(path)
    return user_rehearsal_sections(path.read_text())


def slides():
    # Import the standalone scenes without invoking its file-writing entry point.
    import build_member3_results as member3

    update_member3_rehearsal()
    current = read_speeches(REPORT / "member3_results_rehearsal.md")
    summary, runs, aggregates, _, _ = member3.load_evidence()
    member3.ASSETS.mkdir(exist_ok=True)
    result_charts(summary, runs, assets=member3.ASSETS)
    member3.stopping_chart(aggregates)
    result_scenes = member3.build_slides(aggregates)
    for scene, talk_index in zip(result_scenes, (1, 2, 3, 4, 6)):
        item = TALK[talk_index]
        scene.notes = presenter_note(item, current[item.number])
        scene.cue = (
            "Follow the five-minute team rehearsal; only spoken paragraphs are timed."
        )

    deck = foundation_slides()

    s = base(
        "Experimental design: freeze first, evaluate fresh cases",
        3,
        TALK[0].seconds,
        presenter_note(TALK[0], current[11]),
    )
    for x, value, label in (
        (0.7, "6", "fresh maps"),
        (4.42, "2", "observation seeds"),
        (8.14, "3", "noise factors"),
        (11.86, "160", "actions / case"),
    ):
        rect(s, x, 2.2, 3.4, 1.7, PALE)
        text(s, value, x + 0.25, 2.42, 2.9, 38, TEAL, True)
        text(s, label, x + 0.25, 3.15, 2.9, 19)
    text(
        s,
        "192 final runs = 180 learners + 12 oracle references",
        0.95,
        4.4,
        14,
        28,
        TEAL,
        True,
    )
    text(
        s,
        "Same horizon, sensor access, and matched conditions\nDevelopment selection → frozen settings → held-out study",
        0.95,
        5.35,
        14,
        23,
    )
    callout(s, "Certificate control: compare c=4 retained against c=4 revalidated.")
    deck.append(s)
    deck.extend(result_scenes[:4])

    s = base(
        "Saved dashboard evidence",
        3,
        TALK[5].seconds,
        presenter_note(TALK[5], current[16]),
    )
    picture(s, ASSETS / "dashboard_ui.png", 0.7, 2.0, 10.7, 5.9)
    text(
        s,
        "Saved source replay\nStep 97 · one violation\nMap 100 / obs 3000",
        11.65,
        2.4,
        3.65,
        21,
        TEAL,
        True,
    )
    text(
        s,
        "Before the move:\nLCB −0.280632\nThreshold −0.25\nRetained certificate",
        11.65,
        4.65,
        3.65,
        21,
    )
    deck.append(s)
    deck.append(result_scenes[-1])

    for i, scene in enumerate(deck, 1):
        # Replace results-only counters with the final integrated slide numbers.
        scene.elements = [
            e
            for e in scene.elements
            if not (e["kind"] == "text" and e["x"] >= 13.3 and e["y"] >= 8.5)
        ]
        if scene.title != "Thank You":
            text(scene, f"{i:02d} / 15", 13.4, 8.53, 1.9, 11, MUTED)
    return deck


def powerpoint(deck, output=None, title=None, subject=None, presenter_slot=None):
    presentation = Presentation()
    presentation.slide_width = Inches(WIDTH)
    presentation.slide_height = Inches(HEIGHT)
    presentation.core_properties.title = (
        title or "SNO-MDP · final professor presentation"
    )
    presentation.core_properties.subject = subject or (
        "15 integrated slides: Members 1–2, experimental design, Member 3 results, saved dashboard evidence, and Thank You"
    )
    for index, scene in enumerate(deck, 1):
        s = presentation.slides.add_slide(presentation.slide_layouts[6])
        for e in scene.elements:
            if e["kind"] == "image":
                s.shapes.add_picture(
                    str(e["path"]),
                    Inches(e["x"]),
                    Inches(e["y"]),
                    Inches(e["w"]),
                    Inches(e["h"]),
                )
            elif e["kind"] == "rect":
                shape = s.shapes.add_shape(
                    MSO_AUTO_SHAPE_TYPE.OVAL
                    if e["circle"]
                    else MSO_AUTO_SHAPE_TYPE.RECTANGLE,
                    Inches(e["x"]),
                    Inches(e["y"]),
                    Inches(e["w"]),
                    Inches(e["h"]),
                )
                shape.fill.solid()
                shape.fill.fore_color.rgb = RGBColor.from_string(e["color"][1:])
                if e["edge"]:
                    shape.line.color.rgb = RGBColor.from_string(e["edge"][1:])
                else:
                    shape.line.fill.background()
            else:
                box = s.shapes.add_textbox(
                    Inches(e["x"]), Inches(e["y"]), Inches(e["w"]), Inches(e["h"])
                )
                tf = box.text_frame
                tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
                tf.word_wrap = False
                for line_number, line in enumerate(e["text"].splitlines()):
                    p = tf.paragraphs[0] if line_number == 0 else tf.add_paragraph()
                    p.space_before = p.space_after = Pt(0)
                    p.line_spacing = Pt(e["size"] * 1.25)
                    run = p.add_run()
                    run.text = line
                    run.font.name = "Arial"
                    run.font.size = Pt(e["size"])
                    run.font.bold = e["bold"]
                    run.font.color.rgb = RGBColor.from_string(e["color"][1:])
                    if e["link"]:
                        run.hyperlink.address = e["link"]
        s.notes_slide.notes_text_frame.text = (
            f"{'BACKUP ONLY' if scene.backup else 'MEMBER ' + str(scene.member)} | Slide {index} | "
            f"Slide allocation {scene.seconds}s\n{scene.cue}\n\n{scene.notes}\n\n"
            + (
                presenter_slot
                or "Slide presenter slot: target 3:30 including handoff, hard maximum 4:00. Member 4 uses the dashboard separately."
            )
        )
    presentation.save(output or REPORT / "presentation.pptx")


def render(deck, output=None, assets=None, title=None):
    assets = Path(assets) if assets is not None else ASSETS
    assets.mkdir(parents=True, exist_ok=True)
    with PdfPages(
        output or REPORT / "presentation.pdf",
        metadata={"Title": title or "SNO-MDP · final professor presentation"},
    ) as pdf:
        for i, scene in enumerate(deck, 1):
            fig = plt.figure(figsize=(WIDTH, HEIGHT))
            ax = fig.add_axes((0, 0, 1, 1))
            ax.set_xlim(0, WIDTH)
            ax.set_ylim(HEIGHT, 0)
            ax.axis("off")
            for e in scene.elements:
                if e["kind"] == "image":
                    ax.imshow(
                        Image.open(e["path"]),
                        extent=(e["x"], e["x"] + e["w"], e["y"] + e["h"], e["y"]),
                        aspect="auto",
                        zorder=2,
                    )
                elif e["kind"] == "rect":
                    shape = (
                        patches.Ellipse(
                            (e["x"] + e["w"] / 2, e["y"] + e["h"] / 2), e["w"], e["h"]
                        )
                        if e["circle"]
                        else patches.Rectangle((e["x"], e["y"]), e["w"], e["h"])
                    )
                    shape.set_facecolor(e["color"])
                    shape.set_edgecolor(e["edge"] or "none")
                    ax.add_patch(shape)
                else:
                    for j, line in enumerate(e["text"].splitlines()):
                        ax.text(
                            e["x"],
                            e["y"] + j * e["line_height"],
                            line,
                            fontsize=e["size"],
                            color=e["color"],
                            weight="bold" if e["bold"] else "normal",
                            va="top",
                            fontfamily="Arial",
                        )
            pdf.savefig(fig)
            fig.savefig(assets / f"slide_{i:02d}.png", dpi=90)
            plt.close(fig)


def documents(deck):
    notes = [
        "# Final professor presentation and speaker notes",
        "",
        "Members 1 and 2 retain slides 1–4 and 5–8. Member 3 presents experimental design (9), results (10–13), saved dashboard evidence (14), and Thank You (15). Srinivas then presents the live dashboard.",
        "15 slides total. Superseded results and Q&A backups have been removed. Member 3 follows the five-minute rehearsal, including design and dashboard evidence.",
        "Presenter and timing information is in these notes, not on the audience slides. Graphs use existing frozen results.",
        "See [claim corrections and sources](presentation_claim_review.md) and the [integrated Member 3 rehearsal](member3_results_rehearsal.md). P-ES2 remains literature background; Mars results are not claimed as our experiments.",
        "",
        "[PowerPoint](presentation.pptx) · [PDF](presentation.pdf) · [Browser slides](presentation.html) · [Team timing and demo plan](team_presentation_plan.md)",
        "",
    ]
    sections = []
    for i, s in enumerate(deck, 1):
        notes.extend(
            [
                f"## Slide {i} — {s.title}",
                "",
                "Backup; do not include in the timed talk."
                if s.backup
                else f"**Member {s.member} · slide allocation {s.seconds}s.**",
                "",
                s.cue,
                "",
                s.notes,
                "",
            ]
        )
        sections.append(
            f'<section class="slide" data-member="{s.member}" {"hidden" if i != 1 else ""}>'
            f'<img src="presentation_assets/slide_{i:02d}.png" alt="{html.escape(s.title)}">'
            f"<details><summary>Speaker notes · Member {s.member}</summary><p>{html.escape(s.cue)}</p><pre>{html.escape(s.notes).replace(chr(10), '&#10;')}</pre></details></section>"
        )
    (REPORT / "presentation.md").write_text(
        "\n".join(line.rstrip() for line in "\n".join(notes).splitlines()) + "\n"
    )
    (REPORT / "presentation.html").write_text(
        """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SNO-MDP · final professor presentation</title><style>
body{margin:0;background:#102d36;color:#edf5f5;font:14px Arial,sans-serif}nav{display:flex;gap:12px;align-items:center;padding:12px 20px;flex-wrap:wrap}button,a{background:#eff7f6;color:#145c62;border:0;border-radius:5px;padding:9px 13px;text-decoration:none;cursor:pointer}#position{margin-left:auto}.slide{max-width:1420px;margin:auto}.slide img{width:100%;display:block}details{padding:16px 24px;line-height:1.6}details pre{white-space:pre-wrap;font:inherit}summary{cursor:pointer}[hidden]{display:none!important}@media print{nav,details{display:none}.slide[hidden]{display:block!important}.slide{break-after:page}body{background:white}@page{size:landscape;margin:0}}
</style><nav><button id="prev">← Previous</button><button id="next">Next →</button><a href="http://127.0.0.1:8765" target="_blank" rel="noopener">Dashboard ↗</a><a href="team_presentation_plan.md">Team plan</a><span id="position"></span></nav>"""
        + "".join(sections)
        + """<script>
let index=0;const all=[...document.querySelectorAll('.slide')];function update(){const limit=all.length;index=Math.max(0,Math.min(limit-1,index));all.forEach((s,i)=>s.hidden=i!==index);document.getElementById('position').textContent=(index+1)+' / '+limit+' · Member '+all[index].dataset.member;document.getElementById('prev').disabled=index===0;document.getElementById('next').disabled=index===limit-1;}document.getElementById('prev').onclick=()=>{index--;update()};document.getElementById('next').onclick=()=>{index++;update()};document.onkeydown=e=>{if(e.target.closest('details')||(e.key===' '&&['BUTTON','A'].includes(e.target.tagName)))return;if(e.key==='ArrowRight'||e.key===' '){e.preventDefault();index++;update()}if(e.key==='ArrowLeft'){index--;update()}};update();
</script></html>"""
    )


def main():
    ASSETS.mkdir(exist_ok=True)
    protocol = read(REPORT / "frozen_protocol.json")
    source = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (ROOT / "sno_mdp").glob("*.py")
    }
    if source != protocol["source_sha256"]:
        raise ValueError("Scientific source does not match the frozen protocol.")
    deck = slides()
    assert len(deck) == 15 and Counter(s.member for s in deck if not s.backup) == {
        1: 4,
        2: 4,
        3: 7,
    }
    assert deck[8].title == "Experimental design: freeze first, evaluate fresh cases"
    assert deck[-2].title == "Saved dashboard evidence"
    assert deck[-1].title == "Thank You"
    assert not any(s.backup for s in deck)
    for i, s in enumerate(deck, 1):
        for e in s.elements:
            if (
                e["x"] < 0
                or e["y"] < 0
                or e["x"] + e["w"] > WIDTH + 0.02
                or e["y"] + e["h"] > HEIGHT + 0.02
            ):
                raise ValueError(f"Slide {i} has an element outside the canvas: {e}")
    powerpoint(
        deck,
        presenter_slot="Integrated deck: Members 1–2 retain their existing slots. Member 3 follows the five-minute rehearsal; its team labels 11–17 map to slides 9–15 in this preview. Srinivas presents the live dashboard after Thank You.",
    )
    render(deck)
    for stale in ASSETS.glob("slide_*.png"):
        if int(stale.stem.removeprefix("slide_")) > len(deck):
            stale.unlink()
    documents(deck)
    section = deepcopy(deck[8:])
    for team_number, scene in enumerate(section, 11):
        for element in scene.elements:
            if (
                element["kind"] == "text"
                and element["x"] >= 13.3
                and element["y"] >= 8.5
            ):
                element["text"] = f"{team_number:02d} / 17"
    powerpoint(
        section,
        output=REPORT / "member3_shared_deck_replacement.pptx",
        title="Member 3 · Five-minute results section",
        subject="Seven prepared slides for team positions 11–17; five minutes total",
        presenter_slot="Member 3: five minutes across team slides 11–17, including the handoff to Srinivas. Stage cues and preparation material are not spoken.",
    )
    summary = {
        "main_slides": 15,
        "backup_slides": 0,
        "members": 4,
        "ppt_presenters": [1, 2, 3],
        "slide_allocation_seconds_per_member": {
            str(member): sum(s.seconds for s in deck if s.member == member)
            for member in (1, 2, 3)
        },
        "member_4_live_ui_seconds": 210,
        "member_4_main_slides": [],
        "experimental_design_slide": 9,
        "result_graph_main_slides": [10, 11, 12, 13],
        "saved_dashboard_evidence_slide": 14,
        "thank_you_slide": 15,
        "member3_source_deck": "report/member3_results.pptx",
        "member3_rehearsal_text_preserved": True,
        "member3_rehearsal_section_sha256": {
            str(number): hashlib.sha256(section.encode()).hexdigest()
            for number, section in user_rehearsal_sections(
                (REPORT / "member3_results_rehearsal.md").read_text()
            ).items()
        },
        "source_matches_frozen": True,
        "paper_source_url": "https://proceedings.mlr.press/v119/wachi20a/wachi20a.pdf",
        "claim_review": "report/presentation_claim_review.md",
        "slide_ownership": [
            {
                "slide": i,
                "member": s.member,
                "title": s.title,
                "backup": s.backup,
                "slide_seconds": s.seconds,
            }
            for i, s in enumerate(deck, 1)
        ],
    }
    (ASSETS / "deck_manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        "Built presentation.pptx, presentation.pdf, presentation.html and presentation.md"
    )
    print(
        "15 integrated slides · design + four results + saved dashboard evidence + Thank You · user-edited rehearsal preserved"
    )


if __name__ == "__main__":
    main()
