"""Standalone results deck for Member 3. Does not overwrite the full team deck."""

from collections import defaultdict
import hashlib
import html
import json

import build_team_presentation as deck
import numpy as np
from matplotlib import pyplot as plt

from member3_talk import TALK, read_speeches, write_rehearsal, write_speaker_notes


ROOT, REPORT = deck.ROOT, deck.REPORT
ASSETS = REPORT / "member3_results_assets"
PREFIX = "member3_results"
PROFILES = ("source", "confidence", "revalidated")
NAMES = {
    "source": "ES2 retained c=2",
    "confidence": "ES2 retained c=4",
    "revalidated": "ES2 revalidated c=4",
}


RESULT_TALK_INDICES = (1, 2, 3, 4, 6)


def load_evidence():
    summary = json.loads((ROOT / "results/final/summary.json").read_text())
    runs = json.loads((ROOT / "results/final/runs.json").read_text())
    aggregates = {
        (a["profile"], a["method"], a["noise_factor"]): a for a in summary["aggregates"]
    }
    assert summary["planned_runs"] == summary["completed_runs"] == len(runs) == 192
    assert summary["maps"] == 6 and summary["failed_runs"] == 0
    assert all(
        r["physical_actions"] == 160 and r["stop_reason"] == "horizon_reached"
        for r in runs
    )
    totals = []
    for factor in (0.5, 1, 2):
        for profile in PROFILES:
            selected = [
                r
                for r in runs
                if r["profile"] == profile
                and r["method"] == "sno_es2"
                and r["noise_factor"] == factor
            ]
            assert len(selected) == 12
            a = aggregates[profile, "sno_es2", factor]
            by_map = defaultdict(list)
            for r in selected:
                by_map[r["map_seed"]].append(r)
            assert len(by_map) == 6 and all(len(v) == 2 for v in by_map.values())
            means = [
                np.mean([r["discounted_return"] for r in values])
                for values in by_map.values()
            ]
            assert np.isclose(np.mean(means), a["discounted_return_mean"])
            assert np.isclose(np.std(means, ddof=1), a["discounted_return_map_sd"])
            totals.append(
                {
                    "profile": profile,
                    "noise_factor": factor,
                    "total_unsafe_actions": sum(r["unsafe_actions"] for r in selected),
                    "unsafe_runs": sum(r["unsafe_actions"] > 0 for r in selected),
                    "es2_triggered_runs": sum(r["es2_triggered"] for r in selected),
                    "runs": len(selected),
                    "maps": len(by_map),
                }
            )
    paired = []
    for factor in (0.5, 1, 2):
        by_case = {
            (r["map_seed"], r["observation_seed"], r["profile"]): r
            for r in runs
            if r["method"] == "sno_es2"
            and r["noise_factor"] == factor
            and r["profile"] in ("confidence", "revalidated")
        }
        differences = defaultdict(list)
        for m in range(100, 106):
            for o in (3000, 3001):
                differences[m].append(
                    by_case[m, o, "revalidated"]["discounted_return"]
                    - by_case[m, o, "confidence"]["discounted_return"]
                )
        values = [np.mean(v) for v in differences.values()]
        paired.append(
            {
                "noise_factor": factor,
                "revalidated_minus_retained_c4_mean_return": float(np.mean(values)),
                "paired_map_sd": float(np.std(values, ddof=1)),
                "map_pairs": 6,
            }
        )
    return summary, runs, aggregates, totals, paired


def scene(title, seconds, script=""):
    return deck.base(title, 3, seconds, script)


def stopping_chart(aggregates):
    rows = [
        ("Basic c=2", "source", "sno_basic", "basic"),
        ("ES2 retained\nc=2", "source", "sno_es2", "source"),
        ("ES2 retained\nc=4", "confidence", "sno_es2", "confidence"),
        ("ES2 revalidated\nc=4", "revalidated", "sno_es2", "revalidated"),
    ]
    with plt.rc_context({"font.size": 17, "axes.labelsize": 18}):
        fig, ax = plt.subplots(figsize=(11.3, 5.55))
        fig.subplots_adjust(left=0.1, right=0.98, bottom=0.19, top=0.9)
        deck.chart_style(ax)
        values = [
            aggregates[p, m, 1]["exploration_actions_mean"] for _, p, m, _ in rows
        ]
        ax.bar(
            range(4), values, color=[deck.COLORS[c] for _, _, _, c in rows], width=0.58
        )
        ax.set(ylim=(0, 183), ylabel="Mean safety-exploration actions")
        ax.set_xticks(range(4), [r[0] for r in rows], fontsize=16)
        for i, value in enumerate(values):
            ax.text(
                i, value + 4, f"{value:.2f}", ha="center", fontsize=18, color=deck.INK
            )
        ax.axhline(160, color="#81919a", linestyle="--", linewidth=1)
        fig.savefig(ASSETS / "result_stopping.png", dpi=180, facecolor="white")
        fig.savefig(ASSETS / "result_stopping.svg")
        plt.close(fig)
    deck.write_csv(
        ASSETS / "stopping_results.csv",
        [
            {
                "condition": label.replace("\n", " "),
                "mean_exploration_actions": aggregates[p, m, 1][
                    "exploration_actions_mean"
                ],
                "es2_fraction": aggregates[p, m, 1]["es2_triggered_mean"],
                "runs": 12,
                "maps": 6,
            }
            for label, p, m, _ in rows
        ],
    )


def build_slides(aggregates):
    path = REPORT / f"{PREFIX}_rehearsal.md"
    current = (
        read_speeches(path) if path.exists() else {s.number: s.speech for s in TALK}
    )
    scripts = [current[TALK[i].number] for i in RESULT_TALK_INDICES]
    scenes = []
    saved = (
        aggregates["source", "sno_basic", 1]["exploration_actions_mean"]
        - aggregates["source", "sno_es2", 1]["exploration_actions_mean"]
    )
    specs = [
        (
            "Reward across the same action budget",
            TALK[1].seconds,
            "result_reward_curves.png",
            "Read the curves",
            "Matched noise\n12 runs / condition\n6 independent maps\nγ = 0.99",
            "Means include exploration and travel reward. Oracle uses privileged true-field information.",
        ),
        (
            "Safety and reward must be read together",
            TALK[2].seconds,
            "result_matched_noise.png",
            "Certificate rules",
            "Retained:\nKeeps earlier approvals.\nRevalidated:\nChecks current bounds.\nSafety multiplier = 4.",
            "Return error bars are map SD, not confidence intervals. Equal zero counts do not prove equivalence.",
        ),
        (
            "Robustness when sensor noise is misspecified",
            TALK[3].seconds,
            "result_noise_robustness.png",
            "Unsafe totals",
            "Retained c=2: 98\nRetained c=4: 2\nRevalidated c=4: 0\n36 runs / profile",
            "Revalidation records zero violations in this finite study; this is empirical evidence, not a safety proof.",
        ),
        (
            "ES2 shifts actions from exploration to reward",
            TALK[4].seconds,
            "result_stopping.png",
            "Same-c benefit",
            f"Basic vs ES2 c=2:\n{saved:.2f} actions shifted\n{100 * saved / 160:.1f}% of the budget\n160 total actions",
            "At matched noise, ES2 reallocates the fixed action budget. This does not establish a runtime speedup.",
        ),
    ]
    for i, (title, seconds, image, heading, body, caption) in enumerate(specs):
        s = scene(title, seconds, scripts[i])
        deck.picture(s, ASSETS / image, 0.6, 2.0, 10.85, 5.05)
        deck.card(s, heading, body, 11.65, 2.2, 3.65, 4.15)
        deck.callout(
            s, caption, y=7.22, color=deck.GOLD if i == 2 else deck.PALE, size=19
        )
        deck.text(s, f"Result {i + 1} / 4", 13.35, 8.53, 1.95, 11, deck.MUTED)
        scenes.append(s)

    s = deck.Slide("Thank You", 3, TALK[6].seconds, scripts[4])
    deck.rect(s, 0, 0, deck.WIDTH, deck.HEIGHT, deck.INK)
    deck.rect(s, 0, 0, 0.13, deck.HEIGHT, deck.TEAL)
    for value, size, y, color, bold in [
        ("Thank You", 68, 3.05, "#ffffff", True),
        ("SNO-MDP · Results and evaluation", 23, 4.55, "#c7dfdf", False),
        ("Next: live dashboard demonstration", 20, 6.15, "#c7dfdf", False),
    ]:
        width = deck.font(size, bold).getlength(value) / (4 * 72)
        deck.text(s, value, (deck.WIDTH - width) / 2, y, width + 0.1, size, color, bold)
    deck.rect(s, 7.3, 5.45, 1.4, 0.04, "#dcb36a")
    scenes.append(s)
    return scenes


def detailed_notes():
    path = REPORT / f"{PREFIX}_rehearsal.md"
    write_speaker_notes(REPORT / f"{PREFIX}_speaker_notes.md", read_speeches(path))


def browser_deck(scenes):
    sections = []
    for i, s in enumerate(scenes, 1):
        sections.append(
            f'<section class="slide" {"hidden" if i != 1 else ""}><img src="{ASSETS.name}/slide_{i:02d}.png" alt="{html.escape(s.title)}">'
            f"<details><summary>Speaker notes</summary><pre>{html.escape(s.notes)}</pre></details></section>"
        )
    (REPORT / f"{PREFIX}.html").write_text(
        """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Member 3 · SNO-MDP results</title><style>body{margin:0;background:#102d36;color:#edf5f5;font:14px Arial}nav{display:flex;gap:10px;align-items:center;padding:12px;flex-wrap:wrap}button,a{background:#edf6f5;color:#145c62;border:0;border-radius:5px;padding:9px 12px;cursor:pointer;text-decoration:none}.slide{max-width:1420px;margin:auto}.slide img{width:100%;display:block}details{padding:16px 24px}pre{white-space:pre-wrap;font:14px/1.6 Arial}[hidden]{display:none!important}#position{margin-left:auto}</style>
<nav><button id="prev">← Previous</button><button id="next">Next →</button><a href="member3_results_speaker_notes.md">Detailed preparation notes</a><span id="position"></span></nav>"""
        + "".join(sections)
        + """<script>let index=0;const slides=[...document.querySelectorAll('.slide')];function show(){const n=slides.length;index=Math.max(0,Math.min(n-1,index));slides.forEach((s,i)=>s.hidden=i!==index);document.getElementById('position').textContent=(index+1)+' / '+n+' · Member 3';document.getElementById('prev').disabled=index===0;document.getElementById('next').disabled=index===n-1;}document.getElementById('next').onclick=()=>{index++;show()};document.getElementById('prev').onclick=()=>{index--;show()};document.onkeydown=e=>{if(e.target.closest('details'))return;if(e.key==='ArrowRight'){e.preventDefault();index++;show()}if(e.key==='ArrowLeft'){e.preventDefault();index--;show()}};show();</script></html>"""
    )


def main():
    ASSETS.mkdir(exist_ok=True)
    write_rehearsal(REPORT / f"{PREFIX}_rehearsal.md")
    actual = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (ROOT / "sno_mdp").glob("*.py")
    }
    assert (
        actual
        == json.loads((REPORT / "frozen_protocol.json").read_text())["source_sha256"]
    )
    summary, runs, aggregates, totals, paired = load_evidence()
    deck.result_charts(summary, runs, assets=ASSETS)
    stopping_chart(aggregates)
    scenes = build_slides(aggregates)
    assert len(scenes) == 5 and sum(s.seconds for s in scenes) == 220
    assert all(not s.backup for s in scenes) and scenes[-1].title == "Thank You"
    for i, s in enumerate(scenes, 1):
        for e in s.elements:
            assert (
                0 <= e["x"]
                and 0 <= e["y"]
                and e["x"] + e["w"] <= 16.02
                and e["y"] + e["h"] <= 9.02
            ), (i, e)
    title = "SNO-MDP · Member 3 results and evaluation"
    deck.powerpoint(
        scenes,
        output=REPORT / f"{PREFIX}.pptx",
        title=title,
        subject="Four results slides plus Thank You; subset of the five-minute team results talk",
        presenter_slot="Member 3 full talk: five minutes across team slides 11–17. This standalone subset takes 3:40; design and dashboard evidence take another 1:20.",
    )
    deck.render(scenes, output=REPORT / f"{PREFIX}.pdf", assets=ASSETS, title=title)
    for stale in ASSETS.glob("slide_*.png"):
        if int(stale.stem.removeprefix("slide_")) > len(scenes):
            stale.unlink()
    browser_deck(scenes)
    detailed_notes()
    evidence = {
        "summary": summary,
        "noise_condition_counts": totals,
        "paired_c4_return_differences": paired,
        "main_slides": 4,
        "total_slides": 5,
        "result_slides": 4,
        "closing_slides": 1,
        "backup_slides": 0,
        "slide_titles": [s.title for s in scenes],
        "slide_seconds": [s.seconds for s in scenes],
        "handoff_included_in_slide": 5,
        "target_seconds": 220,
        "full_team_talk_target_seconds": 300,
        "maximum_seconds": 240,
        "corresponding_team_result_slides": [12, 13, 14, 15],
        "local_preview_result_slides": [10, 11, 12, 13],
        "full_team_deck_overwritten": False,
        "source_matches_frozen": True,
    }
    (ASSETS / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    write_rehearsal(REPORT / f"{PREFIX}_rehearsal.md")
    print(
        "Built member3_results.pptx, .pdf, .html, rehearsal script and verified evidence."
    )
    print(
        "4 results slides + Thank You = 5 total · full seven-slide Member 3 talk: 5:00."
    )


if __name__ == "__main__":
    main()
