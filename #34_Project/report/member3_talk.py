"""Five-minute Member 3 script, numbered for slides 11–17 in the team deck."""

from dataclasses import dataclass
from pathlib import Path
import re


@dataclass(frozen=True)
class TalkSlide:
    number: int
    title: str
    seconds: int
    cue: str
    speech: str


TALK = (
    TalkSlide(
        11,
        "Experimental design: freeze first, evaluate fresh cases",
        40,
        "Point to six maps, two repeats, three noise factors, and the common 160-action budget.",
        """Before showing the graphs, here is our evaluation setup. We froze the settings before testing six new 20-by-20 maps, with two sensor-noise repeats and three assumed-noise settings.

We completed 192 runs, including reference runs. Every run used 160 actions, including travel and exploration. This makes the budget comparison fair. To test revalidation separately from confidence tuning, we compare methods with the same safety confidence multiplier.""",
    ),
    TalkSlide(
        12,
        "Reward across the same action budget",
        45,
        "Point to the axes, then compare the grey and teal curves. Identify the dashed reference.",
        """The horizontal axis shows actions; the vertical axis shows discounted return, meaning collected reward with later rewards weighted less.

Compare grey SNO-MDP (Safe Near-Optimal Markov Decision Process, the basic method) with teal ES² (Early Stopping of Exploration of Safety).

With the confidence multiplier at two, early stopping earns 177.56 versus 158.93 for the basic method. The dashed oracle knows the true environment and is a privileged reference. Higher reward alone does not establish safer performance.""",
    ),
    TalkSlide(
        13,
        "Safety and reward must be read together",
        75,
        "Explain the error bars, define the two permission rules, then compare the two conditions at four.",
        """Here, read reward and safety together. The error bars show SD (standard deviation: variation across six map averages), not confidence intervals.

The label c (safety confidence multiplier) controls the width of the model's safety margin. Four requires stronger evidence than two.

Retained means an earlier learned safety permission can remain active even when the current bound fails. Revalidated means learned permissions are checked against current bounds; unsupported permissions are excluded before planning.

At multiplier two, the average is 2.5 unsafe actions per run. Both versions at four record zero here, with lower mean reward. Comparing these two at the same multiplier isolates the permission rule. Since both have zero violations here, this slide alone does not show an additional safety benefit from revalidation.""",
    ),
    TalkSlide(
        14,
        "Robustness when sensor noise is misspecified",
        50,
        "Explain the three assumptions, then point to the side-panel totals: 98, 2, and 0.",
        """Here the actual altitude-sensor noise stays fixed; we change only the assumed noise standard deviation.

A factor of 0.5 underestimates noise, one matches it, and two overestimates it. The graph shows unsafe actions per run; the side panel gives totals across 36 runs per method.

Retained at two records 98 unsafe actions, retained at four records two, and revalidated at four records zero. The two retained-at-four violations occur in one underestimated-noise run. This is limited experimental evidence, not a universal guarantee.""",
    ),
    TalkSlide(
        15,
        "ES2 shifts actions from exploration to reward",
        40,
        "Compare the first two bars and point to the 42.58-action difference.",
        """This graph shows how early stopping changes the fixed budget. Compare the first two bars, both using the confidence multiplier of two.

The basic method spends all 160 actions exploring safety. Early stopping averages 117.42, reallocating 42.58 actions—26.6 percent of the budget—to reward optimization.

Every run still uses 160 actions and collects reward during exploration. This shows a change in priorities, not faster computation.""",
    ),
    TalkSlide(
        16,
        "Saved dashboard evidence",
        40,
        "Identify the saved replay. Point to the failed current bound and the violation count at step 97.",
        """This is saved dashboard evidence. Before the move at step 96, the LCB (lower confidence bound: the model's conservative safety estimate) is about minus 0.281, below the threshold of minus 0.25.

The current bound therefore cannot certify the move, but an earlier retained permission allows it. At step 97, one actual violation is recorded.

This single trajectory illustrates the mechanism; the earlier graphs summarize multiple runs. Srinivas will show the interactive replay.""",
    ),
    TalkSlide(
        17,
        "Thank You",
        10,
        "Hand over to Srinivas without introducing another result.",
        "Thank you. Srinivas will now demonstrate the dashboard and compare individual experiments.",
    ),
)

assert sum(slide.seconds for slide in TALK) == 300


def clock(seconds):
    return f"{seconds // 60}:{seconds % 60:02d}"


def read_speeches(path):
    """Read current user wording; ignore stage cues and timing metadata."""
    value = Path(path).read_text()
    headings = list(re.finditer(r"(?m)^## Slide (1[1-7]) — [^\n]+\n", value))
    speeches = {}
    for i, heading in enumerate(headings):
        end = headings[i + 1].start() if i + 1 < len(headings) else len(value)
        body = value[heading.end() : end].partition("\n---\n")[0]
        paragraphs = [
            paragraph.strip()
            for paragraph in body.strip().split("\n\n")
            if paragraph.strip() and not paragraph.lstrip().startswith("*")
        ]
        speeches[int(heading.group(1))] = "\n\n".join(paragraphs)
    if set(speeches) != set(range(11, 18)):
        raise ValueError("Expected rehearsal sections numbered 11–17.")
    return speeches


def write_rehearsal(path, *, replace=False):
    path = Path(path)
    if path.exists() and not replace:
        return
    lines = [
        "# Member 3 — five-minute results presentation",
        "",
        "**Slides 11–17 · five minutes total, including the handoff.** Speak the paragraphs; italic cues are not spoken. Abbreviations are expanded in brackets when first used.",
        "",
    ]
    elapsed = 0
    for slide in TALK:
        lines.extend(
            [
                f"## Slide {slide.number} — {slide.title}",
                "",
                f"*{clock(elapsed)}–{clock(elapsed + slide.seconds)} · {slide.seconds} seconds. {slide.cue}*",
                "",
                slide.speech,
                "",
            ]
        )
        elapsed += slide.seconds
    path.write_text("\n".join(lines).rstrip() + "\n")


def write_speaker_notes(path, speeches=None):
    speeches = speeches or {slide.number: slide.speech for slide in TALK}
    lines = [
        "# Member 3 — speaker notes for slides 11–17",
        "",
        "**Five minutes total.** The spoken paragraphs match the [rehearsal](member3_results_rehearsal.md). The final preparation section is not part of the timed talk.",
        "",
        "| Slide | Focus | Allocation |",
        "|---|---|---|",
    ]
    lines.extend(f"| {s.number} | {s.title} | {s.seconds} seconds |" for s in TALK)
    elapsed = 0
    for slide in TALK:
        lines.extend(
            [
                "",
                f"## Slide {slide.number} — {slide.title}",
                "",
                f"*{clock(elapsed)}–{clock(elapsed + slide.seconds)}. {slide.cue}*",
                "",
                speeches[slide.number],
            ]
        )
        elapsed += slide.seconds
    lines.extend(
        [
            "",
            "---",
            "",
            "## Preparation only — do not add to the five-minute speech",
            "",
            "**Retained:** previously accepted action permissions remain in the accepted set. A later failing current safety bound does not automatically revoke an earlier learned permission; graph reachability and returnability still restrict execution.",
            "",
            "**Revalidated:** learned permissions require current model support before execution, with reachability and returnability checked again. Supplied initial certificates remain available. Unsupported permissions can become executable again if later observations support them. This does not establish that the model's current bounds are always correct.",
            "",
            "For example, an action could pass yesterday's bound and fail today's updated bound. Retained can preserve that learned permission. Revalidated excludes it until it is supported again, unless it belongs to the supplied initial certificates.",
            "",
            "| Label on a slide | Full name and meaning |",
            "|---|---|",
            "| SNO-MDP | Safe Near-Optimal Markov Decision Process: the base exploration-then-reward method |",
            "| ES² / ES2 | Early Stopping of Exploration of Safety: stops the safety-exploration phase when its auxiliary test passes |",
            "| SD | Standard deviation across six map averages: the plotted return variability |",
            "| STD | Standard deviation of sensor noise: the quantity used in the assumed/actual noise ratio |",
            "| c | Safety confidence multiplier: scales uncertainty in the safety bound; this is not the reward multiplier |",
            "| γ | Gamma, the discount factor: weights later rewards; 0.99 in this study |",
            "| GP | Gaussian process: the model used to predict a function and its uncertainty |",
            "| LCB | Lower confidence bound: the conservative estimate used to check an action's safety margin |",
            "| UCB | Upper confidence bound: predicted reward plus an uncertainty bonus |",
            "| Map 100 / obs 3000 | Map seed 100 and observation seed 3000; the demo is 20×20 with 160 actions |",
            "",
            "At matched noise, ES² mean returns are 177.56 / 157.18 / 155.31 and mean unsafe-action counts are 2.5 / 0 / 0 for retained at two / retained at four / revalidated at four. Across all noise factors, unsafe-action totals are 98 / 2 / 0, with 36 runs per condition. The two violations in the retained-at-four condition occur in one run.",
            "",
            "Evidence: [final results](../results/final/summary.json), [evaluation settings](../configs/research_final.json), [method audit](../docs/method.md), and [demo settings](../configs/demo_source.json).",
        ]
    )
    Path(path).write_text("\n".join(lines).rstrip() + "\n")


def presenter_note(slide, speech):
    return (
        f"MEMBER 3 | Team slide {slide.number} | Target {slide.seconds} seconds\n\n"
        f"DO NOT SPEAK: {slide.cue}\n\n{speech}"
    )
