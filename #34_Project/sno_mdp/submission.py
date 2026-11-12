"""Report, actual presentation PDF, reproduction check, and submission ZIP."""

import hashlib
from pathlib import Path
import tempfile
import textwrap
import zipfile

from .artifacts import read_json, save_json
from .config import Config
from .diagnostic_report import _page, _plot_style
from .experiment import run_experiment

ROOT = Path(__file__).resolve().parent.parent


def reproduce_run(run):
    run = Path(run)
    filenames = (
        "config.json",
        "map.json",
        "summary.json",
        "observations.json.gz",
        "trajectory.json.gz",
        "es2_checks.json.gz",
    )
    with tempfile.TemporaryDirectory(prefix="snomdp-reproduce-") as tmp:
        out = Path(tmp)
        run_experiment(
            Config.load(run / "config.json"),
            out,
            read_json(run / "summary.json")["method"],
            write_replay=False,
        )
        if any(
            (run / name).read_bytes() != (out / name).read_bytes() for name in filenames
        ):
            raise ValueError(
                "Fresh reproduction differs from saved algorithm artifacts"
            )
    result = {
        "checked_files": list(filenames),
        "result": "Byte-identical algorithm artifacts.",
        "limit": "Fresh temporary directory in the same installed environment; independent human reproduction is not attested.",
    }
    save_json(run / "reproduction.json", result)
    return result


def build_submission():
    final, report = ROOT / "results/final", ROOT / "report"
    summary = read_json(final / "summary.json")
    verification = read_json(final / "verification.json")
    calibration = read_json(ROOT / "results/research_calibration/selection.json")
    stopping = read_json(ROOT / "results/research_stopping/summary.json")
    selected = calibration["selected_multiplier"]
    report.mkdir(exist_ok=True)
    table = [
        "| Profile:method | Runs/maps | Mean return | Map SD | Unsafe-run fraction | Failure fraction | ES² fraction |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for a in summary["aggregates"]:
        if a["noise_factor"] in (None, 1):
            table.append(
                f"| {a['profile']}:{a['method']} | {a['runs']}/{a['maps']} | "
                f"{a['discounted_return_mean']:.2f} | {a['discounted_return_map_sd']:.2f} | "
                f"{a['unsafe_run_fraction']:.2%} | {a['failed_mean']:.2%} | {a['es2_triggered_mean']:.2%} |"
            )
    lines = [
        "# SNO-MDP: reproduction, noise robustness, and certificate revalidation",
        "",
        "**Status:** Automated work completed; human team review and names remain to be supplied.",
        "",
        "## Question and contribution",
        "",
        "How do noise assumptions and retention of old safety certificates affect safe exploration, reward, and ES²? "
        "We provide a source-audited headless reimplementation and a current-LCB revalidation extension. "
        "This course project claims empirical implementation and analysis, not a new theorem, exact published curves, or patentability.",
        "",
        "## Method",
        "",
        "Independent 20 × 20 RBF-GP altitude/reward maps, spacing 0.5, discount 0.99, and threshold −0.25. "
        "Directed safety is g(s,a)=z(s)−z(next). Separate GPs learn from noisy local readings; transition uncertainty includes joint covariance. "
        "Exploration uses reachable/returnable certified actions; ES² checks auxiliary-policy successors; reward learning continues online. "
        "[Paper/source deviations](../docs/method.md) are part of the method specification.",
        "",
        "The original profile retains accepted certificates. The extension requires supplied initial certificates or a passing current LCB, "
        "then recomputes reachability/returnability. If the current state leaves that region, motion halts and the run counts as failure. "
        "No evaluator truth selects or vetoes learner actions. Repeated readings use exact Gaussian count/mean sufficient statistics, including jitter divided by count.",
        "",
        "## Development selection and stopping",
        "",
        f"Calibration reran both modes at multipliers 2/3/4 on maps 31/37/41 and two observation seeds. "
        f"The predeclared failure-rate → unsafe-run-rate → negative reference-ratio → multiplier ranking selected **{selected:g}**. "
        "[Selection record](../results/research_calibration/selection.json). No final outcome was used for selection.",
        "",
        f"Stopping validation completed {stopping['completed_runs']} cases at 120/240 actions. Accuracy stops, ES² stops, region loss, "
        "and exploration-budget exhaustion are separate outcomes. [Stopping results](../results/research_stopping/RESULTS.md).",
        "",
        "## Frozen final evaluation",
        "",
        f"{summary['completed_runs']}/{summary['planned_runs']} planned cases completed across {summary['maps']} fresh maps, "
        "two observation seeds, noise factors 0.5/1/2, and a predeclared 160-action horizon. The original c=2 methods, selected-confidence retained ES² control "
        "when distinct, revalidated ES² extension, and privileged oracle use matched conditions.",
        "",
        "Every setup/model/planning/region-loss failure remains in the inventory. Unexecuted actions earn zero extra reward. "
        "Average observation replicates within each map, then average maps equally. SD describes map variability, not confidence intervals. "
        "The oracle is an infinite-horizon policy rolled out for the same finite budget; reference ratios are descriptive.",
        "",
        "Matched-noise results:",
        "",
        *table,
        "",
        f"Across all final conditions: **{summary['failed_runs']} failed runs and {summary['unsafe_actions']} unsafe actions**. "
        "Matched policies and repeated observations are correlated, so these are not independent-trial risk estimates.",
        "",
        "[Every noise condition and result](../results/final/RESULTS.md) · [Dashboard and replays](../results/final/index.html) · "
        "[Frozen protocol](../results/final/protocol.json) · [Every planned case](../results/final/runs.csv)",
        "",
        "## Figures",
        "",
        "- [Reward against actions](../results/final/figures/returns.png)",
        "- [Executable-region growth](../results/final/figures/region_growth.png)",
        "- [Incorrect certificates](../results/final/figures/certificate_errors.png)",
        "- [Safety, failure, exploration and planning cost](../results/final/figures/outcomes.png)",
        "- [Paired map-level differences](../results/final/figures/paired_returns.png)",
        "",
        "## Verification and reproduction",
        "",
        f"Verified all {verification['verified_runs']} final cases and {verification['verified_physical_actions']} physical actions "
        "against directed dynamics, execution certificates, return accounting, map/source hashes, and the frozen protocol. "
        "Numerical, graph, revalidation, fallback, failure, protocol, and integration tests accompany the implementation.",
        "",
        "From the project directory:",
        "",
        "```bash",
        "python3 -m unittest discover -s tests",
        "python3 -m sno_mdp verify --study results/final",
        "python3 -m sno_mdp research --suite configs/research_final.json --output results/final_reproduced --protocol report/frozen_protocol.json",
        "```",
        "",
        "## Limitations",
        "",
        "Finite maps and practical confidence bands establish no uniform safety guarantee. Revalidation can sacrifice reward or completion. "
        "The extension remains distinct from the original method. Kernel/noise assumptions, initial certificates, and observation rules limit transfer. "
        "Native legacy MuJoCo rendering, P-ES², and Mars experiments are outside this submission scope.",
        "",
        "## Presentation, team, and references",
        "",
        "[Presentation PDF](presentation.pdf) · [Browser slides](presentation.html) · [Speaker notes](presentation.md) · [Human team review](team_review.md).",
        "",
        "- [Wachi and Sui, ICML 2020](https://proceedings.mlr.press/v119/wachi20a.html).",
        "- [Local paper](../SNO_MDP_Wachi_Sui_2020_arxiv_v1.pdf).",
        "- [Source revisions, licenses, and checksums](../third_party/README.md).",
        "- [Method audit](../docs/method.md) and [experimental protocol](../docs/experiments.md).",
        "",
    ]
    (report / "report.md").write_text("\n".join(lines), encoding="utf-8")
    slides = [
        (
            "SNO-MDP: safe exploration and reward",
            [
                "Four-person course project; source-audited headless simulation.",
                "Question: how do noise assumptions and certificate retention affect behavior?",
                "Wachi and Sui, ICML 2020; current-LCB revalidation is our project extension.",
            ],
            None,
        ),
        (
            "What the rover knows",
            [
                "Known dynamics and supplied initial safe actions.",
                "Unknown altitude and reward learned from noisy local observations.",
                "True fields belong to the evaluator/oracle only.",
                "Safe action: altitude(source) minus altitude(successor) >= -0.25.",
            ],
            None,
        ),
        (
            "GP inference and SNO-MDP",
            [
                "Separate altitude and reward RBF GPs update online.",
                "Difference variance: Var(z_s)+Var(z_t)-2 Cov(z_s,z_t).",
                "Explore uncertain expandable actions through certified travel.",
                "ES² checks auxiliary-policy successors from every executable state.",
            ],
            None,
        ),
        (
            "Diagnostic finding and extension",
            [
                "Development violations relied on old certificates after current bounds failed.",
                "Baseline retains certificates, following the released source profile.",
                "Extension rechecks current LCBs and rebuilds reachable/returnable paths.",
                "Region-loss halts count as failures; no truth-based action guard.",
            ],
            None,
        ),
        (
            "Development and frozen evaluation",
            [
                f"Selected multiplier {selected:g} from calibration data only.",
                "Stopping tested at 120/240 actions; final horizon fixed at 160.",
                f"{summary['maps']} fresh maps; two observation seeds; three noise assumptions.",
                "Equal budgets, all failures retained, paired comparisons and map-level variability.",
            ],
            None,
        ),
        ("Held-out reward curves", [], "returns.png"),
        ("Executable certificates and errors", [], "certificate_errors.png"),
        ("Safety, completion, exploration and computation", [], "outcomes.png"),
        ("Paired return differences", [], "paired_returns.png"),
        (
            "Findings and practical limits",
            [
                f"Final inventory: {summary['completed_runs']} cases; {summary['failed_runs']} failures; {summary['unsafe_actions']} unsafe actions.",
                "Read per-noise results to assess reward/safety/completion tradeoffs.",
                "Finite experiments do not prove safety or reproduce exact published curves.",
                "Source, protocol, raw records, tests, replays and reproduction commands supplied.",
            ],
            None,
        ),
    ]
    notes, html = [], ""
    with _plot_style():
        from matplotlib.backends.backend_pdf import PdfPages
        from matplotlib.figure import Figure
        import matplotlib.image as mpimg

        with PdfPages(
            report / "presentation.pdf",
            metadata={
                "Title": "SNO-MDP Course Project",
                "CreationDate": None,
                "ModDate": None,
            },
        ) as pdf:
            for i, (title, bullets, image) in enumerate(slides, 1):
                fig = Figure(figsize=(13.33, 7.5), layout="constrained")
                fig.suptitle(title, fontsize=23)
                ax = fig.subplots()
                ax.axis("off")
                if image:
                    ax.imshow(mpimg.imread(final / "figures" / image))
                else:
                    ax.text(
                        0.03,
                        0.9,
                        "\n\n".join("• " + textwrap.fill(t, width=87) for t in bullets),
                        fontsize=18,
                        va="top",
                        transform=ax.transAxes,
                    )
                fig.text(0.95, 0.025, str(i), ha="right", fontsize=10)
                pdf.savefig(fig)
                notes += [
                    f"## Slide {i}: {title}",
                    "",
                    *["- " + b for b in bullets],
                    "",
                ]
                html += "<section style='break-after:page'><h2>" + title + "</h2>"
                if image:
                    notes += [
                        f"Explain axes, map/seed counts, failures, and variability in {image}.",
                        "",
                    ]
                    html += (
                        '<img style="width:100%" src="../results/final/figures/'
                        + image
                        + '" alt="'
                        + title
                        + '">'
                    )
                else:
                    html += (
                        "<ul>"
                        + "".join("<li>" + b + "</li>" for b in bullets)
                        + "</ul>"
                    )
                html += "</section>"
    (report / "presentation.md").write_text(
        "# Presentation and speaker notes\n\n" + "\n".join(notes), encoding="utf-8"
    )
    (report / "presentation.html").write_text(
        _page("SNO-MDP course project", html), encoding="utf-8"
    )
    (report / "team_review.md").write_text(
        "# Human review and actual contributions\n\nNames, actual contributions, and independent human reproduction require team confirmation.\n\n"
        "| Member | Proposed ownership | Actual contribution / cross-review |\n|---|---|---|\n"
        "| Member 1 — add name | Environment, source audit | To confirm |\n"
        "| Member 2 — add name | GP and confidence calibration | To confirm |\n"
        "| Member 3 — add name | Control, ES² and revalidation | To confirm |\n"
        "| Member 4 — add name | Experiments, statistics and report | To confirm |\n\n"
        "- [ ] Each member explains the method and documented deviations.\n"
        "- [ ] Another member reproduces a case in their own environment.\n"
        "- [ ] Add names and confirm actual contributions.\n"
        "- [ ] Check professor-specific format and rehearse the demonstration.\n",
        encoding="utf-8",
    )
    excluded = {".venv", "__pycache__", ".ruff_cache", ".pytest_cache", ".git", "dist"}
    files = [
        p
        for p in ROOT.rglob("*")
        if p.is_file()
        and not any(v in excluded for v in p.relative_to(ROOT).parts)
        and p.name != "submission_manifest.json"
    ]
    save_json(
        report / "submission_manifest.json",
        {
            "files": {
                p.relative_to(ROOT).as_posix(): hashlib.sha256(
                    p.read_bytes()
                ).hexdigest()
                for p in sorted(files)
            },
            "human_review": "Names, actual contributions, independent reproduction and professor format remain to be confirmed.",
        },
    )
    files.append(report / "submission_manifest.json")
    dist = ROOT / "dist"
    dist.mkdir(exist_ok=True)
    target = dist / "SNO_MDP_Project.zip"
    with zipfile.ZipFile(target, "w") as archive:
        for path in files:
            compression = (
                zipfile.ZIP_STORED
                if path.suffix in (".gz", ".png", ".pdf", ".zip", ".npz")
                else zipfile.ZIP_DEFLATED
            )
            archive.write(
                path,
                "SNO_MDP_Project/" + path.relative_to(ROOT).as_posix(),
                compress_type=compression,
            )
    return {
        "report": str(report / "report.md"),
        "slides": str(report / "presentation.pdf"),
        "archive": str(target),
        "files": len(files),
    }
