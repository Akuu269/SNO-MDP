"""One CLI for benchmark runs, controlled studies, and report figures."""

import argparse
from dataclasses import replace
from pathlib import Path

from .config import Config
from .experiment import LEARNING_METHODS, run_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Execute one benchmark episode")
    run.add_argument("--config", type=Path, default=Path("configs/released.json"))
    run.add_argument(
        "--method", choices=(*LEARNING_METHODS, "oracle"), default="sno_es2"
    )
    run.add_argument("--output", type=Path)
    run.add_argument("--horizon", type=int)
    study = sub.add_parser("study", help="Run a matched map/seed/noise suite")
    study.add_argument("--suite", type=Path, default=Path("configs/study_pilot.json"))
    study.add_argument("--output", type=Path)
    plot = sub.add_parser(
        "plot", help="Export an altitude/reward overview from a saved run"
    )
    plot.add_argument("--run", type=Path, required=True)
    diagnose = sub.add_parser(
        "diagnose", help="Audit saved GP confidence and certificate history"
    )
    target = diagnose.add_mutually_exclusive_group(required=True)
    target.add_argument("--run", type=Path)
    target.add_argument("--study", type=Path)
    diagnose.add_argument(
        "--no-plots",
        action="store_true",
        help="Write records and reports without figures",
    )
    research = sub.add_parser("research", help="Run controlled research policies")
    research.add_argument("--suite", type=Path, required=True)
    research.add_argument("--output", type=Path, required=True)
    research.add_argument("--protocol", type=Path)
    freeze = sub.add_parser("freeze", help="Freeze final settings before evaluation")
    freeze.add_argument("--suite", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    verify = sub.add_parser("verify", help="Verify all saved study actions")
    verify.add_argument("--study", type=Path, required=True)
    reproduce = sub.add_parser(
        "reproduce", help="Rerun a case in a temporary directory"
    )
    reproduce.add_argument("--run", type=Path, required=True)
    sub.add_parser("package", help="Build report, slides, and submission ZIP")
    args = parser.parse_args()
    try:
        if args.command == "run":
            config = Config.load(args.config)
            if args.horizon is not None:
                config = replace(config, horizon=args.horizon)
            output = args.output or Path("results") / args.config.stem / args.method
            result = run_experiment(config, output, args.method)
            print(
                "%s | %d actions | discounted return %.4f | unsafe actions %d"
                % (
                    args.method,
                    result["physical_actions"],
                    result["discounted_return"],
                    result["unsafe_actions"],
                )
            )
            print(
                "Transition: %s at %s; stop: %s"
                % (
                    result["transition_reason"],
                    result["transition_step"],
                    result["stop_reason"],
                )
            )
            print("Replay:", output / "index.html")
        elif args.command == "study":
            from .benchmark import run_benchmark

            output = args.output or Path("results") / args.suite.stem
            result = run_benchmark(args.suite, output)
            print(
                "Completed %d runs; %d model failures. Comparison: %s"
                % (result["run_count"], result["failed_runs"], output / "index.html")
            )
        elif args.command == "plot":
            from .figures import plot_overview

            plot_overview(args.run)
            print("Figure:", args.run / "overview.png")
        elif args.command == "diagnose":
            from .diagnostics import diagnose_run, diagnose_study

            if args.run:
                result = diagnose_run(args.run, plots=not args.no_plots)
                print(
                    "Diagnosed %d actions; unsafe actions %d"
                    % (result["executed_actions"], result["unsafe_actions"])
                )
            else:
                result = diagnose_study(args.study, plots=not args.no_plots)
                print(
                    "Diagnosed %d learner runs; unsafe actions %d"
                    % (result["learner_runs"], result["unsafe_actions"])
                )
            print("Safety report:", (args.run or args.study) / "safety/index.html")
        elif args.command == "research":
            from .research import run_research

            print(run_research(args.suite, args.output, args.protocol))
        elif args.command == "freeze":
            from .research import freeze_research

            result = freeze_research(args.suite, args.output)
            print("Frozen", len(result["planned_cases"]), "cases:", args.output)
        elif args.command == "verify":
            from .research import verify_research

            print(verify_research(args.study))
        elif args.command == "reproduce":
            from .submission import reproduce_run

            print(reproduce_run(args.run))
        else:
            from .submission import build_submission

            print(build_submission())
    except (ValueError, RuntimeError, OSError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
