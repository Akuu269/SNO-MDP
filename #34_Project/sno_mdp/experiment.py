"""Source-audited headless benchmark experiments with strict action filtering.

GP beliefs update online in both phases, as in the authors' released driver.
We use certified actions for physical execution and matched total horizons.
See docs/method.md for differences from both source code and paper pseudocode.
"""

import csv
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np

from .agent import SafetyExplorer
from .artifacts import build_manifest, save_json
from .environment import ACTIONS, Evaluator, RoverSession, world_from_config
from .planning import plan_reference
from .reward_planning import check_es2, solve_discounted
from .transition_sets import ConfidenceConflict


LEARNING_METHODS = ("safemdp", "sno_basic", "sno_es2")


def run_experiment(config, output: Path, method="sno_es2", write_replay=True):
    if method not in (*LEARNING_METHODS, "oracle"):
        raise ValueError("Unknown method")
    output = Path(output)
    world = world_from_config(config)
    session = RoverSession(
        world, config.observation_seed, config.reward_noise_std, config.safety_noise_std
    )
    evaluator = Evaluator(world)
    shape = (config.rows, config.cols)
    oracle = method == "oracle"
    reference = (
        plan_reference(
            world, config.discount, config.value_tolerance, config.max_iterations
        )
        if oracle
        else None
    )
    learner = (
        None if oracle else SafetyExplorer(world.grid, session.initial_actions, config)
    )
    phase = "oracle" if oracle else "safety_exploration"
    counts = {"safety_exploration": 0, "reward_optimization": 0, "oracle": 0}
    frames = []
    observations = []
    es2_checks = []
    timings = []
    observation = session.initial_observation
    last_action = last_phase = previous_position = None
    transition_step = transition_reason = None
    cumulative = discounted = 0.0
    unsafe_actions = 0
    stop_reason = diagnostic = None
    true_safe = world.true_action_safety()
    while True:
        observations.append(
            {
                "step": observation.step,
                "position": observation.position,
                "action": last_action.value if last_action else None,
                "used_for_gp": not oracle,
                "samples": [asdict(s) for s in observation.samples],
            }
        )
        began = perf_counter()
        inference_time = 0.0
        action = target = target_action = es2 = None
        plan_residual = None
        if oracle:
            nodes = reference.reachable.ravel()
            optimistic = nodes
            certified = reference.actions
            optimistic_actions = certified
            expanders = np.zeros_like(certified)
        else:
            try:
                snapshot = learner.update(observation)
            except ConfidenceConflict as exc:
                stop_reason, diagnostic = "confidence_conflict", str(exc)
            except (ValueError, RuntimeError) as exc:
                stop_reason, diagnostic = "model_failure", str(exc)
            inference_time = perf_counter() - began
            snapshot = learner.tracker.snapshot
            if snapshot is None:
                nodes = world.initial_actions[:, 0]
                optimistic = nodes
                certified = world.initial_actions
                optimistic_actions = certified
                expanders = np.zeros_like(certified)
            else:
                nodes, optimistic = snapshot.pessimistic, snapshot.optimistic
                certified, optimistic_actions = (
                    snapshot.pessimistic_actions,
                    snapshot.optimistic_actions,
                )
                expanders = snapshot.expanders
        decision_began = perf_counter()
        if (
            not oracle
            and not stop_reason
            and config.certificate_mode == "current_lcb"
            and not nodes[np.ravel_multi_index(session.position, shape)]
        ):
            stop_reason = "certificate_region_lost"
            diagnostic = "Current state lost its reachable/returnable region after current-bound revalidation; motion halted."
        if not stop_reason and session.step_count < config.horizon:
            try:
                if oracle:
                    action = reference.policy[session.position]
                    plan_residual = reference.bellman_residual
                else:
                    if phase == "safety_exploration" and method == "sno_es2":
                        es2 = check_es2(
                            learner.graph,
                            certified,
                            optimistic_actions,
                            learner.reward_prediction.mean,
                            learner.reward_prediction.std,
                            config.reward_confidence_multiplier,
                            config.discount,
                            config.value_tolerance,
                            config.max_iterations,
                        )
                        sources = np.flatnonzero(nodes)
                        successors = learner.graph.transitions[
                            sources, es2.plan.policy[sources]
                        ]
                        outside_sources = sources[~nodes[successors]]
                        es2_checks.append(
                            {
                                "step": session.step_count,
                                "stop": es2.stop,
                                "source_indices": sources.tolist(),
                                "successor_indices": np.flatnonzero(
                                    es2.successors
                                ).tolist(),
                                "outside_source_indices": outside_sources.tolist(),
                                "outside_successor_indices": np.flatnonzero(
                                    es2.successors & ~nodes
                                ).tolist(),
                                "policy": es2.plan.policy.tolist(),
                                "auxiliary_action_reward": es2.auxiliary_reward.tolist(),
                                "bellman_residual": es2.plan.bellman_residual,
                            }
                        )
                    if phase == "safety_exploration" and method != "safemdp":
                        reason = (
                            "es2_condition"
                            if es2 is not None and es2.stop
                            else (
                                "certified_action_accuracy"
                                if learner.accuracy_reached()
                                else None
                            )
                        )
                        if reason:
                            phase = "reward_optimization"
                            transition_step = session.step_count
                            transition_reason = reason
                    if phase == "safety_exploration":
                        action, target, target_action = learner.safety_action(
                            session.position
                        )
                    else:
                        reward_ucb = (
                            learner.reward_prediction.mean
                            + config.reward_confidence_multiplier
                            * learner.reward_prediction.std
                        )
                        plan = solve_discounted(
                            learner.graph,
                            np.minimum(100.0, reward_ucb),
                            nodes,
                            config.discount,
                            config.value_tolerance,
                            config.max_iterations,
                            action_mask=certified,
                        )
                        current = np.ravel_multi_index(session.position, shape)
                        action = ACTIONS[int(plan.policy[current])]
                        plan_residual = plan.bellman_residual
                    source = np.ravel_multi_index(session.position, shape)
                    if not certified[source, ACTIONS.index(action)]:
                        raise ValueError(
                            "Controller selected an uncertified directed action"
                        )
            except (ValueError, RuntimeError) as exc:
                stop_reason, diagnostic = "planning_failure", str(exc)
        if stop_reason is None and session.step_count >= config.horizon:
            stop_reason = "horizon_reached"
        if stop_reason:
            action = target = target_action = None
        decision_seconds = perf_counter() - decision_began
        evaluation = evaluator.evaluate(session.position, previous_position)
        false = certified & ~true_safe

        def reshape(array):
            return array.reshape(shape).tolist()

        prediction = None if oracle else learner.altitude_prediction
        reward_prediction = None if oracle else learner.reward_prediction
        frames.append(
            {
                "step": session.step_count,
                "position": session.position,
                "action": last_action.value if last_action else None,
                "executed_phase": last_phase,
                "next_action": action.value if action else None,
                "next_action_safety_lower": float(
                    learner.transition_prediction.mean[
                        np.ravel_multi_index(session.position, shape),
                        ACTIONS.index(action),
                    ]
                    - config.safety_confidence_multiplier
                    * learner.transition_prediction.std[
                        np.ravel_multi_index(session.position, shape),
                        ACTIONS.index(action),
                    ]
                )
                if action is not None and not oracle
                else None,
                "next_action_initial_certificate": bool(
                    world.initial_actions[
                        np.ravel_multi_index(session.position, shape),
                        ACTIONS.index(action),
                    ]
                )
                if action is not None
                else None,
                "target": target,
                "target_action": target_action,
                "phase": phase,
                "stop_reason": stop_reason,
                "diagnostic": diagnostic,
                "transition_step": transition_step,
                "transition_reason": transition_reason,
                **evaluation,
                "cumulative_reward": cumulative,
                "discounted_return": discounted,
                "unsafe_actions": unsafe_actions,
                "certified_cells": int(nodes.sum()),
                "certified_action_count": int(certified.sum()),
                "optimistic_cells": int(optimistic.sum()),
                "false_certified_actions": int(false.sum()),
                "expander_count": int(expanders.sum()),
                "pessimistic": reshape(nodes),
                "optimistic": reshape(optimistic),
                "certified_actions": certified.tolist(),
                "optimistic_actions": optimistic_actions.tolist(),
                "expander_actions": expanders.tolist(),
                "altitude_mean": reshape(prediction.mean) if prediction else None,
                "altitude_std": reshape(prediction.std) if prediction else None,
                "reward_mean": reshape(reward_prediction.mean)
                if reward_prediction
                else None,
                "reward_std": reshape(reward_prediction.std)
                if reward_prediction
                else None,
                "es2_stop": es2.stop if es2 else None,
                "es2_outside_sources": es2_checks[-1]["outside_source_indices"]
                if es2
                else None,
                "bellman_residual": plan_residual,
            }
        )
        timings.append(
            {
                "step": session.step_count,
                "inference_seconds": inference_time,
                "decision_seconds": decision_seconds,
            }
        )
        if stop_reason:
            break
        previous_position = session.position
        observation = session.step(action)
        scored = evaluator.evaluate(session.position, previous_position)
        cumulative += scored["true_reward"]
        discounted += (
            config.discount ** (session.step_count - 1) * scored["true_reward"]
        )
        unsafe_actions += int(scored["unsafe"])
        counts[phase] += 1
        last_phase = phase
        last_action = action
    summary = {
        "method": method,
        "environment": world.provenance,
        "physical_actions": session.step_count,
        "scheduled_horizon": config.horizon,
        "stop_reason": stop_reason,
        "diagnostic": diagnostic,
        "transition_step": transition_step,
        "transition_reason": transition_reason,
        "phase_actions": counts,
        "budget_exhausted_during_exploration": phase == "safety_exploration"
        and stop_reason == "horizon_reached",
        "initial_certified_cells": len(world.initial_safe),
        "final_certified_cells": frames[-1]["certified_cells"],
        "final_certified_actions": frames[-1]["certified_action_count"],
        "maximum_false_certified_actions": max(
            f["false_certified_actions"] for f in frames
        ),
        "unsafe_actions": unsafe_actions,
        "maximum_violation_severity": max(f["violation_severity"] for f in frames),
        "cumulative_reward": cumulative,
        "discounted_return": discounted,
        "es2_triggered": transition_reason == "es2_condition",
        "es2_checks": len(es2_checks),
        "actual_safety_noise_std": config.safety_noise_std,
        "assumed_safety_noise_std": config.safety_noise_std
        * config.assumed_noise_factor,
        "reward_updates": "online authorized noisy observations in both phases",
        "certificate_mode": config.certificate_mode,
        "safety_confidence_multiplier": config.safety_confidence_multiplier,
        "safety_profile": "source transition differences; certificate mode "
        + config.certificate_mode,
        "not_reproduced": [
            "legacy MuJoCo Engine_GP execution",
            "paper theorem guarantees",
            "P-ES2",
        ],
    }
    maps = {
        "rows": config.rows,
        "cols": config.cols,
        "start": world.start,
        "initial_safe": world.initial_safe,
        "initial_actions": world.initial_actions.tolist(),
        "threshold": world.threshold,
        "spacing": world.spacing,
        "altitude": world.altitude.tolist(),
        "reward": world.reward.tolist(),
        "provenance": world.provenance,
    }
    output.mkdir(parents=True, exist_ok=True)
    for filename, data in [
        ("config.json", config.as_dict()),
        ("map.json", maps),
        ("summary.json", summary),
        ("observations.json.gz", observations),
        ("trajectory.json.gz", frames),
        ("es2_checks.json.gz", es2_checks),
        ("manifest.json", build_manifest(config, maps)),
    ]:
        save_json(
            output / filename,
            data,
            compact=filename.endswith(".gz"),
        )
    columns = [
        "step",
        "phase",
        "action",
        "next_action",
        "certified_cells",
        "certified_action_count",
        "expander_count",
        "false_certified_actions",
        "unsafe_actions",
        "cumulative_reward",
        "discounted_return",
        "stop_reason",
    ]
    for filename, rows, keys in [
        ("progress.csv", frames, columns),
        ("timing.csv", timings, list(timings[0])),
    ]:
        with (output / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows({k: r[k] for k in keys} for r in rows)
    from .visualization import write_visualization

    if write_replay:
        write_visualization(
            output / "index.html", {**maps, "frames": frames, "summary": summary}
        )
    return summary


def run_reference(config, output):
    return run_experiment(config, output, "oracle")
