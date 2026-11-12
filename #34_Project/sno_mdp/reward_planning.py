"""Belief-only discounted planning and Section 4.3 ES2 (not P-ES2).

Actions earn successor-state rewards, including stay. Planning never receives
latent maps. Bounds are used as written, without clipping negative LCB values.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BeliefPlan:
    values: np.ndarray
    policy: np.ndarray
    domain: np.ndarray
    iterations: int
    bellman_residual: float


def solve_discounted(
    graph,
    rewards,
    domain,
    discount=0.95,
    tolerance=1e-10,
    max_iterations=10000,
    action_mask=None,
    action_rewards=None,
):
    rewards = np.array(rewards, dtype=float, copy=True)
    domain = graph.mask(domain).copy()
    if rewards.shape != (graph.size,) or not np.isfinite(rewards).all():
        raise ValueError("Rewards must be finite with one value per state")
    if not domain.any():
        raise ValueError("Planning domain is empty")
    if not np.isfinite(discount) or not 0 <= discount < 1:
        raise ValueError("Discount must be in [0, 1)")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Tolerance must be finite and positive")
    if type(max_iterations) is not int or max_iterations < 1:
        raise ValueError("Iteration limit must be a positive integer")
    successors = graph.transitions
    allowed = domain[successors] & graph.available
    if action_mask is not None:
        mask = np.asarray(action_mask, bool)
        if mask.shape != successors.shape:
            raise ValueError("Action mask shape mismatch")
        allowed &= mask
    if np.any(domain & ~allowed.any(axis=1)):
        raise ValueError("A planning-domain state has no admissible successor")
    immediate = (
        rewards[successors]
        if action_rewards is None
        else np.asarray(action_rewards, float)
    )
    if immediate.shape != successors.shape or not np.isfinite(immediate).all():
        raise ValueError("Invalid action reward matrix")
    values = np.zeros(graph.size)

    def backup(current):
        q = np.where(allowed, immediate + discount * current[successors], -np.inf)
        updated = np.zeros(graph.size)
        updated[domain] = q[domain].max(axis=1)
        return updated, q

    for iteration in range(1, max_iterations + 1):
        updated, _ = backup(values)
        change = np.max(np.abs(updated[domain] - values[domain]))
        values = updated
        if change <= tolerance:
            break
    else:
        raise RuntimeError("Belief value iteration did not converge")
    updated, q = backup(values)
    residual = float(np.max(np.abs(updated[domain] - values[domain])))
    # Policies outside the active domain can enter it in one admissible action.
    # States without such an entry are marked -1, never silently mapped to stay.
    policy = np.full(graph.size, -1, dtype=int)
    can_enter = allowed.any(axis=1)
    policy[can_enter] = np.argmax(q[can_enter], axis=1)
    for array in (values, policy, domain):
        array.setflags(write=False)
    return BeliefPlan(values, policy, domain, iteration, residual)


@dataclass(frozen=True)
class ES2Result:
    stop: bool
    successors: np.ndarray
    auxiliary_reward: np.ndarray
    plan: BeliefPlan


def check_es2(
    graph,
    pessimistic_actions,
    optimistic_actions,
    reward_mean,
    reward_std,
    multiplier=3.0,
    discount=0.95,
    tolerance=1e-10,
    max_iterations=10000,
):
    pessimistic_actions = np.asarray(pessimistic_actions, bool)
    optimistic_actions = np.asarray(optimistic_actions, bool)
    if (
        pessimistic_actions.shape != graph.transitions.shape
        or optimistic_actions.shape != graph.transitions.shape
    ):
        raise ValueError("ES2 requires state-action masks")
    if np.any(pessimistic_actions & ~optimistic_actions):
        raise ValueError("Certified actions must be optimistic")
    pessimistic = pessimistic_actions[:, 0]
    optimistic = optimistic_actions[:, 0]
    mean, std = np.asarray(reward_mean, float), np.asarray(reward_std, float)
    if not pessimistic.any() or np.any(pessimistic & ~optimistic):
        raise ValueError(
            "ES2 requires a nonempty certified subset of the optimistic domain"
        )
    if (
        mean.shape != (graph.size,)
        or std.shape != mean.shape
        or not np.isfinite(mean).all()
        or not np.isfinite(std).all()
        or np.any(std < 0)
    ):
        raise ValueError("Invalid reward belief")
    if not np.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("Reward multiplier must be finite and positive")
    successors = graph.transitions
    auxiliary = mean[successors] + multiplier * std[successors]
    auxiliary[pessimistic_actions] = (mean[successors] - multiplier * std[successors])[
        pessimistic_actions
    ]
    plan = solve_discounted(
        graph,
        mean,
        optimistic,
        discount,
        tolerance,
        max_iterations,
        action_mask=optimistic_actions,
        action_rewards=auxiliary,
    )
    states = np.flatnonzero(pessimistic)
    successors = np.zeros(graph.size, bool)
    successors[graph.transitions[states, plan.policy[states]]] = True
    stop = bool(np.all(~successors | pessimistic))
    successors.setflags(write=False)
    auxiliary.setflags(write=False)
    return ES2Result(stop, successors, auxiliary, plan)
