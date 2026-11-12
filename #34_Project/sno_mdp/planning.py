"""Fully informed, directed-action-safe reference planner."""

from dataclasses import dataclass
import numpy as np
from .environment import ACTIONS
from .graph import TransitionGraph
from .reward_planning import solve_discounted


@dataclass(frozen=True)
class ReferencePlan:
    values: np.ndarray
    reachable: np.ndarray
    actions: np.ndarray
    policy: dict
    iterations: int
    bellman_residual: float


def plan_reference(world, discount=0.99, tolerance=1e-8, max_iterations=5000):
    graph = TransitionGraph.from_grid(world.grid)
    nodes, actions = graph.executable(
        world.true_action_safety(), world.initial_actions[:, 0]
    )
    plan = solve_discounted(
        graph,
        world.reward.ravel(),
        nodes,
        discount,
        tolerance,
        max_iterations,
        action_mask=actions,
    )
    policy = {
        s: ACTIONS[int(plan.policy[i])]
        for i, s in enumerate(world.grid.states())
        if nodes[i]
    }
    return ReferencePlan(
        plan.values.reshape(world.altitude.shape),
        nodes.reshape(world.altitude.shape),
        actions,
        policy,
        plan.iterations,
        plan.bellman_residual,
    )
