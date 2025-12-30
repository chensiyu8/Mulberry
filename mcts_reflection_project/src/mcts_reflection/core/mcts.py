from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

from .types import ReasoningAction, ReasoningState


@dataclass
class MCTSConfig:
    max_iterations: int = 24
    max_depth: int = 8
    ucb_c: float = 1.2

    # Expansion
    num_expand: int = 12
    keep_topk: int = 5

    # Simulation
    num_rollouts: int = 4

    # Reward fusion: R = alpha*R_ans + beta*R_vis + gamma*R_reflect
    alpha_answer: float = 1.0
    beta_vision: float = 0.6
    gamma_reflect: float = 0.4

    reward_threshold: float = 0.0


class Node:
    def __init__(
        self,
        *,
        state: ReasoningState,
        parent: "Node | None" = None,
        action_from_parent: ReasoningAction | None = None,
    ):
        self.state = state
        self.parent = parent
        self.action_from_parent = action_from_parent

        self.children: list[Node] = []
        self.visits: int = 0
        self.value: float = 0.0

        # bookkeeping for rewards
        self.cum_vision_reward: float = 0.0 if parent is None else float(parent.cum_vision_reward + (action_from_parent.vision_reward if action_from_parent else 0.0))
        self.depth: int = 0 if parent is None else parent.depth + 1

        # simulation traces for data export
        self.rollouts: list[dict[str, Any]] = []

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def ucb(self, c: float) -> float:
        if self.parent is None:
            return self.value
        return self.value + c * math.sqrt(math.log(self.parent.visits + 1) / (self.visits + 1))

    def best_child(self, c: float) -> "Node":
        return max(self.children, key=lambda n: n.ucb(c))

    def add_child(self, child: "Node") -> None:
        self.children.append(child)


ExpandFn = Callable[[ReasoningState, int], list[ReasoningAction]]
TransitionFn = Callable[[ReasoningState, ReasoningAction], ReasoningState]
SimulateFn = Callable[[ReasoningState, int], list[dict[str, Any]]]
RewardFn = Callable[[Node, list[dict[str, Any]]], float]


class MCTS:
    """
    Standard MCTS with UCB selection + custom (multi-signal) reward.

    The algorithm follows:
    - Selection: descend by UCB until leaf
    - Expansion: expand leaf with candidate actions, keep top-k by external criterion
    - Simulation: rollout from each expanded child for N times
    - Backpropagation: update value/visits along path
    """

    def __init__(
        self,
        cfg: MCTSConfig,
        *,
        expand: ExpandFn,
        transition: TransitionFn,
        simulate: SimulateFn,
        compute_reward: RewardFn,
    ):
        self.cfg = cfg
        self.expand = expand
        self.transition = transition
        self.simulate = simulate
        self.compute_reward = compute_reward

    def _select(self, root: Node) -> Node:
        cur = root
        while cur.children:
            cur = cur.best_child(self.cfg.ucb_c)
        return cur

    def _backprop(self, node: Node, reward: float) -> None:
        cur: Node | None = node
        while cur is not None:
            cur.visits += 1
            cur.value += (reward - cur.value) / float(cur.visits)
            cur = cur.parent

    def search(self, root_state: ReasoningState) -> Node:
        root = Node(state=root_state)

        for _ in range(self.cfg.max_iterations):
            leaf = self._select(root)
            if leaf.depth >= self.cfg.max_depth:
                # only simulate at terminal depth
                rollouts = self.simulate(leaf.state, self.cfg.num_rollouts)
                leaf.rollouts = rollouts
                reward = self.compute_reward(leaf, rollouts)
                self._backprop(leaf, reward)
                continue

            actions = self.expand(leaf.state, self.cfg.num_expand)
            # keep top-k by action's vision_reward (already computed upstream)
            actions = sorted(actions, key=lambda a: a.vision_reward, reverse=True)[: self.cfg.keep_topk]

            if not actions:
                rollouts = self.simulate(leaf.state, self.cfg.num_rollouts)
                leaf.rollouts = rollouts
                reward = self.compute_reward(leaf, rollouts)
                self._backprop(leaf, reward)
                continue

            # expand children and simulate each
            for a in actions:
                child_state = self.transition(leaf.state, a)
                child = Node(state=child_state, parent=leaf, action_from_parent=a)
                leaf.add_child(child)

                rollouts = self.simulate(child.state, self.cfg.num_rollouts)
                child.rollouts = rollouts
                reward = self.compute_reward(child, rollouts)
                reward = reward if reward >= self.cfg.reward_threshold else 0.0
                self._backprop(child, reward)

        return root

    @staticmethod
    def pick_best_path(root: Node) -> list[Node]:
        """
        Greedy path by node.value (not UCB), for exporting the best found trajectory.
        """
        path: list[Node] = []
        cur = root
        while cur.children:
            cur = max(cur.children, key=lambda n: n.value)
            path.append(cur)
        return path

