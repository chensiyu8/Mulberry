from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MultiModalSample:
    """A single multimodal QA example."""

    sample_id: str
    image_path: str
    question: str
    gt_answer: str
    meta: dict[str, Any]


@dataclass(frozen=True)
class ReasoningState:
    """
    Reasoning state s_t.

    We represent s_t as:
    - prefix_text: current reasoning prefix (includes image description + rationales + steps so far)
    - step_index: next step index to generate (1-based for "### Step k:")
    """

    prefix_text: str
    step_index: int


@dataclass(frozen=True)
class ReasoningAction:
    """A single intermediate reasoning step a_t (natural language)."""

    step_text: str  # must start with "### Step k:"
    vision_reward: float

