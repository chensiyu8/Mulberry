from __future__ import annotations

import re


_RE_WS = re.compile(r"\s+")
_RE_PUNCT = re.compile(r"[^\w\u4e00-\u9fff\.\-\/%]+", re.UNICODE)


def normalize_answer(s: str) -> str:
    """
    Best-effort answer normalization for MathVista-like tasks.

    - lowercases
    - strips common wrappers like "Answer:" / "The final answer is:"
    - removes most punctuation while keeping ., -, /, %
    """
    if s is None:
        return ""
    s = str(s).strip()
    s = s.replace("### The final answer is:", "")
    s = s.replace("The final answer is:", "")
    s = s.replace("Final answer:", "")
    s = s.replace("Answer:", "")
    s = s.strip().lower()
    s = _RE_PUNCT.sub(" ", s)
    s = _RE_WS.sub(" ", s).strip()
    return s


def extract_final_answer_from_reasoning(reasoning: str) -> str:
    """
    Extract the last segment after the final-answer marker used in this repo.
    """
    if reasoning is None:
        return ""
    parts = str(reasoning).split("### The final answer is:")
    if len(parts) >= 2:
        return parts[-1].strip()
    # fallback: try plain marker
    parts = str(reasoning).split("The final answer is:")
    if len(parts) >= 2:
        return parts[-1].strip()
    return reasoning.strip()


_RE_CHOICE = re.compile(r"\b([abcd])\b", re.IGNORECASE)


def extract_choice_letter(s: str) -> str | None:
    """
    If the answer looks like multiple-choice, return A/B/C/D.
    """
    if s is None:
        return None
    s = normalize_answer(s)
    # common forms: "a", "a: 12", "answer a"
    m = _RE_CHOICE.search(s)
    if not m:
        return None
    return m.group(1).upper()


def is_correct_answer(pred: str, gt: str) -> bool:
    """
    Conservative correctness:
    - if both parse as a choice letter => compare letter
    - else compare normalized strings (exact match)
    """
    p_choice = extract_choice_letter(pred)
    g_choice = extract_choice_letter(gt)
    if p_choice and g_choice:
        return p_choice == g_choice
    return normalize_answer(pred) == normalize_answer(gt)

