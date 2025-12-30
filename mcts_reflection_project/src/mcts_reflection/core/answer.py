from __future__ import annotations

import re


_RE_WS = re.compile(r"\s+")
_RE_PUNCT = re.compile(r"[^\w\u4e00-\u9fff\.\-\/%]+", re.UNICODE)
_RE_CHOICE = re.compile(r"\b([abcd])\b", re.IGNORECASE)


def extract_final_answer(reasoning: str) -> str:
    if reasoning is None:
        return ""
    text = str(reasoning)
    parts = text.split("### The final answer is:")
    if len(parts) >= 2:
        return parts[-1].strip()
    parts = text.split("The final answer is:")
    if len(parts) >= 2:
        return parts[-1].strip()
    parts = text.split("Answer:")
    if len(parts) >= 2:
        return parts[-1].strip()
    return text.strip()


def normalize_answer(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    for p in ("### The final answer is:", "The final answer is:", "Final answer:", "Answer:"):
        s = s.replace(p, "")
    s = s.strip().lower()
    s = _RE_PUNCT.sub(" ", s)
    s = _RE_WS.sub(" ", s).strip()
    return s


def extract_choice_letter(s: str) -> str | None:
    if s is None:
        return None
    s = normalize_answer(s)
    m = _RE_CHOICE.search(s)
    if not m:
        return None
    return m.group(1).upper()


def is_correct(pred: str, gt: str) -> bool:
    pc = extract_choice_letter(pred)
    gc = extract_choice_letter(gt)
    if pc and gc:
        return pc == gc
    return normalize_answer(pred) == normalize_answer(gt)

