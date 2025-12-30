from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class MathVistaExample:
    id: str
    image: str
    question: str
    answer: str
    # Optional: choices / metadata
    meta: dict[str, Any]


def _read_json_or_jsonl(path: str) -> list[dict[str, Any]]:
    if path.endswith(".jsonl"):
        out: list[dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                out.append(json.loads(line))
        return out
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict) and "data" in obj and isinstance(obj["data"], list):
        return obj["data"]
    raise ValueError(f"Unsupported MathVista json format: {path}")


def _pick(d: dict[str, Any], keys: Iterable[str]) -> Any | None:
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return None


def load_mathvista(
    data_path: str,
    image_root: str | None = None,
) -> list[MathVistaExample]:
    """
    Load MathVista-like samples from a json/jsonl file.

    Supported key variants (best-effort):
    - image: "image" | "image_path" | "img" | "img_path"
    - question: "question" | "query" | "prompt"
    - answer: "answer" | "gt" | "ground_truth" | "final_answer" | "label"
    - id: "id" | "qid" | "question_id"
    """
    raw = _read_json_or_jsonl(data_path)
    out: list[MathVistaExample] = []
    for idx, d in enumerate(raw):
        qid = _pick(d, ["id", "qid", "question_id"]) or str(idx)
        image = _pick(d, ["image", "image_path", "img", "img_path"])
        question = _pick(d, ["question", "query", "prompt"])
        answer = _pick(d, ["answer", "gt", "ground_truth", "final_answer", "label"])

        if image is None or question is None or answer is None:
            raise ValueError(
                f"Missing required fields at idx={idx}, id={qid}. "
                f"Need image/question/answer, got keys={list(d.keys())}"
            )

        if image_root is not None and not (image.startswith("http://") or image.startswith("https://") or os.path.isabs(image)):
            image = os.path.join(image_root, image)

        out.append(
            MathVistaExample(
                id=str(qid),
                image=str(image),
                question=str(question),
                answer=str(answer),
                meta={k: v for k, v in d.items() if k not in {"id", "qid", "question_id", "image", "image_path", "img", "img_path", "question", "query", "prompt", "answer", "gt", "ground_truth", "final_answer", "label"}},
            )
        )
    return out

