from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Iterable

from ..core.types import MultiModalSample


def _read_json_or_jsonl(path: str) -> list[dict[str, Any]]:
    if path.endswith(".jsonl"):
        rows: list[dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict) and "data" in obj and isinstance(obj["data"], list):
        return obj["data"]
    raise ValueError(f"Unsupported json format: {path}")


def _pick(d: dict[str, Any], keys: Iterable[str]) -> Any | None:
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return None


def load_mathvista(data_path: str, image_root: str | None = None) -> list[MultiModalSample]:
    """
    Best-effort MathVista loader. Accepts common field variants.
    Required:
    - image: image/image_path/img/img_path
    - question: question/query/prompt
    - gt: answer/gt/ground_truth/final_answer/label
    """
    raw = _read_json_or_jsonl(data_path)
    out: list[MultiModalSample] = []
    for idx, d in enumerate(raw):
        sid = str(_pick(d, ["id", "qid", "question_id"]) or idx)
        image = _pick(d, ["image", "image_path", "img", "img_path"])
        question = _pick(d, ["question", "query", "prompt"])
        gt = _pick(d, ["answer", "gt", "ground_truth", "final_answer", "label"])
        if image is None or question is None or gt is None:
            raise ValueError(f"Bad sample idx={idx}, id={sid}, keys={list(d.keys())}")

        image = str(image)
        if image_root is not None and not os.path.isabs(image) and not image.startswith(("http://", "https://")):
            image = os.path.join(image_root, image)

        out.append(
            MultiModalSample(
                sample_id=sid,
                image_path=image,
                question=str(question),
                gt_answer=str(gt),
                meta={k: v for k, v in d.items() if k not in {"id", "qid", "question_id", "image", "image_path", "img", "img_path", "question", "query", "prompt", "answer", "gt", "ground_truth", "final_answer", "label"}},
            )
        )
    return out

