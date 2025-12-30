from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class VisualRewardConfig:
    """
    Action-level visual relevance reward.

    layer_index: choose hidden_states[layer_index] (negative allowed).
    tau: threshold tau in your write-up.
    """

    layer_index: int = -2
    tau: float = 0.2
    min_tokens: int = 3
    agg: Literal["mean_max", "mean_max_clipped"] = "mean_max"


_RE_WORDLIKE = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)


def _is_wordlike(tok: str) -> bool:
    return _RE_WORDLIKE.search(tok) is not None


def _safe_find_sublist(haystack: list[int], needle: list[int]) -> int | None:
    if not needle or len(needle) > len(haystack):
        return None
    for i in range(0, len(haystack) - len(needle) + 1):
        if haystack[i : i + len(needle)] == needle:
            return i
    return None


def infer_vision_token_indices(input_ids: list[int], tokenizer, image_grid_thw=None) -> list[int]:
    # Prefer explicit vision delimiters.
    vs = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    ve = tokenizer.convert_tokens_to_ids("<|vision_end|>")
    if isinstance(vs, int) and isinstance(ve, int):
        try:
            s = input_ids.index(vs)
            e = input_ids.index(ve, s + 1)
            if e > s + 1:
                return list(range(s + 1, e))
        except ValueError:
            pass

    # Common case for Qwen2-VL: repeated <|image_pad|> tokens.
    ip = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    if isinstance(ip, int):
        idxs = [i for i, t in enumerate(input_ids) if t == ip]
        if idxs:
            return idxs

    # Fallback: approximate by t*h*w from image_grid_thw if available.
    if image_grid_thw:
        try:
            t, h, w = image_grid_thw[0]
            n = int(t) * int(h) * int(w)
            if 0 < n < len(input_ids):
                return list(range(0, n))
        except Exception:
            pass
    return []


def infer_appended_token_indices(pre_ids: list[int], full_ids: list[int]) -> list[int]:
    if len(pre_ids) < len(full_ids) and full_ids[: len(pre_ids)] == pre_ids:
        return list(range(len(pre_ids), len(full_ids)))
    start = _safe_find_sublist(full_ids, pre_ids)
    if start is None:
        guess = max(0, len(full_ids) - max(32, len(full_ids) // 4))
        return list(range(guess, len(full_ids)))
    return list(range(start + len(pre_ids), len(full_ids)))


def compute_visual_reward(
    *,
    qwen2vl,
    image_path: str,
    question_text: str,
    prefix_text: str,
    candidate_step_text: str,
    cfg: VisualRewardConfig,
) -> float:
    """
    Compute r_vis(a_t | s_t) using cross-modal hidden states.
    """
    import torch

    # full forward
    full_inputs, full_out = qwen2vl.forward_hidden_states(
        image_path=image_path,
        question_text=question_text,
        prefix_text=prefix_text,
        appended_text=candidate_step_text,
        output_hidden_states=True,
    )

    # prefix-only forward (to locate appended span robustly)
    pre_inputs, _ = qwen2vl.forward_hidden_states(
        image_path=image_path,
        question_text=question_text,
        prefix_text=prefix_text,
        appended_text="",
        output_hidden_states=False,
    )

    hs = full_out.hidden_states[cfg.layer_index][0]  # [seq, dim]
    full_ids = full_inputs["input_ids"][0].tolist()
    pre_ids = pre_inputs["input_ids"][0].tolist()

    image_grid_thw = None
    if "image_grid_thw" in full_inputs:
        try:
            image_grid_thw = full_inputs["image_grid_thw"][0].tolist()
        except Exception:
            image_grid_thw = None

    vis_idxs = infer_vision_token_indices(full_ids, qwen2vl.processor.tokenizer, image_grid_thw=image_grid_thw)
    if not vis_idxs:
        return 0.0

    cand_idxs = infer_appended_token_indices(pre_ids, full_ids)
    if not cand_idxs:
        return 0.0

    toks = qwen2vl.processor.tokenizer.convert_ids_to_tokens([full_ids[i] for i in cand_idxs])
    keep_local = [j for j, t in enumerate(toks) if _is_wordlike(t)]
    if len(keep_local) < cfg.min_tokens:
        return 0.0
    keep_idxs = [cand_idxs[j] for j in keep_local]

    text_emb = torch.nn.functional.normalize(hs[keep_idxs], dim=-1)
    vis_emb = torch.nn.functional.normalize(hs[vis_idxs], dim=-1)
    sim = text_emb @ vis_emb.T  # [n, m]
    max_sim = sim.max(dim=-1).values  # [n]
    score = float(max_sim.mean().item())

    if cfg.agg == "mean_max":
        return score if score >= cfg.tau else 0.0
    if cfg.agg == "mean_max_clipped":
        return float(torch.clamp(max_sim - cfg.tau, min=0).mean().item())
    raise ValueError(f"Unknown agg={cfg.agg}")

