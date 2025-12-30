from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class VisionRewardConfig:
    # Which layer to use from output.hidden_states (negative index allowed).
    layer_index: int = -2
    # Similarity threshold (tau in the write-up).
    tau: float = 0.2
    # How to aggregate token-to-patch similarities.
    agg: Literal["mean_max", "mean_max_clipped"] = "mean_max"
    # Minimum number of candidate tokens required; otherwise reward=0.
    min_tokens: int = 3


_RE_WORDLIKE = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)


def _is_wordlike(token_str: str) -> bool:
    # Heuristic: ignore pure punctuation / whitespace pieces.
    return _RE_WORDLIKE.search(token_str) is not None


def _safe_find_sublist(haystack: list[int], needle: list[int]) -> int | None:
    """Return the first index where needle occurs in haystack, else None."""
    if not needle or len(needle) > len(haystack):
        return None
    # naive scan; sequences are short enough for our use.
    for i in range(0, len(haystack) - len(needle) + 1):
        if haystack[i : i + len(needle)] == needle:
            return i
    return None


def infer_vision_token_span(
    input_ids: "list[int]",
    tokenizer,
    image_grid_thw: "list[list[int]] | None" = None,
) -> list[int]:
    """
    Infer which token positions correspond to visual patch tokens for Qwen2-VL.

    We try multiple robust heuristics because different Qwen2-VL processor/tokenizer
    versions may encode vision tokens differently.
    """
    # 1) Best case: explicit vision start/end delimiters exist.
    vision_start_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    vision_end_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")
    if isinstance(vision_start_id, int) and isinstance(vision_end_id, int):
        try:
            s = input_ids.index(vision_start_id)
            e = input_ids.index(vision_end_id, s + 1)
            if e > s + 1:
                return list(range(s + 1, e))
        except ValueError:
            pass

    # 2) Common case: repeated <|image_pad|> positions mark vision tokens.
    image_pad_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    if isinstance(image_pad_id, int):
        idxs = [i for i, t in enumerate(input_ids) if t == image_pad_id]
        if idxs:
            return idxs

    # 3) Fallback: approximate number of vision tokens from image_grid_thw.
    # image_grid_thw is typically [[t, h, w]]; tokens ~= t*h*w.
    if image_grid_thw:
        try:
            t, h, w = image_grid_thw[0]
            n_vis = int(t) * int(h) * int(w)
            if n_vis > 0 and n_vis < len(input_ids):
                # Heuristic: vision tokens are usually close to the beginning.
                return list(range(0, n_vis))
        except Exception:
            pass

    return []


def infer_appended_text_token_indices(
    pre_input_ids: "list[int]",
    full_input_ids: "list[int]",
    tokenizer,
) -> list[int]:
    """
    Infer which token positions in full correspond to the appended candidate step.

    Strategy:
    - If pre is a prefix of full, it's trivial.
    - Otherwise, locate pre as a sublist within full.
    """
    if len(pre_input_ids) < len(full_input_ids) and full_input_ids[: len(pre_input_ids)] == pre_input_ids:
        return list(range(len(pre_input_ids), len(full_input_ids)))

    start = _safe_find_sublist(full_input_ids, pre_input_ids)
    if start is None:
        # Worst-case fallback: treat the last quarter as appended.
        guess_start = max(0, len(full_input_ids) - max(32, len(full_input_ids) // 4))
        return list(range(guess_start, len(full_input_ids)))

    return list(range(start + len(pre_input_ids), len(full_input_ids)))


def compute_visual_relevance_reward(
    *,
    model,
    processor,
    image,
    question_text: str,
    prefix_text: str,
    candidate_step_text: str,
    cfg: VisionRewardConfig = VisionRewardConfig(),
    device: str | None = None,
) -> float:
    """
    Compute an action-level visual relevance reward for a candidate reasoning step.

    This implements the core idea from your write-up:
    - Run a forward pass to get a cross-modal hidden space (hidden_states[layer]).
    - For each candidate-step text token, compute its max similarity to any image patch token.
    - Average across tokens and apply threshold tau.

    Notes:
    - We do not require an external alignment model.
    - We use cosine similarity in the hidden space.
    - Token selection is heuristic (word-like tokens only).
    """
    import torch

    # Build chat template text exactly as the generation path does, then append prefix/step.
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question_text},
            ],
        },
    ]

    # Qwen2-VL processor supports apply_chat_template; keep consistent with utils.qwen2_vl_forward
    pre_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + (prefix_text or "")
    full_text = pre_text + (candidate_step_text or "")

    pre_inputs = processor(text=[pre_text], images=[image], padding=True, return_tensors="pt")
    full_inputs = processor(text=[full_text], images=[image], padding=True, return_tensors="pt")

    if device is None:
        device = str(getattr(model, "device", "cpu"))
    pre_inputs = pre_inputs.to(device)
    full_inputs = full_inputs.to(device)

    with torch.no_grad():
        out = model(
            **full_inputs,
            output_hidden_states=True,
            return_dict=True,
        )

    hs = out.hidden_states[cfg.layer_index][0]  # [seq, dim]
    input_ids = full_inputs["input_ids"][0].tolist()

    # Determine vision token indices.
    image_grid_thw = None
    if "image_grid_thw" in full_inputs:
        try:
            image_grid_thw = full_inputs["image_grid_thw"][0].tolist()
        except Exception:
            image_grid_thw = None
    vis_idxs = infer_vision_token_span(input_ids, processor.tokenizer, image_grid_thw=image_grid_thw)
    if not vis_idxs:
        return 0.0

    # Determine which tokens belong to appended candidate step.
    pre_ids = pre_inputs["input_ids"][0].tolist()
    cand_idxs = infer_appended_text_token_indices(pre_ids, input_ids, processor.tokenizer)
    if not cand_idxs:
        return 0.0

    # Filter candidate tokens to "visual-related" (word-like) tokens as a proxy for T_vis.
    token_strs = processor.tokenizer.convert_ids_to_tokens([input_ids[i] for i in cand_idxs])
    keep_local = [j for j, s in enumerate(token_strs) if _is_wordlike(s)]
    if len(keep_local) < cfg.min_tokens:
        return 0.0
    keep_idxs = [cand_idxs[j] for j in keep_local]

    text_emb = hs[keep_idxs]  # [n, dim]
    vis_emb = hs[vis_idxs]  # [m, dim]

    # Cosine similarity
    text_emb = torch.nn.functional.normalize(text_emb, dim=-1)
    vis_emb = torch.nn.functional.normalize(vis_emb, dim=-1)
    sim = text_emb @ vis_emb.T  # [n, m]
    max_sim = sim.max(dim=-1).values  # [n]
    score = float(max_sim.mean().item())

    if cfg.agg == "mean_max":
        return score if score >= cfg.tau else 0.0
    if cfg.agg == "mean_max_clipped":
        # encourages surpassing tau: mean(max(sim - tau, 0))
        return float(torch.clamp(max_sim - cfg.tau, min=0).mean().item())
    raise ValueError(f"Unknown agg: {cfg.agg}")

