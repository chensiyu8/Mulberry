from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Any

from PIL import Image
from tqdm import tqdm

from answer_utils import extract_final_answer_from_reasoning, is_correct_answer
from mathvista_io import load_mathvista
from utils import modified_qwen_response
from vision_reward import VisionRewardConfig, compute_visual_relevance_reward


HEADER_PROMPT = """You will be given an image and a question.

Only output the following three sections (exactly, separated by ###), and stop:
### Image Description:
### Rationales:
### Let's think step by step.

Rules:
- Do NOT output any "### Step" sections yet.
- Do NOT output the final answer.
"""


def _step_prompt(step_idx: int) -> str:
    return f"""Continue the reasoning from the given prefix.

Only output ONE next reasoning step, starting with exactly:
### Step {step_idx}:

Rules:
- Do NOT rewrite or modify the prefix.
- Do NOT output any other sections.
- Do NOT output the final answer.
"""


FINISH_PROMPT = """Continue the reasoning from the given prefix and finish the solution.

Rules:
- Do NOT rewrite or modify the prefix.
- Output subsequent steps (### Step k:) as needed.
- Finally output:
### The final answer is:
<answer>
"""


@dataclass
class MCTSConfig:
    max_depth: int = 8
    max_iterations: int = 24
    num_expand: int = 12
    keep_topk_by_vision: int = 5
    num_rollouts: int = 4
    ucb_c: float = 1.2
    alpha_answer: float = 1.0
    beta_vision: float = 0.6
    gamma_reflect: float = 0.4
    reward_threshold: float = 0.0


class Node:
    def __init__(
        self,
        *,
        prefix: str,
        depth: int,
        parent: "Node | None" = None,
        action_text: str | None = None,
        cum_vision: float = 0.0,
    ):
        self.prefix = prefix
        self.depth = depth
        self.parent = parent
        self.action_text = action_text or ""
        self.cum_vision = float(cum_vision)

        self.children: list[Node] = []
        self.visits: int = 0
        self.value: float = 0.0

        # For data export
        self.rollouts: list[dict[str, Any]] = []

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def ucb_score(self, c: float) -> float:
        if self.parent is None:
            return self.value
        return self.value + c * math.sqrt(math.log(self.parent.visits + 1) / (self.visits + 1))

    def best_child(self, c: float) -> "Node":
        assert self.children, "best_child called on leaf"
        return max(self.children, key=lambda n: n.ucb_score(c))

    def add_child(self, child: "Node") -> None:
        self.children.append(child)


def qwen2vl_generate(
    *,
    model,
    processor,
    image_path: str,
    system_prompt: str,
    user_text: str,
    prefix_text: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int | None = None,
) -> str:
    import torch

    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": user_text},
            ],
        },
    ]

    image = Image.open(image_path)
    # Ensure minimal size (some datasets have tiny images)
    w, h = image.size
    if min(w, h) < 28:
        factor = 28 / float(min(w, h))
        image = image.resize((int(w * factor), int(h * factor)))

    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + (prefix_text or "")
    inputs = processor(text=[prompt_text], images=[image], padding=True, return_tensors="pt").to(model.device)

    gen = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=1.0,
    )
    gen_trim = [out[len(inp) :] for inp, out in zip(inputs.input_ids, gen)]
    text = processor.batch_decode(gen_trim, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return (prefix_text or "") + text


def generate_header_prefix(model, processor, image_path: str, question: str, *, temperature: float, top_p: float) -> str:
    # We keep this short; it should only produce the 3 header sections.
    out = qwen2vl_generate(
        model=model,
        processor=processor,
        image_path=image_path,
        system_prompt="You are a helpful assistant.",
        user_text=HEADER_PROMPT + "\n\nQuestion:\n" + question,
        prefix_text="",
        max_new_tokens=512,
        temperature=temperature,
        top_p=top_p,
        seed=None,
    )
    out = modified_qwen_response(out)
    # Hard stop if it accidentally generated steps/answer.
    if "### Step" in out:
        out = out.split("### Step")[0].rstrip() + "\n\n### Let's think step by step.\n"
    if "### The final answer is" in out:
        out = out.split("### The final answer is")[0].rstrip()
    # Ensure it ends with the thinking marker (as prefix).
    if "### Let's think step by step" not in out:
        out = out.rstrip() + "\n\n### Let's think step by step.\n"
    return out if out.endswith("\n") else out + "\n"


def extract_one_step(gen_text: str, step_idx: int) -> str | None:
    """
    Extract exactly one '### Step {step_idx}:' block from model output.
    """
    marker = f"### Step {step_idx}:"
    if marker not in gen_text:
        # try loose marker
        alt = f"Step {step_idx}:"
        if alt in gen_text and gen_text.count(alt) == 1:
            gen_text = gen_text.replace(alt, marker)
        else:
            return None
    after = gen_text.split(marker, 1)[1]
    # stop at next ###
    if "###" in after:
        after = after.split("###", 1)[0]
    step = marker + "\n" + after.strip() + "\n\n"
    # Block forbidden content
    if "final answer" in step.lower():
        return None
    return step


def finish_reasoning(model, processor, image_path: str, question: str, prefix: str, *, temperature: float, top_p: float) -> str:
    out = qwen2vl_generate(
        model=model,
        processor=processor,
        image_path=image_path,
        system_prompt="You are a helpful assistant.",
        user_text=FINISH_PROMPT + "\n\nQuestion:\n" + question,
        prefix_text=prefix,
        max_new_tokens=1024,
        temperature=temperature,
        top_p=top_p,
        seed=None,
    )
    return modified_qwen_response(out)


def estimate_reflection_value(rollouts: list[dict[str, Any]]) -> float:
    """
    Proxy for 'reflection value' (R_reflect) without a second LLM judge:
    - if we see both incorrect and correct rollouts, reflection opportunity exists -> higher.
    - otherwise lower.
    """
    has_correct = any(r.get("is_correct") for r in rollouts)
    has_incorrect = any(not r.get("is_correct") for r in rollouts)
    if has_correct and has_incorrect:
        return 1.0
    if has_correct:
        return 0.2
    return 0.0


def mcts_search_one(
    *,
    model,
    processor,
    image_path: str,
    question: str,
    gt_answer: str,
    cfg: MCTSConfig,
    vision_cfg: VisionRewardConfig,
    temperature: float,
    top_p: float,
    reflection_phrase: str,
) -> dict[str, Any]:
    root_prefix = generate_header_prefix(model, processor, image_path, question, temperature=temperature, top_p=top_p)
    root = Node(prefix=root_prefix, depth=0, parent=None, action_text="", cum_vision=0.0)

    def select(node: Node) -> Node:
        cur = node
        while not cur.is_leaf():
            cur = cur.best_child(cfg.ucb_c)
        return cur

    def expand(node: Node) -> None:
        if node.depth >= cfg.max_depth:
            return
        step_idx = node.depth + 1

        candidates: list[dict[str, Any]] = []
        for i in range(cfg.num_expand):
            gen = qwen2vl_generate(
                model=model,
                processor=processor,
                image_path=image_path,
                system_prompt="You are a helpful assistant.",
                user_text=_step_prompt(step_idx) + "\n\nQuestion:\n" + question,
                prefix_text=node.prefix,
                max_new_tokens=256,
                temperature=temperature,
                top_p=top_p,
                seed=random.randint(1, 10**9),
            )
            gen = modified_qwen_response(gen)
            step = extract_one_step(gen, step_idx)
            if step is None:
                continue

            try:
                v = compute_visual_relevance_reward(
                    model=model,
                    processor=processor,
                    image=image_path,
                    question_text=question,
                    prefix_text=node.prefix,
                    candidate_step_text=step,
                    cfg=vision_cfg,
                    device=str(model.device),
                )
            except Exception:
                v = 0.0
            candidates.append({"step": step, "vision_reward": float(v)})

        candidates.sort(key=lambda x: x["vision_reward"], reverse=True)
        candidates = candidates[: cfg.keep_topk_by_vision]

        for cnd in candidates:
            child = Node(
                prefix=node.prefix + cnd["step"],
                depth=node.depth + 1,
                parent=node,
                action_text=cnd["step"],
                cum_vision=node.cum_vision + cnd["vision_reward"],
            )
            node.add_child(child)

    def simulate(node: Node) -> tuple[float, dict[str, Any]]:
        # Rollout to a full reasoning and compute correctness.
        reasoning = finish_reasoning(model, processor, image_path, question, node.prefix, temperature=temperature, top_p=top_p)
        pred = extract_final_answer_from_reasoning(reasoning)
        ok = is_correct_answer(pred, gt_answer)
        return (1.0 if ok else 0.0), {
            "reasoning": reasoning,
            "pred_answer": pred,
            "is_correct": ok,
        }

    def backprop(node: Node, reward: float) -> None:
        cur = node
        while cur is not None:
            cur.visits += 1
            # running average
            cur.value += (reward - cur.value) / float(cur.visits)
            cur = cur.parent

    for _ in range(cfg.max_iterations):
        leaf = select(root)
        if leaf.visits == 0:
            # do at least one simulation on newly reached leaf
            pass
        expand(leaf)

        # Simulate from either the leaf (if no children) or its new children
        sim_targets = leaf.children if leaf.children else [leaf]

        for n in sim_targets:
            n.rollouts = []
            for _r in range(cfg.num_rollouts):
                ans_reward, rollout = simulate(n)
                n.rollouts.append(rollout)

            p_correct = sum(1 for r in n.rollouts if r["is_correct"]) / max(1, len(n.rollouts))
            r_answer = float(p_correct)
            r_vision = float(n.cum_vision / max(1, n.depth))
            r_reflect = estimate_reflection_value(n.rollouts)

            total = cfg.alpha_answer * r_answer + cfg.beta_vision * r_vision + cfg.gamma_reflect * r_reflect

            if total >= cfg.reward_threshold:
                backprop(n, total)
            else:
                backprop(n, 0.0)

    # Pick best child path by value.
    cur = root
    path: list[Node] = []
    while cur.children:
        cur = max(cur.children, key=lambda x: x.value)
        path.append(cur)

    final_prefix = root.prefix + "".join(n.action_text for n in path)
    final_reasoning = finish_reasoning(model, processor, image_path, question, final_prefix, temperature=temperature, top_p=top_p)
    pred = extract_final_answer_from_reasoning(final_reasoning)
    ok = is_correct_answer(pred, gt_answer)

    # Build one reflection sample if possible: pick one incorrect + one correct rollout from best node
    reflect = None
    if path:
        best_leaf = path[-1]
        correct_roll = next((r for r in best_leaf.rollouts if r.get("is_correct")), None)
        incorrect_roll = next((r for r in best_leaf.rollouts if not r.get("is_correct")), None)
        if correct_roll and incorrect_roll:
            step_idx = best_leaf.depth + 1
            incorrect_next = extract_one_step(incorrect_roll["reasoning"], step_idx) or ""
            correct_next = extract_one_step(correct_roll["reasoning"], step_idx) or ""
            reflect_text = (
                final_prefix
                + incorrect_next
                + (reflection_phrase.rstrip() + "\n\n")
                + (correct_next if correct_next else "")
            )
            reflect = {
                "incorrect_reasoning": incorrect_roll["reasoning"],
                "correct_reasoning": correct_roll["reasoning"],
                "incorrect_step": incorrect_next,
                "correct_step": correct_next,
                "reflection_reasoning": reflect_text,
            }

    return {
        "prefix": root.prefix,
        "path_steps": [n.action_text for n in path],
        "final_reasoning": final_reasoning,
        "pred_answer": pred,
        "is_correct": ok,
        "reflection": reflect,
        "tree": {
            "root_value": root.value,
            "root_visits": root.visits,
        },
    }


def to_sharegpt_record(image_path: str, question: str, assistant: str) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "user", "content": "<image>\n" + question},
            {"role": "assistant", "content": assistant},
        ],
        "images": image_path,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="MathVista json/jsonl path")
    parser.add_argument("--image_root", type=str, default=None, help="Optional root for relative image paths")
    parser.add_argument("--output_search_jsonl", type=str, required=True)
    parser.add_argument("--output_train_json", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--max_depth", type=int, default=8)
    parser.add_argument("--max_iterations", type=int, default=24)
    parser.add_argument("--num_expand", type=int, default=12)
    parser.add_argument("--keep_topk_by_vision", type=int, default=5)
    parser.add_argument("--num_rollouts", type=int, default=4)
    parser.add_argument("--ucb_c", type=float, default=1.2)
    parser.add_argument("--alpha_answer", type=float, default=1.0)
    parser.add_argument("--beta_vision", type=float, default=0.6)
    parser.add_argument("--gamma_reflect", type=float, default=0.4)
    parser.add_argument("--reward_threshold", type=float, default=0.0)
    parser.add_argument("--vision_layer", type=int, default=-2)
    parser.add_argument("--vision_tau", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument(
        "--reflection_phrase",
        type=str,
        default="I think I have made a mistake. Let me rethink it.",
        help="Inserted between incorrect step and corrected continuation.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # Lazy import heavy deps
    import torch
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if not torch.cuda.is_available():
        model = model.to("cpu")
    model.eval()

    processor = AutoProcessor.from_pretrained(args.model_path)

    exs = load_mathvista(args.data_path, image_root=args.image_root)

    os.makedirs(os.path.dirname(args.output_search_jsonl), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_train_json), exist_ok=True)

    cfg = MCTSConfig(
        max_depth=args.max_depth,
        max_iterations=args.max_iterations,
        num_expand=args.num_expand,
        keep_topk_by_vision=args.keep_topk_by_vision,
        num_rollouts=args.num_rollouts,
        ucb_c=args.ucb_c,
        alpha_answer=args.alpha_answer,
        beta_vision=args.beta_vision,
        gamma_reflect=args.gamma_reflect,
        reward_threshold=args.reward_threshold,
    )
    vcfg = VisionRewardConfig(layer_index=args.vision_layer, tau=args.vision_tau)

    train: list[dict[str, Any]] = []

    with open(args.output_search_jsonl, "w", encoding="utf-8") as f:
        for ex in tqdm(exs):
            if not os.path.exists(ex.image):
                # skip missing images but keep pipeline running
                continue

            result = mcts_search_one(
                model=model,
                processor=processor,
                image_path=ex.image,
                question=ex.question if ex.question.startswith("Question:") else "Question: " + ex.question,
                gt_answer=ex.answer,
                cfg=cfg,
                vision_cfg=vcfg,
                temperature=args.temperature,
                top_p=args.top_p,
                reflection_phrase=args.reflection_phrase,
            )

            record = {
                "id": ex.id,
                "image": ex.image,
                "question": ex.question,
                "gt_answer": ex.answer,
                "meta": ex.meta,
                "mcts": result,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

            # Prefer reflection if available; otherwise use the best final reasoning
            assistant_text = (
                result["reflection"]["reflection_reasoning"]
                if result.get("reflection") and result["reflection"] is not None
                else result["final_reasoning"]
            )
            train.append(to_sharegpt_record(ex.image, ex.question, assistant_text))

    with open(args.output_train_json, "w", encoding="utf-8") as f:
        json.dump(train, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

