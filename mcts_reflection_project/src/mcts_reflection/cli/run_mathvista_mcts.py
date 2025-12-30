from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any

from tqdm import tqdm

from ..core.answer import extract_final_answer, is_correct
from ..core.mcts import MCTS, MCTSConfig, Node
from ..core.types import ReasoningAction, ReasoningState
from ..datasets.mathvista import load_mathvista
from ..export.formats import to_sharegpt
from ..models.qwen2vl import Qwen2VL, Qwen2VLGenerateConfig
from ..rewards.visual_relevance import VisualRewardConfig, compute_visual_reward
from ..reflection.generator import ReflectionConfig, build_template_reflection, generate_reflection_with_qwen2vl


HEADER_USER_PROMPT = """请根据问题生成图像描述，并给出解题思路，但不要开始逐步推理。

你必须且只输出以下三个部分（用 ### 分隔），然后停止：
### Image Description:
### Rationales:
### Let's think step by step.

约束：
- 不要输出任何 ### Step
- 不要输出最终答案
"""


def step_user_prompt(step_idx: int) -> str:
    return f"""请基于给定 prefix 继续推理。

你必须且只输出一个“下一步推理动作”，格式必须以这一行开头：
### Step {step_idx}:

约束：
- 不要改写 prefix
- 不要输出其他 section
- 不要输出最终答案
"""


FINISH_USER_PROMPT = """请基于给定 prefix 继续推理并完成解答。

约束：
- 不要改写 prefix
- 可以输出多个后续步骤（### Step k:）
- 最后必须输出：
### The final answer is:
<answer>
"""


def extract_one_step(text: str, step_idx: int) -> str | None:
    marker = f"### Step {step_idx}:"
    if marker not in text:
        # tolerate "Step k:" once
        alt = f"Step {step_idx}:"
        if alt in text and text.count(alt) == 1:
            text = text.replace(alt, marker)
        else:
            return None
    after = text.split(marker, 1)[1]
    if "###" in after:
        after = after.split("###", 1)[0]
    step = marker + "\n" + after.strip() + "\n\n"
    if "final answer" in step.lower():
        return None
    return step


def estimate_reflection_value(rollouts: list[dict[str, Any]]) -> float:
    has_correct = any(r.get("is_correct") for r in rollouts)
    has_incorrect = any(not r.get("is_correct") for r in rollouts)
    if has_correct and has_incorrect:
        return 1.0
    if has_correct:
        return 0.2
    return 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--image_root", type=str, default=None)
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--output_search_jsonl", type=str, required=True)
    parser.add_argument("--output_train_json", type=str, required=True)

    # MCTS
    parser.add_argument("--max_iterations", type=int, default=24)
    parser.add_argument("--max_depth", type=int, default=8)
    parser.add_argument("--ucb_c", type=float, default=1.2)
    parser.add_argument("--num_expand", type=int, default=12)
    parser.add_argument("--keep_topk_by_vision", type=int, default=5)
    parser.add_argument("--num_rollouts", type=int, default=4)
    parser.add_argument("--alpha_answer", type=float, default=1.0)
    parser.add_argument("--beta_vision", type=float, default=0.6)
    parser.add_argument("--gamma_reflect", type=float, default=0.4)
    parser.add_argument("--reward_threshold", type=float, default=0.0)

    # Visual reward
    parser.add_argument("--vision_layer", type=int, default=-2)
    parser.add_argument("--vision_tau", type=float, default=0.2)

    # Generation
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.95)

    # Reflection
    parser.add_argument("--reflection_mode", type=str, default="template", choices=["none", "template", "qwen2vl"])
    parser.add_argument("--reflection_phrase", type=str, default="我发现刚才可能推错了，让我重新思考。")

    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    qwen = Qwen2VL(args.model_path)
    gen_cfg = Qwen2VLGenerateConfig(max_new_tokens=512, temperature=args.temperature, top_p=args.top_p)
    vis_cfg = VisualRewardConfig(layer_index=args.vision_layer, tau=args.vision_tau)
    ref_cfg = ReflectionConfig(mode=args.reflection_mode, reflection_phrase=args.reflection_phrase)

    mcts_cfg = MCTSConfig(
        max_iterations=args.max_iterations,
        max_depth=args.max_depth,
        ucb_c=args.ucb_c,
        num_expand=args.num_expand,
        keep_topk=args.keep_topk_by_vision,
        num_rollouts=args.num_rollouts,
        alpha_answer=args.alpha_answer,
        beta_vision=args.beta_vision,
        gamma_reflect=args.gamma_reflect,
        reward_threshold=args.reward_threshold,
    )

    samples = load_mathvista(args.data_path, image_root=args.image_root)

    os.makedirs(os.path.dirname(args.output_search_jsonl), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_train_json), exist_ok=True)

    train_rows: list[dict[str, Any]] = []

    def expand(state: ReasoningState, n: int) -> list[ReasoningAction]:
        actions: list[ReasoningAction] = []
        for _ in range(n):
            gen = qwen.generate(
                image_path=cur_image,
                question_text=cur_question,
                user_prompt=step_user_prompt(state.step_index),
                prefix_text=state.prefix_text,
                gen_cfg=Qwen2VLGenerateConfig(max_new_tokens=256, temperature=args.temperature, top_p=args.top_p),
                seed=random.randint(1, 10**9),
            )
            step = extract_one_step(gen, state.step_index)
            if step is None:
                continue
            try:
                v = compute_visual_reward(
                    qwen2vl=qwen,
                    image_path=cur_image,
                    question_text=cur_question,
                    prefix_text=state.prefix_text,
                    candidate_step_text=step,
                    cfg=vis_cfg,
                )
            except Exception:
                v = 0.0
            actions.append(ReasoningAction(step_text=step, vision_reward=float(v)))
        return actions

    def transition(state: ReasoningState, action: ReasoningAction) -> ReasoningState:
        return ReasoningState(prefix_text=state.prefix_text + action.step_text, step_index=state.step_index + 1)

    def simulate(state: ReasoningState, n: int) -> list[dict[str, Any]]:
        outs: list[dict[str, Any]] = []
        for _ in range(n):
            reasoning = qwen.generate(
                image_path=cur_image,
                question_text=cur_question,
                user_prompt=FINISH_USER_PROMPT,
                prefix_text=state.prefix_text,
                gen_cfg=Qwen2VLGenerateConfig(max_new_tokens=1024, temperature=args.temperature, top_p=args.top_p),
                seed=random.randint(1, 10**9),
            )
            pred = extract_final_answer(reasoning)
            ok = is_correct(pred, cur_gt)
            outs.append({"reasoning": reasoning, "pred_answer": pred, "is_correct": ok})
        return outs

    def compute_reward(node: Node, rollouts: list[dict[str, Any]]) -> float:
        p_correct = sum(1 for r in rollouts if r["is_correct"]) / max(1, len(rollouts))
        r_ans = float(p_correct)
        r_vis = float(node.cum_vision_reward / max(1, node.depth))
        r_ref = estimate_reflection_value(rollouts)
        return (
            mcts_cfg.alpha_answer * r_ans
            + mcts_cfg.beta_vision * r_vis
            + mcts_cfg.gamma_reflect * r_ref
        )

    mcts = MCTS(mcts_cfg, expand=expand, transition=transition, simulate=simulate, compute_reward=compute_reward)

    with open(args.output_search_jsonl, "w", encoding="utf-8") as fout:
        for s in tqdm(samples):
            if not os.path.exists(s.image_path):
                continue

            cur_image = s.image_path
            cur_question = s.question if s.question.startswith("Question:") else "Question: " + s.question
            cur_gt = s.gt_answer

            # Root prefix (Image Description + Rationales + Let's think ...)
            header = qwen.generate(
                image_path=cur_image,
                question_text=cur_question,
                user_prompt=HEADER_USER_PROMPT,
                prefix_text="",
                gen_cfg=Qwen2VLGenerateConfig(max_new_tokens=512, temperature=args.temperature, top_p=args.top_p),
                seed=random.randint(1, 10**9),
            )
            # Ensure it doesn't accidentally include steps/answer
            if "### Step" in header:
                header = header.split("### Step")[0].rstrip() + "\n\n### Let's think step by step.\n"
            if "### The final answer is" in header:
                header = header.split("### The final answer is")[0].rstrip()
            if "### Let's think step by step" not in header:
                header = header.rstrip() + "\n\n### Let's think step by step.\n"

            root_state = ReasoningState(prefix_text=header if header.endswith("\n") else header + "\n", step_index=1)
            root = mcts.search(root_state)

            best_path = MCTS.pick_best_path(root)
            best_prefix = root_state.prefix_text + "".join(n.action_from_parent.step_text for n in best_path if n.action_from_parent)
            best_reasoning = qwen.generate(
                image_path=cur_image,
                question_text=cur_question,
                user_prompt=FINISH_USER_PROMPT,
                prefix_text=best_prefix,
                gen_cfg=Qwen2VLGenerateConfig(max_new_tokens=1024, temperature=args.temperature, top_p=args.top_p),
                seed=random.randint(1, 10**9),
            )
            best_pred = extract_final_answer(best_reasoning)
            best_ok = is_correct(best_pred, cur_gt)

            # Build one reflection sample if possible: choose one incorrect rollout + one correct rollout from best leaf
            reflection = None
            if best_path:
                leaf = best_path[-1]
                wrong = next((r for r in leaf.rollouts if not r.get("is_correct")), None)
                right = next((r for r in leaf.rollouts if r.get("is_correct")), None)
                if wrong and right and ref_cfg.mode != "none":
                    if ref_cfg.mode == "qwen2vl":
                        reflection = generate_reflection_with_qwen2vl(
                            qwen2vl=qwen,
                            image_path=cur_image,
                            question_text=cur_question,
                            incorrect_reasoning=wrong["reasoning"],
                            correct_reasoning=right["reasoning"],
                            prefix=best_prefix,
                            cfg=ref_cfg,
                        )
                    elif ref_cfg.mode == "template":
                        # Minimal reflection: stitch wrong-next-step + phrase + correct continuation (full correct reasoning)
                        wrong_step = extract_one_step(wrong["reasoning"], leaf.state.step_index) or ""
                        reflection_text = build_template_reflection(
                            prefix=best_prefix,
                            incorrect_step=wrong_step,
                            correct_continuation=right["reasoning"],
                            reflection_phrase=ref_cfg.reflection_phrase,
                        )
                        reflection = {"reflection_text": reflection_text, "incorrect_step": wrong_step}

            row = {
                "id": s.sample_id,
                "image": s.image_path,
                "question": s.question,
                "gt_answer": s.gt_answer,
                "meta": s.meta,
                "mcts": {
                    "root": {"value": root.value, "visits": root.visits},
                    "best": {
                        "final_reasoning": best_reasoning,
                        "pred_answer": best_pred,
                        "is_correct": best_ok,
                        "path_steps": [n.action_from_parent.step_text for n in best_path if n.action_from_parent],
                    },
                    "reflection": reflection,
                },
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

            assistant_text = (
                reflection["reflection_text"]
                if reflection and "reflection_text" in reflection
                else best_reasoning
            )
            train_rows.append(to_sharegpt(s.image_path, s.question, assistant_text))

    with open(args.output_train_json, "w", encoding="utf-8") as f:
        json.dump(train_rows, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

