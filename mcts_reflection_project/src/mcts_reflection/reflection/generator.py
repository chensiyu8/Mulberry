from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ReflectionConfig:
    """
    Reflection generation config.

    mode:
    - "none": no reflection (still exports best reasoning)
    - "template": simple deterministic reflection (no extra model calls)
    - "qwen2vl": use Qwen2-VL to write reflection (extra inference)
    """

    mode: str = "template"
    reflection_phrase: str = "我发现刚才可能推错了，让我重新思考。"


def build_template_reflection(
    *,
    prefix: str,
    incorrect_step: str,
    correct_continuation: str,
    reflection_phrase: str,
) -> str:
    return prefix + incorrect_step + (reflection_phrase.rstrip() + "\n\n") + correct_continuation


def generate_reflection_with_qwen2vl(
    *,
    qwen2vl,
    image_path: str,
    question_text: str,
    incorrect_reasoning: str,
    correct_reasoning: str,
    prefix: str,
    cfg: ReflectionConfig,
) -> dict[str, Any] | None:
    """
    Use Qwen2-VL to produce a reflection trajectory. This follows your write-up:
    provide one wrong path + one correct path, ask the model to:
    - point out wrong step(s)
    - explain how to jump from wrong branch to correct branch
    - output a corrected reasoning continuation
    """
    if cfg.mode != "qwen2vl":
        return None

    prompt = """你将看到同一题目的两条推理：一条错误推理路径与一条正确推理路径。

请生成“过程反思型推理路径”，要求：
1) 明确指出错误推理中最关键的错误步骤（引用原文或概括）。
2) 给出反思（为什么错、缺了什么视觉/逻辑依据）。
3) 从错误步骤的前一步出发，写出正确的后续推理，并完成到最终答案。
4) 输出格式要求：必须保留并沿用给定 prefix，不要改写 prefix；后续使用：
### Step k:
...
### The final answer is:
...

【错误推理】：
{wrong}

【正确推理】：
{right}
"""

    # We pass the reflection prompt as "user_prompt", and let prefix carry the already-built
    # image description + rationales + existing steps.
    out = qwen2vl.generate(
        image_path=image_path,
        question_text=question_text,
        user_prompt=prompt.format(wrong=incorrect_reasoning, right=correct_reasoning),
        prefix_text=prefix,
        gen_cfg=type("Cfg", (), dict(max_new_tokens=1024, temperature=0.7, top_p=0.9, repetition_penalty=1.0, do_sample=True))(),
        seed=None,
    )
    return {"reflection_text": out}

