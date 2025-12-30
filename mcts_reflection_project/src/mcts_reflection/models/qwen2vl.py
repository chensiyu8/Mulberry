from __future__ import annotations

import random
from dataclasses import dataclass

from PIL import Image


@dataclass(frozen=True)
class Qwen2VLGenerateConfig:
    max_new_tokens: int = 512
    temperature: float = 0.9
    top_p: float = 0.95
    repetition_penalty: float = 1.0
    do_sample: bool = True


class Qwen2VL:
    """
    Thin wrapper around Qwen2-VL HF model + processor.

    Responsibilities:
    - build chat-template text exactly once (system+image+question)
    - append reasoning prefix / next-step prompts
    - provide generate() for sampling candidates
    - provide forward_hidden_states() for computing cross-modal representations
    """

    def __init__(self, model_path: str, *, torch_dtype="auto", device_map="auto"):
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        self.model_path = model_path
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
        self.model.eval()

    @property
    def device(self) -> str:
        return str(getattr(self.model, "device", "cpu"))

    def _load_image(self, image_path: str) -> Image.Image:
        img = Image.open(image_path)
        w, h = img.size
        if min(w, h) < 28:
            factor = 28 / float(min(w, h))
            img = img.resize((int(w * factor), int(h * factor)))
        return img

    def build_chat_prefix(self, *, image_path: str, question_text: str, system_prompt: str = "You are a helpful assistant.") -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": question_text},
                ],
            },
        ]
        return self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def generate(
        self,
        *,
        image_path: str,
        question_text: str,
        user_prompt: str,
        prefix_text: str,
        gen_cfg: Qwen2VLGenerateConfig,
        seed: int | None = None,
    ) -> str:
        import torch

        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        chat_prefix = self.build_chat_prefix(image_path=image_path, question_text=user_prompt + "\n\n" + question_text)
        full_text = chat_prefix + (prefix_text or "")

        img = self._load_image(image_path)
        inputs = self.processor(text=[full_text], images=[img], padding=True, return_tensors="pt").to(self.model.device)

        gen_ids = self.model.generate(
            **inputs,
            max_new_tokens=gen_cfg.max_new_tokens,
            do_sample=gen_cfg.do_sample,
            temperature=gen_cfg.temperature,
            top_p=gen_cfg.top_p,
            repetition_penalty=gen_cfg.repetition_penalty,
        )
        gen_trim = [out[len(inp) :] for inp, out in zip(inputs.input_ids, gen_ids)]
        out = self.processor.batch_decode(gen_trim, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return (prefix_text or "") + out

    def forward_hidden_states(
        self,
        *,
        image_path: str,
        question_text: str,
        prefix_text: str,
        appended_text: str,
        output_hidden_states: bool = True,
    ):
        """
        Forward pass on the concatenated text: chat_prefix(question) + prefix_text + appended_text.
        Returns (inputs, model_output) where model_output.hidden_states is available.
        """
        img = self._load_image(image_path)
        chat_prefix = self.build_chat_prefix(image_path=image_path, question_text=question_text)
        full_text = chat_prefix + (prefix_text or "") + (appended_text or "")
        inputs = self.processor(text=[full_text], images=[img], padding=True, return_tensors="pt").to(self.model.device)

        import torch

        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=output_hidden_states, return_dict=True)
        return inputs, out

