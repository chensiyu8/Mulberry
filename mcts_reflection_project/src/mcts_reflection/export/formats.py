from __future__ import annotations

from typing import Any


def to_sharegpt(image_path: str, question: str, assistant: str) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "user", "content": "<image>\n" + question},
            {"role": "assistant", "content": assistant},
        ],
        "images": image_path,
    }

