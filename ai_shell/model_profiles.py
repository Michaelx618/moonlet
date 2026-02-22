"""Declarative model capability profiles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class ModelProfile:
    name: str
    use_chat_tools: bool
    use_chatml_wrap: bool
    description: str


PROFILES: Dict[str, ModelProfile] = {
    "legacy_raw": ModelProfile(
        name="legacy_raw",
        use_chat_tools=False,
        use_chatml_wrap=False,
        description="Raw completion format (no chat wrapping).",
    ),
    "chatml_completion": ModelProfile(
        name="chatml_completion",
        use_chat_tools=False,
        use_chatml_wrap=True,
        description="Chat-tuned models via /completion + chatml wrapping.",
    ),
    "chat_tools": ModelProfile(
        name="chat_tools",
        use_chat_tools=True,
        use_chatml_wrap=False,
        description="OpenAI-style tool calls via /v1/chat/completions.",
    ),
}


def resolve_model_profile(profile_name: str, gguf_path: str = "") -> ModelProfile:
    key = (profile_name or "").strip().lower()
    if key in PROFILES:
        return PROFILES[key]

    gguf = (gguf_path or "").lower()
    # Heuristic defaults for auto mode.
    if any(x in gguf for x in ("deepseek", "coder", "instruct", "llama-3")):
        return PROFILES["chatml_completion"]
    return PROFILES["legacy_raw"]
