"""Role-based generation defaults."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from . import config


@dataclass(frozen=True)
class RoleGenConfig:
    name: str
    max_new: int
    temperature: float
    stop_sequences: List[str]


def resolve_role_config(role: str) -> RoleGenConfig:
    key = (role or "chat").strip().lower()
    table: Dict[str, RoleGenConfig] = {
        "chat": RoleGenConfig("chat", config.CHAT_MAX_NEW, config.TEMPERATURE, ["\nUser:"]),
        "edit": RoleGenConfig("edit", config.PATCH_MAX_NEW, config.PATCH_TEMP, ["\nUser:", "\n\n\n"]),
        "apply": RoleGenConfig("apply", config.PATCH_MAX_NEW, config.PATCH_TEMP, ["\nUser:", "\n\n\n"]),
        "summarize": RoleGenConfig("summarize", config.CHAT_SHORT_MAX_NEW, config.TEMPERATURE, ["\nUser:"]),
        "rerank": RoleGenConfig("rerank", 256, 0.0, ["\nUser:"]),
    }
    return table.get(key, table["chat"])

