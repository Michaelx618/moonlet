"""Shared runtime boot helpers for CLI/server/UI adapters."""

from __future__ import annotations

import os
from typing import Dict, Optional


def build_runtime_env(overrides: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Return a normalized runtime env map for Moonlet processes.

    This keeps adapter launch behavior consistent across CLI and Electron.
    """
    env = dict(os.environ)
    env.setdefault("SC2_MODEL_PROFILE", "auto")
    env.setdefault("SC2_USE_CHATML_WRAP", "1")
    env.setdefault("SC2_USE_CHAT_TOOLS", "0")
    env.setdefault("SC2_APPROVAL_MODE", "1")
    env.setdefault("SC2_AUTO_APPLY_ON_SUCCESS", "1")  # apply edits to disk when pipeline succeeds (stabilize)
    if overrides:
        for key, value in overrides.items():
            if value is None:
                continue
            env[str(key)] = str(value)
    return env
