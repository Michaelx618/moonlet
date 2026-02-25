import json
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ToolRepetitionCheck:
    allow_execution: bool
    message: Optional[str] = None


class ToolRepetitionDetector:
    """Detect consecutive identical tool calls (Roo-style)."""

    def __init__(self, limit: int = 3):
        self._limit = max(0, int(limit))
        self._previous_tool_call_json: Optional[str] = None
        self._consecutive_identical_tool_call_count = 0

    def check(self, tool_name: str, tool_args: Dict[str, Any]) -> ToolRepetitionCheck:
        if self._limit <= 0:
            return ToolRepetitionCheck(allow_execution=True)

        current_tool_call_json = self._serialize(tool_name, tool_args)

        if self._previous_tool_call_json == current_tool_call_json:
            self._consecutive_identical_tool_call_count += 1
        else:
            self._consecutive_identical_tool_call_count = 0
            self._previous_tool_call_json = current_tool_call_json

        if self._consecutive_identical_tool_call_count >= self._limit:
            # Reset so the model can recover with a different strategy.
            self._consecutive_identical_tool_call_count = 0
            self._previous_tool_call_json = None
            return ToolRepetitionCheck(
                allow_execution=False,
                message=f"Tool call repetition limit reached for {tool_name}. Please try a different approach.",
            )

        return ToolRepetitionCheck(allow_execution=True)

    @staticmethod
    def _serialize(tool_name: str, tool_args: Dict[str, Any]) -> str:
        payload = {
            "name": (tool_name or "").strip().lower(),
            "args": tool_args or {},
        }
        return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
