"""Planner stage for core_v2 (separate from deterministic executor)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from . import config
from .model import get_session_cache_key, stream_reply
from .schema_v2 import ActionBatchResultV2, ActionCallV2, parse_action_batch_v2
from .search_replace import parse_search_replace_calls, parse_write_file_calls


@dataclass
class PlannerTraceV2:
    iteration: int
    prompt_len: int
    raw_output: str
    retry_raw_output: str = ""
    parsed_count: int = 0
    rejected_count: int = 0
    notes: List[str] = field(default_factory=list)


def _parse_actions(raw: str) -> ActionBatchResultV2:
    actions: List[ActionCallV2] = []
    for edit in parse_search_replace_calls(raw):
        actions.append(ActionCallV2(name="search_replace", args=edit))
    for edit in parse_write_file_calls(raw):
        actions.append(ActionCallV2(name="write_file", args=edit))
    return parse_action_batch_v2(raw, actions)


def _generate_plan(prompt: str, role: str = "edit") -> str:
    out = stream_reply(
        prompt,
        label="",
        silent=True,
        max_new=getattr(config, "PATCH_MAX_NEW", 2000),
        stop_sequences=["\nUser:", "\n\n\n"],
        temperature=getattr(config, "TEMPERATURE", 0.25),
        cache_key=get_session_cache_key(),
        role=role,
    )
    return (out or "").strip()


def plan_action_batch(
    prompt: str,
    iteration: int,
    strict_retry: bool = True,
) -> tuple[ActionBatchResultV2, PlannerTraceV2]:
    raw = _generate_plan(prompt, role="edit")
    parsed = _parse_actions(raw)
    trace = PlannerTraceV2(
        iteration=iteration,
        prompt_len=len(prompt),
        raw_output=raw,
        parsed_count=len(parsed.parsed_actions),
        rejected_count=len(parsed.rejected_actions),
    )
    if parsed.has_actions or not strict_retry:
        return parsed, trace

    snippet = raw[:300].strip()
    retry_prompt = (
        f"{prompt}\n"
        f"Your previous output could NOT be parsed. First 300 chars:\n---\n{snippet}\n---\n"
        "Output only search_replace(...) or write_file(...) calls."
    )
    retry_raw = _generate_plan(retry_prompt, role="edit")
    parsed_retry = _parse_actions(retry_raw)
    trace.retry_raw_output = retry_raw
    trace.parsed_count = len(parsed_retry.parsed_actions)
    trace.rejected_count = len(parsed_retry.rejected_actions)
    trace.notes.append("strict_retry_used")
    return parsed_retry, trace

