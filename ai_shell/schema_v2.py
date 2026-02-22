"""Strict v2 contracts for CLI-first core execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


ALLOWED_ACTIONS = {
    "search_replace",
    "write_file",
    "read_file",
    "list_files",
    "grep",
    "run_terminal_cmd",
    "run_verify",
}


@dataclass
class VerifySpecV2:
    build_cmd: str = ""
    run_cmd: str = ""
    cwd: str = "."
    timeout_s: int = 60


@dataclass
class SuccessCriterionV2:
    type: str = "tests_pass"
    value: Any = None
    path: str = ""


@dataclass
class ExecutionRequestV2:
    instruction: str
    root: str
    focus_file: str = ""
    allowed_edit_files: List[str] = field(default_factory=list)
    extra_read_files: List[str] = field(default_factory=list)
    max_iterations: int = 5
    verify: VerifySpecV2 = field(default_factory=VerifySpecV2)
    success_criterion: SuccessCriterionV2 = field(default_factory=SuccessCriterionV2)


@dataclass
class ActionCallV2:
    name: str
    args: Dict[str, Any]

    def validate(self) -> Optional[str]:
        if self.name not in ALLOWED_ACTIONS:
            return f"unsupported_action:{self.name}"
        if not isinstance(self.args, dict):
            return f"invalid_action_args:{self.name}"
        return None


@dataclass
class VerifyArtifactsV2:
    build_stdout: str = ""
    build_stderr: str = ""
    run_stdout: str = ""
    run_stderr: str = ""
    return_code: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "build_stdout": self.build_stdout,
            "build_stderr": self.build_stderr,
            "run_stdout": self.run_stdout,
            "run_stderr": self.run_stderr,
            "return_code": self.return_code,
        }


@dataclass
class ActionBatchResultV2:
    parsed_actions: List[ActionCallV2] = field(default_factory=list)
    rejected_actions: List[Dict[str, Any]] = field(default_factory=list)
    raw_output: str = ""

    @property
    def has_actions(self) -> bool:
        return bool(self.parsed_actions)


@dataclass
class ExecutionResultV2:
    ok: bool
    mode_used: str = "core_v2"
    summary: str = ""
    touched: List[str] = field(default_factory=list)
    per_file_diffs: Dict[str, str] = field(default_factory=dict)
    per_file_before: Dict[str, str] = field(default_factory=dict)
    per_file_after: Dict[str, str] = field(default_factory=dict)
    verify_artifacts: VerifyArtifactsV2 = field(default_factory=VerifyArtifactsV2)
    iterations_used: int = 0
    failure_kind: str = ""
    failure_reason: str = ""
    raw_output: str = ""
    action_logs: Dict[str, Any] = field(default_factory=dict)
    traces: List[Dict[str, Any]] = field(default_factory=list)

    def to_meta(self) -> Dict[str, Any]:
        return {
            "mode_used": self.mode_used,
            "ok": self.ok,
            "summary": self.summary,
            "touched": self.touched,
            "files_changed": self.touched,
            "per_file_diffs": self.per_file_diffs,
            "per_file_before": self.per_file_before,
            "per_file_staged": self.per_file_after,
            "applied_directly": self.ok and bool(self.touched),
            "verify_artifacts": self.verify_artifacts.to_dict(),
            "iteration": self.iterations_used,
            "failure_kind": self.failure_kind,
            "failure_reason": self.failure_reason,
            "raw_output": self.raw_output[:2000],
            "core_v2_actions": self.action_logs,
            "core_v2_traces": self.traces,
        }


def parse_action_batch_v2(raw_output: str, action_calls: List[ActionCallV2]) -> ActionBatchResultV2:
    """Validate parsed actions and classify rejections."""
    out = ActionBatchResultV2(raw_output=raw_output)
    for action in action_calls:
        reason = action.validate()
        if reason:
            out.rejected_actions.append({"name": action.name, "reason": reason})
        else:
            out.parsed_actions.append(action)
    return out
