import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import config, prompt_buffer
from .files import _norm_rel_path, get_root, read_single_file_for_context
from .model import stream_reply
from .prompts import _ext
from .router import Route
from .utils import dbg, dbg_dump
from .validation import _validate_prompt_semantics


@dataclass
class FileblockRuntime:
    temp_execute: float
    parse_analysis_packet: Callable[[str], Dict[str, object]]
    append_history: Callable[[str, str], None]
    span: Callable[[List[Dict[str, object]], str], Any]
    build_meta: Callable[..., Dict[str, object]]
    compute_unified_diff: Callable[[str, str, str], str]
    sandbox_validate_candidate_content: Callable[..., Tuple[bool, Optional[str]]]
    normalize_failure_reason: Callable[..., str]
    build_compile_report: Callable[[str], str]
    candidate_fullfile_guard_reason: Callable[[str, str, str], Optional[str]]
    extract_failure_from_output: Callable[[str], Tuple[str, str]]
    build_retry_base_request: Callable[..., str]
    build_noop_report: Callable[[], str]
    set_smoke_meta: Callable[..., None]
    run_smoke_for_candidate: Callable[..., Tuple[bool, Optional[bool], List[str], str]]
    annotate_failure_meta: Callable[..., None]


def extract_fileblock_content(output: str, focus_file: str) -> Tuple[Optional[str], Optional[str]]:
    text = (output or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"```[A-Za-z0-9_-]*", "", text)
    if "[[[file:" in text and "[[[end]]]" not in text:
        dbg("fileblock.normalization=appended_end_marker")
        text = text.rstrip() + "\n[[[end]]]"
    block_re = re.compile(r"\[\[\[file:\s*([^\]]+)\]\]\]\s*(.*?)\s*\[\[\[end\]\]\]", re.DOTALL | re.IGNORECASE)
    m = block_re.search(text)
    if m:
        path = _norm_rel_path(str(m.group(1) or "").strip())
        focus_norm = _norm_rel_path(focus_file)
        if path and focus_norm and path != focus_norm:
            return None, "FILEBLOCK_PATH_MISMATCH"
        return m.group(2), None

    start_re = re.compile(r"\[\[\[file:\s*([^\]]+)\]\]\]\s*", re.IGNORECASE)
    sm = start_re.search(text)
    if not sm:
        return None, "FILEBLOCK_MISSING"
    path = _norm_rel_path(str(sm.group(1) or "").strip())
    focus_norm = _norm_rel_path(focus_file)
    if path and focus_norm and path != focus_norm:
        return None, "FILEBLOCK_PATH_MISMATCH"
    body = text[sm.end() :].strip()
    if not body:
        return None, "FILEBLOCK_EMPTY"
    return body, None


def run_fileblock_write(
    runtime: FileblockRuntime,
    user_text: str,
    focus_file: str,
    silent: bool,
    full_context: bool,
    analysis_packet: str = "",
    sliced_request: str = "",
    focus_content_override: Optional[str] = None,
    diagnostics_hint: str = "",
    hard_fail_on_verify_fail: bool = False,
    force_stage_on_verify_fail: bool = False,
    defer_persist: bool = False,
) -> Dict[str, object]:
    start = time.time()
    trace: List[Dict[str, object]] = []
    model_calls_used = 0
    target_ext = _ext(focus_file)
    original_content = focus_content_override
    if original_content is None:
        original_content = read_single_file_for_context(focus_file).get(focus_file, "")
    analysis_policy = runtime.parse_analysis_packet(analysis_packet)
    preferred_validate_cmd = str(analysis_policy.get("suggested_validate_cmd") or "").strip()
    request_text = (sliced_request or user_text or "").strip()
    if diagnostics_hint:
        request_text += (
            "\n\nRETRY DIAGNOSTICS:\n"
            f"{diagnostics_hint}\n"
            "Fix these issues with minimal edits."
        )

    def _call(prompt: str, span: str) -> str:
        nonlocal model_calls_used
        model_calls_used += 1
        with runtime.span(trace, span):
            return stream_reply(
                prompt,
                silent=silent,
                temperature=runtime.temp_execute,
                max_new=config.DIFF_MAX_NEW,
            )

    prompt, ctx_meta = prompt_buffer.build_prompt(
        "agent",
        request_text,
        focus_file=focus_file,
        full_context=full_context,
        force_file_block=True,
        output_contract="",
        error_message=diagnostics_hint,
        pre_sliced_request=request_text,
        focus_content_override=focus_content_override,
        analysis_packet=analysis_packet,
    )
    output = _call(prompt, "fileblock_attempt_1")
    runtime.append_history(user_text, output)
    dbg_dump("fileblock_output_1", output)

    if str(output).strip() == "[Model timeout]":
        meta = runtime.build_meta(len(prompt), output, 0, False, start, trace, model_calls_used)
        meta.update(ctx_meta)
        meta["mode_used"] = Route.FILE_EDIT.value
        meta["route"] = Route.FILE_EDIT.value
        meta["reject_rule"] = "MODEL_TIMEOUT"
        meta["file_edit_failure_kind"] = "timeout"
        meta["failure_kind"] = "timeout"
        meta["failure_reason"] = f"model timed out after {config.GEN_TIMEOUT}s"
        return {
            "output": "[File edit failed: MODEL_TIMEOUT]",
            "meta": meta,
            "status_prefix": "",
            "blocks_count": 0,
            "retried": False,
            "timeout": True,
        }

    candidate_content, parse_err = extract_fileblock_content(output, focus_file)
    if parse_err or candidate_content is None:
        dbg(f"fileblock.reject_reason={parse_err or 'FILEBLOCK_PARSE_FAIL'}")
        meta = runtime.build_meta(len(prompt), output, 0, False, start, trace, model_calls_used)
        meta.update(ctx_meta)
        meta["mode_used"] = Route.FILE_EDIT.value
        meta["route"] = Route.FILE_EDIT.value
        meta["reject_rule"] = parse_err or "FILEBLOCK_PARSE_FAIL"
        meta["file_edit_failure_kind"] = "format"
        meta["failure_kind"] = "format"
        meta["failure_reason"] = str(parse_err or "fileblock parse failed")
        return {
            "output": f"[File edit failed: {parse_err or 'FILEBLOCK_PARSE_FAIL'}]",
            "meta": meta,
            "status_prefix": "",
            "blocks_count": 0,
            "retried": False,
            "timeout": False,
            "candidate_content": "",
            "candidate_diff": "",
        }

    candidate_diff = runtime.compute_unified_diff(original_content or "", candidate_content, focus_file)
    ok, err = runtime.sandbox_validate_candidate_content(
        focus_file=focus_file,
        candidate_content=candidate_content,
        preferred_validate_cmd=preferred_validate_cmd,
    )
    no_content_change = (candidate_content == (original_content or "")) or (not (candidate_diff or "").strip())
    if no_content_change:
        dbg("fileblock.noop_change_detected")
        meta = runtime.build_meta(len(prompt), output, 0, False, start, trace, model_calls_used)
        meta.update(ctx_meta)
        meta["mode_used"] = Route.FILE_EDIT.value
        meta["route"] = Route.FILE_EDIT.value
        meta["reject_rule"] = "NO_EFFECTIVE_CHANGE"
        meta["failure_kind"] = "noop"
        meta["file_edit_failure_kind"] = "noop"
        meta["failure_reason"] = "model returned no effective file changes"
        meta["noop"] = True
        return {
            "output": "[No-op file_edit: model returned no effective file changes]",
            "meta": meta,
            "status_prefix": "",
            "blocks_count": 0,
            "retried": False,
            "timeout": False,
            "candidate_content": candidate_content,
            "candidate_diff": candidate_diff,
        }
    warning_kind = ""
    warning_reason = ""
    if not ok:
        if hard_fail_on_verify_fail:
            meta = runtime.build_meta(len(prompt), output, 0, False, start, trace, model_calls_used)
            meta.update(ctx_meta)
            meta["mode_used"] = Route.FILE_EDIT.value
            meta["route"] = Route.FILE_EDIT.value
            meta["reject_rule"] = "compile_or_verify_fail"
            meta["file_edit_failure_kind"] = "syntax"
            meta["failure_kind"] = "syntax"
            meta["failure_reason"] = runtime.normalize_failure_reason(err or "compile failed", max_chars=400)
            meta["compile_report"] = runtime.build_compile_report(err or "compile failed")
            meta["diff"] = candidate_diff
            return {
                "output": f"[File edit failed: SYNTAX_FAIL: {(err or 'compile failed')[:500]}]",
                "meta": meta,
                "status_prefix": "",
                "blocks_count": 0,
                "retried": False,
                "timeout": False,
                "candidate_content": candidate_content,
                "candidate_diff": candidate_diff,
            }
        dbg("verify.non_blocking_fail")
        warning_kind = "syntax"
        warning_reason = runtime.normalize_failure_reason(err or "compile failed", max_chars=400)

    collapse_reason = runtime.candidate_fullfile_guard_reason(
        original_content or "",
        candidate_content or "",
        target_ext,
    )
    if collapse_reason:
        meta = runtime.build_meta(len(prompt), output, 0, False, start, trace, model_calls_used)
        meta.update(ctx_meta)
        meta["mode_used"] = Route.FILE_EDIT.value
        meta["route"] = Route.FILE_EDIT.value
        meta["reject_rule"] = "wipe_risk"
        meta["file_edit_failure_kind"] = "rewrite_risk"
        meta["failure_kind"] = "rewrite_risk"
        meta["failure_reason"] = runtime.normalize_failure_reason(collapse_reason, max_chars=400)
        if candidate_diff:
            meta["diff"] = candidate_diff
        return {
            "output": f"[File edit failed: WIPE_RISK: {collapse_reason}]",
            "meta": meta,
            "status_prefix": "",
            "blocks_count": 0,
            "retried": False,
            "timeout": False,
            "candidate_content": candidate_content,
            "candidate_diff": candidate_diff,
        }

    semantic_err = _validate_prompt_semantics(request_text, candidate_content)
    if semantic_err:
        dbg("verify.semantic_non_blocking_fail")
        if hard_fail_on_verify_fail:
            meta = runtime.build_meta(len(prompt), output, 0, False, start, trace, model_calls_used)
            meta.update(ctx_meta)
            meta["mode_used"] = Route.FILE_EDIT.value
            meta["route"] = Route.FILE_EDIT.value
            meta["reject_rule"] = "semantic_fail"
            meta["file_edit_failure_kind"] = "semantic"
            meta["failure_kind"] = "semantic"
            meta["failure_reason"] = runtime.normalize_failure_reason(semantic_err, max_chars=400)
            meta["diff"] = candidate_diff
            return {
                "output": f"[File edit failed: SEMANTIC_FAIL: {semantic_err[:500]}]",
                "meta": meta,
                "status_prefix": "",
                "blocks_count": 0,
                "retried": False,
                "timeout": False,
                "candidate_content": candidate_content,
                "candidate_diff": candidate_diff,
            }
        if not warning_kind:
            warning_kind = "semantic"
        sem_reason = runtime.normalize_failure_reason(semantic_err, max_chars=400)
        warning_reason = f"{warning_reason}; {sem_reason}" if warning_reason else sem_reason

    meta = runtime.build_meta(len(prompt), output, 0, False, start, trace, model_calls_used)
    meta.update(ctx_meta)
    meta["mode_used"] = Route.FILE_EDIT.value
    meta["route"] = Route.FILE_EDIT.value
    meta["diff"] = candidate_diff
    meta["syntax_error_count"] = 0
    if warning_kind:
        meta["verify_warning"] = True
        meta["non_blocking_failure"] = True
        meta["failure_kind"] = warning_kind
        meta["file_edit_failure_kind"] = warning_kind
        meta["failure_reason"] = warning_reason

    if defer_persist:
        meta["deferred_persist"] = True
        return {
            "output": "[Candidate file_edit]",
            "meta": meta,
            "status_prefix": "",
            "blocks_count": 0,
            "retried": False,
            "timeout": False,
            "candidate_content": candidate_content,
            "candidate_diff": candidate_diff,
        }

    force_stage = bool(force_stage_on_verify_fail and warning_kind)
    if force_stage:
        meta["requires_user_decision"] = True
    if config.STAGE_EDITS or force_stage:
        meta["staged"] = True
        meta["staged_file"] = focus_file
        meta["staged_content"] = candidate_content
        return {
            "output": "[Staged file_edit]",
            "meta": meta,
            "status_prefix": "",
            "blocks_count": 0,
            "retried": False,
            "timeout": False,
            "candidate_content": candidate_content,
            "candidate_diff": candidate_diff,
        }

    target = get_root() / focus_file
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(candidate_content, encoding="utf-8")
    return {
        "output": "[Applied file_edit]",
        "meta": meta,
        "status_prefix": "",
        "blocks_count": 0,
        "retried": False,
        "timeout": False,
        "candidate_content": candidate_content,
        "candidate_diff": candidate_diff,
    }


def run_fileblock_edit(
    runtime: FileblockRuntime,
    user_text: str,
    focus_file: str,
    silent: bool,
    full_context: bool,
    analysis_packet: str = "",
    sliced_request: str = "",
) -> Dict[str, object]:
    attempt_total = 2
    baseline_content = read_single_file_for_context(focus_file).get(focus_file, "")
    first = run_fileblock_write(
        runtime=runtime,
        user_text=user_text,
        focus_file=focus_file,
        silent=silent,
        full_context=full_context,
        analysis_packet=analysis_packet,
        sliced_request=sliced_request,
        focus_content_override=baseline_content,
        diagnostics_hint="",
        hard_fail_on_verify_fail=True,
        force_stage_on_verify_fail=False,
    )
    runtime.annotate_failure_meta(first, attempt_index=1, attempt_total=attempt_total, focus_file=focus_file)
    first_meta = first.get("meta", {}) or {}
    first_meta["edit_mode"] = "fileblock"
    retry_trigger = "none"
    retry_hint = ""
    prefer_task_card_retry = False
    first_success = str(first.get("output", "")).startswith("[Applied file_edit]") or str(first.get("output", "")).startswith("[Staged file_edit]")
    if first_success:
        attempted, passed, failures, smoke_report = runtime.run_smoke_for_candidate(
            focus_file=focus_file,
            candidate_content=str(first.get("candidate_content") or ""),
        )
        runtime.set_smoke_meta(first_meta, attempted=attempted, passed=passed, failures=failures)
        first_meta["retry_trigger"] = "none"
        if not attempted or bool(passed):
            dbg("retry.success attempt=1")
            return first
        retry_trigger = "smoke"
        retry_hint = smoke_report or "SMOKE_REPORT:\n- failure: smoke test failed"
        dbg("retry.invoked attempt=2 reason=smoke_fail")
    else:
        runtime.set_smoke_meta(first_meta, attempted=False, passed=None, failures=[])
        reject_rule = str(first_meta.get("reject_rule") or "").strip()
        if bool(first_meta.get("timeout")):
            dbg("retry.skipped reason=timeout_first_attempt")
            return first
        if reject_rule in {"MODEL_TIMEOUT", "WIPE_RISK"}:
            dbg(f"retry.skipped reason=non_retry_rule:{reject_rule}")
            return first

        fail_reason = str(first_meta.get("failure_reason") or "").strip()
        fail_kind = str(first_meta.get("failure_kind") or "").strip()
        if not fail_reason:
            fail_kind, fail_reason = runtime.extract_failure_from_output(str(first.get("output") or ""))
        if fail_kind in {"syntax", "semantic"}:
            retry_hint = runtime.build_compile_report(fail_reason)
            retry_trigger = "compile"
        elif fail_kind == "noop":
            retry_hint = runtime.build_noop_report()
            retry_trigger = "noop"
            prefer_task_card_retry = True
        else:
            retry_hint = fail_reason
            retry_trigger = "none"
        dbg("retry.invoked attempt=2")

    first_meta["retry_trigger"] = retry_trigger
    retry_request = runtime.build_retry_base_request(
        sliced_request=sliced_request,
        user_text=user_text,
        focus_file=focus_file,
        file_content=baseline_content,
        prefer_task_card=prefer_task_card_retry,
    )
    retry_request += (
        "\n\nRETRY RULES:\n"
        "- Return exactly one [[[file: ...]]] block and [[[end]]].\n"
        "- Output full file content only.\n"
        "- No markdown fences or prose.\n"
        "- If diagnostics indicate no-op, ensure output differs from current file.\n"
    )
    second = run_fileblock_write(
        runtime=runtime,
        user_text=user_text,
        focus_file=focus_file,
        silent=silent,
        full_context=full_context,
        analysis_packet=analysis_packet,
        sliced_request=retry_request,
        focus_content_override=baseline_content,
        diagnostics_hint=retry_hint,
        hard_fail_on_verify_fail=False,
        force_stage_on_verify_fail=True,
    )
    runtime.annotate_failure_meta(second, attempt_index=2, attempt_total=attempt_total, focus_file=focus_file)
    second_meta = second.get("meta", {}) or {}
    second_meta["edit_mode"] = "fileblock"
    runtime.set_smoke_meta(second_meta, attempted=False, passed=None, failures=[])
    second_meta["retry_trigger"] = retry_trigger
    if str(second.get("output", "")).startswith("[Applied file_edit]") or str(second.get("output", "")).startswith("[Staged file_edit]"):
        if bool(second_meta.get("non_blocking_failure")):
            second_meta["requires_user_decision"] = True
        dbg("retry.success attempt=2")
        return second
    dbg("retry.failed attempt=2")
    return second
