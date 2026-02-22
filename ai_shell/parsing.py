"""Output parsing for unified diffs."""

import re
from typing import List, Optional, Tuple

from .utils import dbg


class DiffHunk:
    """A single hunk from a unified diff."""

    __slots__ = ("old_start", "old_count", "new_start", "new_count", "lines")

    def __init__(
        self,
        old_start: int = 0,
        old_count: int = 0,
        new_start: int = 0,
        new_count: int = 0,
        lines: Optional[List[Tuple[str, str]]] = None,
    ):
        self.old_start = old_start
        self.old_count = old_count
        self.new_start = new_start
        self.new_count = new_count
        # each entry is (prefix, content) where prefix is one of " ", "+", "-"
        self.lines = lines or []

    def __repr__(self) -> str:
        return (
            f"DiffHunk(@@ -{self.old_start},{self.old_count} "
            f"+{self.new_start},{self.new_count} @@ {len(self.lines)} lines)"
        )


def parse_unified_diff(
    output: str,
    focus_file: str,
    prompt_prefill: str = "",
    file_content: str = "",
    class_methods: Optional[List[str]] = None,
    keep_all_hunks: bool = False,
) -> Tuple[Optional[str], Optional[List[DiffHunk]]]:
    """Parse a unified diff from model output.

    Robust to garbage before/after the diff. Extracts the first valid diff block
    and parses its hunks.

    prompt_prefill: if provided, the exact text that was pre-filled at the end
    of the prompt (diff header + context + start of first + line). The model
    output is a continuation of this, so we prepend it.

    class_methods / keep_all_hunks are retained for compatibility.
    Returns (file_path, hunks) or (None, None) on failure.
    """
    _ = file_content
    _ = class_methods
    _ = keep_all_hunks

    if not output or not focus_file:
        return None, None, False

    raw = output.replace("\r\n", "\n").replace("\r", "\n")
    lines0 = raw.split("\n")
    has_hunk_header = any(ln.startswith("@@") for ln in lines0)
    has_std_headers = ("+++ b/" in raw or "+++ /dev/null" in raw) and ("--- a/" in raw or "--- /dev/null" in raw)
    has_alt_headers = re.search(r"^---\s+\S+", raw, re.MULTILINE) and re.search(r"^\+\+\+\s+\S+", raw, re.MULTILINE)
    has_file_headers = has_std_headers or has_alt_headers
    prefill = (prompt_prefill or "").replace("\r\n", "\n").replace("\r", "\n")
    prefill_diff_like = bool(prefill) and ("--- a/" in prefill and "+++ b/" in prefill and "@@" in prefill)
    if not (has_hunk_header or has_file_headers or prefill_diff_like):
        dbg("parse_unified_diff: output not diff-like; aborting")
        return None, None, False

    if prompt_prefill:
        # Strip overlap: model may echo tail of prefill.
        prefill_clean = prefill
        overlap = 0
        max_check = min(len(prefill_clean), len(raw), 200)
        for k in range(1, max_check + 1):
            suffix = prefill_clean[-k:]
            if raw.startswith(suffix):
                overlap = k
        if overlap > 0:
            dbg(f"parse_unified_diff: stripping {overlap}-char echo overlap")
            raw = prefill_clean + raw[overlap:]
        else:
            last_nl = prefill_clean.rfind("\n")
            partial_line = prefill_clean[last_nl + 1 :] if last_nl >= 0 else ""
            m_partial = re.match(r"^\+(\s*)", partial_line) if partial_line else None
            m_raw = re.match(r"^\+(\s*)", raw)
            if m_partial and m_raw and m_partial.group(1) == m_raw.group(1):
                skip = 1 + len(m_raw.group(1))
                remainder = raw[skip:]
                overlap2 = 0
                max_check2 = min(len(prefill_clean), len(remainder), 200)
                for k in range(1, max_check2 + 1):
                    suffix = prefill_clean[-k:]
                    if remainder.startswith(suffix):
                        overlap2 = k
                if overlap2 > 0:
                    dbg(
                        "parse_unified_diff: double prefix re-emission, "
                        f"skip={skip} + overlap={overlap2}"
                    )
                    raw = prefill_clean + remainder[overlap2:]
                else:
                    dbg(
                        "parse_unified_diff: model re-emitted +indent prefix, "
                        f"merging (skip={skip} chars)"
                    )
                    raw = prefill_clean + remainder
            else:
                raw = prefill_clean + raw
    elif not has_hunk_header:
        # No @@ and no ---/+++ : reject
        dbg("parse_unified_diff: output not diff-like; aborting")
        return None, None, False

    lines = raw.split("\n")

    minus_idx = None
    for i, line in enumerate(lines):
        if line.startswith("--- a/") or line.startswith("--- /dev/null") or line.startswith("--- "):
            minus_idx = i
            break
    plus_idx = None
    if minus_idx is not None:
        for i in range(minus_idx + 1, min(minus_idx + 5, len(lines))):
            if lines[i].startswith("+++ b/") or lines[i].startswith("+++ /dev/null") or lines[i].startswith("+++ "):
                plus_idx = i
                break

    # Hunk-only diff: model outputs @@ without ---/+++ headers. Use focus_file
    if minus_idx is None or plus_idx is None:
        if not has_hunk_header:
            return None, None, False
        first_at = next((i for i, ln in enumerate(lines) if ln.startswith("@@")), None)
        if first_at is None:
            return None, None, False
        dbg("parse_unified_diff: hunk-only diff (no ---/+++), using focus_file")
        minus_idx = first_at - 1 if first_at > 0 else 0
        plus_idx = first_at - 1  # loop starts at plus_idx+1 = first @@ line
        file_a = file_b = focus_file
    else:
        # New file: --- /dev/null; path from +++ b/. Delete: +++ /dev/null; path from --- a/
        # Also accept --- filename and +++ filename (model often omits a/b prefix)
        minus_line = lines[minus_idx]
        plus_line = lines[plus_idx]
        if minus_line.startswith("--- a/"):
            file_a = minus_line[len("--- a/") :].strip().split("\t")[0].strip()
        elif minus_line.startswith("--- /dev/null"):
            file_a = "/dev/null"
        else:
            file_a = minus_line[4:].strip().split("\t")[0].strip()  # --- filename
        if plus_line.startswith("+++ b/"):
            file_b = plus_line[len("+++ b/") :].strip().split("\t")[0].strip()
        elif plus_line.startswith("+++ /dev/null"):
            file_b = "/dev/null"
        else:
            file_b = plus_line[4:].strip().split("\t")[0].strip()  # +++ filename

    def _path_matches(a: str, b: str) -> bool:
        if a == b:
            return True
        # Allow basename match (e.g. Makefile vs w7/Makefile)
        return a and b and (a.endswith("/" + b) or b.endswith("/" + a) or a.split("/")[-1] == b.split("/")[-1])

    if not _path_matches(file_a, focus_file) and not _path_matches(file_b, focus_file):
        dbg(
            "parse_unified_diff: path mismatch: "
            f"a={file_a!r} b={file_b!r} focus={focus_file!r}"
        )
        return None, None, False

    hunks: List[DiffHunk] = []
    hunk_header_re = re.compile(r"^@@\s+-?(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s*@@")

    i = plus_idx + 1
    current_hunk: Optional[DiffHunk] = None

    while i < len(lines):
        line = lines[i]

        if (line.startswith("--- a/") or line.startswith("--- /dev/null") or line.startswith("--- ")) and i > plus_idx + 1:
            dbg("parse_unified_diff: multi-file diff detected, stopping")
            break

        if any(
            marker in line
            for marker in (
                "--- END FILE",
                "--- BEGIN FILE",
                "END FILE ---",
                "BEGIN FILE ---",
                "Return ONLY",
                "Request:",
                "[[[",
                "Output ONLY",
                "Write complete",
                "CONTEXT BEFORE:",
                "CONTEXT AFTER:",
            )
        ):
            dbg(f"parse_unified_diff: hard stop at boundary: {line[:60]}")
            break

        if line.startswith("@@"):
            if current_hunk and current_hunk.lines:
                hunks.append(current_hunk)

            m = hunk_header_re.match(line)
            if m:
                current_hunk = DiffHunk(
                    old_start=int(m.group(1)),
                    old_count=int(m.group(2)) if m.group(2) else 1,
                    new_start=int(m.group(3)),
                    new_count=int(m.group(4)) if m.group(4) else 1,
                )
                # Model sometimes puts first context on same line: @@ -6,6 +6,7 @@ #include...
                rest = line[m.end() :].lstrip()
                if rest:
                    current_hunk.lines.append((" ", rest))
            else:
                current_hunk = DiffHunk()
            i += 1
            continue

        if current_hunk is not None:
            if line.startswith("+"):
                current_hunk.lines.append(("+", line[1:]))
            elif line.startswith("-"):
                current_hunk.lines.append(("-", line[1:]))
            elif line.startswith(" "):
                current_hunk.lines.append((" ", line[1:]))
            elif line == "":
                has_more = False
                for j in range(i + 1, min(i + 3, len(lines))):
                    if lines[j].startswith(("+", "-", " ", "@@")):
                        has_more = True
                        break
                if has_more:
                    current_hunk.lines.append((" ", ""))
                else:
                    break
            elif line and not line.startswith("@@") and not line.startswith("---"):
                # Lenient: model often omits space prefix on context (YAML strips it).
                # Treat as context so we can parse and apply at least one hunk.
                current_hunk.lines.append((" ", line))
            else:
                break
        i += 1

    if current_hunk and current_hunk.lines:
        hunks.append(current_hunk)

    if not hunks:
        payload: List[Tuple[str, str]] = []
        for line in lines[plus_idx + 1 :]:
            if not line:
                continue
            if line.startswith("+") and not line.startswith("+++"):
                payload.append(("+", line[1:]))
                continue
            if line.startswith("-") and not line.startswith("---"):
                payload.append(("-", line[1:]))
                continue
            if line.startswith(("diff --git ", "index ", "new file mode ", "deleted file mode ")):
                continue
            if line == r"\ No newline at end of file":
                continue
            break
        if payload and any(p == "+" for p, _ in payload):
            hunks.append(
                DiffHunk(
                    old_start=1,
                    old_count=1,
                    new_start=1,
                    new_count=1,
                    lines=payload,
                )
            )

    active_hunks = []
    for hunk in hunks:
        has_delta = any(p in ("+", "-") for p, _ in hunk.lines)
        if not has_delta:
            dbg(
                "parse_unified_diff: dropping no-op hunk "
                f"@@ -{hunk.old_start},{hunk.old_count}"
            )
            continue
        active_hunks.append(hunk)
    hunks = active_hunks

    if not hunks:
        dbg("parse_unified_diff: no hunks found (or all were no-op)")
        return None, None, False

    is_delete = file_b == "/dev/null"
    dbg(f"parse_unified_diff: parsed {len(hunks)} active hunk(s) for {focus_file} (delete={is_delete})")
    return focus_file, hunks, is_delete


def hunks_to_diff(hunks: List["DiffHunk"], path: str) -> str:
    """Serialize hunks back to unified diff format."""
    lines = [f"--- a/{path}", f"+++ b/{path}"]
    for hunk in hunks:
        oc = hunk.old_count if hunk.old_count else 1
        nc = hunk.new_count if hunk.new_count else 1
        lines.append(f"@@ -{hunk.old_start},{oc} +{hunk.new_start},{nc} @@")
        for prefix, content in (hunk.lines or []):
            lines.append((prefix if prefix else " ") + content)
    return "\n".join(lines)
