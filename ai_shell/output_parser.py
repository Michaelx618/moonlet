"""Flexible output parser for freedom mode: extract edits from model output.

Tries in order: unified diff, [[[file: path]]] blocks, markdown code blocks.
Returns List[Tuple[path, content]] for apply_blocks, or (path, hunks) for diff apply.
"""

import re
from pathlib import Path
from typing import List, Optional, Tuple

from .files import _norm_rel_path
from .utils import dbg


def strip_code_blocks_for_display(text: str) -> str:
    """Remove code blocks from text, leaving only explanation for agent panel display."""
    if not (text or "").strip():
        return ""
    t = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    # Remove ```lang ... ``` blocks: split by ``` and keep only non-code parts
    parts = re.split(r"```[\w]*\s*\n", t)
    explanation_parts = []
    for i, part in enumerate(parts):
        if i == 0:
            explanation_parts.append(part)
        else:
            # This part started with ``` - find the closing ``` and take text after
            end = part.find("\n```")
            if end >= 0:
                explanation_parts.append(part[end + 4 :])  # after \n```
            # else: unclosed block, discard
    t = "".join(explanation_parts)
    # Remove any remaining ``` lines
    t = re.sub(r"\n?```\s*\n?", "\n", t)
    # Remove [[[file: path]]] ... [[[end]]] blocks
    t = re.sub(r"\[\[\[file:\s*[^\]]+\]\]\].*?\[\[\[end\]\]\]", "", t, flags=re.DOTALL | re.IGNORECASE)
    # Remove unified diff blocks (only when we've seen --- a/ or +++ b/)
    lines = t.split("\n")
    out: list = []
    in_diff = False
    for line in lines:
        if line.startswith("--- a/") or line.startswith("+++ b/") or line.startswith("--- /dev/null"):
            in_diff = True
        if in_diff:
            is_diff_line = (
                line.startswith(("---", "+++", "@@"))
                or (len(line) >= 1 and line[0] in " -+")
            )
            if line.strip() and not is_diff_line:
                in_diff = False
            else:
                continue
        out.append(line)
    t = "\n".join(out)
    result = t.strip()
    # Fallback: if we stripped everything, try to extract "Explanation:" and after
    if not result and "Explanation" in (text or ""):
        idx = (text or "").find("Explanation")
        if idx >= 0:
            result = text[idx:].strip()
    return result


def _looks_like_code(content: str, min_lines: int = 1) -> bool:
    """Heuristic: content looks like source code."""
    content = (content or "").strip()
    if len(content) < 10:
        return False
    lines = content.splitlines()
    if len(lines) < min_lines:
        return False
    # Reject prose
    if any(
        line.strip().lower().startswith(("the ", "this ", "we ", "you ", "i "))
        for line in lines[:3]
    ):
        return False
    return True


def extract_file_blocks(output: str) -> List[Tuple[str, str]]:
    """Extract all [[[file: path]]] ... [[[end]]] blocks from output.

    Returns List[(path, content)]. Paths are normalized relative to repo root.
    """
    if not output:
        return []
    text = (output or "").replace("\r\n", "\n").replace("\r", "\n")

    # Find all blocks: [[[file: path]]] content [[[end]]]
    block_re = re.compile(
        r"\[\[\[file:\s*([^\]]+)\]\]\]\s*(.*?)\s*\[\[\[end\]\]\]",
        re.DOTALL | re.IGNORECASE,
    )
    blocks: List[Tuple[str, str]] = []
    for m in block_re.finditer(text):
        path = _norm_rel_path(str(m.group(1) or "").strip())
        body = (m.group(2) or "").strip()
        if path and body and _looks_like_code(body):
            blocks.append((path, body))
    if blocks:
        dbg(f"output_parser: extracted {len(blocks)} file block(s)")
    return blocks


def _infer_path_from_code(body: str) -> Optional[str]:
    """Infer target file path from code content (e.g. Usage: parentcreates -> parentcreates.c)."""
    if not body:
        return None
    # Usage: programname or usage: programname
    m = re.search(r'[Uu]sage:\s*["\']?([a-zA-Z0-9_]+)', body)
    if m:
        name = (m.group(1) or "").strip()
        if name:
            # Prefer .c for C-like code
            if "include" in body or "fork" in body or "wait" in body or "printf" in body:
                return _norm_rel_path(f"{name}.c")
            if "def " in body or "import " in body:
                return _norm_rel_path(f"{name}.py")
            return _norm_rel_path(f"{name}.c")
    return None


def extract_markdown_blocks(output: str) -> List[Tuple[str, str]]:
    """Extract markdown code blocks with path: ```path or ```lang:path."""
    if not output:
        return []
    text = (output or "").replace("\r\n", "\n").replace("\r", "\n")
    blocks: List[Tuple[str, str]] = []

    # Pattern 1: ```path or ```file: path - has path (group 1) and body (group 2)
    pat1 = re.compile(
        r"```(?:file:\s*)?([A-Za-z0-9_./-]+)\s*\n(.*?)(?=\n```|\Z)",
        re.DOTALL,
    )
    for m in pat1.finditer(text):
        path_part = (m.group(1) or "").strip()
        body = (m.group(2) or "").strip()
        if not body or not _looks_like_code(body):
            continue
        # Skip if path_part is a lang tag (c, cpp, etc.) - pattern 2 handles those
        if path_part.lower() in ("c", "cpp", "cc", "cxx", "py", "js", "ts", "go", "rs"):
            continue
        # Skip "diff" — that's a unified diff block, not a filename
        if path_part.lower() == "diff":
            continue
        path = _norm_rel_path(path_part)
        if path:
            blocks.append((path, body))

    # Pattern 2: ```c, ```cpp, etc. - only body (group 1), infer path from content
    pat2 = re.compile(
        r"```(?:c|cpp|cc|cxx|py|js|ts|go|rs)\s*\n(.*?)(?=\n```|\Z)",
        re.DOTALL | re.IGNORECASE,
    )
    for m in pat2.finditer(text):
        body = (m.group(1) or "").strip()
        if not body or not _looks_like_code(body):
            continue
        inferred = _infer_path_from_code(body)
        if inferred:
            blocks.append((inferred, body))
            dbg(f"output_parser: inferred path {inferred} from lang-only block")

    if blocks:
        dbg(f"output_parser: extracted {len(blocks)} markdown block(s)")
    return blocks


def extract_markdown_lang_only_blocks(
    output: str,
    candidate_paths: Optional[List[str]] = None,
) -> List[Tuple[str, str]]:
    """Extract ```c, ```cpp, etc. blocks (no path), strip markdown, infer path from content or candidates."""
    if not output:
        return []
    text = (output or "").replace("\r\n", "\n").replace("\r", "\n")
    # Match ```lang\n...\n``` (any common lang)
    pat = re.compile(
        r"```(?:c|cpp|cc|cxx|py|js|ts|go|rs|java|makefile?)\s*\n(.*?)(?=\n```|\Z)",
        re.DOTALL | re.IGNORECASE,
    )
    blocks: List[Tuple[str, str]] = []
    candidates = [_norm_rel_path(p) for p in (candidate_paths or [])]
    idx = 0
    for m in pat.finditer(text):
        body = (m.group(1) or "").strip()
        if not body or not _looks_like_code(body):
            continue
        path = _infer_path_from_code(body)
        if not path and idx < len(candidates):
            path = candidates[idx]
            idx += 1
        elif not path and candidates:
            path = candidates[0]
        if path:
            blocks.append((path, body))
            dbg(f"output_parser: extracted lang-only block -> {path}")
    return blocks


def _resolve_path_with_candidates(
    inferred: str, candidate_paths: Optional[List[str]] = None
) -> str:
    """Prefer candidate path that matches inferred basename (e.g. w6/childcreates.c over childcreates.c)."""
    if not inferred:
        return inferred
    base = inferred.split("/")[-1]
    candidates = candidate_paths or []
    for c in candidates:
        norm = _norm_rel_path(c)
        if norm.endswith("/" + base) or norm == base:
            return norm
    return inferred


def parse_flexible_output(
    output: str,
    candidate_paths: Optional[List[str]] = None,
) -> Tuple[Optional[str], Optional[object], Optional[List[Tuple[str, str]]]]:
    """Parse model output into editable format.

    candidate_paths: when stripping markdown blocks without path, use first candidate as fallback.
    Also used to resolve inferred paths (e.g. childcreates.c -> w6/childcreates.c).

    Returns:
        - ("blocks", None, [(path, content), ...]) for file blocks
        - ("diff", (path, hunks), None) for unified diff
        - (None, None, None) if nothing parseable
    """
    if not (output or "").strip():
        return None, None, None

    # 1. Try unified diff FIRST — strip fences, then use path from header (+++ b/ or --- a/)
    from .parsing import parse_unified_diff

    # Step 1: Strip ```diff and ``` fences first
    diff_text = _strip_diff_markdown_fences(output)
    # Step 2: Check indicator and header for path
    path_from_header = _get_path_from_diff_header(diff_text)
    if path_from_header:
        file_path, hunks, is_delete = parse_unified_diff(diff_text, focus_file=path_from_header)
        if file_path and hunks:
            dbg(f"output_parser: parsed unified diff for {file_path} ({len(hunks)} hunks, delete={is_delete})")
            return "diff", (file_path, hunks, is_delete), None

    # 2. Try file blocks
    blocks = extract_file_blocks(output)
    if not blocks:
        blocks = extract_markdown_blocks(output)
    if not blocks:
        blocks = extract_markdown_lang_only_blocks(output, candidate_paths)
    if blocks:
        resolved = [
            (_resolve_path_with_candidates(p, candidate_paths), c) for p, c in blocks
        ]
        return "blocks", None, resolved

    return None, None, None


def _reformat_oneliner_diff(block: str) -> str:
    """Reformat a diff that the model emitted as one line (no newlines) into proper line format."""
    if not block or block.count("\n") > 2:
        return block
    # Split before structural markers: @@ (hunk), --- (file header), +++ (file header)
    # and before diff content lines: space+minus, space+plus
    out = re.sub(r" (?=@@ -)", "\n", block)
    out = re.sub(r" (?=--- a/|--- /dev/null)", "\n", out)
    out = re.sub(r" (?=\+\+\+ b/|\+\+\+ /dev/null)", "\n", out)
    # Diff content: " -line" and " +line" become newlines (avoid splitting inside strings)
    out = re.sub(r" (?=-[^\s]|\+[^\s])", "\n", out)
    return out.strip()


def _strip_diff_markdown_fences(output: str) -> str:
    """Strip ```diff and ``` fences. Return first diff block only (avoids duplicate apply)."""
    if not output:
        return output
    text = (output or "").replace("\r\n", "\n").replace("\r", "\n")
    for m in re.finditer(r"```(?:diff)?\s*\n(.*?)(?=\n```|\Z)", text, re.DOTALL):
        block = (m.group(1) or "").strip()
        # Accept --- a/path (edit), --- /dev/null (new file), +++ /dev/null (delete)
        has_minus = ("--- a/" in block) or ("--- /dev/null" in block)
        has_plus = ("+++ b/" in block) or ("+++ /dev/null" in block)
        if has_minus and has_plus and "@@" in block:
            # Model sometimes emits diff as one line; reformat before parsing
            if block.count("\n") < 2:
                block = _reformat_oneliner_diff(block)
                dbg("output_parser: reformatted one-liner diff")
            # Take only the first diff — model may emit same diff twice
            lines = block.split("\n")
            out_lines = []
            seen_file_header = False
            for i, line in enumerate(lines):
                if (line.startswith("--- a/") or line.startswith("--- /dev/null")) and i > 0 and seen_file_header:
                    break
                if line.startswith("--- a/") or line.startswith("--- /dev/null"):
                    seen_file_header = True
                out_lines.append(line)
            result = "\n".join(out_lines)
            dbg("output_parser: stripped diff markdown fences")
            return result
    return text


def _get_path_from_diff_header(diff_text: str) -> str:
    """Check indicator and header; return path from +++ b/ (preferred) or --- a/ (for delete), else empty."""
    plus_path = minus_path = ""
    for line in (diff_text or "").split("\n"):
        if line.startswith("+++ b/"):
            plus_path = line[len("+++ b/") :].strip().split("\t")[0].strip()
        elif line.startswith("--- a/"):
            minus_path = line[len("--- a/") :].strip().split("\t")[0].strip()
    # For delete (+++ /dev/null), plus_path is empty; use minus_path
    return plus_path or minus_path
