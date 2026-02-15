import difflib
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from . import config
from .utils import dbg

try:
    from file_utils import generate_diff, is_security_concern
except ImportError:
    # Fallback: real diff so green lines always show when file_utils unavailable
    def generate_diff(
        old_content: str,
        new_content: str,
        filepath: str,
        context_lines: int = 3,
    ) -> str:
        old_lines = (old_content or "").splitlines(keepends=True)
        new_lines = (new_content or "").splitlines(keepends=True)
        diff = difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=str(filepath),
            tofile=str(filepath),
            lineterm="",
            n=context_lines,
        )
        return "".join(diff)

    def is_security_concern(*args, **kwargs):
        return False


ROOT_PATH = Path(os.getenv("SC2_ROOT", os.getcwd())).resolve()
INCLUDE_PATHS: Optional[Set[str]] = None


def _norm_rel_path(p: str) -> str:
    p = (p or "").strip().replace("\\", "/")
    p = re.sub(r"^\./+", "", p)
    return Path(p).as_posix()


_ALLOWED_NAMES = {
    # Build systems
    "makefile", "gnumakefile", "cmakelists.txt", "justfile",
    "sconscript", "sconstruct", "meson.build",
    # Containers / infra
    "dockerfile", "containerfile", "vagrantfile",
    "docker-compose.yml", "docker-compose.yaml",
    # Ruby / bundler
    "gemfile", "rakefile", "guardfile",
    # Node / JS
    "gruntfile.js", "gulpfile.js",
    # Config / dotfiles
    ".gitignore", ".gitattributes", ".gitmodules",
    ".dockerignore", ".editorconfig", ".prettierrc",
    ".eslintrc", ".babelrc", ".npmrc", ".nvmrc",
    ".env", ".env.local", ".env.development", ".env.production",
    ".clang-format", ".clang-tidy",
    # CI / CD
    "procfile", "brewfile",
}


def is_allowed_file(path_like) -> bool:
    path_obj = Path(path_like)
    name = path_obj.name
    if name.startswith(".") and name.lower() not in _ALLOWED_NAMES:
        return False
    # Allow known extensionless filenames (Makefile, Dockerfile, etc.)
    if name.lower() in _ALLOWED_NAMES:
        return True
    ext = path_obj.suffix.lower().lstrip(".")
    return ext in config.ALLOWED_EXTS_SET


_BINARY_IMAGE_EXTS = {
    "png", "jpg", "jpeg", "gif", "webp", "bmp", "tiff", "tif", "heic", "heif", "ico",
}


def is_binary_file(path_like) -> bool:
    path_obj = Path(path_like)
    ext = path_obj.suffix.lower().lstrip(".")
    return ext in _BINARY_IMAGE_EXTS


def resolve_path(rel_path: str) -> Path:
    rel_path = _norm_rel_path(rel_path)
    target = (ROOT_PATH / rel_path).resolve()
    try:
        target.relative_to(ROOT_PATH)
    except Exception:
        raise ValueError(f"outside root: target={target} root={ROOT_PATH}")
    return target


def read_file_text(rel_path: str) -> str:
    rel_path = _norm_rel_path(rel_path)
    target = resolve_path(rel_path)
    if not is_allowed_file(target):
        raise PermissionError("file type not allowed")
    if INCLUDE_PATHS is not None and rel_path and not _path_in_include(rel_path):
        raise PermissionError("file not in selected set")
    if not target.exists():
        raise FileNotFoundError(rel_path)
    return target.read_text()


def _write_atomic(target: Path, content: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        delete=False,
        dir=str(target.parent),
        prefix=target.name + ".tmp.",
        encoding="utf-8",
        newline="\n",
    ) as tmp:
        tmp.write(content)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    tmp_path.replace(target)
    try:
        disk = target.read_text(encoding="utf-8")
    except Exception:
        disk = target.read_text()
    if disk != content:
        raise IOError(f"write verification failed for {target}")


def debug_paths(focus_file: str) -> Dict[str, object]:
    focus_file_rel = _norm_rel_path(focus_file)
    info = {
        "root": str(ROOT_PATH),
        "cwd": os.getcwd(),
        "focus_file_rel": focus_file_rel,
    }
    try:
        focus_abs = resolve_path(focus_file_rel)
        info["focus_file_abs"] = str(focus_abs)
        info["exists"] = focus_abs.exists()
        info["allowed_ext"] = is_allowed_file(focus_abs)
    except Exception as exc:
        info["focus_file_abs"] = None
        info["exists"] = False
        info["allowed_ext"] = False
        info["resolve_error"] = str(exc)
    if INCLUDE_PATHS is None:
        info["in_include"] = None
    else:
        info["in_include"] = focus_file_rel in INCLUDE_PATHS
    return info


def write_file_text(rel_path: str, content: str) -> None:
    rel_path = _norm_rel_path(rel_path)
    target = resolve_path(rel_path)
    if not is_allowed_file(target):
        raise PermissionError("file type not allowed")
    # Allow new file creation without include check (model freedom to add files)
    if INCLUDE_PATHS is not None and rel_path and target.exists() and not _path_in_include(rel_path):
        raise PermissionError("file not in selected set")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)


def delete_file(rel_path: str) -> None:
    """Delete a file. Must be under root and pass is_edit_allowed."""
    rel_path = _norm_rel_path(rel_path)
    target = resolve_path(rel_path)
    if not target.exists() or not target.is_file():
        raise FileNotFoundError(rel_path)
    if not is_edit_allowed(rel_path):
        raise PermissionError("file not in selected set")
    if is_security_concern(target):
        raise PermissionError("Security concern: cannot delete system paths")
    target.unlink()


def set_include(paths: List[str]) -> None:
    global INCLUDE_PATHS
    paths = paths if paths is not None else []
    if not paths:
        INCLUDE_PATHS = None
        print("[Include filter cleared]", file=sys.stderr)
        return
    rels: Set[str] = set()
    for p in paths:
        if not p:
            continue
        rels.add(_norm_rel_path(p))
    INCLUDE_PATHS = rels if rels else None
    if INCLUDE_PATHS:
        print(
            f"[Include filter set to {len(INCLUDE_PATHS)} file(s)]",
            file=sys.stderr,
        )
    # Rebuild file index so agent tools (grep, symbols, list_files) see updated set
    try:
        from .index import rebuild_index
        rebuild_index()
    except Exception:
        pass


def get_include() -> List[str]:
    if INCLUDE_PATHS:
        return sorted(str(p) for p in INCLUDE_PATHS)
    return []


def _path_in_include(rel_path: str) -> bool:
    """True if path is in include set (or under an include dir). When include not set, returns True."""
    if not INCLUDE_PATHS:
        return True
    rel = _norm_rel_path(rel_path or "").replace("\\", "/")
    if rel in INCLUDE_PATHS:
        return True
    for p in INCLUDE_PATHS:
        prefix = (p or "").rstrip("/").replace("\\", "/")
        if prefix and (rel == prefix or rel.startswith(prefix + "/")):
            return True
    return False


# Paths the agent must never edit (app source, tooling)
_EDIT_BLOCKLIST_PREFIXES = ("local-app/", "local-app\\", "ai_shell/", "tools/")


def is_edit_allowed(rel_path: str, *, allow_new: bool = False) -> bool:
    """True if the agent may edit this path. Blocks app source and enforces include filter.
    When allow_new=True, new files are allowed (skip include check) so the model can create files freely."""
    rel = _norm_rel_path(rel_path or "")
    if not rel:
        return False
    # Never allow editing Moonlet's own source or tooling
    rel_posix = rel.replace("\\", "/")
    if any(rel_posix.startswith(p) for p in _EDIT_BLOCKLIST_PREFIXES):
        return False
    # When include filter is set, only allow paths in the include set (unless allow_new)
    if INCLUDE_PATHS and not allow_new:
        if rel_posix in INCLUDE_PATHS:
            return True
        for p in INCLUDE_PATHS:
            prefix = (p or "").rstrip("/").replace("\\", "/")
            if prefix and (rel_posix == prefix or rel_posix.startswith(prefix + "/")):
                return True
        return False
    return True


def get_root() -> Path:
    return ROOT_PATH


def set_root(new_root: str) -> Path:
    global ROOT_PATH
    global INCLUDE_PATHS
    # Sanitize: strip file:// prefix, whitespace
    path_str = (new_root or "").strip().replace("\\", "/")
    if path_str.lower().startswith("file://"):
        path_str = path_str[7:]
    candidate = Path(path_str).expanduser().resolve()
    if not candidate.exists() or not candidate.is_dir():
        raise ValueError("root must be an existing directory")
    # Allow opening any existing directory, even if currently empty.
    ROOT_PATH = candidate
    INCLUDE_PATHS = None
    print(f"[Root set to: {ROOT_PATH}]", file=sys.stderr)
    # Rebuild file index so agent tools see the new root
    try:
        from .index import rebuild_index
        rebuild_index()
    except Exception:
        pass
    return ROOT_PATH


def list_repo_files() -> List[str]:
    files: List[str] = []
    dirs: Set[str] = set()
    max_files = config.MAX_LIST_FILES
    root_str = str(ROOT_PATH)
    if not ROOT_PATH.exists():
        print(f"[Warning: ROOT_PATH does not exist: {root_str}]", file=sys.stderr)
        return []
    root_resolved = ROOT_PATH.resolve()
    include_info = (
        f" (include filter: {len(INCLUDE_PATHS) if INCLUDE_PATHS else 'none'})"
    )
    print(f"[Listing files in {root_str}{include_info}]", file=sys.stderr)
    for dirpath, dirnames, filenames in os.walk(ROOT_PATH):
        # Skip ignored dirs; show .dSYM bundle roots but do not recurse into them.
        next_dirnames: List[str] = []
        for d in dirnames:
            if d in config.IGNORE_DIRS or d.startswith("."):
                continue
            d_full = Path(dirpath, d)
            try:
                d_rel = d_full.resolve().relative_to(root_resolved).as_posix() + "/"
            except ValueError:
                continue
            if INCLUDE_PATHS is not None:
                d_prefix = d_rel
                if not any(
                    p == d_rel.rstrip("/") or p.startswith(d_prefix) for p in INCLUDE_PATHS
                ):
                    continue
            dirs.add(d_rel)
            if d.endswith(".dSYM"):
                # Keep bundle visible in Explorer, but don't descend and flood tree.
                continue
            next_dirnames.append(d)
        dirnames[:] = next_dirnames
        for fname in filenames:
            if fname.startswith("."):
                continue
            full_path = Path(dirpath, fname)
            try:
                # Ensure it's relative to ROOT_PATH
                rel = full_path.resolve().relative_to(root_resolved)
            except ValueError:
                # Path is outside ROOT_PATH, skip
                continue
            if not is_allowed_file(rel):
                continue
            rel_posix = rel.as_posix()
            if INCLUDE_PATHS is not None and not _path_in_include(rel_posix):
                continue
            files.append(rel_posix)
            if len(files) >= max_files:
                entries = sorted(dirs) + sorted(files)
                print(f"[Found {len(entries)} entries (hit max files)]", file=sys.stderr)
                return entries
    entries = sorted(dirs) + sorted(files)
    print(f"[Found {len(entries)} entries total]", file=sys.stderr)
    return entries


def read_files_for_context(files: List[str], limit_budget: bool = True) -> Dict[str, str]:
    """Read files for context. Can limit budget to prevent crashes."""
    contents: Dict[str, str] = {}
    if limit_budget:
        # Very conservative: max 3KB total when limiting
        budget = min(config.MAX_CTX_BYTES, config.CTX_CHAR_BUDGET, 3000)
    else:
        budget = min(config.MAX_CTX_BYTES, config.CTX_CHAR_BUDGET)  # Full budget when not limiting

    for rel in files[:10]:  # Limit to first 10 files max
        path = ROOT_PATH / rel
        if not path.is_file():
            continue
        if not is_allowed_file(path):
            continue
        if INCLUDE_PATHS is not None and not _path_in_include(rel):
            continue
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            continue
        size = len(text)
        # Truncate large files to fit budget if limiting
        if limit_budget and size > budget:
            text = text[:budget] + "\n... [truncated]"
            size = budget
        if size > budget:
            continue
        contents[rel] = text
        budget -= size
        if budget <= 0:
            break
    return contents


def read_single_file_for_context(file_path: str) -> Dict[str, str]:
    """Read a single file for context with full content (no truncation)."""
    contents: Dict[str, str] = {}
    file_path = _norm_rel_path(file_path)
    path = ROOT_PATH / file_path
    if not path.is_file():
        return contents
    if not is_allowed_file(path):
        return contents
    if INCLUDE_PATHS is not None and not _path_in_include(file_path):
        return contents
    try:
        text = path.read_text()
        contents[file_path] = text
    except UnicodeDecodeError:
        pass
    return contents


def _is_new_or_empty_file(focus_file: str) -> bool:
    """True for missing files or files with little/no content."""
    if not focus_file:
        return False
    content = read_single_file_for_context(focus_file).get(focus_file, "")
    if not content:
        return True
    return len(content.strip()) < 40


def apply_blocks_with_report(
    blocks: List[Tuple[str, str]], show_diff: bool = True, dry_run: bool = False
) -> Tuple[List[str], List[Dict[str, str]], Dict[str, str], Dict[str, str], Dict[str, str]]:
    """Apply file blocks with optional diff preview.

    Returns (touched, skipped, per_file_diffs, per_file_staged, per_file_before) for UI staging.
    """
    if not blocks:
        dbg("apply_blocks: no blocks to apply")
        return [], [], {}, {}, {}
    dbg(f"apply_blocks: count={len(blocks)}")
    if len(blocks) > 1:
        per_path: Dict[str, Dict[str, Tuple[str, str]]] = {}
        order: List[str] = []
        for raw_path, content in blocks:
            norm_path = _norm_rel_path(raw_path)
            if norm_path not in per_path:
                per_path[norm_path] = {"last": (norm_path, content)}
                order.append(norm_path)
            else:
                per_path[norm_path]["last"] = (norm_path, content)
            if content.strip():
                per_path[norm_path]["last_non_empty"] = (norm_path, content)
            # Track longest content to avoid partial overwrites
            current_longest = per_path[norm_path].get("longest")
            if current_longest is None or len(content) > len(current_longest[1]):
                per_path[norm_path]["longest"] = (norm_path, content)
        normalized_blocks: List[Tuple[str, str]] = []
        for norm_path in order:
            entry = per_path[norm_path]
            # Prefer the longest non-empty content; fall back to last non-empty/last.
            chosen = entry.get("longest")
            if chosen and not chosen[1].strip():
                chosen = None
            if not chosen:
                chosen = entry.get("last_non_empty") or entry["last"]
            normalized_blocks.append(chosen)
        blocks = normalized_blocks
    touched: List[str] = []
    skipped: List[Dict[str, str]] = []
    per_file_diffs: Dict[str, str] = {}
    per_file_staged: Dict[str, str] = {}
    per_file_before: Dict[str, str] = {}
    for rel_path, content in blocks:
        try:
            rel_path = _norm_rel_path(rel_path)
            target = resolve_path(rel_path)
            allow_new = not (target.exists() and target.is_file())
            if not is_edit_allowed(rel_path, allow_new=allow_new):
                skipped.append({"path": rel_path, "reason": "edit_not_allowed"})
                print(f"[skipped] Edit not allowed (app source or outside include): {rel_path}")
                continue
            if is_security_concern(target):
                skipped.append({"path": rel_path, "reason": "security_concern"})
                print(f"[skipped] Security concern: {rel_path}")
                continue
        except ValueError as exc:
            skipped.append({"path": rel_path, "reason": "outside_root"})
            print(f"[skipped] Unsafe path: {rel_path} ({exc})")
            continue
        # Reject obvious language mismatches (e.g. Java/C++ in .py)
        if rel_path.endswith(".py"):
            foreign_markers = (
                "#include",
                "using namespace",
                "public class",
                "static void main",
                "System.out.println",
                "std::",
            )
            if any(marker in content for marker in foreign_markers):
                dbg(f"apply_blocks: rejected non-Python content for {rel_path}")
                continue

        # Generate diff if file exists
        old_content = None
        if target.exists():
            try:
                old_content = target.read_text()
            except Exception:
                old_content = None
        if (
            target.exists()
            and show_diff
            and old_content is not None
            and old_content != content
        ):
            try:
                diff = generate_diff(old_content, content, str(target))
                print(f"\n[Diff for {rel_path}]:")
                print(diff)
                per_file_diffs[rel_path] = diff
            except Exception as exc:
                print(f"[warning] Could not generate diff: {exc}")
        elif not target.exists() and content.strip():
            try:
                diff = generate_diff("", content, str(target))
                per_file_diffs[rel_path] = diff
            except Exception:
                pass
        if old_content is not None and old_content == content:
            dbg(f"apply_blocks: no-op write (identical) for {rel_path}")
            skipped.append({"path": rel_path, "reason": "identical_content"})
            continue
        dbg(f"apply_blocks: writing {rel_path} bytes={len(content)}")
        if not dry_run:
            _write_atomic(target, content)
            print(f"[wrote] {rel_path}")
        touched.append(rel_path)
        per_file_staged[rel_path] = content
        per_file_before[rel_path] = old_content if old_content is not None else ""
    return touched, skipped, per_file_diffs, per_file_staged, per_file_before


def apply_blocks(blocks: List[Tuple[str, str]], show_diff: bool = True) -> List[str]:
    """Apply file blocks with optional diff preview.

    Enhanced with diff generation from Continue codebase.
    """
    if not blocks:
        dbg("apply_blocks: no blocks to apply")
        return []
    dbg(f"apply_blocks: count={len(blocks)}")
    if len(blocks) > 1:
        per_path: Dict[str, Dict[str, Tuple[str, str]]] = {}
        order: List[str] = []
        for raw_path, content in blocks:
            norm_path = _norm_rel_path(raw_path)
            if norm_path not in per_path:
                per_path[norm_path] = {"last": (norm_path, content)}
                order.append(norm_path)
            else:
                per_path[norm_path]["last"] = (norm_path, content)
            if content.strip():
                per_path[norm_path]["last_non_empty"] = (norm_path, content)
            current_longest = per_path[norm_path].get("longest")
            if current_longest is None or len(content) > len(current_longest[1]):
                per_path[norm_path]["longest"] = (norm_path, content)
        normalized_blocks: List[Tuple[str, str]] = []
        for norm_path in order:
            entry = per_path[norm_path]
            chosen = entry.get("longest")
            if chosen and not chosen[1].strip():
                chosen = None
            if not chosen:
                chosen = entry.get("last_non_empty") or entry["last"]
            normalized_blocks.append(chosen)
        blocks = normalized_blocks
    touched: List[str] = []
    for rel_path, content in blocks:
        try:
            rel_path = _norm_rel_path(rel_path)
            target = resolve_path(rel_path)
            if is_security_concern(target):
                print(f"[skipped] Security concern: {rel_path}")
                continue
        except ValueError as exc:
            print(f"[skipped] Unsafe path: {rel_path} ({exc})")
            continue
        # Reject obvious language mismatches (e.g. Java/C++ in .py)
        if rel_path.endswith(".py"):
            foreign_markers = (
                "#include",
                "using namespace",
                "public class",
                "static void main",
                "System.out.println",
                "std::",
            )
            if any(marker in content for marker in foreign_markers):
                dbg(f"apply_blocks: rejected non-Python content for {rel_path}")
                continue

        # Generate diff if file exists
        old_content = None
        if target.exists():
            try:
                old_content = target.read_text()
            except Exception:
                old_content = None
        if (
            target.exists()
            and show_diff
            and old_content is not None
            and old_content != content
        ):
            try:
                diff = generate_diff(old_content, content, str(target))
                print(f"\n[Diff for {rel_path}]:")
                print(diff)
            except Exception as exc:
                print(f"[warning] Could not generate diff: {exc}")
        if old_content is not None and old_content == content:
            dbg(f"apply_blocks: no-op write (identical) for {rel_path}")
            continue

        # Safety: block catastrophic writes (new content far smaller than old)
        if old_content and len(old_content) > 100 and len(content) < len(old_content) * 0.2:
            dbg(f"apply_blocks: BLOCKED catastrophic write for {rel_path} "
                f"({len(old_content)} -> {len(content)} bytes)")
            print(f"[blocked] {rel_path}: refusing to shrink from "
                  f"{len(old_content)} to {len(content)} bytes")
            continue

        dbg(f"apply_blocks: writing {rel_path} bytes={len(content)}")
        _write_atomic(target, content)
        print(f"[wrote] {rel_path}")
        touched.append(rel_path)
    return touched


def apply_line_edits(rel_path: str, edits: List[Dict[str, object]]) -> None:
    """Apply line-based edits (1-based, inclusive) from bottom to top."""
    target = resolve_path(rel_path)
    if is_security_concern(target):
        raise PermissionError("Security concern: cannot edit system paths")
    if not is_allowed_file(target):
        raise PermissionError("file type not allowed")
    if not target.exists():
        raise FileNotFoundError(rel_path)
    if not edits or not isinstance(edits, list):
        raise ValueError("edits must be a non-empty array")

    lines = target.read_text().splitlines()
    normalized: List[Tuple[int, int, str, bool]] = []  # (start, end, repl, is_insert)
    for edit in edits:
        try:
            start = int(edit.get("start_line"))
            end = int(edit.get("end_line"))
        except Exception:
            raise ValueError("start_line and end_line must be integers")
        if start < 1:
            raise ValueError("start_line must be >= 1")
        # Accept multiple key names for replacement (model variants differ)
        _REPL_KEYS = ("replacement", "text", "new_text", "content", "code")
        replacement = ""
        for _k in _REPL_KEYS:
            _v = edit.get(_k)
            if _v:
                replacement = str(_v)
                break
        # end_line < start_line means INSERT before start_line (no deletion)
        is_insert = end < start
        normalized.append((start, end, str(replacement), is_insert))

    normalized.sort(key=lambda x: x[0], reverse=True)
    for start, end, replacement, is_insert in normalized:
        repl_lines = replacement.splitlines()
        if is_insert:
            # Insert before start_line (0-indexed: start-1)
            insert_idx = min(start - 1, len(lines))
            lines[insert_idx:insert_idx] = repl_lines
        else:
            if start > len(lines) + 1:
                raise ValueError("start_line out of range")
            if end > len(lines):
                raise ValueError("end_line out of range")
            lines[start - 1 : end] = repl_lines

    _write_atomic(target, "\n".join(lines) + ("\n" if lines else ""))
# ---------- Unified diff applier ----------

def _line_match(a: str, b: str) -> bool:
    """Compare two lines with whitespace flexibility.
    Exact match first, then stripped match as fallback."""
    if a == b:
        return True
    sa = a.strip()
    sb = b.strip()
    if sa == sb:
        return True
    # Be tolerant to model-produced inline comment drift where a context/removal
    # line incorrectly appends or omits trailing inline comments.
    def _strip_inline_comment(s: str) -> str:
        s2 = re.sub(r"\s*//.*$", "", s)
        s2 = re.sub(r"\s*/\*.*?\*/\s*$", "", s2)
        return s2.strip()
    ca = _strip_inline_comment(sa)
    cb = _strip_inline_comment(sb)
    if ca and cb and ca == cb:
        return True
    # Model diffs sometimes truncate function signature context lines to
    # prefixes like "void foo("; allow prefix matching in that narrow case.
    if sa and sb:
        short, long_ = (sa, sb) if len(sa) <= len(sb) else (sb, sa)
        if len(short) >= 8 and short in long_ and ("(" in short):
            if short.endswith("(") or (")" not in short):
                if long_.startswith(short):
                    return True
    return False


def _find_hunk_location(
    file_lines: List[str],
    expected: List[Tuple[str, str]],
    hint_start: int,
    max_fuzz: int = 2,
) -> int:
    """Find the best position in file_lines to apply a hunk.

    expected: list of (prefix, content) where prefix is " " or "-".
              These lines must exist in the file at the match location.
    hint_start: 0-based line index hint from @@ header.
    max_fuzz: allow skipping up to this many context lines at edges.

    Uses whitespace-flexible matching: models often get indentation
    slightly wrong in diff context lines.

    Returns 0-based index of the match start, or -1 if no match.
    Raises ValueError if multiple equally good matches (ambiguous).
    """
    if not expected:
        # Pure insertion hunk (only "+" lines) — use hint
        return max(0, min(hint_start, len(file_lines)))

    # Build the expected content lines (what should be in the file)
    expect_content = [content for _, content in expected]
    expect_len = len(expect_content)

    # Search window: try near hint first (fast), then fall back to full file.
    search_range = 50

    def _scan(lo: int, hi: int) -> List[Tuple[int, int, int]]:
        # (exact_matches, fuzzy_matches, position)
        out: List[Tuple[int, int, int]] = []
        for pos in range(lo, hi):
            if pos + expect_len > len(file_lines):
                continue
            exact = 0
            fuzzy = 0
            for j in range(expect_len):
                if file_lines[pos + j] == expect_content[j]:
                    exact += 1
                    fuzzy += 1
                elif _line_match(file_lines[pos + j], expect_content[j]):
                    fuzzy += 1
            # Accept if all lines match (exact or fuzzy)
            if fuzzy == expect_len:
                out.append((exact, fuzzy, pos))
            elif fuzzy >= expect_len - max_fuzz and fuzzy > 0:
                # Fuzz: allow up to max_fuzz mismatches at edges
                out.append((exact, fuzzy, pos))
        return out

    def _scan_blank_gap(lo: int, hi: int, max_blank_skip: int = 6) -> List[Tuple[int, int, int]]:
        """Fallback scan that allows extra blank lines in the file."""
        out: List[Tuple[int, int, int]] = []
        for pos in range(lo, hi):
            i = 0
            j = pos
            exact = 0
            fuzzy = 0
            blank_skips = 0
            while i < expect_len and j < len(file_lines):
                if file_lines[j].strip() == "":
                    blank_skips += 1
                    if blank_skips > max_blank_skip:
                        break
                    j += 1
                    continue
                if file_lines[j] == expect_content[i]:
                    exact += 1
                    fuzzy += 1
                    i += 1
                    j += 1
                    continue
                if _line_match(file_lines[j], expect_content[i]):
                    fuzzy += 1
                    i += 1
                    j += 1
                    continue
                break
            if i == expect_len:
                out.append((exact, fuzzy, pos))
        return out

    lo = max(0, hint_start - search_range)
    hi = min(len(file_lines), hint_start + search_range + 1)
    candidates = _scan(lo, hi)

    # If local search misses, do a full-file search. Model-provided @@ line
    # numbers are often approximate, especially in long prompts.
    if not candidates:
        candidates = _scan(0, len(file_lines))

    if not candidates:
        # Final fallback: allow blank-line gaps in file
        candidates = _scan_blank_gap(lo, hi)
        if not candidates:
            candidates = _scan_blank_gap(0, len(file_lines))
        if not candidates:
            return -1

    # Sort: prefer most exact matches, then most fuzzy, then closest to hint
    candidates.sort(key=lambda c: (-c[0], -c[1], abs(c[2] - hint_start)))

    best = candidates[0]
    best_matches = [c for c in candidates if c[0] == best[0] and c[1] == best[1]]

    if len(best_matches) > 1 and best[1] == expect_len:
        # Use hint_start (anchor_line) to break ties — pick closest to hint
        best_matches.sort(key=lambda c: abs(c[2] - hint_start))
        dbg(f"_find_hunk_location: {len(best_matches)} tied matches, "
            f"picking closest to hint {hint_start}: line {best_matches[0][2] + 1}")
        return best_matches[0][2]
    return candidates[0][2]


def _rg_hint_for_expected(
    focus_file: str,
    expected: List[Tuple[str, str]],
    root_override: Optional[Path] = None,
) -> Optional[int]:
    """Use ripgrep to find a likely 0-based start line for a hunk.

    Strategy:
    - pick first non-empty expected line as the anchor
    - run `rg -n -F` against the focus file
    - convert the matched line into a hunk start hint by subtracting
      the anchor's index inside expected
    """
    candidates: List[Tuple[int, str]] = []
    for idx, (_, content) in enumerate(expected):
        stripped = content.strip()
        if not stripped:
            continue
        if stripped in ("{", "}"):
            continue
        if stripped.startswith("//") or stripped.startswith("/*") or stripped.startswith("*"):
            continue
        if stripped.startswith("#"):
            continue
        if len(stripped) < 4:
            continue
        candidates.append((idx, stripped))
        if len(candidates) >= 5:
            break
    if not candidates:
        return None

    try:
        for anchor_idx, anchor_text in candidates:
            search_root = Path(root_override).resolve() if root_override else ROOT_PATH
            res = subprocess.run(
                ["rg", "-n", "-F", anchor_text, focus_file],
                cwd=str(search_root),
                capture_output=True,
                text=True,
            )
            if res.returncode != 0 or not res.stdout.strip():
                continue
            first = res.stdout.splitlines()[0]
            m = re.match(r"^(\d+):", first)
            if not m:
                continue
            line_1 = int(m.group(1))
            hint = max(0, (line_1 - 1) - anchor_idx)
            dbg(f"apply_diff: rg hint for {focus_file}: line {line_1} -> start {hint + 1}")
            return hint
    except Exception:
        return None
    return None


def apply_unified_diff(
    focus_file: str,
    hunks: list,
    anchor_line: int = 0,
    root_override: Optional[Path] = None,
    dry_run: bool = False,
) -> Optional[str]:
    """Apply parsed unified diff hunks to a file using context matching.

    hunks: list of DiffHunk objects (from parse_unified_diff).
    anchor_line: 1-based hint from InsertionAnchor (preferred over @@ numbers).

    Uses context-line matching to find the right location — does NOT
    trust @@ line numbers. Prefers anchor_line if provided.
    """
    rel_focus = _norm_rel_path(focus_file)
    if root_override is not None:
        base_root = Path(root_override).resolve()
        target = (base_root / rel_focus).resolve()
        try:
            target.relative_to(base_root)
        except Exception as exc:
            raise ValueError(f"outside root override: {target}") from exc
        if not target.exists():
            raise FileNotFoundError(focus_file)
    else:
        target = resolve_path(rel_focus)
        if is_security_concern(target):
            raise PermissionError("Security concern: cannot edit system paths")
        if not is_allowed_file(target):
            raise PermissionError("file type not allowed")
        is_new_file = not target.exists()
        if is_new_file:
            # Allow only when diff is "new file" format (--- /dev/null, all hunks insertion-only)
            all_insert = all(
                getattr(h, "old_start", 1) == 0 and getattr(h, "old_count", 1) == 0
                for h in hunks
            )
            if not all_insert:
                raise FileNotFoundError(focus_file)
    if not hunks:
        raise ValueError("No hunks to apply")

    # Safety gate removed: allow multi-hunk insert-only patches.

    file_lines = target.read_text().splitlines() if target.exists() else []

    ordered_hunks = sorted(
        list(hunks),
        key=lambda h: (
            int(getattr(h, "old_start", 0) or 0),
            int(getattr(h, "new_start", 0) or 0),
        ),
    )

    def _context_dump(lines: List[str], center_0: int, radius: int = 6) -> str:
        if not lines:
            return ""
        c = max(0, min(center_0, len(lines) - 1))
        start = max(0, c - radius)
        end = min(len(lines), c + radius + 1)
        out = []
        for idx in range(start, end):
            out.append(f"{idx + 1}| {lines[idx]}")
        return "\n".join(out)

    for hunk in ordered_hunks:
        # Build expected lines (context + removals = what's in the file)
        expected: List[Tuple[str, str]] = []
        for prefix, content in hunk.lines:
            if prefix in (" ", "-"):
                expected.append((prefix, content))

        # Build the replacement exactly as diff intent:
        # context lines + additions, with no automatic spacing edits.
        replacement: List[str] = []
        for prefix, content in hunk.lines:
            if prefix == " ":
                replacement.append(content)
            elif prefix == "+":
                replacement.append(content)
            # "-" lines are removed (not in replacement)

        # Hint: prefer anchor_line from Moonlet, fall back to @@ header
        if anchor_line > 0:
            hint = anchor_line - 1  # 0-based
        elif hunk.old_start > 0:
            hint = hunk.old_start - 1
        else:
            hint = len(file_lines) // 2  # search from middle

        # For insertion-only hunks with no context, use anchor directly
        if not expected and anchor_line > 0:
            insert_pos = min(anchor_line - 1, len(file_lines))
            file_lines[insert_pos:insert_pos] = replacement
            dbg(f"apply_diff: pure insertion at line {insert_pos + 1}, "
                f"inserted {len(replacement)} lines")
            result = "\n".join(file_lines) + ("\n" if file_lines else "")
            if dry_run:
                return result
            _write_atomic(target, result)
            return None

        # Find where this hunk matches in the file.
        # Prefer rg-derived hint when enabled (and no explicit anchor_line).
        pos = -1
        if config.USE_RG_HINT_DEFAULT and anchor_line <= 0 and expected:
            rg_hint = _rg_hint_for_expected(
                rel_focus, expected, root_override=root_override
            )
            if rg_hint is not None:
                dbg(
                    f"apply_diff: rg default hint for {rel_focus}: "
                    f"line {rg_hint + 1} -> start {rg_hint + 1}"
                )
                pos = _find_hunk_location(file_lines, expected, rg_hint, max_fuzz=6)
                if pos == -1:
                    pos = _find_hunk_location(file_lines, expected, rg_hint, max_fuzz=10)

        if pos == -1:
            pos = _find_hunk_location(file_lines, expected, hint)
        if pos == -1:
            pos = _find_hunk_location(file_lines, expected, hint, max_fuzz=6)
        if pos == -1:
            pos = _find_hunk_location(file_lines, expected, hint, max_fuzz=10)
        if pos == -1 and (not config.USE_RG_HINT_DEFAULT or anchor_line > 0):
            rg_hint = _rg_hint_for_expected(
                rel_focus, expected, root_override=root_override
            )
            if rg_hint is not None:
                pos = _find_hunk_location(file_lines, expected, rg_hint, max_fuzz=6)
                if pos == -1:
                    pos = _find_hunk_location(file_lines, expected, rg_hint, max_fuzz=10)
        if pos == -1:
            # Fallback: use @@ line numbers when context match fails (model may have
            # produced diff for slightly different file structure).
            old_start = int(getattr(hunk, "old_start", 0) or 0)
            n_expected = len(expected)
            if old_start > 0:
                pos = old_start - 1  # 0-based
                dbg(
                    f"apply_diff: context match failed, using @@ line-number fallback "
                    f"at line {pos + 1} for {rel_focus}"
                )
                file_lines[pos : pos + n_expected] = replacement
                continue  # next hunk
            expect_preview = [c for _, c in expected[:3]]
            hunk_lines = []
            for p, c in list(getattr(hunk, "lines", []) or [])[:80]:
                hunk_lines.append(f"{p}{c}")
            hunk_text = "\n".join(hunk_lines)
            ctx = _context_dump(file_lines, max(0, hint))
            raise ValueError(
                f"Hunk could not be located in {rel_focus}. "
                f"Expected lines: {expect_preview!r}\n"
                f"HUNK:\n{hunk_text}\n"
                f"CURRENT_CONTEXT:\n{ctx}"
            )

        # Verify removal lines match (whitespace-flexible)
        for j, (prefix, content) in enumerate(expected):
            if prefix == "-":
                if pos + j >= len(file_lines):
                    raise ValueError(
                        f"Removal line at line {pos + j + 1}: unexpected EOF"
                    )
                if not _line_match(file_lines[pos + j], content):
                    got = repr(file_lines[pos + j])
                    ctx = _context_dump(file_lines, pos + j)
                    raise ValueError(
                        f"Removal line mismatch at line {pos + j + 1}: "
                        f"expected {content!r}, got {got}\n"
                        f"CURRENT_CONTEXT:\n{ctx}"
                    )

        # Apply: replace the expected block with the replacement
        n_expected = len(expected)
        file_lines[pos:pos + n_expected] = replacement

        dbg(f"apply_diff: hunk applied at line {pos + 1}, "
            f"removed {n_expected} lines, inserted {len(replacement)} lines")

    result = "\n".join(file_lines) + ("\n" if file_lines else "")
    if dry_run:
        return result
    _write_atomic(target, result)
    return None


def _test_rg_hint_default_behavior() -> Tuple[bool, str]:
    """Internal self-check for rg-first hunk placement (not executed by default)."""
    import tempfile
    from .parsing import DiffHunk
    from .files import set_root

    tmpdir = tempfile.TemporaryDirectory()
    try:
        root = set_root(tmpdir.name)
        path = "t.c"
        content = "\n".join([
            "#include <stdio.h>",
            "",
            "int main() {",
            "  int x = 1;",
            "  printf(\"x=%d\", x);",
            "  return 0;",
            "}",
            "",
        ])
        (root / path).write_text(content)
        hunk = DiffHunk(
            old_start=1,
            old_count=3,
            new_start=1,
            new_count=3,
            lines=[
                (" ", "int main() {"),
                (" ", "  int x = 1;"),
                ("-", "  printf(\"x=%d\", x);"),
                ("+", "  printf(\"x=%d!\", x);"),
                (" ", "  return 0;"),
            ],
        )
        apply_unified_diff(path, [hunk])
        out = (root / path).read_text()
        if "x=%d!" not in out:
            return False, "rg-first apply failed"
        return True, "ok"
    finally:
        tmpdir.cleanup()
