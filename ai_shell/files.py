import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from . import config
from .utils import dbg

from .file_utils_adapter import generate_diff, is_security_concern

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
        print("[Imported files cleared; agent may edit any file under root]", file=sys.stderr)
        return
    rels: Set[str] = set()
    for p in paths:
        if not p:
            continue
        rels.add(_norm_rel_path(p))
    INCLUDE_PATHS = rels if rels else None
    if INCLUDE_PATHS:
        print(
            f"[Imported files (edit allow list): {len(INCLUDE_PATHS)} file(s)]",
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
    """True if path is in include set (or under an include dir). When include not set or indexing disabled, returns True."""
    if not INCLUDE_PATHS or getattr(config, "DISABLE_INDEX", False):
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
    """True if the agent may edit this path.
    Allow list = files imported by the user (INCLUDE_PATHS). When the user has imported files,
    only those paths (or under them) are editable. When no files imported, any path under root is allowed.
    Blocklist (local-app, ai_shell, tools) always applies. When allow_new=True, new files are allowed."""
    rel = _norm_rel_path(rel_path or "")
    if not rel:
        return False
    # Never allow editing Moonlet's own source or tooling
    rel_posix = rel.replace("\\", "/")
    if any(rel_posix.startswith(p) for p in _EDIT_BLOCKLIST_PREFIXES):
        return False
    # Allow list = imported files: when user has imported files, only those (or under them) are editable
    if getattr(config, "DISABLE_INDEX", False):
        return True
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


