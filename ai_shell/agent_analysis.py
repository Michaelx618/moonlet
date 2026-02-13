"""Universal evidence-first prepass analysis."""

import json
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

from . import config
from .files import _norm_rel_path, get_root, read_single_file_for_context

_BINARY_EXTS = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".tif",
    ".heic", ".heif", ".ico", ".pdf", ".zip", ".tar", ".gz", ".7z",
}

_ALLOWED_DOTTED_EXTS = {
    "." + str(ext).lower().lstrip(".") for ext in (config.ALLOWED_EXTS or set())
}


def _is_analysis_text_ext(path: str) -> bool:
    ext = Path(path).suffix.lower()
    if not ext:
        return False
    if ext in _BINARY_EXTS:
        return False
    if _ALLOWED_DOTTED_EXTS:
        return ext in _ALLOWED_DOTTED_EXTS
    return True


def _snippet(path: str, line_no: int, radius: int = 12) -> str:
    content = read_single_file_for_context(path).get(path, "")
    if not content:
        return ""
    lines = content.splitlines()
    if not lines:
        return ""
    start = max(1, line_no - radius)
    end = min(len(lines), line_no + radius)
    out = []
    for i in range(start, end + 1):
        out.append(f"{i}| {lines[i - 1]}")
    return "\n".join(out)


def _parse_local_deps(focus_file: str, focus_content: str) -> List[str]:
    """Read direct local deps only using generic path/module references."""
    root = get_root()
    focus_abs = (root / _norm_rel_path(focus_file)).resolve()
    base_dir = focus_abs.parent

    deps: List[str] = []
    seen: Set[str] = set()

    def _add_if_exists(abs_path: Path) -> None:
        try:
            rel = abs_path.resolve().relative_to(root)
        except Exception:
            return
        p = _norm_rel_path(str(rel))
        if p in seen:
            return
        if abs_path.exists() and abs_path.is_file() and _is_analysis_text_ext(p):
            seen.add(p)
            deps.append(p)

    # 1) Generic quoted path references: "./x", "../x", "x.ext"
    path_refs: Set[str] = set()
    for ref in re.findall(r'["\']([^"\']+)["\']', focus_content):
        r = ref.strip()
        if not r or "://" in r:
            continue
        if r.startswith(("./", "../")) or "/" in r or re.search(r"\.[A-Za-z0-9_]+$", r):
            path_refs.add(r)

    # 2) Generic include-like references with quoted filenames.
    for inc in re.findall(r'^\s*#include\s+"([^"]+)"', focus_content, re.MULTILINE):
        path_refs.add(inc.strip())

    ext_candidates = [
        "", ".py", ".js", ".ts", ".tsx", ".jsx", ".rs", ".go", ".java",
        ".c", ".h", ".cpp", ".hpp", ".json", ".yaml", ".yml",
    ]
    for ref in path_refs:
        has_ext = bool(Path(ref).suffix)
        candidates = [ref] if has_ext else [f"{ref}{suf}" for suf in ext_candidates]
        for c in candidates:
            _add_if_exists(base_dir / c)
            _add_if_exists(root / c)
            _add_if_exists(base_dir / c / "index.py")
            _add_if_exists(base_dir / c / "index.ts")
            _add_if_exists(base_dir / c / "__init__.py")

    # 3) Generic module-style references from import/use-like lines.
    modules: Set[str] = set()
    modules.update(
        re.findall(
            r'^\s*(?:import|from|use|mod|require)\s+([A-Za-z_][A-Za-z0-9_\.]*)',
            focus_content,
            re.MULTILINE,
        )
    )
    for mod in modules:
        rel = mod.replace(".", "/")
        _add_if_exists(base_dir / f"{rel}.py")
        _add_if_exists(base_dir / f"{rel}.ts")
        _add_if_exists(base_dir / f"{rel}.js")
        _add_if_exists(base_dir / f"{rel}.rs")
        _add_if_exists(base_dir / rel / "__init__.py")
        _add_if_exists(base_dir / rel / "index.ts")
        _add_if_exists(base_dir / rel / "index.js")
        _add_if_exists(root / f"{rel}.py")
        _add_if_exists(root / f"{rel}.ts")
        _add_if_exists(root / f"{rel}.js")
        _add_if_exists(root / "src" / f"{rel}.rs")

    return deps[:8]


def _detect_build_system(root: Path) -> str:
    markers = [
        ("make", ["Makefile", "makefile", "GNUmakefile"]),
        ("cargo", ["Cargo.toml"]),
        ("npm", ["package.json"]),
        ("python", ["pyproject.toml", "setup.py", "setup.cfg", "requirements.txt"]),
    ]
    for system, files in markers:
        for name in files:
            if (root / name).exists():
                return system
    return "unknown"


def _suggested_validate_cmd(
    focus_file: str,
    likely_build_system: str,
) -> str:
    ext = Path(focus_file).suffix.lower()
    if ext == ".py":
        return "python -m py_compile {files}"
    if ext in {".c", ".h"}:
        return "cc -fsyntax-only -std=c11 -Wall {files}"
    if ext in {".cc", ".cpp", ".cxx", ".hh", ".hpp", ".hxx"}:
        return "c++ -fsyntax-only -std=c++17 -Wall {files}"
    if likely_build_system == "cargo":
        return "cargo check -q"
    if likely_build_system == "npm":
        return "npm run -s lint"
    if likely_build_system == "make":
        return "make -s"
    return ""


def _extract_symbols_allowed(touch_points: List[Dict[str, str]]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for tp in touch_points or []:
        name = str(tp.get("symbol") or "").strip()
        if not name or name == "<file>":
            continue
        if name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out[:8]


def _collect_dependency_context(
    dep_files: List[str],
    symbols_allowed: List[str],
) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    root = get_root()
    for rel in dep_files[:6]:
        path = root / _norm_rel_path(rel)
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if not text.strip():
            continue
        chosen_symbol = ""
        snippet = ""
        for sym in symbols_allowed:
            m = re.search(rf"(?m)^.*\b{re.escape(sym)}\b.*$", text)
            if not m:
                continue
            chosen_symbol = sym
            line = m.group(0).strip()
            snippet = line[:220]
            break
        if not snippet:
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            if not lines:
                continue
            snippet = lines[0][:220]
            chosen_symbol = chosen_symbol or Path(rel).stem
        out.append(
            {
                "file": _norm_rel_path(rel),
                "symbol": chosen_symbol[:80],
                "snippet": snippet[:220],
            }
        )
        if len(out) >= 2:
            break
    return out




def _extract_touch_points_regex(focus_file: str, content: str) -> List[Dict[str, str]]:
    points: List[Dict[str, str]] = []
    lines = content.splitlines()
    n = len(lines)

    pats = [
        re.compile(r"^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\("),
        re.compile(r"^\s*(?:async\s+)?function\s+([A-Za-z_][A-Za-z0-9_]*)\s*\("),
        re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>"),
        re.compile(r"^\s*(?:fn|func)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\("),
        re.compile(r"^\s*(?:[A-Za-z_][A-Za-z0-9_\s\*:&<>\[\]]+\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*\([^;]*\)\s*\{\s*$"),
    ]

    for i, line in enumerate(lines, start=1):
        s = line.strip()
        if not s or s.startswith("//") or s.startswith("/*") or s.startswith("*"):
            continue
        if s.endswith(";"):
            continue
        name = ""
        for p in pats:
            m = p.match(line)
            if m:
                name = m.group(1)
                break
        if not name:
            continue

        depth = 0
        end = i
        saw_open = False
        for j in range(i, n + 1):
            depth += lines[j - 1].count("{")
            depth -= lines[j - 1].count("}")
            if lines[j - 1].count("{"):
                saw_open = True
            end = j
            if saw_open and depth <= 0 and j > i:
                break

        points.append({"file": focus_file, "symbol": name, "lines": f"L{i}-L{end}"})
        if len(points) >= 4:
            break

    if not points:
        points.append({"file": focus_file, "symbol": "<file>", "lines": "L1-L1"})
    return points


def _extract_touch_points(focus_file: str, content: str) -> List[Dict[str, str]]:
    try:
        from .tools import extract_symbols_treesitter
        symbols = extract_symbols_treesitter(focus_file, content=content)
    except Exception:
        symbols = []

    points: List[Dict[str, str]] = []
    if symbols:
        fn_syms = [s for s in symbols if getattr(s, "kind", "") in ("function", "method")]
        chosen = fn_syms if fn_syms else symbols
        for sym in chosen[:4]:
            points.append(
                {
                    "file": focus_file,
                    "symbol": str(getattr(sym, "name", "")) or "<symbol>",
                    "lines": (
                        f"L{int(getattr(sym, 'line', 1))}-"
                        f"L{int(getattr(sym, 'end_line', getattr(sym, 'line', 1)))}"
                    ),
                }
            )
    if not points:
        points = _extract_touch_points_regex(focus_file, content)
    return points[:4]


def _compact_summary(user_text: str, focus_file: str = "") -> str:
    text = (user_text or "").strip()
    if not text:
        return ""
    focus = _norm_rel_path(focus_file).lower() if focus_file else ""
    stem = Path(focus_file).stem.lower() if focus_file else ""
    action_words = (
        "implement", "fix", "modify", "update", "replace", "refactor",
        "complete", "fill", "add", "remove", "preserve", "todo",
    )
    constraint_words = ("must", "should", "do not", "keep", "preserve", "only", "without")
    narrative_words = (
        "assignment", "lab", "week", "grading", "points", "submit", "starter code",
        "download", "introduction", "overview", "for the sake of", "for simplicity",
        "what to submit", "requirements",
    )
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    chosen: List[str] = []
    for ln in lines:
        low = ln.lower()
        has_action = any(w in low for w in action_words)
        has_constraint = any(w in low for w in constraint_words)
        has_focus = (focus and focus in low) or (stem and stem in low)
        has_narrative = any(w in low for w in narrative_words)
        if has_narrative and not (has_action or has_constraint):
            continue
        if has_action or has_constraint or has_focus:
            chosen.append(ln)
        if len(chosen) >= 3:
            break
    if not chosen:
        chosen = lines[:2] if lines else [text]
    s = re.sub(r"\s+", " ", " ".join(chosen))
    return (s[:197] + "...") if len(s) > 200 else s


def _build_non_negotiables(has_todos: bool) -> List[str]:
    rules = [
        "Keep edits scoped to the request.",
        "Preserve unrelated code.",
        "Do not invent new files unless requested.",
    ]
    if has_todos:
        rules[1] = "Replace placeholders in touched code with working logic."
    return rules[:3]


def _shrink_for_budget(packet: Dict[str, object], max_chars: int) -> Dict[str, object]:
    """Shrink packet while preserving semantic priority."""
    try:
        encoded = json.dumps(packet, separators=(",", ":"), ensure_ascii=True)
    except Exception:
        return packet
    if len(encoded) <= max_chars:
        return packet

    shrunk = dict(packet)
    # 1) Trim supporting context first.
    support = shrunk.get("supporting_context")
    if isinstance(support, list) and support:
        compact_support = []
        for item in support[:1]:
            if not isinstance(item, dict):
                continue
            snippet = str(item.get("snippet") or "")
            compact_support.append(
                {
                    "file": str(item.get("file") or ""),
                    "why": str(item.get("why") or "")[:60],
                    "snippet": snippet[:160],
                }
            )
        shrunk["supporting_context"] = compact_support
    encoded = json.dumps(shrunk, separators=(",", ":"), ensure_ascii=True)
    if len(encoded) <= max_chars:
        return shrunk

    # 2) Trim deps sources.
    deps_sources = shrunk.get("deps_sources")
    if isinstance(deps_sources, list):
        shrunk["deps_sources"] = deps_sources[:3]
    encoded = json.dumps(shrunk, separators=(",", ":"), ensure_ascii=True)
    if len(encoded) <= max_chars:
        return shrunk

    # 4) Trim touch points.
    points = shrunk.get("touch_points")
    if isinstance(points, list):
        shrunk["touch_points"] = points[:2]
    encoded = json.dumps(shrunk, separators=(",", ":"), ensure_ascii=True)
    if len(encoded) <= max_chars:
        return shrunk

    # 4) Last resort: drop support and keep core policy.
    shrunk["supporting_context"] = []
    return shrunk


def compact_analysis_json(packet: Dict[str, object], max_chars: int = 700) -> str:
    """Serialize compact analysis JSON with strict size budget."""
    shrunk = _shrink_for_budget(packet, max_chars=max_chars)
    encoded = json.dumps(shrunk, separators=(",", ":"), ensure_ascii=True)
    if len(encoded) <= max_chars:
        return encoded
    # Final hard cap: emit a minimal valid packet.
    minimal: Dict[str, object] = {
        "task_summary": str(packet.get("task_summary") or "")[:80],
        "deps_confidence": str(packet.get("deps_confidence") or "unknown"),
        "deps_sources": list(packet.get("deps_sources") or [])[:2],
        "include_headers": list(packet.get("include_headers") or [])[:2],
        "hazards": list(packet.get("hazards") or [])[:2],
        "touch_points": list(packet.get("touch_points") or [])[:1],
        "files_allowed": list(packet.get("files_allowed") or [])[:2],
        "symbols_allowed": list(packet.get("symbols_allowed") or [])[:2],
        "likely_build_system": str(packet.get("likely_build_system") or "unknown"),
        "suggested_validate_cmd": str(packet.get("suggested_validate_cmd") or "")[:80],
        "dependency_context": list(packet.get("dependency_context") or [])[:1],
        "constraints": list(packet.get("constraints") or packet.get("non_negotiables") or [])[:2],
    }
    encoded = json.dumps(minimal, separators=(",", ":"), ensure_ascii=True)
    if len(encoded) <= max_chars:
        return encoded
    # If still too long, shrink only summary while keeping valid JSON.
    minimal["task_summary"] = str(minimal.get("task_summary") or "")[:32]
    return json.dumps(minimal, separators=(",", ":"), ensure_ascii=True)


def build_compact_analysis(user_text: str, focus_file: str, max_chars: int = 700) -> Dict[str, object]:
    """Build a compact, language-agnostic analysis packet for prompt injection."""
    focus_file = _norm_rel_path(focus_file)
    focus_content = read_single_file_for_context(focus_file).get(focus_file, "")

    local_refs = _parse_local_deps(focus_file, focus_content)
    hazards: List[str] = []

    has_todos = bool(re.search(r"\b(TODO|FIXME|TBD)\b|placeholder|stub", focus_content, re.IGNORECASE))
    if has_todos:
        hazards.append("Placeholders remain in file.")
    if len(focus_content) > 12000:
        hazards.append("Large file; preserve existing anchors.")
    touch_points = _extract_touch_points(focus_file, focus_content)
    if not touch_points:
        touch_points = [{"file": focus_file, "symbol": "<file>", "lines": "L1-L1"}]

    deps_sources: List[Dict[str, object]] = []
    for ref in local_refs[:5]:
        deps_sources.append(
            {
                "symbol": Path(ref).stem,
                "source": "include",
                "file": ref,
            }
        )
    deps_confidence = "partial" if deps_sources else "unknown"
    symbols_allowed = _extract_symbols_allowed(touch_points)
    files_allowed: List[str] = [focus_file]
    for ref in local_refs[:4]:
        norm = _norm_rel_path(ref)
        if norm not in files_allowed:
            files_allowed.append(norm)
    likely_build_system = _detect_build_system(get_root())
    suggested_validate_cmd = _suggested_validate_cmd(focus_file, likely_build_system)
    dependency_context = _collect_dependency_context(local_refs, symbols_allowed)

    constraints = _build_non_negotiables(has_todos)[:5]
    support: List[Dict[str, str]] = []
    for tp in touch_points[:2]:
        lines = str(tp.get("lines") or "")
        m = re.match(r"L(\d+)", lines)
        line_no = int(m.group(1)) if m else 1
        snip = _snippet(focus_file, line_no, radius=4)
        if snip:
            support.append(
                {
                    "file": focus_file,
                    "why": f"touch {str(tp.get('symbol') or '<symbol>')}",
                    "snippet": "\n".join(snip.splitlines()[:4]),
                }
            )

    packet: Dict[str, object] = {
        "task_summary": _compact_summary(user_text, focus_file),
        "deps_confidence": deps_confidence,
        "deps_sources": deps_sources[:5],
        "include_headers": local_refs[:6],
        "hazards": hazards[:4],
        "touch_points": touch_points[:4],
        "files_allowed": files_allowed[:6],
        "symbols_allowed": symbols_allowed[:8],
        "likely_build_system": likely_build_system,
        "suggested_validate_cmd": suggested_validate_cmd,
        "dependency_context": dependency_context[:2],
        "constraints": constraints,
        "non_negotiables": constraints,
        "supporting_context": support[:2],
    }
    return _shrink_for_budget(packet, max_chars=max_chars)


def prepass(user_text: str, focus_file: str) -> str:
    packet = build_compact_analysis(user_text, focus_file, max_chars=1000)
    return json.dumps(packet, indent=2)
