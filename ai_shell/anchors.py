"""Insertion anchor computation for code editing.

Pipeline:
  1. Tree-sitter parses file structure (classes, methods, line ranges)
  2. Ripgrep locates the target class the user is referring to
  3. AST enriches with Python method signatures (Python only)
  4. Heuristic regex as last resort
"""

import ast
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Optional

from .files import read_single_file_for_context
from .utils import dbg


@dataclass
class InsertionAnchor:
    """Describes exactly where new code should be inserted."""
    anchor_before: List[str]       # 2-5 exact unchanged lines before insertion
    anchor_after: List[str]        # 2-5 exact unchanged lines after insertion
    insert_at_line: int            # 1-based line number for insertion
    indent: str                    # indentation string for new method (e.g., "    ")
    class_name: str                # target class name
    after_method: str              # method name we're inserting after (or "")
    existing_methods: List[str] = None          # method names already in the class
    existing_signatures: List[str] = None       # full signatures (Python AST enrichment)


def _is_python(focus_file: str) -> bool:
    return Path(focus_file).suffix.lower() == ".py"


# ---------------------------------------------------------------------------
# Step 1: Tree-sitter -- parse file structure
# ---------------------------------------------------------------------------


def _treesitter_parse(focus_file: str, file_content: str) -> list:
    """Parse file with tree-sitter. Returns list of Symbol or empty list."""
    try:
        from .tools import extract_symbols_treesitter
        symbols = extract_symbols_treesitter(focus_file, content=file_content)
        return symbols
    except ImportError:
        dbg("treesitter_parse: tree-sitter-languages not installed")
        return []
    except Exception as exc:
        dbg(f"treesitter_parse: failed: {exc}")
        return []


# ---------------------------------------------------------------------------
# Step 2: Ripgrep -- locate target class
# ---------------------------------------------------------------------------


def _fuzzy_match(word: str, target: str, threshold: float = 0.75) -> bool:
    """Check if word is a fuzzy match for target (typo-tolerant).
    Both inputs should be lowercase. Threshold 0.75 catches 1-2 char typos
    in typical class names (4-10 chars)."""
    if len(word) < 3 or len(target) < 3:
        return False
    return SequenceMatcher(None, word, target).ratio() >= threshold


def _locate_target(symbols: list, user_text: str, focus_file: str):
    """Find which class the user is targeting and its methods.

    1. Check tree-sitter symbols for exact class name match
    2. Fuzzy match: handle typos like "queuq" → "Queue"
    3. Ripgrep: class definitions matching user keywords
    4. Fallback: pick class with most methods for 'add' requests
    Returns (target_class_symbol, [method_symbols]) or (None, [])
    """
    text_lower = (user_text or "").lower()
    words = re.findall(r"\w+", text_lower)
    classes = [s for s in symbols if s.kind == "class"]
    all_methods = lambda cls_name: [
        s for s in symbols
        if s.kind in ("function", "method") and s.parent == cls_name
    ]

    # --- Direct name match from tree-sitter symbols ---
    for cls in classes:
        if cls.name.lower() in text_lower:
            methods = all_methods(cls.name)
            dbg(f"locate_target: tree-sitter matched class {cls.name} "
                f"({len(methods)} methods)")
            return cls, methods

    # --- Fuzzy match: handle typos (queuq → Queue) ---
    best_fuzzy = None
    best_ratio = 0.0
    for cls in classes:
        cls_low = cls.name.lower()
        for w in words:
            ratio = SequenceMatcher(None, w, cls_low).ratio()
            if ratio >= 0.75 and ratio > best_ratio and len(w) >= 3:
                best_fuzzy = cls
                best_ratio = ratio
    if best_fuzzy:
        methods = all_methods(best_fuzzy.name)
        dbg(f"locate_target: fuzzy matched class {best_fuzzy.name} "
            f"(ratio={best_ratio:.2f}, {len(methods)} methods)")
        return best_fuzzy, methods

    # --- Ripgrep: search for class definitions matching likely identifiers ---
    try:
        from .tools import grep_symbol_definitions
        skip = {
            "Implement", "Focus", "Only", "Relevant", "Task", "Make", "Now",
            "Modify", "Both", "The", "Note", "Add", "Earlier", "If",
            "However", "When", "What", "Submit", "Return", "Minimal", "Change",
        }
        rg_keywords = []
        for kw in re.findall(r"\b[A-Z][A-Za-z0-9_]+\b", user_text):
            if kw in skip or kw.isupper():
                continue
            # Keep likely symbols only (CamelCase or underscore identifiers).
            if ("_" in kw) or (any(c.islower() for c in kw[1:]) and any(c.isupper() for c in kw[1:])):
                rg_keywords.append(kw)
            elif len(kw) >= 6 and kw[0].isupper() and kw[1:].islower():
                # Allow longer class-like words, reject common short prose.
                rg_keywords.append(kw)
        for kw in rg_keywords[:5]:
            result = grep_symbol_definitions(kw)
            for match in result.matches:
                # Only matches in our focus file
                if match.file.rstrip("./") == focus_file.rstrip("./"):
                    # Cross-reference with tree-sitter symbols
                    for cls in classes:
                        if cls.name == kw:
                            methods = all_methods(cls.name)
                            dbg(f"locate_target: ripgrep matched class {cls.name} "
                                f"in {focus_file} ({len(methods)} methods)")
                            return cls, methods
    except Exception as exc:
        dbg(f"locate_target: ripgrep fallback failed: {exc}")

    # --- Fallback: pick class with most methods for "add" requests ---
    if classes:
        is_add = any(k in text_lower for k in (
            "add ", "insert ", "delete ", "remove ", "create ", "implement ",
        ))
        if is_add:
            best = max(classes, key=lambda c: len(all_methods(c.name)))
            methods = all_methods(best.name)
            if methods:
                dbg(f"locate_target: fallback picked class {best.name} "
                    f"(most methods: {len(methods)})")
                return best, methods

        # Last try: single class in file
        if len(classes) == 1:
            cls = classes[0]
            methods = all_methods(cls.name)
            dbg(f"locate_target: single class {cls.name}")
            return cls, methods

    return None, []


# ---------------------------------------------------------------------------
# Step 3: Build anchor from located symbols
# ---------------------------------------------------------------------------


def _build_anchor_from_symbols(
    target_cls, methods: list, lines: List[str],
    text_lower: str, context_lines: int,
) -> InsertionAnchor:
    """Build InsertionAnchor from tree-sitter Symbol + methods."""

    # --- Determine insertion point ---
    after_method = ""

    # Check if user mentions inserting after a specific method
    for m in methods:
        if f"after {m.name}" in text_lower:
            after_method = m.name
            break
        # Also check underscores replaced with spaces
        if "_" in m.name and f"after {m.name.replace('_', ' ')}" in text_lower:
            after_method = m.name
            break

    if after_method:
        # Insert after the named method
        for m in methods:
            if m.name == after_method:
                insert_line = m.end_line  # 1-based end_line from tree-sitter
                break
        else:
            insert_line = target_cls.end_line
    elif methods:
        # Insert after the last method
        last_method = max(methods, key=lambda m: m.end_line)
        insert_line = last_method.end_line
        after_method = last_method.name
    else:
        # No methods -- insert after class header
        insert_line = target_cls.line

    # --- Compute indent from existing methods ---
    indent = "    "
    if methods:
        first_method_line_idx = methods[0].line - 1
        if 0 <= first_method_line_idx < len(lines):
            method_line = lines[first_method_line_idx]
            indent_match = re.match(r"^(\s*)", method_line)
            if indent_match and indent_match.group(1):
                indent = indent_match.group(1)

    # --- Build context lines ---
    before_start = max(0, insert_line - context_lines)
    after_start = insert_line
    after_end = min(len(lines), insert_line + context_lines)

    anchor_before = lines[before_start:insert_line]
    anchor_after = lines[after_start:after_end]

    # --- Collect existing method names ---
    existing_method_names = [m.name for m in methods]

    dbg(f"anchor: class={target_cls.name} after={after_method} "
        f"insert_at={insert_line + 1} indent={len(indent)} "
        f"before={len(anchor_before)}L after={len(anchor_after)}L "
        f"existing={existing_method_names}")

    return InsertionAnchor(
        anchor_before=anchor_before,
        anchor_after=anchor_after,
        insert_at_line=insert_line + 1,  # 1-based
        indent=indent,
        class_name=target_cls.name,
        after_method=after_method,
        existing_methods=existing_method_names,
    )


# ---------------------------------------------------------------------------
# Step 4: AST enrichment -- Python method signatures
# ---------------------------------------------------------------------------


def _enrich_with_ast(anchor: InsertionAnchor, file_content: str) -> None:
    """For Python files, enrich anchor with full method signatures from AST.

    Populates anchor.existing_signatures with strings like:
      "def __init__(self)"
      "def append(self, data)"
      "def display(self) -> list"
    """
    try:
        tree = ast.parse(file_content)
    except SyntaxError:
        return  # AST can't parse -- keep what tree-sitter gave us

    # Find the target class
    target_class = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == anchor.class_name:
            target_class = node
            break

    if not target_class:
        return

    signatures = []
    for node in target_class.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        # Build parameter list
        params = []
        args = node.args
        # Positional args
        for i, arg in enumerate(args.args):
            param = arg.arg
            # Add type annotation if present
            if arg.annotation:
                try:
                    param += f": {ast.unparse(arg.annotation)}"
                except Exception:
                    pass
            params.append(param)

        # *args
        if args.vararg:
            params.append(f"*{args.vararg.arg}")
        # **kwargs
        if args.kwarg:
            params.append(f"**{args.kwarg.arg}")

        # Build return annotation
        returns = ""
        if node.returns:
            try:
                returns = f" -> {ast.unparse(node.returns)}"
            except Exception:
                pass

        prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
        sig = f"{prefix} {node.name}({', '.join(params)}){returns}"
        signatures.append(sig)

    if signatures:
        anchor.existing_signatures = signatures
        dbg(f"ast_enrich: {anchor.class_name}: {len(signatures)} signatures")


# ---------------------------------------------------------------------------
# Last resort: heuristic regex
# ---------------------------------------------------------------------------


def _heuristic_anchor(
    lines: List[str], text_lower: str, context_lines: int = 3,
) -> Optional[InsertionAnchor]:
    """Last resort: find insertion point by scanning for class/def patterns."""
    class_ranges = []
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith("class ") and (i == 0 or not line[0].isspace()):
            m = re.match(r"class\s+(\w+)", stripped)
            if m:
                cls_name = m.group(1)
                cls_last = i
                last_def_end = i
                for j in range(i + 1, len(lines)):
                    if lines[j].strip() and not lines[j][0].isspace():
                        break
                    cls_last = j
                    if lines[j].lstrip().startswith("def "):
                        for k in range(j + 1, len(lines)):
                            if lines[k].strip() and len(lines[k]) - len(lines[k].lstrip()) <= len(line) - len(line.lstrip()) + 4:
                                break
                            last_def_end = k
                class_ranges.append((cls_name, i, cls_last, last_def_end))

    if not class_ranges:
        return None

    target = None
    for cls_name, cs, ce, lde in class_ranges:
        if cls_name.lower() in text_lower:
            target = (cls_name, cs, ce, lde)
            break
    if not target:
        target = max(class_ranges, key=lambda c: c[2] - c[1])

    cls_name, cls_start, cls_end, last_def_end = target
    insert_line = last_def_end + 1

    indent = "    "
    for j in range(cls_start + 1, min(cls_start + 10, len(lines))):
        if lines[j].lstrip().startswith("def "):
            indent = re.match(r"^(\s*)", lines[j]).group(1)
            break

    before_start = max(0, insert_line - context_lines)
    after_start = insert_line
    after_end = min(len(lines), insert_line + context_lines)

    dbg(f"anchor(heuristic): class={cls_name} insert_at={insert_line + 1} indent={len(indent)}")

    return InsertionAnchor(
        anchor_before=lines[before_start:insert_line],
        anchor_after=lines[after_start:after_end],
        insert_at_line=insert_line + 1,
        indent=indent,
        class_name=cls_name,
        after_method="",
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def compute_insertion_anchor(
    file_content: str, user_text: str, focus_file: str,
    context_lines: int = 3,
) -> Optional[InsertionAnchor]:
    """Compute the insertion point for adding a method to a class.

    Pipeline:
      1. Tree-sitter parses structure (classes, methods, line ranges)
      2. Ripgrep locates which class the user means
      3. For Python: AST enriches with method signatures
      4. Heuristic regex as last resort
    """
    if not file_content or not file_content.strip():
        return None

    lines = file_content.splitlines()
    text_lower = (user_text or "").lower()

    # --- Step 1: Tree-sitter parses structure ---
    symbols = _treesitter_parse(focus_file, file_content)

    # --- Step 2: Locate the target class (tree-sitter + ripgrep) ---
    target_cls, methods = _locate_target(symbols, user_text, focus_file)

    if target_cls:
        # --- Step 3: Build anchor from structure + location ---
        anchor = _build_anchor_from_symbols(
            target_cls, methods, lines, text_lower, context_lines
        )

        # --- Step 4: Python -- enrich with AST signatures ---
        if _is_python(focus_file):
            _enrich_with_ast(anchor, file_content)

        return anchor

    # --- Fallback: AST for Python (if tree-sitter unavailable) ---
    if _is_python(focus_file):
        anchor = _ast_fallback(file_content, user_text, lines, text_lower, context_lines)
        if anchor:
            return anchor

    # --- Last resort: heuristic regex ---
    return _heuristic_anchor(lines, text_lower, context_lines)


# ---------------------------------------------------------------------------
# Structural context for any intent (not just ADD_METHOD)
# ---------------------------------------------------------------------------


def get_structural_context(file_content: str, focus_file: str) -> str:
    """Use tree-sitter to build a structural summary of the file.

    Returns a concise description of the file's structure that helps the
    model understand what exists, e.g.:
      "File structure: includes (lines 1-3), function main (lines 5-30),
       contains: for-loop (line 12), fork() call (line 15)"

    Works for any language tree-sitter supports (C, Python, JS, etc.).
    Returns empty string if tree-sitter is unavailable or parsing fails.
    """
    if not file_content or not file_content.strip():
        return ""

    symbols = _treesitter_parse(focus_file, file_content)
    if not symbols:
        dbg("structural_context: no tree-sitter symbols, using line-count fallback")
        n = len(file_content.splitlines())
        return f"File has {n} lines."

    parts = []
    for sym in symbols:
        kind = getattr(sym, "kind", "symbol")
        name = getattr(sym, "name", "?")
        start = getattr(sym, "line", 0)
        end = getattr(sym, "end_line", 0)
        children = getattr(sym, "children", [])
        entry = f"{kind} {name} (lines {start}-{end})"
        if children:
            child_names = [getattr(c, "name", "?") for c in children[:10]]
            entry += f" contains: {', '.join(child_names)}"
        parts.append(entry)

    summary = "File structure: " + "; ".join(parts[:15])
    dbg(f"structural_context: {summary[:200]}")
    return summary


# ---------------------------------------------------------------------------
# AST fallback (only if tree-sitter is unavailable)
# ---------------------------------------------------------------------------


def _ast_fallback(
    file_content: str, user_text: str, lines: List[str],
    text_lower: str, context_lines: int,
) -> Optional[InsertionAnchor]:
    """Python-only AST fallback when tree-sitter is not installed."""
    try:
        tree = ast.parse(file_content)
    except SyntaxError:
        return None

    classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
    if not classes:
        return None

    # Match class from user text
    target_class = None
    for cls in classes:
        if cls.name.lower() in text_lower:
            target_class = cls
            break
    if not target_class:
        is_add = any(k in text_lower for k in (
            "add ", "insert ", "delete ", "remove ", "create ", "implement ",
        ))
        if is_add:
            best = max(classes, key=lambda c: len([
                n for n in c.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]))
            if any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) for n in best.body):
                target_class = best

    if not target_class:
        return None

    methods = [
        n for n in target_class.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]

    # Insertion point
    insert_line = 0
    after_method = ""
    if methods:
        last = methods[-1]
        insert_line = last.end_lineno
        after_method = last.name
    else:
        insert_line = target_class.lineno
        for n in target_class.body:
            if isinstance(n, ast.Expr) and isinstance(n.value, (ast.Constant, ast.Str)):
                insert_line = n.end_lineno
                break

    # Indent
    indent = "    "
    if methods:
        first_method_line = lines[methods[0].lineno - 1]
        indent = re.match(r"^(\s*)", first_method_line).group(1)
    else:
        class_line = lines[target_class.lineno - 1]
        class_indent = re.match(r"^(\s*)", class_line).group(1)
        indent = class_indent + "    "

    if insert_line <= 0:
        return None

    before_start = max(0, insert_line - context_lines)
    anchor_before = lines[before_start:insert_line]
    anchor_after = lines[insert_line:min(len(lines), insert_line + context_lines)]

    if not anchor_before and not anchor_after:
        return None

    existing_method_names = [
        n.name for n in target_class.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]

    dbg(f"anchor(ast_fallback): class={target_class.name} after={after_method} "
        f"insert_at={insert_line + 1} indent={len(indent)} "
        f"existing={existing_method_names}")

    anchor = InsertionAnchor(
        anchor_before=anchor_before,
        anchor_after=anchor_after,
        insert_at_line=insert_line + 1,
        indent=indent,
        class_name=target_class.name,
        after_method=after_method,
        existing_methods=existing_method_names,
    )
    # Enrich with signatures since we already have the AST
    _enrich_with_ast(anchor, file_content)
    return anchor


# ---------------------------------------------------------------------------
# Cross-file target discovery
# ---------------------------------------------------------------------------


def discover_target_file(
    user_text: str,
    focus_file: Optional[str],
) -> str:
    """Let tree-sitter + ripgrep decide which file to edit.

    Extracts class/symbol names from the user request, then:
      1. If focus_file contains that symbol → keep focus_file.
      2. Otherwise ripgrep across the project → return the file containing it.
      3. If nothing found → return focus_file unchanged.

    This means the user doesn't have to select the exact right file in the
    UI — tree-sitter figures it out.
    """
    if not user_text:
        return focus_file or ""

    # Skip discovery for very long text (e.g. pasted homework/docs).
    # These aren't short "add X to Queue" commands.
    if len(user_text) > 500:
        dbg("discover_target: skipping — text too long for symbol discovery")
        return focus_file or ""

    text_lower = user_text.lower()

    # --- Extract candidate class/symbol names ---
    # Expanded stop list to filter common English words that happen to be
    # capitalised (sentence starts, headings, etc.)
    _STOP = {
        "Both", "All", "Each", "Every", "Them", "That", "This", "The",
        "Well", "Also", "Too", "Class", "Method", "Function", "File",
        "It", "As", "And", "But", "Not", "For", "With", "From",
        "Start", "Stop", "Run", "Open", "Close", "Use", "Try", "Make",
        "Add", "Get", "Set", "Put", "Can", "How", "What", "When",
        "Where", "Which", "Who", "Why", "Now", "Do", "Did", "Does",
        "If", "So", "Or", "On", "In", "To", "Of", "At", "By", "Up",
        "We", "You", "Your", "Its", "Our", "See", "Note", "Only",
        "Please", "Show", "Print", "Call", "Wait", "Let", "New",
        "Task", "Download", "Submit", "Complete", "Ensure",
        "Discuss", "Question", "Example", "Output", "Notice",
        "However", "Earlier", "Modify",
    }

    # CamelCase words (e.g. "Queue", "LinkedList") — only multi-case names
    # like "LinkedList" or words that look like identifiers, not plain English.
    candidates = []
    for w in re.findall(r"\b([A-Z][a-zA-Z0-9]+)\b", user_text):
        if w in _STOP:
            continue
        # Skip words that are ALL CAPS (acronyms like "ASCII", "PCRS") or
        # just a capitalised common word — keep CamelCase and likely class names
        if w.isupper() and len(w) > 2:
            continue  # "ASCII", "PCRS", "TAS"
        candidates.append(w)

    # Cap at 5 candidates to avoid ripgrep spam
    candidates = candidates[:5]

    if not candidates:
        return focus_file or ""

    # --- Check focus_file first (fast path via tree-sitter) ---
    if focus_file:
        content = read_single_file_for_context(focus_file).get(focus_file, "")
        if content:
            symbols = _treesitter_parse(focus_file, content)
            sym_names = {s.name for s in symbols if s.kind == "class"}
            for cand in candidates:
                # Exact match
                if cand in sym_names:
                    dbg(f"discover_target: {cand} already in {focus_file}")
                    return focus_file
                # Fuzzy match (typo tolerance)
                for sn in sym_names:
                    if _fuzzy_match(cand.lower(), sn.lower()):
                        dbg(f"discover_target: fuzzy {cand}≈{sn} in {focus_file}")
                        return focus_file

    # --- Ripgrep across the project ---
    try:
        from .tools import grep_symbol_definitions
        for cand in candidates:
            result = grep_symbol_definitions(cand)
            for match in result.matches:
                found_file = match.file.lstrip("./")
                # Skip the file we already checked (no match there)
                if found_file != (focus_file or "").lstrip("./"):
                    dbg(f"discover_target: {cand} found in {found_file} "
                        f"(overriding {focus_file})")
                    return found_file
                # If ripgrep found it in focus_file, that's fine too
                dbg(f"discover_target: {cand} confirmed in {focus_file} via rg")
                return focus_file
    except Exception as exc:
        dbg(f"discover_target: ripgrep search failed: {exc}")

    return focus_file or ""


# ---------------------------------------------------------------------------
# Multi-class support
# ---------------------------------------------------------------------------


def _is_multi_class_request(user_text: str) -> bool:
    """Detect requests that target multiple classes ('each', 'all', 'both', etc.)."""
    t = (user_text or "").lower()
    return any(k in t for k in (
        "each of them", "all of them", "both of them", "each class",
        "all classes", "both classes", "every class", "to each", "to all",
        "to both",
    ))


def compute_all_anchors(
    file_content: str, user_text: str, focus_file: str,
    context_lines: int = 3,
) -> List[InsertionAnchor]:
    """Return anchors for ALL classes when the request targets multiple classes."""
    if not _is_multi_class_request(user_text):
        a = compute_insertion_anchor(file_content, user_text, focus_file, context_lines)
        return [a] if a else []

    # Multi-class: use tree-sitter to find all classes
    symbols = _treesitter_parse(focus_file, file_content)
    classes = [s for s in symbols if s.kind == "class"]

    if not classes:
        # Fallback to AST for Python
        if _is_python(focus_file):
            try:
                tree = ast.parse(file_content)
                ast_classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
                classes_names = [c.name for c in ast_classes]
            except SyntaxError:
                classes_names = []
        else:
            classes_names = []

        anchors = []
        for name in classes_names:
            cls_text = f"add to class {name}. {user_text}"
            a = compute_insertion_anchor(file_content, cls_text, focus_file, context_lines)
            if a:
                anchors.append(a)
        return anchors

    anchors = []
    lines = file_content.splitlines()
    for cls in classes:
        cls_text = f"add to class {cls.name}. {user_text}"
        a = compute_insertion_anchor(file_content, cls_text, focus_file, context_lines)
        if a:
            anchors.append(a)
    return anchors
