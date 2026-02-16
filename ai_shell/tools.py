"""External tool wrappers: ripgrep and tree-sitter.

These run deterministic tools BEFORE model reasoning (like Continue's tool system).
Each returns structured data — no AI calls.
"""

import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .files import get_include, get_root
from .utils import dbg


# ---------------------------------------------------------------------------
# 1. Ripgrep search — fast "where is X?" across entire repo
# ---------------------------------------------------------------------------

_RG_BIN = "rg"  # assumes ripgrep is on PATH


def _grep_include_paths() -> Optional[List[str]]:
    """When include filter is set, return exact paths to search — only imported files, nothing else."""
    include = get_include()
    if not include:
        return None
    # Use indexed files (respects include) — grep searches only those files
    from .index import get_indexed_files
    return get_indexed_files()


@dataclass
class GrepMatch:
    """A single ripgrep match."""

    file: str  # relative path
    line: int  # 1-based line number
    text: str  # matched line content


@dataclass
class GrepResult:
    """Aggregated grep results."""

    matches: List[GrepMatch] = field(default_factory=list)
    truncated: bool = False
    query: str = ""


def grep_search(
    query: str,
    root: Path = None,
    max_results: int = 100,
    literal: bool = True,
    file_glob: str = "",
    context_lines: int = 0,
    include_paths: Optional[List[str]] = None,
) -> GrepResult:
    """Run ripgrep search across the repo. Returns structured matches.

    Args:
        query: search pattern (escaped to literal by default)
        root: repo root (defaults to get_root())
        max_results: cap on number of matches
        literal: if True, treat query as literal (not regex)
        file_glob: optional glob filter (e.g. "*.py")
        context_lines: lines of context around each match (0 = just the match)
        include_paths: when set, search only these paths (relative to root).
            Used when user has imported specific files; prevents searching app source.
    """
    if root is None:
        root = get_root()

    if not query.strip():
        return GrepResult(query=query)

    cmd = [
        _RG_BIN,
        "--no-heading",
        "--line-number",
        "--max-count",
        str(max_results),
        "--max-columns",
        "200",
        "--max-columns-preview",
    ]

    if literal:
        cmd.append("--fixed-strings")
    if context_lines > 0:
        cmd.extend(["-C", str(context_lines)])
    if file_glob:
        cmd.extend(["--glob", file_glob])

    # Ignore common noise dirs
    for d in ("node_modules", ".git", "__pycache__", ".venv", "core", "extensions"):
        cmd.extend(["--glob", f"!{d}/"])

    # When include filter is set, search ONLY those files — nothing else
    if include_paths is None:
        include_paths = _grep_include_paths()
    if include_paths is not None:
        # Restrict to imported files; empty list = no files to search
        search_paths = include_paths if include_paths else ["/dev/null"]  # no matches
    else:
        search_paths = ["."]
    cmd.extend(["--", query] + search_paths)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(root),
            timeout=10,
        )
    except FileNotFoundError:
        dbg("grep_search: ripgrep (rg) not found on PATH")
        return GrepResult(query=query)
    except subprocess.TimeoutExpired:
        dbg("grep_search: ripgrep timed out")
        return GrepResult(query=query, truncated=True)

    matches = []
    for line in (result.stdout or "").splitlines():
        # Format: file:line:content
        m = re.match(r"^(.+?):(\d+)[:|-](.*)$", line)
        if m:
            matches.append(
                GrepMatch(
                    file=m.group(1),
                    line=int(m.group(2)),
                    text=m.group(3),
                )
            )

    truncated = len(matches) >= max_results
    if truncated:
        dbg(f"grep_search: truncated at {max_results} results")

    dbg(f"grep_search: query={query!r} found {len(matches)} matches")
    return GrepResult(matches=matches, truncated=truncated, query=query)


def grep_symbol_definitions(
    symbol: str,
    root: Path = None,
    file_glob: str = "",
    include_paths: Optional[List[str]] = None,
) -> GrepResult:
    """Find where a symbol is defined (class, def, function, const, etc.)."""
    # Regex: start of line, optional whitespace, def/class/function/const/let/var, then symbol
    pattern = rf"^\s*(def|class|function|async function|const|let|var|fn|func|type|interface|export)\s+{re.escape(symbol)}\b"
    return grep_search(
        pattern,
        root=root,
        literal=False,
        file_glob=file_glob,
        max_results=20,
        include_paths=include_paths,
    )


# ---------------------------------------------------------------------------
# 2. Tree-sitter symbol extraction — accurate multi-language parsing
# ---------------------------------------------------------------------------


@dataclass
class Symbol:
    """A symbol (class, function, method) extracted from source code."""

    name: str
    kind: str  # "class", "function", "method"
    line: int  # 1-based start line
    end_line: int  # 1-based end line
    start_byte: int = 0
    end_byte: int = 0
    parent: str = ""  # parent class name (for methods)


def extract_symbols_treesitter(
    file_path: str,
    content: str = "",
    root: Path = None,
) -> List[Symbol]:
    """Extract symbols from a file using tree-sitter.

    Falls back gracefully if tree-sitter is not installed.
    Supports Python, JavaScript, TypeScript, Go, Rust, Java, C/C++, Ruby, and more.
    """
    if root is None:
        root = get_root()

    try:
        from tree_sitter_languages import get_parser
    except ImportError:
        dbg("extract_symbols_treesitter: tree-sitter-languages not installed")
        return []

    # Map file extension to tree-sitter language name
    ext = os.path.splitext(file_path)[1].lstrip(".")
    lang_map = {
        "py": "python",
        "js": "javascript",
        "jsx": "javascript",
        "ts": "typescript",
        "tsx": "tsx",
        "go": "go",
        "rs": "rust",
        "rb": "ruby",
        "java": "java",
        "c": "c",
        "cpp": "cpp",
        "cc": "cpp",
        "h": "c",
        "hpp": "cpp",
        "cs": "c_sharp",
        "php": "php",
        "swift": "swift",
        "kt": "kotlin",
        "scala": "scala",
        "lua": "lua",
        "sh": "bash",
    }
    lang_name = lang_map.get(ext)
    if not lang_name:
        return []

    # Read content if not provided
    if not content:
        try:
            full_path = root / file_path
            content = full_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return []

    try:
        parser = get_parser(lang_name)
        tree = parser.parse(content.encode())
    except Exception as e:
        dbg(f"extract_symbols_treesitter: parse error for {file_path}: {e}")
        return []

    # Node types that represent symbol definitions
    # These vary by language but tree-sitter uses consistent naming
    symbol_node_types = {
        "class_definition",
        "class_declaration",
        "function_definition",
        "function_declaration",
        "method_definition",
        "method_declaration",
        "async_function_declaration",
        "decorated_definition",  # Python @decorator
        "interface_declaration",
        "type_alias_declaration",
        "struct_item",  # Rust
        "impl_item",  # Rust
        "enum_item",  # Rust
        "trait_item",  # Rust
    }

    symbols: List[Symbol] = []

    def _find_name(node) -> str:
        """Find the identifier name in a node, searching up to 2 levels deep.
        Handles C/C++ where name is inside a declarator grandchild."""
        for child in node.children:
            if child.type in (
                "identifier", "property_identifier", "type_identifier",
            ):
                return child.text.decode()
            # C/C++: function_definition -> function_declarator -> identifier
            if child.type in (
                "function_declarator", "declarator",
                "pointer_declarator", "init_declarator",
            ):
                for grandchild in child.children:
                    if grandchild.type == "identifier":
                        return grandchild.text.decode()
        return ""

    def _walk(node, parent_name=""):
        if node.type in symbol_node_types:
            name = _find_name(node)

            # For decorated_definition (Python), dig into the inner node
            if node.type == "decorated_definition":
                for child in node.children:
                    if child.type in ("class_definition", "function_definition"):
                        _walk(child, parent_name)
                return

            if name:
                kind = "class"
                if "function" in node.type or "method" in node.type:
                    kind = "method" if parent_name else "function"
                elif "struct" in node.type or "impl" in node.type:
                    kind = "class"
                elif "interface" in node.type or "type" in node.type:
                    kind = "class"
                elif "trait" in node.type or "enum" in node.type:
                    kind = "class"

                symbols.append(
                    Symbol(
                        name=name,
                        kind=kind,
                        line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        start_byte=int(getattr(node, "start_byte", 0) or 0),
                        end_byte=int(getattr(node, "end_byte", 0) or 0),
                        parent=parent_name,
                    )
                )

                # Recurse into class-like container bodies for nested members.
                # Use resolved kind (not raw node.type) so struct/interface/trait containers
                # also propagate parent names where supported by the grammar.
                if kind == "class":
                    for child in node.children:
                        _walk(child, name)
                    return

        # Recurse into children
        for child in node.children:
            _walk(child, parent_name)

    _walk(tree.root_node)

    dbg(
        f"treesitter: {file_path}: {len(symbols)} symbols "
        f"({sum(1 for s in symbols if s.kind == 'class')} classes, "
        f"{sum(1 for s in symbols if s.kind in ('function', 'method'))} functions)"
    )

    return symbols
