import re
from pathlib import Path
from typing import List, Optional

from .files import _norm_rel_path, get_root


ADD_METHOD = "ADD_METHOD"
MODIFY_EXISTING = "MODIFY_EXISTING"
CLEAR_RANGE = "CLEAR_RANGE"
DELETE_FILE = "DELETE_FILE"
CREATE_NEW = "CREATE_NEW"
IMPLEMENT_STUBS = "IMPLEMENT_STUBS"
MULTI_COMPONENT = "MULTI_COMPONENT"
UNKNOWN = "UNKNOWN"


def has_stub_placeholders(content: str) -> bool:
    """Language-agnostic placeholder/stub detector for starter code."""
    if not content:
        return False

    # Universal markers
    if re.search(r"\b(TODO|FIXME|TBD)\b|placeholder|stub", content, re.IGNORECASE):
        return True

    # Python-style placeholder
    if re.search(r"^\s*pass\s*(?:#.*)?$", content, re.MULTILINE):
        return True

    # JS/TS placeholder throws
    if re.search(r"throw\s+new\s+Error\s*\(\s*['\"](?:TODO|stub|not implemented)", content, re.IGNORECASE):
        return True

    # Clearly empty/comment-only function bodies in brace languages
    if re.search(
        r"\)\s*\{\s*(?:(?://[^\n]*\n)|(?:/\*[\s\S]*?\*/)|\s)*\}",
        content,
        re.IGNORECASE,
    ):
        return True

    # Clearly unimplemented return-only bodies
    if re.search(
        r"\)\s*\{\s*(?:(?://[^\n]*\n)|(?:/\*[\s\S]*?\*/)|\s)*return\s+0\s*;\s*\}",
        content,
        re.IGNORECASE,
    ):
        return True
    if re.search(
        r"def\s+\w+\s*\([^)]*\)\s*:\s*(?:#.*\n|\s)*return\s+None\b",
        content,
        re.IGNORECASE,
    ):
        return True

    return False


def classify_intent(
    user_text: str, file_content: str, focus_file: Optional[str] = None
) -> str:
    """Classify intent based on user request + current file state.

    Priority order:
      DELETE_FILE > CLEAR_RANGE > IMPLEMENT_STUBS > MULTI_COMPONENT >
      MODIFY_EXISTING > ADD_METHOD > CREATE_NEW > UNKNOWN
    """
    if not user_text:
        return UNKNOWN

    text = user_text.strip().lower()

    # --- EARLY OUT: empty/new file → always CREATE_NEW ---
    if not file_content.strip():
        return CREATE_NEW

    # --- DELETE_FILE: explicit file deletion ---
    if re.search(r"\b(delete|remove)\b.*\b(file|module)\b", text):
        return DELETE_FILE
    if re.search(r"\b(delete|remove)\b.*\b" + re.escape((focus_file or "").lower()) + r"\b", text):
        return DELETE_FILE
    # delete/remove Makefile, Dockerfile, etc.
    if re.search(
        r"\b(delete|remove)\b[^\n]{0,60}\b(makefile|gnumakefile|dockerfile|cmakelists\.txt)\b",
        text,
        re.IGNORECASE,
    ):
        return DELETE_FILE

    # --- CLEAR_RANGE: erase / clear / wipe content ---
    if re.search(r"\b(clear|wipe|erase)\b.*\b(file|contents|content)\b", text):
        return CLEAR_RANGE

    # --- IMPLEMENT_STUBS: fill existing TODO/stub placeholders ---
    has_todo = has_stub_placeholders(file_content or "")
    wants_impl = bool(
        re.search(r"\b(implement|finish|complete|fill(?:\s+in)?|solve)\b", text)
        or re.search(r"\b(write|code)\b.*\b(function|todo|stub|placeholder)\b", text)
    )
    if has_todo and wants_impl:
        return IMPLEMENT_STUBS

    # --- MULTI_COMPONENT: implement multiple things ---
    # If user mentions multiple target files in one request.
    if len(extract_target_files(user_text)) >= 2:
        return MULTI_COMPONENT
    # Multi-target wording should mention files/modules explicitly.
    if (
        re.search(r"\b(also|in addition|as well as)\b", text)
        and re.search(r"\b(file|files|module|modules)\b", text)
    ):
        return MULTI_COMPONENT

    # --- MODIFY_EXISTING: edit / fix / change existing code ---
    if re.search(r"\b(modify|change|update|rename|refactor|replace|rewrite)\b", text):
        return MODIFY_EXISTING

    # If user mentions "function", "method", "class" without "add"
    if re.search(r"\b(function|method|class)\b", text):
        return MODIFY_EXISTING

    # --- ADD_METHOD: add new code to an existing file ---
    if re.search(r"\b(add|insert|append|extend)\b", text):
        return ADD_METHOD
    if re.search(r"\b(add|create|implement)\b.*\b(function|method|class)\b", text):
        return ADD_METHOD
    if re.search(r"\bnew\b.*\b(function|method|class)\b", text):
        return ADD_METHOD

    # --- CREATE_NEW: empty file or explicit creation ---
    if re.search(r"\b(create|make|write)\b.*\b(file|module)\b", text):
        return CREATE_NEW

    return UNKNOWN


def _filename_seems_wrong(focus_file: str, user_text: str) -> bool:
    """True if the focus file name would bias the model away from the request."""
    name = Path(focus_file).stem.lower()
    text = (user_text or "").lower()
    if any(k in text for k in ("test", "unittest", "pytest")):
        return False
    if name.startswith("test") or name.endswith("test") or name == "tests":
        return True
    if name in ("temp", "tmp", "scratch", "untitled", "new", "file"):
        return True
    return False


def _detect_language_ext(text: str, fallback: str = ".py") -> str:
    """Detect the target programming language from context clues in the text."""
    t = text.lower()
    file_mentions = re.findall(r"\b\w+\.(c|cpp|cc|h|hpp|go|rs|java|js|ts|py|rb|sh)\b", t)
    if file_mentions:
        ext = file_mentions[0]
        return f".{ext}"
    c_clues = ("fork(", "fork ", "exec(", "execvp", "waitpid", "pid_t",
               "#include", "printf(", "malloc(", "free(", "stdin", "stdout",
               "argc", "argv", "makefile", ".c file", "gcc", "pipe(",
               "sigaction", "getpid", "processes")
    if any(clue in t for clue in c_clues):
        return ".c"
    cpp_clues = ("cout", "cin", "std::", "vector<", "#include <iostream>",
                 "namespace", ".cpp file", "g++")
    if any(clue in t for clue in cpp_clues):
        return ".cpp"
    java_clues = ("public static void main", "system.out", ".java file",
                  "javac", "extends", "implements")
    if any(clue in t for clue in java_clues):
        return ".java"
    js_clues = ("console.log", "const ", "let ", "require(", ".js file",
                "node ", "npm ")
    if any(clue in t for clue in js_clues):
        return ".js"
    go_clues = ("func main(", "fmt.print", "go run", ".go file", "goroutine")
    if any(clue in t for clue in go_clues):
        return ".go"
    rust_clues = ("fn main(", "let mut", "cargo ", ".rs file", "println!(")
    if any(clue in t for clue in rust_clues):
        return ".rs"
    return fallback


def classify_reference_files(user_text: str, files: List[str]) -> List[str]:
    """Classify which files are reference (read-only context) vs focus (to edit).
    Uses prompt semantics: read/run/try/compile -> reference; modify/edit/update -> focus.
    """
    if not user_text or not files:
        return []
    low = user_text.lower()
    refs: List[str] = []
    for path in files:
        stem = Path(path).stem.lower()
        name = Path(path).name.lower()
        # Reference: read X, run X, try X — require stem as whole word (parent != parentcreates)
        if re.search(rf"\b(?:read|open)\s+[^\n]*\b{re.escape(stem)}\b", low):
            refs.append(path)
        elif re.search(rf"\b(?:run|try|execute)\s+[^\n]*\b{re.escape(stem)}\b", low):
            refs.append(path)
        elif "makefile" in name and re.search(r"\b(?:use|compile)\s+(?:the\s+)?makefile", low):
            refs.append(path)
    return list(dict.fromkeys(refs))


def extract_target_files(user_text: str) -> List[str]:
    """Extract filenames the user actually wants created/modified.

    Target selection rules:
    - File must be mentioned in the request with an action verb nearby (edit, update,
      create, fix, add, modify, delete, etc.) within ~120 chars.
    - Supports: .c, .py, .js, etc. and extensionless names (Makefile, Dockerfile, etc.).
    - File must exist in repo, OR be an explicit create request, OR be a build file
      (Makefile, etc.) with an edit verb.
    - Skips: stdlib headers, incidental #include/import refs, readme.md, package.json.
    """
    if not user_text:
        return []
    _CODE_EXTS = (
        "c", "cpp", "cc", "cxx", "h", "hpp", "hh",
        "py", "pyi", "js", "jsx", "ts", "tsx", "mjs",
        "go", "rs", "java", "kt", "scala", "clj",
        "rb", "php", "swift", "cs", "fs", "vb",
        "lua", "pl", "ex", "exs", "erl", "hs",
        "dart", "zig", "nim", "v", "r",
        "sh", "bash", "zsh",
        "sql", "html", "css", "scss", "vue", "svelte",
    )
    ext_re = "|".join(re.escape(e) for e in _CODE_EXTS)
    file_pat = re.compile(rf"\b([A-Za-z0-9_./-]+\.(?:{ext_re}))\b", re.IGNORECASE)
    # Extensionless build/config files (Makefile, Dockerfile, etc.)
    _EXTENSIONLESS_NAMES = (
        "makefile", "gnumakefile", "cmakelists.txt",
        "dockerfile", "containerfile", "vagrantfile",
        "justfile", "gemfile", "rakefile", "procfile", "brewfile",
    )
    extless_re = "|".join(re.escape(n) for n in _EXTENSIONLESS_NAMES)
    extless_pat = re.compile(rf"\b([A-Za-z0-9_./-]*?(?:{extless_re}))\b", re.IGNORECASE)
    _ACTION_VERBS = {
        "modify", "create", "implement", "write", "update", "change",
        "edit", "fix", "add", "submit", "push", "commit", "complete",
        "finish", "build", "make", "ensure", "updated",
        "delete", "remove",
    }
    _SKIP = {"readme.md", "license.md", "changelog.md", "package.json"}
    _STD_HEADERS = {
        "assert.h", "ctype.h", "errno.h", "float.h", "limits.h", "locale.h",
        "math.h", "setjmp.h", "signal.h", "stdarg.h", "stdbool.h", "stddef.h",
        "stdint.h", "stdio.h", "stdlib.h", "string.h", "time.h", "uchar.h",
        "wchar.h", "wctype.h", "unistd.h", "fcntl.h", "pthread.h", "dirent.h",
        "sys/types.h", "sys/stat.h", "sys/wait.h", "sys/socket.h", "sys/time.h",
        "arpa/inet.h", "netinet/in.h",
    }
    _STDLIB_MODULES = {
        "os", "sys", "re", "math", "json", "time", "pathlib", "typing",
        "collections", "itertools", "functools", "subprocess", "asyncio",
        "logging", "datetime", "random", "statistics", "io", "string", "threading",
        "stdio", "iostream",
    }
    _verb_re = re.compile(
        r"\b(?:" + "|".join(re.escape(v) for v in _ACTION_VERBS) + r")\b",
        re.IGNORECASE,
    )
    all_files = [(m.start(), _norm_rel_path(m.group(1))) for m in file_pat.finditer(user_text)]
    for m in extless_pat.finditer(user_text):
        path = _norm_rel_path(m.group(1))
        if path and Path(path).name.lower() in _EXTENSIONLESS_NAMES:
            all_files.append((m.start(), path))
    all_files.sort(key=lambda t: t[0])

    root = get_root()

    def _exists_in_repo(path: str) -> bool:
        try:
            candidate = (root / _norm_rel_path(path)).resolve()
        except Exception:
            return False
        return candidate.exists() and candidate.is_file()

    def _is_stdlib_ref(path: str) -> bool:
        low = path.lower()
        if low in _STD_HEADERS or low.startswith("sys/"):
            return True
        # Treat bare module names (or direct "<module>.py") as stdlib mentions.
        if "/" not in low:
            stem = Path(low).stem
            if stem in _STDLIB_MODULES:
                return True
        return False

    def _is_explicit_create_request(path: str) -> bool:
        low_text = user_text.lower()
        low_path = path.lower()
        escaped = re.escape(low_path)
        patterns = (
            rf"\b(create|add|make|generate)\b[^\n]{{0,80}}\b(file|module)\b[^\n]{{0,120}}{escaped}\b",
            rf"\b(create|add|make|generate)\b[^\n]{{0,120}}{escaped}\b",
            rf"\bnew\b[^\n]{{0,80}}\b(file|module)\b[^\n]{{0,120}}{escaped}\b",
        )
        return any(re.search(p, low_text, re.IGNORECASE) for p in patterns)

    def _is_extensionless_build_edit(path: str) -> bool:
        """Include Makefile, Dockerfile, etc. when user has edit/update verb (even if file missing)."""
        return Path(path).name.lower() in _EXTENSIONLESS_NAMES

    def _is_incidental_reference(pos: int, fname: str) -> bool:
        low_name = fname.lower()
        stem = Path(fname).stem.lower()
        start = max(0, pos - 80)
        end = min(len(user_text), pos + len(fname) + 80)
        window = user_text[start:end].lower()
        if re.search(rf"#\s*include\s*[<\"]\s*{re.escape(low_name)}\s*[>\"]", window):
            return True
        if re.search(rf"\bimport\s+{re.escape(stem)}\b", window):
            return True
        if re.search(rf"\bfrom\s+{re.escape(stem)}\s+import\b", window):
            return True
        if re.search(rf"\b(?:#\s*include|import|from|require)\b[^\n]{{0,80}}{re.escape(low_name)}", window):
            return True
        return False
    seen: set = set()
    result: List[str] = []
    for pos, fname in all_files:
        low = fname.lower()
        if low in seen or low in _SKIP:
            continue
        if _is_stdlib_ref(fname):
            continue
        if _is_incidental_reference(pos, fname):
            continue
        window = user_text[max(0, pos - 120): pos + len(fname) + 120]
        if _verb_re.search(window):
            if not (
                _exists_in_repo(fname)
                or _is_explicit_create_request(fname)
                or _is_extensionless_build_edit(fname)
            ):
                continue
            seen.add(low)
            result.append(fname)
            continue
    if not result:
        for pos, fname in all_files:
            low = fname.lower()
            if low not in seen and low not in _SKIP:
                if _is_stdlib_ref(fname):
                    continue
                if _is_incidental_reference(pos, fname):
                    continue
                if not (
                    _exists_in_repo(fname)
                    or _is_explicit_create_request(fname)
                    or _is_extensionless_build_edit(fname)
                ):
                    continue
                seen.add(low)
                result.append(fname)
    return result


def derive_filename(user_text: str, focus_file: str) -> str:
    """Derive a meaningful filename from the user's request."""
    mentioned = extract_target_files(user_text)
    if mentioned:
        return mentioned[0]
    fallback_ext = Path(focus_file).suffix or ".py"
    ext = _detect_language_ext(user_text, fallback_ext)
    text = (user_text or "").lower()
    for verb in ("implement", "create", "build", "write", "make", "add"):
        text = text.replace(verb, "")
    words = re.findall(r"\b[a-z]\w{2,}\b", text)
    noise = {
        "the", "and", "for", "with", "from", "this", "that", "all",
        "new", "class", "classes", "function", "method", "some", "file",
    }
    words = [w for w in words if w not in noise]
    if words:
        name = "_".join(words[:3])
        return f"{name}{ext}"
    return focus_file
