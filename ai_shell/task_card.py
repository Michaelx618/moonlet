from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple
from .utils import dbg_dump

TODO_RE = re.compile(r"\b(TODO|FIXME|TBD)\b|placeholder|stub", re.IGNORECASE)
ACTION_RE = re.compile(r"\b(implement|complete|fill|fix|update|replace|refactor|modify)\b", re.IGNORECASE)
MILESTONE_RE = re.compile(r"\b(?:m|milestone)\s*\d+\b", re.IGNORECASE)
SEMANTIC_VERB_RE = re.compile(
    r"\b("
    r"print|write|save|return|compute|calculate|invert|read|parse|validate|format|sort|filter|"
    r"create|spawn|fork|wait|open|close|terminate|exit|preserve|keep|modify|change|"
    r"locate|seek|extract|transform|serialize|deserialize|encode|decode|copy"
    r")\b",
    re.IGNORECASE,
)
SEMANTIC_NOISE_RE = re.compile(
    r"\b(introduction|what to submit|download|ta|debugging|commit|push|submission)\b",
    re.IGNORECASE,
)
OUTPUT_FILE_RE = re.compile(r"\b([A-Za-z0-9_.-]+\.[A-Za-z0-9_]+)\b")
OUTPUT_SIGNAL_RE = re.compile(r"\b(write|written|save|saved|output|new file|generated?)\b", re.IGNORECASE)
COMMAND_SIGNAL_RE = re.compile(r"\b(command-?line|argument|argv|stdin|stdout|file path)\b", re.IGNORECASE)
NUMERIC_SPEC_RE = re.compile(r"\b\d+\s*(?:bits?|bytes?|pixels?|rows?|columns?|times?)\b", re.IGNORECASE)
CODE_IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
OUTPUT_EXT_ALLOW = {
    "bmp",
    "png",
    "jpg",
    "jpeg",
    "gif",
    "txt",
    "csv",
    "json",
    "xml",
    "yaml",
    "yml",
    "bin",
    "dat",
    "out",
}
FORMULA_RE = re.compile(r"\b\d+\s*[-+*/]\s*[A-Za-z_][A-Za-z0-9_]*\b")
UNICODE_FORMULA_RE = re.compile(r"\b\d+\s*[−-]\s*[A-Za-z_][A-Za-z0-9_]*\b")
MODAL_RE = re.compile(
    r"\b("
    r"must|must\s+not|should|should\s+not|required|mandatory|shall|"
    r"exact|exactly|strict|strictly|always|only|without|cannot|can't|"
    r"do\s+not\s+assume|never\s+assume|no\s+guarantee|fail\s+if|match\s+exactly"
    r")\b",
    re.IGNORECASE,
)
ONLY_SHOULD_RE = re.compile(r"\bonly\b[\w\s-]{0,80}\bshould\b", re.IGNORECASE)
AT_MOST_RE = re.compile(r"\bat most\b", re.IGNORECASE)
ORDER_RE = re.compile(r"\b(before|after)\b", re.IGNORECASE)
EACH_RE = re.compile(r"\b(for each|each)\b", re.IGNORECASE)
TASK_HEADER_RE = re.compile(r"^\s*task\s*\d+\s*:", re.IGNORECASE)
SECTION_TASK_RE = re.compile(r"^\s*tasks?\s*:?\s*$|^\s*task\s*\d+\s*:?\s*$", re.IGNORECASE)
SECTION_NOTES_RE = re.compile(r"^\s*notes?\s*:?\s*$", re.IGNORECASE)
SECTION_HINTS_RE = re.compile(r"^\s*hints?\s*:?\s*$", re.IGNORECASE)
SECTION_INLINE_RE = re.compile(r"^\s*(task\s*\d+|tasks?|notes?|hints?)\s*:\s*(.+)$", re.IGNORECASE)
MUST_HAVE_SIGNAL_RE = re.compile(
    r"\b("
    r"must\s+have|must\s+include|must\s+contain|must\s+be|must\s+ahve|"
    r"required(?:\s+to)?\s+(?:have|include|contain|use)|"
    r"should\s+include|has\s+to\s+include|needs?\s+to\s+include|"
    r"mandatory|shall|match\s+exactly|exact\s+match|strict(?:ly)?"
    r")\b",
    re.IGNORECASE,
)
STRICTNESS_SIGNAL_RE = re.compile(
    r"\b("
    r"must|must\s+not|required|mandatory|shall|exact|exactly|"
    r"strict(?:ly)?|no\s+guarantee|do\s+not\s+assume|never\s+assume|"
    r"always|only|without|cannot|can't|fail\s+if|must\s+fail|"
    r"match\s+exactly|exact\s+match|prohibited|forbidden|"
    r"only\s+if|unless|except|at\s+least|at\s+most|no\s+more\s+than"
    r")\b",
    re.IGNORECASE,
)
OUTPUT_CONTRACT_TRIGGER_RE = re.compile(
    r"\b("
    r"output should be|the output should be this|example output|"
    r"prints the following|for example,\s*for the provided file|on screen"
    r")\b",
    re.IGNORECASE,
)
OUTPUT_LABEL_LINE_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9 ()/_-]{2,80})\s*:\s*(.+?)\s*$")
FIELD_IDENTIFIER_RE = re.compile(r"\b(?:[a-z]{1,4}[A-Z][A-Za-z0-9_]*|[A-Z]{2,}[A-Za-z0-9_]{2,})\b")
DEPENDENCY_TRIGGER_RE = re.compile(
    r"\b("
    r"therefore,\s*you\s+need|you\s+need\s+the|use\s+the\s+.+\s+to|to\s+locate|"
    r"no\s+guarantee.+same position|starts?\s+at\s+the\s+same\s+position"
    r")\b",
    re.IGNORECASE,
)
FORMAT_DEFINITION_SIGNAL_RE = re.compile(
    r"\b("
    r"exactly\s+that\s+order|in\s+exactly\s+that\s+order|row\s+by\s+row|left\s+to\s+right|"
    r"three\s+bytes|padding|aligned|multiple\s+of\s+4|bottom\s+up|top\s+down|"
    r"negative\s+\w+|positive\s+\w+|stored\s+[\"“”']?bottom\s+up"
    r")\b",
    re.IGNORECASE,
)
EXACTNESS_SIGNAL_RE = re.compile(
    r"\b("
    r"exact(?:ly)?|match\s+exactly|in\s+exactly\s+that\s+order|"
    r"therefore,\s*you\s+need|no\s+guarantee|do\s+not\s+assume|never\s+assume"
    r")\b",
    re.IGNORECASE,
)
CONSTRAINT_SIGNAL_RE = re.compile(
    r"\b(command-?line|original file|unchanged|not changed|write(?:n)? into a new file|"
    r"bits? per pixel|uncompressed|width.*multiple of 4|size of image data|color depth|"
    r"do not assume|never assume|no guarantee|match exactly|exact match|strict(?:ly)?)\b",
    re.IGNORECASE,
)
OFFSET_SIGNAL_RE = re.compile(r"\b(offset|locate|starts? at|position)\b", re.IGNORECASE)
ALIGN_SIGNAL_RE = re.compile(r"\b(multiple of 4|aligned|padding|pad)\b", re.IGNORECASE)
CHANNEL_ORDER_RE = re.compile(
    r"\b(blue,\s*green,\s*red|bgr|rgb color values.*blue.*green.*red)\b",
    re.IGNORECASE,
)
FORMAT_SIGNAL_RE = re.compile(r"\b(uncompressed|24\s*bit|24\s*bits per pixel|color depth)\b", re.IGNORECASE)
HEADER_REF_RE = re.compile(r"\b(bfOffBits|BITMAPFILEHEADER|BITMAPINFOHEADER)\b", re.IGNORECASE)
METADATA_FIELDS_RE = re.compile(
    r"\b(file size|image width|image height|color depth|size of image data)\b",
    re.IGNORECASE,
)
BACKGROUND_NOISE_RE = re.compile(
    r"\b(our goal|purpose of this|you should see|"
    r"don't hesitate|discuss the following|question\s*\d+|hints?)\b",
    re.IGNORECASE,
)
EDU_NOISE_RE = re.compile(
    r"\b(class|videos?|tas?|whiteboard|discuss|exercise|reading week|starter code)\b",
    re.IGNORECASE,
)
ADMIN_NOISE_RE = re.compile(
    r"\b(commit|push|submit|what to submit|download)\b",
    re.IGNORECASE,
)
ELF_CONTEXT_RE = re.compile(r"\b(elf file format|examine an elf|work we did in class)\b", re.IGNORECASE)
SEMANTIC_SCORE_THRESHOLD = 4
SEMANTIC_BORDERLINE_THRESHOLD = 3
SEMANTIC_MUST_COPY_SCORE_THRESHOLD = 18

# Caps (raised to carry more semantic detail into task cards).
SEMANTIC_CANDIDATE_SCAN_LIMIT = 24
SEMANTIC_SELECTED_LIMIT = 16
WORK_ITEMS_LIMIT = 6
CONSTRAINTS_LIMIT = 10
HAZARDS_LIMIT = 2
TASK_CARD_MAX_LINES = 30


@dataclass
class SymbolSpan:
    name: str
    start: int
    end: int


def _extract_symbol_spans(focus_file: str, content: str) -> List[SymbolSpan]:
    spans: List[SymbolSpan] = []
    try:
        from .tools import extract_symbols_treesitter

        syms = extract_symbols_treesitter(focus_file, content=content) or []
        for sym in syms:
            kind = str(getattr(sym, "kind", "") or "")
            if kind not in {"function", "method", "class"}:
                continue
            name = str(getattr(sym, "name", "") or "").strip()
            if not name:
                continue
            start = int(getattr(sym, "line", 1) or 1)
            end = int(getattr(sym, "end_line", start) or start)
            if end < start:
                end = start
            spans.append(SymbolSpan(name=name, start=start, end=end))
    except Exception:
        spans = []

    if spans:
        return spans

    # Regex fallback across brace + def languages.
    lines = content.splitlines()
    n = len(lines)
    fn_pat = re.compile(
        r"^\s*(?:[A-Za-z_][A-Za-z0-9_\s\*:&<>\[\],]*\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*\([^;]*\)\s*\{\s*$"
    )
    py_pat = re.compile(r"^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")

    i = 1
    while i <= n:
        line = lines[i - 1]
        m = py_pat.match(line)
        if m:
            name = m.group(1)
            indent = len(line) - len(line.lstrip(" \t"))
            end = n
            j = i + 1
            while j <= n:
                lj = lines[j - 1]
                if not lj.strip():
                    j += 1
                    continue
                cur_indent = len(lj) - len(lj.lstrip(" \t"))
                if cur_indent <= indent and re.match(r"^\s*(def|class)\s+", lj):
                    end = j - 1
                    break
                j += 1
            spans.append(SymbolSpan(name=name, start=i, end=end))
            i = end + 1
            continue

        m = fn_pat.match(line)
        if m and m.group(1) not in {"if", "for", "while", "switch", "catch"}:
            name = m.group(1)
            depth = 0
            end = i
            j = i
            saw_open = False
            while j <= n:
                l = lines[j - 1]
                depth += l.count("{")
                depth -= l.count("}")
                if "{" in l:
                    saw_open = True
                end = j
                if saw_open and depth <= 0 and j > i:
                    break
                j += 1
            spans.append(SymbolSpan(name=name, start=i, end=end))
            i = end + 1
            continue
        i += 1

    return spans[:30]


def _find_symbol_for_line(line_no: int, spans: List[SymbolSpan]) -> SymbolSpan | None:
    for span in spans:
        if span.start <= line_no <= span.end:
            return span
    return None


def _is_stub_block(block: str) -> bool:
    if not block.strip():
        return True
    if TODO_RE.search(block):
        return True

    cleaned = re.sub(r"/\*[\s\S]*?\*/", "", block)
    cleaned = re.sub(r"//.*", "", cleaned)
    cleaned = re.sub(r"#.*", "", cleaned)
    cleaned = cleaned.strip()
    if not cleaned:
        return True
    if re.search(r"(?m)^\s*pass\s*$", cleaned):
        return True
    if re.search(r"\bthrow\s+new\s+Error\s*\(\s*['\"](?:TODO|not implemented|stub)", cleaned, re.IGNORECASE):
        return True
    if re.search(r"\breturn\s+(0|NULL|nullptr|None|false)\s*;?\s*$", cleaned, re.IGNORECASE):
        # Only treat as stub if body is tiny.
        code_lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
        if len(code_lines) <= 3:
            return True
    # Empty brace body.
    if re.fullmatch(r"\{\s*\}", cleaned, flags=re.DOTALL):
        return True
    return False


def _extract_preceding_comment(lines: List[str], start_line: int, max_lookback: int = 10) -> str:
    if not lines or start_line <= 1:
        return ""
    out: List[str] = []
    i = start_line - 1
    scanned = 0
    while i > 0 and scanned < max_lookback:
        ln = lines[i - 1]
        scanned += 1
        s = ln.strip()
        if not s:
            if out:
                break
            i -= 1
            continue
        if s.startswith("//") or s.startswith("/*") or s.startswith("*") or s.endswith("*/"):
            out.append(s)
            i -= 1
            continue
        break
    if not out:
        return ""
    text = "\n".join(reversed(out))
    text = re.sub(r"/\*+|\*/", " ", text)
    text = re.sub(r"^\s*//\s?", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\*\s?", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalize_requirement(text: str) -> str:
    s = re.sub(r"\s+", " ", (text or "")).strip(" .-")
    if not s:
        return ""
    s = re.sub(r"^(task\s*\d+\s*:?)", "", s, flags=re.IGNORECASE).strip()
    if not s:
        return ""
    return s[0].upper() + s[1:]


def _iter_section_chunks(user_text: str) -> List[Tuple[str, str]]:
    lines = (user_text or "").splitlines()
    section = "general"
    out: List[Tuple[str, str]] = []

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        if SECTION_TASK_RE.match(line):
            section = "task"
            continue
        if SECTION_NOTES_RE.match(line):
            section = "notes"
            continue
        if SECTION_HINTS_RE.match(line):
            section = "hints"
            continue

        m_inline = SECTION_INLINE_RE.match(line)
        if m_inline:
            hdr = m_inline.group(1).lower()
            if hdr.startswith("task"):
                section = "task"
            elif hdr.startswith("note"):
                section = "notes"
            elif hdr.startswith("hint"):
                section = "hints"
            else:
                section = "general"
            line = m_inline.group(2).strip()
            if not line:
                continue

        # Keep label lines intact for output-contract detection/scoring.
        if OUTPUT_LABEL_LINE_RE.match(line):
            out.append((line, section))
            continue

        parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", line) if p.strip()]
        if not parts:
            continue
        for part in parts:
            out.append((part, section))

    return out


def _extract_output_contract_requirements(user_text: str) -> List[str]:
    lines = (user_text or "").splitlines()
    reqs: List[str] = []
    seen = set()
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].strip()
        if not OUTPUT_CONTRACT_TRIGGER_RE.search(line):
            i += 1
            continue

        labels: List[str] = []
        j = i + 1
        started = False
        while j < n:
            raw = lines[j]
            stripped = raw.strip()
            if not stripped:
                if started:
                    break
                j += 1
                continue
            m_label = OUTPUT_LABEL_LINE_RE.match(stripped)
            if m_label:
                started = True
                label = m_label.group(1).strip()
                if label:
                    labels.append(label)
                j += 1
                continue
            # Allow indented continuation lines after block start.
            if started and raw.startswith((" ", "\t")):
                j += 1
                continue
            if started:
                break
            # A non-label, non-empty line before block start means no block here.
            break

        if labels:
            uniq = []
            local_seen = set()
            for lbl in labels:
                key = lbl.lower()
                if key in local_seen:
                    continue
                local_seen.add(key)
                uniq.append(lbl)
            if uniq:
                req = "Must-have requirement: Match output labels exactly (including punctuation/spaces): " + ", ".join(
                    f"{lbl}:" for lbl in uniq
                )
                key = req.lower()
                if key not in seen:
                    seen.add(key)
                    reqs.append(req)
                req2 = "Must-have requirement: Preserve expected output label strings exactly (including punctuation/spaces)."
                key2 = req2.lower()
                if key2 not in seen:
                    seen.add(key2)
                    reqs.append(req2)
        i = j if j > i else i + 1
    return reqs


def _extract_dependency_requirements(user_text: str) -> List[str]:
    reqs: List[str] = []
    seen = set()
    chunks = [c.strip() for c in re.split(r"(?<=[.!?])\s+|\n+", (user_text or "")) if c.strip()]
    for chunk in chunks:
        low = chunk.lower()
        has_dep = bool(DEPENDENCY_TRIGGER_RE.search(low) or OFFSET_SIGNAL_RE.search(low))
        has_field = bool(FIELD_IDENTIFIER_RE.search(chunk) or HEADER_REF_RE.search(chunk))
        if not (has_dep or has_field):
            continue
        fields = []
        for tok in FIELD_IDENTIFIER_RE.findall(chunk):
            if tok not in fields:
                fields.append(tok)
        named_fields = ", ".join(fields[:4])
        if has_dep and has_field:
            if named_fields:
                req = (
                    f"Use referenced offset/header field identifiers ({named_fields}) "
                    "to locate or derive required data; do not assume fixed positions."
                )
            else:
                req = "Use referenced header/field identifiers to locate or derive required data (do not assume fixed positions)."
        elif has_field:
            if named_fields:
                req = f"Respect prompt-referenced header/field identifiers ({named_fields}) when locating or deriving data."
            else:
                req = "Respect prompt-referenced header/field identifiers when locating or deriving data."
        else:
            req = "Do not assume fixed data positions; follow prompt-stated locator/dependency instructions."
        key = req.lower()
        if key in seen:
            continue
        seen.add(key)
        reqs.append(req)
    # Global hard dependency guard for prompts that explicitly warn start positions vary.
    global_low = (user_text or "").lower()
    if re.search(r"\b(no guarantee|do not assume|never assume)\b", global_low) and re.search(
        r"\b(starts?\s+at|same position|offset|locate)\b", global_low
    ):
        req = "Use prompt-specified offset/header fields to locate target data sections; do not assume fixed start positions."
        key = req.lower()
        if key not in seen:
            seen.add(key)
            reqs.insert(0, req)
    return reqs


def _extract_format_definition_requirements(user_text: str) -> List[str]:
    low = (user_text or "").lower()
    reqs: List[str] = []
    if FORMAT_DEFINITION_SIGNAL_RE.search(low) or CHANNEL_ORDER_RE.search(low):
        reqs.append("Preserve stated format layout rules (ordering, record/row traversal, and directional semantics).")
    if re.search(r"\b(padding|aligned|multiple of 4)\b", low):
        reqs.append("Handle stated alignment/padding constraints from format definitions/notes.")
    if re.search(r"\b(exactly that order|in exactly that order)\b", low):
        reqs.append("Preserve component/channel ordering exactly as specified.")
    return reqs[:3]


def _score_semantic_chunk(chunk: str, symbol_names: Sequence[str], section: str = "general") -> Tuple[int, List[str]]:
    low = chunk.lower()
    score = 0
    reasons: List[str] = []

    if len(chunk) < 20:
        return 0, ["too_short"]

    section_weights = {"task": 5, "notes": 4, "hints": 3}
    if section in section_weights:
        w = section_weights[section]
        score += w
        reasons.append(f"section_{section}:+{w}")

    if TASK_HEADER_RE.search(chunk):
        score += 4
        reasons.append("task_header:+4")
    if MODAL_RE.search(low):
        score += 3
        reasons.append("modal:+3")
    if SEMANTIC_VERB_RE.search(low):
        score += 3
        reasons.append("semantic_verb:+3")
    if OUTPUT_SIGNAL_RE.search(low):
        score += 3
        reasons.append("output_signal:+3")
    if COMMAND_SIGNAL_RE.search(low):
        score += 2
        reasons.append("command_signal:+2")
    if OUTPUT_CONTRACT_TRIGGER_RE.search(low):
        score += 4
        reasons.append("output_contract_signal:+4")
    if OUTPUT_LABEL_LINE_RE.match(chunk):
        score += 6
        reasons.append("output_label_line:+6")
    if CONSTRAINT_SIGNAL_RE.search(low):
        score += 3
        reasons.append("constraint_signal:+3")
    if STRICTNESS_SIGNAL_RE.search(low):
        score += 4
        reasons.append("strictness_signal:+4")
    if EXACTNESS_SIGNAL_RE.search(low):
        score += 3
        reasons.append("exactness_signal:+3")
    if FIELD_IDENTIFIER_RE.search(chunk):
        add = 2
        if DEPENDENCY_TRIGGER_RE.search(low) or OFFSET_SIGNAL_RE.search(low):
            add = 6
        score += add
        reasons.append(f"field_identifier:+{add}")
    if OFFSET_SIGNAL_RE.search(low) and HEADER_REF_RE.search(chunk):
        score += 4
        reasons.append("offset_header_signal:+4")
    if ALIGN_SIGNAL_RE.search(low):
        score += 2
        reasons.append("alignment_signal:+2")
    # Partial channel-order credit even when strict phrase doesn't match.
    if CHANNEL_ORDER_RE.search(low):
        score += 3
        reasons.append("channel_order:+3")
    elif ("blue" in low and "green" in low and "red" in low):
        score += 2
        reasons.append("channel_order_partial:+2")
    if FORMAT_SIGNAL_RE.search(low):
        score += 2
        reasons.append("format_signal:+2")
    # Partial credit by number of metadata fields present.
    meta_hits = len(METADATA_FIELDS_RE.findall(low))
    if meta_hits:
        add = min(4, meta_hits)
        score += add
        reasons.append(f"metadata_fields:+{add}")
    # Accept both ASCII '-' and Unicode '−' formula styles.
    if FORMULA_RE.search(chunk) or UNICODE_FORMULA_RE.search(chunk):
        score += 4
        reasons.append("formula:+4")
    if NUMERIC_SPEC_RE.search(low):
        score += 2
        reasons.append("numeric_spec:+2")
    if ONLY_SHOULD_RE.search(low):
        score += 2
        reasons.append("only_should:+2")
    if AT_MOST_RE.search(low):
        score += 2
        reasons.append("at_most:+2")
    if ORDER_RE.search(low):
        score += 2
        reasons.append("ordering:+2")
    if EACH_RE.search(low):
        score += 1
        reasons.append("each:+1")
    if re.search(r"\b(not changed|unchanged|without changing|do not change|instead of)\b", low):
        score += 2
        reasons.append("preserve_original:+2")
    if ("output should" in low) or ("prints the following" in low):
        score += 3
        reasons.append("output_format_signal:+3")

    if symbol_names:
        matched_syms = [name for name in symbol_names if re.search(rf"\b{re.escape(name)}\b", low)]
        if matched_syms:
            score += min(4, 2 + len(matched_syms))
            reasons.append(f"symbol_match:+{min(4, 2 + len(matched_syms))}")

    code_ident_hits = [tok for tok in CODE_IDENT_RE.findall(chunk) if "_" in tok or any(c.isdigit() for c in tok)]
    if code_ident_hits:
        score += 1
        reasons.append("code_ident:+1")

    if SEMANTIC_NOISE_RE.search(low):
        score -= 3
        reasons.append("semantic_noise:-3")
    if BACKGROUND_NOISE_RE.search(low):
        score -= 3
        reasons.append("background_noise:-3")
    if EDU_NOISE_RE.search(low) and not TASK_HEADER_RE.search(chunk):
        score -= 2
        reasons.append("edu_noise:-2")
    if ADMIN_NOISE_RE.search(low) and not (OUTPUT_SIGNAL_RE.search(low) or "original file" in low):
        score -= 1
        reasons.append("admin_noise:-1")
    if ELF_CONTEXT_RE.search(low):
        score -= 5
        reasons.append("elf_context_noise:-5")
    if "for example" in low and not (OUTPUT_SIGNAL_RE.search(low) or MODAL_RE.search(low) or TASK_HEADER_RE.search(chunk)):
        score -= 2
        reasons.append("example_noise:-2")
    if len(chunk) > 260 and not (FORMULA_RE.search(chunk) or TASK_HEADER_RE.search(chunk)):
        score -= 2
        reasons.append("very_long_chunk:-2")

    return score, reasons


def _has_strong_semantic_signal(chunk: str) -> bool:
    low = chunk.lower()
    return bool(
        HEADER_REF_RE.search(chunk)
        or ALIGN_SIGNAL_RE.search(low)
        or CHANNEL_ORDER_RE.search(low)
        or ("blue" in low and "green" in low and "red" in low)
        or FORMULA_RE.search(chunk)
        or UNICODE_FORMULA_RE.search(chunk)
        or re.search(r"\b(do not|unchanged|without changing|not changed)\b", low)
        or ("output should" in low)
        or ("prints the following" in low)
        or len(METADATA_FIELDS_RE.findall(low)) >= 2
        or OUTPUT_SIGNAL_RE.search(low)
        or STRICTNESS_SIGNAL_RE.search(low)
        or OUTPUT_LABEL_LINE_RE.match(chunk) is not None
        or FIELD_IDENTIFIER_RE.search(chunk) is not None
        or EXACTNESS_SIGNAL_RE.search(low) is not None
    )


def _extract_prompt_semantics(user_text: str, symbol_names: Sequence[str]) -> List[str]:
    reqs: List[str] = []
    seen = set()
    text = (user_text or "").strip()
    if not text:
        return reqs

    # Force-include high-value extraction channels first.
    forced_seed: List[str] = []
    forced_seed.extend(_extract_output_contract_requirements(text))
    forced_seed.extend(_extract_dependency_requirements(text))
    forced_seed.extend(_extract_format_definition_requirements(text))
    for req in forced_seed:
        normalized = _normalize_requirement(req)
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        reqs.append(normalized)
        if len(reqs) >= SEMANTIC_SELECTED_LIMIT:
            return reqs

    # Section-aware sentence-ish chunks from prose and line breaks.
    chunks = _iter_section_chunks(text)
    scored: List[Tuple[int, str, str, List[str]]] = []
    for chunk, section in chunks:
        score, reasons = _score_semantic_chunk(chunk, symbol_names, section=section)
        if score < SEMANTIC_SCORE_THRESHOLD and not (
            score >= SEMANTIC_BORDERLINE_THRESHOLD and _has_strong_semantic_signal(chunk)
        ):
            continue
        scored.append((score, chunk, section, reasons))

    ranked = sorted(scored, key=lambda x: (-x[0], len(x[1])))
    if ranked:
        def _fmt_semantic_candidate(score: int, chunk: str, section: str, reasons: Sequence[str]) -> str:
            cleaned = re.sub(r"\s+", " ", chunk).strip()
            reason_text = ", ".join(reasons[:8])
            return f"[score={score} sec={section}] {cleaned[:220]} || {reason_text}"

        dbg_dump(
            "task_card_semantic_candidates",
            "\n".join(
                _fmt_semantic_candidate(score, chunk, section, reasons)
                for score, chunk, section, reasons in ranked
            ),
        )
        dbg_dump(
            "task_card_semantic_scoring",
            "\n".join(
                f"score={score} reasons={'; '.join(reasons)}"
                for score, _chunk, _section, reasons in ranked[:SEMANTIC_CANDIDATE_SCAN_LIMIT]
            ),
        )

    # Force-include must-have requirements from high-signal candidates first.
    forced: List[str] = []
    forced_seen = set()

    def _add_forced(text_req: str) -> None:
        req = _normalize_requirement(text_req)
        if not req:
            return
        key = req.lower()
        if key in forced_seen:
            return
        forced_seen.add(key)
        forced.append(req)

    for score, chunk, _section, reasons in ranked[:SEMANTIC_CANDIDATE_SCAN_LIMIT]:
        if score < SEMANTIC_MUST_COPY_SCORE_THRESHOLD:
            continue
        has_field_or_offset = any(
            r.startswith("field_identifier:+") or r.startswith("offset_header_signal:+")
            for r in reasons
        )
        strict_and_relevant = (
            STRICTNESS_SIGNAL_RE.search(chunk)
            and (
                CONSTRAINT_SIGNAL_RE.search(chunk.lower())
                or OUTPUT_LABEL_LINE_RE.match(chunk)
                or FORMAT_SIGNAL_RE.search(chunk.lower())
                or ALIGN_SIGNAL_RE.search(chunk.lower())
                or OFFSET_SIGNAL_RE.search(chunk.lower())
            )
        )
        if MUST_HAVE_SIGNAL_RE.search(chunk) or strict_and_relevant or has_field_or_offset:
            forced_line = _normalize_requirement(chunk)
            _add_forced(f"Must-have requirement: {forced_line}")

    if forced:
        dbg_dump("task_card_semantic_forced", "\n".join(forced))
        for req in forced:
            key = req.lower()
            if key in seen:
                continue
            seen.add(key)
            reqs.append(req)
            if len(reqs) >= SEMANTIC_SELECTED_LIMIT:
                return reqs

    for _score, chunk, _section, _reasons in ranked[:SEMANTIC_CANDIDATE_SCAN_LIMIT]:
        low = chunk.lower()
        requirement = ""
        m_formula = FORMULA_RE.search(chunk) or UNICODE_FORMULA_RE.search(chunk)
        if m_formula:
            requirement = f"Apply required formula from prompt (e.g. `{m_formula.group(0)}`)"
        elif OFFSET_SIGNAL_RE.search(low) and HEADER_REF_RE.search(chunk):
            requirement = "Use header-defined data offset to locate the image data region"
        elif ALIGN_SIGNAL_RE.search(low):
            requirement = "Handle row alignment/padding requirements from the format"
        elif CHANNEL_ORDER_RE.search(low):
            requirement = "Preserve required pixel channel ordering from the format"
        elif FORMAT_SIGNAL_RE.search(low):
            requirement = "Respect stated format constraints (e.g., bit depth/compression)"
        elif len(METADATA_FIELDS_RE.findall(low)) >= 3:
            requirement = "Print all required metadata fields from the prompt"
        elif ONLY_SHOULD_RE.search(low):
            requirement = _normalize_requirement(chunk)
        elif AT_MOST_RE.search(low) and SEMANTIC_VERB_RE.search(low):
            requirement = _normalize_requirement(chunk)
        elif ORDER_RE.search(low) and (re.search(r"\b(ensure|wait|terminate|exit|finish)\b", low) or MODAL_RE.search(low)):
            requirement = _normalize_requirement(chunk)
        elif EACH_RE.search(low) and re.search(r"\b(wait|call|create|spawn|fork)\b", low):
            requirement = _normalize_requirement(chunk)
        elif re.search(r"\b(not changed|unchanged|without changing|do not change|instead of)\b", low):
            requirement = "Do not mutate original input; preserve source while producing requested result"
        elif "print" in low and ("following" in low or "output should" in low or "prints the following" in low):
            requirement = "Match required output fields/format from prompt"
        elif MODAL_RE.search(low) and SEMANTIC_VERB_RE.search(low):
            # Keep concise and generic; do not copy long assignment prose.
            cleaned = _normalize_requirement(chunk)
            if len(cleaned) > 120:
                cleaned = cleaned[:117].rstrip() + "..."
            requirement = f"Follow prompt requirement: {cleaned}"

        requirement = _normalize_requirement(requirement)
        if not requirement:
            continue
        key = requirement.lower()
        if key in seen:
            continue
        seen.add(key)
        reqs.append(requirement)
        if len(reqs) >= SEMANTIC_SELECTED_LIMIT:
            break

    return reqs


def _extract_output_target(user_text: str) -> str:
    text = user_text or ""
    candidates: List[tuple[int, int, str]] = []
    for m in OUTPUT_FILE_RE.finditer(text):
        raw = m.group(1).strip("`'\"")
        if "." not in raw:
            continue
        stem_raw, ext_raw = raw.rsplit(".", 1)
        stem = stem_raw.lower()
        ext = ext_raw.lower()
        if not stem or not ext:
            continue
        # Skip sentence artifacts like "code.Create" and unsupported extensions.
        if ext_raw != ext:
            continue
        if ext not in OUTPUT_EXT_ALLOW:
            continue
        # Skip doc asset references unless explicitly output-like.
        if ext in {"png", "jpg", "jpeg", "gif"} and "output" not in stem:
            continue

        context = text[max(0, m.start() - 90): min(len(text), m.end() + 90)]
        score = 0
        if OUTPUT_SIGNAL_RE.search(context):
            score += 3
        if "output" in stem:
            score += 3
        if score <= 0:
            continue
        candidates.append((score, m.start(), raw))

    if not candidates:
        return ""
    candidates.sort(key=lambda x: (-x[0], x[1]))
    return candidates[0][2]


def _extract_semantic_requirements(user_text: str, file_content: str, spans: List[SymbolSpan]) -> List[str]:
    reqs: List[str] = []
    seen = set()
    text = user_text or ""
    low = text.lower()

    def _add(req: str) -> None:
        r = _normalize_requirement(req)
        if not r:
            return
        key = r.lower()
        if key in seen:
            return
        seen.add(key)
        reqs.append(r)

    # High-signal semantic cues from request text.
    out_name = _extract_output_target(text)
    if out_name:
        _add(f"Write transformed result to `{out_name}` and keep input file unchanged")
    symbol_names = [span.name.lower() for span in spans[:30]]
    for req in _extract_prompt_semantics(text, symbol_names):
        _add(req)
    if re.search(r"\bdo not\b.*\bchange\b.*\boriginal\b", low):
        _add("Do not mutate original input data in-place")

    # Fallback from nearby function comments when request text is noisy.
    lines = file_content.splitlines()
    for span in spans[:10]:
        comment = _extract_preceding_comment(lines, span.start)
        if not comment:
            continue
        if SEMANTIC_NOISE_RE.search(comment):
            continue
        if not SEMANTIC_VERB_RE.search(comment):
            continue
        # Keep first sentence only to stay concise.
        first = re.split(r"(?<=[.!?])\s+", comment, maxsplit=1)[0]
        first = re.sub(r"\s+", " ", first).strip()
        if len(first) < 18:
            continue
        _add(f"For `{span.name}`: {first}")

    return reqs[:SEMANTIC_SELECTED_LIMIT]


def _build_constraints(ext: str, semantic_requirements: List[str]) -> List[str]:
    out = [
        "Preserve existing APIs and function signatures.",
        "Do not create new files unless explicitly requested.",
        "Keep unrelated code unchanged.",
    ]
    if semantic_requirements:
        for req in semantic_requirements:
            if len(out) >= CONSTRAINTS_LIMIT:
                break
            line = f"Semantic: {req}."
            if line not in out:
                out.append(line)
    else:
        out.append("Never assume file/stream cursor position; set/restore it explicitly.")
    if ext in {"c", "h", "cc", "cpp", "cxx", "hpp", "hh", "hxx"} and len(out) < CONSTRAINTS_LIMIT:
        out.append("Result should pass C/C++ syntax checks.")
    elif ext == "py" and len(out) < CONSTRAINTS_LIMIT:
        out.append("Result should pass Python syntax checks.")
    return out[:CONSTRAINTS_LIMIT]


def build_task_card(target_file: str, user_text: str, file_content: str, max_lines: int = TASK_CARD_MAX_LINES) -> str:
    """Build concise task card from file + request without assignment restatement."""
    target_file = str(target_file)
    text = user_text or ""
    lines = file_content.splitlines()
    spans = _extract_symbol_spans(target_file, file_content)

    work_items: List[str] = []
    seen = set()
    todo_symbols = set()
    core_work_item_count = 0

    # 1) TODO/FIXME/TBD markers mapped to symbol/line.
    for idx, ln in enumerate(lines, start=1):
        if not TODO_RE.search(ln):
            continue
        span = _find_symbol_for_line(idx, spans)
        if span:
            item = f"Implement TODO in `{span.name}` ({span.start}-{span.end})."
            todo_symbols.add(span.name)
        else:
            item = f"Resolve TODO at line {idx}."
        if item not in seen:
            seen.add(item)
            work_items.append(item)

    # 2) Stub-like functions/classes.
    for span in spans:
        if span.name in todo_symbols:
            continue
        block = "\n".join(lines[span.start - 1:span.end])
        if _is_stub_block(block):
            item = f"Replace stub logic in `{span.name}` ({span.start}-{span.end})."
            if item not in seen:
                seen.add(item)
                work_items.append(item)

    # 3) Explicit function mentions in request.
    low_text = text.lower()
    if ACTION_RE.search(low_text):
        for span in spans[:20]:
            if span.name in todo_symbols:
                continue
            if re.search(rf"\b{re.escape(span.name.lower())}\b", low_text):
                item = f"Update `{span.name}` per user request."
                if item not in seen:
                    seen.add(item)
                    work_items.append(item)

    if not work_items:
        if spans:
            top = spans[0]
            work_items = [f"Implement missing logic in `{top.name}` ({top.start}-{top.end})."]
        else:
            work_items = ["Implement missing logic in target file."]
    core_work_item_count = len(work_items)

    semantic_requirements = _extract_semantic_requirements(text, file_content, spans)
    # Promote semantic requirements to actionable work items (without adding noise).
    for req in semantic_requirements:
        if len(work_items) >= WORK_ITEMS_LIMIT:
            break
        item = f"Satisfy semantic requirement: {req}."
        if item not in seen:
            seen.add(item)
            work_items.append(item)

    constraints = _build_constraints(
        Path(target_file).suffix.lower().lstrip("."),
        semantic_requirements,
    )

    hazards: List[str] = []
    if MILESTONE_RE.search(text):
        hazards.append("Milestone scope is ambiguous; confirm exact required functions.")
    if re.search(r"\b(commit|push|submit|what to submit)\b", text, re.IGNORECASE):
        hazards.append("Request includes submission steps; prioritize code changes only.")
    if not TODO_RE.search(file_content) and len(work_items) <= 1:
        hazards.append("No explicit TODO markers found; inferred work may be incomplete.")
    if not hazards:
        hazards.append("None obvious from current file and request.")

    # Keep card concise.
    work_items = work_items[:WORK_ITEMS_LIMIT]
    constraints = constraints[:CONSTRAINTS_LIMIT]
    hazards = hazards[:HAZARDS_LIMIT]

    def _render() -> List[str]:
        out_lines_local: List[str] = [
            "TASK_CARD",
            f"Target file: {target_file}",
            "",
            "Work items:",
        ]
        out_lines_local.extend([f"- {w}" for w in work_items])
        out_lines_local.append("Constraints:")
        out_lines_local.extend([f"- {c}" for c in constraints])
        out_lines_local.append("Hazards:")
        out_lines_local.extend([f"- {h}" for h in hazards])
        return out_lines_local

    out_lines = _render()

    # Enforce line budget while preserving core TODO/stub work items.
    # Trim least-critical sections first.
    while len(out_lines) > max_lines and len(constraints) > 3:
        constraints.pop()
        out_lines = _render()

    while len(out_lines) > max_lines and len(hazards) > 1:
        hazards.pop()
        out_lines = _render()

    # Only trim non-core work items (typically semantic extras).
    min_keep = max(1, min(core_work_item_count, len(work_items)))
    while len(out_lines) > max_lines and len(work_items) > min_keep:
        work_items.pop()
        out_lines = _render()

    return "\n".join(out_lines)
