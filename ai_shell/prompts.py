"""Prompt helpers used by runtime paths."""

from pathlib import Path


def _ext(path: str) -> str:
    return Path(path).suffix.lower().lstrip(".")


def _language_name_for_ext(ext: str) -> str:
    return {
        "py": "Python",
        "js": "JavaScript",
        "ts": "TypeScript",
        "tsx": "TypeScript/TSX",
        "jsx": "JavaScript/JSX",
        "java": "Java",
        "kt": "Kotlin",
        "go": "Go",
        "rs": "Rust",
        "c": "C",
        "cpp": "C++",
        "cc": "C++",
        "h": "C/C++ header",
        "hpp": "C++ header",
        "json": "JSON",
        "yaml": "YAML",
        "yml": "YAML",
        "md": "Markdown",
        "sh": "Shell",
        "zsh": "Shell",
        "css": "CSS",
    }.get(ext, ext.upper() or "Text")
