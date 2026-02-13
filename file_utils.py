"""File editing utilities ported from Continue codebase."""
import difflib
from pathlib import Path
from typing import List, Tuple, Optional


def generate_diff(
    old_content: str,
    new_content: str,
    filepath: str,
    context_lines: int = 3,
) -> str:
    """Generate unified diff between old and new content.
    
    Ported from extensions/cli/src/tools/writeFile.ts
    """
    old_lines = old_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)
    
    diff = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=filepath,
        tofile=filepath,
        lineterm="",
        n=context_lines,
    )
    return "".join(diff)


def find_search_matches(content: str, search_string: str) -> List[Tuple[int, int]]:
    """Find all occurrences of search_string in content.
    
    Returns list of (start_index, end_index) tuples.
    Ported from core/edit/searchAndReplace/findSearchMatch.ts
    """
    matches = []
    start = 0
    while True:
        idx = content.find(search_string, start)
        if idx == -1:
            break
        matches.append((idx, idx + len(search_string)))
        start = idx + 1
    return matches


def execute_find_and_replace(
    file_content: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
    edit_index: int = 0,
) -> str:
    """Execute find and replace operation with validation.
    
    Ported from core/edit/searchAndReplace/performReplace.ts
    """
    matches = find_search_matches(file_content, old_string)
    
    if not matches:
        raise ValueError(
            f"Edit at index {edit_index}: string not found in file: {old_string!r}"
        )
    
    if replace_all:
        # Apply replacements in reverse order to maintain correct positions
        result = file_content
        for start_idx, end_idx in reversed(matches):
            result = result[:start_idx] + new_string + result[end_idx:]
        return result
    else:
        # For single replacement, check for multiple matches first
        if len(matches) > 1:
            raise ValueError(
                f"Edit at index {edit_index}: String {old_string!r} appears "
                f"{len(matches)} times in the file. Either provide a more specific "
                "string with surrounding context to make it unique, or use "
                "replace_all=True to replace all occurrences."
            )
        
        # Apply single replacement
        start_idx, end_idx = matches[0]
        return (
            file_content[:start_idx] + new_string + file_content[end_idx:]
        )


def execute_multi_find_and_replace(
    file_content: str,
    edits: List[dict],
) -> str:
    """Execute multiple find and replace operations sequentially.
    
    Ported from core/edit/searchAndReplace/performReplace.ts
    
    Args:
        file_content: Original file content
        edits: List of dicts with 'old_string', 'new_string', and optional 'replace_all'
    
    Returns:
        Modified file content
    """
    result = file_content
    
    for edit_index, edit in enumerate(edits):
        old_string = edit.get("old_string", "")
        new_string = edit.get("new_string", "")
        replace_all = edit.get("replace_all", False)
        
        result = execute_find_and_replace(
            result,
            old_string,
            new_string,
            replace_all,
            edit_index,
        )
    
    return result


def validate_file_path(
    file_path: str,
    root_path: Path,
    must_exist: bool = False,
) -> Path:
    """Validate and resolve file path relative to root.
    
    Ported from extensions/cli/src/tools/edit.ts validateAndResolveFilePath
    
    Args:
        file_path: Path to validate (can be absolute or relative)
        root_path: Root directory for relative paths
        must_exist: If True, file must exist
    
    Returns:
        Resolved Path object
    
    Raises:
        ValueError: If path is invalid or outside root
    """
    if not file_path:
        raise ValueError("file_path is required")
    
    # Resolve path
    if Path(file_path).is_absolute():
        resolved = Path(file_path).resolve()
    else:
        resolved = (root_path / file_path).resolve()
    
    # Ensure it's within root
    try:
        resolved.relative_to(root_path.resolve())
    except ValueError:
        raise ValueError(f"Path {file_path} is outside root directory")
    
    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"File {file_path} does not exist")
    
    return resolved


def is_security_concern(filepath: Path) -> bool:
    """Check if file path is a security concern (system files, etc.).
    
    Ported from core/indexing/ignore.ts throwIfFileIsSecurityConcern
    """
    path_str = str(filepath.resolve())
    
    # Common system directories to avoid
    security_paths = [
        "/etc/",
        "/usr/bin/",
        "/usr/sbin/",
        "/bin/",
        "/sbin/",
        "/System/",
        "/Library/Frameworks/",
        "/Applications/",
        "/var/",
        "/private/",
    ]
    
    # Check if path contains security-sensitive directories
    for sec_path in security_paths:
        if sec_path in path_str:
            return True
    
    # Check for hidden system files
    parts = filepath.parts
    for part in parts:
        if part.startswith(".") and part not in [".", "..", ".git", ".venv"]:
            # Allow common project hidden dirs
            continue
    
    return False
