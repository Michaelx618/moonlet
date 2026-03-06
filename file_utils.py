"""File editing utilities."""
import difflib
from pathlib import Path


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
