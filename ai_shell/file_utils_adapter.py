"""Single adapter for file_utils (generate_diff, is_security_concern, validate_file_path).

Import from this module only; avoids repeated try/except across agent_loop, server, search_replace, tool_executor, files.
"""

import difflib
from typing import Any

try:
    from file_utils import (
        generate_diff,
        is_security_concern,
        validate_file_path,
    )
except ImportError:
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

    def is_security_concern(*args: Any, **kwargs: Any) -> bool:
        return False

    def validate_file_path(*args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("file_utils not available")
