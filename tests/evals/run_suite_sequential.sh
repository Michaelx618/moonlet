#!/bin/bash
# Run eval tasks one at a time. Each task runs in a new process; process exits when done.
# Rule: No multiple model processes. One task -> one process -> exit.
# Kills any stale eval process before starting a new one. Hard-kills on timeout.
#
# Usage:
#   ./tests/evals/run_suite_sequential.sh [task_file] [task_indices...]
#   ./tests/evals/run_suite_sequential.sh tests/evals/plan_tasks.json
#   ./tests/evals/run_suite_sequential.sh tests/evals/agent_tasks.json 1 2 3

set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Use venv Python if present (has mlx/mlx-lm)
PYTHON="python3"
if [ -f "$REPO_ROOT/.venv/bin/python" ]; then
  PYTHON="$REPO_ROOT/.venv/bin/python"
fi

# Per-task timeout (seconds). Hard kill after this to prevent runaway/infinite loops.
TASK_TIMEOUT="${EVAL_TASK_TIMEOUT:-600}"

# Resolve timeout command. Required for safe testing (prevents runaway model processes).
# macOS: brew install coreutils -> gtimeout
# Linux: timeout (usually preinstalled)
export PATH="/opt/homebrew/bin:/opt/local/bin:$PATH"
TIMEOUT_CMD=""
if command -v timeout &>/dev/null; then
  TIMEOUT_CMD="timeout -k 5 $TASK_TIMEOUT"
elif command -v gtimeout &>/dev/null; then
  TIMEOUT_CMD="gtimeout -k 5 $TASK_TIMEOUT"
else
  echo "[run_suite_sequential] ERROR: timeout command required. Install with: brew install coreutils" >&2
  exit 1
fi

# Kill any stale eval processes before starting. Prevents multiple model processes.
kill_stale_evals() {
  local killed=0
  if pkill -9 -f "run_single_task\.py" 2>/dev/null; then killed=1; fi
  if pkill -9 -f "eval_harness" 2>/dev/null; then killed=1; fi
  if [ "$killed" -eq 1 ]; then
    echo "[run_suite_sequential] Killed stale eval process(es)" >&2
    sleep 2
  fi
}

TASK_FILE="${1:-tests/evals/plan_tasks.json}"
shift || true

# Resolve task file path
if [[ "$TASK_FILE" != /* ]]; then
  TASK_FILE="$REPO_ROOT/$TASK_FILE"
fi

# Count tasks in JSON
count_tasks() {
  $PYTHON -c "
import json
from pathlib import Path
p = Path('$TASK_FILE')
data = json.loads(p.read_text())
tasks = data.get('tasks', data) if isinstance(data, dict) else data
print(len(tasks))
"
}

TOTAL=$(count_tasks)
echo "[run_suite_sequential] Task file: $TASK_FILE, total tasks: $TOTAL, timeout: ${TASK_TIMEOUT}s"
echo ""

if [ $# -eq 0 ]; then
  INDICES=$(seq 1 "$TOTAL")
else
  INDICES="$*"
fi

# Use path relative to repo root for harness
TASK_FILE_REL="${TASK_FILE#$REPO_ROOT/}"
if [[ "$TASK_FILE_REL" == "$TASK_FILE" ]]; then
  TASK_FILE_REL="$TASK_FILE"
fi

PASSED=0
FAILED=0
for i in $INDICES; do
  kill_stale_evals
  echo "=== Task $i: $TASK_FILE_REL ==="
  if $TIMEOUT_CMD $PYTHON tests/evals/run_single_task.py "$TASK_FILE_REL" "$i"; then
    PASSED=$((PASSED + 1))
  else
    EXIT=$?
    if [ "$EXIT" -eq 124 ] || [ "$EXIT" -eq 137 ]; then
      echo "[run_suite_sequential] Task $i timed out or was killed (exit $EXIT)" >&2
    fi
    FAILED=$((FAILED + 1))
  fi
  echo ""
done

echo "[run_suite_sequential] Done. Passed: $PASSED, Failed: $FAILED"
[ $FAILED -eq 0 ]
