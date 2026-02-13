#!/bin/bash
# Double-click this file to launch Moonlet
export PATH="/usr/local/bin:/opt/homebrew/bin:$PATH"
cd "$(dirname "$0")"

# Log every launch for easier debugging.
LOG_FILE="$PWD/launch.log"
RUNTIME_DEBUG_FILE="$PWD/runtime-debug.log"
if [ -f "$RUNTIME_DEBUG_FILE" ]; then
  cp "$RUNTIME_DEBUG_FILE" "$PWD/runtime-debug.prev.log" 2>/dev/null || true
fi
: > "$RUNTIME_DEBUG_FILE"
{
  echo "=== launch $(date) ==="
  echo "arch: $(uname -m)"
  echo "python3: $(command -v python3) $(python3 -c 'import sys; print(sys.platform)' 2>/dev/null)"
} >> "$LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

# Activate the Python venv
source .venv/bin/activate 2>/dev/null

# Start the Electron app (which spawns the Python server)
cd local-app
# Ensure Electron runs in browser mode, not Node mode.
unset ELECTRON_RUN_AS_NODE
npm start
