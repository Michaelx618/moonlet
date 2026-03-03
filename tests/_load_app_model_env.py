"""
Load environment from local-app/config.json so tests see the same config as the app.

The Electron app (main.js) sets SC2_* and other env vars from config.json when spawning
the Python server. This module does the same when running tests: so temp, top_p, max_new,
model path, debug, root, etc. match the user's environment.

Import and call load_app_model_env() at the top of test modules that run agent/ask/plan,
before importing ai_shell. Tests that need a different workspace (e.g. fixtures) still
call set_root() in setUp; SC2_ROOT from config is applied first, then set_root() overrides
the live root for that test.
"""

import json
import os
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_CFG_PATH = _REPO / "local-app" / "config.json"


def _path_prepend_for_server(existing_path: str = None) -> str:
    """Match main.js buildServerPathEnv: prepend common bins to PATH."""
    existing = (existing_path or os.environ.get("PATH") or "").strip()
    parts = [
        "/opt/homebrew/bin",
        "/usr/local/bin",
        "/usr/bin",
        "/bin",
        "/usr/sbin",
        "/sbin",
        *[p for p in existing.split(":") if p],
    ]
    seen = set()
    dedup = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            dedup.append(p)
    return ":".join(dedup)


def load_app_model_env(reduce_debug_for_tests: bool = False):
    """Set SC2_* and related env from local-app/config.json so tests match the app. Skips if already set.

    If reduce_debug_for_tests is True, debug logging is disabled (SC2_DEBUG, SC2_DEBUG_KV,
    SC2_DEBUG_LOG) to avoid extra RAM from log buffers and dumps; model and sampling stay from config.
    """
    if not _CFG_PATH.exists():
        return
    try:
        with open(_CFG_PATH, "r") as f:
            cfg = json.load(f)
    except Exception:
        return

    # Model (required for agent/ask/plan tests)
    if not os.environ.get("SC2_MLX_MODEL") and cfg.get("mlxModel"):
        os.environ["SC2_MLX_MODEL"] = str(cfg["mlxModel"]).strip()
    if not os.environ.get("SC2_MLX_MODEL_PATH") and cfg.get("mlxModelPath"):
        os.environ["SC2_MLX_MODEL_PATH"] = str(cfg["mlxModelPath"]).strip()

    # Runtime / sampling (same as main.js)
    if "SC2_TEMP" not in os.environ and "temp" in cfg:
        os.environ["SC2_TEMP"] = str(cfg["temp"] if cfg["temp"] is not None else 0.25)
    if "SC2_TOP_P" not in os.environ and "topP" in cfg:
        os.environ["SC2_TOP_P"] = str(cfg["topP"] if cfg["topP"] is not None else 0.9)
    if "SC2_MAX_NEW" not in os.environ and "maxNew" in cfg:
        os.environ["SC2_MAX_NEW"] = str(cfg["maxNew"] if cfg["maxNew"] is not None else 4096)

    # Root: resolve like main.js (path.resolve(repoRoot, cfg.root))
    if "SC2_ROOT" not in os.environ and "root" in cfg:
        root_cfg = (cfg.get("root") or ".").strip() or "."
        resolved = (_REPO / root_cfg).resolve()
        os.environ["SC2_ROOT"] = str(resolved)

    # Debug: from config, or off during tests to save RAM
    if "SC2_DEBUG" not in os.environ:
        os.environ["SC2_DEBUG"] = "0" if reduce_debug_for_tests else ("1" if cfg.get("debug") else "0")
    if "SC2_DEBUG_LOG" not in os.environ:
        os.environ["SC2_DEBUG_LOG"] = "" if reduce_debug_for_tests else str(cfg.get("debugLog") or "")
    if "SC2_DEBUG_KV" not in os.environ:
        os.environ["SC2_DEBUG_KV"] = "0" if reduce_debug_for_tests else ("1" if cfg.get("debugKv") else "0")
    if "SC2_DEBUG_CHAT" not in os.environ and "debugChat" in cfg:
        os.environ["SC2_DEBUG_CHAT"] = "1" if cfg.get("debugChat") else "0"

    # Pipeline / model profile
    if "SC2_PIPELINE_IMPL" not in os.environ:
        os.environ["SC2_PIPELINE_IMPL"] = str(cfg.get("pipelineImpl") or os.environ.get("SC2_PIPELINE_IMPL") or "rail_v3")
    if "SC2_USE_CORE_V2" not in os.environ:
        v = cfg.get("useCoreV2")
        if v is None:
            v = os.environ.get("SC2_USE_CORE_V2", "1")
        os.environ["SC2_USE_CORE_V2"] = str(v)
    if "SC2_MODEL_PROFILE" not in os.environ:
        os.environ["SC2_MODEL_PROFILE"] = str(cfg.get("modelProfile") or os.environ.get("SC2_MODEL_PROFILE") or "auto")
    if "SC2_USE_CHATML_WRAP" not in os.environ:
        v = cfg.get("useChatmlWrap")
        if v is None:
            v = os.environ.get("SC2_USE_CHATML_WRAP", "1")
        os.environ["SC2_USE_CHATML_WRAP"] = str(v)

    if cfg.get("usePipeline") is False and "SC2_USE_STATE_MACHINE_PIPELINE" not in os.environ:
        os.environ["SC2_USE_STATE_MACHINE_PIPELINE"] = "false"

    # Chat tools / approval / auto-apply
    if "SC2_USE_CHAT_TOOLS" not in os.environ:
        os.environ["SC2_USE_CHAT_TOOLS"] = "1" if cfg.get("useChatTools") else "0"
    if "SC2_APPROVAL_MODE" not in os.environ:
        os.environ["SC2_APPROVAL_MODE"] = str(cfg.get("approvalMode", 1))
    if "SC2_AUTO_APPLY_ON_SUCCESS" not in os.environ:
        os.environ["SC2_AUTO_APPLY_ON_SUCCESS"] = str(cfg.get("autoApplyOnSuccess", 1))

    # Bridge (neutral naming)
    if "SC2_USE_EXTERNAL_BRIDGE" not in os.environ:
        v = cfg.get("useExternalBridge") if cfg.get("useExternalBridge") is not None else cfg.get("useContinueBridge")
        os.environ["SC2_USE_EXTERNAL_BRIDGE"] = str(1 if v else 0)
    if "SC2_BRIDGE_CLI_CMD" not in os.environ:
        os.environ["SC2_BRIDGE_CLI_CMD"] = str(cfg.get("bridgeCliCmd") or cfg.get("continueCliCmd") or "")
    if "SC2_BRIDGE_NODE_BIN" not in os.environ:
        os.environ["SC2_BRIDGE_NODE_BIN"] = str(cfg.get("bridgeNodeBin") or cfg.get("continueNodeBin") or "node")
    if "SC2_BRIDGE_TIMEOUT_S" not in os.environ:
        os.environ["SC2_BRIDGE_TIMEOUT_S"] = str(cfg.get("bridgeTimeoutS") or cfg.get("continueTimeoutS") or 180)
    if "SC2_BRIDGE_PRINT_FORMAT" not in os.environ:
        os.environ["SC2_BRIDGE_PRINT_FORMAT"] = str(cfg.get("bridgePrintFormat") or cfg.get("continuePrintFormat") or "")
    if "SC2_BRIDGE_SILENT_PRINT" not in os.environ:
        v = cfg.get("bridgeSilentPrint") if cfg.get("bridgeSilentPrint") is not None else cfg.get("continueSilentPrint")
        os.environ["SC2_BRIDGE_SILENT_PRINT"] = str(1 if v else 0)
    if "SC2_BRIDGE_GLOBAL_DIR" not in os.environ:
        os.environ["SC2_BRIDGE_GLOBAL_DIR"] = str(cfg.get("bridgeGlobalDir") or cfg.get("continueGlobalDir") or "")

    if cfg.get("verifyCommand") and "SC2_VERIFY_CMD" not in os.environ:
        os.environ["SC2_VERIFY_CMD"] = str(cfg["verifyCommand"])

    # Port (main.js sets SC2_PORT for the server)
    if "SC2_PORT" not in os.environ and cfg.get("port") is not None:
        os.environ["SC2_PORT"] = str(int(cfg["port"]))

    # PATH: same order as main.js so tools find the same binaries
    os.environ["PATH"] = _path_prepend_for_server(os.environ.get("PATH"))

    # HuggingFace / Python (main.js sets these for the server process)
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
