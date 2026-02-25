"""Moonlet server entrypoint."""

import signal
import sys
import time
import traceback

from . import config
from .files import get_root
from .server import start_server


def main():
    """Start the Moonlet HTTP server."""
    # Start each app launch with a fresh chat session.
    try:
        from . import state
        state.clear_chat_session()
        print("[Started new chat session]", file=sys.stderr)
    except Exception:
        pass

    # Startup logging
    if getattr(config, "MLX_MODEL", None):
        print(f"[Using MLX model: {config.MLX_MODEL}]", file=sys.stderr)
    elif getattr(config, "MLX_MODEL_PATH", None):
        print(f"[Using MLX model path: {config.MLX_MODEL_PATH}]", file=sys.stderr)
    elif config.GGUF_PATH:
        print(f"[Using GGUF model: {config.GGUF_PATH}]", file=sys.stderr)
    else:
        print(f"[Using HuggingFace model: {config.MODEL_NAME}]", file=sys.stderr)
    print(f"[Root path: {get_root()}]", file=sys.stderr)

    # Build file index for agent tools (grep, symbols, list_files)
    try:
        from .index import rebuild_index
        rebuild_index()
    except Exception as e:
        print(f"[index: startup rebuild failed — {e}]", file=sys.stderr)

    # Load MLX model before binding port so first request doesn't block (client would timeout)
    if getattr(config, "MLX_MODEL", None) or getattr(config, "MLX_MODEL_PATH", None):
        try:
            from .model import ensure_model_loaded
            print("[Loading MLX model (this may take 1–2 min)...]", file=sys.stderr)
            sys.stderr.flush()
            ensure_model_loaded()
            print("[Model load complete]", file=sys.stderr)
        except BaseException:
            print("[Model load failed]", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            sys.stderr.flush()
            print("\nFix the error above (e.g. pip install mlx mlx-lm, enough RAM, or set SC2_MLX_MODEL_PATH to a local cache).", file=sys.stderr)
            sys.exit(1)

    try:
        start_server()
        print(
            f"[Server running on port {config.SERVER_PORT}. Ctrl+C to exit.]",
            file=sys.stderr,
        )
        signal.signal(signal.SIGINT, lambda sig, frame: sys.exit(0))
        while True:
            time.sleep(1)
    except OSError as exc:
        print(f"Failed to start HTTP server on port {config.SERVER_PORT}: {exc}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nServer shutting down...", file=sys.stderr)
        sys.exit(0)
