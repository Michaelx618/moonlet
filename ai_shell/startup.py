"""Moonlet server entrypoint."""

import signal
import sys
import time

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
    if config.GGUF_PATH:
        print(f"[Using GGUF model: {config.GGUF_PATH}]", file=sys.stderr)
    else:
        print(f"[Using HuggingFace model: {config.MODEL_NAME}]", file=sys.stderr)
    print(f"[Root path: {get_root()}]", file=sys.stderr)

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
