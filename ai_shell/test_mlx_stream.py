#!/usr/bin/env python3
"""Test MLX streaming path: load model, call stream_reply_chunks, print each chunk.
Run from repo root with MLX env set, e.g.:
  SC2_MLX_MODEL_PATH=/path/to/cache python -m ai_shell.test_mlx_stream
"""
import os
import sys

# Use local cache so we don't hit network (match config.json)
if "SC2_MLX_MODEL_PATH" not in os.environ:
    os.environ["SC2_MLX_MODEL_PATH"] = (
        "/Users/michael/.cache/huggingface/hub/models--mlx-community--Qwen2.5-Coder-14B-Instruct-4bit/snapshots/29efdbab55a161237ab1e432a3abaf6c7ae2b477"
    )
if "SC2_DEBUG" not in os.environ:
    os.environ["SC2_DEBUG"] = "1"

def main():
    print("Importing ai_shell.model...", flush=True)
    from .model import ensure_model_loaded, stream_reply_chunks, _BACKEND_KIND

    print(f"Backend kind: {_BACKEND_KIND}", flush=True)
    print("Calling ensure_model_loaded() (may take 1-2 min)...", flush=True)
    ensure_model_loaded()
    print("Model loaded. Calling stream_reply_chunks('Say hello in 5 words.')...", flush=True)

    n = 0
    for chunk in stream_reply_chunks("Say hello in 5 words.", max_new=100):
        n += 1
        print(f"  chunk #{n}: {repr(chunk)}", flush=True)
        if n >= 20:
            print("  (stopping after 20 chunks)", flush=True)
            break

    print(f"Done. Received {n} chunks.", flush=True)
    return 0 if n > 0 else 1

if __name__ == "__main__":
    sys.exit(main())
