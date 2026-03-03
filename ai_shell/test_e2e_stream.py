#!/usr/bin/env python3
"""E2E test: start server (same env as app), POST /stream, print SSE events.
Run from repo root:
  SC2_MLX_MODEL_PATH=... SC2_PORT=8002 python -m ai_shell.test_e2e_stream
"""
import json
import os
import subprocess
import sys
import time
import urllib.request
import urllib.error

# Use 8003 by default so we don't conflict with a running app on 8002
PORT = int(os.getenv("SC2_PORT", "8003"))
MLX_PATH = os.getenv(
    "SC2_MLX_MODEL_PATH",
    "/Users/michael/.cache/huggingface/hub/models--mlx-community--Qwen2.5-Coder-14B-Instruct-4bit/snapshots/29efdbab55a161237ab1e432a3abaf6c7ae2b477",
)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main():
    env = {
        **os.environ,
        "SC2_MLX_MODEL_PATH": MLX_PATH,
        "SC2_PORT": str(PORT),
        "SC2_ROOT": REPO_ROOT,
        "SC2_USE_CHATML_WRAP": "1",
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
    }
    if not env.get("SC2_MLX_MODEL") and not env.get("SC2_MLX_MODEL_PATH"):
        env["SC2_MLX_MODEL_PATH"] = MLX_PATH

    print(f"Starting server on port {PORT} (cwd={REPO_ROOT})...", flush=True)
    # Use None for stderr so server logs go to this process's stderr (see [stream]/[agent]/[model] logs)
    proc = subprocess.Popen(
        [sys.executable, os.path.join(REPO_ROOT, "main.py")],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=None,  # inherit so pipeline logs are visible
        text=True,
    )

    try:
        # Wait for server to bind
        base = f"http://127.0.0.1:{PORT}"
        for _ in range(60):
            try:
                with urllib.request.urlopen(f"{base}/health", timeout=1) as r:
                    data = json.loads(r.read().decode())
                    if data.get("status") == "ok":
                        print("Server ready.", flush=True)
                        break
            except Exception:
                time.sleep(0.5)
        else:
            print("Server did not become ready in 60s", flush=True)
            return 1

        # Optional: wait for model to finish loading (otherwise first request blocks in get_model())
        for _ in range(120):
            try:
                with urllib.request.urlopen(f"{base}/health", timeout=2) as r:
                    data = json.loads(r.read().decode())
                    if not data.get("model_loading", False):
                        print("Model loaded.", flush=True)
                        break
            except Exception:
                pass
            time.sleep(1)
        else:
            print("(Model still loading; first request may block)", flush=True)

        # POST /stream
        body = json.dumps({"text": "Reply with exactly: OK", "mode": "chat"}).encode()
        req = urllib.request.Request(
            f"{base}/stream",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        print("POST /stream ...", flush=True)
        with urllib.request.urlopen(req, timeout=90) as resp:
            chunk_count = 0
            event = None
            data_lines = []
            for line in resp:
                line = line.decode("utf-8", errors="replace").rstrip("\n\r")
                if line.startswith("event:"):
                    event = line[6:].strip()
                elif line.startswith("data:"):
                    data_lines.append(line[5:].lstrip())
                elif line == "" and event is not None:
                    data = "\n".join(data_lines)
                    if event == "chunk" and data:
                        chunk_count += 1
                        print(f"  [chunk #{chunk_count}] {repr(data[:80])}{'...' if len(data) > 80 else ''}", flush=True)
                    elif event == "done":
                        print("  [done]", flush=True)
                        break
                    elif event == "action":
                        print(f"  [action] {data[:60]}...", flush=True)
                    elif event == "meta":
                        print("  [meta] (received)", flush=True)
                    event = None
                    data_lines = []

        print(f"\nPipeline test result: {chunk_count} chunk(s) received.", flush=True)
        if chunk_count == 0:
            print("FAIL: no chunks (prompt may not have reached model or SSE not sent)", flush=True)
            return 1
        print("PASS: prompt was sent and model response streamed.", flush=True)
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    sys.exit(main())
