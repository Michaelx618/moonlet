#!/usr/bin/env python3
"""Test KV cache reuse across requests.

Run with: SC2_DEBUG_KV=1 SC2_GGUF=<path> python -m ai_shell.test_kv_cache

Makes two completion requests with the same session cache_key; the second
prompt extends the first. With KV reuse, the server should skip re-processing
the prefix. Check runtime-debug.log for kv_cache: lines.
"""

import os
import sys
import time

os.environ.setdefault("SC2_DEBUG_KV", "1")


def main():
    from ai_shell import config
    from ai_shell.files import set_root, set_include
    from ai_shell.model import get_session_cache_key, stream_reply

    # Use repo as root, no include
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    set_root(repo)
    set_include([])

    key = get_session_cache_key()
    print(f"session_cache_key: {key}")

    # Prompt 1: short
    prompt1 = "SYSTEM:\nYou are orbit.\n\nCONTEXT:\nFILES: a.c\n\nUser: list files\nAssistant:"
    # Prompt 2: extends prompt1 (simulates round 2 of tool loop)
    prompt2 = (
        prompt1
        + "\n[[[list_files]]]\n\nTool results:\n  a.c\n  b.c\n\n"
        "User: Continue. Use the tool results above.\nAssistant:"
    )

    print("Request 1 (cold)...")
    t0 = time.perf_counter()
    r1 = stream_reply(prompt1, silent=True, max_new=32, cache_key=key)
    t1 = time.perf_counter()
    print(f"  reply_len={len(r1)} ms={int((t1-t0)*1000)}")

    print("Request 2 (should reuse KV if prompt extends)...")
    t2 = time.perf_counter()
    r2 = stream_reply(prompt2, silent=True, max_new=32, cache_key=key)
    t3 = time.perf_counter()
    print(f"  reply_len={len(r2)} ms={int((t3-t2)*1000)}")

    print("Done. Check runtime-debug.log for kv_cache: lines (SC2_DEBUG_KV=1)")
    print("With KV reuse, request 2 prompt_per_second should be >> request 1 (only new tokens processed)")


if __name__ == "__main__":
    main()
