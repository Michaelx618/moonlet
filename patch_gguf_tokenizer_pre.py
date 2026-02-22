#!/usr/bin/env python3
"""
Patch tokenizer.ggml.pre in a GGUF file so llama.cpp can load it.

Working model (DeepSeek) uses: tokenizer.ggml.pre = "deepseek-llm"
Broken model (ds67b) uses:    tokenizer.ggml.pre = "gpt2"

Usage:
  python patch_gguf_tokenizer_pre.py input.gguf output.gguf [--pre deepseek-llm]

Example:
  python patch_gguf_tokenizer_pre.py ds67b_merged_q8_0.gguf ds67b_patched.gguf

Requires: pip install gguf tqdm
"""

import argparse
import subprocess
import sys
from pathlib import Path


def patch_gguf_tokenizer_pre(input_path: str, output_path: str, new_pre: str = "deepseek-llm"):
    """Patch tokenizer.ggml.pre using gguf_new_metadata (handles length changes)."""
    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()

    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        sys.exit(1)

    # Use gguf_new_metadata to copy with modified pre-tokenizer
    cmd = [
        sys.executable,
        "-m",
        "gguf.scripts.gguf_new_metadata",
        str(input_path),
        str(output_path),
        "--pre-tokenizer",
        new_pre,
        "--force",
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(result.returncode)
    print(f"Patched tokenizer.ggml.pre to {new_pre!r}, wrote {output_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--pre", default="deepseek-llm")
    args = ap.parse_args()
    patch_gguf_tokenizer_pre(args.input, args.output, args.pre)
