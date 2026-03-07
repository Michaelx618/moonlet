# Plan Task 2 Failure Analysis

## Task
- **Name**: plan-w7-explore-structure
- **Spec**: "What's in this project and how does it fit together?"
- **Root**: tests/fixtures/w7
- **Include**: checkpasswd.c, validate.c, common.h, helper.c, spec.txt, Makefile
- **focus_file**: null
- **extra_read_files**: []

## What Happened

### Round 1
- Model called `read_file` on 10 files: README.md, SOURCES.txt, spec.txt, task.json, common.h, checkpasswd.c, validate.c, pass.txt, pass_example.txt, clean.sh
- Some files (README.md, SOURCES.txt, task.json, pass.txt, pass_example.txt, clean.sh) are **outside** the task's `include` list
- Prompt grew to ~10,270 chars

### Rounds 2–7
- Model's **first token** each round: ` ``` ` (backticks)
- Prompt grew slowly: 10,270 → 11,271 → 11,659 → 12,047 → 12,286 → 12,525 chars (~200–600 chars/round)
- No `[tool] call:` logs in rounds 2–7 → either no tool calls executed, or output truncated
- Process ran ~8+ minutes before being killed

## Root Cause Analysis

### 1. Degenerate output loop
The model repeatedly starts replies with ` ``` `, suggesting it is trying to emit markdown/code blocks instead of a plan. Possible causes:
- Training bias toward code blocks
- Stop sequences or prompt structure pushing it into markdown mode
- Local maximum where it keeps producing similar output

### 2. Broad, underspecified task
- Spec: "What's in this project and how does it fit together?"
- No `focus_file`, no `extra_read_files`
- Model may be unsure what to do after reading files and falls back to markdown

### 3. Possible parser false positives
`extract_function_style_tool_calls` also inspects content inside ` ```...``` ` blocks. If the model emits something like ` ```\nread_file(...)\n``` `, it could be parsed as a tool call and executed, which would explain continued rounds without obvious tool logs (e.g., if logs are truncated or buffered).

### 4. Files outside include
Round 1 read files not in `include`. That may be allowed by the tool executor but can add noise and lengthen the prompt.

## Recommendations

1. **Tighten the task spec** – Make the exploratory task more directive, e.g.:
   - "List the main files and their roles. Then output a brief plan as numbered steps."
2. **Add a reply guard** – After reading files, if the model starts with ` ``` ` or similar, inject a nudge like: "Output your plan as plain text now. Do not use code blocks."
3. **Add early-stop for repetitive output** – If the model’s reply is short and repeats the same pattern (e.g. ` ``` `), reduce the number of rounds or stop early.
4. **Check `include` enforcement** – Ensure `read_file` and similar tools respect `include` so the model only reads allowed files.
