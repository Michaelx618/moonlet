# Prompt and flow: modes, tool choice, and debug log

## Why does the index show 200 files then 4?

The file index is rebuilt whenever **root** or **include** changes:

1. **First rebuild (e.g. 200 files)**  
   On server startup we call `rebuild_index()` once. At that point `root` is usually the **full workspace** (e.g. the Moonlet repo or the folder the app was started from). The index then lists all editable files under that root (up to 200), so you see “index: rebuilt, 200 files … include=none”.

2. **Later rebuilds (e.g. 4 files)**  
   When the client sets an **include filter** (e.g. “only these 4 files”) or a **new root** (e.g. a task folder like `w7`), `set_include()` or `set_root()` is called and triggers another `rebuild_index()`. The index is now restricted to that root/include, so you see “index: rebuilt, 4 files … include=4 path(s)” (or a different root).

So: **200 = full source/workspace**, **4 = narrowed to a task or selected set**. The index is used by agent tools: `list_files`, `grep`, `read_file`, and symbol indexing.

---

## Modes and how we choose what runs

All user requests that do edits or chat go through **POST /stream**. The request body has a **mode** and other fields. We decide what to run like this:

| Mode   | When used | What runs | Tool choice |
|--------|-----------|-----------|-------------|
| **agent** | Default for “do something” (edit, implement, fix). | **Agent loop** (`run_agent`). | **Agent**: Model gets a **single system + user prompt** that includes the tool list (and optionally workspace paths, reference files, focus file). The **model** decides which tools to call (read_file, grep, search_replace, write_file, etc.) by outputting tool calls; we parse and execute them and feed results back. No separate “mode” per tool—one agent prompt, model picks tools. |
| **chat**  | Q&A, no edits. | Same agent loop as agent mode (`run_agent`), but with a **chat** mode flag. | Same tool list; we may add a hint in the prompt that this is “answer in text, don’t edit”. Model can still call read-only tools. |
| **repair** | After verify failed; “fix the build error”. | **Agent loop** with repair text (original spec + last error). | Same as agent; model chooses tools. |

So:

- **We do not choose the tool** per request. The **model** chooses tools from the list we put in the prompt.
- We always use the **agent loop** (one prompt and tool loop).

---

## Does each case have different prompts?

- **Agent loop (agent / chat / repair)**  
  One prompt shape: **system message** (e.g. “You are a systematic coding agent…”) + **tool list / schema** + **workspace paths** (optional) + **@Code / @Folder / reference / focus file** sections + **user request**. The only difference between agent and chat is the **mode** value and any chat-specific hint (“answer in text”). Repair adds the **last_error** (and repair target) into the user text. So: **same prompt structure**, different **content** (spec vs question vs spec+error).


Summary: **Agent loop** uses one prompt style (with tool list + context). See [BORROWED_FROM_CONTINUE.md](BORROWED_FROM_CONTINUE.md) for design origin.

---

## Flow in one picture (agent mode)

```
Client POST /stream { mode: "agent", text: "...", focus_file?, extra_read_files? }
       ↓
Server: mode = agent → run_agent(text, focus_file, mode="agent", …)
       ↓
Build one prompt: system + TOOLS_SYSTEM_HINT + workspace_paths + @Code/@Folder/reference + focus_file + "User request: …"
       ↓
If USE_CHAT_TOOLS: chat_completion_with_tools(…) → model returns tool_calls or final message
Else: stream_reply_chunks(…) → we parse function-style tool calls from text
       ↓
For each tool call: execute_tool_from_kwargs(name, kwargs) → result string
       ↓
Append "Tool results: …" + result to conversation; send back to model (next round)
       ↓
Repeat until model response has no tool calls → that reply is the final answer
       ↓
Server sends final output + meta (touched files, diffs, etc.) to client
```

---

## Tool calls in the debug log

Tool calls are written to the **debug log** (e.g. `runtime-debug.log`) in two ways:

1. **`[tool]` lines**  
   Every tool execution logs a short line, e.g. `[tool] write_file -> checkpasswd.c`. This goes to stderr (if DEBUG_TOOLS) and to `DEBUG_LOG_PATH`.

2. **`[tool_call]` lines**  
   At the start of each tool we also log a **tool_call** line with **tool name** and **sanitized args** (path, pattern, content length, etc., not full file content) so you can see what was called without huge dumps.

So in the log you’ll see:

- **Index**: `[debug] index: rebuilt, N files root=... include=...`
- **Tool**: `[tool_call] name=write_file path=checkpasswd.c content_len=1234`
- **Tool result**: `[tool] write_file -> checkpasswd.c`

If you still “see nothing”, check:

- **DEBUG** (e.g. `SC2_DEBUG=1`) so `[debug]` lines are written.
- **DEBUG_LOG_PATH** (e.g. `SC2_DEBUG_LOG=/path/to/runtime-debug.log`) so all lines go to that file.
- **DEBUG_TOOLS** (default on) so `[tool]` is also printed to stderr.
