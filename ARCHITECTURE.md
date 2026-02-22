# Moonlet architecture

High-level structure of the Moonlet coding agent and how requests flow from the client to tools and back. The agent loop design was borrowed from Continue; see [BORROWED_FROM_CONTINUE.md](BORROWED_FROM_CONTINUE.md).

---

## Entry points

```
main.py                    → launcher
ai_shell/startup.py        → server entry: config, clear chat, rebuild_index(), start_server()
ai_shell/server.py         → HTTP API (ThreadingHTTPServer), route handlers
```

- **Server**: `startup.main()` sets root, rebuilds the file index once, then runs `start_server()`. All agent/edit traffic goes through **POST /stream**.

---

## Request flow (simplified)

```
Client (Electron/local-app)
    │
    │  POST /stream { mode, text, focus_file?, extra_read_files?, ... }
    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  server.APIServer (server.py)                                           │
│  • Validates path, resolves focus_file                                   │
│  • If agent and no focus_file: find_relevant_files(text) → discover     │
│  • Branches on mode                                                     │
└─────────────────────────────────────────────────────────────────────────┘
    │
    └── mode in (agent, chat, repair)  ──►  agent_loop.run_agent()
                                            (build prompt → model → tool loop → meta)
```

After a run, the server may run **verify** (compile/test), and on failure feed the error back as a repair request. Patch proposals (approval mode) and auto-apply are handled in the same handler.

---

## Core components

| Layer | Module | Role |
|-------|--------|------|
| **API** | `server.py` | HTTP routes: `/stream`, `/root`, `/include`, `/proposal/apply`, `/verify`, etc. SSE for streaming; stores patch proposals; triggers verify + repair loop. |
| **Agent** | `agent_loop.py` | Builds one prompt (workspace paths, reference files, focus file, tools hint, user text). Calls model; parses tool calls from reply; runs tools via `tool_executor`; appends results and repeats until no tool calls. |
| **Verify** | `verify.py` | `run_verify()`: build/test and optional runtime test; used by server after edits. |
| **Tools** | `tool_executor.py` | Defines tool list and schema; parses function-style and XML tool calls; `execute_tool_from_kwargs(name, kwargs)` runs read_file, grep, list_files, search_replace, write_file, multi_edit, edit_existing_file, create_new_file, run_terminal_cmd. Logs `[tool]` and `[tool_wrote]` to debug log. |
| **Edit** | `search_replace.py` | Validates and runs single/multi find-and-replace; `execute_search_replace`, `apply_single_multi_edit`; match strategies (exact, trimmed, etc.) in `edit_match.py`. |
| **Model** | `model.py` | Backend abstraction: llama-server vs CLI. `stream_reply_chunks()` for completion-style; `chat_completion_with_tools()` when USE_CHAT_TOOLS. KV cache key for sessions. |
| **Files** | `files.py` | Root and include: `get_root()`, `set_root()`, `get_include()`, `set_include()`. Read/write: `read_file_text`, `write_file_text`. All paths relative to root; `is_edit_allowed`, `_path_in_include`. |
| **Index** | `index.py` | `rebuild_index()` when root/include change; `_list_editable_files()` (cap 200); `get_indexed_files()`, `get_symbols_for_file()`. Used by agent tools and workspace path injection. |
| **Relevance** | `relevance.py` | `find_relevant_files(user_text, open_file)` for focus discovery and pipeline context selection. |
| **State** | `state.py` | Chat turns, change notes, failure notes, task checkpoints (persisted on disk). |
| **Config** | `config.py` | DEBUG, DEBUG_LOG_PATH, USE_CHAT_TOOLS, SERVER_PORT, etc. |

---

## Data flow (agent path)

```
run_agent(text, focus_file, mode, extra_read_files, context_folders)
    │
    ├─► _build_prompt(...)
    │       • _workspace_paths_section()     ← index.get_indexed_files()
    │       • _reference_files_section()    ← extra_read_files, files.read_single_file_for_context
    │       • _focus_file_section()         ← focus_file content
    │       • _code_context_section()       ← optional @Code / @Folder
    │       • TOOLS_SYSTEM_HINT + user text
    │
    ├─► Model: stream_reply_chunks(prompt) or chat_completion_with_tools(...)
    │       → raw reply text (or message + tool_calls)
    │
    ├─► Parse tool calls: extract_function_style_tool_calls(text) or API tool_calls
    │
    ├─► For each call: execute_tool_from_kwargs(name, kwargs)
    │       → tool_executor uses files.*, index.*, search_replace.*, tools.grep_search
    │       → [tool] and [tool_wrote] written to DEBUG_LOG_PATH
    │
    ├─► Append "Tool results: ..." to conversation; loop back to model
    │
    └─► When no tool calls: return (output, meta); server sends meta + optional verify/repair
```

---

## Where things live

- **Prompt text**: Built in `agent_loop._build_prompt()`.
- **Tool list / schema**: `tool_executor.AGENT_TOOLS_JSON`, `TOOLS_SYSTEM_HINT`.
- **Edit execution**: `tool_executor.execute_tool_from_kwargs` → `search_replace.execute_search_replace` / `apply_single_multi_edit`, or `files.write_file_text` for write_file/edit_existing_file/create_new_file.
- **Debug log**: `config.DEBUG_LOG_PATH`; `[tool]` and `[tool_wrote]` lines from `tool_executor`; `[debug]` from `utils.dbg()` when DEBUG is set.
- **Root / include**: `files.ROOT_PATH`, `files.INCLUDE_PATHS`; set via server routes; `index.rebuild_index()` runs on set_root/set_include and at startup.

---

## Diagram (ASCII)

```
                    ┌──────────────┐
                    │ Client       │
                    │ (Electron)   │
                    └──────┬───────┘
                           │ POST /stream
                           ▼
                    ┌──────────────┐
                    │ server.py    │
                    │ /stream      │
                    └──────┬───────┘
                           │
                           ├──────────────────┐
                           ▼                  ▼
                    ┌──────────────┐  ┌─────────────┐
                    │ agent_loop   │  │ verify.py   │
                    │ .py          │  │             │
                    └──────┬───────┘  └──────┬──────┘
                           │                 │
                           ▼                 ▼
                    ┌──────────────┐  ┌──────────────┐
                    │ model.py    │  │ relevance.py │
                    │ (LLM)       │  │ index.py     │
                    └──────┬──────┘  └──────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │ tool_        │
                    │ executor.py  │
                    └──────┬───────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
  ┌──────────┐      ┌──────────────┐    ┌──────────┐
  │ files.py │      │ search_      │    │ tools.py  │
  │ index.py │      │ replace.py   │    │ (grep)    │
  └──────────┘      └──────────────┘    └──────────┘
```

This is the architecture as of the current codebase; for prompt and mode details see **PROMPT_AND_FLOW.md**.
