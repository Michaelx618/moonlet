# What we borrowed from Continue

This project no longer uses "continue" in module or API names; the following list documents what was borrowed from Continue (docs + tool set + flow). See [ARCHITECTURE.md](ARCHITECTURE.md) and [PROMPT_AND_FLOW.md](PROMPT_AND_FLOW.md) for current design.

## Full list: what we took from Continue

Everything below is implemented in this repo and derived from or aligned with Continue (docs + their tool set + flow).

**Flow and loop**
- Agent flow: tools sent with the user request → model may return tool calls → we execute tools → results fed back into the model → repeat until no tool calls (docs.continue.dev/ide-extensions/agent/how-it-works).
- Tool results as context: same as Continue ("fed back into the model as context").
- When the model responds with no tool calls, that reply is the final answer; we stop.

**System message**
- Default agent system message from Continue's `defaultSystemMessages.ts`, stored in `BASE_AGENT_SYSTEM_MESSAGE` (config.py). Override with `SC2_BASE_AGENT_SYSTEM_MESSAGE`.

**Tools (names and semantics) — all 4 edit forms**
- **read_file** (filepath/path).
- **view_subdirectory** / **list_files** (path).
- **grep_search** / **grep** (pattern, path).
- **glob_search** / **glob_file_search** (glob, path).
- **codebase_search** / **codebase_tool** (query).
- **view_repo_map** (no args).
- **view_diff** (path).
- **edit_existing_file**: (1) path + **changes** (full/partial content), or (2) path + old_string + new_string (same as search_replace).
- **single_find_and_replace** / **search_replace** (path, old_string, new_string, **replace_all**).
- **multi_edit** (path, **edits[]** — array of {old_string, new_string, replace_all?}).
- **create_new_file** / **write_file** (path, content). create_new_file rejects if file exists ("use edit tool").
- **run_terminal_command** / **run_terminal_cmd** (command).
- **symbols** (path) — for @Code-style symbol listing.

**Edit logic (ported from Continue core/edit/searchAndReplace)**
- **Match strategies**: exact, trimmed, case-insensitive, whitespace-ignored (`ai_shell/edit_match.py`).
- **validate_single_edit**, **execute_find_and_replace** (with replace_all), **execute_multi_find_and_replace** in `ai_shell/search_replace.py`.
- **Scripts**: Continue's `scripts/` folder is copied into this repo as `scripts/continue/` (install-dependencies.sh, build-packages.js, util/index.js, etc.). Run from repo root or from `vendor/continue` context as needed; see `.cursorignore` (scripts/ is ignored for IDE).

**Tool schemas**
- `TOOLS_SCHEMA`: text description for completion mode (function-style calls). In `ai_shell/tool_executor.py`.
- `AGENT_TOOLS_JSON`: full JSON schema for API tool calls (/v1/chat/completions with tools). Used when `USE_CHAT_TOOLS` is true.
- XML tool-call fallback: we parse `<tool_call>{"name":"...","arguments":{...}}</tool_call>` so we accept the same format Continue uses when native tools aren't available.

**Context (Continue-style context providers)**
- **@File / Reference files**: `extra_read_files` → we inject file content under "Reference files" with explicit path and "(Use path \"…\" in tool calls.)".
- **@CurrentFile**: `focus_file` → we inject content under "Current file (path for tools: \"…\")" and tell the model not to create new directories.
- **@Code**: we inject symbol-level context (functions, classes) with snippets for `focus_file` and `extra_read_files` via `_code_context_section`; also exposed as **symbols** tool.
- **@Folder**: we inject file list and optional content for `context_folders` / `extra_folders` via `_folder_context_section`.
- **Workspace paths**: we inject the list of indexed paths (so the model uses real paths and doesn't invent dirs like `w7/`). Continue doesn't do this; we added it to avoid wrong-path writes.

**Guards and hints**
- Explicit path in every file block so the model uses that path in tool calls.
- **write_file** guard: if the path would create a new top-level directory that doesn't exist in the index and a file with the same basename exists at root, we reject and suggest the workspace path.

**Native API tool calls**
- When `USE_CHAT_TOOLS` is true: we use `chat_completion_with_tools` and structured `tool_calls` from the API (no text parsing). Same idea as Continue's native tool calling.
- API tool name → internal name mapping: view_subdirectory→list_files, grep_search→grep, edit_existing_file→search_replace, create_new_file→write_file, run_terminal_command→run_terminal_cmd.

**Server / API**
- `/stream` and `/v2/execute` accept and pass through: `focus_file`, `extra_read_files`, `context_folders` (or `extra_folders`), plus text and mode.

**Config**
- `BASE_AGENT_SYSTEM_MESSAGE`, `MAX_REF_CONTENT_CHARS`, `MAX_FOCUS_CONTENT_CHARS`, `MAX_READ_FILES_IN_PROMPT`, `MAX_CODE_SNIPPET_LINES`, `MAX_FOLDER_CONTEXT_FILES`, `USE_CHAT_TOOLS`, and related limits.

---

## Where it lives in this repo

- **Agent loop**: `ai_shell/agent_loop.py` — `run_agent`, `_run_agent_api_tools` (when USE_CHAT_TOOLS), `_build_prompt`, `_workspace_paths_section`, `_code_context_section`, `_folder_context_section`, `_reference_files_section`, `_focus_file_section`.
- **Tools**: `ai_shell/tool_executor.py` — `AGENT_TOOLS_JSON` (API schema), `execute_tool_from_kwargs`, `TOOLS_SCHEMA`, write_file guard, create_new_file "file exists" guard, multi_edit, edit_existing_file (changes), symbols tool.
- **Edit match**: `ai_shell/edit_match.py` — `find_search_matches` (exact, trimmed, case-insensitive, whitespace-ignored).
- **Search/replace**: `ai_shell/search_replace.py` — `validate_single_edit`, `execute_find_and_replace` (replace_all), `execute_multi_find_and_replace`, `apply_single_multi_edit`, `parse_multi_edit_calls`.
- **Config**: `ai_shell/config.py` — `BASE_AGENT_SYSTEM_MESSAGE`, limits for context.
- **Server**: `ai_shell/server.py` — `/stream` and `/v2/execute` pass `focus_file`, `extra_read_files`, and `context_folders` (or `extra_folders`) into `run_agent`.

---

## Summary

- **Agent flow** (docs.continue.dev/ide-extensions/agent/how-it-works): tools sent with the user request → model may return tool calls → we run tools → results fed back → repeat until no tool calls.
- **Tool set**: same names and semantics (read_file, view_subdirectory/list_files, grep_search, edit_existing_file/search_replace, create_new_file/write_file, run_terminal_command, codebase_search, view_repo_map, view_diff, etc.). See `TOOLS_SCHEMA` in `ai_shell/tool_executor.py`.
- **Context**: we inject the same *kinds* of context Continue uses (Reference files, Current file, Workspace paths, @Code, @Folder).

## Differences from Continue

| Aspect | Continue | Us |
|--------|----------|-----|
| **Tool calling** | Native (structured `tool_calls` from API) or XML in system message. | Completion API: we parse function-style (and optional XML) from the model's text. If the model doesn't output that format, we don't see tool calls. |
| **Context attachment** | Context items (e.g. @File, @CurrentFile) are resolved and attached by the IDE/core. | We build one prompt with: system + tools hint + workspace paths + reference file contents + user request + focus file content. Paths are explicitly stated so the model uses them. |
| **File discovery** | Model calls view_subdirectory / list_dir / read_file. | We inject workspace path list into the prompt and still support list_files; we also guard write_file to block creating new top-level dirs when the file exists at root. |

To get closer to Continue's behavior: use a backend with **native tool calling** (`USE_CHAT_TOOLS` + chat completions with tools) or add **XML tool parsing** as a fallback.

## What we copied (behavior/UX)

- **Brief explanation when done editing**: We surface the model's final reply (with tool calls stripped) as the response, so the client sees the model's explanation. We also added a system hint: "After making edits, give a brief explanation of what you changed."
- **Tool progress**: We send `event="action"` for each tool call (tool name, args, round) so the client can show progress (e.g. "Reading file...", "Editing...").

## What we didn't copy (behavior/UX)

- **Streaming the final explanation**: Continue likely streams the model's final reply token-by-token so the user sees the explanation as it's generated. We send the full explanation in one chunk when the agent run finishes. To match Continue, we would pass a stream callback into `run_agent` and send each token (or chunk) of the final round as SSE `chunk` events.
- **Explicit "explain after edit" in prompt**: We now ask for a brief explanation in the tool hint (see above). No other gap here.
- **Permission before running tools**: Continue can ask the user for permission before running a tool. We run tools automatically (by design).

## What Continue has that we don't (features)

- **search_web**, **fetch_url_content**, **create_rule_block**, **read_currently_open_file** (we inject current file in prompt instead).
- **Plan mode** (read-only only); we have agent + chat with same tool schema.
- **@Git Diff**, **@Terminal**, **@Docs**, **@Web**, **@Url**, **@Clipboard** context providers.
- MCP server integration; we only have built-in tools.

Continue repo: https://github.com/continuedev/continue
