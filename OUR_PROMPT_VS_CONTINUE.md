# Our Full Prompt vs Continue

How we build the agent prompt versus what we know about Continue (from their docs and our borrowed design). Continue’s exact prompt text is not in this repo; this compares **structure and content** from our code and `BORROWED_FROM_CONTINUE.md` / `ROO_VS_MOONLET_PROMPTS_AND_FLOW.md`.

---

## 1. Our prompt (completion path — one string)

We build a **single prompt string** in `_build_prompt()` and then append tool rounds. Order:

| # | Section | Source | Content |
|---|---------|--------|---------|
| 1 | System | `BASE_AGENT_SYSTEM_MESSAGE` (config) | From Continue’s `defaultSystemMessages.ts`: agent mode, multiple read-only tools in parallel, codeblock language+filename, placeholders for large blocks, “for implementing changes use the edit tools”. |
| 2 | Tools | `TOOLS_SYSTEM_HINT` | Full tool schema in text: read_file, list_files, grep, glob, codebase_search, view_repo_map, view_diff, then edit_existing_file, search_replace, multi_edit, write_file/create_new_file, run_terminal_command. Plus bullets: read before editing, output edit tools for edit requests, edit_existing_file full/partial+placeholders, search_replace/multi_edit/write_file usage, brief explanation after edits, **when the whole task is complete (all requested changes applied everywhere needed) respond with a brief summary and no further tool calls**, decide next step from each tool result, do not repeat the same edit. |
| 3 | Workspace paths | `_workspace_paths_section()` | “Workspace paths (use these exact paths…):” + comma‑separated indexed paths (cap 30). We added this; Continue doesn’t inject a path list like this. |
| 4 | @Code | `_code_context_section()` | “@Code (path):” with symbols (functions/classes) and line ranges + snippets for focus_file and extra_read_files (up to 8 files, 20 symbols/file). |
| 5 | @Folder | `_folder_context_section()` | “@Folder (path):” file list (+ optional content for first 3 files) for context_folders. |
| 6 | Reference files | `_reference_files_section()` | “Reference files (priority context):” + “--- path ---” + full content + “(Use path … in tool calls.)” for extra_read_files (capped). |
| 7 | User request | Fixed | “User request:\n” + the user’s instruction (spec). |
| 8 | Current file | `_focus_file_section()` | “Current file (path for tools: …):” + “---” + full content + “---” + use path in edit_existing_file/write_file. |

Then each tool round we append:

- `Assistant:\n{model_reply}\n\nTool results:\n{combined_results}\n\nUser: …` (with optional “Files already edited…”, “Continue. Use the tool results above.”, or nudge to use edit tools / not repeat).

So the **whole prompt** = (1)–(8) once, then repeated blocks of Assistant + Tool results + User for every round.

---

## 2. Our prompt (API / chat path — messages + tools)

When `USE_CHAT_TOOLS` is true:

- **System message:** Only `BASE_AGENT_SYSTEM_MESSAGE` (no TOOLS_SYSTEM_HINT in the message; tool descriptions come from the API).
- **First user message:** Same content as (3)–(8) above: workspace paths, @Code, @Folder, reference files, “User request:” + spec, focus file. No tools text in the user message.
- **Tools:** Sent as `AGENT_TOOLS_JSON` in the chat completions request (names, descriptions, parameters).
- **Later rounds:** We append assistant message (with `tool_calls`) and user message (tool results). We do **not** refresh workspace/code/folder/ref/focus in later turns; only tool results and optional “Already edited…” prefix change.

So compared to the completion path: same system and same initial user content, but tools are API definitions instead of a big hint block, and there is no per‑turn “environment” refresh.

---

## 3. Continue (as documented in our repo)

From `BORROWED_FROM_CONTINUE.md` and the flow docs:

- **System:** Default agent system message from `defaultSystemMessages.ts` (we use the same text as `BASE_AGENT_SYSTEM_MESSAGE`).
- **Tools:** Sent **with** the user request; either native API `tool_calls` or XML in the system message. Tool **names and semantics** match what we have (read_file, list_files, grep, edit_existing_file, search_replace, multi_edit, create_new_file, run_terminal_command, codebase_search, view_repo_map, view_diff, etc.).
- **Context:** Provided by **context providers** (e.g. @File, @CurrentFile, @Code, @Folder). The IDE/core **resolves** these and attaches the resulting items to the request. So the user (or system) message effectively contains “attached” context; we don’t have a single documented “Continue prompt template” in this repo.
- **User message:** The user’s request plus the attached context (files, current file, symbols, folder listings, etc.). No explicit “Workspace paths:” list in our doc—Continue relies on the model discovering paths via list_dir / read_file.
- **Flow:** Tools → model may return tool calls → we execute → **results fed back into the model as context** → repeat until no tool calls. So tool results are the main per‑turn change, similar to us.
- **When to stop:** “When the model responds with no tool calls, that reply is the final answer.” Continue doesn’t spell out “when the whole task is complete” in our borrowed notes; they rely on the model deciding to stop.

---

## 4. Side‑by‑side comparison

| Aspect | Continue (from our docs) | Us |
|--------|--------------------------|-----|
| **System message** | Same default agent message (we took it from `defaultSystemMessages.ts`). | Same: `BASE_AGENT_SYSTEM_MESSAGE`. |
| **Where tools are described** | With the request (API tool definitions or XML in system). | Completion: full text schema + rules in prompt (`TOOLS_SYSTEM_HINT`). API: only in request as `AGENT_TOOLS_JSON`; system has no tool text. |
| **Explicit “when to finish”** | Not documented; stop when no tool calls. | Explicit: “When the whole task is complete (all requested changes applied everywhere needed), respond with a brief summary and no further tool calls.” |
| **Workspace path list** | Model discovers via list_dir / read_file. | We inject “Workspace paths:” + indexed paths (cap 30) so the model doesn’t invent paths. |
| **Context shape** | Context providers attach @File, @CurrentFile, @Code, @Folder, etc. | We build the same *kinds*: Reference files, Current file, @Code (symbols+snippets), @Folder, “User request:”, plus our workspace path list. |
| **Context per turn** | New context can be attached each turn by the IDE. | Completion: same initial context for the whole run; only “Tool results” + “User: Continue…” change each round. API: same; no per‑turn refresh of workspace/code/folder/ref. |
| **Tool results** | Fed back as context; loop until no tool calls. | Same: we append tool results and a “User: Continue…” (or “Files already edited…”, nudges). |
| **Structure** | Likely system + user (with attached context) + tool definitions. | Completion: one string = system + tools text + all context + user request; then Assistant + Tool results + User each round. API: messages = [system, user]; then assistant + user per round; tools in API. |

---

## 5. Summary

- **Shared with Continue:** Default agent system message, same tool set and semantics, same loop (tools → results fed back → repeat until no tool calls), same *kinds* of context (reference files, current file, @Code, @Folder).
- **Different from Continue:** We put a **full tool schema and rules in the prompt** (completion path) and add an **explicit “when the whole task is complete…”** instruction; we **inject a workspace path list**; we **don’t refresh** workspace/code/folder/ref per turn (same context for the whole run aside from tool results).
- **API path:** We don’t put tool text in the system message (tools are in the API only); initial user message matches our context sections; no per‑turn environment block.
