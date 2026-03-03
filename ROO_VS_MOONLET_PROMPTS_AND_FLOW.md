# Roo vs Moonlet: Prompt Generation and Flow

Comparison of Roo-Code's prompt generation and agent flow with Moonlet's, and what's worth borrowing.

---

## 1. Prompt generation

### Roo

- **System prompt** is built once per *request* (not per task) via `SYSTEM_PROMPT()` in `src/core/prompts/system.ts`. It's a single long string assembled from **sections** in a fixed order:
  1. **Role** – from current mode (Architect, Code, Ask, Debug, Orchestrator); each mode has `roleDefinition` + optional `customInstructions`.
  2. **Markdown rules** – all code/filename refs must be clickable `[\`file\`](path:line)`.
  3. **Tool use** – "Use provider-native tool-calling; no XML; at least one tool per response; prefer many tools when reasonable."
  4. **Tool-use guidelines** – assess what you have vs need, choose the right tool, react to results.
  5. **Capabilities** – what tools do; workspace file list in `environment_details`; execute_command; optional MCP.
  6. **Modes** – list of modes with slug and when-to-use.
  7. **Skills** (optional) – available skills + "mandatory skill check" before user-facing reply.
  8. **Rules** – cwd, paths, no `cd` out, shell chaining, file-edit restrictions, **must use `attempt_completion` when done**, only ask via `ask_followup_question`, not conversational, env_details usage, MCP one-at-a-time, wait for user after each tool.
  9. **System info** – OS, shell, home, workspace dir.
  10. **Objective** – 5 points: analyze task, work sequentially with tools, use capabilities, use `attempt_completion` when done, no pointless back-and-forth.
  11. **Custom instructions** – from `.roo/rules-*/AGENTS.md`, project `AGENTS.md`, etc., merged with mode/global.
- **Tools** are **not** in the system prompt; they're sent only in the API (tool definitions in the request).
- **User message** each turn = processed user/tool content (mentions resolved) + **one** `<environment_details>...</environment_details>` block that includes: visible files, open tabs, running/completed terminals, recently modified files, time, git status, cost, current mode, and (on first turn) recursive workspace file list.

### Moonlet

- **System prompt** is a single configurable string: `BASE_AGENT_SYSTEM_MESSAGE` (Continue-aligned: agent mode, read-only in parallel, codeblock rules, "use the edit tools"). No sections, no mode-specific role, no objective/rules blocks.
- **Tool description** is injected into the **same** prompt as `TOOLS_SYSTEM_HINT` + schema: "function-style calls; use read_file before editing; output edit tools for edit requests; prefer search_replace / edit_existing_file / multi_edit / write_file; brief explanation after edits."
- **User content** is built in `_build_prompt()` / API user message: workspace paths, code context (symbols + snippets), folder context, reference files, "User request: …", optional focus file. No structured "environment" block; no terminals, no "recently modified," no explicit "current mode" or cost.

### Comparison

| Aspect | Roo | Moonlet |
|--------|-----|---------|
| System prompt structure | Sectioned (role, rules, objective, capabilities, modes, skills, custom) | Single block (BASE_AGENT_SYSTEM_MESSAGE) |
| Mode/role | Per-mode role + optional custom instructions | No modes; one global agent message |
| Tool definitions | API only (native tool calls) | In prompt (TOOLS_SYSTEM_HINT + schema) or in API when USE_CHAT_TOOLS |
| "Environment" per turn | Single `<environment_details>` (paths, terminals, recent files, time, git, mode, cost) | Workspace paths + code/folder/ref sections in initial prompt only |
| Objective / completion | Explicit "use attempt_completion when done"; "no back-and-forth" | Implicit (stop when no tool calls); optional nudge to use edit tools |
| Conversational tone | "STRICTLY FORBIDDEN" to start with "Great/Certainly"; "direct and technical" | Not specified |

---

## 2. Flow

### Roo

- **Loop:** One task → stack of "user content" items. Each iteration: pop content → process mentions → strip old env_details → append fresh `getEnvironmentDetails()` → add user message to `apiConversationHistory` → optional context management (condense/truncate) → `getSystemPrompt()` → build clean history (merge consecutive user, strip images) → `api.createMessage(systemPrompt, cleanConversationHistory, metadata)` → stream.
- **Stream:** Assistant content (text + tool_use) is parsed; on each tool_use, `presentAssistantMessage` runs the tool (ask approval → execute) and appends result to `userMessageContent`. Assistant message is saved to history *before* running tools. When stream ends, `userMessageContent` is pushed onto the stack as the next "user content" (tool_results); next iteration adds env_details and becomes the next user message.
- **Tool results** are first accumulated in memory, then (when the stream is done for that turn) become the next user message; optionally flushed to history earlier (e.g. for delegation). So each request sees: … assistant (with tool_uses) → user (tool_results + environment_details).

### Moonlet

- **Completion path** (no USE_CHAT_TOOLS): One growing **prompt string**. Each round: stream reply → parse function-style tool calls → execute tools → append "Assistant:\n{reply}\n\nTool results:\n{combined}\n\nUser: Continue. …" to prompt → repeat until no tool calls (or nudge once if no edits yet / read-only only).
- **Chat API path** (USE_CHAT_TOOLS): `messages` = [system, user]. Each round: `chat_completion_with_tools(messages, tools=AGENT_TOOLS_JSON)` → append assistant message with tool_calls → execute each tool → append user message with tool results → repeat. No separate "environment" block per turn; same user content for round 1 only.

### Comparison

| Aspect | Roo | Moonlet |
|--------|-----|---------|
| Conversation shape | True multi-turn: user / assistant / user / assistant in API history | Completion: single string; Chat: message list but no per-turn env refresh |
| Environment per turn | New `environment_details` every turn (terminals, recent files, etc.) | Same workspace/code/ref context for whole run (or none after round 1 in chat path) |
| System prompt | Rebuilt every request (mode, MCP, skills can change) | Fixed for the whole run |
| Context management | Condense/truncate when over threshold; "effective" history | No compaction; optional supersede for same-path read_file |
| Tool execution | User approval (diff view or direct write); result is structured (e.g. JSON) or plain | Direct execution; result is success/fail string (Continue-aligned) |

---

## 3. Worth borrowing

### High value

1. **Structured "environment" block every turn**
   - **Roo:** Single `<environment_details>` with: workspace paths (first turn), visible files, open tabs, **actively running terminals** (command + new output), **recently modified files** ("re-read before editing"), time, git, mode, cost.
   - **Borrow:** Add an optional `environment_details`-style block (or a short "Context this turn:") to the *user* message each round: e.g. "Open tabs: …", "Recently modified: …", "Running terminals: …".
   - **Why:** Reduces "file changed since you read it" and "command already running" mistakes without changing your tool set.

2. **Explicit "objective" and "when you're done"**
   - **Roo:** "Work through goals sequentially"; "When you've completed the task, use attempt_completion"; "Do not end with questions or offers for further assistance."
   - **Borrow:** Add 2–3 lines to `BASE_AGENT_SYSTEM_MESSAGE` or TOOLS_SYSTEM_HINT: e.g. "Work through the user's request step by step using tools. When the request is fully done, respond with a brief final summary and do not ask follow-up questions unless something is unclear."
   - **Why:** Matches your "stop when no tool calls" behavior and reduces unnecessary "Is there anything else?" endings.

3. **Anti-conversational rule**
   - **Roo:** "STRICTLY FORBIDDEN from starting with 'Great', 'Certainly', 'Okay', 'Sure'. Be direct and technical."
   - **Borrow:** One sentence in the system or tools hint: "Be direct and technical; do not start replies with pleasantries like 'Great' or 'Certainly'."
   - **Why:** Keeps answers short and tool-focused; less filler.

### Medium value

4. **Tool-use guidelines (short)**
   - **Roo:** "Assess what you have vs need"; "Choose the most appropriate tool"; "Do not assume the outcome of any tool use."
   - **Borrow:** You already say "use read_file just before editing" and "reading files alone does not fulfill an edit request." Optionally add: "After each tool result, decide the next step from the result; do not assume success."
   - **Why:** Reinforces using tool output and re-reading after edits.

5. **"Recently modified" list**
   - **Roo:** In env_details: "Recently modified files (re-read before editing): …".
   - **Borrow:** If you track edited paths in the session, append a line like "Recently modified this session: path1, path2. Re-read before editing again." to the user or context block for the next round.
   - **Why:** Cheap and directly addresses stale reads after edits.

6. **Markdown rule for code/filenames**
   - **Roo:** "ALL responses MUST show language construct OR filename as [\`name\`](path:line); line required for syntax."
   - **Borrow:** Only if you care about clickable refs in the UI; otherwise skip.

### Lower priority / structural

7. **Sectioned system prompt**
   - **Roo:** Role, rules, objective, capabilities, modes, skills, custom instructions as separate sections.
   - **Borrow:** Not required for parity. You can get most benefit by adding a few bullets (objective, anti-conversational, "re-read after edit") to your existing single block. A full sectioned builder is optional later (e.g. if you add modes).

8. **Modes and roleDefinition**
   - **Roo:** Architect / Code / Ask / Debug / Orchestrator with different roles and tool sets.
   - **Borrow:** Only if you introduce modes (e.g. "plan only" vs "implement"). Then a short "role" line per mode (like Roo's roleDefinition) is useful.

9. **Rebuild system prompt every request**
   - **Roo:** System prompt rebuilt each API call (mode, MCP, skills can change).
   - **Borrow:** Only if you add modes or dynamic tool lists; otherwise one system message per run is fine.

---

## 4. Summary

- **Prompt generation:** Roo uses a long, sectioned system prompt (role, rules, objective, capabilities, modes, skills, custom) and puts **all** varying context into a single `<environment_details>` block in the user message each turn. Moonlet uses one system block + tools hint and injects workspace/code/folder/ref once.
- **Flow:** Roo is true multi-turn with per-turn env refresh and optional condensing; Moonlet is either one growing prompt (completion) or a message list without per-turn environment.

**Highest-impact borrows:** (1) Per-turn environment block (or minimal "context this turn") including recently modified and running terminals, (2) explicit "when you're done, give a brief summary and don't ask follow-ups," and (3) one-line "be direct, no pleasantries" rule. The rest can be added incrementally if you want closer Roo-like behavior or add modes.
