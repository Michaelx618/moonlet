# Continue context handling (study from their source)

Short summary of how Continue handles multiple files and context. **Source used:** `/Users/michael/continue-main` (you can grep it; workspace may be moonlet so use terminal e.g. `rg "getContextItems" /Users/michael/continue-main -g "*.ts"`). Also: `vendor/continue` in this repo (often cursorignored).

## Context providers (multiple files)

Continue uses **context providers** that return **context items**. Each item has:

- `name` – display name (e.g. base name of file)
- `description` – e.g. relative path or “last 2 path parts”
- `content` – the string that gets injected into the prompt (often full file content in a code block)
- `uri` – `{ type: "file", value: filepath }`

Key providers we care about:

| Provider | Role | Multiple files? | Truncation in provider? |
|--------|------|------------------|--------------------------|
| **CurrentFileContextProvider** | “Currently open file” | No (one file) | No – uses `currentFile.contents` in full |
| **FileContextProvider** | @File – user picks a file | One file per selection; user can add multiple @File | No – `content` is full file from `ide.readFile(fileUri)` |
| **OpenFilesContextProvider** | @Open – all open (or pinned) files | Yes – returns one item per open file | No – each file’s content is full in its item |

So at the **provider** level they do **not** truncate: each context item gets the full file content. Truncation/limits almost certainly happen later when they **assemble** the prompt (total token budget across all items).

## Format of file content in the prompt

- **Current file:**  
  `This is the currently open file:\n\n\`\`\`${relativePathOrBasename}\n${currentFile.contents}\n\`\`\``  
  So: prefix + path + code block with full contents.

- **@File:**  
  `\`\`\`${relativePathOrBasename}\n${content}\n\`\`\``  
  Path in the code block header, then full file content.

- **@Open:**  
  Same pattern per open file: path + code block with full content.

So every file is “path + full content in a block”. No per-file character cap in the provider code.

## How context gets into the prompt (from continue-main)

- **constructMessages** (`gui/src/redux/util/constructMessages.ts`): For each user message, it prepends all that message's context items: `ctxItemParts = item.contextItems.map(ctxItem => ({ type: "text", text: ctxItem.content + "\\n" }))`, then `content = [...ctxItemParts, ...content]`. Full content per item; no truncation.
- **renderContextItems** (`core/util/messageContent.ts`): `contextItems.map((item) => item.content).join("\\n\\n")` — no truncation.
- So provider → constructMessages → message does not trim. Any limit is when sending to the LLM (context window / max_tokens).

## Other providers (for reference)

They also have: CodebaseContextProvider, CodeContextProvider (symbols), FolderContextProvider, DiffContextProvider, RepoMapContextProvider, TerminalContextProvider, DocsContextProvider, etc. See `core/context/providers/index.ts`.

## Takeaways for us

1. **No per-file truncation (we match Continue)** – They push full file content per item; we cap per file (e.g. MAX_FOCUS_CONTENT_CHARS, MAX_REF_CONTENT_CHARS). We’ve increased those caps (e.g. 12k) so we’re less likely to cut off the exact line the model needs.
2. **Multiple files** – They support multiple files by having multiple context items (e.g. one @File per selection, or @Open = one item per open file). We do something similar: one “Current file” + “Reference files” list; we just cap each chunk.
3. **Truncation** – Their truncation is likely at **assembly** time (total token count / context window), not per item. We could later move to a similar model: no per-file cap in the “section” builders, then a single pass that trims or prunes to fit a total token budget.
4. **Path in block** – They put the path in the code block header so the model knows which file is which; we do the same (“Current file (path for tools: …)”, “Reference files … path …”).

## Where to look next

**In `/Users/michael/continue-main` (grep from terminal):**

- **Prompt assembly / token budget:** `rg "countTokens|maxTokens|truncat" /Users/michael/continue-main -g "*.ts"` — token limits may be in config or when building the request.
- **constructMessages:** `gui/src/redux/util/constructMessages.ts` — how history + contextItems become ChatMessage[].
- **messageContent:** `core/util/messageContent.ts` — `renderContextItems`, `normalizeToMessageParts`.
- **Default system message:** `core/llm/defaultSystemMessages.ts` (see BORROWED_FROM_CONTINUE) — prompt wording.
- **Agent flow:** `docs.continue.dev/ide-extensions/agent/how-it-works` — how tools and context are sent.
