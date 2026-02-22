import { Compartment, EditorState, RangeSetBuilder, StateEffect, StateField } from "@codemirror/state";
import {
  Decoration,
  EditorView,
  drawSelection,
  highlightActiveLine,
  highlightActiveLineGutter,
  highlightSpecialChars,
  keymap,
  lineNumbers,
} from "@codemirror/view";
import { defaultKeymap, history, historyKeymap, indentWithTab } from "@codemirror/commands";
import { completionKeymap } from "@codemirror/autocomplete";
import {
  HighlightStyle,
  bracketMatching,
  defaultHighlightStyle,
  indentOnInput,
  syntaxHighlighting,
} from "@codemirror/language";
import { searchKeymap } from "@codemirror/search";
import { cpp } from "@codemirror/lang-cpp";
import { css } from "@codemirror/lang-css";
import { html } from "@codemirror/lang-html";
import { javascript } from "@codemirror/lang-javascript";
import { markdown } from "@codemirror/lang-markdown";
import { python } from "@codemirror/lang-python";
import { tags } from "@lezer/highlight";
import {
  createLspClient,
  fromFileUri,
  isLspSupportedPath,
  lspLanguageIdForPath,
} from "./lsp/client.js";
import {
  createLspCodeMirrorPlugins,
  lspPositionToOffset,
} from "./lsp/codemirror-plugins.js";

const params = new URLSearchParams(window.location.search);
      let API_BASE = params.get("api_base") || "http://127.0.0.1:8000";
      const filesEl = document.getElementById("files");
      const cmHostEl = document.getElementById("cmHost");
      const imagePreviewWrap = document.getElementById("imagePreviewWrap");
      const imagePreviewEl = document.getElementById("imagePreview");
      let currentFileIsBinary = false;
      const responseEl = document.getElementById("response");
      const promptEl = document.getElementById("prompt");
      const modeEl = document.getElementById("mode");
      const statusEl = document.getElementById("status");
      const statusDot = document.getElementById("statusDot");
      const rootEl = document.getElementById("root");
      const cancelBtn = document.getElementById("cancel");
      const refreshBtn = document.getElementById("refresh");
      const pickRootBtn = document.getElementById("pickRoot");
      const clearRootBtn = document.getElementById("clearRoot");
      const tabNameEl = document.getElementById("tabName");
      const runTimerEl = document.getElementById("runTimer");
      const statusBarFileEl = document.getElementById("statusBarFile");
      const unsavedDot = document.getElementById("unsavedDot");
      const welcomeScreen = document.getElementById("welcomeScreen");
      const welcomeSelectFolderBtn = document.getElementById("welcomeSelectFolderBtn");
      const welcomeCreateDirBtn = document.getElementById("welcomeCreateDirBtn");
      const statusBarCursorEl = document.getElementById("statusBarCursor");
      const statusBarLangEl = document.getElementById("statusBarLang");
      const statusBarInfoEl = document.getElementById("statusBarInfo");
      const titleFilepath = document.getElementById("titleFilepath");
      const terminalResizeHandle = document.getElementById("terminalResizeHandle");
      const rightPanelResize = document.getElementById("rightPanelResize");
      const centerTerminal = document.getElementById("centerTerminal");
      const runFileBtn = document.getElementById("runFileBtn");
      const openTerminalBtn = document.getElementById("openTerminalBtn");
      const newChatBtn = document.getElementById("newChatBtn");
      const terminalOutputEl = document.getElementById("terminalOutput");
      const terminalCloseBtn = document.getElementById("terminalCloseBtn");
      const explorerMenuEl = document.getElementById("explorerMenu");
      const menuNewFileEl = document.getElementById("menuNewFile");
      const menuNewFolderEl = document.getElementById("menuNewFolder");
      const menuRenameFileEl = document.getElementById("menuRenameFile");
      const menuDeleteFileEl = document.getElementById("menuDeleteFile");
      const panelTranscript = [];
      let activeTranscriptIndex = -1;
      const PANEL_TRANSCRIPT_LIMIT = 7;
      const PANEL_USER_TEXT_MAX_CHARS = 180;
      const PANEL_ACTIONS_LIMIT = 24;
      const LAST_SELECTED_ROOT_KEY = "moonlet_last_selected_root_v1";
      const LAST_OPENED_FILE_BY_ROOT_KEY = "moonlet_last_opened_file_by_root_v1";
      const expandedUserTurnIds = new Set();
      let panelTurnSeq = 1;
      let includePaths = [];
      let hasUserImported = false;
      let currentFile = null;
      let fileCache = {};
      let suppressViewerEvents = false;
      let saveTimer = null;
      let loadSeq = 0;
      let activeLoadSeq = 0;
      let isLoadingFiles = false;
      let activeRequest = null;
      let serverReady = false;
      let isUnsaved = false;
      let terminalStarted = false;
      let terminalVisible = false;
      let terminalIsPty = false;
      let terminalActiveCwdAbs = "";
      let terminalFallbackLineBuffer = "";
      let terminalFallbackHistory = [];
      let terminalFallbackHistoryIdx = -1;
      let terminalFallbackDraft = "";
      let xterm = null;
      let xtermFit = null;
      let explorerMenuTarget = "";
      let explorerEntries = [];
      const expandedFolders = new Set();
      let lspDocChangeTimer = null;
      let lspCurrentAbsPath = "";

      if (/Mac|iPhone|iPad|iPod/.test(navigator.platform)) {
        document.body.classList.add("darwin");
      }

      // ─── File icon mapping (text-based, no emojis) ───
      function getFileIcon(name) {
        if ((name || "").endsWith("/")) return "📁";
        const ext = name.split('.').pop().toLowerCase();
        const icons = {
          py: 'py', js: 'js', ts: 'ts', tsx: 'tx',
          jsx: 'jx', json: '{}', md: 'md', html: '<>',
          css: '#', rs: 'rs', go: 'go', java: 'jv',
          yaml: 'ym', yml: 'ym', txt: 'tx', c: '.c', cpp: 'c+',
          h: '.h', hpp: 'h+', sh: 'sh', sql: 'sq',
        };
        return icons[ext] || '··';
      }

      function _buildExplorerTree(entries) {
        const root = { name: "", path: "", isDir: true, children: new Map() };
        for (const raw of (entries || [])) {
          const path = String(raw || "");
          if (!path) continue;
          const isDir = path.endsWith("/");
          const clean = isDir ? path.slice(0, -1) : path;
          if (!clean) continue;
          const parts = clean.split("/").filter(Boolean);
          let node = root;
          for (let i = 0; i < parts.length; i++) {
            const part = parts[i];
            const isLeaf = i === parts.length - 1;
            const childIsDir = isLeaf ? isDir : true;
            const full = parts.slice(0, i + 1).join("/");
            let child = node.children.get(part);
            if (!child) {
              child = {
                name: part,
                path: childIsDir ? `${full}/` : full,
                isDir: childIsDir,
                children: new Map(),
              };
              node.children.set(part, child);
            } else if (childIsDir && !child.isDir) {
              child.isDir = true;
              child.path = `${full}/`;
            }
            node = child;
          }
        }
        return root;
      }

      function _sortedChildren(node) {
        return Array.from(node.children.values()).sort((a, b) => {
          if (a.isDir !== b.isDir) return a.isDir ? -1 : 1;
          return a.name.localeCompare(b.name);
        });
      }

      function renderExplorerList() {
        const tree = _buildExplorerTree(explorerEntries);
        filesEl.innerHTML = "";
        const activePath = currentFile || "";

        const renderNode = (node, depth) => {
          for (const child of _sortedChildren(node)) {
            const li = document.createElement("li");
            li.dataset.path = child.path;
            li.style.paddingLeft = `${8 + depth * 14}px`;
            if (activePath && activePath === child.path) li.classList.add("active");

            const expander = document.createElement("button");
            expander.className = "folder-expander";
            if (child.isDir) {
              const hasChildren = child.children && child.children.size > 0;
              if (!hasChildren) expander.classList.add("hidden");
              const expanded = expandedFolders.has(child.path);
              expander.textContent = expanded ? "▾" : "▸";
              expander.onclick = (e) => {
                e.preventDefault();
                e.stopPropagation();
                if (expandedFolders.has(child.path)) expandedFolders.delete(child.path);
                else expandedFolders.add(child.path);
                renderExplorerList();
              };
            } else {
              expander.classList.add("hidden");
              expander.textContent = " ";
            }
            li.appendChild(expander);

            const icon = document.createElement("span");
            icon.className = "file-icon";
            icon.textContent = getFileIcon(child.path);
            li.appendChild(icon);

            const name = document.createElement("span");
            name.className = "file-name";
            name.textContent = child.isDir ? `${child.name}/` : child.name;
            li.appendChild(name);

            li.onclick = () => selectFile(child.path, li);
            filesEl.appendChild(li);

            if (child.isDir && expandedFolders.has(child.path)) {
              renderNode(child, depth + 1);
            }
          }
        };

        renderNode(tree, 0);
      }

      // ─── Language detection ───
      function getLanguageName(name) {
        if (!name) return 'Plain Text';
        const ext = name.split('.').pop().toLowerCase();
        const langs = {
          py: 'Python', js: 'JavaScript', ts: 'TypeScript', tsx: 'TypeScript React',
          jsx: 'JavaScript React', json: 'JSON', md: 'Markdown', html: 'HTML',
          css: 'CSS', scss: 'SCSS', less: 'LESS', rs: 'Rust', go: 'Go',
          java: 'Java', kt: 'Kotlin', c: 'C', cpp: 'C++', h: 'C Header',
          hpp: 'C++ Header', cs: 'C#', php: 'PHP', rb: 'Ruby',
          sh: 'Shell', bash: 'Bash', zsh: 'Zsh', sql: 'SQL',
          yaml: 'YAML', yml: 'YAML', txt: 'Plain Text', xml: 'XML',
          swift: 'Swift', scala: 'Scala', vue: 'Vue', svelte: 'Svelte',
        };
        return langs[ext] || 'Plain Text';
      }

      const HIGHLIGHT_CHAR_CAP = 200000;
      const HIGHLIGHT_LINE_CAP = 4000;
      let modelRunning = false;
      let editorView = null;
      const languageCompartment = new Compartment();
      const behaviorCompartment = new Compartment();
      const lspCompartment = new Compartment();
      const setDiffDecorationsEffect = StateEffect.define();
      const clearDiffDecorationsEffect = StateEffect.define();

      function _isLargeFile(code) {
        if (code.length > HIGHLIGHT_CHAR_CAP) return true;
        let lines = 0;
        let idx = -1;
        while ((idx = code.indexOf("\n", idx + 1)) !== -1) {
          if (++lines > HIGHLIGHT_LINE_CAP) return true;
        }
        return false;
      }

      function languageExtensionForPath(name) {
        const ext = ((name || "").split(".").pop() || "").toLowerCase();
        if (ext === "js" || ext === "jsx") return javascript({ jsx: true });
        if (ext === "ts" || ext === "tsx") return javascript({ typescript: true, jsx: true });
        if (ext === "py") return python();
        if (ext === "html" || ext === "htm") return html();
        if (ext === "css" || ext === "scss" || ext === "less") return css();
        if (ext === "md" || ext === "markdown") return markdown();
        if (ext === "c" || ext === "h" || ext === "cpp" || ext === "cxx" || ext === "cc" || ext === "hpp") return cpp();
        return [];
      }

      function behaviorExtensionsForText(text) {
        if (_isLargeFile(text || "")) return [];
        return [bracketMatching(), indentOnInput()];
      }

      function buildDiffDecorations(doc, addedLines, deletedAtLines) {
        const builder = new RangeSetBuilder();
        const add = Array.isArray(addedLines) ? addedLines : [];
        const del = Array.isArray(deletedAtLines) ? deletedAtLines : [];
        for (const lineNo of add) {
          if (!Number.isInteger(lineNo) || lineNo < 1 || lineNo > doc.lines) continue;
          const line = doc.line(lineNo);
          builder.add(line.from, line.from, Decoration.line({ class: "cm-diff-add" }));
        }
        for (const lineNo of del) {
          if (!Number.isInteger(lineNo) || lineNo < 1 || lineNo > doc.lines) continue;
          const line = doc.line(lineNo);
          builder.add(line.from, line.from, Decoration.line({ class: "cm-diff-del-anchor" }));
        }
        return builder.finish();
      }

      const diffDecorationField = StateField.define({
        create() {
          return Decoration.none;
        },
        update(value, tr) {
          let next = value.map(tr.changes);
          for (const effect of tr.effects) {
            if (effect.is(clearDiffDecorationsEffect)) return Decoration.none;
            if (effect.is(setDiffDecorationsEffect)) {
              const payload = effect.value || {};
              return buildDiffDecorations(
                tr.state.doc,
                payload.addedLines || [],
                payload.deletedAtLines || []
              );
            }
          }
          return next;
        },
        provide: (field) => EditorView.decorations.from(field),
      });

      const cmTheme = EditorView.theme(
        {
          "&": {
            color: "var(--text)",
            backgroundColor: "var(--bg-base)",
          },
          ".cm-scroller": {
            fontFamily: "var(--font-mono)",
            lineHeight: "1.6",
          },
          ".cm-content": {
            padding: "12px 16px",
            minHeight: "100%",
          },
          ".cm-line": {
            padding: "0",
          },
          ".cm-gutters": {
            borderRight: "1px solid var(--border)",
          },
          ".cm-lineNumbers .cm-gutterElement": {
            padding: "0 8px 0 0",
            minWidth: "36px",
          },
        },
        { dark: true }
      );

      const cmHighlightStyle = HighlightStyle.define([
        { tag: tags.keyword, color: "var(--mauve)" },
        { tag: tags.string, color: "var(--green)" },
        { tag: tags.number, color: "var(--peach)" },
        { tag: tags.comment, color: "var(--surface2)", fontStyle: "italic" },
        { tag: [tags.function(tags.variableName), tags.className], color: "var(--accent)" },
      ]);

      function createEditor() {
        if (!cmHostEl) return;
        const initialText = "";
        const baseExtensions = [
          lineNumbers(),
          highlightSpecialChars(),
          drawSelection(),
          history(),
          highlightActiveLine(),
          highlightActiveLineGutter(),
          syntaxHighlighting(defaultHighlightStyle, { fallback: true }),
          syntaxHighlighting(cmHighlightStyle),
          cmTheme,
          diffDecorationField,
          languageCompartment.of([]),
          behaviorCompartment.of(behaviorExtensionsForText(initialText)),
          lspCompartment.of([]),
          keymap.of([
            indentWithTab,
            ...defaultKeymap,
            ...historyKeymap,
            ...searchKeymap,
            ...completionKeymap,
          ]),
          EditorView.updateListener.of((update) => {
            if (update.selectionSet) updateCursorPosition();
            if (update.docChanged) {
              if (!suppressViewerEvents) {
                markUnsaved();
                setStatus("unsaved", "busy");
                scheduleAutoSave();
                scheduleLspDidChange();
              }
              configureEditorForDocument(currentFile, update.state.doc.toString());
            }
          }),
        ];

        editorView = new EditorView({
          state: EditorState.create({
            doc: initialText,
            extensions: baseExtensions,
          }),
          parent: cmHostEl,
        });
      }

      function getEditorText() {
        return editorView ? editorView.state.doc.toString() : "";
      }

      function setEditorText(text) {
        if (!editorView) return;
        const next = String(text ?? "");
        if (next === getEditorText()) return;
        suppressViewerEvents = true;
        editorView.dispatch({
          changes: { from: 0, to: editorView.state.doc.length, insert: next },
        });
        suppressViewerEvents = false;
      }

      function configureEditorForDocument(path, text) {
        if (!editorView) return;
        const large = _isLargeFile(text || "");
        const language = large ? [] : languageExtensionForPath(path);
        const lspExt = lspEnabledForFile(path, text) ? lspPlugins.extensions : [];
        editorView.dispatch({
          effects: [
            languageCompartment.reconfigure(language),
            behaviorCompartment.reconfigure(behaviorExtensionsForText(text || "")),
            lspCompartment.reconfigure(lspExt),
          ],
        });
      }

      function setDiffDecorations(addedLines, deletedAtLines) {
        if (!editorView) return;
        editorView.dispatch({
          effects: setDiffDecorationsEffect.of({
            addedLines: Array.from(addedLines || []),
            deletedAtLines: Array.from(deletedAtLines || []),
          }),
        });
      }

      function clearDiffDecorations() {
        if (!editorView) return;
        editorView.dispatch({ effects: clearDiffDecorationsEffect.of(null) });
      }

      // ─── Cursor position tracking ───
      function updateCursorPosition() {
        if (!editorView) {
          statusBarCursorEl.textContent = "Ln 1, Col 1";
          return;
        }
        const pos = editorView.state.selection.main.head;
        const line = editorView.state.doc.lineAt(pos);
        const col = pos - line.from + 1;
        statusBarCursorEl.textContent = `Ln ${line.number}, Col ${col}`;
      }

      // ─── Unsaved indicator ───
      function markUnsaved() {
        if (!isUnsaved) {
          isUnsaved = true;
          unsavedDot.classList.add("visible");
        }
      }
      function markSaved() {
        isUnsaved = false;
        unsavedDot.classList.remove("visible");
      }

      // ─── Welcome screen ───
      function showWelcome() {
        welcomeScreen.style.display = "";
        if (cmHostEl) cmHostEl.style.display = "none";
        if (imagePreviewWrap) imagePreviewWrap.style.display = "none";
      }
      function hideWelcome() {
        welcomeScreen.style.display = "none";
        if (currentFileIsBinary) {
          if (imagePreviewWrap) imagePreviewWrap.style.display = "flex";
          if (cmHostEl) cmHostEl.style.display = "none";
        } else {
          if (imagePreviewWrap) imagePreviewWrap.style.display = "none";
          if (cmHostEl) cmHostEl.style.display = "";
        }
      }

      function applyModeUI(mode) {
        const next = mode === "chat" ? "chat" : "agent";
        modeEl.value = next;
        localStorage.setItem("moonlet_mode", next);
        promptEl.placeholder = next === "chat"
          ? "Chat with Moonlet..."
          : "Ask the agent... or /run <command>";
      }
      const savedMode = localStorage.getItem("moonlet_mode") || "agent";
      applyModeUI(savedMode);
      modeEl.addEventListener("change", () => applyModeUI(modeEl.value));
      createEditor();

      // ─── Timer ───
      let timerInterval = null;
      let timerStart = 0;
      function startTimer() {
        timerStart = Date.now();
        runTimerEl.className = "run-timer running";
        runTimerEl.textContent = "0.0s";
        if (timerInterval) clearInterval(timerInterval);
        timerInterval = setInterval(() => {
          const elapsed = ((Date.now() - timerStart) / 1000).toFixed(1);
          runTimerEl.textContent = `${elapsed}s`;
        }, 100);
      }
      function stopTimer(type) {
        if (timerInterval) { clearInterval(timerInterval); timerInterval = null; }
        const elapsed = ((Date.now() - timerStart) / 1000).toFixed(1);
        runTimerEl.textContent = `${elapsed}s`;
        runTimerEl.className = `run-timer ${type || 'done'}`;
      }

      // ─── Status ───
      function setStatus(text, type) {
        statusEl.textContent = text;
        statusDot.className = "status-dot";
        if (type === "busy") statusDot.classList.add("busy");
        else if (type === "error") statusDot.classList.add("error");
      }

      function setTerminalVisible(visible) {
        terminalVisible = Boolean(visible);
        if (centerTerminal) {
          centerTerminal.classList.toggle("hidden", !terminalVisible);
        }
        if (terminalVisible) {
          setTimeout(() => {
            try { if (xtermFit) xtermFit.fit(); } catch {}
            try { editorView && editorView.contentDOM.blur(); } catch {}
            try { promptEl.blur(); } catch {}
            try {
              if (xterm) xterm.focus();
              else if (terminalOutputEl) terminalOutputEl.focus();
            } catch {}
          }, 0);
        }
      }

      function appendTerminal(text, stream) {
        if (!text) return;
        if (xterm) {
          xterm.write(text);
        } else {
          terminalOutputEl.textContent += text;
          terminalOutputEl.scrollTop = terminalOutputEl.scrollHeight;
        }
      }

      function writeTerminalLocal(text) {
        if (!text) return;
        if (xterm) xterm.write(text);
        else appendTerminal(text, "stdout");
      }

      function handleTerminalInputData(data) {
        if (!data) return;
        if (!window.electronAPI || !window.electronAPI.terminalWrite) return;

        // PTY mode: passthrough to real terminal process.
        if (terminalIsPty) {
          window.electronAPI.terminalWrite(data);
          return;
        }

        // Fallback mode (no PTY): local line editing + simple history.
        if (data === "\u001b[A") { // ArrowUp: previous command
          if (terminalFallbackHistory.length === 0) return;
          if (terminalFallbackHistoryIdx < 0) {
            terminalFallbackDraft = terminalFallbackLineBuffer;
            terminalFallbackHistoryIdx = terminalFallbackHistory.length - 1;
          } else if (terminalFallbackHistoryIdx > 0) {
            terminalFallbackHistoryIdx -= 1;
          }
          const next = terminalFallbackHistory[terminalFallbackHistoryIdx] || "";
          if (terminalFallbackLineBuffer.length > 0) {
            writeTerminalLocal("\b \b".repeat(terminalFallbackLineBuffer.length));
          }
          terminalFallbackLineBuffer = next;
          if (next) writeTerminalLocal(next);
          return;
        }
        if (data === "\u001b[B") { // ArrowDown: newer command / draft
          if (terminalFallbackHistory.length === 0) return;
          if (terminalFallbackHistoryIdx < 0) return;
          if (terminalFallbackHistoryIdx < terminalFallbackHistory.length - 1) {
            terminalFallbackHistoryIdx += 1;
            const next = terminalFallbackHistory[terminalFallbackHistoryIdx] || "";
            if (terminalFallbackLineBuffer.length > 0) {
              writeTerminalLocal("\b \b".repeat(terminalFallbackLineBuffer.length));
            }
            terminalFallbackLineBuffer = next;
            if (next) writeTerminalLocal(next);
            return;
          }
          // Back to in-progress draft
          if (terminalFallbackLineBuffer.length > 0) {
            writeTerminalLocal("\b \b".repeat(terminalFallbackLineBuffer.length));
          }
          terminalFallbackLineBuffer = terminalFallbackDraft || "";
          terminalFallbackHistoryIdx = -1;
          if (terminalFallbackLineBuffer) writeTerminalLocal(terminalFallbackLineBuffer);
          return;
        }

        if (data === "\r" || data === "\n") {
          writeTerminalLocal("\r\n");
          const submitted = terminalFallbackLineBuffer || "";
          window.electronAPI.terminalWrite(submitted + "\n");
          if (submitted.trim()) {
            const last = terminalFallbackHistory[terminalFallbackHistory.length - 1] || "";
            if (submitted !== last) terminalFallbackHistory.push(submitted);
          }
          terminalFallbackLineBuffer = "";
          terminalFallbackHistoryIdx = -1;
          terminalFallbackDraft = "";
          return;
        }
        if (data === "\u007f") { // Backspace
          if (terminalFallbackLineBuffer.length > 0) {
            terminalFallbackLineBuffer = terminalFallbackLineBuffer.slice(0, -1);
            writeTerminalLocal("\b \b");
          }
          return;
        }
        if (data === "\u0003") { // Ctrl+C
          window.electronAPI.terminalWrite("\u0003");
          terminalFallbackLineBuffer = "";
          terminalFallbackHistoryIdx = -1;
          terminalFallbackDraft = "";
          writeTerminalLocal("^C\r\n");
          return;
        }
        // Ignore escape sequences (arrows, etc.) in fallback mode for now.
        if (data.startsWith("\u001b")) return;

        terminalFallbackLineBuffer += data;
        writeTerminalLocal(data);
      }

      function handleTerminalPasteText(text) {
        if (!text) return;
        if (!window.electronAPI || !window.electronAPI.terminalWrite) return;
        const normalized = text.replace(/\r\n/g, "\n").replace(/\r/g, "\n");

        // In fallback mode:
        // - single-line paste behaves like typing (local echo + editable)
        // - multi-line paste is sent raw to shell to avoid duplicate visual echo
        if (!terminalIsPty) {
          if (normalized.includes("\n")) {
            terminalFallbackLineBuffer = "";
            terminalFallbackHistoryIdx = -1;
            terminalFallbackDraft = "";
          } else {
            writeTerminalLocal(normalized);
            terminalFallbackLineBuffer += normalized;
          }
        }

        window.electronAPI.terminalWrite(text);
      }

      function keyEventToTerminalData(e) {
        const keyMap = {
          Enter: "\n",
          Backspace: "\x7f",
          Tab: "\t",
          ArrowUp: "\x1b[A",
          ArrowDown: "\x1b[B",
          ArrowRight: "\x1b[C",
          ArrowLeft: "\x1b[D",
          Home: "\x1b[H",
          End: "\x1b[F",
          Delete: "\x1b[3~",
          Escape: "\x1b",
        };
        let data = keyMap[e.key];
        if (!data && e.key && e.key.length === 1 && !e.altKey) data = e.key;
        return data || "";
      }

      function ensureXterm() {
        if (xterm) return;
        if (!window.Terminal || !window.FitAddon) {
          if (!terminalOutputEl.dataset.xtermMissingShown) {
            terminalOutputEl.dataset.xtermMissingShown = "1";
            appendTerminal(
              "[terminal] xterm not installed yet; using fallback input mode.\n",
              "stdout",
            );
          }
          return;
        }
        xterm = new window.Terminal({
          cursorBlink: true,
          convertEol: true,
          fontFamily: "JetBrains Mono, Fira Code, SF Mono, Cascadia Code, Consolas, monospace",
          fontSize: 12,
          theme: {
            background: "#11111b",
            foreground: "#cdd6f4",
            cursor: "#89b4fa",
          },
        });
        xtermFit = new window.FitAddon.FitAddon();
        xterm.loadAddon(xtermFit);
        xterm.open(terminalOutputEl);
        terminalOutputEl.tabIndex = 0;
        terminalOutputEl.addEventListener("mousedown", () => {
          setTimeout(() => {
            try { xterm.focus(); } catch {}
          }, 0);
        });
        try { xtermFit.fit(); } catch {}
        xterm.onData((data) => {
          handleTerminalInputData(data);
        });
        const notifyResize = () => {
          if (!xterm || !xtermFit) return;
          try { xtermFit.fit(); } catch {}
          if (window.electronAPI) {
            const cols = xterm.cols || 0;
            const rows = xterm.rows || 0;
            if (window.electronAPI.terminalResize) {
              window.electronAPI.terminalResize({ cols, rows });
            } else if (window.electronAPI.terminalWrite) {
              // no-op fallback: backend can ignore missing resize API
            }
          }
        };
        window.addEventListener("resize", notifyResize);
        const ro = new ResizeObserver(notifyResize);
        ro.observe(terminalOutputEl);
      }

      function relCwdForCurrentFile() {
        if (!currentFile || !currentFile.includes("/")) return "";
        return currentFile.split("/").slice(0, -1).join("/");
      }

      function normalizeAbsPath(p) {
        return String(p || "").replace(/\\/g, "/").replace(/\/+$/, "");
      }

      function rememberLastSelectedRoot(rootPath) {
        const rootAbs = normalizeAbsPath(rootPath || "");
        if (!rootAbs) return;
        try { localStorage.setItem(LAST_SELECTED_ROOT_KEY, rootAbs); } catch {}
      }

      function getLastSelectedRoot() {
        try { return normalizeAbsPath(localStorage.getItem(LAST_SELECTED_ROOT_KEY) || ""); } catch {}
        return "";
      }

      function clearLastSelectedRoot() {
        try { localStorage.removeItem(LAST_SELECTED_ROOT_KEY); } catch {}
      }

      function readLastOpenedMap() {
        try {
          const raw = localStorage.getItem(LAST_OPENED_FILE_BY_ROOT_KEY) || "{}";
          const parsed = JSON.parse(raw);
          if (parsed && typeof parsed === "object") return parsed;
        } catch {}
        return {};
      }

      function writeLastOpenedMap(map) {
        try {
          localStorage.setItem(LAST_OPENED_FILE_BY_ROOT_KEY, JSON.stringify(map || {}));
        } catch {}
      }

      function rememberLastOpenedFileForCurrentRoot(relPath) {
        const rootAbs = normalizeAbsPath(rootEl.value || "");
        const rel = String(relPath || "").trim();
        if (!rootAbs || !rel || rel.endsWith("/")) return;
        const map = readLastOpenedMap();
        map[rootAbs] = rel;
        writeLastOpenedMap(map);
      }

      function forgetLastOpenedFileForCurrentRoot(relPath) {
        const rootAbs = normalizeAbsPath(rootEl.value || "");
        const rel = String(relPath || "").trim();
        if (!rootAbs || !rel) return;
        const map = readLastOpenedMap();
        if (String(map[rootAbs] || "") === rel) {
          delete map[rootAbs];
          writeLastOpenedMap(map);
        }
      }

      async function restoreLastOpenedFileForCurrentRoot() {
        const rootAbs = normalizeAbsPath(rootEl.value || "");
        if (!rootAbs) return false;
        const map = readLastOpenedMap();
        const rel = String(map[rootAbs] || "").trim();
        if (!rel) return false;
        const li = Array.from(filesEl.children).find((n) => (n.dataset.path || "") === rel);
        if (!li) return false;
        await selectFile(rel, li);
        return true;
      }

      function absPathForRel(relPath) {
        const rootAbs = normalizeAbsPath(rootEl.value || "");
        const rel = String(relPath || "").replace(/^[/\\]+/, "");
        if (!rootAbs || !rel) return "";
        return normalizeAbsPath(`${rootAbs}/${rel}`);
      }

      function relPathFromAbs(absPath) {
        const rootAbs = normalizeAbsPath(rootEl.value || "");
        const abs = normalizeAbsPath(absPath || "");
        if (!rootAbs || !abs) return "";
        if (abs === rootAbs) return "";
        if (!abs.startsWith(rootAbs + "/")) return "";
        return abs.slice(rootAbs.length + 1);
      }

      const lspStateText = {
        off: "off",
        starting: "starting",
        indexing: "indexing",
        connected: "connected",
        error: "error",
        stopped: "stopped",
      };

      function setLspStatus(state, message = "") {
        const label = lspStateText[state] || String(state || "off");
        if (statusBarInfoEl) {
          statusBarInfoEl.textContent = `Moonlet · LSP: ${label}`;
          statusBarInfoEl.title = message || `LSP ${label}`;
        }
      }

      const lspPlugins = createLspCodeMirrorPlugins({
        getCurrentAbsPath: () => lspCurrentAbsPath || "",
        requestCompletion: async ({ absPath, position, context }) => {
          if (!lspClient) return null;
          return lspClient.requestCompletion({ absPath, position, context });
        },
        requestHover: async ({ absPath, position }) => {
          if (!lspClient) return null;
          return lspClient.requestHover({ absPath, position });
        },
        requestDefinition: async ({ absPath, position }) => {
          if (!lspClient) return null;
          return lspClient.requestDefinition({ absPath, position });
        },
        onDefinition: async (location) => jumpToDefinitionLocation(location),
      });

      const lspClient = createLspClient({
        electronAPI: window.electronAPI,
        onStatus: ({ state, message }) => {
          setLspStatus(state, message);
        },
        onDiagnostics: (uri, diagnostics) => {
          const absPath = fromFileUri(uri);
          if (!absPath) return;
          lspPlugins.updateDiagnostics(absPath, diagnostics, editorView);
        },
      });
      setLspStatus("off", "LSP idle");

      function lspEnabledForFile(relPath, text) {
        if (!relPath || currentFileIsBinary) return false;
        if (!isLspSupportedPath(relPath)) return false;
        if (!rootEl.value || !normalizeAbsPath(rootEl.value)) return false;
        if (_isLargeFile(text || "")) return false;
        return true;
      }

      async function closeLspForPath(relPath) {
        const absPath = absPathForRel(relPath);
        if (!absPath) return;
        try {
          await lspClient.closeDocument(absPath);
        } catch {
          // ignore close failures
        }
        lspPlugins.clearDiagnostics(absPath, editorView);
        if (lspCurrentAbsPath === absPath) {
          lspCurrentAbsPath = "";
        }
      }

      async function openLspForPath(relPath, text) {
        if (!lspEnabledForFile(relPath, text)) {
          await closeLspForPath(relPath);
          setLspStatus("off", "LSP disabled for this file");
          return;
        }

        const absPath = absPathForRel(relPath);
        const rootAbs = normalizeAbsPath(rootEl.value || "");
        const languageId = lspLanguageIdForPath(relPath);
        if (!absPath || !rootAbs || !languageId) return;

        const ready = await lspClient.ensureSession({ rootAbs, languageId });
        if (!ready) return;

        lspCurrentAbsPath = absPath;
        await lspClient.openDocument({
          absPath,
          languageId,
          text: String(text || ""),
        });
        lspPlugins.refreshDiagnostics(editorView);
      }

      function scheduleLspDidChange() {
        if (!currentFile || !lspCurrentAbsPath) return;
        if (lspDocChangeTimer) clearTimeout(lspDocChangeTimer);
        lspDocChangeTimer = setTimeout(async () => {
          if (!currentFile || !lspCurrentAbsPath) return;
          if (!lspEnabledForFile(currentFile, getEditorText())) return;
          try {
            await lspClient.changeDocument({
              absPath: lspCurrentAbsPath,
              text: getEditorText(),
            });
          } catch {
            // ignore change send failures
          }
        }, 150);
      }

      async function jumpToDefinitionLocation(location) {
        const uri = String(location && location.uri ? location.uri : "");
        const range = location && location.range ? location.range : null;
        const absPath = fromFileUri(uri);
        const relPath = relPathFromAbs(absPath);
        if (!absPath || !relPath) return false;

        let li = Array.from(filesEl.children).find((n) => n.dataset && n.dataset.path === relPath);
        if (!li) {
          await loadFiles();
          li = Array.from(filesEl.children).find((n) => n.dataset && n.dataset.path === relPath);
        }
        if (!li) return false;

        if (currentFile !== relPath) {
          await selectFile(relPath, li);
        }

        if (editorView && range && range.start) {
          const offset = lspPositionToOffset(editorView.state.doc, range.start);
          editorView.dispatch({
            selection: { anchor: offset },
            scrollIntoView: true,
          });
          editorView.focus();
          updateCursorPosition();
        }
        return true;
      }

      function terminalCwdAbsForCurrentFile() {
        const rootAbs = normalizeAbsPath(rootEl.value || "");
        if (!rootAbs) return "";
        const rel = relCwdForCurrentFile();
        if (!rel) return rootAbs;
        return normalizeAbsPath(`${rootAbs}/${rel}`);
      }

      async function ensureTerminalStarted() {
        ensureXterm();
        if (terminalStarted) return;
        if (!window.electronAPI || !window.electronAPI.terminalStart) return;
        const cwdAbs = terminalCwdAbsForCurrentFile() || normalizeAbsPath(rootEl.value || "");
        const res = await window.electronAPI.terminalStart({ cwd: cwdAbs });
        terminalStarted = Boolean(res && res.ok);
        terminalIsPty = Boolean(res && res.pty);
        if (terminalStarted) {
          terminalActiveCwdAbs = normalizeAbsPath(res.cwd || cwdAbs || "");
          appendTerminal(`Moonlet terminal started (${res.shell || "shell"})\n`, "stdout");
          appendTerminal(`cwd: ${res.cwd || cwdAbs || "."}\n\n`, "stdout");
          if (!terminalIsPty) {
            appendTerminal("[terminal] Fallback input mode: type command, press Enter to send.\n\n", "stderr");
          }
          if (xterm && xterm.cols && xterm.rows && window.electronAPI && window.electronAPI.terminalResize) {
            window.electronAPI.terminalResize({ cols: xterm.cols, rows: xterm.rows });
          }
        }
      }

      function sendTerminalCommand(cmd) {
        const text = String(cmd || "").trim();
        if (!text) return;
        if (!window.electronAPI || !window.electronAPI.terminalWrite) return;
        window.electronAPI.terminalWrite(text + "\n");
      }

      function inferRunCommand(pathValue) {
        if (!pathValue) return "";
        const ext = (pathValue.split(".").pop() || "").toLowerCase();
        const q = `"${pathValue}"`;
        if (ext === "py") return `python3 ${q}`;
        if (ext === "js") return `node ${q}`;
        if (ext === "ts") return `npx ts-node ${q}`;
        if (ext === "sh" || ext === "bash" || ext === "zsh") return `bash ${q}`;
        if (ext === "rb") return `ruby ${q}`;
        if (ext === "go") return `go run ${q}`;
        if (ext === "c") return `cc ${q} -o /tmp/moonlet_run && /tmp/moonlet_run`;
        if (ext === "cpp" || ext === "cc" || ext === "cxx") return `c++ ${q} -o /tmp/moonlet_run && /tmp/moonlet_run`;
        if (ext === "rs") return "cargo run";
        return "";
      }

      function updateRunButtonState() {
        const cmd = inferRunCommand(currentFile);
        const runnable = Boolean(currentFile && cmd);
        runFileBtn.disabled = !runnable;
        runFileBtn.style.opacity = runnable ? "1" : "0.45";
        runFileBtn.style.cursor = runnable ? "pointer" : "not-allowed";
        runFileBtn.title = runnable ? `Run ${currentFile}` : "Current file is not runnable";
      }

      function updateTerminalButtonState() {
        const hasImportedFiles = hasUserImported && filesEl.children.length > 0;
        openTerminalBtn.disabled = !hasImportedFiles;
        openTerminalBtn.style.opacity = hasImportedFiles ? "1" : "0.45";
        openTerminalBtn.style.cursor = hasImportedFiles ? "pointer" : "not-allowed";
        openTerminalBtn.title = hasImportedFiles
          ? "Open terminal"
          : "Import files first to enable terminal";
      }

      function summarizeDiff(diffText, fileName) {
        const lines = diffText.split("\n");
        let adds = 0, dels = 0;
        const newFuncs = [];
        const touchedCalls = new Set();
        const touchedFuncs = new Set();
        const cKeywords = new Set(["if","for","while","switch","return","sizeof"]);
        for (const l of lines) {
          const anyFunc = l.match(/^[ +\-]?\s*(?:[A-Za-z_][A-Za-z0-9_\s\*\(\)\[\],]*\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*\([^;]*\)\s*\{?\s*$/);
          if (anyFunc && !cKeywords.has(anyFunc[1])) touchedFuncs.add(anyFunc[1]);
          if (l.startsWith("+") && !l.startsWith("+++")) {
            adds++;
            // Detect new function/method/class definitions
            const fm = l.match(/^\+\s*(def |class |function |async function )(\w+)/);
            if (fm) newFuncs.push(fm[2]);
            const cdef = l.match(/^\+\s*(?:[A-Za-z_][A-Za-z0-9_\s\*\(\)\[\],]*\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*\([^;]*\)\s*\{?\s*$/);
            if (cdef && !cKeywords.has(cdef[1])) newFuncs.push(cdef[1]);
            const cm = l.match(/([A-Za-z_][A-Za-z0-9_]*)\s*\(/g) || [];
            for (const t of cm) {
              const name = t.replace(/\s*\($/, "");
              if (!cKeywords.has(name) && name.length > 2) touchedCalls.add(name);
            }
          } else if (l.startsWith("-") && !l.startsWith("---")) {
            dels++;
            const cm = l.match(/([A-Za-z_][A-Za-z0-9_]*)\s*\(/g) || [];
            for (const t of cm) {
              const name = t.replace(/\s*\($/, "");
              if (!cKeywords.has(name) && name.length > 2) touchedCalls.add(name);
            }
          }
        }
        // Extract file from diff header or use provided
        let file = fileName;
        for (const l of lines) {
          if (l.startsWith("+++ b/")) { file = l.slice(6); break; }
        }
        const parts = [];
        if (file) parts.push(`<span style="color:var(--accent);">${esc(file)}</span>`);
        const changes = [];
        if (adds) changes.push(`<span style="color:var(--green);">+${adds} lines</span>`);
        if (dels) changes.push(`<span style="color:var(--red);">-${dels} lines</span>`);
        if (changes.length) parts.push(changes.join(" "));
        if (newFuncs.length) {
          const uniq = [...new Set(newFuncs)].slice(0, 6);
          const names = uniq.map(n => `<code>${esc(n)}</code>`).join(", ");
          parts.push(`Functions defined/updated: ${names}`);
        }
        const touchedOnly = [...touchedFuncs].filter(n => !newFuncs.includes(n)).slice(0, 6);
        if (touchedOnly.length) {
          parts.push(`Functions touched: ${touchedOnly.map(n => `<code>${esc(n)}</code>`).join(", ")}`);
        }
        if (touchedCalls.size) {
          const uniqTouched = [...touchedCalls].filter(n => !newFuncs.includes(n)).slice(0, 6);
          if (uniqTouched.length) {
            parts.push(`Calls touched: ${uniqTouched.map(n => `<code>${esc(n)}</code>`).join(", ")}`);
          }
        }
        if (!parts.length) return "Changes applied.";
        return parts.join(" &middot; ");
      }

      function renderDiff(diffText) {
        if (!diffText || !diffText.trim()) return "";
        const lines = diffText.split("\n");
        const html = [];
        html.push('<div class="diff-viewer">');
        for (const line of lines) {
          if (line.startsWith("---") || line.startsWith("+++")) {
            html.push(`<span class="diff-line diff-header">${esc(line)}</span>`);
          } else if (line.startsWith("@@")) {
            html.push(`<span class="diff-line diff-hunk-header">${esc(line)}</span>`);
          } else if (line.startsWith("+")) {
            html.push(`<span class="diff-line diff-add">${esc(line)}</span>`);
          } else if (line.startsWith("-")) {
            html.push(`<span class="diff-line diff-del">${esc(line)}</span>`);
          } else {
            html.push(`<span class="diff-line diff-ctx">${esc(line)}</span>`);
          }
        }
        html.push("</div>");
        return html.join("");
      }
      function esc(s) {
        return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
      }

      function truncatePanelUserText(text) {
        const s = String(text || "").trim();
        if (s.length <= PANEL_USER_TEXT_MAX_CHARS) return s;
        return s.slice(0, PANEL_USER_TEXT_MAX_CHARS - 3) + "...";
      }

      function ensureTurnId(row) {
        if (!row) return "";
        if (!row.id) {
          row.id = `turn_${panelTurnSeq++}`;
        }
        return String(row.id);
      }

      function normalizeActionText(item) {
        if (!item) return "";
        if (typeof item === "string") return item.trim();
        const type = String(item.type || "").trim();
        if (type === "tool_call") {
          const tool = String(item.tool || "tool");
          const args = item.args && typeof item.args === "object" ? item.args : {};
          const path = String(args.path || args.filepath || "").trim();
          const cmd = String(args.command || "").trim();
          const pattern = String(args.pattern || "").trim();
          const query = String(args.query || "").trim();
          if (tool === "read_file") return path ? `Read ${path}` : "Read file";
          if (tool === "list_files" || tool === "view_subdirectory") return path ? `List ${path}` : "List files";
          if (tool === "grep" || tool === "grep_search") return pattern ? `Search: ${pattern.slice(0, 50)}${pattern.length > 50 ? "…" : ""}` : "Search";
          if (tool === "search_replace" || tool === "edit_existing_file") return path ? `Edit ${path}` : "Edit file";
          if (tool === "write_file" || tool === "create_new_file") return path ? `Write ${path}` : "Write file";
          if (tool === "multi_edit") return path ? `Edit ${path} (multiple)` : "Multi-edit";
          if (tool === "run_terminal_cmd" || tool === "run_terminal_command") return cmd ? `Run: ${cmd.slice(0, 48)}${cmd.length > 48 ? "…" : ""}` : "Run command";
          if (tool === "symbols") return path ? `Symbols: ${path}` : "Symbols";
          if (tool === "view_diff") return path ? `Diff: ${path}` : "View diff";
          if (tool === "glob_file_search" || tool === "glob_search") return path ? `Glob: ${path}` : "Glob search";
          if (tool === "codebase_search" || tool === "codebase_tool") return query ? `Codebase: ${query.slice(0, 40)}${query.length > 40 ? "…" : ""}` : "Codebase search";
          if (tool === "view_repo_map") return "Repo map";
          return path ? `${tool}: ${path}` : tool;
        }
        if (type === "apply" || type === "reject") {
          const path = String(item.path || "");
          const reason = String(item.reason || "");
          if (type === "apply") return path ? `Applied: ${path}` : "Applied action";
          return reason ? `Rejected: ${path || "action"} (${reason})` : `Rejected: ${path || "action"}`;
        }
        return JSON.stringify(item).slice(0, 180);
      }

      function pushTurnAction(actionItem) {
        if (activeTranscriptIndex < 0 || !panelTranscript[activeTranscriptIndex]) return;
        const turn = panelTranscript[activeTranscriptIndex];
        if (!Array.isArray(turn.actions)) turn.actions = [];
        const text = normalizeActionText(actionItem);
        if (!text) return;
        if (turn.actions.includes(text)) return;
        turn.actions.push(text);
        if (turn.actions.length > PANEL_ACTIONS_LIMIT) {
          turn.actions.splice(0, turn.actions.length - PANEL_ACTIONS_LIMIT);
        }
      }

      function renderTranscriptRows(rows, autoScroll = false) {
        const dataRows = Array.isArray(rows) ? rows.filter(Boolean).slice(-PANEL_TRANSCRIPT_LIMIT) : [];
        if (!dataRows.length) {
          responseEl.textContent = "Ready. Select a file and ask the agent to edit it.";
          responseEl.classList.remove("has-content");
          return;
        }
        const html = [];
        for (const row of dataRows) {
          const rowId = ensureTurnId(row);
          const user = String((row && row.user) || "").trim();
          const assistant = normalizeAssistantHistoryText(String((row && row.assistant) || "").trim());
          if (user) {
            const isLong = user.length > PANEL_USER_TEXT_MAX_CHARS;
            const expanded = isLong && expandedUserTurnIds.has(rowId);
            const shownUser = expanded ? user : truncatePanelUserText(user);
            const toggleBtn = isLong
              ? `<button class="panel-expand-btn" data-turn-id="${esc(rowId)}">${expanded ? "Collapse" : "Expand"}</button>`
              : "";
            html.push(
              `<div class="panel-turn panel-turn-user"><div class="panel-turn-head"><span class="panel-role panel-role-user">You</span>${toggleBtn}</div><div class="panel-turn-body">${esc(shownUser)}</div></div>`
            );
          }
          if (assistant) {
            const actions = Array.isArray(row.actions) ? row.actions : [];
            const actionHtml = actions.length
              ? `<div class="panel-turn-actions" style="margin-top:8px;padding-top:6px;border-top:1px solid var(--border, #333);font-size:0.9em;opacity:0.85;"><div class="panel-turn-actions-title" style="font-weight:600;margin-bottom:4px;">Tools used</div><div class="panel-turn-actions-body">${actions.map((a) => `<div class="panel-turn-action-item">${esc(String(a))}</div>`).join("")}</div></div>`
              : "";
            html.push(
              `<div class="panel-turn panel-turn-assistant"><div class="panel-turn-head"><span class="panel-role panel-role-assistant">Moonlet</span></div><div class="panel-turn-body">${esc(assistant)}</div>${actionHtml}</div>`
            );
          }
        }
        responseEl.innerHTML = html.join("");
        responseEl.classList.add("has-content");
        if (autoScroll) {
          responseEl.scrollTop = responseEl.scrollHeight;
        }
      }

      function renderConversationHistory(entries) {
        renderTranscriptRows(entries, false);
      }

      function renderPanelTranscript() {
        renderTranscriptRows(panelTranscript, true);
      }

      function normalizeAssistantHistoryText(text) {
        const t = String(text || "").trim();
        if (t === "[Staged file_edit]") {
          return "Staged code changes are ready. Review and click Accept or Reject.";
        }
        if (t === "[Applied file_edit]") {
          return "Applied code changes.";
        }
        if (t === "[No-op file_edit: target already up to date]") {
          return "No code changes needed. Target is already up to date.";
        }
        if (t.startsWith("[No-op file_edit:")) {
          return "No effective code changes were produced.";
        }
        return t;
      }

      async function loadConversationHistory() {
        try {
          const data = await api(`/history?limit=${PANEL_TRANSCRIPT_LIMIT}`, {}, 5000);
          const rows = Array.isArray(data && data.history) ? data.history : [];
          panelTranscript.length = 0;
          expandedUserTurnIds.clear();
          for (const row of rows) {
            panelTranscript.push({
              id: `turn_${panelTurnSeq++}`,
              user: String((row && row.user) || ""),
              assistant: String((row && row.assistant) || ""),
              actions: [],
            });
          }
          activeTranscriptIndex = -1;
          if (panelTranscript.length) {
            renderPanelTranscript();
          } else {
            responseEl.textContent = "Ready. Select a file and ask the agent to edit it.";
            responseEl.classList.remove("has-content");
          }
        } catch {
          // Keep default placeholder if history load fails.
        }
      }

      async function startNewChat() {
        try {
          if (serverReady) {
            await api("/history/clear", { method: "POST", body: JSON.stringify({}) }, 5000);
          }
        } catch (_) {
          // Even if backend clear fails, reset visible panel state locally.
        }
        panelTranscript.length = 0;
        activeTranscriptIndex = -1;
        expandedUserTurnIds.clear();
        const echoEl = document.getElementById("userPromptEcho");
        if (echoEl) {
          echoEl.textContent = "";
          echoEl.classList.remove("visible");
        }
        responseEl.textContent = "Started a new chat.";
        responseEl.classList.add("has-content");
        setStatus("New chat started", "");
      }

      responseEl.addEventListener("click", (event) => {
        const btn = event.target && event.target.closest
          ? event.target.closest(".panel-expand-btn")
          : null;
        if (!btn) return;
        const turnId = String(btn.getAttribute("data-turn-id") || "");
        if (!turnId) return;
        if (expandedUserTurnIds.has(turnId)) {
          expandedUserTurnIds.delete(turnId);
        } else {
          expandedUserTurnIds.add(turnId);
        }
        renderPanelTranscript();
      });

      function simplifyResponseText(text) {
        if (!text) return "";
        let cleaned = text;
        cleaned = cleaned.replace(/```[\s\S]*?```/g, "").trim();
        cleaned = cleaned.replace(/\[\[\[file:[\s\S]*?\[\[\[end\]\]\]/g, "").trim();
        cleaned = cleaned.replace(/^\s*FILE:\s.*$/gm, "").trim();
        cleaned = cleaned.replace(/^\s*FILES:\s*$/gm, "").trim();
        cleaned = cleaned.replace(/\s+/g, " ").trim();
        if (cleaned.length > 300) cleaned = cleaned.slice(0, 300) + "...";
        return cleaned;
      }

      // ─── API ───
      async function api(path, opts = {}, timeoutMs = 15000) {
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), timeoutMs);
        try {
          const res = await fetch(API_BASE + path, {
            headers: { "Content-Type": "application/json" },
            signal: controller.signal,
            ...opts,
          });
          if (!res.ok) {
            const text = await res.text();
            throw new Error(`HTTP ${res.status}: ${text}`);
          }
          return res.json();
        } catch (err) {
          if (err && err.name === "AbortError") throw new Error("request timed out");
          throw err;
        } finally { clearTimeout(timer); }
      }

      async function probeHealthAt(base, timeoutMs = 1200) {
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), timeoutMs);
        try {
          const res = await fetch(`${base}/health`, { signal: controller.signal });
          if (!res.ok) return null;
          const data = await res.json();
          if (data && data.status === "ok") return data;
          return null;
        } catch {
          return null;
        } finally {
          clearTimeout(timer);
        }
      }

      async function discoverApiBase() {
        const seen = new Set();
        const candidates = [];
        const add = (base) => {
          const v = String(base || "").trim();
          if (!v || seen.has(v)) return;
          seen.add(v);
          candidates.push(v);
        };
        add(API_BASE);
        add("http://127.0.0.1:8002");
        add("http://127.0.0.1:8000");
        for (let p = 8001; p <= 8010; p++) {
          add(`http://127.0.0.1:${p}`);
        }
        for (const base of candidates) {
          const ok = await probeHealthAt(base, 900);
          if (ok) return base;
        }
        return null;
      }

      async function waitForServer(maxMs = 8000) {
        const start = Date.now();
        while (Date.now() - start < maxMs) {
          try {
            const data = await api("/health", {}, 2000);
            if (data && data.status === "ok") { serverReady = true; return true; }
          } catch {}
          const discovered = await discoverApiBase();
          if (discovered && discovered !== API_BASE) {
            API_BASE = discovered;
            try {
              const data = await api("/health", {}, 2000);
              if (data && data.status === "ok") { serverReady = true; return true; }
            } catch {}
          }
          await new Promise(r => setTimeout(r, 300));
        }
        return false;
      }

      async function ensureServerReadyNow() {
        if (serverReady) return true;
        try {
          const data = await api("/health", {}, 1500);
          if (data && data.status === "ok") {
            serverReady = true;
            return true;
          }
        } catch {}
        const discovered = await discoverApiBase();
        if (discovered && discovered !== API_BASE) {
          API_BASE = discovered;
        }
        try {
          const data = await api("/health", {}, 1500);
          if (data && data.status === "ok") {
            serverReady = true;
            return true;
          }
        } catch {}
        return false;
      }

      // ─── Files ───
      async function loadFiles() {
        const rootValue = rootEl.value.trim();
        if (!rootValue) { setStatus("Select a folder", ""); return false; }
        if (!serverReady) { setStatus("Server not ready", "error"); return false; }
        loadSeq += 1;
        const seq = loadSeq;
        activeLoadSeq = seq;
        isLoadingFiles = true;
        setStatus("Loading files...", "busy");
        try {
          let data = null;
          let attempts = 2;
          while (attempts > 0) {
            try {
              data = await api(`/files?root=${encodeURIComponent(rootValue)}`);
              break;
            } catch (err) {
              const errText = String(err);
              if (errText.includes("HTTP 409")) {
                await api("/root", { method: "POST", body: JSON.stringify({ path: rootValue }) });
                attempts -= 1; continue;
              }
              if (errText.includes("Failed to fetch")) {
                attempts -= 1;
                await new Promise(r => setTimeout(r, 300)); continue;
              }
              throw err;
            }
          }
          if (!data) throw new Error("failed to load files");
          if (seq !== activeLoadSeq) return false;
          const visibleFiles = data.files || [];
          rememberLastSelectedRoot(rootValue);
          explorerEntries = visibleFiles;
          renderExplorerList();
          const count = visibleFiles.length;
          setStatus(`${count} item${count !== 1 ? 's' : ''}`, "");
          updateTerminalButtonState();
          return true;
        } catch (err) {
          if (seq !== activeLoadSeq) return false;
          setStatus("Load failed", "error");
          updateTerminalButtonState();
          console.error(err);
          return false;
        } finally {
          if (seq === activeLoadSeq) isLoadingFiles = false;
        }
      }

      async function loadRoot() {
        try {
          const data = await api("/root");
          rootEl.value = data.root || "";
        } catch {}
      }

      function defaultCreateBaseDir() {
        if (!currentFile) return "";
        const idx = currentFile.lastIndexOf("/");
        return idx >= 0 ? currentFile.slice(0, idx) : "";
      }

      function nextAvailablePath(baseDir, stem, ext, existingSet) {
        let n = 1;
        while (n < 1000) {
          const suffix = n === 1 ? "" : `-${n}`;
          const name = `${stem}${suffix}${ext}`;
          const rel = baseDir ? `${baseDir}/${name}` : name;
          if (!existingSet.has(rel)) return rel;
          n += 1;
        }
        return baseDir ? `${baseDir}/${stem}-${Date.now()}${ext}` : `${stem}-${Date.now()}${ext}`;
      }

      function getSelectedExplorerPath() {
        const active = filesEl.querySelector("li.active");
        if (active && active.dataset && active.dataset.path) {
          return active.dataset.path;
        }
        return currentFile || "";
      }

      async function createExplorerEntry(isDirectory, initialTarget = "") {
        if (!hasUserImported) {
          setStatus("Import files first", "error");
          return;
        }
        const rootAbs = rootEl.value.trim();
        if (!rootAbs) {
          setStatus("Import files first", "error");
          return;
        }
        // If a context-menu target is provided:
        // - folder target => create inside folder
        // - file target => create alongside file
        // If target is empty string (right-click on blank explorer area), create at root.
        // If target is null/undefined (legacy callsite), fallback to current-file directory.
        const hasExplicitTarget = initialTarget !== null && initialTarget !== undefined;
        const baseDir = hasExplicitTarget
          ? (
              initialTarget
                ? (
                    initialTarget.endsWith("/")
                      ? initialTarget.replace(/\/+$/, "")
                      : (initialTarget.includes("/") ? initialTarget.slice(0, initialTarget.lastIndexOf("/")) : "")
                  )
                : ""
            )
          : defaultCreateBaseDir();
        const existing = new Set(
          Array.from(filesEl.querySelectorAll("li[data-path]"))
            .map((n) => (n.dataset.path || "").trim())
            .filter(Boolean)
        );
        let cleanRel = isDirectory
          ? nextAvailablePath(baseDir, "new-folder", "", existing)
          : nextAvailablePath(baseDir, "new-file", ".txt", existing);
        if (!window.electronAPI || !window.electronAPI.fsCreateEntry) {
          setStatus("Create is unavailable", "error");
          return;
        }
        let res = null;
        let createdRel = "";
        try {
          for (let i = 0; i < 30; i++) {
            res = await window.electronAPI.fsCreateEntry({
              root: rootAbs,
              relPath: cleanRel,
              isDirectory,
              content: "",
            });
            if (res && res.ok) break;
            const msg = String((res && res.error) || "");
            if (!msg.toLowerCase().includes("already exists")) break;
            cleanRel = isDirectory
              ? nextAvailablePath(baseDir, "new-folder", "", new Set([...existing, cleanRel]))
              : nextAvailablePath(baseDir, "new-file", ".txt", new Set([...existing, cleanRel]));
          }
          if (!res || !res.ok) {
            setStatus(`Create failed: ${(res && res.error) || "unknown error"}`, "error");
            return;
          }
          if (isDirectory) {
            // Explorer lists files; create a visible starter file inside the folder.
            const starter = `${cleanRel}/new-file.txt`;
            const starterRes = await window.electronAPI.fsCreateEntry({
              root: rootAbs,
              relPath: starter,
              isDirectory: false,
              content: "",
            });
            createdRel = starterRes && starterRes.ok ? starter : cleanRel;
          } else {
            createdRel = cleanRel;
          }
          if (includePaths && includePaths.length && createdRel && !includePaths.includes(createdRel)) {
            includePaths.push(createdRel);
            try {
              await api("/include", { method: "POST", body: JSON.stringify({ paths: includePaths }) });
            } catch (_) {}
          }
          setStatus(`${isDirectory ? "Folder" : "File"} created: ${cleanRel}`, "");
        } finally {
          await loadFiles();
        }
        if (createdRel) {
          const li = Array.from(filesEl.children).find((n) => n.dataset.path === createdRel);
          if (li) await selectFile(createdRel, li);
        }
      }

      async function renameExplorerEntry(targetPath = "") {
        if (!hasUserImported) {
          setStatus("Import files first", "error");
          return;
        }
        const rootAbs = rootEl.value.trim();
        const fromRel = targetPath || getSelectedExplorerPath();
        if (!rootAbs || !fromRel) return;
        const suggested = fromRel;
        const toRelRaw = window.prompt("Rename path to", suggested);
        if (!toRelRaw) return;
        const toRel = toRelRaw.replace(/^[/\\]+/, "").trim();
        if (!toRel || toRel === fromRel) return;
        if (!window.electronAPI || !window.electronAPI.fsRenameEntry) {
          setStatus("Rename is unavailable", "error");
          return;
        }
        let ok = false;
        try {
          const res = await window.electronAPI.fsRenameEntry({
            root: rootAbs,
            fromRelPath: fromRel,
            toRelPath: toRel,
          });
          if (!res || !res.ok) {
            setStatus(`Rename failed: ${(res && res.error) || "unknown error"}`, "error");
            return;
          }
          ok = true;
          if (currentFile === fromRel) {
            await closeLspForPath(fromRel);
          }
          if (includePaths && includePaths.length) {
            const idx = includePaths.indexOf(fromRel);
            if (idx >= 0) includePaths[idx] = toRel;
            try {
              await api("/include", { method: "POST", body: JSON.stringify({ paths: includePaths }) });
            } catch (_) {}
          }
          setStatus(`Renamed to: ${toRel}`, "");
        } finally {
          await loadFiles();
        }
        if (ok) {
          const li = Array.from(filesEl.children).find((n) => n.dataset.path === toRel);
          if (li) await selectFile(toRel, li);
        }
      }

      async function deleteExplorerEntry(targetPath = "") {
        if (!hasUserImported) {
          setStatus("Import files first", "error");
          return;
        }
        const relPath = targetPath || getSelectedExplorerPath();
        const rootAbs = rootEl.value.trim();
        if (!rootAbs || !relPath) return;
        if (!window.confirm(`Delete "${relPath}"?`)) return;
        if (!window.electronAPI || !window.electronAPI.fsDeleteEntry) {
          setStatus("Delete is unavailable", "error");
          return;
        }
        try {
          const res = await window.electronAPI.fsDeleteEntry({
            root: rootAbs,
            relPath,
          });
          if (!res || !res.ok) {
            setStatus(`Delete failed: ${(res && res.error) || "unknown error"}`, "error");
            return;
          }
          if (currentFile === relPath) {
            await closeLspForPath(relPath);
            currentFile = null;
            tabNameEl.textContent = "Welcome";
            titleFilepath.textContent = "";
            statusBarFileEl.textContent = "No file";
            setViewerContent("");
            showWelcome();
            setLspStatus("off", "LSP disabled (no file)");
          }
          forgetLastOpenedFileForCurrentRoot(relPath);
          if (includePaths && includePaths.length) {
            includePaths = includePaths.filter((p) => p !== relPath);
            try {
              await api("/include", { method: "POST", body: JSON.stringify({ paths: includePaths }) });
            } catch (_) {}
          }
          setStatus("Deleted", "");
        } finally {
          await loadFiles();
        }
        updateRunButtonState();
        updateTerminalButtonState();
      }

      function hideExplorerMenu() {
        explorerMenuTarget = "";
        explorerMenuEl.classList.add("hidden");
      }

      function showExplorerMenu(x, y, targetPath = "") {
        explorerMenuTarget = targetPath || "";
        const canMutate = hasUserImported;
        menuNewFileEl.disabled = !canMutate;
        menuNewFolderEl.disabled = !canMutate;
        menuRenameFileEl.disabled = !explorerMenuTarget;
        menuRenameFileEl.textContent = explorerMenuTarget
          ? `Rename "${explorerMenuTarget}"`
          : "Rename";
        menuDeleteFileEl.disabled = !explorerMenuTarget;
        menuDeleteFileEl.textContent = explorerMenuTarget
          ? `Delete "${explorerMenuTarget}"`
          : "Delete";
        explorerMenuEl.classList.remove("hidden");
        const maxX = window.innerWidth - explorerMenuEl.offsetWidth - 8;
        const maxY = window.innerHeight - explorerMenuEl.offsetHeight - 8;
        explorerMenuEl.style.left = `${Math.max(8, Math.min(x, maxX))}px`;
        explorerMenuEl.style.top = `${Math.max(8, Math.min(y, maxY))}px`;
      }

      function showImagePreview(dataUrl) {
        currentFileIsBinary = true;
        if (imagePreviewEl) imagePreviewEl.src = dataUrl || "";
        if (imagePreviewWrap) imagePreviewWrap.style.display = "flex";
        if (cmHostEl) cmHostEl.style.display = "none";
      }

      function hideImagePreview() {
        currentFileIsBinary = false;
        if (imagePreviewEl) imagePreviewEl.src = "";
        if (imagePreviewWrap) imagePreviewWrap.style.display = "none";
        if (cmHostEl) cmHostEl.style.display = "";
      }

      function setViewerContent(text) {
        if (currentFileIsBinary) hideImagePreview();
        setEditorText(text);
        configureEditorForDocument(currentFile, String(text ?? ""));
        updateCursorPosition();
      }

      let activeDiffHighlights = null; // {addedLines: Set, deletedAtLines: Set}
      const preChangeContentByFile = {}; // file path -> content before agent changes
      window._perFileDiffs = window._perFileDiffs || {};

      function showApplySummary(filePath) {
        const path = filePath || currentFile || "";
        const perFile = (window._perFileDiffs && path) ? window._perFileDiffs[path] : "";
        const pending = window._pendingDiff || "";
        const diff = (perFile && perFile.trim()) ? perFile : ((pending && pending.trim()) ? pending : "");
        const summary = diff ? summarizeDiff(diff, path) : (path ? `${esc(path)} &middot; changes applied` : "Changes applied");
        responseEl.innerHTML = `<div style="padding:8px;border:1px solid var(--border);border-radius:8px;background:var(--panel);"><strong>Applied</strong><div style="margin-top:6px;">${summary}</div></div>`;
        responseEl.classList.add("has-content");
      }

      function buildUnifiedDiffFromContents(path, beforeText, afterText) {
        const before = String(beforeText ?? "");
        const after = String(afterText ?? "");
        if (before === after) return "";
        const beforeLines = before.split("\n");
        const afterLines = after.split("\n");
        let prefix = 0;
        while (
          prefix < beforeLines.length &&
          prefix < afterLines.length &&
          beforeLines[prefix] === afterLines[prefix]
        ) {
          prefix++;
        }
        let suffix = 0;
        while (
          suffix < (beforeLines.length - prefix) &&
          suffix < (afterLines.length - prefix) &&
          beforeLines[beforeLines.length - 1 - suffix] === afterLines[afterLines.length - 1 - suffix]
        ) {
          suffix++;
        }
        const oldStart = prefix + 1;
        const newStart = prefix + 1;
        const oldCount = Math.max(0, beforeLines.length - prefix - suffix);
        const newCount = Math.max(0, afterLines.length - prefix - suffix);
        const lines = [
          `--- a/${path}`,
          `+++ b/${path}`,
          `@@ -${oldStart},${oldCount} +${newStart},${newCount} @@`,
        ];
        for (let i = prefix; i < beforeLines.length - suffix; i++) {
          lines.push(`-${beforeLines[i]}`);
        }
        for (let i = prefix; i < afterLines.length - suffix; i++) {
          lines.push(`+${afterLines[i]}`);
        }
        return lines.join("\n");
      }

      async function revertChanges() {
        // Restore all edited files to pre-change state (Ctrl+Z alternative)
        const stagedPaths = window._perFileStaged ? Object.keys(window._perFileStaged) : [];
        let restored = 0;
        for (const p of stagedPaths) {
          const before = preChangeContentByFile[p];
          if (before !== undefined) {
            try {
              await api("/file", { method: "POST", body: JSON.stringify({ path: p, content: before }) });
              restored++;
            } catch { /* continue */ }
          }
        }
        if (currentFile && preChangeContentByFile[currentFile] !== undefined) {
          setViewerContent(preChangeContentByFile[currentFile]);
          scheduleLspDidChange();
        }
        if (restored > 0) {
          markSaved();
          setStatus(`Reverted — ${restored} file${restored !== 1 ? "s" : ""} restored`, "");
        }
        for (const p of stagedPaths) {
          delete window._perFileDiffs?.[p];
          delete preChangeContentByFile[p];
        }
        if (window._perFileStaged) window._perFileStaged = {};
        window._pendingDiff = null;
        document.getElementById("btnRepair").style.display = "none";
        clearDiffHighlights();
      }

      function applyDiffHighlights(diffText) {
        // Parse unified diff to map added and deleted locations in the NEW file view.
        if (!diffText || !diffText.trim()) { clearDiffHighlights(); return; }
        const added = new Set();
        const deletedAt = new Set();
        let newLine = 0;
        for (const line of diffText.split("\n")) {
          if (line.startsWith("@@")) {
            // Parse new file start line: @@ -a,b +c,d @@
            const m = line.match(/\+(\d+)/);
            if (m) newLine = parseInt(m[1]) - 1; // 0-based
            continue;
          }
          if (line.startsWith("+") && !line.startsWith("+++")) {
            newLine++;
            added.add(newLine); // 1-based
          } else if (line.startsWith("-") && !line.startsWith("---")) {
            // Deleted lines do not advance newLine; mark nearest surviving line.
            deletedAt.add(Math.max(1, newLine + 1));
          } else {
            newLine++;
          }
        }
        activeDiffHighlights = { addedLines: added, deletedAtLines: deletedAt };
        // Defer so editor has rendered new content before applying decorations
        requestAnimationFrame(() => {
          setDiffDecorations(added, deletedAt);
        });
        // Show Accept/Reject buttons
        document.getElementById("diffActions").style.display = "flex";
      }

      function clearDiffHighlights() {
        activeDiffHighlights = null;
        clearDiffDecorations();
        document.getElementById("diffActions").style.display = "none";
      }

      async function selectFile(path, li) {
        const previousFile = currentFile;
        if (!li) {
          li = Array.from(filesEl.children).find((n) => n.dataset && n.dataset.path === path);
          if (!li) return;
        }
        Array.from(filesEl.children).forEach(n => n.classList.remove("active"));
        li.classList.add("active");
        if (previousFile && previousFile !== path) {
          await closeLspForPath(previousFile);
        }
        if ((path || "").endsWith("/")) {
          // IDE-like behavior: clicking a folder toggles expand/collapse.
          if (expandedFolders.has(path)) expandedFolders.delete(path);
          else expandedFolders.add(path);
          renderExplorerList();
          currentFile = null;
          currentFileIsBinary = false;
          hideImagePreview();
          tabNameEl.textContent = "Welcome";
          titleFilepath.textContent = "";
          statusBarFileEl.textContent = "No file";
          statusBarLangEl.textContent = "Folder";
          setViewerContent("");
          showWelcome();
          setLspStatus("off", "LSP disabled (folder view)");
          setStatus("Folder toggled", "");
          updateRunButtonState();
          updateTerminalButtonState();
          return;
        }
        currentFile = path;
        rememberLastOpenedFileForCurrentRoot(path);
        // Keep terminal cwd synced to the currently opened file directory.
        if (terminalStarted) {
          const desiredCwdAbs = terminalCwdAbsForCurrentFile();
          const desiredNorm = normalizeAbsPath(desiredCwdAbs);
          const activeNorm = normalizeAbsPath(terminalActiveCwdAbs);
          if (desiredNorm && desiredNorm !== activeNorm) {
            sendTerminalCommand(`cd "${desiredCwdAbs}"`);
            terminalActiveCwdAbs = desiredNorm;
          }
        }
        tabNameEl.textContent = path;
        statusBarFileEl.textContent = path;
        statusBarLangEl.textContent = getLanguageName(path);
        titleFilepath.textContent = path;
        updateRunButtonState();
        updateTerminalButtonState();
        markSaved();
        hideWelcome();
        setStatus("Loading...", "busy");
        try {
          const data = await api(`/file?path=${encodeURIComponent(path)}`);
          if (data && data.binary) {
            const mime = data.mime || "application/octet-stream";
            const dataUrl = `data:${mime};base64,${data.content || ""}`;
            showImagePreview(dataUrl);
            fileCache[path] = "";
            statusBarLangEl.textContent = "Image";
            await closeLspForPath(path);
            setLspStatus("off", "LSP disabled (binary file)");
          } else {
            const diskContent = data.content;
            const pathBase = path.split("/").pop() || path;
            const stagedKey = window._perFileStaged && Object.keys(window._perFileStaged).find(
              (k) => k === path || k.split("/").pop() === pathBase
            );
            const stagedContent = stagedKey ? window._perFileStaged[stagedKey] : undefined;
            const contentToShow = stagedContent !== undefined ? (stagedContent ?? "") : diskContent;
            if (stagedContent !== undefined) {
              preChangeContentByFile[path] = diskContent;
            }
            fileCache[path] = contentToShow;
            setViewerContent(contentToShow);
            currentFileIsBinary = false;
            hideImagePreview();
            await openLspForPath(path, contentToShow);
          }
          updateCursorPosition();
          setStatus("Ready", "");
          // Show diff highlights if this file was changed in a multi-file edit
          const pathBase = path.split("/").pop() || path;
          const diffKey = window._perFileDiffs && Object.keys(window._perFileDiffs).find(
            (k) => k === path || k.split("/").pop() === pathBase
          );
          const diffToShow = diffKey ? window._perFileDiffs[diffKey] : null;
          if (diffToShow && diffToShow.trim()) {
            applyDiffHighlights(diffToShow);
            document.getElementById("diffActions").style.display = "flex";
          } else if (preChangeContentByFile[path] !== undefined && window._perFileStaged) {
            const stagedKey = Object.keys(window._perFileStaged).find(
              (k) => k === path || k.split("/").pop() === pathBase
            );
            if (stagedKey) {
              const beforeContent = preChangeContentByFile[path];
              const afterContent = window._perFileStaged[stagedKey];
              const computed = buildUnifiedDiffFromContents(path, beforeContent, afterContent);
              if (computed) {
                applyDiffHighlights(computed);
                document.getElementById("diffActions").style.display = "flex";
              } else {
                clearDiffHighlights();
              }
            } else {
              clearDiffHighlights();
            }
          } else {
            clearDiffHighlights();
          }
        } catch (err) {
          setStatus("Read error", "error");
          console.error(err);
        }
      }

      // ─── Repair (triggered by Repair button after verify failure) ───
      async function requestRepair() {
        const prompt = window._lastAgentPrompt;
        const lastError = window._lastVerifyError;
        const lastMeta = window._lastAgentMeta;
        if (!prompt || !lastError || !lastMeta) {
          setStatus("No repair context", "error");
          return;
        }
        if (activeRequest) return;
        const iteration = (lastMeta.iteration || 0) + 1;
        window._repairPayload = {
          text: prompt,
          mode: "repair",
          last_error: lastError,
          previous_patches: lastMeta.previous_patches || [],
          iteration,
        };
        promptEl.value = "[Repairing...]";
        await sendPrompt();
      }

      // ─── Send Prompt ───
      async function sendPrompt() {
        let rawText = promptEl.value.trim();
        const useRepairPayload = window._repairPayload;
        if (useRepairPayload) {
          rawText = useRepairPayload.text || "";
          delete window._repairPayload;
        }
        if (!rawText || activeRequest) return;
        if (!(await ensureServerReadyNow())) {
          setStatus("Server not ready", "error");
          responseEl.textContent = "Backend not reachable yet. Try again in a moment.";
          responseEl.classList.add("has-content");
          return;
        }
        const text = rawText;
        // Discard pending staged edits when user sends a new command (don't auto-apply).
        // Skip for repair — we already accepted and wrote; repair returns new patches.
        if (!useRepairPayload && (activeDiffHighlights || (window._pendingDiff && window._pendingDiff.trim()) ||
            (window._perFileStaged && Object.keys(window._perFileStaged).length > 0))) {
          await rejectChanges();
        }
        const preRequestFile = currentFile || "";
        const preRequestContent = preRequestFile && !currentFileIsBinary ? getEditorText() : null;
        promptEl.value = "";
        promptEl.style.height = "auto";
        // Show the sent prompt
        const echoEl = document.getElementById("userPromptEcho");
        echoEl.textContent = text;
        echoEl.classList.add("visible");
        setStatus("Thinking...", "busy");
        modelRunning = true; // pause highlighting to free CPU for model
        startTimer();
        const turn = { id: `turn_${panelTurnSeq++}`, user: text, assistant: "...", actions: [] };
        panelTranscript.push(turn);
        if (panelTranscript.length > PANEL_TRANSCRIPT_LIMIT) {
          panelTranscript.splice(0, panelTranscript.length - PANEL_TRANSCRIPT_LIMIT);
        }
        activeTranscriptIndex = panelTranscript.length - 1;
        renderPanelTranscript();
        let slowTimer = setTimeout(() => setStatus("Model running...", "busy"), 10000);
        // Local shell execution mode: "/run <command>"
        if (text.startsWith("/run ")) {
          const cmd = text.slice(5).trim();
          const cwd = currentFile && currentFile.includes("/")
            ? currentFile.split("/").slice(0, -1).join("/")
            : "";
          try {
            setStatus("Running command...", "busy");
            const data = await api("/run", {
              method: "POST",
              body: JSON.stringify({ cmd, cwd, timeout: 30 }),
            });
            const parts = [];
            parts.push(`$ ${data.cmd}`);
            parts.push(`cwd: ${data.cwd}`);
            parts.push(`exit: ${data.code}`);
            if (data.stdout) {
              parts.push("");
              parts.push("[stdout]");
              parts.push(data.stdout);
            }
            if (data.stderr) {
              parts.push("");
              parts.push("[stderr]");
              parts.push(data.stderr);
            }
            responseEl.textContent = parts.join("\n");
            responseEl.classList.add("has-content");
            if (activeTranscriptIndex >= 0 && panelTranscript[activeTranscriptIndex]) {
              panelTranscript[activeTranscriptIndex].assistant = parts.join("\n");
              renderPanelTranscript();
            }
            const ok = Number(data.code) === 0;
            stopTimer(ok ? "done" : "error");
            setStatus(ok ? "Command finished" : `Command failed (${data.code})`, ok ? "" : "error");
            // If command changed files (e.g. format/build artifacts), refresh list/content.
            if (currentFile) {
              try {
                const fileData = await api(`/file?path=${encodeURIComponent(currentFile)}`);
                setViewerContent(fileData.content);
                await openLspForPath(currentFile, fileData.content);
              } catch {}
            }
          } catch (err) {
            const msg = String(err && err.message ? err.message : err);
            stopTimer("error");
            setStatus("Run failed", "error");
            responseEl.textContent = `Error: ${msg}`;
            responseEl.classList.add("has-content");
            if (activeTranscriptIndex >= 0 && panelTranscript[activeTranscriptIndex]) {
              panelTranscript[activeTranscriptIndex].assistant = `Error: ${msg}`;
              renderPanelTranscript();
            }
          } finally {
            clearTimeout(slowTimer);
            activeRequest = null;
            modelRunning = false;
          }
          return;
        }
        const selectedMode = useRepairPayload ? "repair" : (modeEl.value || "agent");
        // Throttled response rendering state — declared before try so finally can access
        let firstChunk = true;
        let rawResponseBuf = "";
        let responseThrottleTimer = null;
        let metaSummaryLocked = false;
        let pendingNoopRun = false;
        let pendingNoopBody = "No code changes were applied. Target is already up to date.";
        let pendingMetaFocusFile = "";
        function flushResponse() {
          if (metaSummaryLocked) return;
          responseThrottleTimer = null;
          const rendered = rawResponseBuf;
          if (activeTranscriptIndex >= 0 && panelTranscript[activeTranscriptIndex]) {
            panelTranscript[activeTranscriptIndex].assistant = rendered || "...";
            renderPanelTranscript();
            return;
          }
          responseEl.textContent = rawResponseBuf;
          responseEl.classList.add("has-content");
        }
        try {
          const payload = useRepairPayload
            ? { ...useRepairPayload, focus_file: currentFile, file_path: currentFile }
            : { text, mode: selectedMode, focus_file: currentFile, file_path: currentFile };
          const controller = new AbortController();
          activeRequest = controller;
          const res = await fetch(API_BASE + "/stream", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
            signal: controller.signal,
          });
          if (!res.ok) { const t = await res.text(); throw new Error(`HTTP ${res.status}: ${t}`); }
          const reader = res.body.getReader();
          const decoder = new TextDecoder();
          let buffer = "";
          let doneReceived = false;
          let stagedContentApplied = false;
          const handleEvent = (block) => {
            const lines = block.split("\n").filter(Boolean);
            let event = "chunk";
            const dataLines = [];
            for (const line of lines) {
              if (line.startsWith("event:")) event = line.slice(6).trim();
              else if (line.startsWith("data:")) dataLines.push(line.slice(5).trimStart());
            }
            const data = dataLines.join("\n");
            if (event === "chunk") {
              if (!data.trim()) return;
              if (metaSummaryLocked) return;
              if (data.includes("[Retrying with strict file block format]")) return;
              if (firstChunk) { responseEl.textContent = ""; firstChunk = false; }
              rawResponseBuf += data;
              // Throttle: render at most once per 200ms
              if (!responseThrottleTimer) {
                responseThrottleTimer = setTimeout(flushResponse, 200);
              }
            } else if (event === "done") {
              doneReceived = true;
              // In chat mode, never show "Staged for" — that's agent-only; treat as server bug
              const isChat = (selectedMode || "").toLowerCase() === "chat";
              if (isChat && /^Staged for:/.test(String(rawResponseBuf || "").trim())) {
                rawResponseBuf = "Chat mode: no file changes are made. Use agent mode to edit files.";
              }
              // Final flush of response text
              if (responseThrottleTimer) { clearTimeout(responseThrottleTimer); responseThrottleTimer = null; }
              if (rawResponseBuf && !metaSummaryLocked) flushResponse();
            } else if (event === "action") {
              try {
                const actionData = JSON.parse(data);
                pushTurnAction(actionData);
                renderPanelTranscript();
              } catch (_) {}
            } else if (event === "meta") {
              try {
                const d = JSON.parse(data);
                if (Array.isArray(d.agent_actions)) {
                  for (const act of d.agent_actions) {
                    pushTurnAction(act);
                  }
                }
                const coreActions = d.core_v2_actions || {};
                const applied = Array.isArray(coreActions.applied) ? coreActions.applied : [];
                const rejected = Array.isArray(coreActions.rejected) ? coreActions.rejected : [];
                for (const item of applied) {
                  pushTurnAction({ type: "apply", path: item && item.path ? item.path : "", reason: "" });
                }
                for (const item of rejected) {
                  pushTurnAction({
                    type: "reject",
                    path: item && item.path ? item.path : "",
                    reason: item && item.reason ? item.reason : "",
                  });
                }
                stopTimer("done");
                // In chat mode, never process staged/diff meta — chat does not stage files
                const isChatMode = (selectedMode || "").toLowerCase() === "chat";
                if (isChatMode && (d.staged || (d.per_file_staged && Object.keys(d.per_file_staged).length > 0))) {
                  // Ignore staged content; show only explanation/text
                  d.staged = false;
                  d.per_file_staged = {};
                  d.per_file_before = {};
                  d.files_changed = [];
                }
                const dur = d.duration_ms ? `${(d.duration_ms/1000).toFixed(1)}s` : '';
                setStatus(`Done ${dur}`, "");
                const failureKind = String(d.failure_kind || "none");
                const failureReason = String(d.failure_reason || "");
                const noopRun = Boolean(d.noop) || String(d.output || "").startsWith("[No-op file_edit:");
                pendingMetaFocusFile = String(d.focus_file || currentFile || "");
                let renderedMetaSummary = false;
                // Multi-file or single-file result (files_changed or fallback: per_file_staged keys)
                const fc = (d.files_changed && d.files_changed.length > 0)
                  ? d.files_changed
                  : (d.per_file_staged && Object.keys(d.per_file_staged).length > 0)
                    ? Object.keys(d.per_file_staged)
                    : [];
                if (fc.length > 0) {
                  // Multi-file result
                  const explanation = (d.explanation || d.output || rawResponseBuf || "").trim();
                  let html = "";
                  if (explanation) {
                    html += `<div style="margin-bottom:10px;white-space:pre-wrap;">${esc(explanation)}</div>`;
                  }
                  html += `<strong>Changed ${fc.length} file${fc.length !== 1 ? 's' : ''}</strong><br>`;
                  for (const f of fc) {
                    const fdiff = (d.per_file_diffs || {})[f] || "";
                    const fsum = fdiff ? summarizeDiff(fdiff, f) : esc(f);
                    html += `<div style="margin:4px 0;padding:2px 0;border-bottom:1px solid var(--border);">${fsum}</div>`;
                  }
                  responseEl.innerHTML = html;
                  responseEl.classList.add("has-content");
                  rawResponseBuf = "";
                  renderedMetaSummary = true;
                  // Store all per-file diffs for switching between files
                  window._perFileDiffs = { ...(window._perFileDiffs || {}), ...(d.per_file_diffs || {}) };
                  // Store per-file staged content for multi-file Accept
                  if (d.per_file_staged && Object.keys(d.per_file_staged).length > 0) {
                    window._perFileStaged = { ...(window._perFileStaged || {}), ...d.per_file_staged };
                  }
                  // Store before content for diff highlights (before = disk, after = staged)
                  if (d.per_file_before && Object.keys(d.per_file_before).length > 0) {
                    for (const [p, before] of Object.entries(d.per_file_before)) {
                      preChangeContentByFile[p] = before;
                    }
                  }
                  // Set _pendingDiff so Accept/Reject buttons show (match current file or use first changed)
                  const stagedPath = (d.staged_file || fc[0] || "");
                  const base = currentFile ? (currentFile.split("/").pop() || currentFile) : "";
                  let diffKey = currentFile && d.per_file_diffs ? Object.keys(d.per_file_diffs).find(
                    (k) => k === currentFile || k.split("/").pop() === base
                  ) : null;
                  const pd = d.per_file_diffs || {};
                  if (diffKey && pd[diffKey]) {
                    window._pendingDiff = pd[diffKey];
                  } else if (d.diff && d.diff.trim()) {
                    window._pendingDiff = d.diff;
                  } else if (stagedPath && pd[stagedPath]) {
                    window._pendingDiff = pd[stagedPath];
                  } else if (Object.keys(pd).length > 0) {
                    // Fallback: path mismatch (e.g. foo.c vs subdir/foo.c) — use first diff
                    const firstKey = Object.keys(pd).find((k) => k.split("/").pop() === (stagedPath && stagedPath.split("/").pop()) || !stagedPath);
                    window._pendingDiff = pd[firstKey || Object.keys(pd)[0]];
                  }
                  if (window._pendingDiff) {
                    document.getElementById("diffActions").style.display = "flex";
                  }
                  // If changed file not open, switch to it so user sees diff + green lines
                  if (stagedPath && (!currentFile || (currentFile.split("/").pop() !== stagedPath.split("/").pop()))) {
                    const li = Array.from(filesEl.querySelectorAll("[data-path]")).find((n) => {
                      const p = n.dataset.path || "";
                      return p === stagedPath || p.endsWith("/" + stagedPath) || stagedPath.endsWith("/" + p);
                    });
                    if (li) selectFile(stagedPath, li);
                    // selectFile applies diff highlights when loading (uses _perFileDiffs)
                  }
                  // Apply staged content + diff highlights for current file (exact or basename match)
                  if (currentFile && (d.per_file_diffs || d.per_file_staged)) {
                    const base = currentFile.split("/").pop() || currentFile;
                    const stagedKey = d.per_file_staged && Object.keys(d.per_file_staged).find(
                      (k) => k === currentFile || k.split("/").pop() === base
                    );
                    const diffKey = d.per_file_diffs && Object.keys(d.per_file_diffs).find(
                      (k) => k === currentFile || k.split("/").pop() === base
                    );
                    const stagedContent = stagedKey ? d.per_file_staged[stagedKey] : undefined;
                    const beforeContent = (d.per_file_before && stagedKey && d.per_file_before[stagedKey]) || getEditorText();
                    let diffToApply = diffKey && d.per_file_diffs[diffKey] ? d.per_file_diffs[diffKey] : "";
                    if (stagedContent !== undefined) {
                      preChangeContentByFile[currentFile] = beforeContent;
                      setViewerContent(String(stagedContent ?? ""));
                      scheduleLspDidChange();
                      if (!diffToApply) {
                        diffToApply = buildUnifiedDiffFromContents(currentFile, beforeContent, stagedContent);
                      }
                    }
                    if (diffToApply) {
                      window._pendingDiff = diffToApply;
                      if (!diffKey && d.per_file_diffs) {
                        const keyForStore = stagedKey || currentFile;
                        window._perFileDiffs = { ...(window._perFileDiffs || {}), [keyForStore]: diffToApply };
                      }
                      try { applyDiffHighlights(diffToApply); } catch (_) {}
                      document.getElementById("diffActions").style.display = "flex";
                      stagedContentApplied = true;
                    }
                  }
                } else if (d.diff && d.diff.trim()) {
                  // Single-file result
                  window._pendingDiff = d.diff;
                  const diffPath = d.focus_file || currentFile;
                  if (diffPath) {
                    window._perFileDiffs = window._perFileDiffs || {};
                    window._perFileDiffs[diffPath] = d.diff;
                  }
                  // Apply immediately for current file so red/green markers are visible
                  // even on failure paths where no staged content is returned.
                  const pathMatches = currentFile && diffPath && (
                    currentFile === diffPath ||
                    (currentFile.split("/").pop() || currentFile) === (diffPath.split("/").pop() || diffPath)
                  );
                  if (pathMatches) {
                    try {
                      applyDiffHighlights(d.diff);
                    } catch (_) {}
                    stagedContentApplied = true;
                  }
                  document.getElementById("diffActions").style.display = "flex";
                  const explanation = (d.explanation || d.output || rawResponseBuf || "").trim();
                  const summary = summarizeDiff(d.diff, d.focus_file || "");
                  responseEl.innerHTML = explanation
                    ? `<div style="margin-bottom:10px;white-space:pre-wrap;">${esc(explanation)}</div>${summary}`
                    : summary;
                  responseEl.classList.add("has-content");
                  rawResponseBuf = "";
                  renderedMetaSummary = true;
                }
                if (noopRun && !(d.diff && d.diff.trim())) {
                  // Delay noop summary until after final before/after comparison.
                  pendingNoopRun = true;
                  pendingNoopBody = "No code changes were applied. Target is already up to date.";
                } else if (failureKind && failureKind !== "none" && failureKind !== "noop") {
                  const reason = esc(failureReason || "Agent reported a failure.");
                  const tag = esc(failureKind.toUpperCase());
                  const attemptInfo = (d.attempt_index && d.attempt_total)
                    ? ` &middot; attempt ${Number(d.attempt_index)}/${Number(d.attempt_total)}`
                    : "";
                  let body = `<div><strong style="color:var(--red);">${tag}</strong>${attemptInfo}</div><div style="margin-top:4px;">${reason}</div>`;
                  if (d.diff && d.diff.trim()) {
                    body += `<div style="margin-top:6px;">${summarizeDiff(d.diff, d.focus_file || "")}</div>`;
                  }
                  responseEl.innerHTML = `<div style="padding:8px;border:1px solid var(--border);border-radius:8px;background:var(--panel);">${body}</div>`;
                  responseEl.classList.add("has-content");
                  rawResponseBuf = "";
                  renderedMetaSummary = true;
                }
                if (renderedMetaSummary) {
                  metaSummaryLocked = true;
                  if (activeTranscriptIndex >= 0 && panelTranscript[activeTranscriptIndex]) {
                    const metaText = responseEl.textContent ? responseEl.textContent.trim() : "";
                    if (metaText) {
                      panelTranscript[activeTranscriptIndex].assistant = metaText;
                      renderPanelTranscript();
                    }
                  }
                } else if (activeTranscriptIndex >= 0 && panelTranscript[activeTranscriptIndex]) {
                  const explanation = (d.explanation || d.output || "").trim();
                  if (explanation) {
                    panelTranscript[activeTranscriptIndex].assistant = explanation;
                    renderPanelTranscript();
                  }
                }
                // Staged edit mode: show candidate content in editor, do not auto-write.
                const explanation = (d.explanation || d.output || rawResponseBuf || "").trim();
                let stagedContent = (d.per_file_staged && currentFile && d.per_file_staged[currentFile])
                  ? d.per_file_staged[currentFile]
                  : (d.staged_content !== undefined ? d.staged_content : undefined);
                // Fallback: match by basename when exact path fails (e.g. "w6/parentcreates.c" vs "parentcreates.c")
                if (stagedContent === undefined && d.per_file_staged && currentFile) {
                  const base = currentFile.split("/").pop() || currentFile;
                  const match = Object.keys(d.per_file_staged).find((k) => k === currentFile || k.split("/").pop() === base);
                  if (match) stagedContent = d.per_file_staged[match];
                }
                const stagedForCurrent = stagedContent !== undefined && currentFile &&
                  (d.staged_file === currentFile || (d.per_file_staged && d.per_file_staged[currentFile]) ||
                    (d.per_file_staged && Object.keys(d.per_file_staged).some((k) => k === currentFile || k.split("/").pop() === (currentFile.split("/").pop() || currentFile))));
                if (
                  (d.staged || stagedForCurrent) &&
                  stagedContent !== undefined &&
                  currentFile
                ) {
                  // Use server's per_file_before or existing preChange; never getEditorText() after we may have set staged content
                  const base = currentFile.split("/").pop() || currentFile;
                  const stagedKey = d.per_file_staged && Object.keys(d.per_file_staged).find(
                    (k) => k === currentFile || k.split("/").pop() === base
                  );
                  const beforeContent = (d.per_file_before && stagedKey && d.per_file_before[stagedKey]) ||
                    preChangeContentByFile[currentFile] ||
                    getEditorText();
                  if (preChangeContentByFile[currentFile] === undefined) {
                    preChangeContentByFile[currentFile] = beforeContent;
                  }
                  setViewerContent(String(stagedContent ?? ""));
                  scheduleLspDidChange();
                  stagedContentApplied = true;
                  const directDiff = d.per_file_diffs && currentFile && d.per_file_diffs[currentFile];
                  const foundEntry = !directDiff && d.per_file_diffs && currentFile &&
                    Object.entries(d.per_file_diffs).find(([k]) => k === currentFile || k.split("/").pop() === (currentFile.split("/").pop() || currentFile));
                  let diffToApply = directDiff || (foundEntry && foundEntry[1]) || (d.diff && d.diff.trim()) || "";
                  if (!diffToApply) {
                    diffToApply = buildUnifiedDiffFromContents(currentFile, beforeContent, stagedContent);
                  }
                  if (diffToApply) {
                    applyDiffHighlights(diffToApply);
                    document.getElementById("diffActions").style.display = "flex";
                  }
                }
                if (!renderedMetaSummary && (d.staged || (d.per_file_staged && Object.keys(d.per_file_staged).length > 0))) {
                  const explanation = (d.explanation || d.output || rawResponseBuf || "").trim();
                  const stagedFiles = d.per_file_staged ? Object.keys(d.per_file_staged) : (d.staged_file ? [d.staged_file] : []);
                  const appliedDirectly = d.applied_directly === true;
                  const msg = appliedDirectly
                    ? (stagedFiles.length > 1
                      ? `Changes applied to ${stagedFiles.length} files. Use Ctrl+Z to undo or click Revert.`
                      : `Changes applied. Use Ctrl+Z to undo or click Revert.`)
                    : (stagedFiles.length > 1
                      ? `Staged changes for ${stagedFiles.length} files. Review and click Accept or Reject.`
                      : `Staged changes for ${esc(String(d.staged_file || stagedFiles[0] || ""))}. Review and click Accept or Reject.`);
                  responseEl.innerHTML = explanation
                    ? `<div style="margin-bottom:10px;white-space:pre-wrap;">${esc(explanation)}</div><div style="padding:8px;border:1px solid var(--border);border-radius:8px;background:var(--panel);">${msg}</div>`
                    : `<div style="padding:8px;border:1px solid var(--border);border-radius:8px;background:var(--panel);">${msg}</div>`;
                  responseEl.classList.add("has-content");
                  rawResponseBuf = "";
                  renderedMetaSummary = true;
                }
                if (d.per_file_staged && Object.keys(d.per_file_staged).length > 0) {
                  window._perFileStaged = { ...(window._perFileStaged || {}), ...d.per_file_staged };
                }
                if (d.staged_content !== undefined && d.staged_file) {
                  window._perFileStaged = { ...(window._perFileStaged || {}), [d.staged_file]: d.staged_content };
                }
                if ((d.mode_used === "pipeline" || d.mode_used === "pipeline_repair" || d.mode_used === "continue_agent") && (d.touched?.length > 0 || d.per_file_staged)) {
                  window._lastAgentPrompt = text;
                  window._lastAgentMeta = d;
                  document.getElementById("btnRepair").style.display = "none";
                }
                if (d.failure_kind === "max_iterations") {
                  document.getElementById("btnRepair").style.display = "none";
                }
              } catch {}
            } else if (event === "error") {
              throw new Error(data);
            }
          };
          while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            const parts = buffer.split("\n\n");
            buffer = parts.pop() || "";
            for (const part of parts) {
              if (part.trim()) handleEvent(part);
            }
            if (doneReceived) { controller.abort(); break; }
          }
          if (currentFile && !stagedContentApplied) {
            // Only reload the current file when we don't have pending staged content.
            // If we have staged content from meta but it wasn't applied (e.g. focus_file mismatch),
            // or if diffActions are visible from a prior meta, do NOT overwrite the editor with
            // file content — that would make "changes disappear" while Apply/Reject stay visible.
            const stagedKey = window._perFileStaged && (window._perFileStaged[currentFile] !== undefined
              ? currentFile
              : Object.keys(window._perFileStaged || {}).find((k) => k === currentFile || k.split("/").pop() === (currentFile.split("/").pop() || currentFile)));
            const hasPendingStaged = stagedKey && window._perFileStaged[stagedKey];
            const diffKey = window._perFileDiffs && (window._perFileDiffs[currentFile] !== undefined
              ? currentFile
              : Object.keys(window._perFileDiffs || {}).find((k) => k === currentFile || k.split("/").pop() === (currentFile.split("/").pop() || currentFile)));
            const hasPendingDiff = (window._pendingDiff && window._pendingDiff.trim()) ||
              (diffKey && window._perFileDiffs && window._perFileDiffs[diffKey]);
            if (hasPendingStaged || hasPendingDiff) {
              // Preserve editor: show staged content if we have it, else leave as-is
              if (hasPendingStaged) {
                const beforeContent = preChangeContentByFile[currentFile] ?? getEditorText();
                if (preChangeContentByFile[currentFile] === undefined) {
                  preChangeContentByFile[currentFile] = beforeContent;
                }
                const stagedContent = window._perFileStaged[stagedKey];
                setViewerContent(String(stagedContent ?? ""));
                let diffToApply = (diffKey && window._perFileDiffs?.[diffKey]) || window._pendingDiff || "";
                if (!diffToApply) {
                  diffToApply = buildUnifiedDiffFromContents(currentFile, beforeContent, stagedContent);
                }
                if (diffToApply) {
                  window._pendingDiff = diffToApply;
                  if (!diffKey) {
                    window._perFileDiffs = window._perFileDiffs || {};
                    window._perFileDiffs[stagedKey || currentFile] = diffToApply;
                  }
                  applyDiffHighlights(diffToApply);
                  document.getElementById("diffActions").style.display = "flex";
                }
              }
            } else {
            try {
              const fileData = await api(`/file?path=${encodeURIComponent(currentFile)}`);
              const nextContent = String((fileData && fileData.content) ?? "");
              setViewerContent(nextContent);
              await openLspForPath(currentFile, nextContent);
              // Default diff source: compare pre-request and post-request content.
              const canCompare = Boolean(preRequestFile && preRequestFile === currentFile && preRequestContent !== null);
              const computedDiff = canCompare
                ? buildUnifiedDiffFromContents(currentFile, preRequestContent, nextContent)
                : "";
              let finalHasDiff = false;
              if (computedDiff) {
                finalHasDiff = true;
                preChangeContentByFile[currentFile] = String(preRequestContent);
                window._pendingDiff = computedDiff;
                window._perFileDiffs = window._perFileDiffs || {};
                window._perFileDiffs[currentFile] = computedDiff;
                applyDiffHighlights(computedDiff);
                const computedSummary = summarizeDiff(computedDiff, currentFile || pendingMetaFocusFile || "");
                responseEl.innerHTML = computedSummary;
                responseEl.classList.add("has-content");
                rawResponseBuf = "";
                metaSummaryLocked = true;
                if (activeTranscriptIndex >= 0 && panelTranscript[activeTranscriptIndex]) {
                  const metaText = responseEl.textContent ? responseEl.textContent.trim() : "";
                  if (metaText) {
                    panelTranscript[activeTranscriptIndex].assistant = metaText;
                    renderPanelTranscript();
                  }
                }
              } else if (window._pendingDiff) {
                finalHasDiff = Boolean(String(window._pendingDiff || "").trim());
                preChangeContentByFile[currentFile] = preRequestContent !== null
                  ? String(preRequestContent)
                  : getEditorText();
                applyDiffHighlights(window._pendingDiff);
                window._pendingDiff = null;
              }
              if (!finalHasDiff && pendingNoopRun) {
                responseEl.innerHTML = `<div style="padding:8px;border:1px solid var(--border);border-radius:8px;background:var(--panel);">${pendingNoopBody}</div>`;
                responseEl.classList.add("has-content");
                rawResponseBuf = "";
                clearDiffHighlights();
                metaSummaryLocked = true;
                if (activeTranscriptIndex >= 0 && panelTranscript[activeTranscriptIndex]) {
                  const metaText = responseEl.textContent ? responseEl.textContent.trim() : "";
                  if (metaText) {
                    panelTranscript[activeTranscriptIndex].assistant = metaText;
                    renderPanelTranscript();
                  }
                }
              }
            } catch {}
            }
            // Only reload file list if agent wrote/created files
            const wrote = rawResponseBuf.includes("[Applied") || rawResponseBuf.includes("[Repaired");
            if (wrote) await loadFiles();
          }
          if (!doneReceived) setStatus("Done", "");
        } catch (err) {
          const msg = String(err && err.message ? err.message : err);
          if (msg.includes("request cancelled")) {
            stopTimer("");
            setStatus("Cancelled", "");
            responseEl.textContent = "Cancelled.";
            if (activeTranscriptIndex >= 0 && panelTranscript[activeTranscriptIndex]) {
              panelTranscript[activeTranscriptIndex].assistant = "Cancelled.";
              renderPanelTranscript();
            }
          } else {
            stopTimer("error");
            setStatus("Error", "error");
            responseEl.textContent = `Error: ${msg}`;
            responseEl.classList.add("has-content");
            if (activeTranscriptIndex >= 0 && panelTranscript[activeTranscriptIndex]) {
              panelTranscript[activeTranscriptIndex].assistant = `Error: ${msg}`;
              renderPanelTranscript();
            }
          }
        } finally {
          clearTimeout(slowTimer);
          if (responseThrottleTimer) { clearTimeout(responseThrottleTimer); responseThrottleTimer = null; }
          // Final flush if any remaining buffered response
          if (rawResponseBuf && firstChunk === false && !metaSummaryLocked) flushResponse();
          activeRequest = null;
          modelRunning = false;
          document.getElementById("send").disabled = false;
        }
      }

      // ─── Auto-resize prompt input ───
      promptEl.addEventListener("input", () => {
        promptEl.style.height = "auto";
        promptEl.style.height = Math.min(promptEl.scrollHeight, 80) + "px";
      });

      promptEl.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendPrompt(); }
      });

      // ─── Save ───
      async function saveFile() {
        if (!currentFile) return;
        if (currentFileIsBinary) {
          setStatus("Binary file: saving disabled", "error");
          return;
        }
        const content = getEditorText();
        setStatus(`Saving...`, "busy");
        try {
          await api("/file", { method: "POST", body: JSON.stringify({ path: currentFile, content }) });
          markSaved();
          setStatus("Saved", "");
        } catch { setStatus("Save failed", "error"); }
      }
      function scheduleAutoSave() {
        if (!currentFile) return;
        if (currentFileIsBinary) return;
        // Don't autosave while diff highlights are pending (wait for Accept/Reject)
        if (activeDiffHighlights) return;
        if (saveTimer) clearTimeout(saveTimer);
        saveTimer = setTimeout(saveFile, 800);
      }

      // ─── Event bindings ───
      document.getElementById("send").onclick = sendPrompt;
      cancelBtn.onclick = () => { if (activeRequest) activeRequest.abort(); };
      refreshBtn.onclick = loadFiles;
      openTerminalBtn.onclick = async () => {
        if (openTerminalBtn.disabled) {
          setStatus("Import files first", "error");
          return;
        }
        setTerminalVisible(true);
        try {
          setStatus("Starting terminal...", "busy");
          await ensureTerminalStarted();
          if (terminalStarted) {
            setStatus("Terminal ready", "");
            if (xterm) xterm.focus();
          } else {
            setStatus("Terminal unavailable", "error");
          }
        } catch (err) {
          appendTerminal(`[terminal start error] ${String(err)}\n`, "stderr");
          setStatus("Terminal unavailable", "error");
        }
      };

      runFileBtn.onclick = async () => {
        if (runFileBtn.disabled) return;
        if (!currentFile) {
          setStatus("Select a file to run", "error");
          return;
        }
        const cmd = inferRunCommand(currentFile);
        if (!cmd) {
          setStatus("No run command for this file type", "error");
          return;
        }
        setTerminalVisible(true);
        try {
          await ensureTerminalStarted();
          const cwd = relCwdForCurrentFile();
          if (cwd) sendTerminalCommand(`cd "${cwd}"`);
          sendTerminalCommand(cmd);
          setStatus("Running...", "busy");
        } catch (err) {
          setStatus("Terminal start failed", "error");
        }
      };

      terminalCloseBtn.onclick = async () => {
        try {
          if (window.electronAPI && window.electronAPI.terminalStop) {
            await window.electronAPI.terminalStop();
          }
          terminalStarted = false;
          terminalIsPty = false;
          terminalActiveCwdAbs = "";
          terminalFallbackLineBuffer = "";
          terminalFallbackHistory = [];
          terminalFallbackHistoryIdx = -1;
          terminalFallbackDraft = "";
          if (xterm) {
            xterm.clear();
            xterm.write("\x1b[2J\x1b[H");
          } else {
            terminalOutputEl.textContent = "";
          }
          setStatus("Terminal closed", "");
          setTerminalVisible(false);
        } catch (err) {
          setStatus("Failed to close terminal", "error");
        }
      };

      function handleFallbackTerminalKeydown(e) {
        if (!terminalVisible) return;
        if (!terminalStarted) return;
        if (!window.electronAPI || !window.electronAPI.terminalWrite) return;
        if (e.metaKey || e.ctrlKey) return;
        if (xterm && xterm.textarea && document.activeElement === xterm.textarea) return;

        const data = keyEventToTerminalData(e);
        if (!data) return;
        e.preventDefault();
        handleTerminalInputData(data);
      }

      document.addEventListener("keydown", (e) => {
        // Ctrl+Z / Cmd+Z: revert agent changes when available
        if ((e.key === "z" || e.key === "Z") && (e.ctrlKey || e.metaKey) && !e.shiftKey) {
          const hasRevertable = window._perFileStaged && Object.keys(window._perFileStaged).length > 0;
          if (hasRevertable) {
            e.preventDefault();
            revertChanges();
            return;
          }
        }
        handleFallbackTerminalKeydown(e);
      });

      filesEl.addEventListener("contextmenu", (e) => {
        e.preventDefault();
        e.stopPropagation();
        const li = e.target.closest("li");
        const rel = li && li.dataset ? li.dataset.path || "" : "";
        showExplorerMenu(e.clientX, e.clientY, rel);
      });
      menuNewFileEl.onclick = async (e) => {
        e.preventDefault();
        e.stopPropagation();
        const target = explorerMenuTarget;
        hideExplorerMenu();
        await createExplorerEntry(false, target);
      };
      menuNewFolderEl.onclick = async (e) => {
        e.preventDefault();
        e.stopPropagation();
        const target = explorerMenuTarget;
        hideExplorerMenu();
        await createExplorerEntry(true, target);
      };
      menuRenameFileEl.onclick = async (e) => {
        e.preventDefault();
        e.stopPropagation();
        const target = explorerMenuTarget;
        hideExplorerMenu();
        if (!target) return;
        await renameExplorerEntry(target);
      };
      menuDeleteFileEl.onclick = async (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (!explorerMenuTarget) return;
        const target = explorerMenuTarget;
        hideExplorerMenu();
        setStatus(`Deleting ${target}...`, "busy");
        await deleteExplorerEntry(target);
      };
      document.addEventListener("click", (e) => {
        if (!e.target.closest("#explorerMenu")) hideExplorerMenu();
      });
      window.addEventListener("resize", hideExplorerMenu);

      terminalOutputEl.addEventListener("paste", async (e) => {
        if (!terminalVisible) return;
        if (!window.electronAPI || !window.electronAPI.terminalWrite) return;
        const text = e.clipboardData ? e.clipboardData.getData("text") : "";
        if (!text) return;
        if (!terminalStarted) {
          try { await ensureTerminalStarted(); } catch {}
        }
        if (!terminalStarted) return;
        e.preventDefault();
        e.stopPropagation();
        handleTerminalPasteText(text);
      });

      // Capture paste globally while terminal tab is active, so Cmd+V works
      // even if focus is not on the hidden xterm textarea.
      document.addEventListener("paste", async (e) => {
        if (e.defaultPrevented) return;
        if (!terminalVisible) return;
        if (!window.electronAPI || !window.electronAPI.terminalWrite) return;
        const text = e.clipboardData ? e.clipboardData.getData("text") : "";
        if (!text) return;
        if (!terminalStarted) {
          try { await ensureTerminalStarted(); } catch {}
        }
        if (!terminalStarted) return;
        e.preventDefault();
        handleTerminalPasteText(text);
      });

      if (window.electronAPI && window.electronAPI.onTerminalData) {
        window.electronAPI.onTerminalData((payload) => {
          appendTerminal(payload && payload.data ? payload.data : "", payload && payload.stream);
        });
      }
      if (window.electronAPI && window.electronAPI.onTerminalExit) {
        window.electronAPI.onTerminalExit((payload) => {
          const code = payload && payload.code;
          appendTerminal(`\n[terminal exited: ${code}]\n`, "stdout");
          terminalStarted = false;
          if (Number(code) === 0) setStatus("Run finished", "");
          else setStatus("Run finished with errors", "error");
        });
      }

      async function unselectDirectory() {
        try {
          if (activeRequest) activeRequest.abort();
          if (lspDocChangeTimer) {
            clearTimeout(lspDocChangeTimer);
            lspDocChangeTimer = null;
          }
          try { await lspClient.stop(); } catch {}
          lspPlugins.clearAllDiagnostics(editorView);
          lspCurrentAbsPath = "";
          try {
            if (serverReady) {
              await api("/include", { method: "POST", body: JSON.stringify({ paths: [] }) });
            }
          } catch {}
          includePaths = [];
          currentFile = null;
          currentFileIsBinary = false;
          fileCache = {};
          explorerEntries = [];
          expandedFolders.clear();
          filesEl.innerHTML = "";
          rootEl.value = "";
          hasUserImported = false;
          clearLastSelectedRoot();
          window._pendingDiff = null;
          window._perFileDiffs = {};
          clearDiffHighlights();
          hideImagePreview();
          tabNameEl.textContent = "Welcome";
          titleFilepath.textContent = "";
          statusBarFileEl.textContent = "No file";
          statusBarLangEl.textContent = "Plain Text";
          setLspStatus("off", "LSP disabled (no folder selected)");
          setViewerContent("");
          showWelcome();
          hideExplorerMenu();
          markSaved();
          updateRunButtonState();
          updateTerminalButtonState();
          setStatus("No folder selected", "");
        } catch (err) {
          setStatus("Failed to unselect folder", "error");
          console.error(err);
        }
      }

      async function restoreLastSelectedRoot() {
        const savedRoot = getLastSelectedRoot();
        if (!savedRoot) return false;
        try {
          setStatus("Restoring last folder...", "busy");
          const rootData = await api("/root", {
            method: "POST",
            body: JSON.stringify({ path: savedRoot }),
          });
          rootEl.value = rootData.root || savedRoot;
          hasUserImported = true;
          const filesData = await api(`/files?root=${encodeURIComponent(rootEl.value)}`);
          const allEntries = filesData.files || [];
          const fileOnlyPaths = allEntries.filter((p) => p && !p.endsWith("/"));
          includePaths = fileOnlyPaths;
          await api("/include", { method: "POST", body: JSON.stringify({ paths: fileOnlyPaths }) });
          await loadFiles();
          const restoredFile = await restoreLastOpenedFileForCurrentRoot();
          if (restoredFile) {
            setStatus("Restored last folder and file", "");
          } else {
            setStatus("Restored last folder", "");
          }
          return true;
        } catch (err) {
          console.warn("restoreLastSelectedRoot failed", err);
          rootEl.value = "";
          hasUserImported = false;
          filesEl.innerHTML = "";
          return false;
        }
      }

      async function handlePick(importMode = false, createMode = false) {
        try {
          if (!serverReady) { setStatus("Server not ready", "error"); return; }
          const electronAPI = window.electronAPI;
          if (!electronAPI || (importMode && !electronAPI.pickFiles) || (!importMode && !electronAPI.pickRoot)) {
            setStatus("Import unavailable (not running in app)", "error");
            return;
          }
          const res = importMode
            ? await electronAPI.pickFiles()
            : await electronAPI.pickRoot({ allowCreate: createMode });
          if (!res) return;
          let newRoot, filePaths, isDirectory;
          if (importMode) {
            newRoot = res.root || res;
            filePaths = res.paths || [];
            isDirectory = res.hasDir || false;
          } else {
            newRoot = res; filePaths = []; isDirectory = true;
          }
          if (!newRoot) return;
          newRoot = String(newRoot).trim();
          if (!newRoot) return;
          if (lspDocChangeTimer) {
            clearTimeout(lspDocChangeTimer);
            lspDocChangeTimer = null;
          }
          await lspClient.stop();
          lspPlugins.clearAllDiagnostics(editorView);
          lspCurrentAbsPath = "";
          rootEl.value = newRoot;
          setStatus("Setting root...", "busy");
          const rootData = await api("/root", { method: "POST", body: JSON.stringify({ path: newRoot }) });
          rootEl.value = rootData.root || newRoot;
          rememberLastSelectedRoot(rootEl.value || newRoot);
          hasUserImported = true;
          await new Promise(r => setTimeout(r, 200));
          if (isDirectory || !filePaths.length) {
            const filesData = await api(`/files?root=${encodeURIComponent(rootEl.value)}`);
            const allEntries = filesData.files || [];
            const fileOnlyPaths = allEntries.filter((p) => p && !p.endsWith("/"));
            includePaths = fileOnlyPaths;
            await api("/include", { method: "POST", body: JSON.stringify({ paths: fileOnlyPaths }) });
          } else {
            const relPaths = filePaths
              .map(p => { if (!p) return ""; let r = p; if (p.startsWith(newRoot)) { r = p.slice(newRoot.length).replace(/^\/+/, ""); } return r; })
              .filter(Boolean);
            includePaths = relPaths;
            await api("/include", { method: "POST", body: JSON.stringify({ paths: relPaths }) });
          }
          await loadFiles();
          const fileCount = filesEl.children.length;
          const restored = await restoreLastOpenedFileForCurrentRoot();
          if (restored) {
            setStatus("Restored last opened file", "");
          } else {
            setStatus(`${fileCount} file${fileCount !== 1 ? 's' : ''}`, "");
          }
          if (fileCount === 0) {
            setViewerContent("No files found.\n\n1. Check the directory contains code/text files\n2. Files match allowed extensions\n3. Files are not in ignored directories");
          }
        } catch (err) {
          const msg = String(err && err.message ? err.message : err);
          let displayMsg = "Import failed";
          if (msg.includes("HTTP 400") || msg.includes("HTTP 500")) {
            try {
              const rest = msg.replace(/^HTTP \d+:\s*/, "");
              const json = typeof rest === "string" ? JSON.parse(rest) : rest;
              if (json && json.error) {
                displayMsg = String(json.error);
                console.error("Import error (full traceback):\n", displayMsg);
              }
            } catch (_) {}
          } else if (msg.includes("HTTP 409")) {
            displayMsg = "Root mismatch — try again";
          } else if (msg.includes("Failed to fetch") || msg.includes("NetworkError")) {
            displayMsg = "Server unreachable — is it running?";
          } else if (msg.length < 80) {
            displayMsg = msg;
          } else if (msg.includes("local variable") || msg.includes("UnboundLocalError")) {
            displayMsg = msg.slice(0, 120) + " — see moonlet_import_error.txt in project folder";
          }
          const firstLine = displayMsg.split("\n")[0] || displayMsg;
          const hasTraceback = displayMsg.includes("Traceback") || displayMsg.includes("  File ");
          const statusDisplay = hasTraceback
            ? firstLine + " — open Console (Cmd+Option+I) for traceback"
            : (displayMsg.length > 200 ? firstLine + " (see console)" : displayMsg);
          setStatus(statusDisplay, "error");
          console.error(err);
        }
      }

      pickRootBtn.onclick = () => handlePick(true);
      if (newChatBtn) {
        newChatBtn.onclick = () => startNewChat();
      }
      if (clearRootBtn) {
        clearRootBtn.onclick = () => unselectDirectory();
      }
      if (welcomeSelectFolderBtn) {
        welcomeSelectFolderBtn.onclick = () => handlePick(false, false);
      }
      if (welcomeCreateDirBtn) {
        welcomeCreateDirBtn.onclick = () => handlePick(false, true);
      }
      if (window.electronAPI && window.electronAPI.onMenuImport) {
        window.electronAPI.onMenuImport(() => handlePick(true));
      }
      window.addEventListener("beforeunload", () => {
        if (hasUserImported && rootEl.value) {
          rememberLastSelectedRoot(rootEl.value);
        }
        if (lspDocChangeTimer) {
          clearTimeout(lspDocChangeTimer);
          lspDocChangeTimer = null;
        }
        void lspClient.stop();
      });

      // ─── Accept / Reject / Repair buttons ───
      document.getElementById("btnRevert").onclick = revertChanges;
      const btnRepair = document.getElementById("btnRepair");
      if (btnRepair) btnRepair.onclick = () => requestRepair();

      // ─── Terminal resize (center panel) ───
      (function() {
        const panel = document.getElementById("centerTerminal");
        if (!panel || !terminalResizeHandle) return;
        let startY, startH;
        terminalResizeHandle.addEventListener('mousedown', (e) => {
          e.preventDefault();
          startY = e.clientY;
          startH = panel.offsetHeight;
          const onMove = (ev) => {
            const delta = startY - ev.clientY;
            const h = Math.max(80, Math.min(500, startH + delta));
            panel.style.maxHeight = h + 'px';
            panel.style.height = h + 'px';
          };
          const onUp = () => {
            document.removeEventListener('mousemove', onMove);
            document.removeEventListener('mouseup', onUp);
          };
          document.addEventListener('mousemove', onMove);
          document.addEventListener('mouseup', onUp);
        });
      })();

      // ─── Right panel resize (expandable) ───
      (function() {
        const panel = document.getElementById("rightPanel");
        if (!panel || !rightPanelResize) return;
        let startX, startW;
        rightPanelResize.addEventListener('mousedown', (e) => {
          e.preventDefault();
          startX = e.clientX;
          startW = panel.offsetWidth;
          const onMove = (ev) => {
            const delta = ev.clientX - startX;
            const w = Math.max(280, Math.min(window.innerWidth * 0.5, startW - delta));
            panel.style.setProperty('--right-panel-width', w + 'px');
            panel.style.width = w + 'px';
          };
          const onUp = () => {
            document.removeEventListener('mousemove', onMove);
            document.removeEventListener('mouseup', onUp);
          };
          document.addEventListener('mousemove', onMove);
          document.addEventListener('mouseup', onUp);
        });
      })();

      // ─── Init ───
      showWelcome();
      hasUserImported = false;
      filesEl.innerHTML = "";
      rootEl.value = "";
      updateRunButtonState();
      updateTerminalButtonState();
      setTerminalVisible(false);
      pickRootBtn.disabled = true;
      if (clearRootBtn) clearRootBtn.disabled = true;
      if (newChatBtn) newChatBtn.disabled = true;
      document.getElementById("send").disabled = true;
      setStatus("Starting model...", "busy");
      waitForServer(180000).then(async (ok) => {
        serverReady = ok;
        pickRootBtn.disabled = !ok;
        if (clearRootBtn) clearRootBtn.disabled = !ok;
        if (newChatBtn) newChatBtn.disabled = !ok;
        document.getElementById("send").disabled = !ok;
        if (ok) {
          const restored = await restoreLastSelectedRoot();
          loadConversationHistory();
          if (!restored) {
            rootEl.value = "";
            setStatus("Ready. Import files to begin.", "");
          }
        } else {
          setStatus("Still starting...", "busy");
          // Keep checking in background for slow model loads.
          const retry = setInterval(async () => {
            try {
              const data = await api("/health", {}, 2500);
              if (data && data.status === "ok") {
                clearInterval(retry);
                serverReady = true;
                pickRootBtn.disabled = false;
                if (clearRootBtn) clearRootBtn.disabled = false;
                if (newChatBtn) newChatBtn.disabled = false;
                document.getElementById("send").disabled = false;
                const restored = await restoreLastSelectedRoot();
                loadConversationHistory();
                if (!restored) {
                  rootEl.value = "";
                  setStatus("Ready", "");
                }
              }
            } catch {
              // keep retrying
            }
          }, 3000);
        }
      });
