import { autocompletion } from "@codemirror/autocomplete";
import { linter, setDiagnostics } from "@codemirror/lint";
import { EditorView, hoverTooltip, keymap } from "@codemirror/view";

function clamp(n, min, max) {
  return Math.max(min, Math.min(max, n));
}

// NOTE: LSP columns are UTF-16 code units. We currently treat JS string
// indices as UTF-16 offsets, which is correct for BMP and most code, but
// can be imperfect for some astral symbols/grapheme clusters.
export function lspPositionToOffset(doc, position) {
  if (!position || typeof position.line !== "number" || typeof position.character !== "number") {
    return 0;
  }
  const lineNo = clamp(position.line + 1, 1, doc.lines);
  const line = doc.line(lineNo);
  const char = clamp(position.character, 0, line.length);
  return line.from + char;
}

export function offsetToLspPosition(doc, offset) {
  const safe = clamp(offset, 0, doc.length);
  const line = doc.lineAt(safe);
  return {
    line: line.number - 1,
    character: safe - line.from,
  };
}

function lspRangeToFromTo(doc, range) {
  const from = lspPositionToOffset(doc, range && range.start ? range.start : { line: 0, character: 0 });
  const to = lspPositionToOffset(doc, range && range.end ? range.end : { line: 0, character: 0 });
  return { from: Math.min(from, to), to: Math.max(from, to) };
}

function severityToCm(severity) {
  const val = Number(severity || 1);
  if (val === 1) return "error";
  if (val === 2) return "warning";
  if (val === 3) return "info";
  return "hint";
}

function completionKindToType(kind) {
  const k = Number(kind || 0);
  if (k === 2 || k === 3) return "function";
  if (k === 7) return "class";
  if (k === 6) return "variable";
  if (k === 10) return "property";
  if (k === 14) return "keyword";
  if (k === 15) return "snippet";
  if (k === 17) return "file";
  return "text";
}

function hoverToText(hover) {
  if (!hover || !hover.contents) return "";
  const contents = hover.contents;
  if (typeof contents === "string") return contents;
  if (Array.isArray(contents)) {
    return contents.map((x) => hoverToText({ contents: x })).filter(Boolean).join("\n\n");
  }
  if (contents && typeof contents === "object") {
    if (typeof contents.value === "string") return contents.value;
    if (typeof contents.language === "string" && typeof contents.value === "string") {
      return `(${contents.language})\n${contents.value}`;
    }
  }
  return "";
}

function normalizeLocations(result) {
  if (!result) return [];
  const arr = Array.isArray(result) ? result : [result];
  const out = [];
  for (const item of arr) {
    if (!item || typeof item !== "object") continue;
    if (item.uri && item.range) {
      out.push(item);
      continue;
    }
    if (item.targetUri && item.targetSelectionRange) {
      out.push({
        uri: item.targetUri,
        range: item.targetSelectionRange,
      });
    }
  }
  return out;
}

export function createLspCodeMirrorPlugins({
  getCurrentAbsPath,
  requestCompletion,
  requestHover,
  requestDefinition,
  onDefinition,
}) {
  const diagnosticsByAbsPath = new Map();

  function diagnosticsForView(view) {
    const absPath = String((typeof getCurrentAbsPath === "function" && getCurrentAbsPath()) || "");
    const source = absPath ? diagnosticsByAbsPath.get(absPath) || [] : [];
    return source.map((d) => {
      const range = lspRangeToFromTo(view.state.doc, d.range);
      return {
        from: range.from,
        to: range.to,
        severity: severityToCm(d.severity),
        message: String(d.message || "LSP diagnostic"),
        source: d.source ? String(d.source) : "lsp",
      };
    });
  }

  function refreshDiagnostics(view) {
    if (!view) return;
    const diagnostics = diagnosticsForView(view);
    view.dispatch(setDiagnostics(view.state, diagnostics));
  }

  async function completionSource(context) {
    const absPath = String((typeof getCurrentAbsPath === "function" && getCurrentAbsPath()) || "");
    if (!absPath || typeof requestCompletion !== "function") return null;

    const position = offsetToLspPosition(context.state.doc, context.pos);
    const triggerKind = context.explicit ? 1 : 2;

    let result = null;
    try {
      result = await requestCompletion({
        absPath,
        position,
        context: { triggerKind },
      });
    } catch {
      return null;
    }

    const items = Array.isArray(result)
      ? result
      : Array.isArray(result && result.items)
      ? result.items
      : [];
    if (!items.length) return null;

    const options = items.slice(0, 100).map((item) => {
      const label = String(item.label || "");
      const detail = String(item.detail || "");
      const type = completionKindToType(item.kind);
      const insertText = String(item.insertText || label);
      const textEdit = item.textEdit && item.textEdit.range ? item.textEdit : null;

      const option = {
        label,
        detail,
        type,
      };

      option.apply = (view, _completion, from, to) => {
        if (textEdit) {
          const mapped = lspRangeToFromTo(view.state.doc, textEdit.range);
          view.dispatch({
            changes: {
              from: mapped.from,
              to: mapped.to,
              insert: String(textEdit.newText || ""),
            },
            selection: { anchor: mapped.from + String(textEdit.newText || "").length },
            scrollIntoView: true,
          });
          return;
        }
        view.dispatch({
          changes: {
            from,
            to,
            insert: insertText,
          },
          selection: { anchor: from + insertText.length },
        });
      };

      return option;
    });

    return {
      from: context.pos,
      options,
      validFor: /^\w*$/,
    };
  }

  const hoverExt = hoverTooltip(async (view, pos) => {
    const absPath = String((typeof getCurrentAbsPath === "function" && getCurrentAbsPath()) || "");
    if (!absPath || typeof requestHover !== "function") return null;

    try {
      const hover = await requestHover({
        absPath,
        position: offsetToLspPosition(view.state.doc, pos),
      });
      const text = hoverToText(hover).trim();
      if (!text) return null;

      const range = hover && hover.range ? lspRangeToFromTo(view.state.doc, hover.range) : { from: pos, to: pos };
      return {
        pos: range.from,
        end: range.to,
        above: true,
        create() {
          const dom = document.createElement("div");
          dom.style.maxWidth = "520px";
          dom.style.whiteSpace = "pre-wrap";
          dom.style.padding = "8px 10px";
          dom.style.borderRadius = "6px";
          dom.style.background = "var(--bg-overlay)";
          dom.style.color = "var(--text)";
          dom.style.border = "1px solid var(--border)";
          dom.style.fontFamily = "var(--font-mono)";
          dom.style.fontSize = "12px";
          dom.textContent = text;
          return { dom };
        },
      };
    } catch {
      return null;
    }
  });

  async function runDefinition(view, pos) {
    const absPath = String((typeof getCurrentAbsPath === "function" && getCurrentAbsPath()) || "");
    if (!absPath || typeof requestDefinition !== "function") return false;

    let res = null;
    try {
      res = await requestDefinition({
        absPath,
        position: offsetToLspPosition(view.state.doc, pos),
      });
    } catch {
      return false;
    }

    const locations = normalizeLocations(res);
    if (!locations.length) return false;
    if (typeof onDefinition === "function") {
      return Boolean(await onDefinition(locations[0]));
    }
    return false;
  }

  const definitionKeymap = keymap.of([
    {
      key: "F12",
      run(view) {
        const pos = view.state.selection.main.head;
        void runDefinition(view, pos);
        return true;
      },
    },
  ]);

  const cmdClickHandler = EditorView.domEventHandlers({
    mousedown(event, view) {
      const isMod = event.metaKey || event.ctrlKey;
      if (!isMod || event.button !== 0) return false;
      const pos = view.posAtCoords({ x: event.clientX, y: event.clientY });
      if (pos == null) return false;
      event.preventDefault();
      void runDefinition(view, pos);
      return true;
    },
  });

  const lintExt = linter(() => [], { delay: 0 });
  const completionExt = autocompletion({
    override: [completionSource],
    activateOnTyping: true,
    maxRenderedOptions: 80,
  });

  function updateDiagnostics(absPath, diagnostics, view) {
    const key = String(absPath || "");
    if (!key) return;
    diagnosticsByAbsPath.set(key, Array.isArray(diagnostics) ? diagnostics : []);
    refreshDiagnostics(view);
  }

  function clearDiagnostics(absPath, view) {
    const key = String(absPath || "");
    if (key) diagnosticsByAbsPath.delete(key);
    refreshDiagnostics(view);
  }

  function clearAllDiagnostics(view) {
    diagnosticsByAbsPath.clear();
    refreshDiagnostics(view);
  }

  return {
    extensions: [lintExt, completionExt, hoverExt, definitionKeymap, cmdClickHandler],
    updateDiagnostics,
    clearDiagnostics,
    clearAllDiagnostics,
    refreshDiagnostics,
  };
}
