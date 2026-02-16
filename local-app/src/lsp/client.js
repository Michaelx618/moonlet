function normalizeAbsPath(p) {
  return String(p || "").replace(/\\/g, "/").replace(/\/+$/, "");
}

export function isLspSupportedPath(relPath) {
  const ext = ((String(relPath || "").split(".").pop() || "").toLowerCase());
  return [
    "c", "h", "cc", "cpp", "cxx", "hpp", "hh", "hxx",
    "py", "pyi",
    "java",
    "js", "jsx", "mjs", "cjs",
    "ts", "tsx",
    "go",
    "rs",
    "json",
    "yaml", "yml",
    "html", "htm",
    "css", "scss", "less",
    "sh", "bash", "zsh",
  ].includes(ext);
}

export function lspLanguageIdForPath(relPath) {
  const ext = ((String(relPath || "").split(".").pop() || "").toLowerCase());
  if (["c", "h"].includes(ext)) return "c";
  if (["cc", "cpp", "cxx", "hpp", "hh", "hxx"].includes(ext)) return "cpp";
  if (["py", "pyi"].includes(ext)) return "python";
  if (ext === "java") return "java";
  if (["js", "jsx", "mjs", "cjs"].includes(ext)) return "javascript";
  if (["ts", "tsx"].includes(ext)) return "typescript";
  if (ext === "go") return "go";
  if (ext === "rs") return "rust";
  if (ext === "json") return "json";
  if (["yaml", "yml"].includes(ext)) return "yaml";
  if (["html", "htm"].includes(ext)) return "html";
  if (["css", "scss", "less"].includes(ext)) return "css";
  if (["sh", "bash", "zsh"].includes(ext)) return "shell";
  return "";
}

export function toFileUri(absPath) {
  const norm = normalizeAbsPath(absPath);
  if (!norm) return "";
  const withLeading = norm.startsWith("/") ? norm : `/${norm}`;
  return `file://${encodeURI(withLeading)}`;
}

export function fromFileUri(uri) {
  const u = String(uri || "");
  if (!u.startsWith("file://")) return "";
  const withoutScheme = u.replace(/^file:\/\//, "");
  try {
    return decodeURI(withoutScheme);
  } catch {
    return withoutScheme;
  }
}

export function createLspClient({ electronAPI, onStatus, onDiagnostics }) {
  let sessionId = "";
  let sessionRootAbs = "";
  let sessionLanguageId = "";
  let initialized = false;
  let nextId = 1;
  let activeUri = "";

  const docs = new Map(); // uri -> {version, absPath, languageId}
  const pending = new Map(); // id -> {resolve,reject,timer}

  function emitStatus(state, message = "") {
    if (typeof onStatus === "function") onStatus({ state, message });
  }

  function onLspMessage(payload) {
    const sid = String(payload && payload.sessionId ? payload.sessionId : "");
    if (!sid || sid !== sessionId) return;

    const msg = payload && payload.message ? payload.message : null;
    if (!msg || typeof msg !== "object") return;

    if (Object.prototype.hasOwnProperty.call(msg, "id")) {
      const id = Number(msg.id);
      const entry = pending.get(id);
      if (entry) {
        pending.delete(id);
        clearTimeout(entry.timer);
        if (Object.prototype.hasOwnProperty.call(msg, "error")) {
          entry.reject(new Error(String(msg.error && msg.error.message ? msg.error.message : JSON.stringify(msg.error))));
        } else {
          entry.resolve(msg.result);
        }
      }
      return;
    }

    if (msg.method === "textDocument/publishDiagnostics") {
      const uri = String(msg.params && msg.params.uri ? msg.params.uri : "");
      const diagnostics = Array.isArray(msg.params && msg.params.diagnostics)
        ? msg.params.diagnostics
        : [];
      if (typeof onDiagnostics === "function") onDiagnostics(uri, diagnostics);
      return;
    }

    if (msg.method === "moonlet/status") {
      const state = String(msg.params && msg.params.state ? msg.params.state : "connected");
      const message = String(msg.params && msg.params.message ? msg.params.message : "");
      emitStatus(state, message);
      return;
    }

    if (msg.method === "window/logMessage") {
      // Ignore LSP log messages (e.g. compile_commands.json, indexing) — don't overwrite status with "error"
      return;
    }

    if (msg.method === "$/progress") {
      const kind = String(msg.params && msg.params.value && msg.params.value.kind ? msg.params.value.kind : "");
      if (kind === "begin" || kind === "report") emitStatus("indexing", "LSP indexing");
      if (kind === "end") emitStatus("connected", "LSP connected");
    }
  }

  if (electronAPI && typeof electronAPI.onLspMessage === "function") {
    electronAPI.onLspMessage((payload) => onLspMessage(payload));
  }

  async function send(message) {
    if (!electronAPI || typeof electronAPI.lspSend !== "function") {
      throw new Error("LSP IPC bridge is unavailable");
    }
    const res = await electronAPI.lspSend({ sessionId, message });
    if (!res || !res.ok) throw new Error(String((res && res.error) || "LSP send failed"));
  }

  async function notify(method, params) {
    await send({ jsonrpc: "2.0", method, params });
  }

  async function request(method, params, timeoutMs = 8000) {
    const id = nextId++;
    await send({ jsonrpc: "2.0", id, method, params });

    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        pending.delete(id);
        reject(new Error(`LSP request timeout: ${method}`));
      }, timeoutMs);
      pending.set(id, { resolve, reject, timer });
    });
  }

  async function ensureSession({ rootAbs, languageId }) {
    const root = normalizeAbsPath(rootAbs);
    if (!root) {
      emitStatus("off", "LSP disabled (no workspace root)");
      return false;
    }
    if (!languageId) {
      emitStatus("off", "LSP disabled (unsupported file type)");
      return false;
    }

    if (
      sessionId &&
      initialized &&
      sessionRootAbs === root &&
      sessionLanguageId === languageId
    ) {
      return true;
    }

    if (
      sessionId &&
      (sessionRootAbs !== root || sessionLanguageId !== languageId)
    ) {
      await stop();
    }

    if (!electronAPI || typeof electronAPI.lspStart !== "function") {
      emitStatus("error", "LSP IPC bridge missing");
      return false;
    }

    emitStatus("starting", "Starting LSP...");
    const startRes = await electronAPI.lspStart({ languageId, rootAbs: root });
    if (!startRes || !startRes.ok || !startRes.sessionId) {
      emitStatus("error", String((startRes && startRes.error) || "Failed to start LSP server"));
      return false;
    }

    sessionId = String(startRes.sessionId);
    sessionRootAbs = root;
    sessionLanguageId = languageId;
    initialized = false;

    try {
      await request("initialize", {
        processId: null,
        rootUri: toFileUri(root),
        capabilities: {
          textDocument: {
            hover: {},
            completion: {
              completionItem: {
                snippetSupport: false,
              },
            },
            definition: {},
            publishDiagnostics: {},
          },
        },
        clientInfo: {
          name: "moonlet",
          version: "0.1.0",
        },
      }, 10000);

      await notify("initialized", {});
      initialized = true;
      emitStatus("connected", "LSP connected");
      return true;
    } catch (err) {
      emitStatus("error", String(err && err.message ? err.message : err));
      return false;
    }
  }

  async function openDocument({ absPath, languageId, text }) {
    const uri = toFileUri(absPath);
    if (!uri || !sessionId || !initialized) return false;

    if (activeUri && activeUri !== uri && docs.has(activeUri)) {
      await closeDocumentByUri(activeUri);
    }

    const existing = docs.get(uri);
    const version = existing ? existing.version + 1 : 1;
    docs.set(uri, {
      version,
      absPath,
      languageId,
    });

    if (existing) {
      await notify("textDocument/didChange", {
        textDocument: { uri, version },
        contentChanges: [{ text: String(text || "") }],
      });
    } else {
      await notify("textDocument/didOpen", {
        textDocument: {
          uri,
          languageId,
          version,
          text: String(text || ""),
        },
      });
    }

    activeUri = uri;
    return true;
  }

  async function changeDocument({ absPath, text }) {
    const uri = toFileUri(absPath);
    const doc = docs.get(uri);
    if (!uri || !doc || !sessionId || !initialized) return false;

    doc.version += 1;
    await notify("textDocument/didChange", {
      textDocument: { uri, version: doc.version },
      contentChanges: [{ text: String(text || "") }],
    });
    return true;
  }

  async function closeDocument(absPath) {
    const uri = toFileUri(absPath);
    if (!uri) return false;
    return closeDocumentByUri(uri);
  }

  async function closeDocumentByUri(uri) {
    if (!uri || !docs.has(uri) || !sessionId || !initialized) {
      docs.delete(uri);
      if (activeUri === uri) activeUri = "";
      return false;
    }

    await notify("textDocument/didClose", {
      textDocument: { uri },
    });
    docs.delete(uri);
    if (activeUri === uri) activeUri = "";
    return true;
  }

  async function requestCompletion({ absPath, position, context }) {
    const uri = toFileUri(absPath);
    if (!uri || !docs.has(uri) || !initialized) return null;
    return request("textDocument/completion", {
      textDocument: { uri },
      position,
      context: context || { triggerKind: 1 },
    });
  }

  async function requestHover({ absPath, position }) {
    const uri = toFileUri(absPath);
    if (!uri || !docs.has(uri) || !initialized) return null;
    return request("textDocument/hover", {
      textDocument: { uri },
      position,
    });
  }

  async function requestDefinition({ absPath, position }) {
    const uri = toFileUri(absPath);
    if (!uri || !docs.has(uri) || !initialized) return null;
    return request("textDocument/definition", {
      textDocument: { uri },
      position,
    });
  }

  async function stop() {
    if (!sessionId) return;

    for (const [id, pendingReq] of pending.entries()) {
      clearTimeout(pendingReq.timer);
      pendingReq.reject(new Error("LSP session stopped"));
      pending.delete(id);
    }

    docs.clear();
    activeUri = "";
    initialized = false;

    if (electronAPI && typeof electronAPI.lspStop === "function") {
      try {
        await electronAPI.lspStop({ sessionId });
      } catch {
        // ignore
      }
    }

    sessionId = "";
    sessionRootAbs = "";
    sessionLanguageId = "";
    emitStatus("off", "LSP stopped");
  }

  return {
    ensureSession,
    openDocument,
    changeDocument,
    closeDocument,
    requestCompletion,
    requestHover,
    requestDefinition,
    stop,
  };
}
