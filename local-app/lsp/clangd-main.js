const { spawn } = require("child_process");
const path = require("path");
const { JsonRpcMessageParser, encodeJsonRpcMessage } = require("./jsonrpc");

function nowIso() {
  return new Date().toISOString();
}

function normalizeAbsPath(p) {
  return String(p || "").replace(/\\/g, "/").replace(/\/+$/, "");
}

const DEFAULT_SERVER_BY_LANGUAGE = {
  c: { command: "clangd", args: ["--stdio"], configKey: "clangdPath", envKey: "CLANGD_PATH" },
  cpp: { command: "clangd", args: ["--stdio"], configKey: "clangdPath", envKey: "CLANGD_PATH" },
  python: { command: "pyright-langserver", args: ["--stdio"], configKey: "pyrightPath", envKey: "PYRIGHT_PATH" },
  java: { command: "jdtls", args: [], configKey: "jdtlsPath", envKey: "JDTLS_PATH" },
  javascript: {
    command: "typescript-language-server",
    args: ["--stdio"],
    configKey: "typescriptLanguageServerPath",
    envKey: "TYPESCRIPT_LANGUAGE_SERVER_PATH",
  },
  typescript: {
    command: "typescript-language-server",
    args: ["--stdio"],
    configKey: "typescriptLanguageServerPath",
    envKey: "TYPESCRIPT_LANGUAGE_SERVER_PATH",
  },
  go: { command: "gopls", args: [], configKey: "goplsPath", envKey: "GOPLS_PATH" },
  rust: { command: "rust-analyzer", args: [], configKey: "rustAnalyzerPath", envKey: "RUST_ANALYZER_PATH" },
  yaml: { command: "yaml-language-server", args: ["--stdio"], configKey: "yamlLanguageServerPath", envKey: "YAML_LANGUAGE_SERVER_PATH" },
  json: { command: "vscode-json-language-server", args: ["--stdio"], configKey: "jsonLanguageServerPath", envKey: "JSON_LANGUAGE_SERVER_PATH" },
  html: { command: "vscode-html-language-server", args: ["--stdio"], configKey: "htmlLanguageServerPath", envKey: "HTML_LANGUAGE_SERVER_PATH" },
  css: { command: "vscode-css-language-server", args: ["--stdio"], configKey: "cssLanguageServerPath", envKey: "CSS_LANGUAGE_SERVER_PATH" },
  shell: { command: "bash-language-server", args: ["start"], configKey: "bashLanguageServerPath", envKey: "BASH_LANGUAGE_SERVER_PATH" },
};

function supportedLanguages() {
  return Object.keys(DEFAULT_SERVER_BY_LANGUAGE);
}

function resolveServerSpec(languageId, loadConfig) {
  const cfg = (typeof loadConfig === "function" ? loadConfig() : {}) || {};
  const base = DEFAULT_SERVER_BY_LANGUAGE[languageId];
  if (!base) return null;

  const customMap = cfg.lspServers && typeof cfg.lspServers === "object" ? cfg.lspServers : {};
  const custom = customMap[languageId];

  let command = String(
    (custom && typeof custom === "object" ? custom.command : "") ||
      (base.configKey ? cfg[base.configKey] : "") ||
      (base.envKey ? process.env[base.envKey] : "") ||
      base.command ||
      ""
  ).trim();
  if (!command) command = base.command;

  let args = Array.isArray(base.args) ? [...base.args] : [];
  if (custom && typeof custom === "object" && Array.isArray(custom.args)) {
    args = custom.args.map((x) => String(x));
  }

  return { command, args };
}

function createSessionStatusMessage(state, message, extra = {}) {
  return {
    jsonrpc: "2.0",
    method: "moonlet/status",
    params: {
      state,
      message,
      ts: nowIso(),
      ...extra,
    },
  };
}

function createClangdMain({ sendToRenderer, loadConfig, repoRoot }) {
  const sessions = new Map(); // sessionId -> session
  const sessionByRoot = new Map(); // `${languageId}:${rootAbs}` -> sessionId

  function emitMessage(sessionId, message) {
    if (!sendToRenderer) return;
    sendToRenderer("lsp-message", { sessionId, message });
  }

  function emitStatus(sessionId, state, message, extra = {}) {
    emitMessage(sessionId, createSessionStatusMessage(state, message, extra));
  }

  function getSession(sessionId) {
    return sessions.get(String(sessionId || "")) || null;
  }

  function sessionKey(languageId, rootAbs) {
    return `${languageId}:${normalizeAbsPath(rootAbs)}`;
  }

  function startSession(args = {}) {
    const languageId = String(args.languageId || "").trim() || "cpp";
    const rootAbs = normalizeAbsPath(args.rootAbs || repoRoot || process.cwd());

    if (!supportedLanguages().includes(languageId)) {
      return {
        ok: false,
        error: `Unsupported LSP languageId: ${languageId}. Supported: ${supportedLanguages().join(", ")}`,
      };
    }

    const key = sessionKey(languageId, rootAbs);
    const existingId = sessionByRoot.get(key);
    if (existingId && sessions.has(existingId)) {
      const existing = sessions.get(existingId);
      return {
        ok: true,
        sessionId: existing.id,
        reused: true,
        command: existing.command,
        args: existing.args,
      };
    }

    const server = resolveServerSpec(languageId, loadConfig);
    if (!server || !server.command) {
      return {
        ok: false,
        error: `No LSP server configured for ${languageId}`,
      };
    }
    const sessionId = `${languageId}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    const proc = spawn(server.command, server.args || [], {
      cwd: rootAbs || repoRoot || process.cwd(),
      env: {
        ...process.env,
      },
      stdio: ["pipe", "pipe", "pipe"],
    });

    const parser = new JsonRpcMessageParser();
    const session = {
      id: sessionId,
      proc,
      parser,
      key,
      languageId,
      rootAbs,
      command: server.command,
      args: server.args || [],
      alive: true,
    };

    sessions.set(sessionId, session);
    sessionByRoot.set(key, sessionId);

    emitStatus(sessionId, "starting", `Starting ${languageId} LSP (${server.command})...`, {
      rootAbs,
      languageId,
      command: server.command,
    });

    proc.stdout.on("data", (chunk) => {
      parser.push(chunk);
    });

    proc.stderr.on("data", (chunk) => {
      const text = String(chunk || "").trim();
      if (!text) return;
      emitMessage(sessionId, {
        jsonrpc: "2.0",
        method: "moonlet/log",
        params: { level: "stderr", message: text },
      });
    });

    proc.on("spawn", () => {
      emitStatus(sessionId, "indexing", `${languageId} LSP connected. Indexing...`, {
        rootAbs,
        languageId,
        command: server.command,
      });
    });

    proc.on("error", (err) => {
      const message = String(err && err.message ? err.message : err);
      emitStatus(sessionId, "error", `Failed to start ${languageId} LSP (${server.command}): ${message}`, {
        rootAbs,
        languageId,
        command: server.command,
      });
    });

    proc.on("exit", (code, signal) => {
      const s = sessions.get(sessionId);
      if (!s) return;
      s.alive = false;
      sessions.delete(sessionId);
      sessionByRoot.delete(key);
      emitStatus(
        sessionId,
        "stopped",
        `${languageId} LSP exited (code=${code}, signal=${signal || ""})`,
        { code, signal }
      );
    });

    parser.on("message", (message) => {
      emitMessage(sessionId, message);
      if (
        message &&
        message.method === "$/progress" &&
        message.params &&
        message.params.value &&
        message.params.value.kind
      ) {
        const kind = message.params.value.kind;
        if (kind === "end") {
          emitStatus(sessionId, "connected", "LSP connected");
        } else if (kind === "begin" || kind === "report") {
          emitStatus(sessionId, "indexing", "LSP indexing");
        }
      }
    });

    parser.on("error", (err) => {
      emitStatus(sessionId, "error", String(err && err.message ? err.message : err));
    });

    return {
      ok: true,
      sessionId,
      reused: false,
      command: server.command,
      args: server.args || [],
    };
  }

  function stopSession(args = {}) {
    const sessionId = String(args.sessionId || "").trim();
    const session = getSession(sessionId);
    if (!session) return { ok: true };

    session.alive = false;
    sessions.delete(sessionId);
    sessionByRoot.delete(session.key);

    try {
      session.proc.kill("SIGTERM");
    } catch (_) {
      // ignore
    }

    return { ok: true };
  }

  function sendMessage(args = {}) {
    const sessionId = String(args.sessionId || "").trim();
    const message = args.message;
    const session = getSession(sessionId);
    if (!session || !session.proc || !session.proc.stdin || !session.alive) {
      return { ok: false, error: "LSP session is not running" };
    }

    if (!message || typeof message !== "object") {
      return { ok: false, error: "Invalid JSON-RPC message payload" };
    }

    try {
      session.proc.stdin.write(encodeJsonRpcMessage(message));
      return { ok: true };
    } catch (err) {
      return { ok: false, error: String(err && err.message ? err.message : err) };
    }
  }

  function stopAll() {
    for (const [sessionId, session] of Array.from(sessions.entries())) {
      try {
        session.proc.kill("SIGTERM");
      } catch (_) {
        // ignore
      }
      sessions.delete(sessionId);
      sessionByRoot.delete(session.key);
    }
  }

  return {
    startSession,
    stopSession,
    sendMessage,
    stopAll,
  };
}

module.exports = {
  createClangdMain,
};
