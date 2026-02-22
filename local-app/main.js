const electron = require("electron");
if (!electron || typeof electron !== "object" || !electron.app) {
  console.error(
    "Moonlet must be launched by Electron browser process. " +
      "Unset ELECTRON_RUN_AS_NODE and try again."
  );
  process.exit(1);
}
const {
  app,
  BrowserWindow,
  ipcMain,
  dialog,
  Menu,
} = electron;
const path = require("path");
const fs = require("fs");
const { spawn } = require("child_process");
const net = require("net");
const { createClangdMain } = require("./lsp/clangd-main");
const ptyBackends = [];
const ptyBackendLoadErrors = [];
let ptyBackendsLoaded = false;

const repoRoot = path.join(__dirname, "..");
let serverProc = null;
let serverPort = null;
let terminalProc = null;
let terminalPty = null;
let mainWindow = null;
let lspMain = null;

function ensurePtyBackendsLoaded() {
  if (ptyBackendsLoaded) return;
  ptyBackendsLoaded = true;
  // Lazy-load PTY backends only when terminal is started.
  // This avoids app startup aborts from native module ABI mismatch.
  for (const mod of ["node-pty", "node-pty-prebuilt-multiarch"]) {
    if (ptyBackends.length > 0) break;
    try {
      ptyBackends.push({ name: mod, api: require(mod) });
      continue;
    } catch (_) {
      // fall through
    }
    try {
      const abs = path.join(__dirname, "node_modules", mod);
      ptyBackends.push({ name: mod, api: require(abs) });
    } catch (errAbs) {
      ptyBackendLoadErrors.push(`${mod}: ${String(errAbs)}`);
    }
  }
}

function resolveInsideRoot(rootAbs, relPath) {
  const root = path.resolve(String(rootAbs || ""));
  const rel = String(relPath || "").replace(/^[/\\]+/, "");
  const target = path.resolve(root, rel);
  const within =
    target === root || target.startsWith(root + path.sep);
  if (!within) return null;
  return { root, rel, target };
}

function buildTerminalEnv() {
  const base = { ...process.env };
  const pathParts = [
    "/opt/homebrew/bin",
    "/usr/local/bin",
    "/usr/bin",
    "/bin",
    "/usr/sbin",
    "/sbin",
    ...(String(base.PATH || "").split(":").filter(Boolean)),
  ];
  const dedup = [];
  for (const p of pathParts) {
    if (!dedup.includes(p)) dedup.push(p);
  }
  base.PATH = dedup.join(":");
  base.TERM = base.TERM || "xterm-256color";
  return base;
}

function buildServerPathEnv(existingPath) {
  const localLlamaBin = path.join(repoRoot, "tools", "llama.cpp", "bin");
  const pathParts = [
    localLlamaBin,
    "/opt/homebrew/bin",
    "/usr/local/bin",
    "/usr/bin",
    "/bin",
    "/usr/sbin",
    "/sbin",
    ...(String(existingPath || "").split(":").filter(Boolean)),
  ];
  const dedup = [];
  for (const p of pathParts) {
    if (!dedup.includes(p)) dedup.push(p);
  }
  return dedup.join(":");
}

function resolveShell() {
  if (process.platform === "win32") {
    return process.env.COMSPEC || "cmd.exe";
  }
  const candidates = [
    process.env.SHELL,
    "/bin/zsh",
    "/bin/bash",
    "/bin/sh",
    "zsh",
    "bash",
    "sh",
  ].filter(Boolean);
  for (const c of candidates) {
    try {
      if (c.startsWith("/") && fs.existsSync(c)) return c;
      if (!c.startsWith("/")) return c;
    } catch (_) {
      // ignore invalid candidate
    }
  }
  return "/bin/sh";
}

function sendToRenderer(channel, payload) {
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send(channel, payload);
  }
}

function loadConfig() {
  const cfgPath = path.join(__dirname, "config.json");
  try {
    const raw = fs.readFileSync(cfgPath, "utf-8");
    return JSON.parse(raw);
  } catch {
    return {};
  }
}

function findAvailablePort(startPort) {
  return new Promise((resolve) => {
    const tryPort = (port) => {
      const tester = net
        .createServer()
        .once("error", () => {
          tryPort(port + 1);
        })
        .once("listening", () => {
          tester.close(() => resolve(port));
        })
        .listen(port, "127.0.0.1");
    };
    tryPort(startPort);
  });
}

async function startServer() {
  const cfg = loadConfig();
  const venvPython = path.join(repoRoot, ".venv", "bin", "python3");
  const python = fs.existsSync(venvPython) ? venvPython : "python3";
  const gguf = cfg.ggufPath || process.env.SC2_GGUF;
  if (!gguf) {
    console.warn("SC2_GGUF not set; cannot start server automatically.");
    return null;
  }
  const desiredPort = Number(cfg.port || process.env.SC2_PORT || 8000);
  const portNum = await findAvailablePort(desiredPort);
  const port = String(portNum);
  const env = {
    ...process.env,
    PATH: buildServerPathEnv(process.env.PATH),
    PYTHONDONTWRITEBYTECODE: "1",
    SC2_GGUF: gguf,
    SC2_CTX_TOK: String(cfg.ctxTok || process.env.SC2_CTX_TOK || 8192),
    SC2_THREADS: String(cfg.threads || process.env.SC2_THREADS || 4),
    SC2_GPU_LAYERS: String(
      cfg.gpuLayers !== undefined
        ? cfg.gpuLayers
        : process.env.SC2_GPU_LAYERS || 0
    ),
    SC2_TEMP: String(cfg.temp || process.env.SC2_TEMP || 0.25),
    SC2_TOP_P: String(cfg.topP || process.env.SC2_TOP_P || 0.9),
    SC2_MAX_NEW: String(cfg.maxNew || process.env.SC2_MAX_NEW || 4096),
    SC2_PORT: port,
    SC2_ROOT: cfg.root
      ? path.resolve(repoRoot, cfg.root)
      : process.env.SC2_ROOT || repoRoot,
    SC2_DEBUG: cfg.debug ? "1" : process.env.SC2_DEBUG,
    SC2_DEBUG_LOG: cfg.debugLog || process.env.SC2_DEBUG_LOG,
    SC2_DEBUG_KV: cfg.debugKv ? "1" : process.env.SC2_DEBUG_KV,
    SC2_DEBUG_CHAT: cfg.debugChat ? "1" : process.env.SC2_DEBUG_CHAT,
    SC2_PREFER_CLI: String(
      cfg.preferCli !== undefined
        ? cfg.preferCli
        : process.env.SC2_PREFER_CLI || 0
    ),
    SC2_LLAMA_CLI: cfg.llamaCliPath || process.env.SC2_LLAMA_CLI,
    SC2_USE_LLAMA_SERVER: String(
      cfg.useLlamaServer !== undefined
        ? cfg.useLlamaServer
        : process.env.SC2_USE_LLAMA_SERVER || 1
    ),
    SC2_LLAMA_SERVER_BIN:
      cfg.llamaServerPath || process.env.SC2_LLAMA_SERVER_BIN,
    SC2_LLAMA_SERVER_HOST:
      cfg.llamaServerHost || process.env.SC2_LLAMA_SERVER_HOST || "127.0.0.1",
    SC2_LLAMA_SERVER_PORT: String(
      cfg.llamaServerPort || process.env.SC2_LLAMA_SERVER_PORT || 8012
    ),
    SC2_PIPELINE_IMPL: String(
      cfg.pipelineImpl || process.env.SC2_PIPELINE_IMPL || "rail_v3"
    ),
    SC2_USE_CORE_V2: String(
      cfg.useCoreV2 !== undefined
        ? cfg.useCoreV2
        : process.env.SC2_USE_CORE_V2 || 1
    ),
    SC2_MODEL_PROFILE: String(
      cfg.modelProfile || process.env.SC2_MODEL_PROFILE || "auto"
    ),
    SC2_USE_CHATML_WRAP: String(
      cfg.useChatmlWrap !== undefined
        ? cfg.useChatmlWrap
        : process.env.SC2_USE_CHATML_WRAP || 1
    ),
    ...(cfg.usePipeline === false ? { SC2_USE_STATE_MACHINE_PIPELINE: "false" } : {}),
    SC2_USE_LEGACY_PIPELINE: String(
      cfg.useLegacyPipeline !== undefined
        ? cfg.useLegacyPipeline
        : process.env.SC2_USE_LEGACY_PIPELINE || 0
    ),
    SC2_USE_CHAT_TOOLS: cfg.useChatTools ? "1" : "0",
    SC2_APPROVAL_MODE: String(
      cfg.approvalMode !== undefined
        ? cfg.approvalMode
        : process.env.SC2_APPROVAL_MODE || 1
    ),
    SC2_AUTO_APPLY_ON_SUCCESS: String(
      cfg.autoApplyOnSuccess !== undefined
        ? cfg.autoApplyOnSuccess
        : process.env.SC2_AUTO_APPLY_ON_SUCCESS || 1
    ),
    SC2_USE_CONTINUE_BRIDGE: String(
      cfg.useContinueBridge !== undefined
        ? cfg.useContinueBridge
        : process.env.SC2_USE_CONTINUE_BRIDGE || 0
    ),
    SC2_CONTINUE_CLI_CMD: String(
      cfg.continueCliCmd || process.env.SC2_CONTINUE_CLI_CMD || ""
    ),
    SC2_CONTINUE_NODE_BIN: String(
      cfg.continueNodeBin || process.env.SC2_CONTINUE_NODE_BIN || "node"
    ),
    SC2_CONTINUE_TIMEOUT_S: String(
      cfg.continueTimeoutS || process.env.SC2_CONTINUE_TIMEOUT_S || 180
    ),
    SC2_CONTINUE_PRINT_FORMAT: String(
      cfg.continuePrintFormat || process.env.SC2_CONTINUE_PRINT_FORMAT || ""
    ),
    SC2_CONTINUE_SILENT_PRINT: String(
      cfg.continueSilentPrint !== undefined
        ? cfg.continueSilentPrint
        : process.env.SC2_CONTINUE_SILENT_PRINT || 0
    ),
    SC2_CONTINUE_GLOBAL_DIR: String(
      cfg.continueGlobalDir || process.env.SC2_CONTINUE_GLOBAL_DIR || ""
    ),
    ...(cfg.verifyCommand ? { SC2_VERIFY_CMD: String(cfg.verifyCommand) } : {}),
  };

  const script = path.join(repoRoot, "main.py");
  if (portNum !== desiredPort) {
    console.warn(`Port ${desiredPort} in use, using ${portNum} instead`);
  }
  console.log(`Starting server: ${python} ${script} on port ${port}`);
  // Use arch -arm64 to force native execution even if Electron runs under Rosetta
  serverProc = spawn("/usr/bin/arch", ["-arm64", python, script], {
    cwd: repoRoot,
    env,
    stdio: "inherit",
  });
  serverProc.on("exit", (code, signal) => {
    console.log(`Server exited (code=${code}, signal=${signal})`);
  });
  return port;
}

function createWindow(port) {
  const apiBase = `http://127.0.0.1:${port || 8000}`;
  const iconPath = path.join(repoRoot, "Moonlet.app", "Contents", "Resources", "AppIcon.icns");
  const win = new BrowserWindow({
    width: 1280,
    height: 860,
    minWidth: 900,
    minHeight: 600,
    icon: iconPath,
    title: "Moonlet",
    titleBarStyle: process.platform === "darwin" ? "hiddenInset" : "default",
    trafficLightPosition: { x: 14, y: 12 },
    backgroundColor: "#11111b",
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      preload: path.join(__dirname, "preload.js"),
    },
  });

  win.loadFile(path.join(__dirname, "index.html"), {
    search: `api_base=${encodeURIComponent(apiBase)}`,
  });
  mainWindow = win;
  win.webContents.on("did-finish-load", () => {
    console.log(`Renderer loaded: ${win.webContents.getURL()}`);
  });
  win.webContents.on("did-fail-load", (_e, code, desc, url) => {
    console.error(`Renderer load failed code=${code} desc=${desc} url=${url}`);
  });
  win.webContents.on("console-message", (_e, level, message, line, sourceId) => {
    const lvl = ["log", "warn", "error"][Math.max(0, Math.min(2, Number(level) - 1))] || "log";
    const src = sourceId || "renderer";
    const ln = Number(line) || 0;
    const out = `[renderer:${lvl}] ${src}:${ln} ${message}`;
    if (lvl === "error") console.error(out);
    else if (lvl === "warn") console.warn(out);
    else console.log(out);
  });
  win.on("closed", () => {
    if (mainWindow === win) mainWindow = null;
  });

  const menuTemplate = [
    ...(process.platform === "darwin"
      ? [
          {
            label: app.name,
            submenu: [
              { role: "about" },
              { type: "separator" },
              { role: "quit" },
            ],
          },
        ]
      : []),
    {
      label: "File",
      submenu: [
        {
          label: "Import Files…",
          accelerator: "CmdOrCtrl+O",
          click: () => {
            win.webContents.send("menu-import");
          },
        },
        { type: "separator" },
        { role: process.platform === "darwin" ? "close" : "quit" },
      ],
    },
    { role: "editMenu" },
    { role: "viewMenu" },
    { role: "windowMenu" },
  ];
  const menu = Menu.buildFromTemplate(menuTemplate);
  Menu.setApplicationMenu(menu);

  win.webContents.on("context-menu", (_event, params) => {
    const template = [];
    if (params.isEditable) {
      template.push(
        { role: "undo" },
        { role: "redo" },
        { type: "separator" },
        { role: "cut" },
        { role: "copy" },
        { role: "paste" },
        { role: "selectAll" }
      );
    } else if (params.selectionText) {
      template.push({ role: "copy" });
    }
    if (!template.length) return;
    const ctx = Menu.buildFromTemplate(template);
    ctx.popup({ window: win });
  });
}

app.whenReady().then(async () => {
  // Set dock icon and name on macOS
  if (process.platform === "darwin" && app.dock) {
    const iconPath = path.join(repoRoot, "Moonlet.app", "Contents", "Resources", "AppIcon.icns");
    try { app.dock.setIcon(iconPath); } catch (_) {}
  }
  app.setName("Moonlet");

  const port = await startServer();
  serverPort = port;
  createWindow(port);
  lspMain = createClangdMain({
    sendToRenderer,
    loadConfig,
    repoRoot,
  });

  ipcMain.handle("pick-root", async (_event, opts = {}) => {
    const allowCreate = Boolean(opts && opts.allowCreate);
    const props = ["openDirectory"];
    if (allowCreate) props.push("createDirectory");
    const res = await dialog.showOpenDialog({
      properties: props,
    });
    if (res.canceled || !res.filePaths.length) return null;
    return res.filePaths[0];
  });

  ipcMain.handle("pick-files", async () => {
    const res = await dialog.showOpenDialog({
      properties: ["openFile", "openDirectory", "multiSelections"],
      filters: [{ name: "All Files", extensions: ["*"] }],
    });

    if (res.canceled || !res.filePaths.length) return null;
    const validPaths = res.filePaths.filter((p) => {
      try {
        const stat = fs.statSync(p);
        return stat.isDirectory() || stat.isFile();
      } catch {
        return false;
      }
    });

    if (!validPaths.length) return null;

    const first = validPaths[0];
    let root = first;
    let hasDir = false;
    try {
      const stat = fs.statSync(first);
      if (stat.isFile()) {
        root = path.dirname(first);
      } else {
        hasDir = true;
      }
    } catch {
      // ignore
    }

    const result = { root, paths: validPaths, hasDir };
    return result;
  });

  ipcMain.handle("fs-create-entry", async (_evt, args = {}) => {
    try {
      const resolved = resolveInsideRoot(args.root, args.relPath);
      if (!resolved || !resolved.rel) {
        return { ok: false, error: "Invalid path" };
      }
      const { target } = resolved;
      const exists = fs.existsSync(target);
      if (exists) return { ok: false, error: "Path already exists" };
      const isDirectory = Boolean(args.isDirectory);
      if (isDirectory) {
        fs.mkdirSync(target, { recursive: true });
      } else {
        fs.mkdirSync(path.dirname(target), { recursive: true });
        fs.writeFileSync(target, String(args.content || ""), "utf-8");
      }
      return { ok: true, path: target };
    } catch (err) {
      return { ok: false, error: String(err) };
    }
  });

  ipcMain.handle("fs-delete-entry", async (_evt, args = {}) => {
    try {
      const resolved = resolveInsideRoot(args.root, args.relPath);
      if (!resolved || !resolved.rel) {
        return { ok: false, error: "Invalid path" };
      }
      const { target } = resolved;
      if (!fs.existsSync(target)) return { ok: true };
      fs.rmSync(target, { recursive: true, force: true });
      return { ok: true };
    } catch (err) {
      return { ok: false, error: String(err) };
    }
  });

  ipcMain.handle("fs-rename-entry", async (_evt, args = {}) => {
    try {
      const src = resolveInsideRoot(args.root, args.fromRelPath);
      const dst = resolveInsideRoot(args.root, args.toRelPath);
      if (!src || !dst || !src.rel || !dst.rel) {
        return { ok: false, error: "Invalid path" };
      }
      if (!fs.existsSync(src.target)) {
        return { ok: false, error: "Source does not exist" };
      }
      if (fs.existsSync(dst.target)) {
        return { ok: false, error: "Destination already exists" };
      }
      fs.mkdirSync(path.dirname(dst.target), { recursive: true });
      fs.renameSync(src.target, dst.target);
      return { ok: true, from: src.target, to: dst.target };
    } catch (err) {
      return { ok: false, error: String(err) };
    }
  });

  ipcMain.handle("terminal-start", async (_evt, args = {}) => {
    ensurePtyBackendsLoaded();
    const requestedCwd = String(args.cwd || "").trim();
    let cwd = repoRoot;
    if (requestedCwd) {
      const candidate = path.resolve(requestedCwd);
      if (fs.existsSync(candidate) && fs.statSync(candidate).isDirectory()) {
        cwd = candidate;
      }
    }

    if (terminalPty) {
      try {
        terminalPty.kill();
      } catch (_) {
        // ignore
      }
      terminalPty = null;
    }

    if (terminalProc && !terminalProc.killed) {
      try {
        terminalProc.kill("SIGTERM");
      } catch (_) {
        // ignore
      }
      terminalProc = null;
    }

    const shell = resolveShell();
    const terminalEnv = buildTerminalEnv();

    if (ptyBackends.length === 0 && ptyBackendLoadErrors.length > 0) {
      sendToRenderer("terminal-data", {
        stream: "stderr",
        data:
          "[terminal] PTY backend load errors:\n- " +
          ptyBackendLoadErrors.join("\n- ") +
          "\n",
      });
    }

    // Preferred path: true pseudo-terminal for live echo/cursor/editing.
    if (ptyBackends.length > 0) {
      try {
        for (const backend of ptyBackends) {
          try {
            const ptyArgs =
              process.platform === "win32"
                ? []
                : ["-i"];
            terminalPty = backend.api.spawn(shell, ptyArgs, {
              name: "xterm-256color",
              cols: 120,
              rows: 30,
              cwd,
              env: terminalEnv,
            });
            terminalPty.onData((data) => {
              sendToRenderer("terminal-data", { stream: "stdout", data: String(data || "") });
            });
            terminalPty.onExit(({ exitCode, signal }) => {
              sendToRenderer("terminal-exit", { code: exitCode, signal });
              terminalPty = null;
            });
            sendToRenderer("terminal-data", {
              stream: "stderr",
              data: `[terminal] PTY backend active: ${backend.name}\n`,
            });
            return { ok: true, cwd, shell, pty: true };
          } catch (err) {
            sendToRenderer("terminal-data", {
              stream: "stderr",
              data: `[terminal] PTY backend ${backend.name} failed (${String(err)}).\n`,
            });
            terminalPty = null;
          }
        }
      } catch (_) {
        // fall through
      }
    }

    sendToRenderer("terminal-data", {
      stream: "stderr",
      data: `[terminal] PTY backend unavailable (loaded: ${
        ptyBackends.map((b) => b.name).join(", ") || "none"
      }). Falling back to basic shell mode.\n`,
    });

    const shellArgs =
      process.platform === "win32"
        ? ["/Q", "/K"]
        : ["-i"];

    try {
      terminalProc = spawn(shell, shellArgs, {
        cwd,
        env: terminalEnv,
        stdio: ["pipe", "pipe", "pipe"],
      });
    } catch (err) {
      return { ok: false, cwd, shell, pty: false, error: String(err) };
    }

    terminalProc.stdout.on("data", (chunk) => {
      sendToRenderer("terminal-data", { stream: "stdout", data: chunk.toString() });
    });
    terminalProc.stderr.on("data", (chunk) => {
      sendToRenderer("terminal-data", { stream: "stderr", data: chunk.toString() });
    });
    terminalProc.on("exit", (code, signal) => {
      sendToRenderer("terminal-exit", { code, signal });
      terminalProc = null;
    });
    terminalProc.on("error", (err) => {
      sendToRenderer("terminal-data", { stream: "stderr", data: `\n[terminal error] ${String(err)}\n` });
      terminalProc = null;
    });

    return { ok: true, cwd, shell, pty: false };
  });

  ipcMain.handle("terminal-stop", async () => {
    if (terminalPty) {
      try {
        terminalPty.kill();
      } catch (err) {
        return { ok: false, error: String(err) };
      }
      terminalPty = null;
      return { ok: true };
    }
    if (!terminalProc || terminalProc.killed) return { ok: true };
    try {
      terminalProc.kill("SIGTERM");
    } catch (err) {
      return { ok: false, error: String(err) };
    }
    terminalProc = null;
    return { ok: true };
  });

  ipcMain.on("terminal-write", (_evt, text) => {
    if (terminalPty) {
      try {
        terminalPty.write(String(text || ""));
      } catch (_) {
        // ignore broken PTY
      }
      return;
    }
    if (!terminalProc || terminalProc.killed) return;
    try {
      terminalProc.stdin.write(String(text || ""));
    } catch (_) {
      // ignore broken pipe
    }
  });

  ipcMain.on("terminal-resize", (_evt, size = {}) => {
    if (!terminalPty) return;
    const cols = Number(size.cols || 0);
    const rows = Number(size.rows || 0);
    if (!Number.isFinite(cols) || !Number.isFinite(rows)) return;
    if (cols <= 0 || rows <= 0) return;
    try {
      terminalPty.resize(cols, rows);
    } catch (_) {
      // ignore
    }
  });

  ipcMain.handle("lsp/start", async (_evt, args = {}) => {
    if (!lspMain) return { ok: false, error: "LSP manager unavailable" };
    try {
      return lspMain.startSession(args || {});
    } catch (err) {
      return { ok: false, error: String(err) };
    }
  });

  ipcMain.handle("lsp/stop", async (_evt, args = {}) => {
    if (!lspMain) return { ok: false, error: "LSP manager unavailable" };
    try {
      return lspMain.stopSession(args || {});
    } catch (err) {
      return { ok: false, error: String(err) };
    }
  });

  ipcMain.handle("lsp/send", async (_evt, args = {}) => {
    if (!lspMain) return { ok: false, error: "LSP manager unavailable" };
    try {
      return lspMain.sendMessage(args || {});
    } catch (err) {
      return { ok: false, error: String(err) };
    }
  });

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow(port);
    }
  });
});

app.on("before-quit", () => {
  // Clear llama-server KV cache before shutdown
  if (serverPort) {
    try {
      const http = require("http");
      const req = http.get(`http://127.0.0.1:${serverPort}/clear-cache`, () => {});
      req.on("error", () => {});
      req.end();
    } catch (_) {}
  }
  if (serverProc) {
    serverProc.kill();
    // Ensure llama-server (child of Python) is killed; it can orphan when parent dies
    try {
      require("child_process").spawnSync("pkill", ["-f", "llama-server"], {
        stdio: "ignore",
        timeout: 2000,
      });
    } catch (_) {}
  }
  if (lspMain) {
    try { lspMain.stopAll(); } catch (_) {}
  }
  if (terminalProc) {
    terminalProc.kill();
  }
  if (terminalPty) {
    try { terminalPty.kill(); } catch (_) {}
    terminalPty = null;
  }
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
  }
});
