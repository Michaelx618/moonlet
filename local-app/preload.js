const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("electronAPI", {
  pickRoot: (opts) => ipcRenderer.invoke("pick-root", opts || {}),
  pickFiles: () => ipcRenderer.invoke("pick-files"),
  fsCreateEntry: (args) => ipcRenderer.invoke("fs-create-entry", args || {}),
  fsDeleteEntry: (args) => ipcRenderer.invoke("fs-delete-entry", args || {}),
  fsRenameEntry: (args) => ipcRenderer.invoke("fs-rename-entry", args || {}),
  terminalStart: (args) => ipcRenderer.invoke("terminal-start", args || {}),
  terminalStop: () => ipcRenderer.invoke("terminal-stop"),
  terminalWrite: (text) => ipcRenderer.send("terminal-write", text || ""),
  terminalResize: (size) => ipcRenderer.send("terminal-resize", size || {}),
  lspStart: (args) => ipcRenderer.invoke("lsp/start", args || {}),
  lspStop: (args) => ipcRenderer.invoke("lsp/stop", args || {}),
  lspSend: (args) => ipcRenderer.invoke("lsp/send", args || {}),
  onLspMessage: (cb) => {
    ipcRenderer.removeAllListeners("lsp-message");
    ipcRenderer.on("lsp-message", (_evt, payload) => cb && cb(payload));
  },
  onTerminalData: (cb) => {
    ipcRenderer.removeAllListeners("terminal-data");
    ipcRenderer.on("terminal-data", (_evt, payload) => cb && cb(payload));
  },
  onTerminalExit: (cb) => {
    ipcRenderer.removeAllListeners("terminal-exit");
    ipcRenderer.on("terminal-exit", (_evt, payload) => cb && cb(payload));
  },
  onMenuImport: (cb) => {
    ipcRenderer.removeAllListeners("menu-import");
    ipcRenderer.on("menu-import", () => cb && cb());
  },
});
