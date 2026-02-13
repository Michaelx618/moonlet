const { EventEmitter } = require("events");

class JsonRpcMessageParser extends EventEmitter {
  constructor() {
    super();
    this._buffer = Buffer.alloc(0);
    this._expectedLength = null;
  }

  push(chunk) {
    if (!chunk || chunk.length === 0) return;
    const next = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
    this._buffer = Buffer.concat([this._buffer, next]);
    this._drain();
  }

  _drain() {
    while (true) {
      if (this._expectedLength == null) {
        const headerEnd = this._buffer.indexOf("\r\n\r\n");
        if (headerEnd < 0) return;

        const headerText = this._buffer.slice(0, headerEnd).toString("utf8");
        this._buffer = this._buffer.slice(headerEnd + 4);

        const length = this._parseContentLength(headerText);
        if (!Number.isFinite(length) || length < 0) {
          this.emit("error", new Error("Invalid JSON-RPC Content-Length header"));
          this._expectedLength = null;
          continue;
        }
        this._expectedLength = length;
      }

      if (this._buffer.length < this._expectedLength) return;

      const body = this._buffer.slice(0, this._expectedLength);
      this._buffer = this._buffer.slice(this._expectedLength);
      this._expectedLength = null;

      try {
        const message = JSON.parse(body.toString("utf8"));
        this.emit("message", message);
      } catch (err) {
        this.emit("error", new Error(`Invalid JSON-RPC payload: ${String(err)}`));
      }
    }
  }

  _parseContentLength(headerText) {
    const lines = String(headerText || "")
      .split("\r\n")
      .map((x) => x.trim())
      .filter(Boolean);
    for (const line of lines) {
      const m = /^Content-Length\s*:\s*(\d+)$/i.exec(line);
      if (m) return Number(m[1]);
    }
    return NaN;
  }
}

function encodeJsonRpcMessage(message) {
  const json = JSON.stringify(message ?? {});
  const body = Buffer.from(json, "utf8");
  const header = Buffer.from(`Content-Length: ${body.length}\r\n\r\n`, "utf8");
  return Buffer.concat([header, body]);
}

module.exports = {
  JsonRpcMessageParser,
  encodeJsonRpcMessage,
};
