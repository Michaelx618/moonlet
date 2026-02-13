# Local Agent UI (no login)

Minimal web UI that talks to the local `main.py` server. No cloud, no login.

## Start the model server
```bash
cd /Users/michael/moonlet
SC2_GGUF="/Users/michael/models/starcoder2-15b-instruct-Q4_K_M.gguf" \
SC2_CTX_TOK=8192 \
SC2_THREADS=8 \
SC2_GPU_LAYERS=0 \
SC2_PORT=8000 \
python3 main.py
```

## Open the UI
Desktop (Electron, no browser) — auto-starts the local model server:
```bash
cd /Users/michael/moonlet/local-app
npm install
npm start
```
This opens a desktop window that talks to the local server at `http://127.0.0.1:8000`.
Server is auto-started by Electron using settings in `local-app/config.json`.

Browser fallback:
- Open `local-app/index.html` directly, or
- Serve it: `cd local-app && python3 -m http.server 5500` → http://127.0.0.1:5500

## Configure server defaults
Edit `local-app/config.json`:
- `ggufPath`: absolute path to your model file
- `ctxTok`: context length (tokens)
- `threads`: CPU threads
- `gpuLayers`: Metal offload layers (0 for CPU only)
- `port`: server port (UI will use this automatically)
- `root`: working directory (relative to repo root)

In the app, use “Pick” to choose a folder (like Cursor’s file picker); it sets the root and reloads files.

## Features
- List and open files.
- View/edit file content.
- Send prompts to the agent (`agent` mode writes files; `chat` mode just replies).
- Save the current file back to disk.
- Run current file from a `Run` button (language-aware command mapping).
- Integrated terminal tab with persistent local shell and streamed output.
- Manual terminal command input (Enter to run).

## Endpoints used (main.py)
- `POST /agent` `{ "text": "..." }`
- `POST /chat` `{ "text": "..." }`
- `GET /files`
- `GET /file?path=relative/path`
- `POST /file` `{ "path": "relative/path", "content": "..." }`

All traffic stays on `127.0.0.1`; no login required.***
