# Tests for Ask, Plan, and Agent routes

These tests verify that the three modes (ask, plan, agent) work as intended, with unit tests (no model), integration tests (real model), and API tests (HTTP `/stream` with mocked handlers).

## Test files

| File | Description |
|------|-------------|
| **test_ask_plan_guards.py** | Unit tests: Ask and Plan routes reject write tool calls (search_replace, write_file, edit_existing_file). Uses a mocked model so no `SC2_MLX_MODEL` is required. Sets `SC2_MLX_MODEL=dummy` so `ai_shell` imports succeed. |
| **test_three_routes_integration.py** | Integration tests: real `run_ask`, `run_plan`, `run_agent` with human-like, vague prompts. **Requires `SC2_MLX_MODEL` (or `SC2_MLX_MODEL_PATH`)**; skipped otherwise. Uses `tests/fixtures/w7` as workspace. |
| **test_stream_api_modes.py** | API tests: POST `/stream` with `mode=ask`, `mode=plan`, `mode=agent`, and `mode=chat` (treated as ask). Starts the HTTP server in a background thread and mocks `run_ask`/`run_plan`/`run_agent` so no model is used. Asserts correct handler is called and SSE meta is returned. |
| **test_agent_create_file.py** | Integration tests that the **agent** can use `create_new_file` / `write_file` for non-trivial tasks: create README from existing code, create a header and include it, create a run script, create helper.c + helper.h. Requires `SC2_MLX_MODEL`. Writes into `tests/fixtures/w7`. |

## How to run

**Unit tests (no model, fast):**
```bash
cd /path/to/moonlet
SC2_MLX_MODEL=dummy python3 -m unittest tests.test_ask_plan_guards -v
```

**Integration tests (real model, slow):**
```bash
SC2_MLX_MODEL=mlx-community/Llama-3.2-1B-Instruct python3 -m unittest tests.test_three_routes_integration -v
```

**API tests (mocked handlers, starts server):**
```bash
SC2_MLX_MODEL=dummy python3 -m unittest tests.test_stream_api_modes -v
```

**Create-file (agent) integration tests (one module, one process):**
```bash
SC2_MLX_MODEL=your/model python3 -m unittest tests.test_agent_create_file -v
```

**Integration tests one at a time (recommended: one test per process, no RAM buildup):**
```bash
cd /path/to/moonlet
python3 tests/run_integration_one_at_a_time.py              # all 18 tests
python3 tests/run_integration_one_at_a_time.py --first 1    # smoke: one test only
python3 tests/run_integration_one_at_a_time.py --skip-port-check   # if no app is running
```
Uses `local-app/config.json` (or env) for model; each test runs in a separate process that exits when done. Close the Moonlet app first. One test can take 2–5 minutes (model load + agent rounds).

**Unit + API tests (no model):**
```bash
SC2_MLX_MODEL=dummy python3 -m unittest tests.test_ask_plan_guards tests.test_stream_api_modes -v
```

## Hard / human-like cases (integration)

- **Ask:** "what does the code in here do?", "can you explain this file?", "what's in this project?"
- **Plan:** "how would we add password validation?", "figure out what needs to change to make it work", "what would we need to do to get the build passing?"
- **Agent:** "add a comment at the top of the file saying // checkpasswd", "replace the blank lines before return 0 with a single comment: // TODO"

These simulate vague or natural user requests and assert that ask/plan never touch files and return valid meta.
