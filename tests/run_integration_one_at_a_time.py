"""
Run integration tests one at a time, each in its own process.

Each test gets a fresh process: model loads, one test runs, process exits (no RAM buildup).
Use this instead of running the full suite in one process.

  cd /path/to/moonlet
  python3 tests/run_integration_one_at_a_time.py              # all tests
  python3 tests/run_integration_one_at_a_time.py --first 1    # smoke: only first test

Before running: close the Moonlet app so no server/LLM is using GPU/RAM. The script checks
ports 8000/8002 and exits if something is already listening (so you don't run tests on top of the app).

Expects local-app/config.json (or SC2_MLX_MODEL / SC2_MLX_MODEL_PATH) for model config.
"""

import argparse
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO / "local-app" / "config.json"
FIXTURES_W7 = REPO / "tests" / "fixtures" / "w7"

# Ports the app typically uses (main.js / config.json)
SERVER_PORTS = (8000, 8002)


def _port_in_use(port: int) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            s.connect(("127.0.0.1", port))
            return True
    except (OSError, socket.error, socket.timeout):
        return False


def _warn_if_server_running():
    try:
        in_use = [p for p in SERVER_PORTS if _port_in_use(p)]
    except Exception:
        in_use = []  # if check fails (e.g. restricted env), proceed
    if in_use:
        print(
            f"Port(s) {in_use} are in use. Stop the Moonlet app (and any python main.py server) first,\n"
            "then run the tests again. This avoids two processes using the model/GPU at once.",
            file=sys.stderr,
        )
        sys.exit(1)


def _preflight_checks(skip_port_check: bool) -> None:
    """Fail fast with clear messages if env or fixtures are missing."""
    if not FIXTURES_W7.is_dir():
        print(
            f"Fixtures not found: {FIXTURES_W7}\n"
            "Integration tests need tests/fixtures/w7 (e.g. checkpasswd.c, validate.c, Makefile).",
            file=sys.stderr,
        )
        sys.exit(1)
    has_env_model = os.environ.get("SC2_MLX_MODEL") or os.environ.get("SC2_MLX_MODEL_PATH")
    if has_env_model:
        return
    if not CONFIG_PATH.exists():
        print(
            "No model config. Create local-app/config.json with mlxModel or mlxModelPath,\n"
            "or set SC2_MLX_MODEL or SC2_MLX_MODEL_PATH.",
            file=sys.stderr,
        )
        sys.exit(1)
    try:
        with open(CONFIG_PATH) as f:
            cfg = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"Could not read config: {CONFIG_PATH} — {e}", file=sys.stderr)
        sys.exit(1)
    if not (cfg.get("mlxModel") or cfg.get("mlxModelPath")):
        print(
            "local-app/config.json has no mlxModel or mlxModelPath. Set one, or set SC2_MLX_MODEL / SC2_MLX_MODEL_PATH.",
            file=sys.stderr,
        )
        sys.exit(1)

# One test per process; order: agent create-file, then ask/plan/agent routes. Skip _ExitAfterAllTests.
INTEGRATION_TESTS = [
    # test_agent_create_file
    "tests.test_agent_create_file.TestAgentCreateFile.test_create_readme_from_source",
    "tests.test_agent_create_file.TestAgentCreateFile.test_create_common_header_and_include_in_checkpasswd",
    "tests.test_agent_create_file.TestAgentCreateFile.test_create_run_script_that_builds_and_runs",
    "tests.test_agent_create_file.TestAgentCreateFile.test_create_helper_c_and_header_with_trim_newline",
    "tests.test_agent_create_file.TestAgentCreateFile.test_create_build_doc_from_makefile",
    "tests.test_agent_create_file.TestAgentCreateFile.test_create_gitignore_for_build_artifacts",
    "tests.test_agent_create_file.TestAgentCreateFile.test_create_pass_example_from_format",
    "tests.test_agent_create_file.TestAgentCreateFile.test_create_usage_from_comments",
    "tests.test_agent_create_file.TestAgentCreateFile.test_create_clean_script",
    "tests.test_agent_create_file.TestAgentCreateFile.test_create_sources_list",
    # test_three_routes_integration
    "tests.test_three_routes_integration.TestAskRouteIntegration.test_vague_question_what_does_this_do",
    "tests.test_three_routes_integration.TestAskRouteIntegration.test_slightly_ambiguous_question",
    "tests.test_three_routes_integration.TestAskRouteIntegration.test_ask_no_focus_file",
    "tests.test_three_routes_integration.TestPlanRouteIntegration.test_vague_plan_request",
    "tests.test_three_routes_integration.TestPlanRouteIntegration.test_plan_explore_codebase",
    "tests.test_three_routes_integration.TestPlanRouteIntegration.test_plan_no_focus",
    "tests.test_three_routes_integration.TestAgentRouteIntegration.test_agent_vague_edit_request",
    "tests.test_three_routes_integration.TestAgentRouteIntegration.test_agent_explicit_small_edit",
]


def main():
    parser = argparse.ArgumentParser(description="Run integration tests one per process.")
    parser.add_argument(
        "--first", type=int, default=None, metavar="N",
        help="Run only the first N tests (e.g. --first 1 for a quick smoke test).",
    )
    parser.add_argument(
        "--skip-port-check", action="store_true",
        help="Do not check if server ports are in use (use if you know no app is running).",
    )
    args = parser.parse_args()
    if not args.skip_port_check:
        _warn_if_server_running()
    _preflight_checks(args.skip_port_check)
    tests = INTEGRATION_TESTS
    if args.first is not None:
        tests = INTEGRATION_TESTS[: max(1, args.first)]
        print(f"Running first {len(tests)} test(s) only (--first {args.first}).")
    os.chdir(REPO)
    python = sys.executable
    failed = []
    log_path = REPO / "tests" / "integration_failures.log"
    for i, test_id in enumerate(tests, 1):
        print(f"\n[{i}/{len(tests)}] Running {test_id} ...")
        sys.stdout.flush()
        r = subprocess.run(
            [python, "-m", "unittest", test_id, "-v"],
            cwd=REPO,
            env=os.environ.copy(),
            capture_output=True,
            text=True,
        )
        if r.stdout:
            print(r.stdout, end="" if r.stdout.endswith("\n") else "\n")
        if r.stderr:
            print(r.stderr, end="" if r.stderr.endswith("\n") else "\n", file=sys.stderr)
        if r.returncode != 0:
            failed.append(test_id)
            try:
                with open(log_path, "a") as lf:
                    lf.write(f"\n{'='*60}\n")
                    lf.write(f"FAILED {test_id}\n")
                    lf.write(f"Time: {__import__('datetime').datetime.now().isoformat()}\n")
                    lf.write(f"Return code: {r.returncode}\n")
                    lf.write("--- stdout ---\n")
                    lf.write(r.stdout or "")
                    lf.write("\n--- stderr ---\n")
                    lf.write(r.stderr or "")
                    lf.write("\n")
            except Exception:
                pass
    if failed:
        print(f"\nFailed: {failed}")
        print(f"Failure log appended to: {log_path}")
        sys.exit(1)
    print("\nAll passed.")


if __name__ == "__main__":
    main()
