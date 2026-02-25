import os
from .model_profiles import resolve_model_profile

# Model/runtime knobs
MODEL_NAME = os.getenv("SC2_MODEL", "bigcode/starcoder2-3b")
# Default output budget for model replies
MAX_NEW = int(os.getenv("SC2_MAX_NEW", "8000"))
PLAN_MAX_NEW = int(os.getenv("SC2_PLAN_MAX_NEW", "4096"))
DIFF_MAX_NEW = int(os.getenv("SC2_DIFF_MAX_NEW", "4096"))
PATCH_MAX_NEW = int(os.getenv("SC2_PATCH_MAX_NEW", "2000"))
CHAT_MAX_NEW = int(os.getenv("SC2_CHAT_MAX_NEW", "768"))
CHAT_SHORT_MAX_NEW = int(os.getenv("SC2_CHAT_SHORT_MAX_NEW", "128"))
CHAT_SHORT_INPUT_CHARS = int(os.getenv("SC2_CHAT_SHORT_INPUT_CHARS", "24"))
TEMPERATURE = float(os.getenv("SC2_TEMP", "0.25"))
PATCH_TEMP = float(os.getenv("SC2_PATCH_TEMP", "0.1"))  # Lower for PATCH to reduce hallucination
TOP_P = float(os.getenv("SC2_TOP_P", "0.9"))
DEBUG = os.getenv("SC2_DEBUG", "").lower() in ("1", "true", "yes")
# SC2_DEBUG_KV=1: log KV cache slot, session key, and server response cache stats
DEBUG_KV_CACHE = os.getenv("SC2_DEBUG_KV", "").lower() in ("1", "true", "yes")
# SC2_DEBUG_CHAT=1: log chat prompt, context size, rounds, tools used (for debugging "no context")
DEBUG_CHAT = os.getenv("SC2_DEBUG_CHAT", "").lower() in ("1", "true", "yes")
DEBUG_LOG_PATH = os.getenv(
    "SC2_DEBUG_LOG", "/Users/michael/moonlet/runtime-debug.log"
)
# SC2_DEBUG_DUMP_VERBOSE=1: write full model output to debug log (no truncation).
# Default: true (full dump). Set to false for truncated output.
DEBUG_DUMP_VERBOSE = os.getenv("SC2_DEBUG_DUMP_VERBOSE", "true").lower() in ("1", "true", "yes")
DEBUG_DUMP_MAX_LINES = int(os.getenv("SC2_DEBUG_DUMP_MAX_LINES", "20"))
DEBUG_DUMP_MAX_CHARS = int(os.getenv("SC2_DEBUG_DUMP_MAX_CHARS", "2000"))
GEN_TIMEOUT = int(os.getenv("SC2_GEN_TIMEOUT", "300"))
MAX_MODEL_CALLS = int(os.getenv("SC2_MAX_MODEL_CALLS", "2"))
REP_WINDOW = int(os.getenv("SC2_REP_WINDOW", "40"))
REP_MIN_CHARS = int(os.getenv("SC2_REP_MIN_CHARS", "200"))
REP_MIN_TOKENS = int(os.getenv("SC2_REP_MIN_TOKENS", "120"))
USE_RG_HINT_DEFAULT = os.getenv("SC2_USE_RG_HINT_DEFAULT", "true").lower() != "false"
SOFT_DIFF_POSTCHECK = os.getenv("SC2_SOFT_DIFF_POSTCHECK", "false").lower() != "false"
SKIP_COMPILE_CHECKS = os.getenv("SC2_SKIP_COMPILE_CHECKS", "false").lower() != "false"
RELAX_SEMANTIC_CHECKS = os.getenv("SC2_RELAX_SEMANTIC_CHECKS", "false").lower() != "false"
REVERT_ON_FAILURE = os.getenv("SC2_REVERT_ON_FAILURE", "false").lower() in ("1", "true", "yes")
# If true, stage edits for UI Accept/Reject. If false, apply directly to disk (user can Ctrl+Z or Revert).
STAGE_EDITS = os.getenv("SC2_STAGE_EDITS", "false").lower() in ("1", "true", "yes")
# Analysis symbol allowlist can over-constrain edits; keep off by default.
ENFORCE_ANALYSIS_ALLOWLIST = os.getenv("SC2_ENFORCE_ANALYSIS_ALLOWLIST", "false").lower() in ("1", "true", "yes")

# Relevance pipeline: when True, find_relevant_files returns [] (no FILES fed to model)
DISABLE_RELEVANCE = os.getenv("SC2_DISABLE_RELEVANCE", "true").lower() in ("1", "true", "yes")

# Semantic search (optional: pip install sentence-transformers)
SEMANTIC_SEARCH_ENABLED = os.getenv("SC2_SEMANTIC_SEARCH", "").lower() in ("1", "true", "yes")
EMBEDDING_MODEL = os.getenv("SC2_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
SEMANTIC_TOP_K = int(os.getenv("SC2_SEMANTIC_TOP_K", "5"))

# Patch protocol (unified diff contract) knobs
PATCH_LEGACY_FILEBLOCK_FALLBACK = os.getenv(
    "SC2_PATCH_LEGACY_FILEBLOCK_FALLBACK", "false"
).lower() in ("1", "true", "yes")
PATCH_REJECT_BINARY_LIKE = os.getenv("SC2_PATCH_REJECT_BINARY_LIKE", "true").lower() in (
    "1",
    "true",
    "yes",
)
PATCH_STRICT_PARSE = os.getenv("SC2_PATCH_STRICT_PARSE", "true").lower() in (
    "1",
    "true",
    "yes",
)
PATCH_MAX_FILES_PER_DIFF = int(os.getenv("SC2_PATCH_MAX_FILES_PER_DIFF", "5"))
PATCH_MAX_HUNKS_PER_FILE = int(os.getenv("SC2_PATCH_MAX_HUNKS_PER_FILE", "24"))
PATCH_CONTEXT_WINDOW_MIN = int(os.getenv("SC2_PATCH_CONTEXT_WINDOW_MIN", "80"))
PATCH_CONTEXT_WINDOW_MAX = int(os.getenv("SC2_PATCH_CONTEXT_WINDOW_MAX", "200"))
PATCH_FULL_FILE_MAX_LINES = int(os.getenv("SC2_PATCH_FULL_FILE_MAX_LINES", "500"))
PATCH_TINY_SINGLE_HUNK_MAX_LINES = int(
    os.getenv("SC2_PATCH_TINY_SINGLE_HUNK_MAX_LINES", "200")
)
# Freedom mode: cap total context chars to avoid bloated prompts (default 5000)
FREEDOM_CONTEXT_MAX_CHARS = int(os.getenv("SC2_FREEDOM_CONTEXT_MAX_CHARS", "5000"))
LLAMA_SERVER_CACHE_SLOTS = int(
    os.getenv("SC2_LLAMA_SERVER_CACHE_SLOTS", "32")
)
# Optional GGUF path for local quantized model (llama.cpp backend)
GGUF_PATH = os.getenv("SC2_GGUF")
# Optional MLX model (HuggingFace id or local path). When set, use mlx_lm instead of GGUF.
# Example: SC2_MLX_MODEL=mlx-community/Llama-3.2-3B-Instruct-4bit
# For faster load (skip hub), set SC2_MLX_MODEL_PATH to the cache snapshot dir, e.g.:
#   ~/.cache/huggingface/hub/models--mlx-community--Qwen2.5-Coder-14B-Instruct-4bit/snapshots/<commit_hash>
MLX_MODEL = os.getenv("SC2_MLX_MODEL", "").strip() or None
MLX_MODEL_PATH = os.getenv("SC2_MLX_MODEL_PATH", "").strip() or None
# KV cache size for MLX (max prompt tokens to cache; improves throughput). 0 = use mlx_lm default.
MLX_MAX_KV_SIZE = int(os.getenv("SC2_MLX_MAX_KV_SIZE", "16384"))
# Hard cap on agent prompt length when using MLX (reduce hallucination from oversized context).
MLX_AGENT_PROMPT_MAX_CHARS = int(os.getenv("SC2_MLX_AGENT_PROMPT_MAX_CHARS", "32000"))
# When False, MLX uses same ChatML wrap as GGUF (no tokenizer chat template). Can reduce edit hallucination.
MLX_USE_CHAT_TEMPLATE = os.getenv("SC2_MLX_USE_CHAT_TEMPLATE", "false").lower() in ("1", "true", "yes")
MODEL_PROFILE_NAME = os.getenv("SC2_MODEL_PROFILE", "auto").strip().lower()
MODEL_PROFILE = resolve_model_profile(MODEL_PROFILE_NAME, GGUF_PATH or "")
GGUF_CTX = int(os.getenv("SC2_CTX_TOK", "4096"))
GGUF_THREADS = int(os.getenv("SC2_THREADS", str(os.cpu_count() or 4)))
GGUF_GPU_LAYERS = int(os.getenv("SC2_GPU_LAYERS", "-1"))
LLAMA_SERVER_ENABLED = os.getenv("SC2_USE_LLAMA_SERVER", "true").lower() in ("1", "true", "yes")
LLAMA_SERVER_HOST = os.getenv("SC2_LLAMA_SERVER_HOST", "127.0.0.1")
LLAMA_SERVER_PORT = int(os.getenv("SC2_LLAMA_SERVER_PORT", "8012"))
LLAMA_SERVER_BIN = os.getenv("SC2_LLAMA_SERVER_BIN", "")
LLAMA_SERVER_START_TIMEOUT = int(os.getenv("SC2_LLAMA_SERVER_START_TIMEOUT", "90"))
# Use chat completions with tools for direct tool calls (no text parsing). Requires --jinja.
if os.getenv("SC2_USE_CHAT_TOOLS") is not None:
    USE_CHAT_TOOLS = os.getenv("SC2_USE_CHAT_TOOLS", "false").lower() in ("1", "true", "yes")
else:
    USE_CHAT_TOOLS = bool(MODEL_PROFILE.use_chat_tools)
# Wrap raw prompts in chatml tags for chat-finetuned models via /completion endpoint.
# Default true: most GGUF models (DeepSeek-Coder, etc.) are chat-tuned and need this.
if os.getenv("SC2_USE_CHATML_WRAP") is not None:
    USE_CHATML_WRAP = os.getenv("SC2_USE_CHATML_WRAP", "true").lower() in ("1", "true", "yes")
else:
    USE_CHATML_WRAP = bool(MODEL_PROFILE.use_chatml_wrap)


# Agent/runtime knobs
IGNORE_DIRS = {".git", "node_modules", "__pycache__"}
MAX_CTX_BYTES = int(os.getenv("SC2_CTX_BYTES", "4000"))
MAX_LIST_FILES = int(os.getenv("SC2_MAX_FILES", "40"))
SERVER_PORT = int(os.getenv("SC2_PORT", "8000"))
BUFFER_OUTPUT = os.getenv("SC2_BUFFER", "1") not in {"0", "false", "False"}
CHAT_BUFFER_PROMPT_CHARS = int(os.getenv("SC2_CHAT_BUFFER_PROMPT_CHARS", "3200"))
CHAT_BUFFER_INPUT_CHARS = int(os.getenv("SC2_CHAT_BUFFER_INPUT_CHARS", "400"))
# Chat mode: max files to include in context (0 = list only, no content; 12 = read up to 12 files)
CHAT_INCLUDE_MAX_FILES = int(os.getenv("SC2_CHAT_INCLUDE_MAX_FILES", "12"))
STATE_PATH = os.getenv("SC2_STATE_PATH", "~/.moonlet_state.json")
DISABLE_HISTORY = os.getenv("SC2_DISABLE_HISTORY", "false").lower() in ("1", "true", "yes")
MAX_BUFFER_CHARS = int(os.getenv("SC2_MAX_BUFFER_CHARS", "200000"))
# Prompt char budget (raise to reduce truncation; ~4 chars/token). Default 8000 tokens worth.
CTX_CHAR_BUDGET = int(
    os.getenv(
        "SC2_CTX_CHAR_BUDGET",
        "32000",  # ~8000 tokens; was (GGUF_CTX or 2048) * 3
    )
)
# Multi-file agent limits
MAX_PLAN_FILES = int(os.getenv("SC2_MAX_PLAN_FILES", "5"))
MAX_MULTI_MODEL_CALLS = int(os.getenv("SC2_MAX_MULTI_CALLS", "8"))

# Agent tool loop: max rounds (SC2_MAX_TOOL_ROUNDS). 0 = no cap (Continue-style: run until
# model produces output without tool calls). If > 0, loop exits after that many rounds.
MAX_TOOL_ROUNDS = int(os.getenv("SC2_MAX_TOOL_ROUNDS", "0"))
# When True, do not restrict read/edit to imported files; allow any path under root (index/include filter disabled).
# Default True for now; set SC2_DISABLE_INDEX=0 to re-enable the include filter.
DISABLE_INDEX = os.getenv("SC2_DISABLE_INDEX", "true").lower() in ("1", "true", "yes")
# Index gate: when true, keep index empty until user imports/selects files.
INDEX_REQUIRES_IMPORT = os.getenv("SC2_INDEX_REQUIRES_IMPORT", "true").lower() in ("1", "true", "yes")
# Content-addressed indexing: SQLite catalog, incremental refresh, code snippets, chunks, FTS, embeddings.
# When True, rebuild_index() also runs the indexer refresh. Off by default for now; set SC2_CONTINUE_INDEX=1 to enable.
CONTINUE_INDEX_ENABLED = False  # os.getenv("SC2_CONTINUE_INDEX", "false").lower() in ("1", "true", "yes")
# Which index artifacts to build (all default true except embeddings, which requires SEMANTIC_SEARCH_ENABLED).
INDEX_ENABLE_CODE_SNIPPETS = os.getenv("SC2_INDEX_CODE_SNIPPETS", "true").lower() in ("1", "true", "yes")
INDEX_ENABLE_FTS = os.getenv("SC2_INDEX_FTS", "true").lower() in ("1", "true", "yes")
INDEX_ENABLE_CHUNKS = os.getenv("SC2_INDEX_CHUNKS", "true").lower() in ("1", "true", "yes")
# Embeddings index: explicit SC2_INDEX_EMBEDDINGS, or default to SEMANTIC_SEARCH_ENABLED (embed model available).
_index_emb_env = os.getenv("SC2_INDEX_EMBEDDINGS", "").strip().lower()
if _index_emb_env in ("0", "false", "no"):
    INDEX_ENABLE_EMBEDDINGS = False
elif _index_emb_env in ("1", "true", "yes"):
    INDEX_ENABLE_EMBEDDINGS = True
else:
    INDEX_ENABLE_EMBEDDINGS = os.getenv("SC2_SEMANTIC_SEARCH", "").lower() in ("1", "true", "yes")
# Chunk size (lines per chunk) for chunk index.
INDEX_CHUNK_MAX_LINES = int(os.getenv("SC2_INDEX_CHUNK_MAX_LINES", "300"))

APPROVAL_MODE = os.getenv("SC2_APPROVAL_MODE", "true").lower() in ("1", "true", "yes")
AUTO_APPLY_ON_SUCCESS = os.getenv("SC2_AUTO_APPLY_ON_SUCCESS", "true").lower() in ("1", "true", "yes")
MAX_REPAIR_ITERATIONS = int(os.getenv("SC2_MAX_REPAIR_ITERATIONS", "5"))
# search_replace: when True, use tool loop (feed results back to model); when False, one-shot
USE_SEARCH_REPLACE_LOOP = os.getenv("SC2_USE_SEARCH_REPLACE_LOOP", "false").lower() in (
    "1",
    "true",
    "yes",
)
MAX_EDIT_ROUNDS = int(os.getenv("SC2_MAX_EDIT_ROUNDS", "4"))
# Add one-shot example in prompt to match training format (SC2_PIPELINE_ONE_SHOT=0 to disable)
PIPELINE_ONE_SHOT = os.getenv("SC2_PIPELINE_ONE_SHOT", "1").lower() not in ("0", "false", "no")
VERIFY_CMD = os.getenv("SC2_VERIFY_CMD", "").strip() or None
VERIFY_TIMEOUT = int(os.getenv("SC2_VERIFY_TIMEOUT", "60"))
# Max retries after verify failure (compile error -> feed error to model -> re-edit -> re-verify)
MAX_VERIFY_RETRIES = int(os.getenv("SC2_MAX_VERIFY_RETRIES", "2"))
# Optional override for running the built executable (e.g. "./checkpasswd < input.txt")
RUN_CMD = os.getenv("SC2_RUN_CMD", "").strip() or None
# Timeout for running the executable after build (seconds, 0 = no timeout)
RUN_TIMEOUT = int(os.getenv("SC2_RUN_TIMEOUT", "0"))
# Cap tool results fed back to model (0 = no limit, full freedom for read results)
MAX_TOOL_RESULT_CHARS = int(os.getenv("SC2_MAX_TOOL_RESULT_CHARS", "0"))
# Log tool calls to stderr (set SC2_DEBUG_TOOLS=0 to disable)
DEBUG_TOOLS = os.getenv("SC2_DEBUG_TOOLS", "1").lower() not in ("0", "false", "no")
# Disable focus file context (no file content in prompt; set SC2_DISABLE_FOCUS_FILE=1)
DISABLE_FOCUS_FILE = os.getenv("SC2_DISABLE_FOCUS_FILE", "1").lower() in ("1", "true", "yes")
# When true, prompt lists file paths only (model uses [[[read:path]]]).
# When false (default), file content is always embedded in the prompt.
USE_AGENT_READ_TOOLS = os.getenv("SC2_USE_AGENT_READ_TOOLS", "false").lower() in ("1", "true", "yes")
# Prompt caps for complex tasks (avoid dropping important details)
MAX_INSTRUCTION_CHARS = int(os.getenv("SC2_MAX_INSTRUCTION_CHARS", "1500"))
MAX_FOCUS_CONTENT_CHARS = int(os.getenv("SC2_MAX_FOCUS_CONTENT_CHARS", "12000"))
MAX_REF_CONTENT_CHARS = int(os.getenv("SC2_MAX_REF_CONTENT_CHARS", "12000"))
MAX_READ_FILES_IN_PROMPT = int(os.getenv("SC2_MAX_READ_FILES_IN_PROMPT", "15"))
# @Code / @Folder context (Continue-style)
MAX_CODE_SNIPPET_LINES = int(os.getenv("SC2_MAX_CODE_SNIPPET_LINES", "40"))
MAX_FOLDER_CONTEXT_FILES = int(os.getenv("SC2_MAX_FOLDER_CONTEXT_FILES", "25"))
# Continue default agent system message (aligned with core/llm/defaultSystemMessages.ts DEFAULT_AGENT_SYSTEM_MESSAGE)
_BASE_AGENT_DEFAULT = """\
<important_rules>
You are in agent mode.

If you need to use multiple tools, you can call multiple read-only tools simultaneously.

Always include the language and file name in the info string when you write code blocks.
If you are editing "src/main.py" for example, your code block should start with ```python src/main.py

For larger codeblocks (>20 lines), use brief language-appropriate placeholders for unmodified sections, e.g. '// ... existing code ...'

However, only output codeblocks for suggestion and demonstration purposes, for example, when enumerating multiple hypothetical options. For implementing changes, use the edit tools.

</important_rules>"""
BASE_AGENT_SYSTEM_MESSAGE = os.getenv("SC2_BASE_AGENT_SYSTEM_MESSAGE", _BASE_AGENT_DEFAULT.strip())

# RAIL v3 budgets/interlocks
RAIL_MAX_ACTIONS = int(os.getenv("SC2_RAIL_MAX_ACTIONS", "5"))
RAIL_MAX_PATCH_FILES = int(os.getenv("SC2_RAIL_MAX_PATCH_FILES", "3"))
RAIL_MAX_PATCH_LINES = int(os.getenv("SC2_RAIL_MAX_PATCH_LINES", "120"))
RAIL_SHADOW_ROOT = os.getenv("SC2_RAIL_SHADOW_ROOT", "/tmp/shadow_repo")
RAIL_MAX_ITERATIONS = int(os.getenv("SC2_RAIL_MAX_ITERATIONS", "8"))
RAIL_TOP_K_RELEVANT = int(os.getenv("SC2_RAIL_TOP_K_RELEVANT", "5"))
RAIL_ACTION_MAX_NEW = int(os.getenv("SC2_RAIL_ACTION_MAX_NEW", "600"))

ALLOWED_EXTS = {
    # Documents / markup
    "txt", "md", "mdx", "rst", "adoc", "tex", "org",
    # Web
    "html", "htm", "css", "scss", "sass", "less", "styl",
    "js", "jsx", "mjs", "cjs",
    "ts", "tsx", "mts", "cts",
    "vue", "svelte", "astro",
    # Data / config
    "json", "jsonc", "json5",
    "yaml", "yml", "toml", "ini", "cfg", "conf",
    "xml", "xsl", "xslt", "svg", "plist",
    # Images
    "png", "jpg", "jpeg", "gif", "webp", "bmp", "tiff", "tif", "heic", "heif", "ico",
    "env", "properties",
    "csv", "tsv",
    # Python
    "py", "pyi", "pyx", "pxd",
    # C / C++
    "c", "h", "cc", "cpp", "cxx", "hpp", "hxx", "hh", "m", "mm",
    # C# / .NET
    "cs", "csx", "fs", "fsx", "fsi", "vb",
    # Java / JVM
    "java", "kt", "kts", "groovy", "gradle", "scala", "clj", "cljs",
    # Go
    "go", "mod", "sum",
    # Rust
    "rs",
    # Swift / Objective-C
    "swift",
    # Ruby
    "rb", "erb", "rake", "gemspec",
    # PHP
    "php", "blade",
    # Shell / scripting
    "sh", "bash", "zsh", "fish", "ps1", "psm1", "bat", "cmd",
    # Perl
    "pl", "pm",
    # Lua
    "lua",
    # R
    "r", "rmd",
    # Elixir / Erlang
    "ex", "exs", "erl", "hrl",
    # Haskell
    "hs", "lhs",
    # OCaml / F#
    "ml", "mli",
    # Dart / Flutter
    "dart",
    # Zig / Nim / V
    "zig", "nim", "v",
    # SQL / database
    "sql", "prisma", "graphql", "gql",
    # Docker / CI / infra
    "tf", "hcl",
    # Protobuf / schemas
    "proto", "thrift", "avsc",
    # Assembly
    "asm", "s",
    # Misc
    "cmake", "mk", "mak",
    "patch", "diff",
    "lock",
}
ALLOWED_EXTS_SET = ALLOWED_EXTS  # alias to avoid accidental mutation
