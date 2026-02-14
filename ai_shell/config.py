import os

# Model/runtime knobs
MODEL_NAME = os.getenv("SC2_MODEL", "bigcode/starcoder2-3b")
# Default output budget for model replies
MAX_NEW = int(os.getenv("SC2_MAX_NEW", "4096"))
PLAN_MAX_NEW = int(os.getenv("SC2_PLAN_MAX_NEW", "4096"))
DIFF_MAX_NEW = int(os.getenv("SC2_DIFF_MAX_NEW", "4096"))
CHAT_MAX_NEW = int(os.getenv("SC2_CHAT_MAX_NEW", "768"))
CHAT_SHORT_MAX_NEW = int(os.getenv("SC2_CHAT_SHORT_MAX_NEW", "128"))
CHAT_SHORT_INPUT_CHARS = int(os.getenv("SC2_CHAT_SHORT_INPUT_CHARS", "24"))
TEMPERATURE = float(os.getenv("SC2_TEMP", "0.25"))
TOP_P = float(os.getenv("SC2_TOP_P", "0.9"))
DEBUG = os.getenv("SC2_DEBUG", "").lower() in ("1", "true", "yes")
DEBUG_LOG_PATH = os.getenv(
    "SC2_DEBUG_LOG", "/Users/michael/moonlet/runtime-debug.log"
)
# SC2_DEBUG_DUMP_VERBOSE=1: write full model output to debug log (no truncation).
# Default: true (full dump). Set to false for truncated output.
DEBUG_DUMP_VERBOSE = os.getenv("SC2_DEBUG_DUMP_VERBOSE", "true").lower() in ("1", "true", "yes")
DEBUG_DUMP_MAX_LINES = int(os.getenv("SC2_DEBUG_DUMP_MAX_LINES", "20"))
DEBUG_DUMP_MAX_CHARS = int(os.getenv("SC2_DEBUG_DUMP_MAX_CHARS", "2000"))
GEN_TIMEOUT = int(os.getenv("SC2_GEN_TIMEOUT", "120"))
MAX_MODEL_CALLS = int(os.getenv("SC2_MAX_MODEL_CALLS", "2"))
REP_WINDOW = int(os.getenv("SC2_REP_WINDOW", "40"))
REP_MIN_CHARS = int(os.getenv("SC2_REP_MIN_CHARS", "200"))
REP_MIN_TOKENS = int(os.getenv("SC2_REP_MIN_TOKENS", "120"))
USE_RG_HINT_DEFAULT = os.getenv("SC2_USE_RG_HINT_DEFAULT", "true").lower() != "false"
SOFT_DIFF_POSTCHECK = os.getenv("SC2_SOFT_DIFF_POSTCHECK", "false").lower() != "false"
SKIP_COMPILE_CHECKS = os.getenv("SC2_SKIP_COMPILE_CHECKS", "false").lower() != "false"
RELAX_SEMANTIC_CHECKS = os.getenv("SC2_RELAX_SEMANTIC_CHECKS", "false").lower() != "false"
REVERT_ON_FAILURE = os.getenv("SC2_REVERT_ON_FAILURE", "false").lower() in ("1", "true", "yes")
# If true, do not write model edits to disk automatically.
# Candidate content is returned for UI Accept/Reject flow.
STAGE_EDITS = os.getenv("SC2_STAGE_EDITS", "true").lower() in ("1", "true", "yes")
# Analysis symbol allowlist can over-constrain edits; keep off by default.
ENFORCE_ANALYSIS_ALLOWLIST = os.getenv("SC2_ENFORCE_ANALYSIS_ALLOWLIST", "false").lower() in ("1", "true", "yes")

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
LLAMA_SERVER_CACHE_SLOTS = int(
    os.getenv("SC2_LLAMA_SERVER_CACHE_SLOTS", "32")
)
# Optional GGUF path for local quantized model (llama.cpp backend)
GGUF_PATH = os.getenv("SC2_GGUF")
GGUF_CTX = int(os.getenv("SC2_CTX_TOK", "4096"))
GGUF_THREADS = int(os.getenv("SC2_THREADS", str(os.cpu_count() or 4)))
GGUF_GPU_LAYERS = int(os.getenv("SC2_GPU_LAYERS", "-1"))
LLAMA_SERVER_ENABLED = os.getenv("SC2_USE_LLAMA_SERVER", "true").lower() in ("1", "true", "yes")
LLAMA_SERVER_HOST = os.getenv("SC2_LLAMA_SERVER_HOST", "127.0.0.1")
LLAMA_SERVER_PORT = int(os.getenv("SC2_LLAMA_SERVER_PORT", "8012"))
LLAMA_SERVER_BIN = os.getenv("SC2_LLAMA_SERVER_BIN", "")
LLAMA_SERVER_START_TIMEOUT = int(os.getenv("SC2_LLAMA_SERVER_START_TIMEOUT", "90"))


# Agent/runtime knobs
IGNORE_DIRS = {".git", "node_modules", "__pycache__"}
MAX_CTX_BYTES = int(os.getenv("SC2_CTX_BYTES", "4000"))
MAX_LIST_FILES = int(os.getenv("SC2_MAX_FILES", "40"))
SERVER_PORT = int(os.getenv("SC2_PORT", "8000"))
BUFFER_OUTPUT = os.getenv("SC2_BUFFER", "1") not in {"0", "false", "False"}
CHAT_BUFFER_PROMPT_CHARS = int(os.getenv("SC2_CHAT_BUFFER_PROMPT_CHARS", "3200"))
CHAT_BUFFER_INPUT_CHARS = int(os.getenv("SC2_CHAT_BUFFER_INPUT_CHARS", "400"))
STATE_PATH = os.getenv("SC2_STATE_PATH", "~/.moonlet_state.json")
DISABLE_HISTORY = os.getenv("SC2_DISABLE_HISTORY", "false").lower() in ("1", "true", "yes")
MAX_BUFFER_CHARS = int(os.getenv("SC2_MAX_BUFFER_CHARS", "200000"))
CTX_CHAR_BUDGET = int(
    os.getenv(
        "SC2_CTX_CHAR_BUDGET",
        str((GGUF_CTX if GGUF_CTX else 2048) * 3),
    )
)
# Multi-file agent limits
MAX_PLAN_FILES = int(os.getenv("SC2_MAX_PLAN_FILES", "5"))
MAX_MULTI_MODEL_CALLS = int(os.getenv("SC2_MAX_MULTI_CALLS", "8"))

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
