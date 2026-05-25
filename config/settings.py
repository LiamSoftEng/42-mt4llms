import os
from dotenv import load_dotenv

load_dotenv()

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HUGGINGFACE_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

# API Keys
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
GROK_API_KEY = os.getenv("GROK_API_KEY")

# Model names (TODO: change please, based on knowledge cutoffs + call costs)
CLAUDE_MODEL = "claude-sonnet-4-6"
OPENAI_MODEL = "gpt-5-nano"
GEMINI_MODEL = "gemini-2.5-flash"
DEEPSEEK_MODEL = "minimaxai/minimax-m2.7"
CODE_EMBEDDING_MODEL = "Salesforce/codet5p-220m"
CODE_TOKENIZER_MODEL = "Salesforce/codet5-base"

# Groq model options (uncomment one)
# GROK_MODEL = "openai/gpt-oss-20b"
GROK_MODEL = "openai/gpt-oss-120b"
# GROK_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
# GROK_MODEL = "llama-3.3-70b-versatile"
# GROK_MODEL = "llama-3.1-8b-instant"
# GROK_MODEL = "groq/compound"
# GROK_MODEL = "groq/compound-mini"
# GROK_MODEL = "allam-2-7b"
# GROK_MODEL = "meta-llama/llama-prompt-guard-2-22m"
# GROK_MODEL = "meta-llama/llama-prompt-guard-2-86m"
# GROK_MODEL = "openai/gpt-oss-safeguard-20b"
# GROK_MODEL = "qwen/qwen3-32b"

# Generation params (TODO: adjust as needed)
MAX_TOKENS = 4096
TEMPERATURE = 0.2  # Deterministic outputs for reproducibility

# Dataset
LCB_DATASET = "bzantium/livecodebench"
LCB_VERSION = "release_latest"
N_SAMPLES = None  # Set to an int (e.g. 10) to limit samples during dev

# Output
RESULTS_DIR = "results"
COST_TRACKING_PATH = os.getenv("COST_TRACKING_PATH", os.path.join(RESULTS_DIR, ".model_costs.json"))

# Cost tracking (USD per 1M tokens). Update these with current provider rates.
COST_RATES_PER_MILLION = {
    CLAUDE_MODEL: {"prompt": 0.0, "completion": 0.0},
    OPENAI_MODEL: {"prompt": 0.05, "completion": 0.40},
    GEMINI_MODEL: {"prompt": 0.0, "completion": 0.0},
    DEEPSEEK_MODEL: {"prompt": 0.0, "completion": 0.0},
    GROK_MODEL: {"prompt": 0.05, "completion": 0.40},
}