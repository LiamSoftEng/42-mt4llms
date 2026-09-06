import os
from dotenv import load_dotenv

load_dotenv()

HUGGINGFACE_ENV_DEFAULTS = {
    "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    "HUGGINGFACE_HUB_DISABLE_PROGRESS_BARS": "1",
    "TRANSFORMERS_VERBOSITY": "error",
    "TRANSFORMERS_NO_ADVISORY_WARNINGS": "1",
}


def configure_huggingface_environment() -> None:
    for name, value in HUGGINGFACE_ENV_DEFAULTS.items():
        os.environ.setdefault(name, value)


configure_huggingface_environment()

# API Keys
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Model names
MR_HELPER_MODEL = "gpt-5.6-luna"
CLAUDE_MODEL = "claude-sonnet-4-6"
OPENAI_MODEL = "gpt-5.6-sol"
GEMINI_MODEL = "gemini-2.5-flash"
CODE_EMBEDDING_MODEL = "Salesforce/codet5p-220m"
CODE_TOKENIZER_MODEL = "Salesforce/codet5-base"

# Generation parameters
MAX_TOKENS = 8192
TEMPERATURE = 0  # Deterministic outputs for reproducibility

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
    OPENAI_MODEL: {"prompt": 5, "completion": 30},
    GEMINI_MODEL: {"prompt": 0.0, "completion": 0.0},
}

# ---------------------------------------------------------------------------
# Metamorphic-relation violation detection (per-question t-test baseline)
# ---------------------------------------------------------------------------
# Paths to repeat-only (no-transformation) run folders, used as a
# per-question baseline distribution for t-test-based violation detection.
# Each entry is a directory containing repeat-*.json files (same format
# PipelineRunner writes under base/repeats/); every file contributes one
# independent sample per question. Directories may cover disjoint question
# sets (here, split by difficulty) — merged by question_id when building
# the baseline. See evaluation.metrics.load_repeat_baseline.
REPEAT_BASELINE_DIRS = [
    "results/gpt-5-6-sol/base/repeats1_hard/repeats",
    "results/gpt-5-6-sol/base/repeats2_medium/repeats",
    "results/gpt-5-6-sol/base/repeats3_easy/repeats",
]

# Significance level for the one-sample t-test in
# evaluation.metrics.compute_ttest_violation. A followup metric value is
# flagged as a violation when its p-value is below this.
TTEST_ALPHA = 0.05