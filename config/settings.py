import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# Model names
CLAUDE_MODEL = "claude-sonnet-4-6"
OPENAI_MODEL = "gpt-4o"
GEMINI_MODEL = "gemini-2.0-flash"
DEEPSEEK_MODEL = "google/gemma-2-2b-it"

# Generation params
MAX_TOKENS = 1024
TEMPERATURE = 0.2  # Deterministic outputs for reproducibility

# Dataset
LCB_DATASET = "bzantium/livecodebench"
LCB_VERSION = "release_v2"
N_SAMPLES = None  # Set to an int (e.g. 10) to limit samples during dev

# Output
RESULTS_DIR = "results"