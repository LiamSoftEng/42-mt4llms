# LLM MT Pipeline

Run LiveCodeBench problems through multiple LLMs and collect their outputs for metamorphic testing.

---

## Setup

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```
Then open `.env` and fill in your keys.

---

## File Structure

```
llm_mt_pipeline/
├── config/settings.py              # model names, dataset config
├── data/loader.py                  # loads LiveCodeBench from HuggingFace
├── models/
│   ├── base.py                     # abstract LLM interface
│   ├── __init__.py
│   ├── claude_model.py
│   ├── openai_model.py
│   ├── gemini_model.py
│   └── deepseek_model.py           # uses NVIDIA API
├── evaluation/
│   ├── runner.py                   # sends prompts to models, saves results
│   ├── executor.py                 # decodes test cases, dispatches to harness
│   └── harness/
│       ├── stdin.py                # runs Codeforces-style problems via subprocess
│       └── functional.py          # runs LeetCode-style problems via function call
├── explore_dataset.py              # inspect dataset fields (no API calls)
├── main.py                         # entry point for generating responses
├── runtests.py                    # entry point for testing responses
└── results/                        # output JSON files (one per model, gitignored)
```

---

## Usage

**Explore the dataset (no API calls, run this first, downloads/caches the data)**
```powershell
python explore_dataset.py
```

**Run a model**
```powershell
# 5 samples, DeepSeek only
python main.py --models deepseek --n 5

# Only problems from 2024 onwards (dates in yyyy-mm-dd)
python main.py --models deepseek --after 2024-01-01

# Combine filters
python main.py --models deepseek --after 2024-01-01 --n 10

# Multiple models
python main.py --models claude openai --n 5

# All models, full dataset
python main.py
```

| Flag | Description | Example |
|---|---|---|
| `--models` | One or more of: `claude` `openai` `gemini` `deepseek` | `--models deepseek claude` |
| `--n` | Limit number of samples | `--n 10` |
| `--after` | Only problems on or after this date | `--after 2024-01-01` |

**Test the responses**
```powershell
python run_tests.py --results results\google_gemma-2-2b-it.json
python run_tests.py --results results\google_gemma-2-2b-it.json --n 5
```
replace the results\<yourresults.json>
---

## Output

Each run saves a JSON file to `results\` named after the model:

```json
{
  "question_id": "1873_A",
  "question_title": "A. Short Sort",
  "difficulty": "easy",
  "platform": "codeforces",
  "model_name": "google/gemma-2-2b-it",
  "prompt": "...",
  "response": "```python\n...```",
  "error": ""
}
```

If a model call fails, `error` will contain the message and `response` will be empty. The pipeline continues to the next sample rather than crashing.

---

## Notes

- The dataset downloads once (~2GB) and is cached at `C:\Users\<you>\.cache\huggingface\`
- Testing only uses public test cases, need to do the decoding of private tests
- Start with `--n 5` to verify everything works before running the full dataset
- Re-activate the virtual environment each session with `venv\Scripts\activate`

## Cutoff date samples

- After Feb2025: 131
- After March2025: 80
- After April2025: 12
- After May2025: 0