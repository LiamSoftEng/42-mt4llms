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
42-mt4llms/
├── config/settings.py
├── data/
│   ├── loader.py
│   └── decoder.py
├── llms/
│   ├── base.py
│   ├── __init__.py
│   ├── claude_model.py
│   ├── openai_model.py
│   ├── gemini_model.py
│   └── deepseek_model.py
├── evaluation/
│   ├── runner.py
│   ├── executor.py
│   └── harness/
│       ├── stdin.py
│       └── functional.py
├── metamorphic/
│   ├── __init__.py
│   ├── relations_list.py
│   └── relations/
│       ├── __init__.py
│       ├── mr0.py
│       └── mr1.py
├── explore_dataset.py
├── main.py
├── runtests.py
└── results/
    └── <model>.json
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
| `--mr` | Apply a metamorphic relation alongside the normal run | `--mr 1` |

**Run with a metamorphic relation**
```powershell
# Run with MR0 (control — prompt unchanged, use to verify pipeline)
python main.py --models deepseek --n 5 --mr 0

# Run with MR1 (dead context)
python main.py --models deepseek --n 5 --mr 1
```

**Test the responses**
```powershell
python runtests.py --results results\google_gemma-2-2b-it.json
python runtests.py --results results\google_gemma-2-2b-it.json --n 5
```
replace the results\<yourresults.json>

---
## Output
Each `main.py` run saves a JSON file to `results\` named after the model:
```json
{
  "question_id": "1873_A",
  "question_title": "A. Short Sort",
  "difficulty": "easy",
  "platform": "codeforces",
  "model_name": "google/gemma-2-2b-it",
  "prompt": "...",
  "response": "...",
  "public_test_cases": "...",
  "private_test_cases": "...",
  "metadata": "...",
  "error": "",
  "relation": "dead_context",
  "followup_prompt": "...",
  "followup_response": "...",
  "followup_error": ""
}
```
`relation`, `followup_prompt`, `followup_response`, and `followup_error` are only populated when `--mr` is used.

If a model call fails, `error` will contain the message and `response` will be empty. The pipeline continues to the next sample rather than crashing.

`runtests.py` automatically detects the presence of followup responses and computes MR verdicts:
```
  [source]   Test 1 [functional]: PASS
  [followup] Test 1 [functional]: PASS
  MR [dead_context]: HOLDS
```

---
## Metamorphic Relations
Relations are defined in `metamorphic/relations/` and registered in `metamorphic/registry.py`.

Each MR class implements:
- `name` — identifier string used in results
- `transform(prompt)` — takes original prompt, returns transformed followup prompt
- `verify(source_passed, followup_passed)` — returns whether the MR holds

To add a new MR, create `metamorphic/relations/mrN.py` and add an entry to `registry.py`.

| MR | Name | Description |
|---|---|---|
| 0 | `do_nothing` | Control — returns prompt unchanged. Should always hold. |
| 1 | `dead_context` | Insert irrelevant sentence into prompt. Output should be identical. |

---
## Notes
- The dataset downloads once (~2GB) and is cached at `C:\Users\<you>\.cache\huggingface\`
- Testing uses private test cases decoded from base64+zlib+pickle format
- Start with `--n 5` to verify everything works before running the full dataset
- Re-activate the virtual environment each session with `venv\Scripts\activate`

## Cutoff date samples
- After Feb 2025: 131
- After Mar 2025: 80
- After Apr 2025: 12
- After May 2025: 0