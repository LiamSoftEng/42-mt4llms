# LLM MT Pipeline
Run LiveCodeBench problems through multiple LLMs, capture source and followup outputs, and evaluate with metamorphic relations and test harnesses.
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
│   ├── deepseek_model.py
│   └── grok_model.py
├── evaluation/
│   ├── runner.py
│   ├── executor.py
│   ├── metrics.py
│   └── harness/
│       ├── stdin.py
│       └── functional.py
├── metamorphic/
│   ├── __init__.py
│   ├── relations_list.py
│   └── relations/
│       ├── __init__.py
│       ├── base_mr.py
│       └── mrs.py
├── explore_dataset.py
├── main.py
├── runtests.py
└── results/
  ├── <model>/
  │   ├── base.json
  │   ├── base.tests.log
  │   ├── mr-04-persona_role_prompting.json
  │   ├── mr-04-persona_role_prompting.tests.log
  │   ├── all_mrs.json
  │   └── all_mrs.tests.log
  └── dataset_preview.json
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
python main.py --models grok --after 2025-01-01 --n 10
# Combine filters and run tests
python main.py --models grok --after 2025-01-01 --n 10 --run-tests
# Multiple models
python main.py --models claude openai --n 5
# All models, full dataset
python main.py
```
| Flag | Description | Example |
|---|---|---|
| `--models` | One or more of: `claude` `openai` `gemini` `deepseek` `grok` | `--models deepseek claude` |
| `--n` | Limit number of samples | `--n 10` |
| `--after` | Only problems on or after this date | `--after 2024-01-01` |
| `--mr` | Apply a metamorphic relation alongside the normal run | `--mr 4` |
| `--mr-only` | Skip base LLM call and generate followup only (requires `--mr`) | `--mr 4 --mr-only` |

**Run with a metamorphic relation**
```powershell
# Run with MR0 (control - prompt unchanged, use to verify pipeline)
python main.py --models deepseek --n 5 --mr 0

# Run with MR4 (persona role prompting)
python main.py --models deepseek --n 5 --mr 4

# Run MR4 without base generation
python main.py --models deepseek --n 5 --mr 4 --mr-only
```

**Test the responses**
```powershell
python runtests.py --results results\google_gemma-2-2b-it\base.json
python runtests.py --results results\google_gemma-2-2b-it\base.json --n 5
python runtests.py --results results\google_gemma-2-2b-it\base.json --stdout
python runtests.py --results results\google_gemma-2-2b-it\base.json --debug
python runtests.py --results results\google_gemma-2-2b-it\mr-04-persona_role_prompting.json --followup-only
```
Replace `results\<model>\<file>.json` with your file.

---
## Output
Each `main.py` run saves a JSON file under `results\<model>\`.
Base runs use `base.json`; MR runs use `mr-<id>-<name>.json`.
```json
{
  "question_id": "1873_A",
  "question_title": "A. Short Sort",
  "difficulty": "easy",
  "platform": "codeforces",
  "model_name": "google/gemma-2-2b-it",
  "instruction_prefix": "...",
  "instruction_suffix": "...",
  "question": "...",
  "response": "...",
  "public_test_cases": "...",
  "private_test_cases": "...",
  "metadata": "...",
  "error": "",
  "relation": "persona_role_prompting",
  "followup_instruction_prefix": "...",
  "followup_instruction_suffix": "...",
  "followup_question": "...",
  "followup_response": "...",
  "followup_error": ""
}
```
Followup fields are only populated when `--mr` is used. If a model call fails, `error` will contain the message and `response` will be empty; the pipeline continues to the next sample.

`runtests.py` prints per-test verdicts, MR holds/violations, and a metrics summary table using NCLOC and cyclomatic complexity (radon):
```
  [source]   Test 1 [functional]: PASS
  [followup] Test 1 [functional]: PASS
  MR [dead_context]: HOLDS

  ID               Src NCLOC   Src CC  Flw NCLOC   Flw CC   dNCLOC      dCC       MR
  1873_A                 14     1.00         13     1.00       -1     +0.00    HOLDS
```

---
## Metamorphic Relations
Relations are defined in `metamorphic/relations/mrs.py` and registered in `metamorphic/relations_list.py`.

Each MR class implements:
- `name` - identifier string used in results
- `transform(...)` - returns transformed question or instruction
- `verify(source_passed, followup_passed)` - returns whether the MR holds

To add a new MR, implement it in `metamorphic/relations/mrs.py` and register it in `metamorphic/relations_list.py`.

| ID | Name | Target | Description |
|---|---|---|---|
| 0 | `do_nothing` | question | Control relation that returns the original prompt unchanged. |
| 1 | `dead_context` | question | Insert an irrelevant sentence into the prompt. |
| 2 | `constraint_redundant_spec` | instruction | Inject a redundant language constraint already implied by syntax. |
| 3 | `format_structural_equivalence` | question | Convert the prompt into a structured bulleted list. |
| 4 | `persona_role_prompting` | instruction | Prepend a persona to the prompt. |
| 5 | `context_extraneous_narrative` | question | Inject an irrelevant backstory without changing requirements. |
| 6 | `polarity_negative_formulation` | instruction | Rewrite a return constraint into an equivalent negative form. |
| 7 | `identifier_explicit_naming` | instruction | Add explicit internal naming constraints for variables. |
| 8 | `output_formatting_constraint` | instruction | Add constraints about outputting raw code only. |
| 9 | `hint_trivial_algorithm` | instruction | Add a trivial algorithmic suggestion. |
| 10 | `order_requirement_permutation` | question | Permute the order of independent constraints. |
| 11 | `verbosity_lexical_expansion` | question | Add brief definitions for common programming terms. |

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