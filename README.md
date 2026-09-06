# LLM Metamorphic Testing Pipeline

This project runs LiveCodeBench problems through supported LLMs, applies optional metamorphic relations (MRs), executes the generated Python code against test cases, and measures code variability across repeated generations.

## Quick Start

### 1. Set up the environment

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Add the API keys for the providers you intend to use to `.env`. Model names, generation settings, embedding models, result paths, and cost rates are configured in `config/settings.py`.

### 2. Check the dataset

```powershell
python explore_dataset.py
```

The LiveCodeBench dataset is downloaded and cached on first use. Add
`--write-preview` only when you want to save the first ten records under the
configured results directory.

### 3. Run a small experiment

```powershell
python main.py --models openai --n 5 --mr 1 --repeats 3 --run-tests
```

This command:

1. Loads five problems.
2. Generates each original prompt three times.
3. Applies MR1 and generates each transformed prompt three times.
4. Saves generation results after every problem.
5. Runs the generated code against the available tests.
6. Writes metrics and an HTML report.

## Python Run Configuration

For repeatable experiments, edit `CONFIG` in `config/run_config.py`:

```python
CONFIG = {
    "models": ["openai"],
    "n": 5,
    "after": "2025-04-01",
    "difficulty": ["easy", "medium", "hard"],  # Five questions from each difficulty.
    "mr": [1, 4, 7],  # Or "all" to run every registered MR.
    "mr_only": False,
    "run_tests": True,
    "repeats": 3,
    "resume": True,
}
```

Run it directly:

```powershell
python config\run_config.py
```

Use `None`, `""`, or `[]` to omit a value. Boolean flags are included only when set to `True`. A key can also be commented out.

## Command-Line Usage

```powershell
python main.py [options]
```

| Option | Description |
|---|---|
| `--models MODEL ...` | Models to run: `claude`, `openai`, or `gemini`. Defaults to all. |
| `--n N` | Limit the number of dataset problems. |
| `--after YYYY-MM-DD` | Include problems on or after the specified date. |
| `--difficulty LEVEL ...` | Run each listed difficulty separately; `--n` applies to each difficulty. |
| `--mr ID ...` | Apply one or more relation IDs, or use `--mr all`. |
| `--mr-only` | Reuse saved base responses and generate only MR follow-up responses. Requires `--mr`. |
| `--repeats N` | Generate each applicable prompt N times. Defaults to `1`. |
| `--resume` | Skip existing entries that already contain the required code blocks. |
| `--run-tests` | Evaluate results after generation and write split outputs plus an HTML report. |

### Common examples

Run a base experiment:

```powershell
python main.py --models openai --n 5
```

Run an MR experiment:

```powershell
python main.py --models openai --n 10 --mr 4
```

Run several or all registered MRs:

```powershell
python main.py --models openai --n 10 --mr 3 4 5
python main.py --models openai --n 10 --mr all
```

Measure source and MR prompt variability:

```powershell
python main.py --models openai --n 10 --mr 4 --repeats 5 --run-tests
```

Generate only follow-up responses using previously saved base results:

```powershell
python main.py --models openai --n 10 --mr 4 --mr-only
```

Resume an interrupted experiment:

```powershell
python main.py --models openai --n 50 --mr 4 --repeats 5 --resume
```

## Repeats and LLM Calls

The number of calls depends on the number of problems, prompts, repeats, and models.

```text
Base run:
calls = problems x repeats x models

MR run:
calls = problems x repeats x 2 prompts x models

MR-only run:
calls = problems x repeats x models
```

For example, five problems with one model, MR4, and three repeats produce:

```text
5 problems x 2 prompts x 3 repeats = 30 LLM calls
```

`TEMPERATURE` is currently configured in `config/settings.py`. A value of `0` reduces randomness, although provider output may still vary.

## Resume Behavior

Results are saved after each completed problem. If a run stops, add `--resume` to continue from the existing result file.

An entry is skipped only when the required response contains an extractable `python` code block:

| Run type | Required code block |
|---|---|
| Base | Source response |
| MR | Source and follow-up responses |
| MR-only | Follow-up response |

Entries without the required code are generated again. The new entry replaces the incomplete entry, preventing duplicate question IDs.

## Testing Existing Results

Use `runtests.py` when generation has already completed:

```powershell
python runtests.py --results results\<model>\<result-file>.json --stdout
```

Available options:

| Option | Description |
|---|---|
| `--results PATH` | Input result JSON file. Required. |
| `--n N` | Test only the first N entries. |
| `--stdout` | Print output while also writing the test log. |
| `--debug` | Print extracted code and generated execution scripts. |
| `--out PATH` | Set the test log path. |
| `--csv PATH` | Export the metrics table to CSV. |
| `--followup-only` | Test the follow-up response as the primary response. |
| `--write-outputs` | Write split content/results files and generate the HTML report. |

Examples:

```powershell
python runtests.py --results results\gpt-5-nano\base\base.json --stdout

python runtests.py `
  --results results\gpt-5-nano\mr-04-persona_role_prompting.json `
  --csv results\gpt-5-nano\mr-04.metrics.csv `
  --write-outputs `
  --stdout
```

`main.py --run-tests` automatically invokes `runtests.py` with `--write-outputs`.

## Repeat Variability Testing

### Generate repeated samples

Add `--repeats N` to generate each prompt N times. Each repeat is saved as a separate file under the base results folder:

```powershell
python main.py --models openai --n 2  --repeats 3  --difficulty easy
```

This generates:

```text
results/<model>/base/
|-- base.json
`-- repeats/
    |-- repeat-01.json
    |-- repeat-02.json
    `-- repeat-03.json
```

Every `repeat-NN.json` contains the same set of problems; `response_samples` holds that run's independently generated code sample for each problem.

### Test and compare the repeats

`compare_repeats.py` re-runs the test suite against every repeat file, then compares pass rates and code metrics across the runs:

```powershell
python compare_repeats.py --model gpt-5-nano
```

or point at a repeats folder directly:

```powershell
python compare_repeats.py --dir results\gpt-5-nano\base\repeats
```

Options:

| Option | Description |
|---|---|
| `--model NAME` | Look under `results/<model>/base/repeats`. |
| `--dir PATH` | Explicit path to a repeats folder. |
| `--csv PATH` | Where to write the comparison CSV (default: `comparison.csv` in the repeats folder). |
| `--html PATH` | Where to write the HTML comparison report (default: `comparison.html` in the repeats folder). |

For each repeat file it writes a per-run test log (`repeat-NN.tests.log`) and metrics CSV (`repeat-NN.metrics.csv`), then produces two comparison artifacts in the repeats folder:

- `comparison.csv` — one row per repeat: pass rate, extractable rate, and mean NCLOC / CC / tokens / LLM call time.
- `comparison.html` — stat boxes, a per-run summary table, and a per-question breakdown table with each question's PASS/FAIL badge per repeat and mean metrics.

The console output also reports the mean/std pass rate across repeats and how many questions flipped between pass and fail.

## Variability Metrics

Repeated code samples are compared using CodeT5 embeddings and cosine similarity.

```text
distance = 1 - cosine similarity
```

| Metric | Meaning |
|---|---|
| `CodeSim` | Similarity between the first source and first follow-up code samples. Higher means more similar. |
| `SrcDist` | Mean pairwise distance among repeated source generations. |
| `SrcStd` | Standard deviation of source pairwise distances. |
| `FlwDist` | Mean pairwise distance among repeated follow-up generations. |
| `FlwStd` | Standard deviation of follow-up pairwise distances. |
| `Var>Src` | `YES` when `FlwDist` is greater than `SrcDist`. |
| `NCLOC` | Non-comment logical lines of code. |
| `CC` | Average cyclomatic complexity. |

Interpretation:

- Higher mean distance means the prompt produced more varied code on average.
- Higher standard deviation means the amount of variation was less consistent.
- With two generated samples there is only one pairwise distance, so `StdDist` is `0.0`. Use at least three repeats for a meaningful standard deviation.
- Missing or unextractable code blocks are excluded from variability calculations.

## Result Files

Raw generation results are saved under:

```text
results/<model>/<difficulty>/base/base.json
results/<model>/<difficulty>/mr-<id>-<relation>/mr-<id>-<relation>.json
```

Supplying multiple difficulties runs them back to back. For example,
`--n 20 --difficulty easy medium hard` runs 20 questions at each difficulty
(60 questions per MR in total) and keeps their artifacts separate. If
`--difficulty` is omitted, the legacy unscoped `results/<model>/...` layout is
used.

The model directory uses the configured model name from `config/settings.py`, sanitized for use as a folder name. For example, the `openai` CLI alias currently writes under `results/gpt-5-nano/`.

When `--write-outputs` is used, additional files are created:

```text
results/<model>/<difficulty>/
|-- base/
|   |-- base.json
|   |-- base.html
|   `-- base.tests.log
|   `-- base.results.json
|   `-- repeats/              Present only when --repeats > 1
|       |-- repeat-01.json
|       |-- repeat-01.tests.log
|       |-- repeat-01.metrics.csv
|       |-- comparison.csv
|       `-- comparison.html
|-- mr-<id>-<relation>/
|   |-- mr-<id>-<relation>.json
|   |-- followup.json
|   |-- followup.results.json
|   |-- mr-<id>-<relation>.html
|   `-- mr-<id>-<relation>.tests.log
|-- all_mrs/
|   |-- all_mrs.json
|   `-- all_mrs.tests.log
```

Test logs default to the input result filename with `.tests.log` appended. CSV output is written only when `--csv` is supplied.

## Metamorphic Relations

| ID | Name | Target | Transformation |
|---:|---|---|---|
| 0 | `do_nothing` | Prompt description | Returns the original prompt unchanged as a control. |
| 1 | `irrelevant_context` | Prompt description | Inserts an irrelevant sentence into the prompt. |
| 2 | `constraint_redundant_spec` | Instruction | Inserts a related sentence that does not affect the solution. |
| 3 | `format_structural_equivalence` | Prompt description | Converts the prompt into a structured bulleted list. |
| 4 | `persona_role_prompting` | Instruction | Prepends a persona to the prompt. |
| 5 | `polarity_negative_formulation` | Prompt description | Rewrites the output requirement into an equivalent negative form. |
| 6 | `identifier_explicit_naming` | Instruction | Adds explicit internal variable-naming constraints. |
| 7 | `order_requirement_permutation` | Prompt description | Reorders independent constraints while preserving meaning. |
| 8 | `remove_sentences` | Prompt description | Removes exactly one sentence. |
| 9 | `antonym_substitution` | Prompt description | Substitutes key terms with antonyms and adjusts surrounding logic. |
| 10 | `synonym_substitution` | Prompt description | Substitutes key terms with synonyms while preserving meaning. |
| 11 | `loop_while_only` | Instruction | Requires while loops and forbids for loops. |
| 12 | `full_commenting` | Instruction | Requires an inline comment on every non-blank solution line. |
| 13 | `dummy_variable` | Instruction | Requires unused dummy variables at the start of the solution. |
| 14 | `guard_clause_conditionals` | Instruction | Requires guard clauses instead of nested if-else logic. |
| 15 | `use_recursion` | Instruction | Requires recursion and forbids iterative loops. |
| 16 | `no_builtin_libraries` | Instruction | Forbids Python standard-library modules. |
| 17 | `mandated_library` | Prompt description | Requires a helper-selected standard-library module. |
| 18 | `scope_input_narrowing` | Prompt description | Adds a generic guarantee about the input domain. |
| 19 | `round_trip_paraphrase` | Prompt description | Replaces the problem with a helper-generated paraphrase. |

Relations are implemented in `metamorphic/relations/mrs.py` and registered in `metamorphic/relations_list.py`.

To add a relation:

1. Add a class derived from `BaseMR`.
2. Set its `target` to `PD` (prompt description) or `I` (instruction).
3. Implement `name` and `transform(...)`.
4. Register it in `metamorphic/relations_list.py`.

## Project Structure

```text
config/
  run_config.py          Runnable experiment configuration
  settings.py            Models, generation settings, paths, and costs
data/
  loader.py              LiveCodeBench loading and prompt construction
  decoder.py             Private test-case decoding
evaluation/
  runner.py              Generation pipeline and partial-result saving
  executor.py            Code extraction and test dispatch
  metrics.py             Complexity, similarity, and variability metrics
  results_writer.py      Split JSON output and report orchestration
  report.py              HTML report generation
  harness/               Functional and stdin execution harnesses
llms/                    Provider-specific model clients
metamorphic/             MR registry and implementations
results/                 Generated responses, metrics, logs, and reports
main.py                  Main generation CLI
runtests.py              Evaluation and export CLI
compare_repeats.py       Tests and compares per-repeat runs
run_all_mrs.py           Runs all registered MRs
```

## Notes

- Start with `--n 5` before running large experiments.
- Private tests are decoded from the LiveCodeBench data at evaluation time.
- The embedding model is downloaded on first variability analysis.
- Update cost rates in `config/settings.py` when provider pricing changes.
