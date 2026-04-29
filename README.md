# LLM MT Pipeline

Run LiveCodeBench problems through multiple LLMs and collect their outputs for metamorphic testing.

---

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env          # then fill in your API keys
```

---

## File Structure

```
llm_mt_pipeline/
├── config/settings.py        # model names, dataset config
├── data/loader.py            # loads LiveCodeBench from HuggingFace
├── models/
│   ├── base.py               # abstract LLM interface
│   ├── claude_model.py
│   ├── openai_model.py
│   ├── gemini_model.py
│   └── deepseek_model.py     # uses NVIDIA API
├── evaluation/runner.py      # sends prompts to models, saves results
├── explore_dataset.py        # inspect dataset fields (no API calls)
├── main.py                   # entry point
├── run_tests.py              # execute model responses against test cases
└── results/                  # output JSON files (one per model)
```

---

## Usage

**Explore the dataset**
```bash
python explore_dataset.py
```

**Run a model**
```bash
python main.py --models deepseek --n 5     # 5 samples, DeepSeek only
python main.py --models claude openai      # two models, full dataset
python main.py                             # all models, full dataset
```

Available models: `claude` `openai` `gemini` `deepseek`

**Test the responses**
```bash
python run_tests.py --results results/google_gemma-2-2b-it.json
python run_tests.py --results results/google_gemma-2-2b-it.json --n 5
```

---

## Output

Each run saves a JSON file to `results/` with one entry per problem:

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

---

## Notes

- `results/` and `.env` are gitignored — never committed
- `run_tests.py` currently supports `stdin`-type problems (Codeforces). LeetCode functional-type problems are skipped
- Start with `--n 5` to verify everything works before running the full dataset