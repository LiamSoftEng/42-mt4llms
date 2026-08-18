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

The LiveCodeBench dataset is downloaded and cached on first use.

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
