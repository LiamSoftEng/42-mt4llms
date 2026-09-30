from pathlib import Path

from config.settings import RESULTS_DIR


def safe_name(model_name: str) -> str:
    return model_name.replace("/", "_").replace(".", "-")


def mr_folder_name(relation_name: str, relation_id: int | None) -> str:
    if relation_id is not None:
        return f"mr-{relation_id:02d}-{relation_name}"
    return f"mr-{relation_name}"


def build_model_results_dir(model_name: str, difficulty: str | None = None) -> Path:
    """Return the model's result root, optionally scoped to one difficulty."""
    model_dir = Path(RESULTS_DIR) / safe_name(model_name)
    if difficulty:
        model_dir = model_dir / difficulty.lower()
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def build_results_dir(
    model_name: str,
    relation_name: str | None = None,
    relation_id: int | None = None,
    difficulty: str | None = None,
) -> Path:
    model_dir = build_model_results_dir(model_name, difficulty)
    if relation_name:
        results_dir = model_dir / mr_folder_name(relation_name, relation_id)
    else:
        results_dir = model_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def build_results_path(
    model_name: str,
    relation_name: str | None = None,
    relation_id: int | None = None,
    difficulty: str | None = None,
) -> Path:
    if relation_name:
        filename = f"{mr_folder_name(relation_name, relation_id)}.json"
        return build_results_dir(model_name, relation_name, relation_id, difficulty) / filename

    base_dir = build_model_results_dir(model_name, difficulty) / "base"
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir / "base.json"


def build_repeats_dir(model_name: str, difficulty: str | None = None) -> Path:
    """Folder holding one result file per --repeats run, inside the base folder."""
    base_dir = build_results_path(model_name, None, None, difficulty).parent
    repeats_dir = base_dir / "repeats"
    repeats_dir.mkdir(parents=True, exist_ok=True)
    return repeats_dir


def build_repeat_path(model_name: str, repeat_index: int, difficulty: str | None = None) -> Path:
    """Path for a single repeat's result file. repeat_index is 0-based."""
    return build_repeats_dir(model_name, difficulty) / f"repeat-{repeat_index + 1:02d}.json"
