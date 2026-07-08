from pathlib import Path

from config.settings import RESULTS_DIR


def safe_name(model_name: str) -> str:
    return model_name.replace("/", "_").replace(".", "-")


def mr_folder_name(relation_name: str, relation_id: int | None) -> str:
    if relation_id is not None:
        return f"mr-{relation_id:02d}-{relation_name}"
    return f"mr-{relation_name}"


def build_results_dir(
    model_name: str,
    relation_name: str | None = None,
    relation_id: int | None = None,
) -> Path:
    model_dir = Path(RESULTS_DIR) / safe_name(model_name)
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
) -> Path:
    if relation_name:
        filename = f"{mr_folder_name(relation_name, relation_id)}.json"
        return build_results_dir(model_name, relation_name, relation_id) / filename

    base_dir = Path(RESULTS_DIR) / safe_name(model_name) / "base"
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir / "base.json"
