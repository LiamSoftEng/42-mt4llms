from pathlib import Path

from config.settings import RESULTS_DIR


def safe_name(model_name: str) -> str:
    return model_name.replace("/", "_").replace(".", "-")


def mr_folder_name(relation_name: str, relation_id: int | None) -> str:
    if relation_id is not None:
        return f"mr-{relation_id:02d}-{relation_name}"
    return f"mr-{relation_name}"


def build_results_path(
    model_name: str,
    relation_name: str | None = None,
    relation_id: int | None = None,
) -> Path:
    model_dir = Path(RESULTS_DIR) / safe_name(model_name)
    model_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{mr_folder_name(relation_name, relation_id)}.json" if relation_name else "base.json"
    return model_dir / filename
