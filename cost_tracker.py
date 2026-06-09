from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from config.settings import COST_RATES_PER_MILLION, COST_TRACKING_PATH

_WARNED = False


def update_costs(model_name: str, usage: dict | None) -> None:
    if not usage:
        return

    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    if prompt_tokens is None and completion_tokens is None:
        return

    prompt_tokens = int(prompt_tokens or 0)
    completion_tokens = int(completion_tokens or 0)
    total_tokens = prompt_tokens + completion_tokens

    rates = COST_RATES_PER_MILLION.get(model_name)
    missing_rate = False
    if not rates:
        rates = {"prompt": 0.0, "completion": 0.0}
        missing_rate = True

    prompt_rate = float(rates.get("prompt", 0.0))
    completion_rate = float(rates.get("completion", 0.0))
    cost = (prompt_tokens / 1_000_000) * prompt_rate + (completion_tokens / 1_000_000) * completion_rate

    try:
        path = Path(COST_TRACKING_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = _load_payload(path)
        model_entry = data["models"].setdefault(
            model_name,
            {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0.0,
                "last_updated": "",
            },
        )

        model_entry["prompt_tokens"] += prompt_tokens
        model_entry["completion_tokens"] += completion_tokens
        model_entry["total_tokens"] += total_tokens
        model_entry["cost_usd"] += cost
        model_entry["last_updated"] = _utc_now()

        if missing_rate:
            model_entry["missing_rate"] = True
        else:
            model_entry.pop("missing_rate", None)

        _save_payload(path, data)
    except Exception as exc:
        _warn_once(f"Cost tracking skipped: {exc}")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _default_payload() -> dict:
    return {
        "version": 1,
        "currency": "NZD",
        "unit": "per_million_tokens",
        "models": {},
    }


def _load_payload(path: Path) -> dict:
    if not path.exists():
        return _default_payload()

    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            return _default_payload()
    except (OSError, json.JSONDecodeError):
        return _default_payload()

    data.setdefault("version", 1)
    data.setdefault("currency", "NZD")
    data.setdefault("unit", "per_million_tokens")
    if not isinstance(data.get("models"), dict):
        data["models"] = {}
    return data


def _save_payload(path: Path, data: dict) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=True)
    temp_path.replace(path)


def _warn_once(message: str) -> None:
    global _WARNED
    if _WARNED:
        return
    _WARNED = True
    print(f"[cost_tracker] {message}", file=sys.stderr)
