from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from config.settings import COST_RATES_PER_MILLION, COST_TRACKING_PATH, MR_HELPER_MODEL

_WARNED = False
_MODEL_CALL_COUNTS = {"main": 0, "helper": 0}
_RUN_COSTS: dict[str, dict[str, dict]] = {"main": {}, "helper": {}}


def reset_model_call_counts() -> None:
    for key in _MODEL_CALL_COUNTS:
        _MODEL_CALL_COUNTS[key] = 0
    for role in _RUN_COSTS:
        _RUN_COSTS[role].clear()


def record_model_call(kind: str) -> None:
    _MODEL_CALL_COUNTS[kind] += 1


def get_model_call_counts() -> dict[str, int]:
    return dict(_MODEL_CALL_COUNTS)


def get_run_cost_summary() -> dict[str, dict[str, dict]]:
    return {
        role: {model: dict(values) for model, values in models.items()}
        for role, models in _RUN_COSTS.items()
    }


def reconcile_cost_file() -> None:
    """Upgrade ledger metadata and recover entries that were explicitly unpriced."""
    try:
        path = Path(COST_TRACKING_PATH)
        if not path.exists():
            return
        data = _load_payload(path)
        _save_payload(path, data)
    except Exception as exc:
        _warn_once(f"Cost ledger reconciliation skipped: {exc}")


def update_costs(model_name: str, usage: dict | None, role: str = "main") -> None:
    if not usage:
        return
    if role not in _RUN_COSTS:
        raise ValueError(f"Unknown cost role: {role}")

    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    if prompt_tokens is None and completion_tokens is None:
        return

    prompt_tokens = int(prompt_tokens or 0)
    completion_tokens = int(completion_tokens or 0)
    cached_prompt_tokens = min(int(usage.get("cached_prompt_tokens") or 0), prompt_tokens)
    uncached_prompt_tokens = prompt_tokens - cached_prompt_tokens
    total_tokens = prompt_tokens + completion_tokens

    rates = COST_RATES_PER_MILLION.get(model_name)
    missing_rate = False
    if not rates:
        rates = {"prompt": 0.0, "completion": 0.0}
        missing_rate = True

    prompt_rate = float(rates.get("prompt", 0.0))
    cached_prompt_rate = float(rates.get("cached_prompt", prompt_rate))
    completion_rate = float(rates.get("completion", 0.0))
    cost = (
        (uncached_prompt_tokens / 1_000_000) * prompt_rate
        + (cached_prompt_tokens / 1_000_000) * cached_prompt_rate
        + (completion_tokens / 1_000_000) * completion_rate
    )

    try:
        path = Path(COST_TRACKING_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = _load_payload(path)
        model_entry = data["models"].setdefault(
            model_name,
            {
                "prompt_tokens": 0,
                "cached_prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0.0,
                "last_updated": "",
            },
        )

        model_entry["prompt_tokens"] += prompt_tokens
        model_entry["cached_prompt_tokens"] = model_entry.get("cached_prompt_tokens", 0) + cached_prompt_tokens
        model_entry["completion_tokens"] += completion_tokens
        model_entry["total_tokens"] += total_tokens
        model_entry["cost_usd"] += cost
        model_entry["last_updated"] = _utc_now()
        model_entry["last_rates_per_million_usd"] = {
            "prompt": prompt_rate,
            "cached_prompt": cached_prompt_rate,
            "completion": completion_rate,
        }

        role_entry = model_entry.setdefault("usage_by_role", {}).setdefault(
            role,
            _empty_usage_entry(),
        )
        _add_usage(role_entry, prompt_tokens, cached_prompt_tokens, completion_tokens, cost)

        if missing_rate:
            model_entry["missing_rate"] = True
        else:
            model_entry.pop("missing_rate", None)

        run_entry = _RUN_COSTS[role].setdefault(model_name, _empty_usage_entry())
        _add_usage(run_entry, prompt_tokens, cached_prompt_tokens, completion_tokens, cost)
        if missing_rate:
            run_entry["missing_rate"] = True

        _save_payload(path, data)
    except Exception as exc:
        _warn_once(f"Cost tracking skipped: {exc}")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _empty_usage_entry() -> dict:
    return {
        "prompt_tokens": 0,
        "cached_prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cost_usd": 0.0,
    }


def _add_usage(
    entry: dict,
    prompt_tokens: int,
    cached_prompt_tokens: int,
    completion_tokens: int,
    cost_usd: float,
) -> None:
    entry["prompt_tokens"] = entry.get("prompt_tokens", 0) + prompt_tokens
    entry["cached_prompt_tokens"] = entry.get("cached_prompt_tokens", 0) + cached_prompt_tokens
    entry["completion_tokens"] = entry.get("completion_tokens", 0) + completion_tokens
    entry["total_tokens"] = entry.get("total_tokens", 0) + prompt_tokens + completion_tokens
    entry["cost_usd"] = entry.get("cost_usd", 0.0) + cost_usd


def _default_payload() -> dict:
    return {
        "version": 2,
        "currency": "USD",
        "unit": "per_million_tokens",
        "models": {},
        "totals": _empty_usage_entry(),
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

    data["version"] = 2
    data["currency"] = "USD"
    data.setdefault("unit", "per_million_tokens")
    if not isinstance(data.get("models"), dict):
        data["models"] = {}
    _upgrade_model_entries(data)
    _refresh_totals(data)
    return data


def _save_payload(path: Path, data: dict) -> None:
    _refresh_totals(data)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=True)
    temp_path.replace(path)


def _upgrade_model_entries(data: dict) -> None:
    """Add role metadata and recover only costs that were previously unpriced."""
    for model_name, entry in data["models"].items():
        entry.setdefault("cached_prompt_tokens", 0)
        if not isinstance(entry.get("usage_by_role"), dict):
            historical_role = "helper" if model_name == MR_HELPER_MODEL else "main"
            entry["usage_by_role"] = {
                historical_role: {
                    "prompt_tokens": entry.get("prompt_tokens", 0),
                    "cached_prompt_tokens": entry.get("cached_prompt_tokens", 0),
                    "completion_tokens": entry.get("completion_tokens", 0),
                    "total_tokens": entry.get("total_tokens", 0),
                    "cost_usd": entry.get("cost_usd", 0.0),
                }
            }

        rates = COST_RATES_PER_MILLION.get(model_name)
        if entry.get("missing_rate") and rates:
            prompt_tokens = int(entry.get("prompt_tokens", 0))
            cached_tokens = min(int(entry.get("cached_prompt_tokens", 0)), prompt_tokens)
            completion_tokens = int(entry.get("completion_tokens", 0))
            recovered_cost = (
                ((prompt_tokens - cached_tokens) / 1_000_000) * float(rates.get("prompt", 0.0))
                + (cached_tokens / 1_000_000)
                * float(rates.get("cached_prompt", rates.get("prompt", 0.0)))
                + (completion_tokens / 1_000_000) * float(rates.get("completion", 0.0))
            )
            entry["cost_usd"] = recovered_cost
            for role_entry in entry["usage_by_role"].values():
                role_entry["cost_usd"] = recovered_cost
            entry["recovered_previously_unpriced_cost"] = True
            entry["last_rates_per_million_usd"] = {
                "prompt": float(rates.get("prompt", 0.0)),
                "cached_prompt": float(rates.get("cached_prompt", rates.get("prompt", 0.0))),
                "completion": float(rates.get("completion", 0.0)),
            }
            entry.pop("missing_rate", None)


def _refresh_totals(data: dict) -> None:
    totals = _empty_usage_entry()
    totals["by_role"] = {"main": _empty_usage_entry(), "helper": _empty_usage_entry()}
    for entry in data.get("models", {}).values():
        for key in _empty_usage_entry():
            totals[key] += entry.get(key, 0)
        for role, role_entry in entry.get("usage_by_role", {}).items():
            target = totals["by_role"].setdefault(role, _empty_usage_entry())
            for key in _empty_usage_entry():
                target[key] += role_entry.get(key, 0)
    data["totals"] = totals


def _warn_once(message: str) -> None:
    global _WARNED
    if _WARNED:
        return
    _WARNED = True
    print(f"[cost_tracker] {message}", file=sys.stderr)
