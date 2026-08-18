# 42-mt4llms — Full Codebase Reference

Auto-generated from the working tree. File contents below are stored in 
4-space-indented code blocks so any literal ``` inside source is preserved.

- Total files: 38
- Total bytes: 194,829

## Excluded from this document

- `venv/` — virtual environment
- `__pycache__/` — bytecode caches
- `.git/` — repository metadata
- `results/` — generated experiment outputs
- `.env` — local API keys (do not commit / share)

---

## Project Structure

```text
.env.example
.gitignore
compare_repeats.py
config/run_config.py
config/settings.py
cost_tracker.py
data/decoder.py
data/loader.py
evaluation/executor.py
evaluation/harness/functional.py
evaluation/harness/stdin.py
evaluation/metric_violations.py
evaluation/metrics.py
evaluation/paths.py
evaluation/report.py
evaluation/results_writer.py
evaluation/runner.py
explore_dataset.py
INSTRUCTIONS.md
llms/__init__.py
llms/base.py
llms/claude_model.py
llms/deepseek_model.py
llms/gemini_model.py
llms/grok_model.py
llms/mr_helper.py
llms/openai_model.py
main.py
metamorphic/__init__.py
metamorphic/relations/__init__.py
metamorphic/relations/base_mr.py
metamorphic/relations/mrs.py
metamorphic/relations/phrases.py
metamorphic/relations_list.py
README.md
requirements.txt
run_all_mrs.py
runtests.py
```

---

## .env.example

```text
File: .env.example (156 bytes)
```

    ANTHROPIC_API_KEY=your_anthropic_key_here
    OPENAI_API_KEY=your_openai_key_here
    GOOGLE_API_KEY=your_google_key_here
    DEEPSEEK_API_KEY=your_deepseek_key_here

---

## .gitignore

```text
File: .gitignore (196 bytes)
```

    # Environment
    .env
    venv/
     
    # Results
    results/
     
    # HuggingFace cache
    ~/.cache/huggingface/
     
    # Python
    __pycache__/
    *.py[cod]
    *.egg-info/
    dist/
    build/
     
    # OS
    .DS_Store
    Thumbs.db


---

## compare_repeats.py

```text
File: compare_repeats.py (13614 bytes)
```

    """
    Run tests against every per-repeat result file produced by a base run with
    --repeats N (see runner.py), then compare pass rate and code metrics across
    the repeats.

    Usage:
        python compare_repeats.py --model gpt-5-nano
        python compare_repeats.py --dir results/gpt-5-nano/base/repeats
        python compare_repeats.py --model gpt-5-nano --csv results/gpt-5-nano/base/repeats/comparison.csv
        python compare_repeats.py --model gpt-5-nano --html results/gpt-5-nano/base/repeats/comparison.html
    """

    import argparse
    import csv
    import sys
    from contextlib import redirect_stdout, redirect_stderr
    from pathlib import Path
    from statistics import mean, stdev

    from runtests import run_tests, _Tee
    from evaluation.paths import build_results_path
    from evaluation.report import _CSS, _fmt_duration, _pass_badge


    def _resolve_dir(model: str | None, dir_arg: str | None) -> Path:
        if dir_arg:
            return Path(dir_arg)
        if not model:
            raise SystemExit("Provide --model or --dir")
        return build_results_path(model, None, None).parent / "repeats"


    def _mean_metric(rows: list[dict], key: str) -> float | None:
        values = [r[key] for r in rows if isinstance(r.get(key), (int, float)) and r.get(key) != -1]
        return round(mean(values), 3) if values else None


    def _fmt_number(value: float | None) -> str:
        if value is None:
            return "N/A"
        return f"{value:,.1f}"


    def _mean_list(values: list) -> float | None:
        nums = [v for v in values if isinstance(v, (int, float)) and v != -1]
        return mean(nums) if nums else None


    def _write_html_report(
        per_repeat: list[dict],
        per_question: dict[str, dict],
        flips: int,
        total_questions: int,
        repeats_dir: Path,
        out_path: Path,
    ) -> None:
        """Write a comparison HTML report styled identically to the runtests reports."""
        n_runs = len(per_repeat)

        pass_rates = [r["pass_rate"] for r in per_repeat if r["pass_rate"] is not None]
        extract_rates = [r["extractable_rate"] for r in per_repeat if r["extractable_rate"] is not None]
        mean_pass = mean(pass_rates) if pass_rates else None
        mean_extract = mean(extract_rates) if extract_rates else None

        def _avg(key: str) -> float | None:
            values = [r[key] for r in per_repeat if isinstance(r.get(key), (int, float))]
            return mean(values) if values else None

        mean_ncloc = _avg("mean_ncloc")
        mean_cc = _avg("mean_cc")
        mean_tokens = _avg("mean_total_tokens")
        mean_call = _avg("mean_llm_call_time")
        inconsistent_pct = round(flips / total_questions * 100) if total_questions else 0

        stats_html = f"""
            <div class="stat-box">
                <div class="stat-label">Runs compared</div>
                <div class="stat-value">{n_runs}</div>
                <div class="stat-sub">repeat-*.json files</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Mean pass rate</div>
                <div class="stat-value">{mean_pass * 100:.1f}%</div>
                <div class="stat-sub">across {n_runs} runs</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Mean extractable</div>
                <div class="stat-value">{mean_extract * 100:.1f}%</div>
                <div class="stat-sub">across {n_runs} runs</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Inconsistent pass/fail</div>
                <div class="stat-value">{flips}/{total_questions}</div>
                <div class="stat-sub">{inconsistent_pct}% of questions</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Mean NCLOC</div>
                <div class="stat-value">{_fmt_number(mean_ncloc)}</div>
                <div class="stat-sub">code size</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Mean CC</div>
                <div class="stat-value">{_fmt_number(mean_cc)}</div>
                <div class="stat-sub">complexity</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Mean tokens</div>
                <div class="stat-value">{_fmt_number(mean_tokens)}</div>
                <div class="stat-sub">total per problem</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Mean LLM call time</div>
                <div class="stat-value">{_fmt_duration(mean_call)}</div>
                <div class="stat-sub">per problem</div>
            </div>"""

        rows_html = ""
        for r in per_repeat:
            pass_pct = f"{r['pass_rate'] * 100:.1f}%" if r["pass_rate"] is not None else "N/A"
            ext_pct = f"{r['extractable_rate'] * 100:.1f}%" if r["extractable_rate"] is not None else "N/A"
            rows_html += f"""
                <tr>
                    <td><code>{r['file']}</code></td>
                    <td>{pass_pct}</td>
                    <td>{ext_pct}</td>
                    <td>{_fmt_number(r['mean_ncloc'])}</td>
                    <td>{_fmt_number(r['mean_cc'])}</td>
                    <td>{_fmt_number(r['mean_total_tokens'])}</td>
                    <td>{_fmt_duration(r['mean_llm_call_time'])}</td>
                </tr>"""

        table_headers = """
                <th>Run</th>
                <th>Pass%</th>
                <th>Extr%</th>
                <th>NCLOC</th>
                <th>CC</th>
                <th>Tokens</th>
                <th>Call time</th>"""

        # ── Per-question breakdown ────────────────────────────────────────────────
        question_rows_html = ""
        for q in per_question.values():
            passes = q["pass"]
            run_badges = "".join(
                f"<td>{_pass_badge(p)}</td>" for p in passes
            )
            non_none = {p for p in passes if p is not None}
            consistent = len(non_none) <= 1
            if consistent and non_none:
                consistency_badge = '<span class="badge holds">YES</span>'
            elif consistent:
                consistency_badge = '<span class="badge na">N/A</span>'
            else:
                consistency_badge = '<span class="badge violated">NO</span>'

            title = q.get("question_title") or ""
            difficulty = q.get("difficulty") or ""
            platform = q.get("platform") or ""
            question_rows_html += f"""
                <tr>
                    <td>
                        <code>{q["question_id"]}</code><br>
                        <small>{title}</small><br>
                        <span class="tag">{difficulty}</span>
                        <span class="tag">{platform}</span>
                    </td>
                    {run_badges}
                    <td>{consistency_badge}</td>
                    <td>{_fmt_number(_mean_list(q["ncloc"]))}</td>
                    <td>{_fmt_number(_mean_list(q["cc"]))}</td>
                    <td>{_fmt_number(_mean_list(q["tokens"]))}</td>
                    <td>{_fmt_duration(_mean_list(q["call_time"]))}</td>
                </tr>"""

        run_head_cells = "".join(f"<th>R{i+1}</th>" for i in range(n_runs))
        question_headers = f"""
                <th>Problem</th>
                {run_head_cells}
                <th>Consistent</th>
                <th>NCLOC</th>
                <th>CC</th>
                <th>Tokens</th>
                <th>Call time</th>"""

        question_tables_html = ""
        if question_rows_html:
            question_tables_html = f"""
            <h2>Per-question breakdown</h2>
            <div class="table-wrap">
            <table>
              <thead><tr>{question_headers}</tr></thead>
              <tbody>{question_rows_html}</tbody>
            </table>
            </div>"""

        html = f"""<!DOCTYPE html>
    <html lang="en">
    <head>
    <meta charset="UTF-8">
    <title>Repeat comparison — {repeats_dir}</title>
    <style>
    {_CSS}
      h2 {{ font-size: 1.1rem; margin: 2rem 0 0.75rem; color: #212529; }}
    </style>
    </head>
    <body>
    <h1>Repeat comparison</h1>
    <div class="subtitle">{repeats_dir} — {n_runs} runs</div>

    <div class="stats">
    {stats_html}
    </div>
    <h2>Run summary</h2>
    <div class="table-wrap">
    <table>
      <thead><tr>{table_headers}</tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
    </div>
    {question_tables_html}
    </body>
    </html>"""

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(html)
        print(f"  HTML report -> {out_path}")


    def compare(repeats_dir: Path, csv_out: str | None = None, html_out: str | None = None) -> None:
        files = sorted(repeats_dir.glob("repeat-*.json"))
        if not files:
            print(f"No repeat files found in {repeats_dir}")
            return

        per_repeat = []
        per_question: dict[str, dict] = {}

        for f in files:
            log_path = f.with_suffix(".tests.log")
            with open(log_path, "w", encoding="utf-8") as log_fh:
                with redirect_stdout(_Tee(sys.stdout, log_fh)), redirect_stderr(_Tee(sys.stdout, log_fh)):
                    metric_rows = run_tests(
                        str(f),
                        csv_path=str(f.with_suffix(".metrics.csv")),
                        write_outputs=False,
                    )

            total = len(metric_rows)
            passed = sum(1 for r in metric_rows if r.get("source_passed"))
            extractable = sum(1 for r in metric_rows if r.get("source_extractable"))

            per_repeat.append({
                "file": f.name,
                "total": total,
                "passed": passed,
                "pass_rate": round(passed / total, 4) if total else None,
                "extractable_rate": round(extractable / total, 4) if total else None,
                "mean_ncloc": _mean_metric(metric_rows, "source_ncloc"),
                "mean_cc": _mean_metric(metric_rows, "source_cc"),
                "mean_total_tokens": _mean_metric(metric_rows, "source_total_tokens"),
                "mean_llm_call_time": _mean_metric(metric_rows, "source_llm_call_time"),
            })

            for row in metric_rows:
                q = per_question.setdefault(row["id"], {
                    "question_id": row["id"],
                    "question_title": row.get("question_title"),
                    "difficulty": row.get("difficulty"),
                    "platform": row.get("platform"),
                    "pass": [],
                    "ncloc": [],
                    "cc": [],
                    "tokens": [],
                    "call_time": [],
                })
                q["pass"].append(row.get("source_passed"))
                q["ncloc"].append(row.get("source_ncloc"))
                q["cc"].append(row.get("source_cc"))
                q["tokens"].append(row.get("source_total_tokens"))
                q["call_time"].append(row.get("source_llm_call_time"))

        # ── Comparison table ─────────────────────────────────────────────────────
        print(f"\n{'='*72}")
        print(f"  Repeat comparison — {len(files)} runs from {repeats_dir}")
        print(f"{'='*72}")
        print(f"  {'File':<16} {'Pass%':>7} {'Extr%':>7} {'NCLOC':>7} {'CC':>6} {'Tokens':>9} {'CallT':>7}")
        print(f"  {'-'*16} {'-'*7} {'-'*7} {'-'*7} {'-'*6} {'-'*9} {'-'*7}")
        for r in per_repeat:
            pass_pct = f"{r['pass_rate']*100:.1f}%" if r["pass_rate"] is not None else "N/A"
            ext_pct = f"{r['extractable_rate']*100:.1f}%" if r["extractable_rate"] is not None else "N/A"
            ncloc_str = r["mean_ncloc"] if r["mean_ncloc"] is not None else "N/A"
            cc_str = r["mean_cc"] if r["mean_cc"] is not None else "N/A"
            tok_str = r["mean_total_tokens"] if r["mean_total_tokens"] is not None else "N/A"
            callt_str = r["mean_llm_call_time"] if r["mean_llm_call_time"] is not None else "N/A"
            print(
                f"  {r['file']:<16} {pass_pct:>7} {ext_pct:>7} "
                f"{ncloc_str:>7} {cc_str:>6} {tok_str:>9} {callt_str:>7}"
            )

        # ── Cross-repeat summary ─────────────────────────────────────────────────
        pass_rates = [r["pass_rate"] for r in per_repeat if r["pass_rate"] is not None]
        print()
        if len(pass_rates) > 1:
            print(f"  Pass rate across repeats: mean={mean(pass_rates):.4f}  std={stdev(pass_rates):.4f}")
        elif pass_rates:
            print(f"  Pass rate: {pass_rates[0]:.4f}")

        flips = sum(
            1 for q in per_question.values()
            if len({r for r in q["pass"] if r is not None}) > 1
        )
        print(f"  Questions with inconsistent pass/fail across repeats: {flips}/{len(per_question)}")

        path = Path(csv_out) if csv_out else repeats_dir / "comparison.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(per_repeat[0].keys()))
            writer.writeheader()
            writer.writerows(per_repeat)
        print(f"\n  Wrote comparison CSV -> {path}")

        html_path = Path(html_out) if html_out else repeats_dir / "comparison.html"
        _write_html_report(
            per_repeat,
            per_question,
            flips,
            len(per_question),
            repeats_dir,
            html_path,
        )


    if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="Compare test results across --repeats runs")
        parser.add_argument("--model", help="Model name (looks under results/<model>/base/repeats)")
        parser.add_argument("--dir", help="Explicit path to a repeats folder")
        parser.add_argument("--csv", help="Path to write the comparison CSV (default: comparison.csv inside the repeats folder)")
        parser.add_argument("--html", help="Path to write the comparison HTML report (default: comparison.html inside the repeats folder)")
        args = parser.parse_args()

        compare(_resolve_dir(args.model, args.dir), csv_out=args.csv, html_out=args.html)


---

## config/run_config.py

```text
File: config/run_config.py (1431 bytes)
```

    """
    Runnable pipeline configuration.

    Run the pipeline:
        python config/run_config.py
    """

    import subprocess
    import sys
    from pathlib import Path


    CONFIG = {
        "models": ["openai"],
        "n": 1,
        "after": None,
        "difficulty": ["easy"],
        "mr": 5,
        "mr_only": False,
        "run_tests": True,
        "repeats": 1,
        "resume": False,
    }

    FLAG_NAMES = {
        "models": "--models",
        "n": "--n",
        "after": "--after",
        "difficulty": "--difficulty",
        "mr": "--mr",
        "mr_only": "--mr-only",
        "run_tests": "--run-tests",
        "repeats": "--repeats",
        "resume": "--resume",
    }


    def _is_empty(value) -> bool:
        return value is None or value == "" or value == []


    def build_command(config: dict) -> list[str]:
        repo_root = Path(__file__).resolve().parents[1]
        cmd = [sys.executable, str(repo_root / "main.py")]

        for key, flag in FLAG_NAMES.items():
            value = config.get(key)
            if _is_empty(value):
                continue
            if isinstance(value, bool):
                if value:
                    cmd.append(flag)
                continue
            if isinstance(value, list):
                cmd.append(flag)
                cmd.extend(str(item) for item in value)
                continue
            cmd.extend([flag, str(value)])

        return cmd


    def main() -> int:
        cmd = build_command(CONFIG)
        return subprocess.run(cmd, check=False).returncode


    if __name__ == "__main__":
        raise SystemExit(main())


---

## config/settings.py

```text
File: config/settings.py (3003 bytes)
```

    import os
    from dotenv import load_dotenv

    load_dotenv()

    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HUGGINGFACE_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

    # API Keys
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    GROK_API_KEY = os.getenv("GROK_API_KEY")

    # Model names (TODO: change please, based on knowledge cutoffs + call costs)
    MR_HELPER_MODEL = "gpt-5.6-luna" #find fast, cheap model for MR helper, just putting gpt-5-nano for now
    CLAUDE_MODEL = "claude-sonnet-4-6"
    OPENAI_MODEL = "gpt-5.6-sol"
    GEMINI_MODEL = "gemini-2.5-flash"
    DEEPSEEK_MODEL = "google/gemma-2-2b-it"
    # DEEPSEEK_MODEL = "minimaxai/minimax-m2.7"
    CODE_EMBEDDING_MODEL = "Salesforce/codet5p-220m"
    CODE_TOKENIZER_MODEL = "Salesforce/codet5-base"

    # Groq model options (uncomment one)
    # GROK_MODEL = "openai/gpt-oss-20b"
    GROK_MODEL = "openai/gpt-oss-120b"
    # GROK_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
    # GROK_MODEL = "llama-3.3-70b-versatile"
    # GROK_MODEL = "llama-3.1-8b-instant"
    # GROK_MODEL = "groq/compound"
    # GROK_MODEL = "groq/compound-mini"
    # GROK_MODEL = "allam-2-7b"
    # GROK_MODEL = "meta-llama/llama-prompt-guard-2-22m"
    # GROK_MODEL = "meta-llama/llama-prompt-guard-2-86m"
    # GROK_MODEL = "openai/gpt-oss-safeguard-20b"
    # GROK_MODEL = "qwen/qwen3-32b"

    # Generation params (TODO: adjust as needed)
    MR_HELPER_TOKENS = 8192
    MAX_TOKENS = 8192
    TEMPERATURE = 0  # Deterministic outputs for reproducibility

    # Dataset
    LCB_DATASET = "bzantium/livecodebench"
    LCB_VERSION = "release_latest"
    N_SAMPLES = None  # Set to an int (e.g. 10) to limit samples during dev

    # Output
    RESULTS_DIR = "results"
    COST_TRACKING_PATH = os.getenv("COST_TRACKING_PATH", os.path.join(RESULTS_DIR, ".model_costs.json"))

    # Cost tracking (USD per 1M tokens). Update these with current provider rates.
    COST_RATES_PER_MILLION = {
        CLAUDE_MODEL: {"prompt": 0.0, "completion": 0.0},
        OPENAI_MODEL: {"prompt": 5, "completion": 30},
        GEMINI_MODEL: {"prompt": 0.0, "completion": 0.0},
        DEEPSEEK_MODEL: {"prompt": 0.0, "completion": 0.0},
        GROK_MODEL: {"prompt": 0.05, "completion": 0.40},
    }

    METRIC_VARIANCE_THRESHOLDS = {
        "ncloc": 5,
        "cyclomatic_complexity": 2.0,
        "llm_call_time": 10.0,
        "total_tokens": 500,
        "token_efficiency": 0.5,
        "code_sim": 0.85,  # minimum allowed value, not a delta
    }

    def get_metric_threshold(metric_name: str, question_id: str | None = None) -> float | None:
        # Future: check a per-question override table first, keyed by question_id.
        # For now this always falls through to the global default.
        return METRIC_VARIANCE_THRESHOLDS.get(metric_name)

---

## cost_tracker.py

```text
File: cost_tracker.py (3337 bytes)
```

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


---

## data/decoder.py

```text
File: data/decoder.py (779 bytes)
```

    """
    Decodes private test cases from the LiveCodeBench dataset.
    Private tests are stored as base64 -> zlib -> pickle -> JSON string.
    """

    import base64
    import zlib
    import pickle
    import json


    def decode_private_test_cases(raw: str) -> list:
        """
        Decode private test cases into a list of {input, output, testtype} dicts.
        Returns [] if decoding fails.
        """
        if not raw or not raw.strip():
            return []
        try:
            compressed = base64.b64decode(raw)
            decompressed = zlib.decompress(compressed)
            json_str = pickle.loads(decompressed)
            result = json.loads(json_str)
            if isinstance(result, list):
                return result
        except Exception as e:
            print(f"  [decode] Failed: {e}")
        return []

---

## data/loader.py

```text
File: data/loader.py (3117 bytes)
```

    from datetime import datetime
    from datasets import load_dataset
    from config.settings import LCB_DATASET, LCB_VERSION, N_SAMPLES
    INSTRUCTION_PREFIX = "Solve the following coding problem. Return ONLY the final code, no explanation, no code comments\n"
    INSTRUCTION_SUFFIX = "\nReturn your solution as a Python code block (```python ... ```)."

    def load_livecode_bench(n_samples: int = None, after: str = None, difficulty: str | list[str] | None = None):
        """
        Load LiveCodeBench from HuggingFace.

        Each sample contains:
          - question_id:        unique identifier
          - question_title:     short title of the problem
          - question_content:   full problem statement (HTML/markdown)
          - difficulty:         'easy' | 'medium' | 'hard'
          - platform:           'leetcode' | 'atcoder' | 'codeforces'
          - contest_date:       when the problem was published
          - starter_code:       function stub to complete (may be empty)
          - public_test_cases:  example tests shown in the problem
          - private_test_cases: hidden evaluation tests (gzipped + base64)

        Args:
            n_samples:  limit number of samples returned.
            after:      only include problems on or after this date e.g. '2024-01-01'
            difficulty: only include problems matching this difficulty. Accepts a
                        single value ('easy') or a list ('easy', 'medium'). Matching
                        is case-insensitive.
        """
        dataset = load_dataset(LCB_DATASET, name=LCB_VERSION, split="test")

        if after is not None:
            cutoff = datetime.strptime(after, "%Y-%m-%d")
            dataset = dataset.filter(
                lambda x: x["contest_date"].replace(tzinfo=None) >= cutoff
            )
            print(f"Filtered to problems on or after {after}: {len(dataset)} samples remaining")

        if difficulty is not None:
            allowed = {difficulty} if isinstance(difficulty, str) else set(difficulty)
            allowed = {d.lower() for d in allowed}
            dataset = dataset.filter(lambda x: x["difficulty"].lower() in allowed)
            print(f"Filtered to difficulty {sorted(allowed)}: {len(dataset)} samples remaining")

        # The raw dataset is ordered oldest-first by contest_date. Sort newest-first
        # so --n takes the most recent problems (within any after/difficulty filter)
        # rather than the oldest ones.
        dataset = dataset.sort("contest_date", reverse=True)

        limit = n_samples if n_samples is not None else N_SAMPLES
        if limit is not None:
            dataset = dataset.select(range(min(limit, len(dataset))))

        print(f"Loaded {len(dataset)} samples from {LCB_DATASET} ({LCB_VERSION})")
        return dataset

    def build_code_question(sample: dict) -> str:
        parts = []
        parts.append(f"Problem: {sample['question_title']}\n")
        parts.append(sample["question_content"])
        if sample.get("starter_code", "").strip():
            parts.append(f"\nUse this starter code:\n```python\n{sample['starter_code']}\n```")
        return "\n".join(parts)


    def build_prompt(question: str, prefix: str = INSTRUCTION_PREFIX, suffix: str = INSTRUCTION_SUFFIX) -> str:
        return prefix + question + suffix

---

## evaluation/executor.py

```text
File: evaluation/executor.py (3117 bytes)
```

    """
    Runs public test cases against model responses.
    """
    import json
    import re
    import time


    def extract_code(response: str) -> str | None:
        """
        Pull the Python code block out of the model response.

        Tries, in order:
          1. ```python ... ``` fenced block (preferred, most explicit)
          2. ``` ... ``` fenced block with no language tag
          3. Raw response treated as code, if it has no fence markers at all
             (some models emit bare code with no markdown fencing)
        Returns None only if the response is empty or contains fence markers
        that don't resolve into a matched block (i.e. genuinely unparseable).
        """
        if not response:
            return None

        match = re.search(r"```python\s*(.*?)```", response, re.DOTALL)
        if match:
            return match.group(1).strip()

        match = re.search(r"```\s*(.*?)```", response, re.DOTALL)
        if match:
            return match.group(1).strip()

        stripped = response.strip()
        if stripped and "```" not in stripped:
            return stripped

        return None


    def get_all_tests(entry: dict) -> list[dict]:
        """Decode private test cases from the raw base64 string at test time."""
        from data.decoder import decode_private_test_cases
        raw = entry.get("private_test_cases", "")
        return decode_private_test_cases(raw)


    def get_func_name(entry: dict) -> str | None:
        """Extract the function name from metadata (used for functional problems)."""
        raw = entry.get("metadata", "{}")
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
                if isinstance(raw, str):
                    raw = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                return None
        if not isinstance(raw, dict):
            return None
        return raw.get("func_name", None)


    def run_test(
        code: str,
        test: dict,
        func_name: str | None,
        debug: bool = False,
    ) -> tuple[bool, str, str, float]:
        """
        Run a single test case against the extracted code.
        Returns (passed, actual_output, error, elapsed_seconds).
        elapsed_seconds is the wall-clock time spent inside the harness.
        """
        testtype = test.get("testtype", "stdin")
        inp = test.get("input", "")
        expected = test.get("output", "")

        t0 = time.perf_counter()

        if testtype == "stdin":
            from evaluation.harness.stdin import run, check
            actual, error = run(code, inp)
            passed = check(actual, expected) if not error else False

        elif testtype == "functional":
            if not func_name:
                elapsed = time.perf_counter() - t0
                return False, "", "No func_name in metadata", elapsed
            from evaluation.harness.functional import run, check
            actual, error = run(code, func_name, inp, debug=debug)
            passed = check(actual, expected) if not error else False

        else:
            elapsed = time.perf_counter() - t0
            return False, "", f"Unknown testtype: {testtype}", elapsed

        elapsed = time.perf_counter() - t0
        return passed, actual, error, elapsed

---

## evaluation/harness/functional.py

```text
File: evaluation/harness/functional.py (3049 bytes)
```

    """
    Harness for functional style problems (LeetCode).
    The model returns a Solution class with a method. We construct a self-contained
    script that calls the function with test inputs and prints the result.
    """

    import subprocess
    import sys
    import tempfile
    import os
    import json


    RUNNER_TEMPLATE = """from typing import List, Dict, Tuple, Optional, Set
    {code}

    import json

    sol = Solution()
    result = sol.{func_name}({args})
    print(json.dumps(result, separators=(',', ':')))
    """


    def parse_args(test_input: str) -> str:
        """
        LeetCode inputs are newline-separated JSON values, one per argument.
        e.g. '"[1,2,3]"\n"4"' -> '[1,2,3], 4'
        Parse each line as JSON so Python receives the correct types.
        """
        args = []
        for line in test_input.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                # Each line is a JSON-encoded value
                parsed = json.loads(line)
                args.append(repr(parsed))
            except json.JSONDecodeError:
                # Fall back to using the raw line
                args.append(line)
        return ", ".join(args)


    def run(code: str, func_name: str, test_input: str, timeout: int = 10, debug: bool = False) -> tuple[str, str]:
        """
        Execute a functional-style solution.

        Returns:
            (stdout, stderr)
        """
        args = parse_args(test_input)
        script = RUNNER_TEMPLATE.format(code=code, func_name=func_name, args=args)

        if debug:
            print("\n--- GENERATED SCRIPT ---")
            print(script)
            print("--- END SCRIPT ---\n")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
            f.write(script)
            tmp_path = f.name

        try:
            result = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return result.stdout.strip(), result.stderr.strip()
        except subprocess.TimeoutExpired:
            return "", "TIMEOUT"
        finally:
            os.unlink(tmp_path)


    def check(actual: str, expected: str) -> bool:
        """
        Compare actual vs expected output.
        Parses both as JSON for structural comparison, falls back to string compare.
        Also handles LeetCode's 'true'/'false' vs Python's 'True'/'False'.
        """
        # Normalise Python booleans to JSON booleans for comparison
        actual = actual.strip().replace("True", "true").replace("False", "false")
        expected = expected.strip().replace("True", "true").replace("False", "false")

        try:
            actual_val = json.loads(actual)
            expected_val = json.loads(expected)
            if isinstance(actual_val, list) and isinstance(expected_val, list):
                if sorted(str(x) for x in actual_val) == sorted(str(x) for x in expected_val):
                    return True
            return actual_val == expected_val
        except (json.JSONDecodeError, TypeError):
            return actual == expected

---

## evaluation/harness/stdin.py

```text
File: evaluation/harness/stdin.py (1296 bytes)
```

    """
    Harness for stdin/stdout style problems (Codeforces).
    The model's code is run as a subprocess with test input piped to stdin,
    and stdout is compared against the expected output.
    """

    import subprocess
    import sys
    import tempfile
    import os


    def run(code: str, test_input: str, timeout: int = 10) -> tuple[bool, str, str]:
        """
        Execute code with the given stdin input.

        Returns:
            (passed, actual_output, error_message)
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
            f.write(code)
            tmp_path = f.name

        try:
            result = subprocess.run(
                [sys.executable, tmp_path],
                input=test_input,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return "", "TIMEOUT"
        finally:
            os.unlink(tmp_path)


    def check(actual: str, expected: str) -> bool:
        """Compare outputs line by line, ignoring trailing whitespace."""
        actual_lines = [l.strip() for l in actual.strip().splitlines()]
        expected_lines = [l.strip() for l in expected.strip().splitlines()]
        return actual_lines == expected_lines

---

## evaluation/metric_violations.py

```text
File: evaluation/metric_violations.py (2510 bytes)
```

    """
    Shared metric-violation checking logic.

    Used by both results_writer.py (violations.json log) and report.py
    (per-cell highlighting in the HTML reports), so the two always agree on
    what counts as a metamorphic-relation violation. Lives in its own module
    rather than in results_writer.py (its original home) because report.py
    needs it too, and results_writer.py already imports from report.py —
    report.py importing back from results_writer.py would be circular.
    """

    from config.settings import METRIC_VARIANCE_THRESHOLDS, get_metric_threshold


    def check_metric_violations(base_row: dict, followup_row: dict) -> dict:
        """
        Compare base vs followup metric values against METRIC_VARIANCE_THRESHOLDS.

        Every metric except code_sim is checked as a max-allowed |delta| between
        base and followup. code_sim is checked as a minimum allowed value on the
        followup alone, since there's no base code_sim to diff against.

        Returns a dict of {metric_name: {"base":..., "followup":..., "delta":...,
        "threshold":...}} for every metric that breached its threshold. Metrics
        missing on either side, or with no configured threshold, are skipped.
        """
        question_id = followup_row.get("question_id") or base_row.get("question_id")
        violations: dict = {}

        for metric_name in METRIC_VARIANCE_THRESHOLDS:
            threshold = get_metric_threshold(metric_name, question_id)
            if threshold is None:
                continue

            if metric_name == "code_sim":
                followup_value = followup_row.get("code_sim")
                if not isinstance(followup_value, (int, float)):
                    continue
                if followup_value < threshold:
                    violations["code_sim"] = {
                        "base": None,
                        "followup": followup_value,
                        "delta": None,
                        "threshold": threshold,
                    }
                continue

            base_value = base_row.get(metric_name)
            followup_value = followup_row.get(metric_name)
            if not isinstance(base_value, (int, float)) or not isinstance(followup_value, (int, float)):
                continue

            delta = followup_value - base_value
            if abs(delta) > threshold:
                violations[metric_name] = {
                    "base": base_value,
                    "followup": followup_value,
                    "delta": delta,
                    "threshold": threshold,
                }

        return violations

---

## evaluation/metrics.py

```text
File: evaluation/metrics.py (12131 bytes)
```

    """
    Code metrics computed on extracted code blocks using radon.
    """

    from functools import lru_cache
    from itertools import combinations
    import os
    from statistics import mean, stdev
    import textwrap
    import warnings

    from radon.raw import analyze
    from radon.complexity import cc_visit

    from evaluation.executor import extract_code


    def _codesim_debug(message: str) -> None:
        if os.getenv("CODE_SIM_DEBUG") == "1":
            print(f"  [codesim] {message}")


    # ---------------------------------------------------------------------------
    # Embedding backend
    # ---------------------------------------------------------------------------

    @lru_cache(maxsize=1)
    def _get_code_embedder():
        from sentence_transformers import SentenceTransformer
        from config.settings import CODE_EMBEDDING_MODEL

        os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")
        try:
            return SentenceTransformer(CODE_EMBEDDING_MODEL, model_kwargs={"use_safetensors": False})
        except Exception as exc:
            _codesim_debug(f"sentence-transformers init failed: {exc}")
            return None


    @lru_cache(maxsize=1)
    def _get_transformer_encoder():
        from config.settings import CODE_EMBEDDING_MODEL, CODE_TOKENIZER_MODEL

        os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("HUGGINGFACE_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

        warnings.filterwarnings(
            "ignore",
            message="`resume_download` is deprecated*",
            category=FutureWarning,
            module="huggingface_hub",
        )

        from transformers import AutoConfig, AutoModel, AutoTokenizer, T5EncoderModel
        import torch
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()

        tokenizer_model = CODE_TOKENIZER_MODEL or CODE_EMBEDDING_MODEL
        config = AutoConfig.from_pretrained(CODE_EMBEDDING_MODEL)
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, use_fast=True)
        except Exception as exc:
            _codesim_debug(f"fast tokenizer load failed for {tokenizer_model}: {exc}")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, use_fast=False)

        if config.model_type in {"t5", "mt5"}:
            model = T5EncoderModel.from_pretrained(CODE_EMBEDDING_MODEL, use_safetensors=False)
        else:
            model = AutoModel.from_pretrained(CODE_EMBEDDING_MODEL, use_safetensors=False)
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        return tokenizer, model, device


    def _encode_with_transformers(code_a: str, code_b: str):
        import torch

        tokenizer, model, device = _get_transformer_encoder()
        code_a = str(code_a)
        code_b = str(code_b)
        inputs = tokenizer([code_a, code_b], return_tensors="pt", padding=True, truncation=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        token_embeddings = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"]
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = (token_embeddings * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        pooled = summed / counts
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return pooled[0], pooled[1]


    # ---------------------------------------------------------------------------
    # Similarity / distance
    # ---------------------------------------------------------------------------

    def compute_code_similarity(code_a: str | None, code_b: str | None) -> float | None:
        """
        Compute cosine similarity between two code snippets using embeddings.
        Returns None when inputs are missing or embedding fails.
        """
        if not code_a or not code_b:
            return None
        try:
            emb_a, emb_b = _encode_with_transformers(code_a, code_b)
            import torch
            score = torch.nn.functional.cosine_similarity(emb_a, emb_b, dim=0).item()
            return float(score)
        except Exception as exc:
            _codesim_debug(f"transformers fallback failed: {exc}")
            return None


    def compute_code_distance(code_a: str | None, code_b: str | None) -> float | None:
        """
        Compute embedding distance as 1 - cosine similarity.
        Returns None when similarity cannot be computed.
        """
        similarity = compute_code_similarity(code_a, code_b)
        if similarity is None:
            return None
        return float(1 - similarity)


    def compute_pairwise_code_variability(codes: list[str | None]) -> dict:
        """
        Measure variability across repeated generated code samples.

        Each code sample is embedded and compared against every other sample.
        Standard deviation is computed over pairwise embedding distances, so
        higher values mean the prompt produced less consistent code across runs.
        """
        valid_codes = [code for code in codes if code]
        distances = []
        similarities = []

        for code_a, code_b in combinations(valid_codes, 2):
            similarity = compute_code_similarity(code_a, code_b)
            if similarity is None:
                continue
            similarities.append(float(similarity))
            distances.append(float(1 - similarity))

        if not distances:
            return {
                "n_codes": len(valid_codes),
                "n_pairs": 0,
                "mean_similarity": None,
                "std_similarity": None,
                "mean_distance": None,
                "std_distance": None,
            }

        return {
            "n_codes": len(valid_codes),
            "n_pairs": len(distances),
            "mean_similarity": round(mean(similarities), 6),
            "std_similarity": round(stdev(similarities), 6) if len(similarities) > 1 else 0.0,
            "mean_distance": round(mean(distances), 6),
            "std_distance": round(stdev(distances), 6) if len(distances) > 1 else 0.0,
        }


    # ---------------------------------------------------------------------------
    # Static code metrics
    # ---------------------------------------------------------------------------

    def compute_ncloc(code: str) -> int:
        """
        Non-comment lines of code — blank lines and comment lines excluded.
        Uses radon's lloc (logical lines of code).
        Returns -1 if code cannot be parsed.
        """
        try:
            result = analyze(code)
            return result.lloc
        except Exception:
            return -1


    def compute_cyclomatic_complexity(code: str) -> float:
        """
        Average cyclomatic complexity across all functions/methods in the code.

        radon's cc_visit only walks function/method bodies, so it has nothing to
        measure in flat/procedural code with no def blocks (common for
        LeetCode/AtCoder-style solutions that just read input, compute, and
        print — in practice this is most of this dataset's solutions). Rather
        than return a sentinel for those, wrap the whole module body in a
        synthetic function so cc_visit has something to walk, and measure the
        program's actual branching complexity as a single unit.

        This only changes the *parsed structure* radon sees (one extra level of
        indentation) — it doesn't execute the code, so it's safe even for code
        containing multi-line string literals; any leading whitespace added
        inside a string's content has no effect on complexity.

        Returns -1 if the code cannot be parsed at all (real code, not the
        wrapped version). -2 is still possible for the rare case where even the
        wrapped code yields no measurable blocks (e.g. inconsistent
        tabs/spaces prevents wrapping from parsing).
        """
        try:
            blocks = cc_visit(code)
        except Exception:
            return -1

        if not blocks:
            try:
                wrapped = "def _synthetic_wrapper():\n" + textwrap.indent(code, "    ")
                blocks = cc_visit(wrapped)
            except Exception:
                blocks = []
            if not blocks:
                return -2

        return round(sum(b.complexity for b in blocks) / len(blocks), 2)


    def compute_metrics(code: str | None) -> dict:
        """
        Compute static metrics (NCLOC, CCN) for a code block.
        Returns a dict with ncloc and cyclomatic_complexity.
        """
        if not code:
            return {"ncloc": -1, "cyclomatic_complexity": -1}
        return {
            "ncloc": compute_ncloc(code),
            "cyclomatic_complexity": compute_cyclomatic_complexity(code),
        }


    # ---------------------------------------------------------------------------
    # Extraction
    # ---------------------------------------------------------------------------

    def compute_extractable_code(response: str | None) -> bool:
        """
        Return True if a Python code block can be extracted from the response.
        """
        if not response:
            return False
        return extract_code(response) is not None


    # ---------------------------------------------------------------------------
    # Test pass metrics
    # ---------------------------------------------------------------------------

    def compute_passes_tests(test_results: list[tuple]) -> bool | None:
        """
        Return True if ALL test cases passed, False if any failed.
        Each element of test_results is a tuple whose first element is a bool.
        Returns None when the list is empty.
        """
        if not test_results:
            return None
        return all(result[0] for result in test_results)


    def compute_passes_same_tests(
        source_results: list[tuple],
        followup_results: list[tuple],
    ) -> bool | None:
        """
        Return True if the followup passes every test that the source passed.

        A followup that regresses on any individually-passing source test returns
        False, even if its overall pass count is the same or higher.
        Returns None when either list is empty or lengths differ.
        """
        if not source_results or not followup_results:
            return None
        if len(source_results) != len(followup_results):
            return None
        for (src_passed, *_), (flw_passed, *_) in zip(source_results, followup_results):
            if src_passed and not flw_passed:
                return False
        return True


    # ---------------------------------------------------------------------------
    # Token metrics
    # ---------------------------------------------------------------------------

    def compute_token_count(usage: dict | None) -> dict:
        """
        Extract token counts from a usage dict.
        Returns prompt, completion, total, and thinking token counts.
        All values are None when usage is not available.
        """
        if not usage:
            return {
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
                "thinking_tokens": None,
            }
        return {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "thinking_tokens": usage.get("thinking_tokens"),
        }


    def compute_token_efficiency(usage: dict | None) -> float | None:
        """
        Token efficiency = completion_tokens / thinking_tokens.

        Measures how many output tokens were produced per thinking/reasoning token
        spent. A higher value means more output relative to internal reasoning cost.
        Returns None when thinking tokens are unavailable or zero (non-reasoning
        models, or models that didn't use extended thinking on this call).
        """
        if not usage:
            return None
        thinking = usage.get("thinking_tokens")
        completion = usage.get("completion_tokens")
        if not thinking or not completion:
            return None
        return round(completion / thinking, 6)

---

## evaluation/paths.py

```text
File: evaluation/paths.py (1759 bytes)
```

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


    def build_repeats_dir(model_name: str) -> Path:
        """Folder holding one result file per --repeats run, inside the base folder."""
        base_dir = build_results_path(model_name, None, None).parent
        repeats_dir = base_dir / "repeats"
        repeats_dir.mkdir(parents=True, exist_ok=True)
        return repeats_dir


    def build_repeat_path(model_name: str, repeat_index: int) -> Path:
        """Path for a single repeat's result file. repeat_index is 0-based."""
        return build_repeats_dir(model_name) / f"repeat-{repeat_index + 1:02d}.json"


---

## evaluation/report.py

```text
File: evaluation/report.py (25598 bytes)
```

    """
    Generates an HTML comparison report from base and followup results files.
    The merged report (all.html) is composed from the verbatim bodies of the
    individual per-run HTML reports, so its styling is identical to theirs.
    """

    from pathlib import Path
    import json
    import re

    from evaluation.paths import safe_name
    from evaluation.metric_violations import check_metric_violations


    def _fmt(value, digits: int = 4) -> str:
        if isinstance(value, float):
            return f"{value:.{digits}f}"
        if value is None or value == -1:
            return "N/A"
        return str(value)


    def _delta(a, b) -> str:
        if isinstance(a, (int, float)) and isinstance(b, (int, float)) and a != -1 and b != -1:
            d = b - a
            return f"{d:+.2f}" if isinstance(d, float) else f"{d:+d}"
        return "N/A"


    def _pct(numerator, denominator) -> str:
        if isinstance(numerator, int) and isinstance(denominator, int) and denominator > 0:
            return f"{round(numerator / denominator * 100)}%"
        return "N/A"


    def _mr_badge(holds: bool | None, metric_violations: dict | None = None) -> str:
        """
        Badge for the overall MR verdict. A row is VIOLATED if either the test
        verdict diverged (holds is False) or any metric breached its configured
        threshold (metric_violations non-empty) — even if the test verdict
        itself still holds.
        """
        metric_violations = metric_violations or {}
        if holds is False or metric_violations:
            names = ", ".join(metric_violations)
            title = f' title="Metric threshold(s) breached: {names}"' if names else ""
            return f'<span class="badge violated"{title}>VIOLATED</span>'
        if holds is True:
            return '<span class="badge holds">HOLDS</span>'
        return '<span class="badge na">N/A</span>'


    def _violation_span(text: str, detail: dict | None) -> str:
        """Wrap a cell's text in a red span when `detail` (a single entry from
        check_metric_violations) indicates that metric breached its threshold."""
        if not detail:
            return text
        thr = detail.get("threshold")
        d = detail.get("delta")
        if d is not None and thr is not None:
            title = f' title="&Delta;{d:+.2f} exceeds allowed threshold of {thr:g}"'
        elif thr is not None:
            title = f' title="below minimum threshold of {thr:g}"'
        else:
            title = ""
        return f'<span class="delta-neg"{title}>{text}</span>'


    def _pass_badge(passed: bool | None) -> str:
        if passed is True:
            return '<span class="badge holds">PASS</span>'
        if passed is False:
            return '<span class="badge violated">FAIL</span>'
        return '<span class="badge na">N/A</span>'


    def _bool_badge(val: bool | None, true_label: str = "YES", false_label: str = "NO") -> str:
        if val is True:
            return f'<span class="badge holds">{true_label}</span>'
        if val is False:
            return f'<span class="badge violated">{false_label}</span>'
        return '<span class="badge na">N/A</span>'


    def _load(path: Path) -> list[dict]:
        if not path or not path.exists():
            return []
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return []
        if not isinstance(data, list):
            return []
        return [r for r in data if isinstance(r, dict)]


    def _load_by_id(path: Path) -> dict[str, dict]:
        return {
            r["question_id"]: r
            for r in _load(path)
            if isinstance(r.get("question_id"), str) and r["question_id"]
        }


    def _fmt_tokens(prompt, completion, thinking) -> str:
        """Format token counts as prompt+completion (thinking) compactly."""
        parts = []
        if isinstance(prompt, int):
            parts.append(f"P:{prompt:,}")
        if isinstance(completion, int):
            parts.append(f"C:{completion:,}")
        if isinstance(thinking, int) and thinking:
            parts.append(f"T:{thinking:,}")
        return " ".join(parts) if parts else "N/A"


    def _fmt_duration(seconds) -> str:
        if not isinstance(seconds, (int, float)):
            return "N/A"
        seconds = round(seconds, 1)
        hours, remainder = divmod(int(seconds), 3600)
        minutes, secs = divmod(remainder, 60)
        if hours:
            return f"{hours}h {minutes:02d}m {secs:02d}s"
        if minutes:
            return f"{minutes}m {secs:02d}s"
        return f"{seconds:.1f}s"


    def _row_total_time(b: dict, f: dict, has_mr: bool) -> float | None:
        parts = [b.get("llm_call_time"), b.get("test_run_time")]
        if has_mr:
            parts += [f.get("llm_call_time"), f.get("test_run_time")]
        nums = [p for p in parts if isinstance(p, (int, float))]
        return sum(nums) if nums else None


    def _relation_name_from_folder(folder_name: str) -> str:
        """Extract the MR relation name from a folder like 'mr-04-persona_role_prompting'."""
        if folder_name.startswith("mr-"):
            parts = folder_name.split("-", 2)
            if len(parts) == 3:
                return parts[2]
        return folder_name


    def _build_section(base: dict, followup: dict, relation_name: str | None) -> dict:
        """
        Render the stats boxes and data table for one relation run.

        base and followup are dicts keyed by question_id (from base.results.json /
        followup.results.json). Returns a dict with:
            stats_html, table_headers, rows_html
        """
        has_mr = bool(followup) and bool(relation_name)
        all_ids = list(base.keys()) or list(followup.keys())

        # Per-row metric-threshold violations (config.settings.METRIC_VARIANCE_THRESHOLDS),
        # computed once and reused below for both the aggregate stat boxes and the
        # per-row cell highlighting, so the two never disagree on what "violated" means.
        row_metric_violations: dict[str, dict] = {}
        if has_mr:
            for qid in all_ids:
                mv = check_metric_violations(base.get(qid, {}), followup.get(qid, {}))
                if mv:
                    row_metric_violations[qid] = mv

        # ── Aggregate stats ──────────────────────────────────────────────────────
        base_total = len(base)
        base_passed = sum(1 for r in base.values() if r.get("passed"))
        base_failed = base_total - base_passed
        base_extractable = sum(1 for r in base.values() if r.get("extractable_code"))

        flw_total = len(followup)
        flw_passed = sum(1 for r in followup.values() if r.get("passed"))
        flw_failed = flw_total - flw_passed
        flw_extractable = sum(1 for r in followup.values() if r.get("extractable_code"))

        # A row now counts as holding only if the test verdict held AND no metric
        # breached its threshold — previously this only looked at mr_holds, so
        # metric-only violations were silently counted as "holds".
        mr_holds = sum(
            1 for qid in all_ids
            if followup.get(qid, {}).get("mr_holds") is True and qid not in row_metric_violations
        )
        mr_violated = sum(
            1 for qid in all_ids
            if followup.get(qid, {}).get("mr_holds") is False or qid in row_metric_violations
        )
        metric_violated_count = len(row_metric_violations)

        same_tests_count = sum(1 for qid in all_ids if followup.get(qid, {}).get("passes_same_tests") is True)
        diff_tests_count = sum(1 for qid in all_ids if followup.get(qid, {}).get("passes_same_tests") is False)

        def _sum_time(rows: dict, key: str) -> float:
            return sum(r[key] for r in rows.values() if isinstance(r.get(key), (int, float)))

        total_time = (
            _sum_time(base, "llm_call_time") + _sum_time(base, "test_run_time")
            + _sum_time(followup, "llm_call_time") + _sum_time(followup, "test_run_time")
        )

        # ── Build rows ───────────────────────────────────────────────────────────
        rows_html = ""
        for qid in all_ids:
            b = base.get(qid, {})
            f = followup.get(qid, {})

            title = b.get("question_title") or f.get("question_title") or qid
            difficulty = b.get("difficulty") or f.get("difficulty") or ""
            platform = b.get("platform") or f.get("platform") or ""

            # Test results
            b_passed = b.get("passed")
            f_passed = f.get("passed") if has_mr else None
            mr_holds_val = f.get("mr_holds") if has_mr else None
            passes_same = f.get("passes_same_tests") if has_mr else None
            metric_violations = row_metric_violations.get(qid, {}) if has_mr else {}

            b_tests = f'{b.get("tests_passed", "?")}/{b.get("tests_total", "?")}'
            f_tests = f'{f.get("tests_passed", "?")}/{f.get("tests_total", "?")}' if has_mr else "N/A"

            # Static metrics
            b_ncloc = b.get("ncloc", -1)
            b_cc = b.get("cyclomatic_complexity", -1)
            f_ncloc = f.get("ncloc", -1) if has_mr else None
            f_cc = f.get("cyclomatic_complexity", -1) if has_mr else None

            # Similarity / variability
            code_sim = f.get("code_sim") if has_mr else None
            b_mean_dist = b.get("mean_distance")
            f_mean_dist = f.get("mean_distance") if has_mr else None
            b_std_dist = b.get("std_distance")
            f_std_dist = f.get("std_distance") if has_mr else None

            # Tokens
            b_tok = _fmt_tokens(b.get("prompt_tokens"), b.get("completion_tokens"), b.get("thinking_tokens"))
            f_tok = _fmt_tokens(f.get("prompt_tokens"), f.get("completion_tokens"), f.get("thinking_tokens")) if has_mr else "N/A"
            b_eff = b.get("token_efficiency")
            f_eff = f.get("token_efficiency") if has_mr else None

            # Timing
            b_llm_t = f'{b.get("llm_call_time"):.1f}s' if isinstance(b.get("llm_call_time"), (int, float)) else "N/A"
            f_llm_t = f'{f.get("llm_call_time"):.1f}s' if has_mr and isinstance(f.get("llm_call_time"), (int, float)) else "N/A"
            b_test_t = f'{b.get("test_run_time"):.2f}s' if isinstance(b.get("test_run_time"), (int, float)) else "N/A"
            f_test_t = f'{f.get("test_run_time"):.2f}s' if has_mr and isinstance(f.get("test_run_time"), (int, float)) else "N/A"

            row_total_time = _fmt_duration(_row_total_time(b, f, has_mr))

            # Highlight cells for metrics that breached their configured threshold
            # (config.settings.METRIC_VARIANCE_THRESHOLDS).
            delta_ncloc_html = _violation_span(_delta(b_ncloc, f_ncloc), metric_violations.get("ncloc"))
            delta_cc_html = _violation_span(_delta(b_cc, f_cc), metric_violations.get("cyclomatic_complexity"))
            code_sim_html = _violation_span(_fmt(code_sim), metric_violations.get("code_sim"))
            f_tok_html = _violation_span(
                f_tok, metric_violations.get("total_tokens") or metric_violations.get("token_efficiency")
            )
            f_llm_t_html = _violation_span(f_llm_t, metric_violations.get("llm_call_time"))

            if has_mr:
                rows_html += f"""
                <tr>
                    <td>
                        <code>{qid}</code><br>
                        <small>{title}</small><br>
                        <span class="tag">{difficulty}</span>
                        <span class="tag">{platform}</span>
                    </td>
                    <td>{b_tests}<br>{_pass_badge(b_passed)}</td>
                    <td>{f_tests}<br>{_pass_badge(f_passed)}</td>
                    <td>{_mr_badge(mr_holds_val, metric_violations)}<br>{_bool_badge(passes_same, "SAME", "DIFF")}</td>
                    <td>{_fmt(b_ncloc, 0)}</td>
                    <td>{_fmt(f_ncloc, 0) if f_ncloc is not None else "N/A"}</td>
                    <td>{delta_ncloc_html}</td>
                    <td>{_fmt(b_cc)}</td>
                    <td>{_fmt(f_cc) if f_cc is not None else "N/A"}</td>
                    <td>{delta_cc_html}</td>
                    <td>{code_sim_html}</td>
                    <td>{_fmt(b_mean_dist)}<br><small>±{_fmt(b_std_dist)}</small></td>
                    <td>{_fmt(f_mean_dist)}<br><small>±{_fmt(f_std_dist)}</small></td>
                    <td>{b_tok}<br><small>eff:{_fmt(b_eff, 3)}</small></td>
                    <td>{f_tok_html}<br><small>eff:{_fmt(f_eff, 3)}</small></td>
                    <td>{b_llm_t}<br><small>test:{b_test_t}</small></td>
                    <td>{f_llm_t_html}<br><small>test:{f_test_t}</small></td>
                    <td>{row_total_time}</td>
                </tr>"""
            else:
                rows_html += f"""
                <tr>
                    <td>
                        <code>{qid}</code><br>
                        <small>{title}</small><br>
                        <span class="tag">{difficulty}</span>
                        <span class="tag">{platform}</span>
                    </td>
                    <td>{b_tests}<br>{_pass_badge(b_passed)}</td>
                    <td>{_fmt(b_ncloc, 0)}</td>
                    <td>{_fmt(b_cc)}</td>
                    <td>{_fmt(b_mean_dist)}<br><small>±{_fmt(b_std_dist)}</small></td>
                    <td>{b_tok}<br><small>eff:{_fmt(b_eff, 3)}</small></td>
                    <td>{b_llm_t}<br><small>test:{b_test_t}</small></td>
                </tr>"""

        # ── Aggregate stat boxes ─────────────────────────────────────────────────
        base_pct = round(base_passed / base_total * 100) if base_total else 0
        extract_pct = round(base_extractable / base_total * 100) if base_total else 0

        mr_section = ""
        followup_stat = ""
        same_tests_stat = ""

        if has_mr:
            flw_pct = round(flw_passed / flw_total * 100) if flw_total else 0
            flw_extract_pct = round(flw_extractable / flw_total * 100) if flw_total else 0
            mr_total = mr_holds + mr_violated
            hold_pct = round(mr_holds / mr_total * 100) if mr_total else 0
            same_pct = round(same_tests_count / mr_total * 100) if mr_total else 0

            followup_stat = f"""
            <div class="stat-box">
                <div class="stat-label">Followup pass rate</div>
                <div class="stat-value">{flw_pct}%</div>
                <div class="stat-sub">{flw_passed} passed / {flw_failed} failed / {flw_total} total</div>
                <div class="stat-sub">Extractable: {flw_extract_pct}% ({flw_extractable}/{flw_total})</div>
            </div>"""

            mr_section = f"""
            <div class="stat-box">
                <div class="stat-label">MR [{relation_name}]</div>
                <div class="stat-value">{hold_pct}% hold rate</div>
                <div class="stat-sub">{mr_holds} holds / {mr_violated} violations / {mr_total} total</div>
                <div class="stat-sub">Of which {metric_violated_count} breached a metric threshold</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Passes same tests</div>
                <div class="stat-value">{same_pct}%</div>
                <div class="stat-sub">{same_tests_count} same / {diff_tests_count} different</div>
            </div>"""

        total_time_stat = f"""
        <div class="stat-box">
            <div class="stat-label">Total time</div>
            <div class="stat-value">{_fmt_duration(total_time)}</div>
            <div class="stat-sub">LLM calls + test execution, all rows</div>
        </div>"""

        stats_html = f"""
            <div class="stat-box">
                <div class="stat-label">Base pass rate</div>
                <div class="stat-value">{base_pct}%</div>
                <div class="stat-sub">{base_passed} passed / {base_failed} failed / {base_total} total</div>
                <div class="stat-sub">Extractable: {extract_pct}% ({base_extractable}/{base_total})</div>
            </div>
            {followup_stat}
            {mr_section}
            {same_tests_stat}
            {total_time_stat}"""

        # ── Table headers ────────────────────────────────────────────────────────
        if has_mr:
            table_headers = """
                <th>Problem</th>
                <th>Base tests</th>
                <th>Followup tests</th>
                <th>MR / same tests</th>
                <th>Base NCLOC</th>
                <th>Flw NCLOC</th>
                <th>ΔNCLOC</th>
                <th>Base CC</th>
                <th>Flw CC</th>
                <th>ΔCC</th>
                <th>Code sim</th>
                <th>Base dist ±std</th>
                <th>Flw dist ±std</th>
                <th>Base tokens / eff</th>
                <th>Flw tokens / eff</th>
                <th>Base LLM / test time</th>
                <th>Flw LLM / test time</th>
                <th>Total time</th>"""
        else:
            table_headers = """
                <th>Problem</th>
                <th>Tests</th>
                <th>NCLOC</th>
                <th>CC</th>
                <th>Mean dist ±std</th>
                <th>Tokens / eff</th>
                <th>LLM / test time</th>"""

        return {
            "stats_html": stats_html,
            "table_headers": table_headers,
            "rows_html": rows_html,
        }


    _CSS = """
      body { font-family: system-ui, sans-serif; margin: 2rem; background: #f8f9fa; color: #212529; font-size: 0.9rem; }
      h1 { font-size: 1.4rem; margin-bottom: 0.25rem; }
      .subtitle { color: #6c757d; font-size: 0.85rem; margin-bottom: 1.5rem; }

      .stats { display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 2rem; }
      .stat-box { background: white; border: 1px solid #dee2e6; border-radius: 8px; padding: 1rem 1.5rem; min-width: 170px; }
      .stat-label { font-size: 0.72rem; color: #6c757d; text-transform: uppercase; letter-spacing: 0.05em; }
      .stat-value { font-size: 1.6rem; font-weight: 700; margin: 0.25rem 0; }
      .stat-sub { font-size: 0.78rem; color: #6c757d; }

      .table-wrap { overflow-x: auto; }
      table { width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); white-space: nowrap; }
      th { background: #343a40; color: white; padding: 0.5rem 0.7rem; text-align: left; font-size: 0.75rem; }
      td { padding: 0.5rem 0.7rem; border-bottom: 1px solid #dee2e6; font-size: 0.8rem; vertical-align: top; white-space: nowrap; }
      td:first-child { white-space: normal; min-width: 180px; }
      tr:last-child td { border-bottom: none; }
      tr:hover td { background: #f1f3f5; }

      code { font-size: 0.75rem; color: #e83e8c; }
      small { color: #6c757d; font-size: 0.72rem; }
      .tag { display: inline-block; background: #e9ecef; border-radius: 4px; padding: 0.1rem 0.4rem; font-size: 0.7rem; color: #495057; margin-right: 2px; }

      .badge { display: inline-block; border-radius: 4px; padding: 0.12rem 0.45rem; font-size: 0.72rem; font-weight: 600; }
      .holds { background: #d1e7dd; color: #0f5132; }
      .violated { background: #f8d7da; color: #842029; }
      .na { background: #e9ecef; color: #6c757d; }

      .delta-pos { color: #198754; }
      .delta-neg { color: #dc3545; font-weight: 700; cursor: help; }
    """


    def _render_document(model_name: str, relation_name: str | None, section: dict) -> str:
        if relation_name:
            subtitle = f"MR: {relation_name}"
        else:
            subtitle = "Base run — no MR applied"

        content = (
            '<div class="stats">\n'
            f'{section["stats_html"]}\n'
            "</div>\n"
            '<div class="table-wrap">\n'
            "<table>\n"
            f'  <thead><tr>{section["table_headers"]}</tr></thead>\n'
            f'  <tbody>{section["rows_html"]}</tbody>\n'
            "</table>\n"
            "</div>"
        )

        return f"""<!DOCTYPE html>
    <html lang="en">
    <head>
    <meta charset="UTF-8">
    <title>MT Report — {model_name}</title>
    <style>
    {_CSS}
    </style>
    </head>
    <body>
    <h1>MT Report — {model_name}</h1>
    <div class="subtitle">{subtitle}</div>

    {content}
    </body>
    </html>"""


    def _write_html_report(
        base_results_path: Path,
        followup_results_path: Path | None,
        out_path: Path,
        model_name: str,
        relation_name: str | None,
    ) -> None:
        base = _load_by_id(base_results_path)
        followup = _load_by_id(followup_results_path) if followup_results_path else {}
        section = _build_section(base, followup, relation_name)
        html = _render_document(model_name, relation_name, section)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html)


    def generate_html_report(
        base_results_path: Path,
        followup_results_path: Path | None,
        out_path: Path,
        model_name: str,
        relation_name: str | None,
    ) -> None:
        _write_html_report(base_results_path, followup_results_path, out_path, model_name, relation_name)
        print(f"  HTML report -> {out_path}")

    _MERGED_CSS = """
      .toc { background: white; border: 1px solid #dee2e6; border-radius: 8px; padding: 1rem 1.5rem; margin-bottom: 2rem; }
      .toc strong { font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; color: #6c757d; }
      .toc ul { columns: 3; column-gap: 2rem; margin: 0.6rem 0 0; padding-left: 1.1rem; list-style: disc; }
      .toc li { margin-bottom: 0.3rem; font-size: 0.85rem; break-inside: avoid; }
      .toc a { color: #0d6efd; text-decoration: none; }
      .toc a:hover { text-decoration: underline; }

      .report-section { margin-top: 3rem; padding-top: 1.75rem; border-top: 3px solid #343a40; }
      .report-section:first-of-type { margin-top: 0; padding-top: 0; border-top: none; }
      .report-section h2 { font-size: 1.2rem; margin: 0 0 1rem; color: #212529; }
    """


    def generate_merged_html_report(
        model_name: str,
        results_dir: str,
        out_path: Path | None = None,
    ) -> Path:
        """
        Build a single merged report (results/<model>/all.html) with one clearly
        labeled, visually separated section per relation (base + every mr-*),
        plus a table of contents linking to each section.

        Sections are built directly from each run's results JSON via
        _build_section(), the same function individual per-run reports use, so
        structure and styling stay identical and in sync automatically. This
        does NOT read back already-rendered HTML files, unlike the previous
        approach, which regex-extracted and stripped headers from individual
        report bodies and left the merged page with no section markers at all.
        """
        model_dir = Path(results_dir) / safe_name(model_name)
        out_path = out_path or model_dir / "all.html"

        base_results = model_dir / "base" / "base.results.json"
        base = _load_by_id(base_results) if base_results.exists() else {}

        # Still write the individual per-run HTML reports as standalone artifacts
        # (base.html, mr-XX-name.html) — just no longer used as an input here.
        if base_results.exists():
            _write_html_report(base_results, None, model_dir / "base" / "base.html", model_name, None)

        sections: list[tuple[str, str, dict]] = []  # (anchor_id, label, section_dict)

        if base:
            sections.append(("base", "Base (no MR)", _build_section(base, {}, None)))

        if model_dir.exists():
            for folder in sorted(p for p in model_dir.iterdir() if p.is_dir() and p.name.startswith("mr-")):
                flw_results = folder / "followup.results.json"
                if not flw_results.exists():
                    continue
                followup = _load_by_id(flw_results)
                if not followup:
                    continue
                relation_name = _relation_name_from_folder(folder.name)

                # Still write the individual per-run report as a standalone artifact.
                _write_html_report(
                    base_results, flw_results, folder / f"{folder.name}.html", model_name, relation_name
                )

                section = _build_section(base, followup, relation_name)
                sections.append((folder.name, f"MR: {relation_name}", section))

        if not sections:
            print(f"  No results found to merge for {model_name}")
            return out_path

        # ── Table of contents ────────────────────────────────────────────────────
        toc_items = "\n".join(
            f'    <li><a href="#{anchor}">{label}</a></li>'
            for anchor, label, _ in sections
        )
        toc_html = f'<nav class="toc"><strong>Sections</strong><ul>\n{toc_items}\n  </ul></nav>'

        # ── Section bodies ───────────────────────────────────────────────────────
        section_blocks = []
        for anchor, label, section in sections:
            section_blocks.append(f"""
    <section id="{anchor}" class="report-section">
      <h2>{label}</h2>
      <div class="stats">
        {section["stats_html"]}
      </div>
      <div class="table-wrap">
        <table>
          <thead><tr>{section["table_headers"]}</tr></thead>
          <tbody>{section["rows_html"]}</tbody>
        </table>
      </div>
    </section>""")

        content = "\n".join(section_blocks)

        merged = f"""<!DOCTYPE html>
    <html lang="en">
    <head>
    <meta charset="UTF-8">
    <title>MT Report — {model_name}</title>
    <style>
    {_CSS}
    {_MERGED_CSS}
    </style>
    </head>
    <body>
    <h1>MT Report — {model_name}</h1>
    {toc_html}
    {content}
    </body>
    </html>"""

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(merged)
        print(f"  Merged HTML report -> {out_path}")
        return out_path

---

## evaluation/results_writer.py

```text
File: evaluation/results_writer.py (14743 bytes)
```

    """
    Writes split base/followup content files and results JSON files,
    then triggers HTML report generation.
    """

    import json
    import time
    from pathlib import Path

    from evaluation.report import generate_html_report
    from evaluation.paths import build_results_dir, mr_folder_name, safe_name
    from evaluation.metric_violations import check_metric_violations


    def _merge_by_question_id(existing_path: Path, new_entries: list[dict]) -> list[dict]:
        """Merge new_entries into whatever is already at existing_path, keyed by
        question_id. New entries replace existing ones with the same id; anything
        else already on disk is preserved."""
        existing: list[dict] = []
        if existing_path.exists():
            try:
                with open(existing_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, list):
                    existing = [e for e in loaded if isinstance(e, dict)]
            except (OSError, json.JSONDecodeError):
                existing = []

        incoming_ids = {e.get("question_id") for e in new_entries}
        return [e for e in existing if e.get("question_id") not in incoming_ids] + new_entries


    def write_base_content(results: list[dict], model_name: str, results_dir: str) -> Path:
        """
        Write base LLM content to results/<model>/base/base.json.
        Strips followup fields. Merges into any existing base.json rather
        than overwriting it, since `results` here may only be a subset
        (e.g. a single --mr run's questions), while base.json is meant to
        be a running ledger across all runs.
        """
        FOLLOWUP_FIELDS = {
            "relation", "followup_instruction_prefix", "followup_instruction_suffix",
            "followup_question", "followup_response", "followup_response_samples",
            "followup_error", "followup_prompt_tokens", "followup_completion_tokens",
            "followup_total_tokens", "followup_thinking_tokens", "followup_token_efficiency",
            "followup_llm_call_time",
        }
        out_dir = Path(results_dir) / safe_name(model_name) / "base"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "base.json"

        stripped = [
            {k: v for k, v in entry.items() if k not in FOLLOWUP_FIELDS}
            for entry in results
        ]

        merged = _merge_by_question_id(out_path, stripped)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)
        print(f"  Base content  -> {out_path}")
        return out_path


    def write_followup_content(
        results: list[dict],
        model_name: str,
        relation_name: str,
        relation_id: int | None,
        results_dir: str,
    ) -> Path:
        """
        Write followup LLM content to results/<model>/mr-XX-name/followup.json.
        Keeps question context + followup fields only.
        """
        KEEP_FIELDS = {
            "question_id", "question_title", "difficulty", "platform", "model_name",
            "question", "public_test_cases", "private_test_cases", "metadata",
            "relation", "followup_instruction_prefix", "followup_instruction_suffix",
            "followup_question", "followup_response", "followup_response_samples",
            "followup_error", "followup_prompt_tokens", "followup_completion_tokens",
            "followup_total_tokens", "followup_thinking_tokens", "followup_token_efficiency",
            "followup_llm_call_time",
        }
        folder = mr_folder_name(relation_name, relation_id)
        out_dir = Path(results_dir) / safe_name(model_name) / folder
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "followup.json"

        filtered = [
            {k: v for k, v in entry.items() if k in KEEP_FIELDS}
            for entry in results
            if entry.get("followup_response")
        ]
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(filtered, f, indent=2, ensure_ascii=False)
        print(f"  Followup content -> {out_path}")
        return out_path


    def write_results_json(metric_rows: list[dict], out_path: Path, merge: bool = False) -> None:
        """Write per-problem metrics and pass/fail counts to a results JSON file.

        If merge=True, merges into any existing file at out_path by question_id
        instead of overwriting it outright — used for base.results.json, which
        is a running ledger across all --mr runs, not a per-run artifact.
        """
        rows = _merge_by_question_id(out_path, metric_rows) if merge else metric_rows
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)
        print(f"  Results JSON  -> {out_path}")


    def write_all_outputs(
        results: list[dict],
        metric_rows: list[dict],
        model_name: str,
        relation_name: str | None,
        relation_id: int | None,
        results_dir: str,
    ) -> None:
        """
        Write all split output files and generate the HTML report.
        Called at the end of runtests.py when --run-tests is active.
        """
        has_mr = bool(relation_name)

        # Write content files
        base_content_path = write_base_content(results, model_name, results_dir)
        followup_content_path = None
        if has_mr:
            followup_content_path = write_followup_content(
                results, model_name, relation_name, relation_id, results_dir
            )

        # ── Base results JSON ────────────────────────────────────────────────────
        base_results_path = base_content_path.parent / "base.results.json"
        base_rows = []
        for row in metric_rows:
            src_var = row.get("source_variability") or {}
            base_rows.append({
                "question_id": row["id"],
                "question_title": row.get("question_title"),
                "difficulty": row.get("difficulty"),
                "platform": row.get("platform"),

                # Test pass metrics
                "extractable_code": row.get("source_extractable"),
                "passes_tests": row.get("source_passes_tests"),
                "passed": row.get("source_passed"),          # legacy alias
                "tests_passed": row.get("tests_passed"),
                "tests_failed": row.get("tests_failed"),
                "tests_total": row.get("tests_total"),
                "test_run_time": row.get("source_test_run_time"),

                # Static code metrics
                "ncloc": row.get("source_ncloc"),
                "cyclomatic_complexity": row.get("source_cc"),

                # Variability
                "mean_distance": src_var.get("mean_distance"),
                "std_distance": src_var.get("std_distance"),
                "n_pairs": src_var.get("n_pairs"),

                # Token metrics
                "prompt_tokens": row.get("source_prompt_tokens"),
                "completion_tokens": row.get("source_completion_tokens"),
                "total_tokens": row.get("source_total_tokens"),
                "thinking_tokens": row.get("source_thinking_tokens"),
                "token_efficiency": row.get("source_token_efficiency"),

                # Timing
                "llm_call_time": row.get("source_llm_call_time"),
            })
        write_results_json(base_rows, base_results_path, merge=True)

        # ── Followup results JSON ────────────────────────────────────────────────
        followup_results_path = None
        if has_mr:
            folder = mr_folder_name(relation_name, relation_id)
            followup_results_path = (
                Path(results_dir) / safe_name(model_name) / folder / "followup.results.json"
            )
            followup_rows = []
            for row in metric_rows:
                if row.get("followup_ncloc") is None:
                    continue
                flw_var = row.get("followup_variability") or {}
                followup_rows.append({
                    "question_id": row["id"],
                    "question_title": row.get("question_title"),
                    "difficulty": row.get("difficulty"),
                    "platform": row.get("platform"),

                    # Test pass metrics
                    "extractable_code": row.get("followup_extractable"),
                    "passes_tests": row.get("followup_passes_tests"),
                    "passes_same_tests": row.get("passes_same_tests"),
                    "passed": row.get("followup_passed"),    # legacy alias
                    "tests_passed": row.get("followup_tests_passed"),
                    "tests_failed": row.get("followup_tests_failed"),
                    "tests_total": row.get("followup_tests_total"),
                    "test_run_time": row.get("followup_test_run_time"),

                    # Static code metrics
                    "ncloc": row.get("followup_ncloc"),
                    "cyclomatic_complexity": row.get("followup_cc"),

                    # Variability
                    "mean_distance": flw_var.get("mean_distance"),
                    "std_distance": flw_var.get("std_distance"),
                    "n_pairs": flw_var.get("n_pairs"),

                    # MR
                    "mr_holds": row.get("mr_holds"),
                    "code_sim": row.get("code_sim"),

                    # Token metrics
                    "prompt_tokens": row.get("followup_prompt_tokens"),
                    "completion_tokens": row.get("followup_completion_tokens"),
                    "total_tokens": row.get("followup_total_tokens"),
                    "thinking_tokens": row.get("followup_thinking_tokens"),
                    "token_efficiency": row.get("followup_token_efficiency"),

                    # Timing
                    "llm_call_time": row.get("followup_llm_call_time"),
                })
            write_results_json(followup_rows, followup_results_path)

        # ── HTML report ──────────────────────────────────────────────────────────
        safe = safe_name(model_name)
        if has_mr:
            out_dir = build_results_dir(model_name, relation_name, relation_id)
            html_path = out_dir / f"{out_dir.name}.html"
        else:
            html_path = Path(results_dir) / safe / "base" / "base.html"

        generate_html_report(
            base_results_path=base_results_path,
            followup_results_path=followup_results_path,
            out_path=html_path,
            model_name=model_name,
            relation_name=relation_name,
        )


    def write_violations_log(model_name: str, results_dir: str) -> Path:
        """
        Scan each per-MR followup results file for violated metamorphic relations,
        where a violation is either a failed test-verdict match (mr_holds == False)
        or a metric delta exceeding its configured threshold (METRIC_VARIANCE_THRESHOLDS).
        Source-side metrics are joined from the base running ledger by question_id.

        The log is JSONL (one object per line), matching the helper_call_log
        format, with a timestamp first on each record.
        """
        model_dir = Path(results_dir) / safe_name(model_name)
        out_path = model_dir / "violations.json"

        base_by_qid: dict[str, dict] = {}
        base_path = model_dir / "base" / "base.results.json"
        if base_path.exists():
            try:
                with open(base_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, list):
                    base_by_qid = {
                        e.get("question_id"): e
                        for e in loaded
                        if isinstance(e, dict) and isinstance(e.get("question_id"), str) and e["question_id"]
                    }
            except (OSError, json.JSONDecodeError):
                base_by_qid = {}

        violations: list[dict] = []
        if model_dir.exists():
            for folder in sorted(p for p in model_dir.iterdir() if p.is_dir() and p.name.startswith("mr-")):
                flw_path = folder / "followup.results.json"
                if not flw_path.exists():
                    continue
                try:
                    with open(flw_path, "r", encoding="utf-8") as f:
                        rows = json.load(f)
                except (OSError, json.JSONDecodeError):
                    continue
                if not isinstance(rows, list):
                    continue
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    qid = row.get("question_id")
                    b = base_by_qid.get(qid, {})

                    test_violation = row.get("mr_holds") is False
                    metric_violations = check_metric_violations(b, row)

                    if not test_violation and not metric_violations:
                        continue

                    violations.append({
                        "model": model_name,
                        "relation": folder.name,
                        "question_id": qid,
                        "question_title": row.get("question_title"),
                        "difficulty": row.get("difficulty"),
                        "platform": row.get("platform"),
                        "source_passed": b.get("passed"),
                        "source_tests_passed": b.get("tests_passed"),
                        "source_tests_failed": b.get("tests_failed"),
                        "source_tests_total": b.get("tests_total"),
                        "followup_passed": row.get("passed"),
                        "followup_tests_passed": row.get("tests_passed"),
                        "followup_tests_failed": row.get("tests_failed"),
                        "followup_tests_total": row.get("tests_total"),
                        "passes_same_tests": row.get("passes_same_tests"),
                        "code_sim": row.get("code_sim"),
                        "test_violation": test_violation,
                        "metric_violations": metric_violations,
                    })

        violations.sort(key=lambda v: (v.get("relation") or "", v.get("question_id") or ""))
        model_dir.mkdir(parents=True, exist_ok=True)
        now = time.time()
        with open(out_path, "w", encoding="utf-8") as f:
            for v in violations:
                f.write(json.dumps({"timestamp": now, **v}, ensure_ascii=False) + "\n")
        print(f"  Violations log -> {out_path} ({len(violations)} violations)")
        return out_path


    def write_consolidated_outputs(model_name: str, results_dir: str) -> None:
        """
        Write the post-test consolidated outputs for a model:
          - merged HTML report of base + all per-MR runs
          - violations log
        Safe to call repeatedly; reads whatever results exist on disk.
        """
        from evaluation.report import generate_merged_html_report

        generate_merged_html_report(model_name, results_dir)
        write_violations_log(model_name, results_dir)

---

## evaluation/runner.py

```text
File: evaluation/runner.py (19416 bytes)
```

    import json
    import time
    from dataclasses import dataclass, asdict, field
    from typing import List
    from pathlib import Path

    from llms.base import BaseLLM
    from data.loader import build_code_question, build_prompt, INSTRUCTION_PREFIX, INSTRUCTION_SUFFIX
    from evaluation.executor import extract_code
    from evaluation.paths import build_results_path, build_repeat_path, build_repeats_dir
    from config.settings import RESULTS_DIR


    def _load_base_results(model_name: str) -> dict[str, dict]:
        base_path = build_results_path(model_name, None, None)
        if not base_path.exists():
            print(f"  Base results not found for MR-only run: {base_path}")
            return {}
        try:
            with open(base_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            print(f"  Failed to read base results for MR-only run: {base_path}")
            return {}
        if not isinstance(data, list):
            print(f"  Unexpected base results format (expected list): {base_path}")
            return {}
        return {
            qid: entry
            for entry in data
            if isinstance(entry, dict)
            and isinstance((qid := entry.get("question_id")), str)
            and qid
        }


    def _load_existing_results(path: Path) -> list[dict]:
        if not path.exists():
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            print(f"  Existing results could not be read, starting fresh: {path}")
            return []
        if not isinstance(data, list):
            print(f"  Existing results are not a list, starting fresh: {path}")
            return []
        return [entry for entry in data if isinstance(entry, dict)]


    def _entry_has_code(entry: dict, response_key: str) -> bool:
        response = entry.get(response_key, "")
        return isinstance(response, str) and extract_code(response) is not None


    def _entry_complete_for_resume(entry: dict, relation=None, mr_only: bool = False) -> bool:
        if not isinstance(entry.get("question_id"), str) or not entry["question_id"]:
            return False
        if mr_only:
            return _entry_has_code(entry, "followup_response")
        if relation:
            return _entry_has_code(entry, "response") and _entry_has_code(entry, "followup_response")
        return _entry_has_code(entry, "response")


    def _extract_usage(usage: dict | None) -> tuple[int | None, int | None, int | None, int | None]:
        """Return (prompt_tokens, completion_tokens, total_tokens, thinking_tokens)."""
        if not usage:
            return None, None, None, None
        return (
            usage.get("prompt_tokens"),
            usage.get("completion_tokens"),
            usage.get("total_tokens"),
            usage.get("thinking_tokens"),
        )


    def _accumulate_usage(totals: dict, usage: dict | None) -> None:
        if not usage:
            return
        for key in ("prompt_tokens", "completion_tokens", "total_tokens", "thinking_tokens"):
            value = usage.get(key)
            if value is not None:
                totals[key] = totals.get(key, 0) + value
                totals["reported"] = True


    @dataclass
    class LLMResult:
        question_id: str
        question_title: str
        difficulty: str
        platform: str
        model_name: str
        instruction_prefix: str
        instruction_suffix: str
        question: str
        response: str
        public_test_cases: str
        private_test_cases: str
        metadata: str
        error: str = ""
        response_samples: list[str] = field(default_factory=list)

        # Token counts — source
        prompt_tokens: int | None = None
        completion_tokens: int | None = None
        total_tokens: int | None = None
        thinking_tokens: int | None = None          # reasoning/thinking budget consumed
        token_efficiency: float | None = None       # completion_tokens / thinking_tokens

        # Timing — source
        llm_call_time: float | None = None          # wall-clock seconds for generate()

        # MT fields — only populated when --mr is used
        relation: str = ""
        followup_instruction_prefix: str = ""
        followup_instruction_suffix: str = ""
        followup_question: str = ""
        followup_response: str = ""
        followup_error: str = ""
        followup_response_samples: list[str] = field(default_factory=list)

        # Token counts — followup
        followup_prompt_tokens: int | None = None
        followup_completion_tokens: int | None = None
        followup_total_tokens: int | None = None
        followup_thinking_tokens: int | None = None
        followup_token_efficiency: float | None = None

        # Timing — followup
        followup_llm_call_time: float | None = None


    class PipelineRunner:
        def __init__(
            self,
            models: List[BaseLLM],
            dataset,
            delay: float = 0.5,
            relation=None,
            relation_id: int | None = None,
            mr_only: bool = False,
            prompt_repeats: int = 1,
            resume: bool = False,
        ):
            self.models = models
            self.dataset = dataset
            self.delay = delay
            self.relation = relation
            self.relation_id = relation_id
            self.mr_only = mr_only
            self.prompt_repeats = max(1, prompt_repeats)
            self.resume = resume
            Path(RESULTS_DIR).mkdir(exist_ok=True)

        def run(self):
            for model in self.models:
                print(f"\n{'='*50}")
                print(f"Running model: {model.model_name}")
                print(f"{'='*50}")

                out_path = build_results_path(
                    model.model_name,
                    self.relation.name if self.relation else None,
                    self.relation_id,
                )
                results = _load_existing_results(out_path) if self.resume else []
                completed_ids = {
                    entry.get("question_id")
                    for entry in results
                    if _entry_complete_for_resume(entry, self.relation, self.mr_only)
                }
                if self.resume and completed_ids:
                    print(f"  Resume enabled: loaded {len(completed_ids)} completed code results from {out_path}")

                token_totals: dict = {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "thinking_tokens": 0,
                    "reported": False,
                }

                base_results = {}
                if self.relation and self.mr_only:
                    base_results = _load_base_results(model.model_name)

                # For a plain base run (no --mr) with --repeats > 1, each repeat is
                # also written out as its own full results file under base/repeats/,
                # so it can be tested and compared independently of the others.
                write_repeat_files = self.relation is None and self.prompt_repeats > 1
                repeat_results: list[list[dict]] = (
                    [[] for _ in range(self.prompt_repeats)] if write_repeat_files else []
                )

                for i, sample in enumerate(self.dataset):
                    if self.resume and sample["question_id"] in completed_ids:
                        print(
                            f"  [{i+1}/{len(self.dataset)}] {sample['question_id']} "
                            f"({sample['difficulty']}) ... skipped (already complete)"
                        )
                        continue

                    question = build_code_question(sample)
                    prompt = build_prompt(question)

                    print(f"  [{i+1}/{len(self.dataset)}] {sample['question_id']} ({sample['difficulty']})", end=" ... ")

                    # ── Source response ──────────────────────────────────────────
                    response = ""
                    response_samples = []
                    usage_samples: list[dict | None] = []
                    call_time_samples: list[float | None] = []
                    error = ""
                    usage = None
                    llm_call_time: float | None = None

                    if not (self.relation and self.mr_only):
                        for repeat_index in range(self.prompt_repeats):
                            try:
                                current_response = model.generate(prompt)
                                current_usage = model.last_usage
                                current_call_time = model.last_call_time
                            except Exception as e:
                                current_response = ""
                                current_usage = None
                                current_call_time = None
                                if repeat_index == 0:
                                    error = str(e)

                            response_samples.append(current_response)
                            usage_samples.append(current_usage)
                            call_time_samples.append(current_call_time)
                            if repeat_index == 0:
                                response = current_response
                                usage = current_usage
                                llm_call_time = current_call_time
                            _accumulate_usage(token_totals, current_usage)
                            if repeat_index < self.prompt_repeats - 1:
                                time.sleep(self.delay)

                    if self.relation and self.mr_only and base_results:
                        base_entry = base_results.get(sample["question_id"])
                        if base_entry:
                            response = base_entry.get("response", "")
                            response_samples = base_entry.get("response_samples") or ([response] if response else [])
                            error = base_entry.get("error", "")
                            llm_call_time = base_entry.get("llm_call_time")

                    prompt_tokens, completion_tokens, total_tokens, thinking_tokens = _extract_usage(usage)
                    token_efficiency: float | None = None
                    if thinking_tokens and completion_tokens:
                        token_efficiency = round(completion_tokens / thinking_tokens, 6)

                    # ── Followup response ────────────────────────────────────────
                    followup_question = ""
                    followup_instruction_prefix = ""
                    followup_instruction_suffix = ""
                    followup_response = ""
                    followup_response_samples = []
                    followup_error = ""
                    followup_prompt_tokens = None
                    followup_completion_tokens = None
                    followup_total_tokens = None
                    followup_thinking_tokens = None
                    followup_token_efficiency = None
                    followup_llm_call_time: float | None = None

                    if self.relation:
                        target = getattr(self.relation, "target", "PD")

                        if target == "PD":
                            followup_question = self.relation.transform(question)
                            followup_instruction_prefix = INSTRUCTION_PREFIX
                            followup_instruction_suffix = INSTRUCTION_SUFFIX
                            followup_prompt = build_prompt(followup_question)
                        elif target == "I":
                            followup_question = question
                            followup_instruction_prefix, followup_instruction_suffix = self.relation.transform(
                                INSTRUCTION_PREFIX, INSTRUCTION_SUFFIX
                            )
                            followup_prompt = build_prompt(question, followup_instruction_prefix, followup_instruction_suffix)

                        for repeat_index in range(self.prompt_repeats):
                            try:
                                current_followup_response = model.generate(followup_prompt)
                                followup_usage = model.last_usage
                                current_followup_call_time = model.last_call_time
                            except Exception as e:
                                current_followup_response = ""
                                followup_usage = None
                                current_followup_call_time = None
                                if repeat_index == 0:
                                    followup_error = str(e)

                            followup_response_samples.append(current_followup_response)
                            if repeat_index == 0:
                                followup_response = current_followup_response
                                (
                                    followup_prompt_tokens,
                                    followup_completion_tokens,
                                    followup_total_tokens,
                                    followup_thinking_tokens,
                                ) = _extract_usage(followup_usage)
                                followup_llm_call_time = current_followup_call_time
                                if followup_thinking_tokens and followup_completion_tokens:
                                    followup_token_efficiency = round(
                                        followup_completion_tokens / followup_thinking_tokens, 6
                                    )
                            _accumulate_usage(token_totals, followup_usage)
                            if repeat_index < self.prompt_repeats - 1:
                                time.sleep(self.delay)

                        if self.mr_only:
                            print(f"ok (mr-only + {self.relation.name} [{target}], repeats={self.prompt_repeats})")
                        else:
                            print(f"ok (+ {self.relation.name} [{target}], repeats={self.prompt_repeats})")
                    else:
                        print(f"ok (repeats={self.prompt_repeats})" if self.prompt_repeats > 1 else "ok")

                    result_entry = asdict(LLMResult(
                        question_id=sample["question_id"],
                        question_title=sample["question_title"],
                        difficulty=sample["difficulty"],
                        platform=sample["platform"],
                        model_name=model.model_name,
                        instruction_prefix=INSTRUCTION_PREFIX,
                        instruction_suffix=INSTRUCTION_SUFFIX,
                        question=question,
                        response=response,
                        response_samples=response_samples,
                        public_test_cases=json.dumps(sample.get("public_test_cases", [])),
                        private_test_cases=sample.get("private_test_cases", ""),
                        metadata=json.dumps(sample.get("metadata", {})),
                        error=error,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens,
                        thinking_tokens=thinking_tokens,
                        token_efficiency=token_efficiency,
                        llm_call_time=llm_call_time,
                        relation=self.relation.name if self.relation else "",
                        followup_instruction_prefix=followup_instruction_prefix,
                        followup_instruction_suffix=followup_instruction_suffix,
                        followup_question=followup_question,
                        followup_response=followup_response,
                        followup_response_samples=followup_response_samples,
                        followup_error=followup_error,
                        followup_prompt_tokens=followup_prompt_tokens,
                        followup_completion_tokens=followup_completion_tokens,
                        followup_total_tokens=followup_total_tokens,
                        followup_thinking_tokens=followup_thinking_tokens,
                        followup_token_efficiency=followup_token_efficiency,
                        followup_llm_call_time=followup_llm_call_time,
                    ))
                    results = [entry for entry in results if entry.get("question_id") != sample["question_id"]]
                    results.append(result_entry)
                    completed_ids.add(sample["question_id"])
                    self._save_to_path(out_path, results, quiet=True)

                    # ── Per-repeat result files (base run only, repeats > 1) ───────
                    if write_repeat_files:
                        for idx, rep_response in enumerate(response_samples):
                            rep_usage = usage_samples[idx] if idx < len(usage_samples) else None
                            rep_ptok, rep_ctok, rep_ttok, rep_thtok = _extract_usage(rep_usage)
                            rep_eff = None
                            if rep_thtok and rep_ctok:
                                rep_eff = round(rep_ctok / rep_thtok, 6)

                            rep_entry = asdict(LLMResult(
                                question_id=sample["question_id"],
                                question_title=sample["question_title"],
                                difficulty=sample["difficulty"],
                                platform=sample["platform"],
                                model_name=model.model_name,
                                instruction_prefix=INSTRUCTION_PREFIX,
                                instruction_suffix=INSTRUCTION_SUFFIX,
                                question=question,
                                response=rep_response,
                                public_test_cases=json.dumps(sample.get("public_test_cases", [])),
                                private_test_cases=sample.get("private_test_cases", ""),
                                metadata=json.dumps(sample.get("metadata", {})),
                                error=error if idx == 0 else "",
                                prompt_tokens=rep_ptok,
                                completion_tokens=rep_ctok,
                                total_tokens=rep_ttok,
                                thinking_tokens=rep_thtok,
                                token_efficiency=rep_eff,
                                llm_call_time=call_time_samples[idx] if idx < len(call_time_samples) else None,
                            ))
                            repeat_results[idx] = [
                                e for e in repeat_results[idx] if e.get("question_id") != sample["question_id"]
                            ]
                            repeat_results[idx].append(rep_entry)
                            self._save_to_path(build_repeat_path(model.model_name, idx), repeat_results[idx], quiet=True)

                    time.sleep(self.delay)

                self._save_to_path(out_path, results)

                if write_repeat_files:
                    print(f"  Saved {self.prompt_repeats} per-repeat result files -> {build_repeats_dir(model.model_name)}")

                if token_totals["reported"]:
                    thinking_total = token_totals.get("thinking_tokens", 0) or 0
                    thinking_str = f", thinking={thinking_total}" if thinking_total else ""
                    print(
                        "  Token usage (total): "
                        f"prompt={token_totals['prompt_tokens']}, "
                        f"completion={token_totals['completion_tokens']}, "
                        f"total={token_totals['total_tokens']}"
                        f"{thinking_str}"
                    )
                else:
                    print("  Token usage (total): not reported by provider")

            print("\nAll models complete.")

        def _save_to_path(self, out_path: Path, results: list, quiet: bool = False):
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            if not quiet:
                print(f"\n  Saved {len(results)} results -> {out_path}")


---

## explore_dataset.py

```text
File: explore_dataset.py (1365 bytes)
```

    """
    Run this first to inspect the LiveCodeBench dataset structure
    before kicking off the full pipeline.

    Usage:
        python explore_dataset.py
    """

    from datasets import load_dataset
    from config.settings import LCB_DATASET, LCB_VERSION
    import json
    from datetime import datetime
    from pathlib import Path


    ds = load_dataset(
        LCB_DATASET,
        name=LCB_VERSION,
        split="test",
    )

    print(f"\nTotal samples: {len(ds)}")
    print(f"Columns: {ds.column_names}\n")

    # Show one sample in detail
    sample = ds[20]
    for key, value in sample.items():
        if isinstance(value, (list, dict)):
            preview = json.dumps(value, indent=2)[:300] + "..."
        else:
            preview = str(value)[:300]
        print(f"--- {key} ---")
        print(preview)
        print()

    # Write first 10 samples to results/dataset_preview.json
    Path("results").mkdir(exist_ok=True)
    out_path = Path("results") / "dataset_preview.json"
     
    records = []
    for sample in ds.select(range(10)):
        record = {}
        for key, value in sample.items():
            if isinstance(value, datetime):
                record[key] = value.isoformat()
            else:
                record[key] = value
        records.append(record)
     
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
     
    print(f"Saved {len(records)} samples -> {out_path}")


---

## INSTRUCTIONS.md

```text
File: INSTRUCTIONS.md (1439 bytes)
```

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
        "mr": 1,
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

---

## llms/__init__.py

```text
File: llms/__init__.py (282 bytes)
```

    from .claude_model import ClaudeModel
    from .openai_model import OpenAIModel
    from .gemini_model import GeminiModel
    from .deepseek_model import DeepSeekModel
    from .grok_model import GrokModel

    __all__ = ["ClaudeModel", "OpenAIModel", "GeminiModel", "DeepSeekModel", "GrokModel"]

---

## llms/base.py

```text
File: llms/base.py (2699 bytes)
```

    from abc import ABC, abstractmethod
    import time
    from typing import Callable, TypeVar

    from cost_tracker import update_costs

    T = TypeVar("T")


    class BaseLLM(ABC):
        """Abstract base class for all LLM wrappers."""

        @property
        def last_usage(self) -> dict | None:
            """Return the most recent token usage, if available."""
            return getattr(self, "_last_usage", None)
        
        @property
        def last_call_time(self) -> float | None:
            """Return wall-clock seconds for the most recent generate() call."""
            return getattr(self, "_last_call_time", None)

        def _set_last_call_time(self, elapsed: float) -> None:
            """Store wall-clock duration of the most recent generate() call."""
            self._last_call_time = elapsed

        def _set_last_usage(self, usage: dict | None) -> None:
            """Store token usage from the most recent request."""
            self._last_usage = usage
            update_costs(self.model_name, usage)

        def _record_timed_call(self, call: Callable[[], T]) -> T:
            """Run a provider request and store its elapsed wall-clock time."""
            t0 = time.perf_counter()
            try:
                return call()
            finally:
                self._set_last_call_time(time.perf_counter() - t0)

        def _set_openai_compatible_usage(self, response) -> None:
            """Store token usage from OpenAI-compatible response objects."""
            usage = getattr(response, "usage", None)
            if usage is None:
                self._set_last_usage(None)
                return

            thinking_tokens = None
            details = getattr(usage, "completion_tokens_details", None)
            if details is not None:
                thinking_tokens = getattr(details, "reasoning_tokens", None)

            self._set_last_usage({
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
                "thinking_tokens": thinking_tokens,
            })

        @abstractmethod
        def generate(self, prompt: str) -> str:
            """Send a prompt and return the response text."""
            pass

        @property
        @abstractmethod
        def model_name(self) -> str:
            """Identifier string for this model (used in results output)."""
            pass

        @property
        def last_finish_reason(self) -> str | None:
            """Return the finish/stop reason for the most recent generate() call."""
            return getattr(self, "_last_finish_reason", None)

        def _set_last_finish_reason(self, reason: str | None) -> None:
            self._last_finish_reason = reason

---

## llms/claude_model.py

```text
File: llms/claude_model.py (1335 bytes)
```

    import anthropic
    from .base import BaseLLM
    from config.settings import ANTHROPIC_API_KEY, CLAUDE_MODEL, MAX_TOKENS, TEMPERATURE


    class ClaudeModel(BaseLLM):
        def __init__(self, model: str = CLAUDE_MODEL):
            self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            self._model_name = model

        @property
        def model_name(self) -> str:
            return self._model_name

        def generate(self, prompt: str) -> str:
            message = self.client.messages.create(
                model=self._model_name,
                max_tokens=MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}],
            )
            usage = getattr(message, "usage", None)
            if usage is not None:
                prompt_tokens = getattr(usage, "input_tokens", None)
                completion_tokens = getattr(usage, "output_tokens", None)
                total_tokens = None
                if prompt_tokens is not None and completion_tokens is not None:
                    total_tokens = prompt_tokens + completion_tokens
                self._set_last_usage({
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                })
            else:
                self._set_last_usage(None)
            return message.content[0].text

---

## llms/deepseek_model.py

```text
File: llms/deepseek_model.py (1122 bytes)
```

    from openai import OpenAI
    from .base import BaseLLM
    from config.settings import DEEPSEEK_API_KEY, DEEPSEEK_MODEL, MAX_TOKENS, TEMPERATURE

    #NOTE: This ended up not actually using the deepseek LLM, its just a tester with free online api keys (google gemma in this case)
    class DeepSeekModel(BaseLLM):
        def __init__(self, model: str = DEEPSEEK_MODEL):
            self.client = OpenAI(
                base_url="https://integrate.api.nvidia.com/v1",
                api_key=DEEPSEEK_API_KEY,
            )
            self._model_name = model

        @property
        def model_name(self) -> str:
            return self._model_name

        def generate(self, prompt: str) -> str:
            completion = self._record_timed_call(
                lambda: self.client.chat.completions.create(
                    model=self._model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=TEMPERATURE,
                    top_p=0.7,
                    max_tokens=MAX_TOKENS,
                )
            )
            self._set_openai_compatible_usage(completion)
            return completion.choices[0].message.content


---

## llms/gemini_model.py

```text
File: llms/gemini_model.py (1369 bytes)
```

    from google import genai
    from google.genai import types
    from .base import BaseLLM
    from config.settings import GOOGLE_API_KEY, GEMINI_MODEL, MAX_TOKENS, TEMPERATURE

    class GeminiModel(BaseLLM):
        def __init__(self, model: str = GEMINI_MODEL):
            self.client = genai.Client(api_key=GOOGLE_API_KEY)
            self._model_name = model
     
        @property
        def model_name(self) -> str:
            return self._model_name
     
        def generate(self, prompt: str) -> str:
            response = self.client.models.generate_content(
                model=self._model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                ),
            )
            usage = getattr(response, "usage_metadata", None)
            if usage is not None:
                prompt_tokens = getattr(usage, "prompt_token_count", None)
                completion_tokens = getattr(usage, "candidates_token_count", None)
                total_tokens = getattr(usage, "total_token_count", None)
                self._set_last_usage({
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                })
            else:
                self._set_last_usage(None)
            return response.text


---

## llms/grok_model.py

```text
File: llms/grok_model.py (934 bytes)
```

    from openai import OpenAI
    from .base import BaseLLM
    from config.settings import GROK_API_KEY, GROK_MODEL, MAX_TOKENS, TEMPERATURE


    class GrokModel(BaseLLM):
        def __init__(self, model: str = GROK_MODEL):
            self.client = OpenAI(
                api_key=GROK_API_KEY,
                base_url="https://api.groq.com/openai/v1",
            )
            self._model_name = model

        @property
        def model_name(self) -> str:
            return self._model_name

        def generate(self, prompt: str) -> str:
            response = self._record_timed_call(
                lambda: self.client.chat.completions.create(
                    model=self._model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                )
            )
            self._set_openai_compatible_usage(response)
            return response.choices[0].message.content


---

## llms/mr_helper.py

```text
File: llms/mr_helper.py (2838 bytes)
```

    from openai import OpenAI
    from config.settings import OPENAI_API_KEY, MR_HELPER_MODEL, MR_HELPER_TOKENS, RESULTS_DIR
    import json
    import time
    from pathlib import Path

    HELPER_LOG_PATH = Path(RESULTS_DIR) / "helper_call_log.jsonl"

    def _log_helper_call(system: str, user: str, response_text: str, finish_reason, usage) -> None:
        HELPER_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(HELPER_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "timestamp": time.time(),
                "system_prompt": system,
                "user_input": user,
                "raw_response": response_text,
                "finish_reason": finish_reason,
                "usage": usage,
            }) + "\n")

    class MRHelper:

        def __init__(self, model: str = MR_HELPER_MODEL):
            self.client = OpenAI(api_key=OPENAI_API_KEY)
            self._model_name = model
            self._last_usage: dict | None = None

        @property
        def model_name(self) -> str:
            return self._model_name

        @property
        def last_usage(self) -> dict | None:
            return self._last_usage
        
        @property
        def last_finish_reason(self):
            return getattr(self, "_last_finish_reason", None)

        def transform(self, system: str, user: str) -> str:
            request = {
                "model": self._model_name,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            }

            try:
                response = self.client.chat.completions.create(**request)
            except Exception as exc:
                print(f"  [mr_helper] call failed ({exc}), returning input unchanged")
                self._last_usage = None
                self._last_finish_reason = None
                _log_helper_call(system, user, "", None, None)
                return user

            choice = response.choices[0]
            self._last_finish_reason = choice.finish_reason

            usage = getattr(response, "usage", None)
            if usage is not None:
                self._last_usage = {
                    "prompt_tokens": getattr(usage, "prompt_tokens", None),
                    "completion_tokens": getattr(usage, "completion_tokens", None),
                    "total_tokens": getattr(usage, "total_tokens", None),
                }
            else:
                self._last_usage = None

            result = choice.message.content.strip()
            _log_helper_call(system, user, result, self._last_finish_reason, self._last_usage)
            return result


    _helper: MRHelper | None = None


    def _get_helper() -> MRHelper:
        global _helper
        if _helper is None:
            _helper = MRHelper()
        return _helper


    def mr_llm_transform(system: str, user: str) -> str:
        return _get_helper().transform(system, user)

---

## llms/openai_model.py

```text
File: llms/openai_model.py (1183 bytes)
```

    from openai import OpenAI
    from .base import BaseLLM
    from config.settings import OPENAI_API_KEY, OPENAI_MODEL, MAX_TOKENS, TEMPERATURE


    class OpenAIModel(BaseLLM):
        def __init__(self, model: str = OPENAI_MODEL):
            self.client = OpenAI(api_key=OPENAI_API_KEY)
            self._model_name = model

        @property
        def model_name(self) -> str:
            return self._model_name

        def generate(self, prompt: str) -> str:
            request = {
                "model": self._model_name,
                "messages": [{"role": "user", "content": prompt}],
            }
            # if "gpt-5-nano" not in self._model_name:
                # request["temperature"] = TEMPERATURE
            if self._model_name.startswith("gpt-5"):
                request["max_completion_tokens"] = MAX_TOKENS
                request["reasoning_effort"] = "medium"
            else:
                request["max_tokens"] = MAX_TOKENS

            response = self._record_timed_call(lambda: self.client.chat.completions.create(**request))
            self._set_openai_compatible_usage(response)
            self._set_last_finish_reason(response.choices[0].finish_reason)
            return response.choices[0].message.content

---

## main.py

```text
File: main.py (4283 bytes)
```

    import argparse
    import subprocess
    import sys

    from data.loader import load_livecode_bench
    from evaluation.paths import build_results_path
    from evaluation.runner import PipelineRunner
    from llms import ClaudeModel, OpenAIModel, GeminiModel, DeepSeekModel, GrokModel
    from metamorphic.relations_list import RELATIONS

    MODEL_CHOICES = {
        "claude": ClaudeModel,
        "openai": OpenAIModel,
        "gemini": GeminiModel,
        "deepseek": DeepSeekModel,
        "grok": GrokModel,
    }

    def get_relation(mr_id: int):
        if mr_id not in RELATIONS:
            available = list(RELATIONS.keys())
            raise ValueError(f"Unknown MR: {mr_id}. Available: {available}")
        entry = RELATIONS[mr_id]
        return entry["class"]()

    def _run_tests(model_name: str, relation_name: str | None, relation_id: int | None, followup_only: bool) -> int:
        results_path = build_results_path(model_name, relation_name, relation_id)
        if not results_path.exists():
            print(f"Results file not found for tests: {results_path}")
            return 1
        cmd = [sys.executable, "runtests.py", "--results", str(results_path), "--write-outputs"]
        if followup_only:
            cmd.append("--followup-only")
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            print(f"Tests failed for {model_name} (exit {result.returncode})")
        return result.returncode

    def parse_args():
        parser = argparse.ArgumentParser(description="LLM MT Pipeline")
        parser.add_argument(
            "--models",
            nargs="+",
            choices=MODEL_CHOICES.keys(),
            default=list(MODEL_CHOICES.keys()),
            help="Which models to run (default: all). e.g. --models claude openai",
        )
        parser.add_argument(
            "--n",
            type=int,
            default=None,
            help="Limit number of samples. e.g. --n 10",
        )
        parser.add_argument(
            "--after",
            type=str,
            default=None,
            help="Only include problems on or after this date. e.g. --after 2024-01-01",
        )
        parser.add_argument(
            "--difficulty",
            nargs="+",
            choices=["easy", "medium", "hard"],
            default=None,
            help="Only include problems of these difficulties. e.g. --difficulty easy medium",
        )
        parser.add_argument(
            "--mr",
            type=int,
            default=None,
            help="Apply a metamorphic relation alongside the normal run. e.g. --mr 1",
        )
        parser.add_argument(
            "--mr-only",
            action="store_true",
            help="When used with --mr, skip the base LLM call and generate followup only",
        )
        parser.add_argument(
            "--run-tests",
            action="store_true",
            help="Run runtests.py on each model's results after generation",
        )
        parser.add_argument(
            "--repeats",
            type=int,
            default=1,
            help="Generate each source/followup prompt this many times to measure prompt variability",
        )
        parser.add_argument(
            "--resume",
            action="store_true",
            help="Continue from an existing results file by skipping entries with required code blocks",
        )
        return parser.parse_args()

    def main() -> int:
        args = parse_args()

        dataset = load_livecode_bench(n_samples=args.n, after=args.after, difficulty=args.difficulty)
        models = [MODEL_CHOICES[name]() for name in args.models]
        relation = get_relation(args.mr) if args.mr is not None else None
        if args.mr_only and args.mr is None:
            print("--mr-only requires --mr to be set")
            return 2
        if args.repeats < 1:
            print("--repeats must be at least 1")
            return 2

        print(f"Running models: {[m.model_name for m in models]}, on metamorphic relation: {args.mr is not None}")

        runner = PipelineRunner(
            models=models,
            dataset=dataset,
            relation=relation,
            relation_id=args.mr,
            mr_only=args.mr_only,
            prompt_repeats=args.repeats,
            resume=args.resume,
        )
        runner.run()

        if args.run_tests:
            for model in models:
                relation_name = relation.name if relation else None
                exit_code = _run_tests(model.model_name, relation_name, args.mr, args.mr_only)
                if exit_code != 0:
                    return exit_code
        return 0

    if __name__ == "__main__":
        raise SystemExit(main())


---

## metamorphic/__init__.py

```text
File: metamorphic/__init__.py (0 bytes)
```



---

## metamorphic/relations/__init__.py

```text
File: metamorphic/relations/__init__.py (0 bytes)
```



---

## metamorphic/relations/base_mr.py

```text
File: metamorphic/relations/base_mr.py (241 bytes)
```

    class BaseMR:
        target = "PD"
     
        def transform(self, text: str) -> str:
            raise NotImplementedError
     
        def verify(self, source_passed: bool, followup_passed: bool) -> bool:
            return source_passed == followup_passed

---

## metamorphic/relations/mrs.py

```text
File: metamorphic/relations/mrs.py (8242 bytes)
```

    import re, random
    from metamorphic.relations.base_mr import BaseMR
    from metamorphic.relations import phrases
    from metamorphic.relations.phrases import _pick
    from llms.mr_helper import mr_llm_transform
    from evaluation.executor import extract_code


    def _insert_after_first_line(prompt: str, line: str) -> str:
        lines = prompt.split("\n")
        if lines and lines[0] != "":
            lines.insert(1, line)
            return "\n".join(lines)
        if lines and lines[0] == "":
            lines.insert(0, line)
            return "\n".join(lines)
        return line


    def _prepend_line(prompt: str, line: str) -> str:
        if not prompt:
            return line
        return f"{line}\n{prompt}"


    def _strip_list_prefix(text: str) -> str:
        if text.startswith("-") or text.startswith("*"):
            return text.lstrip("-* ").strip()
        match = re.match(r"^\d+[\.)]\s*(.*)$", text)
        if match:
            return match.group(1)
        return text


    def _to_bulleted(prompt: str) -> str:
        lines = prompt.split("\n")
        out = []
        in_code = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("```"):
                in_code = not in_code
                out.append(line)
                continue
            if in_code or stripped == "":
                out.append(line)
                continue
            out.append(f"- {_strip_list_prefix(stripped)}")
        return "Structured prompt:\n" + "\n".join(out)

    def _remove_random_sentence(text: str) -> str:
        """Split on sentence boundaries and delete one at random, preserving original spacing."""
        sentences = re.findall(r"[^.!?]*[.!?]+\s*|[^.!?]+$", text)
        candidates = [i for i, s in enumerate(sentences) if s.strip()]
        if candidates:
            del sentences[random.choice(candidates)]
        return "".join(sentences)

    def _clean_library_name(raw: str) -> str:
        name = raw.strip().splitlines()[0].strip()
        name = name.strip("`\"'. ")
        if not name or len(name) > 40:
            name = "a suitable standard library module"
        return name

    class MR0(BaseMR):
        target = "PD"

        @property
        def name(self) -> str:
            return "do_nothing"

        def transform(self, text: str) -> str:
            return text


    class MR1(BaseMR):
        target = "PD"

        @property
        def name(self) -> str:
            return "irrelevant_context"

        def transform(self, text: str) -> str:
            return _insert_after_first_line(text, _pick(phrases.IRRELEVANT_CONTEXT))


    class MR2(BaseMR):
        target = "I"

        @property
        def name(self) -> str:
            return "constraint_redundant_spec"

        def transform(self, prefix: str, suffix: str) -> tuple[str, str]:
            return _insert_after_first_line(prefix, _pick(phrases.REDUNDANT_CONTEXT)), suffix


    class MR3(BaseMR):
        target = "PD"

        @property
        def name(self) -> str:
            return "format_structural_equivalence"

        def transform(self, text: str) -> str:
            return _to_bulleted(text)


    class MR4(BaseMR):
        target = "I"

        @property
        def name(self) -> str:
            return "persona_role_prompting"

        def transform(self, prefix: str, suffix: str) -> tuple[str, str]:
            return _prepend_line(prefix, _pick(phrases.PERSONAS)), suffix


    class MR5(BaseMR):
        target = "PD"

        @property
        def name(self) -> str:
            return "polarity_negative_formulation"

        def transform(self, text: str) -> str:
            return mr_llm_transform(phrases.NEGATION_PROMPT, text)


    class MR6(BaseMR):
        target = "I"

        @property
        def name(self) -> str:
            return "identifier_explicit_naming"

        def transform(self, prefix: str, suffix: str) -> tuple[str, str]:
            return (_insert_after_first_line(prefix, phrases.make_identifier_constraint(phrases.WORDS)), suffix)


    class MR7(BaseMR):
        target = "PD"

        @property
        def name(self) -> str:
            return "order_requirement_permutation"

        def transform(self, text: str) -> str:
            return mr_llm_transform(phrases.PERMUTATION_PROMPT, text)


    class MR8(BaseMR):
        target = "PD"

        @property
        def name(self) -> str:
            return "remove_sentences"

        def transform(self, text: str) -> str:
            return _remove_random_sentence(text)


    class MR9(BaseMR):
        target = "PD"

        @property
        def name(self) -> str:
            return "antonym_substitution"

        def transform(self, text: str) -> str:
            return mr_llm_transform(phrases.ANTONYM_PROMPT_DIFF, text)


    class MR10(BaseMR):
        target = "PD"

        @property
        def name(self) -> str:
            return "synonym_substitution"

        def transform(self, text: str) -> str:
            return mr_llm_transform(phrases.SYNONYM_PROMPT, text)


    class MR11(BaseMR):
        target = "I"

        @property
        def name(self) -> str:
            return "loop_while_only"

        def transform(self, prefix: str, suffix: str) -> tuple[str, str]:
            return _insert_after_first_line(prefix, _pick(phrases.LOOP_TYPE_CONSTRAINT)), suffix


    class MR12(BaseMR):
        target = "I"

        @property
        def name(self) -> str:
            return "full_commenting"

        def transform(self, prefix: str, suffix: str) -> tuple[str, str]:
            return _insert_after_first_line(prefix, phrases.FULL_COMMENTING_CONSTRAINT), suffix


    class MR13(BaseMR):
        target = "I"

        @property
        def name(self) -> str:
            return "dummy_variable"

        def transform(self, prefix: str, suffix: str) -> tuple[str, str]:
            return _insert_after_first_line(prefix, _pick(phrases.DUMMY_VARIABLE_CONSTRAINT)), suffix


    class MR14(BaseMR):
        target = "I"

        @property
        def name(self) -> str:
            return "guard_clause_conditionals"

        def transform(self, prefix: str, suffix: str) -> tuple[str, str]:
            return _insert_after_first_line(prefix, phrases.GUARD_CLAUSE_CONSTRAINT), suffix


    class MR15(BaseMR):
        target = "I"

        @property
        def name(self) -> str:
            return "use_recursion"

        def transform(self, prefix: str, suffix: str) -> tuple[str, str]:
            return _insert_after_first_line(prefix, phrases.RECURSION_CONSTRAINT), suffix


    class MR16(BaseMR):
        target = "I"

        @property
        def name(self) -> str:
            return "no_builtin_libraries"

        def transform(self, prefix: str, suffix: str) -> tuple[str, str]:
            return _insert_after_first_line(prefix, phrases.NO_BUILTIN_LIBS), suffix


    class MR17(BaseMR):
        """LLM-assisted: ask the helper which standard-library module fits this
        problem, then require the model under test to use it."""
        target = "PD"

        @property
        def name(self) -> str:
            return "mandated_library"

        def transform(self, text: str) -> str:
            suggestion = mr_llm_transform(phrases.LIBRARY_SUGGESTION_PROMPT, text)
            library = _clean_library_name(suggestion)
            constraint = f"You must use Python's {library} in your solution."
            print(f"  [mr17] helper suggested: {library}")
            return _insert_after_first_line(text, constraint)

    class MR18(BaseMR):
        target = "PD"

        @property
        def name(self) -> str:
            return "scope_input_narrowing"

        def transform(self, text: str) -> str:
            return _insert_after_first_line(text, _pick(phrases.SCOPE_NARROWING))


    class MR19(BaseMR):
        """LLM-assisted: have the helper solve the problem in one call, then, in a
        separate call with no access to the original problem text, describe only
        the resulting code. That description becomes the new problem statement."""
        target = "PD"

        @property
        def name(self) -> str:
            return "round_trip_paraphrase"

        def transform(self, text: str) -> str:
            solve_response = mr_llm_transform(phrases.ROUND_TRIP_SOLVE_PROMPT, text)
            code = extract_code(solve_response)
            if not code:
                # Helper failed to produce a fenced block — fall back to the raw
                # response so the MR doesn't silently return an empty prompt.
                code = solve_response
            explanation = mr_llm_transform(phrases.ROUND_TRIP_EXPLAIN_PROMPT, code)
            return explanation

---

## metamorphic/relations/phrases.py

```text
File: metamorphic/relations/phrases.py (13150 bytes)
```

    import random


    def _pick(variants: list[str]) -> str:
        return random.choice(variants)


    def make_identifier_constraint(words: list[str]) -> str:
        var1, var2 = random.sample(words, 2)
        template = _pick(IDENTIFIER_CONSTRAINT)
        return template.format(variable1=var1, variable2=var2)


    WORDS = [
        "whisper",
        "tmp",
        "garden",
        "foo",
        "copper",
        "val2",
        "q7z",
        "thunder",
        "a1",
        "ribbon",
        "cnt2",
        "marble",
        "do_thing",
        "velvet",
        "m2p",
    ]
    IRRELEVANT_CONTEXT = [
        "Fun fact: honey never spoils if stored properly.",
        "The Great Barrier Reef is visible from space.",
        "This question was inspired by a conversation overheard at a coffee shop.",
        "Octopuses have three hearts and blue blood.",
        "This scenario is loosely based on an old folk tale from Eastern Europe.",
        "The Eiffel Tower can grow taller in summer due to heat expansion.",
        "A similar version of this puzzle once appeared in a math olympiad.",
        "Bananas are technically classified as berries.",
        "This example was adapted from a textbook published in the 1990s.",
        "The average adult has around 60,000 miles of blood vessels in their body.",
        "This dataset was collected across 12 different research sites.",
        "Researchers have long debated the optimal way to teach coding to beginners.",
        "A similar approach is sometimes used in supply chain optimization problems.",
        "Many students find this next part surprisingly tricky.",
        "The silent astronaut carefully navigated her lunar rover through the crystal caverns.",
    ]
    REDUNDANT_CONTEXT = [
        "Note: This problem was originally presented in an international programming contest.",
        "Follow the programming language's required syntax rules (e.g., indentation for scope in Python).",
        "Make sure your code compiles or runs without syntax errors before submitting.",
        "Use meaningful variable names to improve code readability.",
        "Remember that most languages are case sensitive when it comes to variable and function names.",
        "It's good practice to test your solution on multiple inputs before finalizing it.",
        "Standard input and output methods should be used unless otherwise specified.",
        "Code style and formatting conventions may vary by language but don't change the underlying logic.",
        "Efficient solutions are appreciated, but correctness is the primary requirement.",
        "Off-by-one errors are a common source of bugs, so double check loop boundaries.",
        "Version control is recommended for tracking changes, though it has no bearing on the output.",
        "Do not return anything other than what the specification requires.",
        "Some languages require explicit type declarations, while others infer types automatically.",
        "Whitespace within expressions is generally ignored by most compilers and interpreters.",
        "Definitions: iterate means loop through each element; concatenate means append one string to the end of another.",
    ]
    PERSONAS = [
        "You are an expert senior software engineer writing production-grade code.",
        "You are a competitive programmer optimizing for speed and efficiency under contest constraints.",
        "You are a meticulous code reviewer who catches every edge case and bug.",
        "You are a computer science professor writing algorithms to first-year students.",
        "You are a security-focused engineer auditing code for vulnerabilities.",
        "You are a systems programmer working close to the hardware, optimizing for memory and performance.",
        "You are a junior developer just learning best practices, careful and methodical in your approach.",
        "You are a technical interviewer evaluating a candidate's problem-solving process step by step.",
        "You are an open-source maintainer who values clean, well-documented, idiomatic code.",
        "You are a data scientist writing code to prototype and validate an algorithmic idea quickly.",
        "You are a full-stack developer balancing readability with performance across the entire code.",
        "You are a compiler engineer who thinks in terms of parsing, syntax trees, and formal language rules.",
        "You are a DevOps engineer writing scripts that must be robust and reliable in automated pipelines.",
        "You are a research engineer implementing an algorithm straight from an academic paper, prioritizing correctness over polish.",
    ]

    NEGATION_PROMPT = (
        "Rewrite only the return/output specification using a logically equivalent negative "
        "formulation. Before rewriting, verify that the rewritten sentence is true if and only "
        "if the original sentence is true. Do not modify any other part of the "
        "problem. Output only the entire rewritten problem text."
    )

    IDENTIFIER_CONSTRAINT = [
        "Call your temporary variable {variable1} and your final answer {variable2}.",
        "The input variable should be named {variable1}; the output variable should be named {variable2}.",
        "Name your first value {variable1} and your second value {variable2}.",
        "Call your main variable {variable1} and your supporting variable {variable2}.",
        "The primary variable should be called {variable1}; the secondary variable should be called {variable2}.",
        ]

    PERMUTATION_PROMPT = (
        "Rewrite the programming problem below by reordering independent constraints, requirements, "
        "or setup sentences into a different but logically equivalent order. Do not change, remove, "
        "add, or reword any requirement, constraint, or detail — only reorder sentences or clauses "
        "that do not depend on one another. "
        "Output only the rewritten problem text with no preamble."
    )

    ANTONYM_PROMPT_DIFF = (
        "Rewrite the programming problem below by substituting key terms with their antonyms "
        "(e.g., 'maximum' -> 'minimum', 'increasing' -> 'decreasing', 'at least' -> 'at most'), "
        "ensuring the substitutions make sense in context.\n\n"
        "Critical: the rewritten problem must stay internally consistent. The antonym substitution "
        "changes the problem's logic, so any sample inputs, sample outputs, or worked-example "
        "explanations in the original problem will likely no longer be correct under the new "
        "logic. For every example in the problem:\n"
        "  - Recompute what the correct output actually is for that sample input, given the "
        "new (antonym-substituted) problem statement — do not just copy the original output.\n"
        "  - Rewrite any explanation of why that output is correct so it matches the new logic.\n"
        "  - Sample input values may be kept as-is; only outputs and explanations need to change "
        "to match the new problem.\n\n"
        "Output only the rewritten problem text with no preamble."
    )

    ANTONYM_PROMPT_EQUIV = (
        "Rewrite the programming problem below by substituting key terms with their antonyms "
        "ensuring that the substitutions make sense in context"
        "and adjusting the surrounding logic so the problem remains semantically equivalent "
        "(e.g. 'ascending order' → 'not in descending order'). "
        "All numeric constraints, input/output formats, and the required answer must stay identical. "
        "Output only the rewritten problem text with no preamble."
    )

    SYNONYM_PROMPT = (
        "Rewrite the programming problem below by substituting key domain terms and verbs "
        "ensuring that the substitutions make sense in context"
        "Preserve all numeric constraints, input/output formats, and the exact meaning. "
        "Do not add or remove any requirements. "
        "Output only the rewritten problem text with no preamble."
    )

    LOOP_TYPE_CONSTRAINT = (
        "You must use only while loops for all iteration. Do not use for loops.",
        "You must use only for loops for all iteration. Do not use while loops.",
        "You must use nested loops to solve this problem. A single loop is not allowed.",
        "You must use only a single loop with no nesting. Nested loops are not allowed.",
    )

    FULL_COMMENTING_CONSTRAINT = (
        "Add a concise inline comment to every line of your solution explaining what that line does. "
        "Every non-blank line of code must have a comment."
    )

    DUMMY_VARIABLE_CONSTRAINT = (
        "At the start of your solution, declare a variable named _unused_flag and set it to True. Declare a variable named unused_iteration and set it to 0. Do not use these variables anywhere else in your solution.",
        "Before writing any logic, declare a variable named _dummy_check and set it to False. Declare a variable named _placeholder_count and set it to 0. Do not use these variables anywhere else in your code.",
        "At the top of your solution, add a variable _skip_marker set to None and a variable _reserved_slot set to -1. Do not reference either variable again.",
        "Declare two unused variables at the beginning of your solution: _idle_state set to True and _padding_value set to 0. Neither should appear anywhere else in the code.",
        "As the first lines of your solution, initialize _noop_flag to False and _spare_index to 0. These declarations must remain unused for the rest of your code.",
        "Your solution must begin with the declaration of _sentinel_flag (set to True) and _extra_counter (set to 1). Do not use these variables in any subsequent logic.",
        "Include _placeholder_bool set to False and _filler_number set to 99 at the start of your code. These must not be referenced or used anywhere else.",
        "At the beginning of your solution, create a variable _stub_flag with the value True and a variable _reserve_count with the value 0. Leave both unused for the remainder of the solution.",
        "First, declare _inactive_marker as None and _buffer_index as -1. Do not use either variable elsewhere in your solution.",
        "Add two dummy declarations at the start of your solution: _unused_toggle = False and _slack_variable = 0. These variables should not be used anywhere else.",
        "At the very beginning of your code, declare _dormant_flag and set it to True, and declare _padding_counter and set it to 5. Do not use these anywhere else in your solution.",
        "Your code must start with two unused variables: _quiet_flag set to False and _phantom_index set to -1. Neither variable should be used again.",
        "Declare _idle_marker set to None and _spare_counter set to 0 before any other code. These must remain unreferenced for the rest of the solution.",
        "At the top of your solution, initialize _hollow_flag to True and _extra_slot to 1. Do not use these variables anywhere else in your code.",
        "Begin your solution by declaring _void_flag set to False and _reserve_index set to 0. These variables must not be used anywhere else in the solution.",
    )

    GUARD_CLAUSE_CONSTRAINT = (
        "All conditional logic must be expressed as guard clauses (early returns) rather than nested if-else statements. "
        "Do not use nested if-else statements."
    )

    RECURSION_CONSTRAINT = (
        "You must implement your solution using recursion. "
        "Do not use any iterative loops (for or while)."
    )

    NO_BUILTIN_LIBS = (
        "You must not import or use any standard library modules; implement everything from scratch using only core language syntax."
    )

    LIBRARY_SUGGESTION_PROMPT = (
        "Given the following programming problem, name exactly one Python standard-library module "
        "that would be genuinely useful for solving it efficiently (for example: itertools, "
        "bisect, functools). "
        "Respond with only the name, nothing else no explanation, no punctuation, no surrounding text."
    )

    SCOPE_NARROWING = [
        "Additional guarantee: none of the input values will ever be zero.",
        "Additional guarantee: all numeric inputs are guaranteed to be strictly positive.",
        "Additional guarantee: the input size is always a power of two.",
        "Additional guarantee: the input contains no duplicate values.",
        "Additional guarantee: all inputs fit within the range of a 32-bit signed integer.",
        "Additional guarantee: the input is already sorted in non-decreasing order.",
    ]

    ROUND_TRIP_SOLVE_PROMPT = (
        "Solve the following programming problem. Respond with only a single fenced "
        "Python code block containing your complete solution, no explanation, no preamble."
    )

    ROUND_TRIP_EXPLAIN_PROMPT = (
        "Here is a Python solution to a programming problem. Without seeing the original "
        "problem statement, write a clear, complete natural-language description of what "
        "this code does, as if writing a detailed docstring for someone else to implement "
        "from scratch. Explicitly restate any input format, output format, or constraints "
        "on the input that the code implies (through parsing, bounds checks, loop ranges, "
        "or similar), so your description fully specifies the problem on its own. Someone "
        "reading only your description, with no access to this code, must be able to write "
        "correct code from it alone. Output only that description, with no preamble and no code."
    )

---

## metamorphic/relations_list.py

```text
File: metamorphic/relations_list.py (3871 bytes)
```

    from metamorphic.relations.mrs import (
        MR0,
        MR1,
        MR2,
        MR3,
        MR4,
        MR5,
        MR6,
        MR7,
        MR8,
        MR9,
        MR10,
        MR11,
        MR12,
        MR13,
        MR14,
        MR15,
        MR16,
        MR17,
        MR18,
        MR19,
    )

    RELATIONS = {
        0:  {"class": MR0,  "name": "do_nothing",                     "target": "PD", "description": "Control relation that returns the original prompt and output unchanged."},
        1:  {"class": MR1,  "name": "irrelevant_context",             "target": "PD", "description": "Insert irrelevant sentence into prompt"},
        2:  {"class": MR2,  "name": "redundant_context",              "target": "I",  "description": "Insert a related sentence which has no effect on the solution."},
        3:  {"class": MR3,  "name": "format_structural_equivalence",  "target": "PD", "description": "Convert the prompt into a structured bulleted list."},
        4:  {"class": MR4,  "name": "persona_role_prompting",         "target": "I",  "description": "Prepend a persona to the prompt."},
        5:  {"class": MR5,  "name": "polarity_negative_formulation",  "target": "PD", "description": "LLM-assisted rewrite of the return/output requirement into an equivalent negative form."},
        6:  {"class": MR6,  "name": "identifier_explicit_naming",     "target": "I",  "description": "Add explicit internal naming constraints for variables."},
        7:  {"class": MR7,  "name": "order_requirement_permutation",  "target": "PD", "description": "LLM-assisted reordering of independent constraints/sentences, preserving meaning."},
        8:  {"class": MR8,  "name": "remove_sentences",               "target": "PD", "description": "Random removal of exactly one sentence."},
        9:  {"class": MR9,  "name": "antonym_substitution",           "target": "PD", "description": "LLM-assisted substitution of key terms with antonyms, rewriting surrounding logic to remain equivalent."},
        10: {"class": MR10, "name": "synonym_substitution",           "target": "PD", "description": "LLM-assisted substitution of key terms in the question with synonyms while preserving semantics."},
        11: {"class": MR11, "name": "loop_while_only",                "target": "I",  "description": "Constrain the solution to use only while loops; no for loops permitted."},
        12: {"class": MR12, "name": "full_commenting",                "target": "I",  "description": "Require an inline comment on every non-blank line of the solution."},
        13: {"class": MR13, "name": "dummy_variable",                 "target": "I",  "description": "Require declaration of an unused dummy variable (_unused_flag) at the start of the solution."},
        14: {"class": MR14, "name": "guard_clause_conditionals",      "target": "I",  "description": "Require all conditional logic to be expressed as guard clauses (early returns) rather than nested if-else."},
        15: {"class": MR15, "name": "use_recursion",                  "target": "I",  "description": "Require a recursive implementation; no iterative loops permitted."},
        16: {"class": MR16, "name": "no_builtin_libraries",           "target": "I",  "description": "Forbid use of any Python standard-library module; require the solution to be implemented manually."},
        17: {"class": MR17, "name": "mandated_library",               "target": "PD", "description": "LLM-assisted: ask the helper which standard-library module fits this problem, then require the model under test to use it."},
        18: {"class": MR18, "name": "scope_input_narrowing",          "target": "PD", "description": "State an additional (generic, not problem-verified) guarantee about the input domain."},
        19: {"class": MR19, "name": "round_trip_paraphrase",          "target": "PD", "description": "LLM-assisted: replace the problem statement with a solve-then-describe paraphrase of it, generated by the helper model."},
    }

---

## README.md

```text
File: README.md (13199 bytes)
```

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
        "mr": 1,
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
    | `--models MODEL ...` | Models to run: `claude`, `openai`, `gemini`, `deepseek`, or `grok`. Defaults to all. |
    | `--n N` | Limit the number of dataset problems. |
    | `--after YYYY-MM-DD` | Include problems on or after the specified date. |
    | `--mr ID` | Apply the metamorphic relation with the specified ID. |
    | `--mr-only` | Reuse saved base responses and generate only MR follow-up responses. Requires `--mr`. |
    | `--repeats N` | Generate each applicable prompt N times. Defaults to `1`. |
    | `--resume` | Skip existing entries that already contain the required code blocks. |
    | `--run-tests` | Evaluate results after generation and write split outputs plus an HTML report. |

    ### Common examples

    Run a base experiment:

    ```powershell
    python main.py --models deepseek --n 5
    ```

    Run an MR experiment:

    ```powershell
    python main.py --models openai --n 10 --mr 4
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
    | `--write-outputs` | Write split content/results files, generate the HTML report, and refresh the merged `all.html` + `violations.json`. |

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
    results/<model>/base/base.json
    results/<model>/mr-<id>-<relation>/mr-<id>-<relation>.json
    ```

    The model directory uses the configured model name from `config/settings.py`, sanitized for use as a folder name. For example, the `openai` CLI alias currently writes under `results/gpt-5-nano/`.

    When `--write-outputs` is used, additional files are created:

    ```text
    results/<model>/
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
    |-- all.html              Merged HTML report of the base run + all MR runs
    `-- violations.json       Log of every MR violation (mr_holds == False)
    ```

    `all.html` and `violations.json` are consolidated automatically once testing
    finishes — at the end of `main.py --run-tests` (per model) and after every
    `runtests.py` run that uses `--write-outputs`. They are regenerated from
    whatever results exist on disk, so they stay up to date as runs are added.

    Each `violations.json` entry records the relation, question, source vs followup
    pass status and test counts, `passes_same_tests`, and code similarity for that
    violation.

    Test logs default to the input result filename with `.tests.log` appended. CSV output is written only when `--csv` is supplied.

    ## Metamorphic Relations

    | ID | Name | Target | Transformation |
    |---:|---|---|---|
    | 0 | `do_nothing` | Question | Leaves the prompt unchanged as a control. |
    | 1 | `dead_context` | Question | Adds an irrelevant sentence. |
    | 2 | `constraint_redundant_spec` | Instruction | Adds a redundant language constraint. |
    | 3 | `format_structural_equivalence` | Question | Converts the prompt into a structured list. |
    | 4 | `persona_role_prompting` | Instruction | Adds a software-engineer persona. |
    | 5 | `context_extraneous_narrative` | Question | Adds irrelevant narrative context. |
    | 6 | `polarity_negative_formulation` | Instruction | Rewrites a return constraint in negative form. |
    | 7 | `identifier_explicit_naming` | Instruction | Adds explicit variable-naming constraints. |
    | 8 | `output_formatting_constraint` | Instruction | Adds raw-code output constraints. |
    | 9 | `hint_trivial_algorithm` | Instruction | Adds a simple algorithmic hint. |
    | 10 | `order_requirement_permutation` | Question | Reorders independent constraints. |
    | 11 | `verbosity_lexical_expansion` | Question | Adds short definitions for common terms. |

    Relations are implemented in `metamorphic/relations/mrs.py` and registered in `metamorphic/relations_list.py`.

    To add a relation:

    1. Add a class derived from `BaseMR`.
    2. Set its `target` to `question` or `instruction`.
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


---

## requirements.txt

```text
File: requirements.txt (160 bytes)
```

    anthropic
    openai
    google-genai
    datasets
    python-dotenv
    pandas
    radon
    sentence-transformers
    torch
    sentencepiece
    transformers==4.38.2
    tokenizers==0.15.2


---

## run_all_mrs.py

```text
File: run_all_mrs.py (2467 bytes)
```

    import json
    import subprocess
    import sys
    from pathlib import Path

    from metamorphic.relations_list import RELATIONS
    from config.settings import GROK_MODEL, DEEPSEEK_MODEL
    from evaluation.paths import build_results_path


    def _load_run_results(model_name: str, relation_name: str, relation_id: int) -> list[dict] | None:
        src = build_results_path(model_name, relation_name, relation_id)
        if not src.exists():
            print(f"Expected results file not found: {src}")
            return None
        with open(src, "r", encoding="utf-8") as f:
            return json.load(f)


    def main() -> int:
        base_results_path = build_results_path(DEEPSEEK_MODEL, None, None)
        model_dir = base_results_path.parent.parent
        out_dir = model_dir / "all_mrs"
        out_path = out_dir / "all_mrs.json"
        combined_results: list[dict] = []
        mr_ids = sorted(RELATIONS.keys())
        for mr_id in mr_ids:
            mr_name = RELATIONS[mr_id]["name"]
            print(f"\n=== Running MR {mr_id}: {mr_name} ===")
            result = subprocess.run(
                [
                    sys.executable,
                    "main.py",
                    "--models",
                    "deepseek",
                    "--n",
                    "1",
                    "--mr",
                    str(mr_id),
                ],
                check=False,
            )
            if result.returncode != 0:
                print(f"MR {mr_id} failed with exit code {result.returncode}")
                return result.returncode
            run_results = _load_run_results(DEEPSEEK_MODEL, mr_name, mr_id)
            if run_results is None:
                return 1
            combined_results.extend(run_results)

        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(combined_results, f, indent=2, ensure_ascii=False)
        print(f"\nSaved all MR results -> {out_path}")

        tests_out_path = out_dir / "all_mrs.tests.log"
        test_result = subprocess.run(
            [
                sys.executable,
                "runtests.py",
                "--results",
                str(out_path),
                "--out",
                str(tests_out_path),
            ],
            check=False,
        )
        if test_result.returncode != 0:
            print(f"Tests failed with exit code {test_result.returncode}")
            return test_result.returncode
        print(f"Saved test log -> {tests_out_path}")
        return 0


    if __name__ == "__main__":
        raise SystemExit(main())


---

## runtests.py

```text
File: runtests.py (25398 bytes)
```

    """
    Run all test cases against model responses.
    If the results file contains followup responses (from --mr), tests both
    and computes MR verdict.

    Usage:
        python runtests.py --results results/grok/base/base.json
        python runtests.py --results results/grok/base/base.json --n 5
        python runtests.py --results results/grok/base/base.json --stdout
        python runtests.py --results results/grok/mr-05-context_extraneous_narrative.json --followup-only
        python runtests.py --results results/grok/mr-05-context_extraneous_narrative.json --csv results/grok/metrics.csv
    """

    import json
    import argparse
    import csv
    import sys
    from contextlib import redirect_stdout, redirect_stderr
    from pathlib import Path

    from evaluation.executor import (
        extract_code, get_all_tests, get_func_name, run_test,
    )
    from evaluation.metrics import (
        compute_metrics,
        compute_code_similarity,
        compute_pairwise_code_variability,
        compute_extractable_code,
        compute_passes_same_tests,
        compute_passes_tests,
        compute_token_count,
        compute_token_efficiency,
    )


    def _response_samples(entry: dict, sample_key: str, primary_key: str) -> list[str]:
        samples = entry.get(sample_key)
        if isinstance(samples, list) and samples:
            return [s for s in samples if isinstance(s, str)]
        primary = entry.get(primary_key, "")
        return [primary] if isinstance(primary, str) and primary else []


    def _code_samples(responses: list[str]) -> list[str | None]:
        return [extract_code(r) for r in responses]


    def _fmt(value, digits: int = 4) -> str:
        return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "N/A"


    def _delta(value) -> str:
        return f"{value:+}" if isinstance(value, (int, float)) else value


    def _variation_greater(source_mean_dist, followup_mean_dist) -> str:
        if isinstance(source_mean_dist, (int, float)) and isinstance(followup_mean_dist, (int, float)):
            return "YES" if followup_mean_dist > source_mean_dist else "NO"
        return "N/A"


    def _write_metrics_csv(csv_path: str | Path, metric_rows: list[dict], has_mr: bool) -> None:
        path = Path(csv_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        base_fields = [
            "id",
            "extractable_code", "passes_tests", "source_passed",
            "tests_passed", "tests_failed", "tests_total",
            "test_run_time",
            "source_ncloc", "source_cc",
            "source_mean_distance", "source_std_distance", "source_pairs",
            "prompt_tokens", "completion_tokens", "total_tokens",
            "thinking_tokens", "token_efficiency",
            "llm_call_time",
        ]
        mr_extra_fields = [
            "followup_extractable", "followup_passes_tests", "passes_same_tests", "followup_passed",
            "followup_tests_passed", "followup_tests_failed", "followup_tests_total",
            "followup_test_run_time",
            "followup_ncloc", "followup_cc",
            "followup_mean_distance", "followup_std_distance", "followup_pairs",
            "followup_prompt_tokens", "followup_completion_tokens", "followup_total_tokens",
            "followup_thinking_tokens", "followup_token_efficiency",
            "followup_llm_call_time",
            "code_sim", "mr_holds",
        ]
        fieldnames = base_fields + (mr_extra_fields if has_mr else [])

        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in metric_rows:
                src_var = row.get("source_variability") or {}
                flw_var = row.get("followup_variability") or {}
                output = {
                    "id": row.get("id"),
                    "extractable_code": row.get("source_extractable"),
                    "passes_tests": row.get("source_passes_tests"),
                    "source_passed": row.get("source_passed"),
                    "tests_passed": row.get("tests_passed"),
                    "tests_failed": row.get("tests_failed"),
                    "tests_total": row.get("tests_total"),
                    "test_run_time": row.get("source_test_run_time"),
                    "source_ncloc": row.get("source_ncloc"),
                    "source_cc": row.get("source_cc"),
                    "source_mean_distance": src_var.get("mean_distance"),
                    "source_std_distance": src_var.get("std_distance"),
                    "source_pairs": src_var.get("n_pairs"),
                    "prompt_tokens": row.get("source_prompt_tokens"),
                    "completion_tokens": row.get("source_completion_tokens"),
                    "total_tokens": row.get("source_total_tokens"),
                    "thinking_tokens": row.get("source_thinking_tokens"),
                    "token_efficiency": row.get("source_token_efficiency"),
                    "llm_call_time": row.get("source_llm_call_time"),
                    # MR fields
                    "followup_extractable": row.get("followup_extractable"),
                    "followup_passes_tests": row.get("followup_passes_tests"),
                    "passes_same_tests": row.get("passes_same_tests"),
                    "followup_passed": row.get("followup_passed"),
                    "followup_tests_passed": row.get("followup_tests_passed"),
                    "followup_tests_failed": row.get("followup_tests_failed"),
                    "followup_tests_total": row.get("followup_tests_total"),
                    "followup_test_run_time": row.get("followup_test_run_time"),
                    "followup_ncloc": row.get("followup_ncloc"),
                    "followup_cc": row.get("followup_cc"),
                    "followup_mean_distance": flw_var.get("mean_distance"),
                    "followup_std_distance": flw_var.get("std_distance"),
                    "followup_pairs": flw_var.get("n_pairs"),
                    "followup_prompt_tokens": row.get("followup_prompt_tokens"),
                    "followup_completion_tokens": row.get("followup_completion_tokens"),
                    "followup_total_tokens": row.get("followup_total_tokens"),
                    "followup_thinking_tokens": row.get("followup_thinking_tokens"),
                    "followup_token_efficiency": row.get("followup_token_efficiency"),
                    "followup_llm_call_time": row.get("followup_llm_call_time"),
                    "code_sim": row.get("code_sim"),
                    "mr_holds": row.get("mr_holds"),
                }
                writer.writerow({field: output.get(field) for field in fieldnames})
        print(f"\n  Wrote metrics CSV -> {path}")


    def test_response(
        response: str,
        entry: dict,
        func_name: str | None,
        debug: bool,
        label: str = "source",
    ) -> tuple[bool | None, int, int, int, str | None, list[tuple], float]:
        """
        Run all tests for one response.

        Returns:
            all_passed      bool | None  — None means skipped (no code / no tests)
            tests_passed    int
            tests_failed    int
            tests_total     int
            code            str | None   — extracted code block
            test_results    list[tuple]  — raw (passed, actual, error, elapsed) per test
            total_elapsed   float        — cumulative wall-clock seconds across all tests
        """
        code = extract_code(response)
        if not code:
            print(f"  [{label}] SKIP — no code block found")
            if debug:
                print(f"    Response: {response[:200]}")
            return None, 0, 0, 0, None, [], 0.0

        tests = get_all_tests(entry)
        if not tests:
            print(f"  [{label}] SKIP — no test cases")
            return None, 0, 0, 0, None, [], 0.0

        tests_passed = 0
        tests_failed = 0
        test_results: list[tuple] = []
        total_elapsed = 0.0

        for i, test in enumerate(tests):
            testtype = test.get("testtype", "stdin")
            passed, actual, error, elapsed = run_test(code, test, func_name, debug=debug)
            test_results.append((passed, actual, error, elapsed))
            total_elapsed += elapsed

            if passed:
                tests_passed += 1
            else:
                tests_failed += 1
                print(f"  [{label}] Test {i+1} [{testtype}]: FAIL")
                if error:
                    print(f"    Error:    {error[:200]}")
                else:
                    print(f"    Expected: {test.get('output', '')[:80].strip()}")
                    print(f"    Got:      {actual[:80].strip()}")

        all_passed = tests_failed == 0
        total = tests_passed + tests_failed
        print(f"  [{label}] {tests_passed}/{total} passed  ({total_elapsed:.2f}s)")
        return all_passed, tests_passed, tests_failed, total, code, test_results, total_elapsed


    def run_tests(
        results_path: str,
        n: int = None,
        debug: bool = False,
        followup_only: bool = False,
        csv_path: str | None = None,
        write_outputs: bool = False,
    ) -> list[dict]:
        with open(results_path, encoding="utf-8") as f:
            results = json.load(f)

        if n:
            results = results[:n]

        total = passed = failed = skipped = 0
        mr_holds_count = mr_violated_count = 0
        has_mr = any(r.get("relation") for r in results) and not followup_only
        metric_rows = []

        for entry in results:
            qid = entry["question_id"]
            print(f"\n{'='*60}")
            print(f"  {qid} — {entry['question_title']} ({entry['difficulty']} / {entry['platform']})")
            print(f"{'='*60}")

            total += 1

            if entry.get("error"):
                print(f"  [SKIP] LLM error: {entry['error']}")
                skipped += 1
                continue

            func_name = get_func_name(entry)

            # Pull token fields from entry (written by runner.py)
            src_tokens = compute_token_count({
                "prompt_tokens": entry.get("prompt_tokens"),
                "completion_tokens": entry.get("completion_tokens"),
                "total_tokens": entry.get("total_tokens"),
                "thinking_tokens": entry.get("thinking_tokens"),
            })
            src_token_efficiency = compute_token_efficiency({
                "completion_tokens": entry.get("completion_tokens"),
                "thinking_tokens": entry.get("thinking_tokens"),
            })
            src_llm_call_time = entry.get("llm_call_time")

            flw_tokens = compute_token_count({
                "prompt_tokens": entry.get("followup_prompt_tokens"),
                "completion_tokens": entry.get("followup_completion_tokens"),
                "total_tokens": entry.get("followup_total_tokens"),
                "thinking_tokens": entry.get("followup_thinking_tokens"),
            })
            flw_token_efficiency = compute_token_efficiency({
                "completion_tokens": entry.get("followup_completion_tokens"),
                "thinking_tokens": entry.get("followup_thinking_tokens"),
            })
            flw_llm_call_time = entry.get("followup_llm_call_time")

            # ── followup-only mode ───────────────────────────────────────────────
            if followup_only:
                primary_response = entry.get("followup_response") or entry.get("response", "")
                label = "followup" if entry.get("followup_response") else "source"
                all_passed, tp, tf, tt, code, test_results, run_time = test_response(
                    primary_response, entry, func_name, debug, label=label
                )
                if all_passed is None:
                    skipped += 1
                    continue
                if all_passed:
                    passed += 1
                else:
                    failed += 1

                metrics = compute_metrics(code)
                sample_key = "followup_response_samples" if label == "followup" else "response_samples"
                primary_key = "followup_response" if label == "followup" else "response"
                variability = compute_pairwise_code_variability(
                    _code_samples(_response_samples(entry, sample_key, primary_key))
                )
                metric_rows.append({
                    "id": qid,
                    "question_title": entry.get("question_title"),
                    "difficulty": entry.get("difficulty"),
                    "platform": entry.get("platform"),
                    "source_extractable": compute_extractable_code(primary_response),
                    "source_passes_tests": compute_passes_tests(test_results),
                    "source_passed": all_passed,
                    "tests_passed": tp, "tests_failed": tf, "tests_total": tt,
                    "source_test_run_time": round(run_time, 4),
                    "source_ncloc": metrics["ncloc"],
                    "source_cc": metrics["cyclomatic_complexity"],
                    "source_variability": variability,
                    "source_prompt_tokens": src_tokens["prompt_tokens"],
                    "source_completion_tokens": src_tokens["completion_tokens"],
                    "source_total_tokens": src_tokens["total_tokens"],
                    "source_thinking_tokens": src_tokens["thinking_tokens"],
                    "source_token_efficiency": src_token_efficiency,
                    "source_llm_call_time": src_llm_call_time,
                })
                continue

            # ── Source ──────────────────────────────────────────────────────────
            src_response = entry["response"]
            src_passed, src_tp, src_tf, src_tt, src_code, src_test_results, src_run_time = test_response(
                src_response, entry, func_name, debug, label="source"
            )

            if src_passed is None:
                skipped += 1
                continue
            if src_passed:
                passed += 1
            else:
                failed += 1

            src_metrics = compute_metrics(src_code)
            src_var = compute_pairwise_code_variability(
                _code_samples(_response_samples(entry, "response_samples", "response"))
            )

            row: dict = {
                "id": qid,
                "question_title": entry.get("question_title"),
                "difficulty": entry.get("difficulty"),
                "platform": entry.get("platform"),

                # Source test metrics
                "source_extractable": compute_extractable_code(src_response),
                "source_passes_tests": compute_passes_tests(src_test_results),
                "source_passed": src_passed,
                "tests_passed": src_tp, "tests_failed": src_tf, "tests_total": src_tt,
                "source_test_run_time": round(src_run_time, 4),

                # Source code metrics
                "source_ncloc": src_metrics["ncloc"],
                "source_cc": src_metrics["cyclomatic_complexity"],
                "source_variability": src_var,

                # Source token metrics
                "source_prompt_tokens": src_tokens["prompt_tokens"],
                "source_completion_tokens": src_tokens["completion_tokens"],
                "source_total_tokens": src_tokens["total_tokens"],
                "source_thinking_tokens": src_tokens["thinking_tokens"],
                "source_token_efficiency": src_token_efficiency,
                "source_llm_call_time": src_llm_call_time,

                # Followup placeholders
                "followup_extractable": None,
                "followup_passes_tests": None,
                "passes_same_tests": None,
                "followup_passed": None,
                "followup_tests_passed": None, "followup_tests_failed": None, "followup_tests_total": None,
                "followup_test_run_time": None,
                "followup_ncloc": None, "followup_cc": None,
                "followup_variability": None,
                "followup_prompt_tokens": None,
                "followup_completion_tokens": None,
                "followup_total_tokens": None,
                "followup_thinking_tokens": None,
                "followup_token_efficiency": None,
                "followup_llm_call_time": None,
                "code_sim": None, "mr_holds": None,
            }

            # ── Followup ─────────────────────────────────────────────────────────
            if entry.get("relation") and entry.get("followup_response"):
                flw_response = entry["followup_response"]
                flw_passed, flw_tp, flw_tf, flw_tt, flw_code, flw_test_results, flw_run_time = test_response(
                    flw_response, entry, func_name, debug, label="followup"
                )

                if flw_passed is not None:
                    holds = src_passed == flw_passed
                    verdict = "HOLDS" if holds else "VIOLATED"
                    print(f"  MR [{entry['relation']}]: {verdict}")
                    if holds:
                        mr_holds_count += 1
                    else:
                        mr_violated_count += 1
                    row["mr_holds"] = holds
                    row["followup_passed"] = flw_passed
                    row["followup_tests_passed"] = flw_tp
                    row["followup_tests_failed"] = flw_tf
                    row["followup_tests_total"] = flw_tt
                    row["followup_test_run_time"] = round(flw_run_time, 4)

                row["followup_extractable"] = compute_extractable_code(flw_response)
                row["followup_passes_tests"] = compute_passes_tests(flw_test_results)
                row["passes_same_tests"] = compute_passes_same_tests(src_test_results, flw_test_results)

                flw_metrics = compute_metrics(flw_code)
                row["followup_ncloc"] = flw_metrics["ncloc"]
                row["followup_cc"] = flw_metrics["cyclomatic_complexity"]
                row["followup_variability"] = compute_pairwise_code_variability(
                    _code_samples(_response_samples(entry, "followup_response_samples", "followup_response"))
                )
                row["followup_prompt_tokens"] = flw_tokens["prompt_tokens"]
                row["followup_completion_tokens"] = flw_tokens["completion_tokens"]
                row["followup_total_tokens"] = flw_tokens["total_tokens"]
                row["followup_thinking_tokens"] = flw_tokens["thinking_tokens"]
                row["followup_token_efficiency"] = flw_token_efficiency
                row["followup_llm_call_time"] = flw_llm_call_time

                if src_code and flw_code:
                    row["code_sim"] = compute_code_similarity(src_code, flw_code)

            metric_rows.append(row)

        # ── Summary ──────────────────────────────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"  Results: {Path(results_path).name}")
        print(f"  Model:   {results[0]['model_name'] if results else 'unknown'}")
        print(f"  PASSED:  {passed}/{total}")
        print(f"  FAILED:  {failed}/{total}")
        print(f"  SKIPPED: {skipped}/{total}")
        if has_mr:
            mr_total = mr_holds_count + mr_violated_count
            print(f"  MR [{results[0].get('relation')}] HOLDS:    {mr_holds_count}/{mr_total}")
            print(f"  MR [{results[0].get('relation')}] VIOLATED: {mr_violated_count}/{mr_total}")

        if metric_rows:
            print()
            if has_mr:
                print(
                    f"  {'ID':<15} {'Src':>6} {'Flw':>6} {'SameTst':>7} {'NCLOC':>6} {'dNCL':>5} "
                    f"{'CC':>6} {'dCC':>5} {'Sim':>7} {'SrcDst':>7} {'FlwDst':>7} {'Var>':>5} "
                    f"{'TkEff':>6} {'LLMs':>6} {'MR':>8}"
                )
                print(
                    f"  {'-'*15} {'-'*6} {'-'*6} {'-'*7} {'-'*6} {'-'*5} "
                    f"{'-'*6} {'-'*5} {'-'*7} {'-'*7} {'-'*7} {'-'*5} "
                    f"{'-'*6} {'-'*6} {'-'*8}"
                )
                for r in metric_rows:
                    src_var = r.get("source_variability") or {}
                    flw_var = r.get("followup_variability") or {}
                    src_dist = src_var.get("mean_distance")
                    flw_dist = flw_var.get("mean_distance")
                    ncloc_str = f"{r['source_ncloc']}/{r['followup_ncloc'] if r['followup_ncloc'] is not None else 'N/A'}"
                    cc_str = f"{_fmt(r['source_cc'], 1)}/{_fmt(r['followup_cc'], 1) if r['followup_cc'] is not None else 'N/A'}"
                    mr = ("HOLDS" if r["mr_holds"] else "VIOLATED") if r["mr_holds"] is not None else "N/A"
                    src_res = f"{r['tests_passed']}/{r['tests_total']}" if r.get("tests_total") else "N/A"
                    flw_res = (
                        f"{r.get('followup_tests_passed', '?')}/{r.get('followup_tests_total', '?')}"
                        if r.get("followup_tests_total") else "N/A"
                    )
                    same_tst = (
                        "YES" if r.get("passes_same_tests") is True
                        else "NO" if r.get("passes_same_tests") is False
                        else "N/A"
                    )
                    eff = _fmt(r.get("source_token_efficiency"), 3) if r.get("source_token_efficiency") is not None else "N/A"
                    llm_t = f"{r['source_llm_call_time']:.1f}s" if r.get("source_llm_call_time") is not None else "N/A"
                    d_ncloc = _delta(
                        r["followup_ncloc"] - r["source_ncloc"]
                        if r["followup_ncloc"] is not None and r["source_ncloc"] >= 0 and r["followup_ncloc"] >= 0
                        else "N/A"
                    )
                    d_cc = _delta(
                        round(r["followup_cc"] - r["source_cc"], 2)
                        if r["followup_cc"] is not None and r["source_cc"] >= 0 and r["followup_cc"] >= 0
                        else "N/A"
                    )
                    print(
                        f"  {r['id']:<15} {src_res:>6} {flw_res:>6} {same_tst:>7} {ncloc_str:>6} {d_ncloc:>5} "
                        f"{cc_str:>6} {d_cc:>5} "
                        f"{_fmt(r.get('code_sim'), 4):>7} {_fmt(src_dist, 4):>7} {_fmt(flw_dist, 4):>7} "
                        f"{_variation_greater(src_dist, flw_dist):>5} "
                        f"{eff:>6} {llm_t:>6} {mr:>8}"
                    )
            else:
                print(
                    f"  {'ID':<15} {'Tests':>8} {'Extract':>7} {'NCLOC':>6} {'CC':>6} "
                    f"{'MeanDst':>8} {'StdDst':>7} {'TkEff':>6} {'LLMs':>6} {'TestT':>6}"
                )
                print(
                    f"  {'-'*15} {'-'*8} {'-'*7} {'-'*6} {'-'*6} "
                    f"{'-'*8} {'-'*7} {'-'*6} {'-'*6} {'-'*6}"
                )
                for r in metric_rows:
                    src_var = r.get("source_variability") or {}
                    tests_str = f"{r.get('tests_passed', '?')}/{r.get('tests_total', '?')}"
                    ext = "YES" if r.get("source_extractable") else "NO"
                    eff = _fmt(r.get("source_token_efficiency"), 3) if r.get("source_token_efficiency") is not None else "N/A"
                    llm_t = f"{r['source_llm_call_time']:.1f}s" if r.get("source_llm_call_time") is not None else "N/A"
                    test_t = f"{r['source_test_run_time']:.2f}s" if r.get("source_test_run_time") is not None else "N/A"
                    print(
                        f"  {r['id']:<15} {tests_str:>8} {ext:>7} {r['source_ncloc']:>6} {r['source_cc']:>6} "
                        f"{_fmt(src_var.get('mean_distance'), 4):>8} {_fmt(src_var.get('std_distance'), 4):>7} "
                        f"{eff:>6} {llm_t:>6} {test_t:>6}"
                    )

        if csv_path:
            _write_metrics_csv(csv_path, metric_rows, has_mr)

        if write_outputs and results:
            from evaluation.results_writer import write_all_outputs
            from config.settings import RESULTS_DIR
            model_name = results[0]["model_name"]
            relation_name = results[0].get("relation") or None
            relation_id = None
            stem = Path(results_path).stem
            if stem.startswith("mr-"):
                parts = stem.split("-")
                if len(parts) >= 2:
                    try:
                        relation_id = int(parts[1])
                    except ValueError:
                        pass
            write_all_outputs(results, metric_rows, model_name, relation_name, relation_id, RESULTS_DIR)

        print(f"{'='*60}")
        return metric_rows


    class _Tee:
        def __init__(self, *streams):
            self._streams = streams

        def write(self, data):
            for stream in self._streams:
                stream.write(data)

        def flush(self):
            for stream in self._streams:
                stream.flush()


    if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="Run test cases against model responses")
        parser.add_argument("--results", required=True, help="Path to results JSON file")
        parser.add_argument("--n", type=int, default=None, help="Limit to first N problems")
        parser.add_argument("--debug", action="store_true", help="Print extracted code and generated scripts")
        parser.add_argument("--out", type=str, default=None, help="Path to write test output log")
        parser.add_argument("--csv", type=str, default=None, help="Path to write metrics CSV")
        parser.add_argument("--stdout", action="store_true", help="Also print output to the terminal")
        parser.add_argument("--followup-only", action="store_true", help="Test followup responses only")
        parser.add_argument("--write-outputs", action="store_true", help="Write split files and HTML report")
        args = parser.parse_args()

        out_path = Path(args.out) if args.out else Path(args.results).with_suffix(".tests.log")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "w", encoding="utf-8") as f:
            stream = _Tee(sys.stdout, f) if args.stdout else f
            with redirect_stdout(stream), redirect_stderr(stream):
                run_tests(
                    args.results, args.n,
                    debug=args.debug,
                    followup_only=args.followup_only,
                    csv_path=args.csv,
                    write_outputs=args.write_outputs,
                )


---
