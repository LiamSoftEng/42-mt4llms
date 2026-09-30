# 42-mt4llms — Full Codebase Reference

Auto-generated from the working tree. File contents below are stored in
4-space-indented code blocks so any literal ``` inside source is preserved.

- Total files: 38
- Total bytes: 219267

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
evaluation/metric_rows.py
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
llms/gemini_model.py
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
tests/test_regressions.py
```

---

## .env.example

```text
File: .env.example (117 bytes)
```

    ANTHROPIC_API_KEY=your_anthropic_key_here
    OPENAI_API_KEY=your_openai_key_here
    GOOGLE_API_KEY=your_google_key_here


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
File: compare_repeats.py (15818 bytes)
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
    import json
    import sys
    from contextlib import redirect_stdout, redirect_stderr
    from pathlib import Path
    from statistics import mean, stdev
    
    from runtests import Tee, run_tests
    from evaluation.paths import build_results_path
    from evaluation.report import REPORT_CSS, format_duration, pass_badge
    
    
    def _resolve_dir(model: str | None, dir_arg: str | None) -> Path:
        if dir_arg:
            return Path(dir_arg)
        if not model:
            raise SystemExit("Provide --model or --dir")
        return build_results_path(model, None, None).parent / "repeats"
    
    
    def _numeric_mean(values, digits: int | None = None) -> float | None:
        numbers = [value for value in values if isinstance(value, (int, float)) and value != -1]
        if not numbers:
            return None
        result = mean(numbers)
        return round(result, digits) if digits is not None else result
    
    
    def _fmt_number(value: float | None) -> str:
        if value is None:
            return "N/A"
        return f"{value:,.1f}"
    
    
    _QUESTION_METRIC_KEYS = ("pass", "ncloc", "cc", "tokens", "call_time")
    
    
    def _question_row(entry: dict, completed_runs: int) -> dict:
        """Create an aligned per-question row, prefilled for earlier repeats."""
        return {
            "question_id": entry.get("question_id") or entry["id"],
            "question_title": entry.get("question_title"),
            "difficulty": entry.get("difficulty"),
            "platform": entry.get("platform"),
            **{key: [None] * completed_runs for key in _QUESTION_METRIC_KEYS},
        }
    
    
    def _record_repeat_questions(
        per_question: dict[str, dict],
        raw_results: list[dict],
        metric_rows: list[dict],
        run_index: int,
    ) -> None:
        """Add one repeat while preserving an explicit slot for every question."""
        raw_by_id = {
            entry.get("question_id"): entry
            for entry in raw_results
            if isinstance(entry, dict) and entry.get("question_id")
        }
        metrics_by_id = {row["id"]: row for row in metric_rows}
    
        for q in per_question.values():
            for key in _QUESTION_METRIC_KEYS:
                q[key].append(None)
    
        for qid in dict.fromkeys([*raw_by_id, *metrics_by_id]):
            entry = metrics_by_id.get(qid) or raw_by_id[qid]
            if qid not in per_question:
                per_question[qid] = _question_row(entry, run_index + 1)
    
            row = metrics_by_id.get(qid)
            if row is None:
                continue
            q = per_question[qid]
            q["pass"][run_index] = row.get("source_passed")
            q["ncloc"][run_index] = row.get("source_ncloc")
            q["cc"][run_index] = row.get("source_cc")
            q["tokens"][run_index] = row.get("source_total_tokens")
            q["call_time"][run_index] = row.get("source_llm_call_time")
    
    
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
    
        mean_pass_text = f"{mean_pass * 100:.1f}%" if mean_pass is not None else "N/A"
        mean_extract_text = f"{mean_extract * 100:.1f}%" if mean_extract is not None else "N/A"
    
        stats_html = f"""
            <div class="stat-box">
                <div class="stat-label">Runs compared</div>
                <div class="stat-value">{n_runs}</div>
                <div class="stat-sub">repeat-*.json files</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Mean pass rate</div>
                <div class="stat-value">{mean_pass_text}</div>
                <div class="stat-sub">across {n_runs} runs</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Mean extractable</div>
                <div class="stat-value">{mean_extract_text}</div>
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
                <div class="stat-value">{format_duration(mean_call)}</div>
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
                    <td>{format_duration(r['mean_llm_call_time'])}</td>
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
                f"<td>{pass_badge(p)}</td>" for p in passes
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
                    <td>{_fmt_number(_numeric_mean(q["ncloc"]))}</td>
                    <td>{_fmt_number(_numeric_mean(q["cc"]))}</td>
                    <td>{_fmt_number(_numeric_mean(q["tokens"]))}</td>
                    <td>{format_duration(_numeric_mean(q["call_time"]))}</td>
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
    {REPORT_CSS}
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
    
        for run_index, f in enumerate(files):
            with open(f, encoding="utf-8") as results_fh:
                raw_results = json.load(results_fh)
    
            log_path = f.with_suffix(".tests.log")
            with open(log_path, "w", encoding="utf-8") as log_fh:
                with redirect_stdout(Tee(sys.stdout, log_fh)), redirect_stderr(Tee(sys.stdout, log_fh)):
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
                "mean_ncloc": _numeric_mean((row.get("source_ncloc") for row in metric_rows), 3),
                "mean_cc": _numeric_mean((row.get("source_cc") for row in metric_rows), 3),
                "mean_total_tokens": _numeric_mean((row.get("source_total_tokens") for row in metric_rows), 3),
                "mean_llm_call_time": _numeric_mean((row.get("source_llm_call_time") for row in metric_rows), 3),
            })
    
            # run_tests deliberately omits rows that cannot be evaluated (for
            # example an LLM error or a response with no extractable code). Build
            # the question set from the raw results as well, so those omissions
            # remain in their correct repeat column and render as N/A.
            _record_repeat_questions(per_question, raw_results, metric_rows, run_index)
    
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
            writer.writerows(
                {key: "N/A" if value is None else value for key, value in row.items()}
                for row in per_repeat
            )
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
File: config/run_config.py (1548 bytes)
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
        "n": 4,
        "after": None,
        "difficulty": ["easy"],
        "mr": [1],  # Use a list of IDs, one ID, "all", or None.
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
File: config/settings.py (2741 bytes)
```

    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    HUGGINGFACE_ENV_DEFAULTS = {
        "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        "HUGGINGFACE_HUB_DISABLE_PROGRESS_BARS": "1",
        "TRANSFORMERS_VERBOSITY": "error",
        "TRANSFORMERS_NO_ADVISORY_WARNINGS": "1",
    }
    
    
    def configure_huggingface_environment() -> None:
        for name, value in HUGGINGFACE_ENV_DEFAULTS.items():
            os.environ.setdefault(name, value)
    
    
    configure_huggingface_environment()
    
    # API Keys
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    
    # Model names
    MR_HELPER_MODEL = "gpt-5-nano"
    CLAUDE_MODEL = "claude-sonnet-4-6"
    OPENAI_MODEL = "gpt-5-nano"
    GEMINI_MODEL = "gemini-2.5-flash"
    CODE_EMBEDDING_MODEL = "Salesforce/codet5p-220m"
    CODE_TOKENIZER_MODEL = "Salesforce/codet5-base"
    
    # Generation parameters
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
    }
    
    # ---------------------------------------------------------------------------
    # Metamorphic-relation violation detection (per-question t-test baseline)
    # ---------------------------------------------------------------------------
    # Paths to repeat-only (no-transformation) run folders, used as a
    # per-question baseline distribution for t-test-based violation detection.
    # Each entry is a directory containing repeat-*.json files (same format
    # PipelineRunner writes under base/repeats/); every file contributes one
    # independent sample per question. Directories may cover disjoint question
    # sets (here, split by difficulty) — merged by question_id when building
    # the baseline. See evaluation.metrics.load_repeat_baseline.
    REPEAT_BASELINE_DIRS = [
        "results/gpt-5-6-sol/base/repeats1_hard/repeats",
        "results/gpt-5-6-sol/base/repeats2_medium/repeats",
        "results/gpt-5-6-sol/base/repeats3_easy/repeats",
    ]
    
    # Significance level for the one-sample t-test in
    # evaluation.metrics.compute_ttest_violation. A followup metric value is
    # flagged as a violation when its p-value is below this.
    TTEST_ALPHA = 0.05


---

## cost_tracker.py

```text
File: cost_tracker.py (3657 bytes)
```

    from __future__ import annotations
    
    import json
    import sys
    from datetime import datetime, timezone
    from pathlib import Path
    
    from config.settings import COST_RATES_PER_MILLION, COST_TRACKING_PATH
    
    _WARNED = False
    _MODEL_CALL_COUNTS = {"main": 0, "helper": 0}
    
    
    def reset_model_call_counts() -> None:
        for key in _MODEL_CALL_COUNTS:
            _MODEL_CALL_COUNTS[key] = 0
    
    
    def record_model_call(kind: str) -> None:
        _MODEL_CALL_COUNTS[kind] += 1
    
    
    def get_model_call_counts() -> dict[str, int]:
        return dict(_MODEL_CALL_COUNTS)
    
    
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
File: data/loader.py (3182 bytes)
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
File: evaluation/harness/functional.py (3050 bytes)
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
    
    
    def run(code: str, func_name: str, test_input: str, timeout: int = 100, debug: bool = False) -> tuple[str, str]:
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
File: evaluation/harness/stdin.py (1298 bytes)
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
    
    
    def run(code: str, test_input: str, timeout: int = 100) -> tuple[str, str]:
        """
        Execute code with the given stdin input.
    
        Returns:
            The subprocess's ``(stdout, stderr)`` text.
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

## evaluation/metric_rows.py

```text
File: evaluation/metric_rows.py (5985 bytes)
```

    """Shared serializers for the metric rows produced by ``runtests.py``."""
    
    from evaluation.metric_violations import compute_violation_field
    
    BASE_CSV_FIELDS = [
        "id",
        "extractable_code", "passes_tests", "source_passed",
        "tests_passed", "tests_failed", "tests_total", "test_run_time",
        "source_ncloc", "source_cc",
        "source_mean_distance", "source_std_distance", "source_pairs",
        "prompt_tokens", "completion_tokens", "total_tokens",
        "thinking_tokens", "token_efficiency", "llm_call_time",
    ]
    
    FOLLOWUP_CSV_FIELDS = [
        "followup_extractable", "followup_passes_tests", "passes_same_tests", "followup_passed",
        "followup_tests_passed", "followup_tests_failed", "followup_tests_total",
        "followup_test_run_time", "followup_ncloc", "followup_cc",
        "followup_mean_distance", "followup_std_distance", "followup_pairs",
        "followup_prompt_tokens", "followup_completion_tokens", "followup_total_tokens",
        "followup_thinking_tokens", "followup_token_efficiency", "followup_llm_call_time",
        "code_sim", "mr_holds",
    ]
    
    
    def serialize_metric_side(row: dict, side: str) -> dict:
        """Convert an internal metric row to the stable per-side results schema."""
        if side not in {"source", "followup"}:
            raise ValueError(f"Unknown metric side: {side}")
    
        followup = side == "followup"
        prefix = "followup_" if followup else "source_"
        variability = row.get(f"{prefix}variability") or {}
        tests_prefix = "followup_" if followup else ""
    
        output = {
            "question_id": row["id"],
            "question_title": row.get("question_title"),
            "difficulty": row.get("difficulty"),
            "platform": row.get("platform"),
            "extractable_code": row.get(f"{prefix}extractable"),
            "passes_tests": row.get(f"{prefix}passes_tests"),
            "passed": row.get(f"{prefix}passed"),
            "tests_passed": row.get(f"{tests_prefix}tests_passed"),
            "tests_failed": row.get(f"{tests_prefix}tests_failed"),
            "tests_total": row.get(f"{tests_prefix}tests_total"),
            "test_run_time": row.get(f"{prefix}test_run_time"),
            "ncloc": row.get(f"{prefix}ncloc"),
            "cyclomatic_complexity": row.get(f"{prefix}cc"),
            "mean_distance": variability.get("mean_distance"),
            "std_distance": variability.get("std_distance"),
            "n_pairs": variability.get("n_pairs"),
            "prompt_tokens": row.get(f"{prefix}prompt_tokens"),
            "completion_tokens": row.get(f"{prefix}completion_tokens"),
            "total_tokens": row.get(f"{prefix}total_tokens"),
            "thinking_tokens": row.get(f"{prefix}thinking_tokens"),
            "token_efficiency": row.get(f"{prefix}token_efficiency"),
            "llm_call_time": row.get(f"{prefix}llm_call_time"),
        }
        if followup:
            output.update({
                "passes_same_tests": row.get("passes_same_tests"),
                "mr_holds": row.get("mr_holds"),
                "code_sim": row.get("code_sim"),
            })
            # Tri-state overall verdict for this row: "true" (violated), "false"
            # (clean), or "n/a" (nothing was comparable — no test verdict and no
            # repeat baseline for this question). See
            # evaluation.metric_violations.compute_violation_field.
            output["violation"] = compute_violation_field(output["question_id"], output)
        return output
    
    
    def serialize_metric_csv_row(row: dict, has_mr: bool) -> dict:
        """Flatten a metric row to the existing CSV column names."""
        source = serialize_metric_side(row, "source")
        output = {
            "id": source["question_id"],
            "extractable_code": source["extractable_code"],
            "passes_tests": source["passes_tests"],
            "source_passed": source["passed"],
            "tests_passed": source["tests_passed"],
            "tests_failed": source["tests_failed"],
            "tests_total": source["tests_total"],
            "test_run_time": source["test_run_time"],
            "source_ncloc": source["ncloc"],
            "source_cc": source["cyclomatic_complexity"],
            "source_mean_distance": source["mean_distance"],
            "source_std_distance": source["std_distance"],
            "source_pairs": source["n_pairs"],
            "prompt_tokens": source["prompt_tokens"],
            "completion_tokens": source["completion_tokens"],
            "total_tokens": source["total_tokens"],
            "thinking_tokens": source["thinking_tokens"],
            "token_efficiency": source["token_efficiency"],
            "llm_call_time": source["llm_call_time"],
        }
        if has_mr:
            followup = serialize_metric_side(row, "followup")
            output.update({
                "followup_extractable": followup["extractable_code"],
                "followup_passes_tests": followup["passes_tests"],
                "passes_same_tests": followup["passes_same_tests"],
                "followup_passed": followup["passed"],
                "followup_tests_passed": followup["tests_passed"],
                "followup_tests_failed": followup["tests_failed"],
                "followup_tests_total": followup["tests_total"],
                "followup_test_run_time": followup["test_run_time"],
                "followup_ncloc": followup["ncloc"],
                "followup_cc": followup["cyclomatic_complexity"],
                "followup_mean_distance": followup["mean_distance"],
                "followup_std_distance": followup["std_distance"],
                "followup_pairs": followup["n_pairs"],
                "followup_prompt_tokens": followup["prompt_tokens"],
                "followup_completion_tokens": followup["completion_tokens"],
                "followup_total_tokens": followup["total_tokens"],
                "followup_thinking_tokens": followup["thinking_tokens"],
                "followup_token_efficiency": followup["token_efficiency"],
                "followup_llm_call_time": followup["llm_call_time"],
                "code_sim": followup["code_sim"],
                "mr_holds": followup["mr_holds"],
            })
        return output


---

## evaluation/metric_violations.py

```text
File: evaluation/metric_violations.py (5633 bytes)
```

    """
    Shared metric-violation checking logic.
    
    Used by both results_writer.py (violations.json log), report.py (per-cell
    highlighting in the HTML reports), and metric_rows.py (the "violation" field
    on each followup.results.json entry) — so all three always agree on what
    counts as a metamorphic-relation violation. Lives in its own module rather
    than in results_writer.py (its original home) because report.py needs it
    too, and results_writer.py already imports from report.py — report.py
    importing back from results_writer.py would be circular.
    
    Violation detection is a per-question one-sample t-test: each followup
    metric value is tested against a baseline distribution built from repeat
    (no-transformation) runs for that same question (config.settings.
    REPEAT_BASELINE_DIRS), rather than a fixed +/- delta threshold. See
    evaluation.metrics.load_repeat_baseline and
    evaluation.metrics.compute_ttest_violation.
    """
    
    from config.settings import TTEST_ALPHA
    from evaluation.metrics import load_repeat_baseline, compute_ttest_violation
    
    # Metric names checked against the repeat baseline. Must match both the
    # field names in evaluation.metrics.load_repeat_baseline()'s output and the
    # keys on the followup row produced by evaluation.metric_rows.serialize_metric_side.
    _METRIC_NAMES = ("ncloc", "cyclomatic_complexity", "llm_call_time", "total_tokens", "token_efficiency", "code_sim")
    
    # scipy.stats.ttest_1samp treats the repeat baseline as the sample and the
    # follow-up value as popmean. Structural/resource metrics use two-sided tests
    # when a change in either direction is relevant. For directional performance
    # metrics, ``less`` detects a slower follow-up and ``greater`` detects lower
    # efficiency or similarity.
    _METRIC_ALTERNATIVES = {
        "ncloc": "two-sided",
        "cyclomatic_complexity": "two-sided",
        "llm_call_time": "less",
        "total_tokens": "two-sided",
        "token_efficiency": "greater",
        "code_sim": "greater",
    }
    
    # Static-metric error sentinels: compute_metrics() returns these instead of a
    # real value when code couldn't be parsed. They're valid ints but meaningless
    # as measurements, so they must never be handed to the t-test as a real
    # followup value.
    _ERROR_SENTINELS = {
        "ncloc": {-1},
        "cyclomatic_complexity": {-1, -2},
    }
    
    
    def _is_valid_metric_value(metric_name: str, value) -> bool:
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and value not in _ERROR_SENTINELS.get(metric_name, set())
        )
    
    
    def check_metric_violations(question_id: str, followup_row: dict) -> dict:
        """
        Test each of _METRIC_NAMES on the followup row against that question's
        repeat-run baseline distribution via a one-sample t-test.
    
        Returns a dict of {metric_name: ttest_result} for every metric whose
        t-test came back significant (p < config.settings.TTEST_ALPHA).
        ttest_result is the dict returned by compute_ttest_violation (n_baseline,
        baseline_mean, baseline_std, followup, t_stat, p_value, alternative,
        significant).
    
        A metric is silently skipped (not counted as a violation) when there's
        no baseline for this question at all, the followup value is a static-
        metric error sentinel (-1/-2, meaning the code didn't parse), fewer than
        two baseline samples exist for that specific metric, or the followup
        value is otherwise non-numeric.
        """
        baseline = load_repeat_baseline()
        q_baseline = baseline.get(question_id)
        if not q_baseline:
            return {}
    
        violations: dict = {}
        for metric_name in _METRIC_NAMES:
            followup_value = followup_row.get(metric_name)
            if not _is_valid_metric_value(metric_name, followup_value):
                continue
    
            baseline_samples = list(q_baseline.get(metric_name, ()))
            result = compute_ttest_violation(
                baseline_samples,
                followup_value,
                alpha=TTEST_ALPHA,
                alternative=_METRIC_ALTERNATIVES[metric_name],
            )
            if result is not None and result["significant"]:
                violations[metric_name] = result
    
        return violations
    
    
    def compute_violation_field(question_id: str, followup_row: dict) -> str:
        """
        Overall tri-state MR verdict for one followup row, written into the
        "violation" field of followup.results.json entries:
    
          "true"  — the test verdict flipped (mr_holds is False), or at least
                    one metric's t-test against the repeat baseline came back
                    significant.
          "false" — a test verdict was recorded and held (mr_holds is True) and
                    no metric was significant, OR no test verdict exists
                    (mr_holds is None) but the question has a repeat baseline to
                    compare metrics against and none of them were significant.
          "n/a"   — nothing could actually be compared: no test verdict AND no
                    repeat baseline exists for this question at all.
        """
        metric_violations = check_metric_violations(question_id, followup_row)
        mr_holds = followup_row.get("mr_holds")
    
        if mr_holds is False or metric_violations:
            return "true"
        if mr_holds is True:
            return "false"
    
        # mr_holds is None: no test-based comparison was made for this row (e.g.
        # no extractable followup code, or no private tests). Only "false" if we
        # at least had baseline data to check metrics against; otherwise nothing
        # was actually compared.
        if not load_repeat_baseline().get(question_id):
            return "n/a"
        return "false"


---

## evaluation/metrics.py

```text
File: evaluation/metrics.py (19185 bytes)
```

    """
    Code metrics computed on extracted code blocks using radon.
    """
    
    from functools import lru_cache
    import io
    from itertools import combinations
    import os
    from statistics import mean, stdev
    import textwrap
    import tokenize
    import warnings
    
    from radon.raw import analyze
    from radon.complexity import cc_visit
    
    from evaluation.executor import extract_code
    from config.settings import configure_huggingface_environment
    
    
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
        configure_huggingface_environment()
    
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
    
    def _strip_comments_for_codesim(code: str) -> str:
        """Remove Python comments without changing strings or stored code."""
        try:
            filtered = []
            skip_comment_linebreak = False
            for token in tokenize.generate_tokens(io.StringIO(code).readline):
                if token.type == tokenize.COMMENT:
                    skip_comment_linebreak = not token.line[:token.start[1]].strip()
                    continue
                if skip_comment_linebreak and token.type in (tokenize.NL, tokenize.NEWLINE):
                    skip_comment_linebreak = False
                    continue
                filtered.append((token.type, token.string))
            return tokenize.untokenize(filtered)
        except (IndentationError, tokenize.TokenError):
            return code
    
    
    def compute_code_similarity(code_a: str | None, code_b: str | None) -> float | None:
        """
        Compute cosine similarity between two code snippets using embeddings.
        Returns None when inputs are missing or embedding fails.
        """
        if not code_a or not code_b:
            return None
        try:
            emb_a, emb_b = _encode_with_transformers(
                _strip_comments_for_codesim(code_a),
                _strip_comments_for_codesim(code_b),
            )
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
    
    
    def compute_pairwise_similarity_list(codes: list[str | None]) -> list[float]:
        """
        Raw list of pairwise cosine similarities among valid code samples, one
        entry per unique pair. compute_pairwise_code_variability() only returns
        the aggregated mean/std of these values — this exists because the t-test
        baseline for code_sim needs the individual pairwise values themselves,
        not a summary.
        """
        valid_codes = [c for c in codes if c]
        similarities = []
        for code_a, code_b in combinations(valid_codes, 2):
            similarity = compute_code_similarity(code_a, code_b)
            if similarity is not None:
                similarities.append(float(similarity))
        return similarities
    
    
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
    
    
    # ---------------------------------------------------------------------------
    # Repeat baseline / t-test violation detection
    # ---------------------------------------------------------------------------
    
    @lru_cache(maxsize=1)
    def load_repeat_baseline() -> dict[str, dict[str, tuple[float, ...]]]:
        """
        Build {question_id: {metric_name: (values...)}} from the repeat-only
        runs listed in config.settings.REPEAT_BASELINE_DIRS. Each directory is
        scanned for repeat-*.json files; every file contributes one independent
        sample per question. Directories may cover disjoint question sets (here,
        split by difficulty: hard/medium/easy) — merged by question_id across
        all directories.
    
        ncloc/cyclomatic_complexity are computed fresh from each entry's response
        code (never stored anywhere). llm_call_time/total_tokens/token_efficiency
        are read directly since those already exist per-entry.
    
        code_sim is a special case: instead of one value per file, its baseline
        is the full list of pairwise CodeT5 similarities between every pair of
        extracted code samples for that question across all repeat files — i.e.
        "how similar do two independent generations of this question tend to
        be," which the source-vs-followup code_sim score gets tested against.
    
        Cached: computed once per process, since it re-extracts code, recomputes
        NCLOC/CC, and runs embeddings for every repeat file on first access —
        with three difficulty folders this can take a while the first time it's
        called; subsequent calls in the same process are free.
        """
        import json
        from pathlib import Path
        from config.settings import REPEAT_BASELINE_DIRS
    
        metric_keys = ("ncloc", "cyclomatic_complexity", "llm_call_time", "total_tokens", "token_efficiency")
        baseline: dict[str, dict[str, list[float]]] = {}
        codes_by_question: dict[str, list[str | None]] = {}
    
        for dir_path in REPEAT_BASELINE_DIRS:
            for file_path in sorted(Path(dir_path).glob("repeat-*.json")):
                try:
                    with open(file_path, encoding="utf-8") as f:
                        entries = json.load(f)
                except (OSError, json.JSONDecodeError):
                    continue
                if not isinstance(entries, list):
                    continue
    
                for entry in entries:
                    if not isinstance(entry, dict):
                        continue
                    qid = entry.get("question_id")
                    if not qid:
                        continue
    
                    q_metrics = baseline.setdefault(qid, {k: [] for k in metric_keys})
                    codes_by_question.setdefault(qid, [])
    
                    code = extract_code(entry.get("response") or "")
                    codes_by_question[qid].append(code)
    
                    code_metrics = compute_metrics(code)
                    if code_metrics["ncloc"] not in (-1, -2):
                        q_metrics["ncloc"].append(code_metrics["ncloc"])
                    if code_metrics["cyclomatic_complexity"] not in (-1, -2):
                        q_metrics["cyclomatic_complexity"].append(code_metrics["cyclomatic_complexity"])
    
                    llm_call_time = entry.get("llm_call_time")
                    if isinstance(llm_call_time, (int, float)):
                        q_metrics["llm_call_time"].append(llm_call_time)
    
                    total_tokens = entry.get("total_tokens")
                    if isinstance(total_tokens, (int, float)):
                        q_metrics["total_tokens"].append(total_tokens)
    
                    token_efficiency = entry.get("token_efficiency")
                    if isinstance(token_efficiency, (int, float)):
                        q_metrics["token_efficiency"].append(token_efficiency)
    
        result: dict[str, dict[str, tuple[float, ...]]] = {}
        for qid, metrics_dict in baseline.items():
            entry_out = {k: tuple(v) for k, v in metrics_dict.items()}
            entry_out["code_sim"] = tuple(compute_pairwise_similarity_list(codes_by_question.get(qid, [])))
            result[qid] = entry_out
    
        return result
    
    
    def compute_ttest_violation(
        baseline_samples: list[float],
        followup_value,
        alpha: float = 0.05,
        alternative: str = "two-sided",
    ) -> dict | None:
        """
        One-sample t-test: is followup_value consistent with the per-question
        baseline distribution built from repeat (no-transformation) runs?
    
        ``alternative`` uses scipy's orientation, where the repeat baseline is
        the sample and ``followup_value`` is the hypothesised population mean:
        ``less`` detects a follow-up above the baseline, while ``greater``
        detects a follow-up below it.
    
        Returns None if untestable (fewer than 2 baseline samples, or a
        non-numeric followup value) rather than raising or false-flagging.
        """
        if not isinstance(followup_value, (int, float)) or isinstance(followup_value, bool):
            return None
        if len(baseline_samples) < 2:
            return None
        from scipy import stats as _stats
        try:
            t_stat, p_value = _stats.ttest_1samp(
                baseline_samples,
                popmean=followup_value,
                alternative=alternative,
            )
        except Exception:
            return None
    
        return {
            "n_baseline": len(baseline_samples),
            "baseline_mean": round(mean(baseline_samples), 4),
            "baseline_std": round(stdev(baseline_samples), 4) if len(baseline_samples) > 1 else 0.0,
            "followup": followup_value,
            "t_stat": round(float(t_stat), 4),
            "p_value": round(float(p_value), 6),
            "alternative": alternative,
            "significant": bool(p_value < alpha),
        }


---

## evaluation/paths.py

```text
File: evaluation/paths.py (1813 bytes)
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
File: evaluation/report.py (26210 bytes)
```

    """
    Generates HTML comparison reports from base and followup results files.
    Merged reports reuse the same section builder and styles as individual reports.
    """
    
    from pathlib import Path
    import json
    
    from evaluation.paths import safe_name
    from evaluation.metric_violations import check_metric_violations
    
    
    def format_value(value, digits: int = 4) -> str:
        if value is None or value in (-1, -2):
            return "N/A"
        if isinstance(value, float):
            return f"{value:.{digits}f}"
        return str(value)
    
    
    def format_decimal(value, digits: int = 4) -> str:
        """Format numeric terminal-table values with a fixed decimal precision."""
        if value is None or value in (-1, -2):
            return "N/A"
        return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "N/A"
    
    
    def _delta(a, b) -> str:
        if (
            isinstance(a, (int, float))
            and isinstance(b, (int, float))
            and not isinstance(a, bool)
            and not isinstance(b, bool)
            and a not in (-1, -2)
            and b not in (-1, -2)
        ):
            d = b - a
            return f"{d:+.2f}" if isinstance(d, float) else f"{d:+d}"
        return "N/A"
    
    
    def _mr_badge(holds: bool | None, metric_violations: dict | None = None) -> str:
        """
        Badge for the overall MR verdict. A row is VIOLATED if either the test
        verdict diverged (holds is False) or any metric's t-test against its
        per-question repeat baseline came back significant (metric_violations
        non-empty) — even if the test verdict itself still holds.
        """
        metric_violations = metric_violations or {}
        if holds is False or metric_violations:
            names = ", ".join(metric_violations)
            title = f' title="Flagged by t-test: {names}"' if names else ""
            return f'<span class="badge violated"{title}>VIOLATED</span>'
        if holds is True:
            return '<span class="badge holds">HOLDS</span>'
        return '<span class="badge na">N/A</span>'
    
    
    def _violation_span(text: str, detail: dict | None) -> str:
        """Wrap a cell's text in a red span when `detail` (a single entry from
        check_metric_violations) indicates that metric's t-test against its
        per-question repeat baseline came back significant."""
        if not detail:
            return text
        p = detail.get("p_value")
        mean_b = detail.get("baseline_mean")
        std_b = detail.get("baseline_std")
        followup_v = detail.get("followup")
    
        bits = []
        if p is not None:
            bits.append(f"p={p:.4g}")
        if mean_b is not None and std_b is not None:
            bits.append(f"baseline={mean_b:.3g}&plusmn;{std_b:.3g}")
        if isinstance(followup_v, (int, float)):
            bits.append(f"followup={followup_v:.3g}")
        title = f' title="{"; ".join(bits)}"' if bits else ""
        return f'<span class="delta-neg"{title}>{text}</span>'
    
    
    def pass_badge(passed: bool | None) -> str:
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
    
    
    def format_duration(seconds) -> str:
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
    
        # Per-row metric violations, computed once and reused below for both the
        # aggregate stat boxes and the per-row cell highlighting, so the two
        # never disagree on what "violated" means. Each row is checked via a
        # one-sample t-test against that question's repeat-run baseline (see
        # evaluation.metric_violations.check_metric_violations).
        row_metric_violations: dict[str, dict] = {}
        if has_mr:
            for qid in all_ids:
                mv = check_metric_violations(qid, followup.get(qid, {}))
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
        # was flagged by its t-test — previously this only looked at mr_holds, so
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
    
            row_total_time = format_duration(_row_total_time(b, f, has_mr))
    
            # Highlight cells for metrics whose t-test against the repeat baseline
            # came back significant (config.settings.TTEST_ALPHA).
            delta_ncloc_html = _violation_span(_delta(b_ncloc, f_ncloc), metric_violations.get("ncloc"))
            delta_cc_html = _violation_span(_delta(b_cc, f_cc), metric_violations.get("cyclomatic_complexity"))
            code_sim_html = _violation_span(format_value(code_sim), metric_violations.get("code_sim"))
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
                    <td>{b_tests}<br>{pass_badge(b_passed)}</td>
                    <td>{f_tests}<br>{pass_badge(f_passed)}</td>
                    <td>{_mr_badge(mr_holds_val, metric_violations)}<br>{_bool_badge(passes_same, "SAME", "DIFF")}</td>
                    <td>{format_value(b_ncloc, 0)}</td>
                    <td>{format_value(f_ncloc, 0) if f_ncloc is not None else "N/A"}</td>
                    <td>{delta_ncloc_html}</td>
                    <td>{format_value(b_cc)}</td>
                    <td>{format_value(f_cc) if f_cc is not None else "N/A"}</td>
                    <td>{delta_cc_html}</td>
                    <td>{code_sim_html}</td>
                    <td>{format_value(b_mean_dist)}<br><small>±{format_value(b_std_dist)}</small></td>
                    <td>{format_value(f_mean_dist)}<br><small>±{format_value(f_std_dist)}</small></td>
                    <td>{b_tok}<br><small>eff:{format_value(b_eff, 3)}</small></td>
                    <td>{f_tok_html}<br><small>eff:{format_value(f_eff, 3)}</small></td>
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
                    <td>{b_tests}<br>{pass_badge(b_passed)}</td>
                    <td>{format_value(b_ncloc, 0)}</td>
                    <td>{format_value(b_cc)}</td>
                    <td>{format_value(b_mean_dist)}<br><small>±{format_value(b_std_dist)}</small></td>
                    <td>{b_tok}<br><small>eff:{format_value(b_eff, 3)}</small></td>
                    <td>{b_llm_t}<br><small>test:{b_test_t}</small></td>
                </tr>"""
    
        # ── Aggregate stat boxes ─────────────────────────────────────────────────
        base_pct = round(base_passed / base_total * 100) if base_total else 0
        extract_pct = round(base_extractable / base_total * 100) if base_total else 0
    
        mr_section = ""
        followup_stat = ""
    
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
                <div class="stat-sub">Of which {metric_violated_count} flagged by a metric t-test</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Passes same tests</div>
                <div class="stat-value">{same_pct}%</div>
                <div class="stat-sub">{same_tests_count} same / {diff_tests_count} different</div>
            </div>"""
    
        total_time_stat = f"""
        <div class="stat-box">
            <div class="stat-label">Total time</div>
            <div class="stat-value">{format_duration(total_time)}</div>
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
    
    
    REPORT_CSS = """
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
    {REPORT_CSS}
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
    {REPORT_CSS}
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
File: evaluation/results_writer.py (10648 bytes)
```

    """
    Writes split base/followup content files and results JSON files,
    then triggers HTML report generation.
    """
    
    import json
    import time
    from dataclasses import fields
    from pathlib import Path
    
    from evaluation.report import generate_html_report
    from evaluation.paths import build_results_dir, mr_folder_name, safe_name
    from evaluation.metric_violations import check_metric_violations
    from evaluation.metric_rows import serialize_metric_side
    from evaluation.runner import LLMResult
    
    
    _RESULT_FIELD_NAMES = tuple(field.name for field in fields(LLMResult))
    _FOLLOWUP_FIELDS = {
        name for name in _RESULT_FIELD_NAMES
        if name == "relation" or name.startswith("followup_")
    }
    _BASE_FIELDS = set(_RESULT_FIELD_NAMES) - _FOLLOWUP_FIELDS
    _FOLLOWUP_CONTEXT_FIELDS = (
        set(_RESULT_FIELD_NAMES[:_RESULT_FIELD_NAMES.index("error")])
        - {"instruction_prefix", "instruction_suffix", "response"}
    )
    _FOLLOWUP_KEEP_FIELDS = _FOLLOWUP_CONTEXT_FIELDS | _FOLLOWUP_FIELDS
    
    
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
        out_dir = Path(results_dir) / safe_name(model_name) / "base"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "base.json"
    
        stripped = [
            {k: v for k, v in entry.items() if k in _BASE_FIELDS}
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
        folder = mr_folder_name(relation_name, relation_id)
        out_dir = Path(results_dir) / safe_name(model_name) / folder
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "followup.json"
    
        filtered = [
            {k: v for k, v in entry.items() if k in _FOLLOWUP_KEEP_FIELDS}
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
        base_rows = [serialize_metric_side(row, "source") for row in metric_rows]
        write_results_json(base_rows, base_results_path, merge=True)
    
        # ── Followup results JSON ────────────────────────────────────────────────
        followup_results_path = None
        if has_mr:
            folder = mr_folder_name(relation_name, relation_id)
            followup_results_path = (
                Path(results_dir) / safe_name(model_name) / folder / "followup.results.json"
            )
            followup_rows = [
                serialize_metric_side(row, "followup")
                for row in metric_rows
                if row.get("followup_ncloc") is not None
            ]
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
        or a metric flagged by its one-sample t-test against the per-question
        repeat baseline (see evaluation.metric_violations.check_metric_violations).
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
                    metric_violations = check_metric_violations(qid, row)
    
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


---

## evaluation/runner.py

```text
File: evaluation/runner.py (19663 bytes)
```

    import json
    import time
    from dataclasses import asdict, dataclass, field, replace
    from typing import List
    from pathlib import Path
    
    from llms.base import BaseLLM
    from data.loader import build_code_question, build_prompt, INSTRUCTION_PREFIX, INSTRUCTION_SUFFIX
    from evaluation.paths import build_results_path, build_repeat_path, build_repeats_dir
    from config.settings import RESULTS_DIR
    from cost_tracker import record_model_call
    
    
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
    
    
    def _entry_complete_for_resume(entry: dict, relation=None, mr_only: bool = False) -> bool:
        return isinstance(entry.get("question_id"), str) and bool(entry["question_id"])
    
    
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
    
    
    def _token_efficiency(usage: dict | None) -> float | None:
        _, completion_tokens, _, thinking_tokens = _extract_usage(usage)
        if thinking_tokens and completion_tokens:
            return round(completion_tokens / thinking_tokens, 6)
        return None
    
    
    @dataclass
    class GeneratedResponses:
        responses: list[str] = field(default_factory=list)
        usages: list[dict | None] = field(default_factory=list)
        call_times: list[float | None] = field(default_factory=list)
        finish_reasons: list[str | None] = field(default_factory=list)
        error: str = ""
    
        @property
        def primary_response(self) -> str:
            return self.responses[0] if self.responses else ""
    
        @property
        def primary_usage(self) -> dict | None:
            return self.usages[0] if self.usages else None
    
        @property
        def primary_call_time(self) -> float | None:
            return self.call_times[0] if self.call_times else None
    
        @property
        def primary_finish_reason(self) -> str | None:
            return self.finish_reasons[0] if self.finish_reasons else None
    
    
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
        finish_reason: str | None = None
    
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
        followup_finish_reason: str | None = None
    
    
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
    
        def _generate_responses(
            self,
            model: BaseLLM,
            prompt: str,
            token_totals: dict,
        ) -> GeneratedResponses:
            generated = GeneratedResponses()
            for repeat_index in range(self.prompt_repeats):
                try:
                    record_model_call("main")
                    response = model.generate(prompt)
                    usage = model.last_usage
                    call_time = model.last_call_time
                    finish_reason = model.last_finish_reason
                except Exception as exc:
                    response = ""
                    usage = None
                    call_time = None
                    finish_reason = None
                    if repeat_index == 0:
                        generated.error = str(exc)
    
                generated.responses.append(response)
                generated.usages.append(usage)
                generated.call_times.append(call_time)
                generated.finish_reasons.append(finish_reason)
                _accumulate_usage(token_totals, usage)
                if repeat_index < self.prompt_repeats - 1:
                    time.sleep(self.delay)
            return generated
    
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
                    print(f"  Resume enabled: loaded {len(completed_ids)} existing results from {out_path}")
    
                token_totals: dict = {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "thinking_tokens": 0,
                    "reported": False,
                }
    
                base_path = build_results_path(model.model_name, None, None)
                base_entries = _load_existing_results(base_path) if self.relation else []
                base_results = {
                    entry["question_id"]: entry
                    for entry in base_entries
                    if isinstance(entry.get("question_id"), str) and entry["question_id"]
                }
    
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
                    source = GeneratedResponses()
                    base_entry = base_results.get(sample["question_id"]) if self.relation else None
                    base_status = ""
    
                    if base_entry:
                        base_status = "base exists"
                    else:
                        source = self._generate_responses(model, prompt, token_totals)
                        if self.relation:
                            base_status = "base generated"
    
                    response = source.primary_response
                    response_samples = source.responses
                    error = source.error
                    usage = source.primary_usage
                    llm_call_time = source.primary_call_time
                    finish_reason = source.primary_finish_reason
    
                    if base_entry:
                        response = base_entry.get("response", "")
                        response_samples = base_entry.get("response_samples") or ([response] if response else [])
                        error = base_entry.get("error", "")
                        prompt_tokens = base_entry.get("prompt_tokens")
                        completion_tokens = base_entry.get("completion_tokens")
                        total_tokens = base_entry.get("total_tokens")
                        thinking_tokens = base_entry.get("thinking_tokens")
                        token_efficiency = base_entry.get("token_efficiency")
                        llm_call_time = base_entry.get("llm_call_time")
                        finish_reason = base_entry.get("finish_reason")
                    else:
                        prompt_tokens, completion_tokens, total_tokens, thinking_tokens = _extract_usage(usage)
                        token_efficiency = _token_efficiency(usage)
    
                    source_result = LLMResult(
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
                        finish_reason=finish_reason,
                    )
    
                    if self.relation and not base_entry:
                        base_entry = asdict(source_result)
                        base_entries.append(base_entry)
                        base_results[sample["question_id"]] = base_entry
                        self._save_to_path(base_path, base_entries, quiet=True)
    
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
                    followup_finish_reason: str | None = None
    
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
    
                        followup = self._generate_responses(model, followup_prompt, token_totals)
                        followup_response = followup.primary_response
                        followup_response_samples = followup.responses
                        followup_error = followup.error
                        (
                            followup_prompt_tokens,
                            followup_completion_tokens,
                            followup_total_tokens,
                            followup_thinking_tokens,
                        ) = _extract_usage(followup.primary_usage)
                        followup_token_efficiency = _token_efficiency(followup.primary_usage)
                        followup_llm_call_time = followup.primary_call_time
                        followup_finish_reason = followup.primary_finish_reason
    
                        if self.mr_only:
                            print(
                                f"ok ({base_status}, mr-only + {self.relation.name} "
                                f"[{target}], repeats={self.prompt_repeats})"
                            )
                        else:
                            print(
                                f"ok ({base_status} + {self.relation.name} "
                                f"[{target}], repeats={self.prompt_repeats})"
                            )
                    else:
                        print(f"ok (repeats={self.prompt_repeats})" if self.prompt_repeats > 1 else "ok")
    
                    result = replace(
                        source_result,
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
                        followup_finish_reason=followup_finish_reason,
                    )
                    result_entry = asdict(result)
                    results = [entry for entry in results if entry.get("question_id") != sample["question_id"]]
                    results.append(result_entry)
                    completed_ids.add(sample["question_id"])
                    self._save_to_path(out_path, results, quiet=True)
    
                    # ── Per-repeat result files (base run only, repeats > 1) ───────
                    if write_repeat_files:
                        for idx, rep_response in enumerate(response_samples):
                            rep_usage = source.usages[idx] if idx < len(source.usages) else None
                            rep_ptok, rep_ctok, rep_ttok, rep_thtok = _extract_usage(rep_usage)
                            rep_entry = asdict(replace(
                                result,
                                response=rep_response,
                                response_samples=[],
                                error=error if idx == 0 else "",
                                prompt_tokens=rep_ptok,
                                completion_tokens=rep_ctok,
                                total_tokens=rep_ttok,
                                thinking_tokens=rep_thtok,
                                token_efficiency=_token_efficiency(rep_usage),
                                llm_call_time=source.call_times[idx] if idx < len(source.call_times) else None,
                                finish_reason=(
                                    source.finish_reasons[idx] if idx < len(source.finish_reasons) else None
                                ),
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
File: explore_dataset.py (2012 bytes)
```

    """
    Run this first to inspect the LiveCodeBench dataset structure
    before kicking off the full pipeline.
    
    Usage:
        python explore_dataset.py [--write-preview]
    """
    
    import argparse
    import json
    from datetime import datetime
    from pathlib import Path
    
    from datasets import load_dataset
    from config.settings import LCB_DATASET, LCB_VERSION, RESULTS_DIR
    
    
    def parse_args():
        parser = argparse.ArgumentParser(description="Inspect the LiveCodeBench dataset")
        parser.add_argument(
            "--write-preview",
            action="store_true",
            help="Write the first ten records to RESULTS_DIR/dataset_preview.json",
        )
        return parser.parse_args()
    
    
    def write_preview(dataset) -> Path:
        out_dir = Path(RESULTS_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "dataset_preview.json"
        records = []
        for sample in dataset.select(range(min(10, len(dataset)))):
            records.append({
                key: value.isoformat() if isinstance(value, datetime) else value
                for key, value in sample.items()
            })
    
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        return out_path
    
    
    def main() -> None:
        args = parse_args()
        dataset = load_dataset(LCB_DATASET, name=LCB_VERSION, split="test")
    
        print(f"\nTotal samples: {len(dataset)}")
        print(f"Columns: {dataset.column_names}\n")
        if not len(dataset):
            return
    
        sample = dataset[min(20, len(dataset) - 1)]
        for key, value in sample.items():
            if isinstance(value, (list, dict)):
                preview = json.dumps(value, indent=2)[:300] + "..."
            else:
                preview = str(value)[:300]
            print(f"--- {key} ---")
            print(preview)
            print()
    
        if args.write_preview:
            out_path = write_preview(dataset)
            print(f"Saved {min(10, len(dataset))} samples -> {out_path}")
    
    
    if __name__ == "__main__":
        main()


---

## INSTRUCTIONS.md

```text
File: INSTRUCTIONS.md (1489 bytes)
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


---

## llms/__init__.py

```text
File: llms/__init__.py (176 bytes)
```

    from .claude_model import ClaudeModel
    from .openai_model import OpenAIModel
    from .gemini_model import GeminiModel
    
    __all__ = ["ClaudeModel", "OpenAIModel", "GeminiModel"]


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
File: llms/claude_model.py (1474 bytes)
```

    import anthropic
    from .base import BaseLLM
    from config.settings import ANTHROPIC_API_KEY, CLAUDE_MODEL, MAX_TOKENS
    
    
    class ClaudeModel(BaseLLM):
        def __init__(self, model: str = CLAUDE_MODEL):
            self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            self._model_name = model
    
        @property
        def model_name(self) -> str:
            return self._model_name
    
        def generate(self, prompt: str) -> str:
            message = self._record_timed_call(
                lambda: self.client.messages.create(
                    model=self._model_name,
                    max_tokens=MAX_TOKENS,
                    messages=[{"role": "user", "content": prompt}],
                )
            )
            self._set_last_finish_reason(getattr(message, "stop_reason", None))
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

## llms/gemini_model.py

```text
File: llms/gemini_model.py (1834 bytes)
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
            response = self._record_timed_call(
                lambda: self.client.models.generate_content(
                    model=self._model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        max_output_tokens=MAX_TOKENS,
                        temperature=TEMPERATURE,
                    ),
                )
            )
            candidates = getattr(response, "candidates", None) or []
            finish_reason = getattr(candidates[0], "finish_reason", None) if candidates else None
            if finish_reason is not None and not isinstance(finish_reason, str):
                finish_reason = getattr(finish_reason, "name", None) or str(finish_reason)
            self._set_last_finish_reason(finish_reason)
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

## llms/mr_helper.py

```text
File: llms/mr_helper.py (2980 bytes)
```

    from openai import OpenAI
    from config.settings import OPENAI_API_KEY, MR_HELPER_MODEL, RESULTS_DIR
    import json
    import time
    from pathlib import Path
    
    from cost_tracker import record_model_call, update_costs
    
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
    
            record_model_call("helper")
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
    
            update_costs(self.model_name, self._last_usage)
            result = (choice.message.content or "").strip()
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
File: llms/openai_model.py (1068 bytes)
```

    from openai import OpenAI
    from .base import BaseLLM
    from config.settings import OPENAI_API_KEY, OPENAI_MODEL, MAX_TOKENS
    
    
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
File: main.py (6020 bytes)
```

    import argparse
    import subprocess
    import sys
    
    from cost_tracker import get_model_call_counts, reset_model_call_counts
    from data.loader import load_livecode_bench
    from evaluation.paths import build_results_path
    from evaluation.runner import PipelineRunner
    from llms import ClaudeModel, GeminiModel, OpenAIModel
    from metamorphic.relations_list import RELATIONS
    
    MODEL_CHOICES = {
        "claude": ClaudeModel,
        "openai": OpenAIModel,
        "gemini": GeminiModel,
    }
    
    def get_relation(mr_id: int):
        if mr_id not in RELATIONS:
            available = list(RELATIONS.keys())
            raise ValueError(f"Unknown MR: {mr_id}. Available: {available}")
        entry = RELATIONS[mr_id]
        return entry["class"]()
    
    
    def parse_mr(value: str) -> int | str:
        if value.lower() == "all":
            return "all"
        try:
            return int(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("MR must be an integer ID or 'all'") from exc
    
    
    def resolve_relation_ids(values: list[int | str] | None) -> list[int]:
        if not values:
            return []
        if "all" in values:
            if len(values) != 1:
                raise ValueError("'all' cannot be combined with individual MR IDs")
            return sorted(RELATIONS)
    
        relation_ids = list(dict.fromkeys(values))
        unknown = [mr_id for mr_id in relation_ids if mr_id not in RELATIONS]
        if unknown:
            raise ValueError(f"Unknown MR(s): {unknown}. Available: {sorted(RELATIONS)}")
        return relation_ids
    
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
            nargs="+",
            type=parse_mr,
            default=None,
            metavar="ID|all",
            help="Apply one or more metamorphic relations. e.g. --mr 1 4 7 or --mr all",
        )
        parser.add_argument(
            "--mr-only",
            action="store_true",
            help="When used with --mr, reuse shared base results and generate any missing bases",
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
    
        try:
            relation_ids = resolve_relation_ids(args.mr)
        except ValueError as exc:
            print(exc)
            return 2
        if args.mr_only and not relation_ids:
            print("--mr-only requires --mr to be set")
            return 2
        if args.repeats < 1:
            print("--repeats must be at least 1")
            return 2
    
        dataset = load_livecode_bench(n_samples=args.n, after=args.after, difficulty=args.difficulty)
        models = [MODEL_CHOICES[name]() for name in args.models]
        runs = relation_ids or [None]
        print(f"Running models: {[model.model_name for model in models]}, MRs: {relation_ids or ['base']}")
        reset_model_call_counts()
    
        try:
            for relation_id in runs:
                relation = get_relation(relation_id) if relation_id is not None else None
                runner = PipelineRunner(
                    models=models,
                    dataset=dataset,
                    relation=relation,
                    relation_id=relation_id,
                    mr_only=args.mr_only,
                    prompt_repeats=args.repeats,
                    resume=args.resume,
                )
                runner.run()
    
                if args.run_tests:
                    for model in models:
                        relation_name = relation.name if relation else None
                        exit_code = _run_tests(
                            model.model_name,
                            relation_name,
                            relation_id,
                            args.mr_only,
                        )
                        if exit_code != 0:
                            return exit_code
        finally:
            call_counts = get_model_call_counts()
            total_calls = call_counts["main"] + call_counts["helper"]
            print(
                f"\nModel calls (total): main={call_counts['main']}, "
                f"helper={call_counts['helper']}, combined={total_calls}"
            )
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
File: metamorphic/relations/base_mr.py (115 bytes)
```

    class BaseMR:
        target = "PD"
    
        def transform(self, text: str) -> str:
            raise NotImplementedError


---

## metamorphic/relations/mrs.py

```text
File: metamorphic/relations/mrs.py (8322 bytes)
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
        lines = raw.strip().splitlines()
        name = lines[0].strip() if lines else ""
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
            if not solve_response.strip():
                return text
            code = extract_code(solve_response)
            if not code:
                # Helper failed to produce a fenced block — fall back to the raw
                # response so the MR doesn't silently return an empty prompt.
                code = solve_response
            explanation = mr_llm_transform(phrases.ROUND_TRIP_EXPLAIN_PROMPT, code)
            return explanation if explanation.strip() else text


---

## metamorphic/relations/phrases.py

```text
File: metamorphic/relations/phrases.py (12649 bytes)
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
File: metamorphic/relations_list.py (3875 bytes)
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
        2:  {"class": MR2,  "name": "constraint_redundant_spec",        "target": "I",  "description": "Insert a related sentence which has no effect on the solution."},
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
File: README.md (13965 bytes)
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
    | `--difficulty LEVEL ...` | Include only `easy`, `medium`, and/or `hard` problems. |
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
File: run_all_mrs.py (2419 bytes)
```

    import json
    import subprocess
    import sys
    from metamorphic.relations_list import RELATIONS
    from config.settings import OPENAI_MODEL
    from evaluation.paths import build_results_path
    
    
    def _load_run_results(model_name: str, relation_name: str, relation_id: int) -> list[dict] | None:
        src = build_results_path(model_name, relation_name, relation_id)
        if not src.exists():
            print(f"Expected results file not found: {src}")
            return None
        with open(src, "r", encoding="utf-8") as f:
            return json.load(f)
    
    
    def main() -> int:
        base_results_path = build_results_path(OPENAI_MODEL, None, None)
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
                    "openai",
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
            run_results = _load_run_results(OPENAI_MODEL, mr_name, mr_id)
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
File: runtests.py (22203 bytes)
```

    """
    Run all test cases against model responses.
    If the results file contains followup responses (from --mr), tests both
    and computes MR verdict.
    
    Usage:
        python runtests.py --results results/openai/base/base.json
        python runtests.py --results results/openai/base/base.json --n 5
        python runtests.py --results results/openai/base/base.json --stdout
        python runtests.py --results results/openai/mr-05-polarity_negative_formulation.json --followup-only
        python runtests.py --results results/openai/base/base.json --csv results/openai/metrics.csv
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
    from evaluation.report import format_decimal
    from evaluation.metric_rows import BASE_CSV_FIELDS, FOLLOWUP_CSV_FIELDS, serialize_metric_csv_row
    
    
    def _response_samples(entry: dict, sample_key: str, primary_key: str) -> list[str]:
        samples = entry.get(sample_key)
        if isinstance(samples, list) and samples:
            return [s for s in samples if isinstance(s, str)]
        primary = entry.get(primary_key, "")
        return [primary] if isinstance(primary, str) and primary else []
    
    
    def _code_samples(responses: list[str]) -> list[str | None]:
        return [extract_code(r) for r in responses]
    
    
    def _delta(value) -> str:
        return f"{value:+}" if isinstance(value, (int, float)) else value
    
    
    def _variation_greater(source_mean_dist, followup_mean_dist) -> str:
        if isinstance(source_mean_dist, (int, float)) and isinstance(followup_mean_dist, (int, float)):
            return "YES" if followup_mean_dist > source_mean_dist else "NO"
        return "N/A"
    
    
    def _write_metrics_csv(csv_path: str | Path, metric_rows: list[dict], has_mr: bool) -> None:
        path = Path(csv_path)
        path.parent.mkdir(parents=True, exist_ok=True)
    
        fieldnames = BASE_CSV_FIELDS + (FOLLOWUP_CSV_FIELDS if has_mr else [])
    
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in metric_rows:
                writer.writerow(serialize_metric_csv_row(row, has_mr))
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
    
    
    def evaluate_response(
        response: str,
        entry: dict,
        func_name: str | None,
        debug: bool,
        label: str,
        sample_key: str,
        primary_key: str,
    ) -> dict:
        """Execute and measure one source or followup response."""
        passed, tests_passed, tests_failed, tests_total, code, test_results, run_time = test_response(
            response, entry, func_name, debug, label=label
        )
        metrics = compute_metrics(code)
        return {
            "passed": passed,
            "tests_passed": tests_passed,
            "tests_failed": tests_failed,
            "tests_total": tests_total,
            "test_run_time": round(run_time, 4),
            "code": code,
            "test_results": test_results,
            "extractable": compute_extractable_code(response),
            "passes_tests": compute_passes_tests(test_results),
            "ncloc": metrics["ncloc"],
            "cc": metrics["cyclomatic_complexity"],
            "variability": compute_pairwise_code_variability(
                _code_samples(_response_samples(entry, sample_key, primary_key))
            ),
        }
    
    
    def _metric_side(evaluation: dict, side: str) -> dict:
        """Map an evaluated response to the internal metric-row field names."""
        prefix = "followup_" if side == "followup" else "source_"
        tests_prefix = "followup_" if side == "followup" else ""
        return {
            f"{prefix}extractable": evaluation["extractable"],
            f"{prefix}passes_tests": evaluation["passes_tests"],
            f"{prefix}passed": evaluation["passed"],
            f"{tests_prefix}tests_passed": evaluation["tests_passed"],
            f"{tests_prefix}tests_failed": evaluation["tests_failed"],
            f"{tests_prefix}tests_total": evaluation["tests_total"],
            f"{prefix}test_run_time": evaluation["test_run_time"],
            f"{prefix}ncloc": evaluation["ncloc"],
            f"{prefix}cc": evaluation["cc"],
            f"{prefix}variability": evaluation["variability"],
        }
    
    
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
                evaluation = evaluate_response(
                    primary_response,
                    entry,
                    func_name,
                    debug,
                    label,
                    "followup_response_samples" if label == "followup" else "response_samples",
                    "followup_response" if label == "followup" else "response",
                )
                if evaluation["passed"] is None:
                    skipped += 1
                    continue
                if evaluation["passed"]:
                    passed += 1
                else:
                    failed += 1
    
                metric_rows.append({
                    "id": qid,
                    "question_title": entry.get("question_title"),
                    "difficulty": entry.get("difficulty"),
                    "platform": entry.get("platform"),
                    **_metric_side(evaluation, "source"),
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
            source = evaluate_response(
                src_response,
                entry,
                func_name,
                debug,
                "source",
                "response_samples",
                "response",
            )
    
            if source["passed"] is None:
                skipped += 1
                continue
            if source["passed"]:
                passed += 1
            else:
                failed += 1
    
            row: dict = {
                "id": qid,
                "question_title": entry.get("question_title"),
                "difficulty": entry.get("difficulty"),
                "platform": entry.get("platform"),
    
                **_metric_side(source, "source"),
    
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
                followup = evaluate_response(
                    flw_response,
                    entry,
                    func_name,
                    debug,
                    "followup",
                    "followup_response_samples",
                    "followup_response",
                )
    
                if followup["passed"] is not None:
                    holds = source["passed"] == followup["passed"]
                    verdict = "HOLDS" if holds else "VIOLATED"
                    print(f"  MR [{entry['relation']}]: {verdict}")
                    if holds:
                        mr_holds_count += 1
                    else:
                        mr_violated_count += 1
                    row["mr_holds"] = holds
    
                row.update(_metric_side(followup, "followup"))
    
                row["passes_same_tests"] = compute_passes_same_tests(
                    source["test_results"], followup.get("test_results", [])
                )
                row["followup_prompt_tokens"] = flw_tokens["prompt_tokens"]
                row["followup_completion_tokens"] = flw_tokens["completion_tokens"]
                row["followup_total_tokens"] = flw_tokens["total_tokens"]
                row["followup_thinking_tokens"] = flw_tokens["thinking_tokens"]
                row["followup_token_efficiency"] = flw_token_efficiency
                row["followup_llm_call_time"] = flw_llm_call_time
    
                if source["code"] and followup.get("code"):
                    row["code_sim"] = compute_code_similarity(source["code"], followup["code"])
    
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
                    cc_str = f"{format_decimal(r['source_cc'], 1)}/{format_decimal(r['followup_cc'], 1) if r['followup_cc'] is not None else 'N/A'}"
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
                    eff = format_decimal(r.get("source_token_efficiency"), 3) if r.get("source_token_efficiency") is not None else "N/A"
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
                        f"{format_decimal(r.get('code_sim'), 4):>7} {format_decimal(src_dist, 4):>7} {format_decimal(flw_dist, 4):>7} "
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
                    eff = format_decimal(r.get("source_token_efficiency"), 3) if r.get("source_token_efficiency") is not None else "N/A"
                    llm_t = f"{r['source_llm_call_time']:.1f}s" if r.get("source_llm_call_time") is not None else "N/A"
                    test_t = f"{r['source_test_run_time']:.2f}s" if r.get("source_test_run_time") is not None else "N/A"
                    print(
                        f"  {r['id']:<15} {tests_str:>8} {ext:>7} {r['source_ncloc']:>6} {r['source_cc']:>6} "
                        f"{format_decimal(src_var.get('mean_distance'), 4):>8} {format_decimal(src_var.get('std_distance'), 4):>7} "
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
    
    
    class Tee:
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
            stream = Tee(sys.stdout, f) if args.stdout else f
            with redirect_stdout(stream), redirect_stderr(stream):
                run_tests(
                    args.results, args.n,
                    debug=args.debug,
                    followup_only=args.followup_only,
                    csv_path=args.csv,
                    write_outputs=args.write_outputs,
                )


---

## tests/test_regressions.py

```text
File: tests/test_regressions.py (11167 bytes)
```

    import json
    from pathlib import Path
    from tempfile import TemporaryDirectory
    import unittest
    from types import SimpleNamespace
    from unittest.mock import patch
    
    from cost_tracker import get_model_call_counts, reset_model_call_counts
    from evaluation.metric_violations import check_metric_violations
    from evaluation.metrics import _strip_comments_for_codesim, compute_code_similarity, compute_ttest_violation
    from evaluation.report import format_decimal, format_value
    from evaluation.runner import PipelineRunner
    from llms.claude_model import ClaudeModel
    from llms.gemini_model import GeminiModel
    from llms.mr_helper import MRHelper
    from metamorphic.relations.mrs import MR19, _clean_library_name
    
    
    class MRRegressionTests(unittest.TestCase):
        def test_empty_library_name_uses_fallback(self):
            self.assertEqual(_clean_library_name(""), "a suitable standard library module")
    
        @patch("metamorphic.relations.mrs.mr_llm_transform", return_value="")
        def test_mr19_empty_solve_response_returns_original(self, transform):
            original = "Original problem"
    
            self.assertEqual(MR19().transform(original), original)
            transform.assert_called_once()
    
        @patch(
            "metamorphic.relations.mrs.mr_llm_transform",
            side_effect=["```python\nprint('ok')\n```", ""],
        )
        def test_mr19_empty_explanation_returns_original(self, transform):
            original = "Original problem"
    
            self.assertEqual(MR19().transform(original), original)
            self.assertEqual(transform.call_count, 2)
    
    
    class MetricRegressionTests(unittest.TestCase):
        @patch("evaluation.metrics._encode_with_transformers")
        def test_codesim_removes_comments_but_preserves_hashes_in_strings(self, encode):
            encode.side_effect = RuntimeError("stop after preprocessing")
    
            compute_code_similarity(
                'value = "# retained"  # removed\n',
                "value = '# also retained'\n# removed too\n",
            )
    
            code_a, code_b = encode.call_args.args
            self.assertIn('"# retained"', code_a)
            self.assertNotIn("# removed", code_a)
            self.assertIn("'# also retained'", code_b)
            self.assertNotIn("# removed too", code_b)
            self.assertEqual(
                _strip_comments_for_codesim("value = 1\n"),
                _strip_comments_for_codesim("# heading\nvalue = 1  # explanation\n"),
            )
    
        @patch("evaluation.metric_violations.load_repeat_baseline")
        def test_static_metric_error_sentinels_are_not_compared(self, load_baseline):
            load_baseline.return_value = {
                "question": {
                    "ncloc": (10, 11, 12),
                    "cyclomatic_complexity": (2.0, 2.5, 3.0),
                }
            }
            violations = check_metric_violations(
                "question",
                {"ncloc": -1, "cyclomatic_complexity": -2},
            )
    
            self.assertNotIn("ncloc", violations)
            self.assertNotIn("cyclomatic_complexity", violations)
    
        def test_one_sided_ttests_only_flag_the_configured_direction(self):
            baseline = [9, 10, 10, 10, 11]
    
            above = compute_ttest_violation(baseline, 12, alternative="less")
            below = compute_ttest_violation(baseline, 8, alternative="less")
            self.assertTrue(above["significant"])
            self.assertFalse(below["significant"])
    
            below = compute_ttest_violation(baseline, 8, alternative="greater")
            above = compute_ttest_violation(baseline, 12, alternative="greater")
            self.assertTrue(below["significant"])
            self.assertFalse(above["significant"])
    
        @patch("evaluation.metric_violations.load_repeat_baseline")
        def test_metric_tests_use_mixed_directions(self, load_baseline):
            samples = (9, 10, 10, 10, 11)
            load_baseline.return_value = {
                "question": {
                    "ncloc": samples,
                    "cyclomatic_complexity": samples,
                    "llm_call_time": samples,
                    "total_tokens": samples,
                    "token_efficiency": samples,
                    "code_sim": samples,
                }
            }
    
            violations = check_metric_violations(
                "question",
                {
                    "ncloc": 8,
                    "cyclomatic_complexity": 8,
                    "llm_call_time": 8,
                    "total_tokens": 8,
                    "token_efficiency": 12,
                    "code_sim": 12,
                },
            )
    
            self.assertEqual(
                set(violations),
                {"ncloc", "cyclomatic_complexity", "total_tokens"},
            )
    
            violations = check_metric_violations(
                "question",
                {
                    "llm_call_time": 12,
                    "token_efficiency": 8,
                    "code_sim": 8,
                },
            )
            self.assertEqual(
                set(violations),
                {"llm_call_time", "token_efficiency", "code_sim"},
            )
    
        def test_mr_runs_append_and_reuse_shared_base_results(self):
            reset_model_call_counts()
            calls = []
    
            def generate(prompt):
                calls.append(prompt)
                return f"response-{len(calls)}"
    
            model = SimpleNamespace(
                model_name="test-model",
                generate=generate,
                last_usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                last_call_time=0.1,
                last_finish_reason="stop",
            )
            samples = [
                {
                    "question_id": qid,
                    "question_title": qid,
                    "question_content": "Solve it.",
                    "difficulty": "easy",
                    "platform": "test",
                }
                for qid in ("q1", "q2")
            ]
    
            with TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
    
                def result_path(_model_name, relation_name=None, relation_id=None):
                    if relation_name:
                        return root / f"mr-{relation_id}.json"
                    return root / "base.json"
    
                relation_1 = SimpleNamespace(name="first", target="PD", transform=lambda question: question)
                relation_2 = SimpleNamespace(name="second", target="PD", transform=lambda question: question)
    
                with patch("evaluation.runner.build_results_path", side_effect=result_path), patch("builtins.print") as output:
                    PipelineRunner([model], samples[:1], delay=0, relation=relation_1, relation_id=1).run()
                    PipelineRunner([model], samples, delay=0, relation=relation_2, relation_id=2).run()
    
                base_entries = json.loads((root / "base.json").read_text(encoding="utf-8"))
                second_mr = json.loads((root / "mr-2.json").read_text(encoding="utf-8"))
                printed = " ".join(" ".join(map(str, call.args)) for call in output.call_args_list)
    
            self.assertEqual(len(calls), 5)
            self.assertEqual(get_model_call_counts(), {"main": 5, "helper": 0})
            self.assertEqual([entry["question_id"] for entry in base_entries], ["q1", "q2"])
            self.assertEqual(base_entries[0]["response"], "response-1")
            self.assertEqual(second_mr[0]["response"], "response-1")
            self.assertEqual(second_mr[1]["response"], "response-4")
            self.assertIn("base exists", printed)
            self.assertIn("base generated", printed)
    
        def test_static_metric_error_sentinels_display_as_unavailable(self):
            for sentinel in (-1, -1.0, -2, -2.0):
                with self.subTest(sentinel=sentinel):
                    self.assertEqual(format_value(sentinel), "N/A")
                    self.assertEqual(format_decimal(sentinel), "N/A")
    
    
    class ProviderMetadataTests(unittest.TestCase):
        @patch("llms.mr_helper.update_costs")
        def test_helper_usage_is_sent_to_cost_tracker(self, update_costs):
            reset_model_call_counts()
            usage = SimpleNamespace(prompt_tokens=10, completion_tokens=4, total_tokens=14)
            response = SimpleNamespace(
                choices=[SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content="answer"),
                )],
                usage=usage,
            )
            helper = MRHelper.__new__(MRHelper)
            helper.client = SimpleNamespace(
                chat=SimpleNamespace(
                    completions=SimpleNamespace(create=lambda **kwargs: response),
                ),
            )
            helper._model_name = "helper-model"
            helper._last_usage = None
    
            with patch("llms.mr_helper._log_helper_call"):
                self.assertEqual(helper.transform("system", "user"), "answer")
    
            update_costs.assert_called_once_with(
                "helper-model",
                {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
            )
            self.assertEqual(get_model_call_counts(), {"main": 0, "helper": 1})
    
        @patch("llms.base.update_costs")
        def test_claude_records_timing_and_stop_reason(self, _update_costs):
            message = SimpleNamespace(
                content=[SimpleNamespace(text="answer")],
                stop_reason="end_turn",
                usage=SimpleNamespace(input_tokens=10, output_tokens=4),
            )
            model = ClaudeModel.__new__(ClaudeModel)
            model._model_name = "claude-test"
            model.client = SimpleNamespace(
                messages=SimpleNamespace(create=lambda **kwargs: message),
            )
    
            self.assertEqual(model.generate("prompt"), "answer")
            self.assertIsInstance(model.last_call_time, float)
            self.assertEqual(model.last_finish_reason, "end_turn")
    
        @patch("llms.base.update_costs")
        def test_gemini_records_timing_and_finish_reason(self, _update_costs):
            finish_reason = SimpleNamespace(name="MAX_TOKENS")
            response = SimpleNamespace(
                text="answer",
                candidates=[SimpleNamespace(finish_reason=finish_reason)],
                usage_metadata=SimpleNamespace(
                    prompt_token_count=10,
                    candidates_token_count=4,
                    total_token_count=14,
                ),
            )
            model = GeminiModel.__new__(GeminiModel)
            model._model_name = "gemini-test"
            model.client = SimpleNamespace(
                models=SimpleNamespace(generate_content=lambda **kwargs: response),
            )
    
            self.assertEqual(model.generate("prompt"), "answer")
            self.assertIsInstance(model.last_call_time, float)
            self.assertEqual(model.last_finish_reason, "MAX_TOKENS")
    
        def test_runner_collects_finish_reason(self):
            model = SimpleNamespace(
                last_usage={"prompt_tokens": 1, "completion_tokens": 1},
                last_call_time=0.25,
                last_finish_reason="stop",
                generate=lambda prompt: "answer",
            )
            runner = PipelineRunner([], [], delay=0)
            totals = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "thinking_tokens": 0,
                "reported": False,
            }
    
            generated = runner._generate_responses(model, "prompt", totals)
    
            self.assertEqual(generated.primary_finish_reason, "stop")
    
    
    if __name__ == "__main__":
        unittest.main()

