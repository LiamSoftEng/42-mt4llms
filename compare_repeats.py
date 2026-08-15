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
