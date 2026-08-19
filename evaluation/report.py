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

        row_total_time = format_duration(_row_total_time(b, f, has_mr))

        # Highlight cells for metrics that breached their configured threshold
        # (config.settings.METRIC_VARIANCE_THRESHOLDS).
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
