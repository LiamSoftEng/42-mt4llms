"""
Generates an HTML comparison report from base and followup results files.
"""

from pathlib import Path
import json


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


def _mr_badge(holds: bool | None) -> str:
    if holds is True:
        return '<span class="badge holds">HOLDS</span>'
    if holds is False:
        return '<span class="badge violated">VIOLATED</span>'
    return '<span class="badge na">N/A</span>'


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
    with open(path, encoding="utf-8") as f:
        return json.load(f)


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


def generate_html_report(
    base_results_path: Path,
    followup_results_path: Path | None,
    out_path: Path,
    model_name: str,
    relation_name: str | None,
) -> None:
    base = {r["question_id"]: r for r in _load(base_results_path)}
    followup = {r["question_id"]: r for r in _load(followup_results_path)} if followup_results_path else {}
    has_mr = bool(followup) and bool(relation_name)

    all_ids = list(base.keys()) or list(followup.keys())

    # ── Aggregate stats ──────────────────────────────────────────────────────
    base_total = len(base)
    base_passed = sum(1 for r in base.values() if r.get("passed"))
    base_failed = base_total - base_passed
    base_extractable = sum(1 for r in base.values() if r.get("extractable_code"))

    flw_total = len(followup)
    flw_passed = sum(1 for r in followup.values() if r.get("passed"))
    flw_failed = flw_total - flw_passed
    flw_extractable = sum(1 for r in followup.values() if r.get("extractable_code"))

    mr_holds = sum(1 for qid in all_ids if followup.get(qid, {}).get("mr_holds") is True)
    mr_violated = sum(1 for qid in all_ids if followup.get(qid, {}).get("mr_holds") is False)

    same_tests_count = sum(1 for qid in all_ids if followup.get(qid, {}).get("passes_same_tests") is True)
    diff_tests_count = sum(1 for qid in all_ids if followup.get(qid, {}).get("passes_same_tests") is False)

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
                <td>{_mr_badge(mr_holds_val)}<br>{_bool_badge(passes_same, "SAME", "DIFF")}</td>
                <td>{_fmt(b_ncloc, 0)}</td>
                <td>{_fmt(f_ncloc, 0) if f_ncloc is not None else "N/A"}</td>
                <td>{_delta(b_ncloc, f_ncloc)}</td>
                <td>{_fmt(b_cc)}</td>
                <td>{_fmt(f_cc) if f_cc is not None else "N/A"}</td>
                <td>{_delta(b_cc, f_cc)}</td>
                <td>{_fmt(code_sim)}</td>
                <td>{_fmt(b_mean_dist)}<br><small>±{_fmt(b_std_dist)}</small></td>
                <td>{_fmt(f_mean_dist)}<br><small>±{_fmt(f_std_dist)}</small></td>
                <td>{b_tok}<br><small>eff:{_fmt(b_eff, 3)}</small></td>
                <td>{f_tok}<br><small>eff:{_fmt(f_eff, 3)}</small></td>
                <td>{b_llm_t}<br><small>test:{b_test_t}</small></td>
                <td>{f_llm_t}<br><small>test:{f_test_t}</small></td>
                <td><span class="badge na">N/A</span></td>
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
        </div>
        <div class="stat-box">
            <div class="stat-label">Passes same tests</div>
            <div class="stat-value">{same_pct}%</div>
            <div class="stat-sub">{same_tests_count} same / {diff_tests_count} different</div>
        </div>"""

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
            <th>Constraint adherence</th>"""
    else:
        table_headers = """
            <th>Problem</th>
            <th>Tests</th>
            <th>NCLOC</th>
            <th>CC</th>
            <th>Mean dist ±std</th>
            <th>Tokens / eff</th>
            <th>LLM / test time</th>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>MT Report — {model_name}</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 2rem; background: #f8f9fa; color: #212529; font-size: 0.9rem; }}
  h1 {{ font-size: 1.4rem; margin-bottom: 0.25rem; }}
  .subtitle {{ color: #6c757d; font-size: 0.85rem; margin-bottom: 1.5rem; }}

  .stats {{ display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 2rem; }}
  .stat-box {{ background: white; border: 1px solid #dee2e6; border-radius: 8px; padding: 1rem 1.5rem; min-width: 170px; }}
  .stat-label {{ font-size: 0.72rem; color: #6c757d; text-transform: uppercase; letter-spacing: 0.05em; }}
  .stat-value {{ font-size: 1.6rem; font-weight: 700; margin: 0.25rem 0; }}
  .stat-sub {{ font-size: 0.78rem; color: #6c757d; }}

  .table-wrap {{ overflow-x: auto; }}
  table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); white-space: nowrap; }}
  th {{ background: #343a40; color: white; padding: 0.5rem 0.7rem; text-align: left; font-size: 0.75rem; }}
  td {{ padding: 0.5rem 0.7rem; border-bottom: 1px solid #dee2e6; font-size: 0.8rem; vertical-align: top; white-space: nowrap; }}
  td:first-child {{ white-space: normal; min-width: 180px; }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover td {{ background: #f1f3f5; }}

  code {{ font-size: 0.75rem; color: #e83e8c; }}
  small {{ color: #6c757d; font-size: 0.72rem; }}
  .tag {{ display: inline-block; background: #e9ecef; border-radius: 4px; padding: 0.1rem 0.4rem; font-size: 0.7rem; color: #495057; margin-right: 2px; }}

  .badge {{ display: inline-block; border-radius: 4px; padding: 0.12rem 0.45rem; font-size: 0.72rem; font-weight: 600; }}
  .holds {{ background: #d1e7dd; color: #0f5132; }}
  .violated {{ background: #f8d7da; color: #842029; }}
  .na {{ background: #e9ecef; color: #6c757d; }}

  .delta-pos {{ color: #198754; }}
  .delta-neg {{ color: #dc3545; }}
</style>
</head>
<body>
<h1>MT Report — {model_name}</h1>
<div class="subtitle">{f"MR: {relation_name}" if relation_name else "Base run — no MR applied"}</div>

<div class="stats">
  <div class="stat-box">
    <div class="stat-label">Base pass rate</div>
    <div class="stat-value">{base_pct}%</div>
    <div class="stat-sub">{base_passed} passed / {base_failed} failed / {base_total} total</div>
    <div class="stat-sub">Extractable: {extract_pct}% ({base_extractable}/{base_total})</div>
  </div>
  {followup_stat}
  {mr_section}
</div>

<div class="table-wrap">
<table>
  <thead><tr>{table_headers}</tr></thead>
  <tbody>{rows_html}</tbody>
</table>
</div>
</body>
</html>"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  HTML report -> {out_path}")