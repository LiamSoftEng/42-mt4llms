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


def _load(path: Path) -> list[dict]:
    if not path or not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


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

    # Aggregate stats
    base_total = len(base)
    base_passed = sum(1 for r in base.values() if r.get("passed"))
    base_failed = base_total - base_passed

    flw_total = len(followup)
    flw_passed = sum(1 for r in followup.values() if r.get("passed"))
    flw_failed = flw_total - flw_passed

    mr_holds = sum(1 for qid in all_ids if followup.get(qid, {}).get("mr_holds") is True)
    mr_violated = sum(1 for qid in all_ids if followup.get(qid, {}).get("mr_holds") is False)

    # Build rows
    rows_html = ""
    for qid in all_ids:
        b = base.get(qid, {})
        f = followup.get(qid, {})

        b_passed = b.get("passed")
        f_passed = f.get("passed") if has_mr else None
        mr_holds_val = f.get("mr_holds") if has_mr else None

        b_ncloc = b.get("ncloc", -1)
        b_cc = b.get("cyclomatic_complexity", -1)
        f_ncloc = f.get("ncloc", -1) if has_mr else None
        f_cc = f.get("cyclomatic_complexity", -1) if has_mr else None

        b_dist = b.get("mean_distance")
        f_dist = f.get("mean_distance") if has_mr else None
        code_sim = f.get("code_sim") if has_mr else None

        title = b.get("question_title") or f.get("question_title") or qid
        difficulty = b.get("difficulty") or f.get("difficulty") or ""
        platform = b.get("platform") or f.get("platform") or ""

        b_tests = f'{b.get("tests_passed", "?")}/{b.get("tests_total", "?")}'
        f_tests = f'{f.get("tests_passed", "?")}/{f.get("tests_total", "?")}' if has_mr else "N/A"

        if has_mr:
            rows_html += f"""
            <tr>
                <td><code>{qid}</code><br><small>{title}</small><br><span class="tag">{difficulty} / {platform}</span></td>
                <td>{b_tests} {_pass_badge(b_passed)}</td>
                <td>{f_tests} {_pass_badge(f_passed)}</td>
                <td>{_mr_badge(mr_holds_val)}</td>
                <td>{_fmt(b_ncloc, 0)}</td>
                <td>{_fmt(f_ncloc, 0) if f_ncloc is not None else 'N/A'}</td>
                <td>{_delta(b_ncloc, f_ncloc)}</td>
                <td>{_fmt(b_cc)}</td>
                <td>{_fmt(f_cc) if f_cc is not None else 'N/A'}</td>
                <td>{_delta(b_cc, f_cc)}</td>
                <td>{_fmt(code_sim)}</td>
                <td>{_fmt(b_dist)}</td>
                <td>{_fmt(f_dist)}</td>
            </tr>"""
        else:
            rows_html += f"""
            <tr>
                <td><code>{qid}</code><br><small>{title}</small><br><span class="tag">{difficulty} / {platform}</span></td>
                <td>{b_tests} {_pass_badge(b_passed)}</td>
                <td>{_fmt(b_ncloc, 0)}</td>
                <td>{_fmt(b_cc)}</td>
                <td>{_fmt(b_dist)}</td>
            </tr>"""

    # Header section
    mr_section = ""
    if has_mr:
        mr_total = mr_holds + mr_violated
        hold_pct = round(mr_holds / mr_total * 100) if mr_total else 0
        mr_section = f"""
        <div class="stat-box">
            <div class="stat-label">MR [{relation_name}]</div>
            <div class="stat-value">{hold_pct}% hold rate</div>
            <div class="stat-sub">{mr_holds} holds / {mr_violated} violations</div>
        </div>"""

    followup_stat = ""
    if has_mr:
        flw_pct = round(flw_passed / flw_total * 100) if flw_total else 0
        followup_stat = f"""
        <div class="stat-box">
            <div class="stat-label">Followup Pass Rate</div>
            <div class="stat-value">{flw_pct}%</div>
            <div class="stat-sub">{flw_passed} passed / {flw_failed} failed / {flw_total} total</div>
        </div>"""

    base_pct = round(base_passed / base_total * 100) if base_total else 0

    # Table headers
    if has_mr:
        table_headers = """
            <th>Problem</th>
            <th>Base Tests</th>
            <th>Followup Tests</th>
            <th>MR</th>
            <th>Base NCLOC</th>
            <th>Flw NCLOC</th>
            <th>ΔNCLOC</th>
            <th>Base CC</th>
            <th>Flw CC</th>
            <th>ΔCC</th>
            <th>CodeSim</th>
            <th>Base Dist</th>
            <th>Flw Dist</th>"""
    else:
        table_headers = """
            <th>Problem</th>
            <th>Tests</th>
            <th>NCLOC</th>
            <th>CC</th>
            <th>Mean Dist</th>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>MT Report — {model_name}</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 2rem; background: #f8f9fa; color: #212529; }}
  h1 {{ font-size: 1.4rem; margin-bottom: 0.25rem; }}
  .subtitle {{ color: #6c757d; font-size: 0.9rem; margin-bottom: 1.5rem; }}
  .stats {{ display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 2rem; }}
  .stat-box {{ background: white; border: 1px solid #dee2e6; border-radius: 8px; padding: 1rem 1.5rem; min-width: 160px; }}
  .stat-label {{ font-size: 0.75rem; color: #6c757d; text-transform: uppercase; letter-spacing: 0.05em; }}
  .stat-value {{ font-size: 1.6rem; font-weight: 700; margin: 0.25rem 0; }}
  .stat-sub {{ font-size: 0.8rem; color: #6c757d; }}
  table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  th {{ background: #343a40; color: white; padding: 0.6rem 0.8rem; text-align: left; font-size: 0.8rem; white-space: nowrap; }}
  td {{ padding: 0.6rem 0.8rem; border-bottom: 1px solid #dee2e6; font-size: 0.85rem; vertical-align: top; }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover td {{ background: #f1f3f5; }}
  code {{ font-size: 0.8rem; color: #e83e8c; }}
  small {{ color: #6c757d; }}
  .tag {{ display: inline-block; background: #e9ecef; border-radius: 4px; padding: 0.1rem 0.4rem; font-size: 0.7rem; color: #495057; }}
  .badge {{ display: inline-block; border-radius: 4px; padding: 0.15rem 0.5rem; font-size: 0.75rem; font-weight: 600; }}
  .holds {{ background: #d1e7dd; color: #0f5132; }}
  .violated {{ background: #f8d7da; color: #842029; }}
  .na {{ background: #e9ecef; color: #6c757d; }}
</style>
</head>
<body>
<h1>MT Report — {model_name}</h1>
<div class="subtitle">{f"MR: {relation_name}" if relation_name else "Base run — no MR applied"}</div>

<div class="stats">
  <div class="stat-box">
    <div class="stat-label">Base Pass Rate</div>
    <div class="stat-value">{base_pct}%</div>
    <div class="stat-sub">{base_passed} passed / {base_failed} failed / {base_total} total</div>
  </div>
  {followup_stat}
  {mr_section}
</div>

<table>
  <thead><tr>{table_headers}</tr></thead>
  <tbody>{rows_html}</tbody>
</table>
</body>
</html>"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  HTML report -> {out_path}")