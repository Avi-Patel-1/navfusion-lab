from __future__ import annotations

import html
from pathlib import Path
from typing import Any


def _metric_table(rows: list[dict[str, Any]]) -> str:
    has_experiment = any("experiment" in row for row in rows)
    first_cols = "<th>Experiment</th>" if has_experiment else ""
    header = f"<tr>{first_cols}<th>Filter</th><th>Profile</th><th>Position RMSE (m)</th><th>Velocity RMSE (m/s)</th><th>Rejected</th><th>Dropout max error (m)</th><th>Diverged</th></tr>"
    body = []
    for row in rows:
        experiment_cell = f"<td>{html.escape(str(row.get('experiment', '')))}</td>" if has_experiment else ""
        body.append(
            "<tr>"
            f"{experiment_cell}"
            f"<td>{html.escape(str(row.get('filter', '')))}</td>"
            f"<td>{html.escape(str(row.get('profile', '')))}</td>"
            f"<td>{float(row.get('position_rmse_m', 0.0)):.3f}</td>"
            f"<td>{float(row.get('velocity_rmse_mps', 0.0)):.3f}</td>"
            f"<td>{int(row.get('rejected_updates', 0))}</td>"
            f"<td>{float(row.get('max_error_during_dropout_m', 0.0)):.3f}</td>"
            f"<td>{html.escape(str(row.get('diverged', False)).lower())}</td>"
            "</tr>"
        )
    return f"<table>{header}{''.join(body)}</table>"


def _metadata(summary: dict[str, Any]) -> str:
    items = [
        ("Best result", summary.get("best_result", summary.get("best_profile", ""))),
        ("Samples", summary.get("samples", "")),
        ("Measurements", summary.get("measurement_count", "")),
        ("Results", len(summary.get("results", []))),
    ]
    cards = []
    for label, value in items:
        if value == "":
            continue
        cards.append(f'<div class="card"><span>{html.escape(label)}</span><strong>{html.escape(str(value))}</strong></div>')
    return f'<div class="cards">{"".join(cards)}</div>' if cards else ""


def write_report(run_dir: Path, summary: dict[str, Any]) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    title = html.escape(str(summary.get("name", "Sensor Fusion Run")))
    rows = summary.get("results", summary.get("profiles", []))
    plot_links = []
    plots_dir = run_dir / "plots"
    if plots_dir.exists():
        for plot in sorted(plots_dir.glob("*.svg")):
            rel = plot.relative_to(run_dir)
            plot_links.append(f'<figure><img src="{html.escape(str(rel))}" alt="{html.escape(plot.stem)}"><figcaption>{html.escape(plot.stem)}</figcaption></figure>')
    content = f'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 28px; color: #172026; background: #f8fafb; }}
    h1 {{ margin-bottom: 4px; }}
    .meta {{ color: #54616c; margin-bottom: 16px; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 10px; max-width: 960px; margin: 12px 0 20px; }}
    .card {{ background: white; border: 1px solid #d9e2e7; padding: 10px 12px; }}
    .card span {{ display: block; color: #60717d; font-size: 12px; margin-bottom: 4px; }}
    .card strong {{ font-size: 17px; }}
    .table-wrap {{ overflow-x: auto; max-width: 100%; }}
    table {{ border-collapse: collapse; margin: 16px 0 28px; min-width: 720px; background: white; }}
    th, td {{ border: 1px solid #cfd8dc; padding: 8px 10px; text-align: right; }}
    th:first-child, td:first-child, th:nth-child(2), td:nth-child(2), th:nth-child(3), td:nth-child(3) {{ text-align: left; }}
    th {{ background: #eef3f6; }}
    figure {{ margin: 20px 0; }}
    img {{ max-width: 100%; border: 1px solid #d9e2e7; }}
    figcaption {{ color: #54616c; font-size: 12px; margin-top: 4px; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <div class="meta">Navigation filter validation report</div>
  {_metadata(summary)}
  <div class="table-wrap">{_metric_table(rows)}</div>
  {''.join(plot_links)}
</body>
</html>
'''
    path = run_dir / "report.html"
    path.write_text(content)
    return path
