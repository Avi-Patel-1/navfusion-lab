from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any


def _sanitize_identifier(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_]", "_", value.strip())
    if not cleaned:
        cleaned = "value"
    if cleaned[0].isdigit():
        cleaned = f"v_{cleaned}"
    return cleaned.lower()


def _read_numeric_csv(path: Path, max_rows: int | None = None) -> tuple[list[str], list[list[float]]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if max_rows is not None:
        rows = rows[: int(max_rows)]
    numeric_columns: list[str] = []
    for name in fieldnames:
        values = [row.get(name, "") for row in rows]
        convertible = True
        saw_value = False
        for value in values:
            if value in {"", None}:
                continue
            try:
                float(value)
                saw_value = True
            except (TypeError, ValueError):
                convertible = False
                break
        if convertible and saw_value:
            numeric_columns.append(name)
    matrix: list[list[float]] = []
    for row in rows:
        matrix.append([0.0 if row.get(name, "") in {"", None} else float(row[name]) for name in numeric_columns])
    return numeric_columns, matrix


def _format_matrix(name: str, columns: list[str], matrix: list[list[float]]) -> str:
    rows = len(matrix)
    cols = len(columns)
    lines = [
        f"// {name} columns: {', '.join(columns)}",
        f"static const unsigned int {name}_rows = {rows}u;",
        f"static const unsigned int {name}_cols = {cols}u;",
        f"static const double {name}[{max(rows, 1)}][{max(cols, 1)}] = {{",
    ]
    if rows == 0 or cols == 0:
        lines.append("  {0.0}")
    else:
        for row in matrix:
            payload = ", ".join(f"{value:.12g}" for value in row)
            lines.append(f"  {{{payload}}},")
    lines.append("};")
    return "\n".join(lines)


def export_run_c_header(run_dir: str | Path, out_path: str | Path, include_patterns: list[str] | None = None, max_rows: int | None = None) -> dict[str, Any]:
    """Export numeric run CSV files as C-compatible static double arrays."""
    run = Path(run_dir)
    out = Path(out_path)
    patterns = include_patterns or ["truth.csv", "measurements.csv", "estimates_*.csv", "innovations_*.csv"]
    csv_paths: list[Path] = []
    for pattern in patterns:
        csv_paths.extend(sorted(run.glob(pattern)))
    unique_paths = []
    seen = set()
    for path in csv_paths:
        if path.is_file() and path not in seen:
            seen.add(path)
            unique_paths.append(path)
    sections = [
        "/* Reference vectors exported from fusion_sandbox. */",
        "#ifndef FUSION_SANDBOX_REFERENCE_VECTORS_H",
        "#define FUSION_SANDBOX_REFERENCE_VECTORS_H",
        "",
    ]
    manifest = []
    for path in unique_paths:
        columns, matrix = _read_numeric_csv(path, max_rows=max_rows)
        name = _sanitize_identifier(path.stem)
        sections.append(_format_matrix(name, columns, matrix))
        sections.append("")
        manifest.append({"file": str(path.relative_to(run)), "array": name, "rows": len(matrix), "cols": len(columns)})
    sections.append("#endif")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(sections) + "\n")
    return {"run": str(run), "out": str(out), "arrays": manifest, "array_count": len(manifest)}
