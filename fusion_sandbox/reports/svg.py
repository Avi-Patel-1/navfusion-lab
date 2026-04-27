from __future__ import annotations

import html
from pathlib import Path
from typing import Iterable

import numpy as np


COLORS = ["#0f766e", "#2563eb", "#b45309", "#be123c", "#4f46e5", "#15803d", "#7c3aed", "#64748b"]


def _scale(values: np.ndarray, lo: float, hi: float, pixels_lo: float, pixels_hi: float) -> np.ndarray:
    span = max(hi - lo, 1e-12)
    return pixels_lo + (values - lo) / span * (pixels_hi - pixels_lo)


def _bounds(series: Iterable[np.ndarray]) -> tuple[float, float]:
    arrays = [np.asarray(s, dtype=float).reshape(-1) for s in series if np.asarray(s).size]
    if not arrays:
        return 0.0, 1.0
    values = np.concatenate(arrays)
    if values.size == 0:
        return 0.0, 1.0
    lo = float(np.nanmin(values))
    hi = float(np.nanmax(values))
    if abs(hi - lo) < 1e-12:
        lo -= 1.0
        hi += 1.0
    return lo, hi


def _svg_frame(width: int, height: int, title: str, y_label: str) -> str:
    return f'''<rect width="100%" height="100%" fill="white"/>
<text x="52" y="30" font-family="Arial" font-size="18" font-weight="700">{html.escape(title)}</text>
<line x1="52" y1="{height-46}" x2="{width-24}" y2="{height-46}" stroke="#263238"/>
<line x1="52" y1="42" x2="52" y2="{height-46}" stroke="#263238"/>
<text x="52" y="{height-14}" font-family="Arial" font-size="12">time (s)</text>
<text x="{width-190}" y="{height-14}" font-family="Arial" font-size="12">{html.escape(y_label)}</text>
'''


def write_series_svg(path: Path, xs: list[float], ys: list[float], title: str, y_label: str, annotations: list[list[float]] | None = None) -> None:
    write_multi_series_svg(path, xs, [(title, ys)], title, y_label, annotations)


def write_multi_series_svg(
    path: Path,
    xs: list[float],
    series: list[tuple[str, list[float]]],
    title: str,
    y_label: str,
    annotations: list[list[float]] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 820, 320
    pad_l, pad_r, pad_t, pad_b = 52, 24, 42, 46
    x = np.asarray(xs, dtype=float)
    if x.size == 0:
        x = np.array([0.0, 1.0])
    xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))
    ymin, ymax = _bounds([np.asarray(vals, dtype=float) for _, vals in series])
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        _svg_frame(width, height, title, y_label),
    ]
    for window in annotations or []:
        x0, x1 = float(window[0]), float(window[1])
        px0 = float(_scale(np.array([x0]), xmin, xmax, pad_l, width - pad_r)[0])
        px1 = float(_scale(np.array([x1]), xmin, xmax, pad_l, width - pad_r)[0])
        pieces.append(f'<rect x="{px0:.1f}" y="{pad_t}" width="{max(px1-px0, 1.0):.1f}" height="{height-pad_t-pad_b}" fill="#fef3c7" opacity="0.72"/>')
    for idx, (label, vals) in enumerate(series):
        y = np.asarray(vals, dtype=float)
        usable_x = x[: y.size]
        px = _scale(usable_x, xmin, xmax, pad_l, width - pad_r)
        py = _scale(y, ymin, ymax, height - pad_b, pad_t)
        points = " ".join(f"{a:.1f},{b:.1f}" for a, b in zip(px, py) if np.isfinite(a) and np.isfinite(b))
        color = COLORS[idx % len(COLORS)]
        pieces.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.3" points="{points}"/>')
        pieces.append(f'<text x="{width-180}" y="{56 + idx*16}" font-family="Arial" font-size="12" fill="{color}">{html.escape(label)}</text>')
    pieces.append("</svg>")
    path.write_text("\n".join(pieces))


def write_xy_svg(path: Path, xs: list[float], ys: list[float], title: str, x_label: str = "x (m)", y_label: str = "y (m)") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 560, 460
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    xmin, xmax = _bounds([x])
    ymin, ymax = _bounds([y])
    px = _scale(x, xmin, xmax, 52, width - 24)
    py = _scale(y, ymin, ymax, height - 48, 42)
    points = " ".join(f"{a:.1f},{b:.1f}" for a, b in zip(px, py))
    path.write_text(
        f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="52" y="30" font-family="Arial" font-size="18" font-weight="700">{html.escape(title)}</text>
<line x1="52" y1="{height-48}" x2="{width-24}" y2="{height-48}" stroke="#263238"/>
<line x1="52" y1="42" x2="52" y2="{height-48}" stroke="#263238"/>
<polyline fill="none" stroke="#2563eb" stroke-width="2.2" points="{points}"/>
<text x="52" y="{height-16}" font-family="Arial" font-size="12">{html.escape(x_label)}</text>
<text x="{width-120}" y="{height-16}" font-family="Arial" font-size="12">{html.escape(y_label)}</text>
</svg>'''
    )


def write_bar_svg(path: Path, labels: list[str], values: list[float], title: str, y_label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 760, 340
    values_arr = np.asarray(values, dtype=float)
    ymax = max(float(np.nanmax(values_arr)) if values_arr.size else 1.0, 1e-9)
    bar_w = max((width - 110) / max(len(labels), 1), 1.0)
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="100%" height="100%" fill="white"/><text x="52" y="30" font-family="Arial" font-size="18" font-weight="700">{html.escape(title)}</text>',
        f'<line x1="52" y1="{height-60}" x2="{width-24}" y2="{height-60}" stroke="#263238"/><line x1="52" y1="42" x2="52" y2="{height-60}" stroke="#263238"/>',
        f'<text x="{width-175}" y="{height-16}" font-family="Arial" font-size="12">{html.escape(y_label)}</text>',
    ]
    for i, (label, value) in enumerate(zip(labels, values)):
        h = float(value) / ymax * (height - 116)
        x = 62 + i * bar_w
        y = height - 60 - h
        pieces.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w*0.66:.1f}" height="{h:.1f}" fill="{COLORS[i % len(COLORS)]}"/>')
        pieces.append(f'<text x="{x:.1f}" y="{height-40}" font-family="Arial" font-size="11" transform="rotate(25 {x:.1f},{height-40})">{html.escape(label)}</text>')
    pieces.append("</svg>")
    path.write_text("\n".join(pieces))
