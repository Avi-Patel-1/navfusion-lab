from __future__ import annotations

from pathlib import Path
from typing import Any

from .reports.csv import write_csv
from .reports.json import write_json
from .reports.svg import write_series_svg as svg_series


__all__ = ["svg_series", "write_csv", "write_json"]
