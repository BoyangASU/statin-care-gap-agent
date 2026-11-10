"""Load the project configuration (configs/default.json by default)."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | None = None) -> dict:
    p = Path(path) if path else ROOT / "configs" / "default.json"
    cfg = json.loads(p.read_text())
    for key in ("synthea_csv_dir", "database"):
        if not Path(cfg[key]).is_absolute():
            cfg[key] = str(ROOT / cfg[key])
    return cfg


def index_date_for(cfg, con):
    """Index date from the config, or the last observation date in the data."""
    from datetime import date
    from .db import data_end_date
    v = cfg.get("index_date")
    return date.fromisoformat(v) if v else data_end_date(con)
