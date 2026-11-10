"""Load Synthea CSVs into a DuckDB database."""
import _common
from statin_agent.config import load_config
from statin_agent.db import build_database

p = _common.base_parser(__doc__)
p.add_argument("--csv-dir", help="override synthea_csv_dir from the config")
a = p.parse_args()
cfg = load_config(a.config)
counts = build_database(a.csv_dir or cfg["synthea_csv_dir"], cfg["database"])
for k, v in counts.items():
    print(f"{k:>14}: {v}")
print(f"database -> {cfg['database']}")
