"""DuckDB data layer over MIMIC-IV `hosp` CSV exports (Demo 2.2 or full 2.2+).

`build_database(hosp_dir, db_path)` loads the hosp tables the statin task needs
into one DuckDB file, with explicit types for the columns that type sniffing
gets wrong (ICD codes with leading zeros, NDCs, free-text lab values), and adds
one convenience view, `admissions_x`, with age at admission and length of stay.

Column names were checked against the MIMIC-IV Demo 2.2 files themselves.
The full MIMIC-IV hosp module has the same layout, so the same code loads it.

Notes on MIMIC-IV conventions that matter downstream:
- ICD codes are stored without the dot ("41401", "I214") and with an
  `icd_version` column (9 or 10). We trim whitespace on load.
- Dates are shifted into the future (21xx) per patient; intervals within a
  patient are real. Age at admission = anchor_age + (year(admittime) - anchor_year).
- `labevents.hadm_id` is NULL for outpatient labs, so labs should be joined by
  subject_id and charttime window, not by hadm_id.

Usage (from the repository root):
    python -m mimic.db --hosp-dir /scratch/$USER/mimic-iv-demo/hosp --db data/mimic_demo.duckdb
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import duckdb

# Column type overrides per table. Columns not listed are auto-detected.
# Timestamps are listed explicitly so a sniffing mistake cannot turn them into text.
TS = "TIMESTAMP"
TABLES: dict[str, dict[str, str]] = {
    "patients": {"subject_id": "BIGINT", "gender": "VARCHAR", "anchor_age": "INTEGER",
                 "anchor_year": "INTEGER", "anchor_year_group": "VARCHAR", "dod": "DATE"},
    "admissions": {"subject_id": "BIGINT", "hadm_id": "BIGINT", "admittime": TS,
                   "dischtime": TS, "deathtime": TS, "edregtime": TS, "edouttime": TS,
                   "admission_type": "VARCHAR", "admission_location": "VARCHAR",
                   "discharge_location": "VARCHAR", "race": "VARCHAR",
                   "hospital_expire_flag": "INTEGER"},
    "diagnoses_icd": {"subject_id": "BIGINT", "hadm_id": "BIGINT", "seq_num": "INTEGER",
                      "icd_code": "VARCHAR", "icd_version": "INTEGER"},
    "d_icd_diagnoses": {"icd_code": "VARCHAR", "icd_version": "INTEGER", "long_title": "VARCHAR"},
    "procedures_icd": {"subject_id": "BIGINT", "hadm_id": "BIGINT", "seq_num": "INTEGER",
                       "chartdate": "DATE", "icd_code": "VARCHAR", "icd_version": "INTEGER"},
    "d_icd_procedures": {"icd_code": "VARCHAR", "icd_version": "INTEGER", "long_title": "VARCHAR"},
    "prescriptions": {"subject_id": "BIGINT", "hadm_id": "BIGINT", "pharmacy_id": "BIGINT",
                      "poe_id": "VARCHAR", "starttime": TS, "stoptime": TS,
                      "drug_type": "VARCHAR", "drug": "VARCHAR", "formulary_drug_cd": "VARCHAR",
                      "gsn": "VARCHAR", "ndc": "VARCHAR", "prod_strength": "VARCHAR",
                      "dose_val_rx": "VARCHAR", "dose_unit_rx": "VARCHAR",
                      "doses_per_24_hrs": "DOUBLE", "route": "VARCHAR"},
    "pharmacy": {"subject_id": "BIGINT", "hadm_id": "BIGINT", "pharmacy_id": "BIGINT",
                 "poe_id": "VARCHAR", "starttime": TS, "stoptime": TS, "entertime": TS,
                 "verifiedtime": TS, "expirationdate": TS, "medication": "VARCHAR",
                 "status": "VARCHAR", "route": "VARCHAR", "frequency": "VARCHAR"},
    "emar": {"subject_id": "BIGINT", "hadm_id": "BIGINT", "emar_id": "VARCHAR",
             "emar_seq": "BIGINT", "poe_id": "VARCHAR", "pharmacy_id": "BIGINT",
             "charttime": TS, "scheduletime": TS, "storetime": TS,
             "medication": "VARCHAR", "event_txt": "VARCHAR"},
    "labevents": {"labevent_id": "BIGINT", "subject_id": "BIGINT", "hadm_id": "BIGINT",
                  "specimen_id": "BIGINT", "itemid": "INTEGER", "charttime": TS,
                  "storetime": TS, "value": "VARCHAR", "valuenum": "DOUBLE",
                  "valueuom": "VARCHAR", "ref_range_lower": "DOUBLE",
                  "ref_range_upper": "DOUBLE", "flag": "VARCHAR", "priority": "VARCHAR",
                  "comments": "VARCHAR"},
    "d_labitems": {"itemid": "INTEGER", "label": "VARCHAR", "fluid": "VARCHAR",
                   "category": "VARCHAR"},
    "omr": {"subject_id": "BIGINT", "chartdate": "DATE", "seq_num": "INTEGER",
            "result_name": "VARCHAR", "result_value": "VARCHAR"},
    "transfers": {"subject_id": "BIGINT", "hadm_id": "BIGINT", "transfer_id": "BIGINT",
                  "eventtype": "VARCHAR", "careunit": "VARCHAR", "intime": TS, "outtime": TS},
    "services": {"subject_id": "BIGINT", "hadm_id": "BIGINT", "transfertime": TS,
                 "prev_service": "VARCHAR", "curr_service": "VARCHAR"},
    "drgcodes": {"subject_id": "BIGINT", "hadm_id": "BIGINT", "drg_type": "VARCHAR",
                 "drg_code": "VARCHAR", "description": "VARCHAR"},
}
# Large tables that are not needed for the statin task; load only on request.
OPTIONAL_TABLES = ("emar_detail", "poe", "poe_detail", "microbiologyevents",
                   "hcpcsevents", "d_hcpcs", "provider")

INDEXES = {
    "admissions": ["hadm_id", "subject_id"],
    "diagnoses_icd": ["hadm_id"],
    "procedures_icd": ["hadm_id"],
    "prescriptions": ["hadm_id"],
    "emar": ["hadm_id"],
    "labevents": ["subject_id"],
    "omr": ["subject_id"],
}


def _csv_path(hosp_dir: Path, name: str) -> Path:
    for ext in (".csv.gz", ".csv"):
        p = hosp_dir / f"{name}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"{name}.csv.gz not found in {hosp_dir}")


def _resolve_hosp_dir(path: str | Path) -> Path:
    """Accept either the hosp folder itself or its parent (e.g. .../mimic-iv-demo)."""
    p = Path(path).expanduser()
    if (p / "hosp").is_dir():
        p = p / "hosp"
    if not (p / "admissions.csv.gz").exists() and not (p / "admissions.csv").exists():
        raise FileNotFoundError(f"No admissions.csv(.gz) under {p}; point --hosp-dir at the hosp folder")
    return p


def _load_sql(path: Path, types: dict[str, str]) -> str:
    type_map = "{" + ", ".join(f"'{k}': '{v}'" for k, v in types.items()) + "}"
    # header + explicit types; everything else sniffed from a large sample.
    return (f"SELECT * FROM read_csv('{path.as_posix()}', header=true, "
            f"types={type_map}, sample_size=100000)")


ADMISSIONS_X = """
CREATE OR REPLACE VIEW admissions_x AS
SELECT a.*,
       p.gender,
       p.anchor_age + (year(a.admittime) - p.anchor_year) AS age_at_admit,
       p.dod,
       date_diff('minute', a.admittime, a.dischtime) / 1440.0 AS los_days
FROM admissions a JOIN patients p USING (subject_id)
"""


def build_database(hosp_dir: str | Path, db_path: str | Path,
                   extra_tables: tuple[str, ...] = ()) -> dict:
    """Load MIMIC-IV hosp CSVs into a DuckDB file. Returns row counts per table."""
    hosp = _resolve_hosp_dir(hosp_dir)
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    counts: dict[str, int] = {}
    todo = list(TABLES.items()) + [(t, {}) for t in extra_tables]
    for name, types in todo:
        t0 = time.time()
        con.execute(f"CREATE OR REPLACE TABLE {name} AS {_load_sql(_csv_path(hosp, name), types)}")
        if "icd_code" in types:
            con.execute(f"UPDATE {name} SET icd_code = trim(icd_code)")
        counts[name] = con.execute(f"SELECT count(*) FROM {name}").fetchone()[0]
        print(f"  {name:<18} {counts[name]:>12,} rows  ({time.time() - t0:.1f}s)", flush=True)
    for table, cols in INDEXES.items():
        for col in cols:
            con.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_{col} ON {table}({col})")
    con.execute(ADMISSIONS_X)
    con.execute("CREATE OR REPLACE TABLE meta AS SELECT ? AS source_dir, now() AS built_at",
                [str(hosp)])
    con.close()
    return counts


def connect(db_path: str | Path, read_only: bool = True) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(db_path), read_only=read_only)


def check_schema(con) -> list[str]:
    """Return a list of problems if expected columns are missing (empty list = OK)."""
    problems = []
    for name, types in TABLES.items():
        cols = {r[0] for r in con.execute(f"DESCRIBE {name}").fetchall()}
        missing = [c for c in types if c not in cols]
        if missing:
            problems.append(f"{name}: missing {missing}")
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--hosp-dir", required=True,
                    help="folder with MIMIC-IV hosp *.csv.gz (or its parent)")
    ap.add_argument("--db", default="data/mimic_demo.duckdb", help="output DuckDB file")
    ap.add_argument("--extra", nargs="*", default=[], choices=OPTIONAL_TABLES,
                    help="also load these optional tables")
    args = ap.parse_args()
    print(f"Loading {args.hosp_dir} -> {args.db}")
    counts = build_database(args.hosp_dir, args.db, tuple(args.extra))
    with connect(args.db) as con:
        problems = check_schema(con)
        n_adm, n_pat = con.execute(
            "SELECT count(*), count(DISTINCT subject_id) FROM admissions_x").fetchone()
    print(f"Done: {n_pat} patients, {n_adm} admissions, {len(counts)} tables.")
    print("Schema check:", "OK" if not problems else problems)


if __name__ == "__main__":
    main()
