"""Reference labels for the MIMIC agent (task A, patient-level screening).

One row per patient: the index admission, the reference facts and the category
from the prototype's guideline rules. Written to <out-dir>/labels.csv, which
`mimic.run_agent` samples from and `mimic.evaluate` scores against.

Usage (from the repository root):
    python -m mimic.build_labels --db data/mimic_demo.duckdb --out-dir results_mimic/demo
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import duckdb
import pandas as pd

from mimic.guideline import classify

from mimic.agent.record import MimicRecord, index_admissions, reference_evidence, reference_facts


def build_labels(con, min_age=40, max_age=80, lookback_days=365) -> pd.DataFrame:
    rows = []
    for subject_id, hadm_id in index_admissions(con, min_age, max_age):
        rec = MimicRecord(con, subject_id, hadm_id)
        f = reference_facts(rec, lookback_days)
        cat, risk = classify(f)
        ev = reference_evidence(rec)
        rows.append({"patient_id": str(subject_id), "index_hadm_id": hadm_id,
                     "index_time": rec.index_time, "label": cat, "pce_risk": risk, **asdict(f),
                     **{f"codes_{k}": ";".join(v) for k, v in ev.items()}})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--db", default="data/mimic_demo.duckdb")
    ap.add_argument("--out-dir", default="results_mimic/demo")
    ap.add_argument("--min-age", type=int, default=40)
    ap.add_argument("--max-age", type=int, default=80)
    ap.add_argument("--lookback-days", type=int, default=365)
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(a.db, read_only=True)
    df = build_labels(con, a.min_age, a.max_age, a.lookback_days)
    df.to_csv(out / "labels.csv", index=False)
    print(f"{len(df)} patients (age {a.min_age}-{a.max_age} at the index admission) -> {out / 'labels.csv'}")
    print(df["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
