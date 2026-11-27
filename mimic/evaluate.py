"""Evaluate a MIMIC agent run with the prototype's metrics (mimic/evaluation.py, a copy of the prototype's):
the three configurations (single pass, + self-check, + self-consistency and deferral),
field-level extraction errors, confusion matrix, risk-coverage, optional bootstrap CIs.

Usage:  python -m mimic.evaluate --labels-dir results_mimic/demo --run simulated_n81_k3 [--bootstrap 2000]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from mimic import evaluation as ev


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--labels-dir", default="results_mimic/demo")
    p.add_argument("--run", required=True, help="folder name under <labels-dir>/runs/")
    p.add_argument("--bootstrap", type=int, default=0)
    a = p.parse_args()
    base = Path(a.labels_dir)
    run = base / "runs" / a.run
    labels = pd.read_csv(base / "labels.csv", dtype={"patient_id": str})
    dec = ev.load_decisions(run / "decisions.jsonl")
    for d in dec:
        d["patient_id"] = str(d["patient_id"])
    labels = labels[labels["patient_id"].isin({d["patient_id"] for d in dec})]

    table = ev.summary_table(labels, dec)
    fe = ev.field_errors(labels, dec).merge(ev.field_errors(labels, dec, "final"), on="field",
                                            suffixes=("_first_attempt", "_after_self_check"))
    conf = ev.confusion(labels, dec)
    rc = ev.risk_coverage(labels, dec)
    tokens = sum(d.get("input_tokens", 0) + d.get("output_tokens", 0) for d in dec)
    pd.set_option("display.width", 160)
    print(f"Run {a.run}: {len(dec)} patients, {tokens:,} tokens\n")
    print("Label distribution:\n" + labels["label"].value_counts().to_string() + "\n")
    print(table.to_string(), "\n")
    print("Field-level extraction errors (sample 1):\n" + fe.to_string(index=False), "\n")
    print("Confusion (rows = reference, cols = final decision):\n" + conf.to_string())
    if a.bootstrap:
        ci = ev.bootstrap_table(labels, dec, B=a.bootstrap)
        print("\n95% bootstrap CIs:\n" + ci.to_string(index=False))
        ci.to_csv(run / "summary_ci.csv", index=False)
        print("\nPaired test, single pass vs full pipeline:", ev.paired_test(labels, dec))
    table.to_csv(run / "summary.csv"); fe.to_csv(run / "field_errors.csv", index=False)
    conf.to_csv(run / "confusion.csv"); rc.to_csv(run / "risk_coverage.csv", index=False)
    (run / "summary.json").write_text(json.dumps({"tokens": tokens, "table": table.to_dict("index")},
                                                 indent=2, default=str))
    print(f"\nSaved tables to {run}")


if __name__ == "__main__":
    main()
