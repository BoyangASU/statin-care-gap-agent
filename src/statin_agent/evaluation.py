"""Metrics for the three configurations, risk-coverage, calibration and
field-level error analysis. Inputs: the reference labels and the decisions file.
"""
from __future__ import annotations
import json
from collections import Counter

import numpy as np
import pandas as pd

from .guideline import CATEGORIES

FACT_FIELDS = ["on_statin", "statin_intolerance", "ascvd", "diabetes", "smoker", "bp_treated",
               "hypertension", "ldl", "total_chol", "hdl", "sbp"]


def load_decisions(path) -> list[dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def config_predictions(decisions: list[dict]) -> pd.DataFrame:
    """One row per patient with the prediction and deferral flag under each configuration."""
    rows = []
    for d in decisions:
        s0 = d["samples"][0]
        rows.append({
            "patient_id": d["patient_id"],
            "single_pred": s0["first_category"], "single_defer": False,
            "check_pred": s0["final_category"], "check_defer": bool(s0["final_problems"]),
            "full_pred": d["category"], "full_defer": d["deferred"],
            "confidence": d["confidence"], "verified": d["verified"],
            "tokens": d.get("input_tokens", 0) + d.get("output_tokens", 0),
        })
    return pd.DataFrame(rows)


def _summ(y, pred, defer):
    auto = ~defer
    correct = (pred == y)
    gap_true, gap_pred = (y == "indicated"), (pred == "indicated") & auto
    tp = int((gap_true & gap_pred).sum())
    return {
        "n": int(len(y)),
        "accuracy_all": round(float(correct.mean()), 3),
        "coverage": round(float(auto.mean()), 3),
        "accuracy_automated": round(float(correct[auto].mean()), 3) if auto.any() else float("nan"),
        "wrong_automated": int((~correct & auto).sum()),
        "deferred": int(defer.sum()),
        "care_gap_precision": round(tp / max(int(gap_pred.sum()), 1), 3),
        "care_gap_recall_automated": round(tp / max(int(gap_true.sum()), 1), 3),
    }


def summary_table(labels: pd.DataFrame, decisions: list[dict]) -> pd.DataFrame:
    df = config_predictions(decisions).merge(labels[["patient_id", "label"]], on="patient_id")
    y = df["label"].to_numpy()
    out = {}
    for key, name in [("single", "single pass"), ("check", "+ self-check"),
                      ("full", "+ self-consistency + deferral")]:
        out[name] = _summ(y, df[f"{key}_pred"].to_numpy(), df[f"{key}_defer"].to_numpy())
    return pd.DataFrame(out).T


def risk_coverage(labels, decisions) -> pd.DataFrame:
    """Sweep the deferral threshold over the confidence score (verification failures always defer)."""
    df = config_predictions(decisions).merge(labels[["patient_id", "label"]], on="patient_id")
    correct = (df["full_pred"] == df["label"]).to_numpy()
    conf = df["confidence"].to_numpy() * df["verified"].to_numpy()
    rows = []
    for t in sorted(set(conf) | {0.0}):
        keep = conf >= t
        rows.append({"threshold": t, "coverage": keep.mean(),
                     "risk": 1 - correct[keep].mean() if keep.any() else 0.0})
    return pd.DataFrame(rows)


def calibration(labels, decisions) -> pd.DataFrame:
    df = config_predictions(decisions).merge(labels[["patient_id", "label"]], on="patient_id")
    df["correct"] = df["full_pred"] == df["label"]
    return df.groupby("confidence").agg(n=("correct", "size"), accuracy=("correct", "mean")).reset_index()


def field_errors(labels: pd.DataFrame, decisions: list[dict], attempt="first") -> pd.DataFrame:
    """How often each extracted fact disagrees with the reference fact (first sample)."""
    ref = labels.set_index("patient_id")
    errs, n = Counter(), 0
    for d in decisions:
        eps = d["samples"][0].get("episodes") or []
        if not eps:
            continue
        ep = eps[0] if attempt == "first" else eps[-1]
        r, f = ref.loc[d["patient_id"]], ep.get("facts") or {}
        n += 1
        for k in FACT_FIELDS:
            a, b = f.get(k), r[k]
            b = None if (isinstance(b, float) and np.isnan(b)) else b
            if isinstance(b, (np.bool_, bool)) or k in ("on_statin",):
                bad = bool(a) != bool(b)
            elif a is None or b is None:
                bad = (a is None) != (b is None)
            else:
                bad = abs(float(a) - float(b)) >= 0.6
            errs[k] += bad
    return pd.DataFrame({"field": FACT_FIELDS,
                         "error_rate": [round(errs[k] / max(n, 1), 3) for k in FACT_FIELDS]})


def confusion(labels, decisions, key="full_pred") -> pd.DataFrame:
    df = config_predictions(decisions).merge(labels[["patient_id", "label"]], on="patient_id")
    return pd.crosstab(df["label"], df[key]).reindex(index=CATEGORIES, columns=CATEGORIES, fill_value=0)


# ------------------------------------------------------------ uncertainty
CONFIGS = [("single", "single pass"), ("check", "+ self-check"),
           ("full", "+ self-consistency + deferral")]
CI_METRICS = ["accuracy_all", "coverage", "accuracy_automated", "wrong_automated_rate",
              "care_gap_precision", "care_gap_recall_automated"]


def _metrics_arrays(y, pred, defer):
    auto = ~defer
    correct = pred == y
    gap_true, gap_pred = y == "indicated", (pred == "indicated") & auto
    tp = (gap_true & gap_pred).sum()
    return {
        "accuracy_all": correct.mean(),
        "coverage": auto.mean(),
        "accuracy_automated": correct[auto].mean() if auto.any() else np.nan,
        "wrong_automated_rate": (~correct & auto).mean(),
        "care_gap_precision": tp / gap_pred.sum() if gap_pred.any() else np.nan,
        "care_gap_recall_automated": tp / gap_true.sum() if gap_true.any() else np.nan,
    }


def bootstrap_table(labels, decisions, B=2000, seed=0) -> pd.DataFrame:
    """Point estimates with 95% percentile intervals from a patient-level bootstrap."""
    df = config_predictions(decisions).merge(labels[["patient_id", "label"]], on="patient_id")
    y = df["label"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(df), size=(B, len(df)))
    rows = []
    for key, name in CONFIGS:
        pred, defer = df[f"{key}_pred"].to_numpy(), df[f"{key}_defer"].to_numpy().astype(bool)
        point = _metrics_arrays(y, pred, defer)
        boots = [_metrics_arrays(y[i], pred[i], defer[i]) for i in idx]
        for m in CI_METRICS:
            vals = np.array([b[m] for b in boots], dtype=float)
            vals = vals[~np.isnan(vals)]
            lo, hi = (np.percentile(vals, [2.5, 97.5]) if len(vals) else (np.nan, np.nan))
            rows.append({"configuration": name, "metric": m, "estimate": round(float(point[m]), 3),
                         "ci_low": round(float(lo), 3), "ci_high": round(float(hi), 3)})
    return pd.DataFrame(rows)


def paired_test(labels, decisions, a="single", b="full") -> dict:
    """Exact McNemar test on per-patient 'wrong automated decision' between two configurations."""
    from math import comb
    df = config_predictions(decisions).merge(labels[["patient_id", "label"]], on="patient_id")
    wa = (df[f"{a}_pred"] != df["label"]) & ~df[f"{a}_defer"].astype(bool)
    wb = (df[f"{b}_pred"] != df["label"]) & ~df[f"{b}_defer"].astype(bool)
    n01, n10 = int((~wa & wb).sum()), int((wa & ~wb).sum())   # discordant pairs
    n = n01 + n10
    p = min(1.0, 2 * sum(comb(n, k) for k in range(0, min(n01, n10) + 1)) / 2 ** n) if n else 1.0
    return {"wrong_only_in_" + a: n10, "wrong_only_in_" + b: n01, "p_value_exact_mcnemar": float(f"{p:.3g}")}
