"""Reference (ground-truth) fact extraction from a patient record.

This encodes the correct reading of the record: only results within the
look-back window, only physiologically plausible values, only medications
active on the index date, the most recent smoking status, and all ASCVD codes.
It is used to build labels and by the verifier, never by the agent itself.
"""
from __future__ import annotations
from .codes import (ASCVD_SNOMED, DIABETES_T2_SNOMED, HYPERTENSION_SNOMED, LOINC, PLAUSIBLE,
                    CURRENT_SMOKER_VALUES, is_statin, is_antihypertensive)
from .db import PatientRecord
from .guideline import Facts, classify


def _active(rec: PatientRecord, row) -> bool:
    return row["stop"] is None or row["stop"] > rec.index_date


def latest_valid(rec: PatientRecord, key: str, lookback_days: int):
    """Most recent plausible numeric value within the look-back window."""
    lo, hi = PLAUSIBLE[key]
    for r in rec.observations(loinc=LOINC[key], limit=200):
        if rec.days_before_index(r["date"]) > lookback_days:
            return None
        try:
            v = float(r["value"])
        except (TypeError, ValueError):
            continue
        if lo <= v <= hi:
            return round(v, 1)
    return None


def reference_facts(rec: PatientRecord, lookback_days: int = 365) -> Facts:
    demo = rec.demographics()
    conds = rec.conditions()
    chronic = ASCVD_SNOMED.keys() | DIABETES_T2_SNOMED.keys()   # these never resolve
    codes = {c["code"] for c in conds if c["stop"] is None or c["stop"] > rec.index_date
             or c["code"] in chronic}
    meds = [m for m in rec.medications() if _active(rec, m)]
    smoke = rec.observations(loinc=LOINC["smoking"], limit=1)
    allergies = [a["description"].lower() for a in rec.allergies()]
    return Facts(
        age=demo["age"], sex=demo["sex"], race=demo["race"],
        on_statin=any(is_statin(m["description"]) for m in meds),
        statin_intolerance=any(is_statin(a) for a in allergies),
        ascvd=bool(codes & ASCVD_SNOMED.keys()),
        diabetes=bool(codes & DIABETES_T2_SNOMED.keys()),
        hypertension=bool(codes & HYPERTENSION_SNOMED.keys()),
        bp_treated=any(is_antihypertensive(m["description"]) for m in meds),
        smoker=bool(smoke) and any(s in str(smoke[0]["value"]).lower() for s in CURRENT_SMOKER_VALUES),
        ldl=latest_valid(rec, "ldl", lookback_days),
        total_chol=latest_valid(rec, "total_chol", lookback_days),
        hdl=latest_valid(rec, "hdl", lookback_days),
        sbp=latest_valid(rec, "sbp", lookback_days),
    )


def reference_label(rec: PatientRecord, lookback_days: int = 365) -> tuple[str, float | None, Facts]:
    f = reference_facts(rec, lookback_days)
    cat, risk = classify(f)
    return cat, risk, f
