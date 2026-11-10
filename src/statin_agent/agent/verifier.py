"""Self-check: re-derive what can be checked from the record and compare.

The verifier is deliberately narrower than the reference labels. It checks
facts that can be verified mechanically (lab currency and validity, active
statin, that allergies were checked, internal consistency). It does NOT know
the full clinical answer, so errors it cannot see (missed ASCVD codes,
misread smoking status, missed antihypertensives) are left to
self-consistency and deferral.
"""
from __future__ import annotations
from ..codes import is_statin
from ..facts import latest_valid
from ..guideline import CATEGORIES, Facts, classify

LAB_NAMES = {"ldl": "LDL", "total_chol": "total cholesterol", "hdl": "HDL", "sbp": "systolic BP"}


def verify(rec, ep, lookback_days: int = 365) -> list[str]:
    if ep.error or ep.category not in CATEGORIES:
        return [f"No valid decision was submitted ({ep.error or ep.category})."]
    f, problems = ep.facts or {}, []
    for key, name in LAB_NAMES.items():
        ref = latest_valid(rec, key, lookback_days)
        got = f.get(key)
        same = (got is None and ref is None) or (
            got is not None and ref is not None and abs(float(got) - ref) < 0.6)
        if not same:
            problems.append(f"{key}={got}, but the most recent plausible {name} within "
                            f"{lookback_days} days is {ref}. Re-check dates (365-day window) and "
                            f"discard implausible values.")
    meds = rec.medications()
    active = any(is_statin(m["description"]) and (m["stop"] is None or m["stop"] > rec.index_date)
                 for m in meds)
    if f.get("on_statin") is not None and bool(f["on_statin"]) != active:
        problems.append(f"on_statin={f['on_statin']} is inconsistent with medication stop dates.")
    if "list_allergies" not in {t["tool"] for t in ep.trace}:
        problems.append("Allergies were not checked for statin intolerance (call list_allergies).")
    try:
        cat, _ = classify(Facts(**{k: f.get(k) for k in Facts.__dataclass_fields__}))
        if cat != ep.category:
            problems.append(f"Submitted category {ep.category} does not follow from the submitted "
                            f"facts (guideline gives {cat}).")
    except Exception as e:
        problems.append(f"Submitted facts are incomplete or malformed: {e}.")
    return problems
