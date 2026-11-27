"""Self-check for the MIMIC agent: re-derive what can be checked mechanically.

Like the prototype's verifier, this is deliberately narrower than the reference
labels. It checks:
  1. a valid decision was submitted;
  2. the four numbers (LDL, total cholesterol, HDL, systolic BP) equal the most
     recent plausible value within the window;
  3. on_statin agrees with the index admission's statin orders and eMAR;
  4. every cited ICD code exists in this patient's record up to the index
     admission and belongs to the fact it was cited for (no invented codes);
  5. diagnoses were actually listed (process check);
  6. the category follows from the submitted facts.
It does NOT check whether a coded fact set to false was missed (a missed ASCVD,
diabetes or smoking code), or bp_treated; those errors are left to
self-consistency and deferral, and are measured against the reference.
"""
from __future__ import annotations

from mimic import codes as C
from mimic.guideline import CATEGORIES, Facts, classify

from .record import MimicRecord, latest_valid, reference_on_statin

LAB_NAMES = {"ldl": "LDL", "total_chol": "total cholesterol", "hdl": "HDL", "sbp": "systolic BP"}
FACT_SPECS = {
    "ascvd": list(C.ASCVD_DX.values()) + list(C.REVASC_PROC.values()),
    "diabetes": [C.RISK_DX["diabetes"]],
    "hypertension": [C.RISK_DX["hypertension"]],
    "smoker": [C.RISK_DX["current_smoker"]],
    "statin_intolerance": [C.JUSTIFIED_DX[k] for k in ("statin_adverse_effect", "drug_myopathy",
                                                       "rhabdomyolysis", "liver_disease", "pregnancy")],
}


def _record_codes(rec: MimicRecord) -> dict[str, int]:
    """code -> ICD version, for every diagnosis and procedure code up to the index admission."""
    out = {}
    for r in rec.diagnoses():
        out[r["icd_code"]] = r["icd_version"]
    for r in rec.procedures():
        out.setdefault(r["icd_code"], r["icd_version"])
    return out


def verify(rec: MimicRecord, ep, lookback_days: int = 365) -> list[str]:
    if ep.error or ep.category not in CATEGORIES:
        return [f"No valid decision was submitted ({ep.error or ep.category})."]
    f, problems = ep.facts or {}, []
    for key, name in LAB_NAMES.items():
        ref = latest_valid(rec, key, lookback_days)
        got = f.get(key)
        try:
            same = (got is None and ref is None) or (
                got is not None and ref is not None and abs(float(got) - ref) < 0.6)
        except (TypeError, ValueError):
            same = False
        if not same:
            problems.append(f"{key}={got}, but the most recent plausible {name} within {lookback_days} "
                            f"days of the index time is {ref}. Re-check days_before_index and discard "
                            f"implausible values.")
    if f.get("on_statin") is not None and bool(f["on_statin"]) != reference_on_statin(rec):
        problems.append(f"on_statin={f['on_statin']} is inconsistent with the index admission's statin "
                        f"orders and administrations (24-hour order rule, 48-hour dose rule; earlier "
                        f"admissions do not count).")
    codes = _record_codes(rec)
    cited = getattr(ep, "evidence_codes", None) or {}
    for fact, specs in FACT_SPECS.items():
        for code in cited.get(fact) or []:
            code = str(code).replace(".", "").strip().upper()
            if code not in codes:
                problems.append(f"Code {code} cited for {fact} is not in this patient's record.")
            elif not any(C.icd_match(code, codes[code], s) for s in specs):
                problems.append(f"Code {code} cited for {fact} does not define {fact}.")
        if f.get(fact) and not cited.get(fact):
            problems.append(f"{fact}=true but no supporting ICD code was cited in evidence_codes.")
    if "list_diagnoses" not in {t["tool"] for t in ep.trace}:
        problems.append("Diagnoses were not reviewed (call list_diagnoses).")
    try:
        cat, _ = classify(Facts(**{k: f.get(k) for k in Facts.__dataclass_fields__}))
        if cat != ep.category:
            problems.append(f"Submitted category {ep.category} does not follow from the submitted facts "
                            f"(guideline gives {cat}).")
    except Exception as e:
        problems.append(f"Submitted facts are incomplete or malformed: {e}.")
    return problems
