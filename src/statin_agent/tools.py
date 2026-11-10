"""Tools the agent can call. Every tool is bound to ONE patient record.

The data tools return the record as it is (stale results, implausible values,
stopped medications included). Applying the clinical rules to that raw data is
the agent's job; this is where real agents make mistakes, and what the
verifier and the evaluation measure.
"""
from __future__ import annotations
from datetime import date
from .db import PatientRecord
from .guideline import Facts, classify, pce_10yr_risk, CATEGORIES


def _iso(d):
    return d.isoformat() if isinstance(d, date) else d


# ---------------------------------------------------------------- data tools
def get_demographics(rec: PatientRecord):
    return rec.demographics()


def list_conditions(rec: PatientRecord):
    """One row per diagnosis code: first onset and (if resolved) the resolution date."""
    out = {}
    for c in rec.conditions():
        r = out.setdefault(c["code"], {"code": str(c["code"]), "description": c["description"],
                                       "onset": _iso(c["start"]), "resolved": _iso(c["stop"])})
        r["resolved"] = _iso(c["stop"])            # keep the most recent episode's status
    return list(out.values())


def list_medications(rec: PatientRecord):
    """One row per drug: first start, most recent start, and most recent stop (null = ongoing)."""
    out = {}
    for m in rec.medications():
        r = out.setdefault(m["description"], {"description": m["description"],
                                              "first_start": _iso(m["start"])})
        r["most_recent_start"] = _iso(m["start"])
        r["most_recent_stop"] = _iso(m["stop"])
    return list(out.values())


def list_allergies(rec: PatientRecord):
    return [{"description": a["description"], "start": _iso(a["start"]), "stop": _iso(a["stop"])}
            for a in rec.allergies()]


def get_observations(rec: PatientRecord, loinc: str | None = None, keyword: str | None = None,
                     limit: int = 10):
    """Most recent observations first, filtered by LOINC code or a keyword in the name."""
    rows = rec.observations(loinc=loinc, keyword=keyword, limit=min(int(limit), 50))
    return [{"date": _iso(r["date"]), "days_before_index": rec.days_before_index(r["date"]),
             "loinc": r["loinc"], "name": r["description"], "value": r["value"], "units": r["units"]}
            for r in rows]


# ---------------------------------------------------------- calculator tools
def compute_ascvd_risk(rec, age, sex, race, total_chol, hdl, sbp, bp_treated, smoker, diabetes):
    r = pce_10yr_risk(float(age), sex, race, float(total_chol), float(hdl), float(sbp),
                      bool(bp_treated), bool(smoker), bool(diabetes))
    return {"ten_year_ascvd_risk": round(r, 4)}


def check_statin_guideline(rec, **facts):
    f = Facts(**{k: facts.get(k) for k in Facts.__dataclass_fields__})
    cat, risk = classify(f)
    return {"category": cat, "ten_year_ascvd_risk": None if risk is None else round(risk, 4)}


TOOL_FUNCS = {
    "get_demographics": get_demographics,
    "list_conditions": list_conditions,
    "list_medications": list_medications,
    "list_allergies": list_allergies,
    "get_observations": get_observations,
    "compute_ascvd_risk": compute_ascvd_risk,
    "check_statin_guideline": check_statin_guideline,
}

_num = {"type": ["number", "null"]}
FACT_SCHEMA = {
    "type": "object",
    "properties": {
        "age": {"type": "number"}, "sex": {"type": "string", "enum": ["male", "female"]},
        "race": {"type": "string"}, "on_statin": {"type": "boolean"},
        "statin_intolerance": {"type": "boolean"}, "ascvd": {"type": "boolean"},
        "diabetes": {"type": "boolean"}, "smoker": {"type": "boolean"},
        "bp_treated": {"type": "boolean"}, "hypertension": {"type": "boolean"},
        "ldl": _num, "total_chol": _num, "hdl": _num, "sbp": _num,
    },
    "required": list(Facts.__dataclass_fields__),
}

TOOL_SPECS = [
    {"name": "get_demographics", "description": "Age at the index date, sex, race, and the index date.",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "list_conditions", "description": "All recorded diagnoses (SNOMED CT code, description, onset, resolved date or null).",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "list_medications", "description": "All recorded medications with first start, most recent start and most recent stop date (null = ongoing).",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "list_allergies", "description": "Recorded allergies and intolerances.",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "get_observations", "description": "Labs, vitals and survey answers, most recent first. Filter by LOINC code or by a keyword in the name. Each row includes days_before_index.",
     "parameters": {"type": "object", "properties": {
         "loinc": {"type": "string", "description": "LOINC code, e.g. 18262-6 for LDL."},
         "keyword": {"type": "string", "description": "Case-insensitive text in the observation name."},
         "limit": {"type": "integer", "default": 10}}}},
    {"name": "compute_ascvd_risk", "description": "10-year ASCVD risk from the 2013 Pooled Cohort Equations.",
     "parameters": {"type": "object", "properties": {
         k: FACT_SCHEMA["properties"][k] for k in
         ("age", "sex", "race", "total_chol", "hdl", "sbp", "bp_treated", "smoker", "diabetes")},
         "required": ["age", "sex", "race", "total_chol", "hdl", "sbp", "bp_treated", "smoker", "diabetes"]}},
    {"name": "check_statin_guideline", "description": "Apply the statin guideline (ACC/AHA 2018, USPSTF 2022) to the facts you extracted. Use null for missing values.",
     "parameters": FACT_SCHEMA},
]

SUBMIT_SPEC = {
    "name": "submit_decision",
    "description": "Submit the final decision for this patient. Call exactly once, at the end.",
    "parameters": {"type": "object", "properties": {
        "category": {"type": "string", "enum": CATEGORIES},
        "facts": FACT_SCHEMA,
        "evidence": {"type": "string", "description": "Dates and values you relied on."},
        "rationale": {"type": "string"}},
        "required": ["category", "facts", "rationale"]},
}
ALL_SPECS = TOOL_SPECS + [SUBMIT_SPEC]


def execute(rec: PatientRecord, name: str, args: dict):
    if name not in TOOL_FUNCS:
        return {"error": f"Unknown tool {name}."}
    try:
        return TOOL_FUNCS[name](rec, **(args or {}))
    except Exception as e:                  # report tool errors back to the agent
        return {"error": f"{type(e).__name__}: {e}"}
