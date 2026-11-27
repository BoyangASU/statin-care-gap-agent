"""Tools the MIMIC agent can call. Every tool is bound to ONE MimicRecord
(one patient, nothing after the index admission's discharge).

The data tools return the record as it is: stale labs, implausible values,
orders that were cancelled at once, history codes next to current-use codes.
Applying the rules to that raw data is the agent's job, which is what the
verifier and the evaluation measure.

The two calculator tools and the fact schema are the prototype's, copied unchanged.
"""
from __future__ import annotations

from mimic.guideline import CATEGORIES, Facts, classify, pce_10yr_risk

from .record import MimicRecord, LAB_NAMES


# ------------------------------------------------- calculator tools (as in the prototype)
def compute_ascvd_risk(rec, age, sex, race, total_chol, hdl, sbp, bp_treated, smoker, diabetes):
    r = pce_10yr_risk(float(age), sex, race, float(total_chol), float(hdl), float(sbp),
                      bool(bp_treated), bool(smoker), bool(diabetes))
    return {"ten_year_ascvd_risk": round(r, 4)}


def check_statin_guideline(rec, **facts):
    f = Facts(**{k: facts.get(k) for k in Facts.__dataclass_fields__})
    cat, risk = classify(f)
    return {"category": cat, "ten_year_ascvd_risk": None if risk is None else round(risk, 4)}


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


def get_demographics(rec: MimicRecord):
    return rec.demographics()


def list_admissions(rec: MimicRecord):
    return rec.admissions()


DX_COLUMNS = ["icd_code", "icd_version", "title", "last_seen", "n_admissions", "on_index_admission"]


def list_diagnoses(rec: MimicRecord, keyword: str | None = None):
    """Compact table (columns + rows): long records have hundreds of codes."""
    rows = rec.diagnoses(keyword=keyword)
    return {"n_codes": len(rows), "columns": DX_COLUMNS, "rows": [[r[c] for c in DX_COLUMNS] for r in rows]}


def rows_as_dicts(table: dict) -> list[dict]:
    return [dict(zip(table["columns"], r)) for r in table["rows"]]


def list_procedures(rec: MimicRecord):
    return rec.procedures()


def list_medication_orders(rec: MimicRecord, keyword: str | None = None, all_admissions: bool = False):
    return rec.medication_orders(keyword=keyword, all_admissions=bool(all_admissions))


def list_administrations(rec: MimicRecord, keyword: str | None = None):
    return rec.administrations(keyword=keyword)


def get_labs(rec: MimicRecord, name: str | None = None, keyword: str | None = None, limit: int = 10):
    return rec.labs(name=name, keyword=keyword, limit=limit)


def get_blood_pressure(rec: MimicRecord, limit: int = 10):
    return rec.blood_pressure(limit=limit)


TOOL_FUNCS = {
    "get_demographics": get_demographics,
    "list_admissions": list_admissions,
    "list_diagnoses": list_diagnoses,
    "list_procedures": list_procedures,
    "list_medication_orders": list_medication_orders,
    "list_administrations": list_administrations,
    "get_labs": get_labs,
    "get_blood_pressure": get_blood_pressure,
    "compute_ascvd_risk": compute_ascvd_risk,
    "check_statin_guideline": check_statin_guideline,
}

_kw = {"type": "string", "description": "Case-insensitive text to filter by."}
TOOL_SPECS = [
    {"name": "get_demographics",
     "description": "Age at the index admission, sex, race, the index time (discharge of the index admission) and its hadm_id.",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "list_admissions",
     "description": "All hospital admissions up to and including the index admission, oldest first, with principal diagnosis and discharge location.",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "list_diagnoses",
     "description": ("ICD-9 and ICD-10 diagnosis codes from ALL admissions up to the index admission, one row per code "
                     "(codes have no dots, e.g. I214 = I21.4), newest first, as a table of columns and rows. "
                     "on_index_admission says whether the code is on the index admission. Use keyword to narrow."),
     "parameters": {"type": "object", "properties": {"keyword": _kw}}},
    {"name": "list_procedures",
     "description": "ICD-9 / ICD-10-PCS procedure codes from all admissions up to the index admission (e.g. CABG, PCI).",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "list_medication_orders",
     "description": ("Inpatient medication orders grouped by drug: first start, last stop, and for the index admission "
                     "last_stop_hours_before_discharge (negative = still running after discharge). "
                     "Default: index admission only; all_admissions=true shows earlier admissions too."),
     "parameters": {"type": "object", "properties": {
         "keyword": {"type": "string", "description": "Text in the drug name, e.g. 'statin' or 'lisinopril'."},
         "all_admissions": {"type": "boolean", "default": False}}}},
    {"name": "list_administrations",
     "description": "Doses recorded by nurses (eMAR) on the index admission, newest first, with hours_before_discharge.",
     "parameters": {"type": "object", "properties": {"keyword": _kw}}},
    {"name": "get_labs",
     "description": (f"Blood test results up to the index time, newest first, inpatient and outpatient. "
                     f"name is one of {sorted(LAB_NAMES)}; or use keyword for other tests. "
                     "Each row has days_before_index and the upper limit of normal."),
     "parameters": {"type": "object", "properties": {
         "name": {"type": "string", "enum": sorted(LAB_NAMES)},
         "keyword": _kw, "limit": {"type": "integer", "default": 10}}}},
    {"name": "get_blood_pressure",
     "description": "Outpatient blood pressure readings (systolic/diastolic mmHg) up to the index date, newest first, with days_before_index.",
     "parameters": {"type": "object", "properties": {"limit": {"type": "integer", "default": 10}}}},
    {"name": "compute_ascvd_risk", "description": "10-year ASCVD risk from the 2013 Pooled Cohort Equations.",
     "parameters": {"type": "object", "properties": {
         k: FACT_SCHEMA["properties"][k] for k in
         ("age", "sex", "race", "total_chol", "hdl", "sbp", "bp_treated", "smoker", "diabetes")},
         "required": ["age", "sex", "race", "total_chol", "hdl", "sbp", "bp_treated", "smoker", "diabetes"]}},
    {"name": "check_statin_guideline",
     "description": "Apply the statin guideline (ACC/AHA 2018, USPSTF 2022) to the facts you extracted. Use null for missing values.",
     "parameters": FACT_SCHEMA},
]

SUBMIT_SPEC = {
    "name": "submit_decision",
    "description": "Submit the final decision for this patient. Call exactly once, at the end.",
    "parameters": {"type": "object", "properties": {
        "category": {"type": "string", "enum": CATEGORIES},
        "facts": FACT_SCHEMA,
        "evidence_codes": {
            "type": "object",
            "description": ("ICD codes you relied on for each coded fact that you set to true, exactly as listed "
                            "by list_diagnoses / list_procedures, e.g. {\"ascvd\": [\"I214\"], \"diabetes\": [\"25000\"]}."),
            "properties": {k: {"type": "array", "items": {"type": "string"}} for k in
                           ("ascvd", "diabetes", "hypertension", "smoker", "statin_intolerance")}},
        "evidence": {"type": "string", "description": "Dates and values you relied on."},
        "rationale": {"type": "string"}},
        "required": ["category", "facts", "rationale"]},
}
ALL_SPECS = TOOL_SPECS + [SUBMIT_SPEC]


def execute(rec: MimicRecord, name: str, args: dict):
    if name not in TOOL_FUNCS:
        return {"error": f"Unknown tool {name}."}
    try:
        return TOOL_FUNCS[name](rec, **(args or {}))
    except Exception as e:                  # report tool errors back to the agent
        return {"error": f"{type(e).__name__}: {e}"}
