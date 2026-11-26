"""Code sets for the MIMIC-IV statin task (DRAFT, for descriptive stats).

ICD codes in MIMIC-IV are stored without dots, so every set below is a list of
prefixes per ICD version. `icd_sql(spec)` turns a spec into a SQL predicate on
(icd_code, icd_version). Every prefix was checked against the dictionary tables
`d_icd_diagnoses` / `d_icd_procedures` (see `python -m mimic.codes --db ...`).

These lists are a draft for discussion. The final Phase-1 definitions will be
fixed only after the category design is confirmed.
"""
from __future__ import annotations

import re

# A spec maps ICD version -> list of code prefixes; optional "exclude" maps
# version -> prefixes to drop (applied after the include match).
Spec = dict

# --- ASCVD diagnoses (2018 AHA/ACC "clinical ASCVD") -------------------------
ASCVD_DX: dict[str, Spec] = {
    "acute_mi": {9: ["410"], 10: ["I21", "I22"]},
    "old_mi": {9: ["412"], 10: ["I252"]},
    # Angina, other acute and chronic ischemic heart disease (CHD).
    # Coronary aneurysm/dissection (414.1x, I25.4x) is excluded: not atherosclerotic.
    "chd": {9: ["411", "413", "414"], 10: ["I20", "I24", "I25"],
            "exclude": {9: ["4141"], 10: ["I252", "I254"]}},
    # Ischemic stroke. ICD-9 433/434 with 5th digit 1 = "with cerebral infarction";
    # 436 (acute but ill-defined CVD) was used in ICD-9-era stroke measures.
    "ischemic_stroke": {9: ["43301", "43311", "43321", "43331", "43381", "43391",
                            "43401", "43411", "43491", "436"],
                        10: ["I63"]},
    "tia": {9: ["435"], 10: ["G45"], "exclude": {10: ["G454"]}},  # G45.4 = transient global amnesia
    # Atherosclerosis of native/graft arteries of the extremities, PAD unspecified.
    "pad": {9: ["4402", "4403", "4439"],
            10: ["I702", "I703", "I704", "I705", "I706", "I707", "I739"]},
    # History of coronary revascularization (status codes).
    "revasc_history": {9: ["V4581", "V4582"], 10: ["Z951", "Z955", "Z9861"]},
}
# Groups that make an admission an "index" ASCVD admission when coded as the
# principal diagnosis (seq_num = 1): the acute event that brought them in.
INDEX_EVENT_GROUPS = ("acute_mi", "chd", "ischemic_stroke", "tia")

# --- Coronary revascularization procedures during the stay ------------------
REVASC_PROC: dict[str, Spec] = {
    # ICD-9-CM vol. 3: 36.1x bypass; ICD-10-PCS 0210-0213 bypass coronary artery.
    "cabg": {9: ["361"], 10: ["0210", "0211", "0212", "0213"]},
    # ICD-9: 00.66 PTCA, 36.06/36.07 stents, 36.09 other removal of obstruction
    # (36.01/36.02/36.05 were retired before MIMIC's ICD-9 era and are absent);
    # ICD-10-PCS: 0270-0273 dilation, 02C0-02C3 extirpation of coronary artery.
    "pci": {9: ["0066", "3606", "3607", "3609"],
            10: ["0270", "0271", "0272", "0273", "02C0", "02C1", "02C2", "02C3"]},
}

# --- Candidate "justified no statin" diagnoses ------------------------------
JUSTIFIED_DX: dict[str, Spec] = {
    # Statin-specific or likely statin-related harm.
    "statin_adverse_effect": {9: ["E9422"], 10: ["T466X5"]},  # adverse effect of antihyperlipidemics
    "drug_myopathy": {9: ["3594"], 10: ["G720"]},             # toxic / drug-induced myopathy
    "rhabdomyolysis": {9: ["72888"], 10: ["M6282"]},
    # Active liver disease / decompensated cirrhosis / hepatic failure.
    "liver_disease": {9: ["570", "5712", "5715", "5722", "5724"],
                      10: ["K703", "K704", "K72", "K717", "K7460", "K7469", "K766", "K767"]},
    "pregnancy": {9: ["V22", "V23"] + [str(c) for c in range(630, 680)],
                  10: ["O", "Z33", "Z34"]},
    # Comfort care / palliative encounter.
    "palliative": {9: ["V667"], 10: ["Z515"]},
    # Myalgia / myositis, unspecified (weak signal: not statin-specific).
    "myalgia": {9: ["7291"], 10: ["M791"]},
    # Do-not-resuscitate status (context, not a reason by itself).
    "dnr": {9: ["V4986"], 10: ["Z66"]},
    # Dialysis-dependent kidney disease (2018 guideline: initiation not recommended).
    "esrd_dialysis": {9: ["5856", "V4511"], 10: ["N186", "Z992"]},
}

# --- Risk factors for the patient-level (primary + secondary prevention) task ---
RISK_DX: dict[str, Spec] = {
    "diabetes": {9: ["250"], 10: ["E10", "E11", "E13"]},
    "hypertension": {9: ["401", "402", "403", "404", "405"],
                     10: ["I10", "I11", "I12", "I13", "I15", "I16"]},
    # Current tobacco use. History codes (V15.82, Z87.891) mean FORMER smoker.
    "current_smoker": {9: ["3051"], 10: ["F172", "Z720"]},
    "former_smoker": {9: ["V1582"], 10: ["Z87891"]},
}
ANTIHYPERTENSIVE_REGEX = (r"(lisinopril|enalapril|ramipril|captopril|benazepril|quinapril|"
                          r"losartan|valsartan|irbesartan|olmesartan|candesartan|telmisartan|"
                          r"amlodipine|nifedipine|diltiazem|verapamil|felodipine|"
                          r"metoprolol|atenolol|carvedilol|bisoprolol|labetalol|nadolol|propranolol|"
                          r"hydrochlorothiazide|chlorthalidone|indapamide|spironolactone|"
                          r"hydralazine|clonidine|doxazosin|terazosin|sacubitril)")

# --- Labs (d_labitems itemids, Blood) ---------------------------------------
LAB_ITEMS = {
    "alt": [50861],
    "ast": [50878],
    "ck": [50910],
    "total_chol": [50907],
    "hdl": [50904],
    "ldl": [50905, 50906],   # calculated, measured
    "triglycerides": [51000],
}
DEFAULT_ULN = {"alt": 40.0, "ast": 40.0, "ck": 200.0}  # used when ref_range_upper is missing

# --- Drugs -------------------------------------------------------------------
STATIN_GENERICS = ("atorvastatin", "simvastatin", "rosuvastatin", "pravastatin",
                   "lovastatin", "pitavastatin", "fluvastatin")
STATIN_BRANDS = ("lipitor", "zocor", "crestor", "pravachol", "mevacor", "altoprev",
                 "livalo", "zypitamag", "lescol", "vytorin", "caduet", "advicor",
                 "simcor", "liptruzet", "roszet", "ezallor", "flolipid", "atorvaliq")
# Regex with a word boundary before the stem so "nystatin" never matches.
STATIN_REGEX = r"(^|[^a-z])(" + "|".join(
    [g.replace("statin", "") + "statin" for g in STATIN_GENERICS] + list(STATIN_BRANDS)) + ")"
NONSTATIN_LIPID_REGEX = r"(ezetimibe|zetia|alirocumab|praluent|evolocumab|repatha|inclisiran|leqvio|bempedoic|nexlizet|nexletol)"
# eMAR events that mean the dose was actually given.
EMAR_GIVEN = ("Administered", "Delayed Administered", "Partial Administered",
              "Administered in Other Location", "Administered Bolus from IV Drip")

_statin_re = re.compile(STATIN_REGEX)


def is_statin(name: str | None) -> bool:
    return bool(name) and bool(_statin_re.search(name.lower()))


def icd_sql(spec: Spec, code_col: str = "icd_code", ver_col: str = "icd_version") -> str:
    """SQL predicate that is true when (code, version) matches the spec."""
    parts = []
    excl = spec.get("exclude", {})
    for ver in (9, 10):
        prefixes = spec.get(ver, [])
        if not prefixes:
            continue
        inc = " OR ".join(f"starts_with({code_col}, '{p}')" for p in prefixes)
        cond = f"({ver_col} = {ver} AND ({inc})"
        if excl.get(ver):
            ex = " OR ".join(f"starts_with({code_col}, '{p}')" for p in excl[ver])
            cond += f" AND NOT ({ex})"
        parts.append(cond + ")")
    return "(" + " OR ".join(parts) + ")" if parts else "FALSE"


def icd_match(code: str, version: int, spec: Spec) -> bool:
    code = code.strip()
    if not any(code.startswith(p) for p in spec.get(version, [])):
        return False
    return not any(code.startswith(p) for p in spec.get("exclude", {}).get(version, []))


def _audit(db_path: str):
    """Print how many dictionary codes each prefix matches, plus sample titles."""
    import duckdb
    con = duckdb.connect(db_path, read_only=True)
    for table, groups in (("d_icd_diagnoses", {**ASCVD_DX, **JUSTIFIED_DX}),
                          ("d_icd_procedures", REVASC_PROC)):
        print(f"\n== {table}")
        for name, spec in groups.items():
            for ver in (9, 10):
                for p in spec.get(ver, []):
                    n, title = con.execute(
                        f"SELECT count(*), min(long_title) FROM {table} "
                        f"WHERE icd_version = ? AND starts_with(icd_code, ?)", [ver, p]).fetchone()
                    flag = "" if n else "   <-- NO MATCH"
                    print(f"  {name:<22} ICD-{ver:<2} {p:<7} {n:>5} codes  e.g. {str(title)[:60]}{flag}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Audit code prefixes against the ICD dictionaries")
    ap.add_argument("--db", default="data/mimic_demo.duckdb")
    _audit(ap.parse_args().db)
