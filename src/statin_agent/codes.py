"""Clinical code sets used by Synthea (SNOMED CT, LOINC) and drug-name lists.

Kept in one place so the reference logic and the documentation agree.
"""

# --- SNOMED CT: clinical ASCVD (secondary-prevention group) -----------------
ASCVD_SNOMED = {
    22298006: "Myocardial infarction",
    401314000: "Acute NSTEMI",
    401303003: "Acute STEMI",
    399211009: "History of myocardial infarction",
    414545008: "Ischemic heart disease",
    399261000: "History of coronary artery bypass grafting",
    230690007: "Cerebrovascular accident (stroke)",
}
# Diabetes: the diagnosis itself, plus complications coded "due to diabetes" (Synthea
# often records these without a separate 44054006 code). Prediabetes (714628002) and
# hyperglycemia (80394007) are NOT diabetes.
DIABETES_T2_SNOMED = {
    44054006: "Diabetes mellitus type 2",
    127013003: "Disorder of kidney due to diabetes mellitus",
    90781000119102: "Microalbuminuria due to type 2 diabetes mellitus",
    157141000119108: "Proteinuria due to type 2 diabetes mellitus",
    368581000119106: "Neuropathy due to type 2 diabetes mellitus",
    1551000119108: "Nonproliferative diabetic retinopathy due to type II diabetes mellitus",
    1501000119109: "Proliferative diabetic retinopathy due to type II diabetes mellitus",
    97331000119101: "Macular edema and retinopathy due to type 2 diabetes mellitus",
}
HYPERTENSION_SNOMED = {59621000: "Essential hypertension"}

# --- LOINC -------------------------------------------------------------------
LOINC = {
    "total_chol": "2093-3",   # Cholesterol [Mass/volume] in Serum or Plasma
    "ldl": "18262-6",         # Cholesterol in LDL, direct assay
    "hdl": "2085-9",          # Cholesterol in HDL
    "sbp": "8480-6",          # Systolic blood pressure
    "smoking": "72166-2",     # Tobacco smoking status
    "a1c": "4548-4",          # Hemoglobin A1c
}
# Physiologically plausible ranges; values outside are treated as invalid.
PLAUSIBLE = {"total_chol": (50, 1000), "ldl": (10, 600), "hdl": (5, 200), "sbp": (60, 300)}
CURRENT_SMOKER_VALUES = ("smokes tobacco daily", "smokes tobacco", "current")

# --- Drug classes (matched on lowercase ingredient names) --------------------
STATINS = ("atorvastatin", "simvastatin", "rosuvastatin", "pravastatin",
           "lovastatin", "pitavastatin", "fluvastatin")
ANTIHYPERTENSIVES = ("lisinopril", "amlodipine", "hydrochlorothiazide", "losartan",
                     "metoprolol", "atenolol", "bisoprolol", "valsartan",
                     "chlorthalidone", "enalapril", "ramipril", "carvedilol")


def is_statin(desc: str) -> bool:
    d = desc.lower()
    return any(s in d for s in STATINS)          # avoids matching "nystatin"


def is_antihypertensive(desc: str) -> bool:
    d = desc.lower()
    return any(s in d for s in ANTIHYPERTENSIVES)
