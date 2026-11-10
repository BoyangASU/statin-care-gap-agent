"""Cheap screening features for every patient, from structured data only.

These are what a health system can pull with one SQL query, before any agent
runs. They are deliberately the *naive* versions of the facts the agent must
reason about: the latest LDL regardless of date or plausibility, any statin
ever listed, any ASCVD or diabetes code ever recorded. The expensive step (the
agent, or a clinician) is what resolves dates, stop dates and implausible values.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from ..codes import ASCVD_SNOMED, DIABETES_T2_SNOMED, HYPERTENSION_SNOMED, LOINC, is_statin

FEATURES = ["age", "male", "black", "ascvd_any", "diabetes_any", "hypertension_any",
            "statin_ever", "ldl_latest", "days_since_ldl", "n_ldl", "n_conditions",
            "n_medications"]


def screening_features(con, patient_ids, index_date) -> pd.DataFrame:
    con = con.cursor()                       # own cursor: no clash with open results
    ids = pd.DataFrame({"patient_id": list(patient_ids)})
    con.register("pool_ids", ids)
    q = lambda sql: con.execute(sql, [index_date] * sql.count("?")).df()
    demo = q("""SELECT p.patient_id, date_diff('year', birthdate, ?::DATE) AS age,
                       (gender = 'M')::INT AS male, (race = 'black')::INT AS black
                FROM patients p JOIN pool_ids USING (patient_id)""")
    codes = lambda s: ",".join(str(c) for c in s)
    cond = q(f"""SELECT patient_id,
                   max((code IN ({codes(ASCVD_SNOMED)}))::INT) AS ascvd_any,
                   max((code IN ({codes(DIABETES_T2_SNOMED)}))::INT) AS diabetes_any,
                   max((code IN ({codes(HYPERTENSION_SNOMED)}))::INT) AS hypertension_any,
                   count(DISTINCT code) AS n_conditions
                FROM conditions JOIN pool_ids USING (patient_id)
                WHERE start <= ? GROUP BY patient_id""")
    meds = q("""SELECT patient_id, description FROM medications JOIN pool_ids USING (patient_id)
                WHERE start <= ?""")
    med = (meds.assign(statin=meds["description"].map(is_statin))
               .groupby("patient_id").agg(statin_ever=("statin", "max"),
                                          n_medications=("description", "nunique")).reset_index())
    ldl = q(f"""SELECT patient_id, arg_max(TRY_CAST(value AS DOUBLE), date) AS ldl_latest,
                       date_diff('day', max(date), ?::DATE) AS days_since_ldl, count(*) AS n_ldl
                FROM observations JOIN pool_ids USING (patient_id)
                WHERE loinc = '{LOINC["ldl"]}' AND date <= ? GROUP BY patient_id""")
    con.unregister("pool_ids")
    df = ids
    for part in (demo, cond, med, ldl):
        df = df.merge(part, on="patient_id", how="left")
    df["statin_ever"] = df["statin_ever"].fillna(False).astype(int)
    df["days_since_ldl"] = df["days_since_ldl"].fillna(9999)     # never measured
    df["ldl_latest"] = df["ldl_latest"].fillna(0.0)
    df = df.fillna(0)
    return df[["patient_id"] + FEATURES]


def minmax(X: np.ndarray) -> np.ndarray:
    """Same scaling as the SDM notebook (MinMaxScaler), column-wise to [0, 1]."""
    lo, hi = X.min(0), X.max(0)
    return (X - lo) / np.where(hi > lo, hi - lo, 1.0)
