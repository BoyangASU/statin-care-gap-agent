"""Patient-level statin screening on MIMIC-IV (longitudinal EHR view, DRAFT).

This mirrors the Synthea prototype task as closely as MIMIC allows: for each
patient at an index date, use ALL earlier records (every prior admission's
diagnoses and procedures, inpatient and outpatient labs, outpatient blood
pressure from `omr`) to fill the same 14 facts, then apply the prototype's
reference rules (`statin_agent.guideline.classify`: 2018 AHA/ACC + USPSTF 2022,
PCE 10-year risk). Primary and secondary prevention are both covered.

Index date (draft choice): discharge time of each patient's last admission
that did not end in death, hospice or palliative care (patients with no such
admission are left out). At that moment everything earlier in the record is
available, as it would be for a screening pass over an EHR.

What MIMIC cannot see: outpatient prescriptions and outpatient diagnoses.
"on_statin" therefore means a statin was active at the end of the index
admission; a statin prescribed only by an outpatient doctor is invisible
(MIMIC-IV-ED `medrecon` and discharge notes can fill part of this later).

Usage (from the repository root):
    python -m mimic.patient_level --db data/mimic_demo.duckdb --out-dir results_mimic/patient_level
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import duckdb

from mimic import codes as C
from mimic.describe import build_admission_table, _md_table

try:  # reuse the prototype's guideline logic unchanged
    from statin_agent.guideline import Facts, classify, CATEGORIES
except ImportError:  # repository not pip-installed: add src/ to the path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from statin_agent.guideline import Facts, classify, CATEGORIES

LAB = {k: ", ".join(str(i) for i in v) for k, v in C.LAB_ITEMS.items()}
PLAUSIBLE = {"ldl": (10, 600), "total_chol": (50, 1000), "hdl": (5, 200), "sbp": (60, 300)}


def _dx_any(spec) -> str:
    return f"bool_or({C.icd_sql(spec, 'd.icd_code', 'd.icd_version')})"


def build_patient_table(con, lookback_days: int | None) -> None:
    """Create temp table `pt_facts`: one row per patient at the index date."""
    build_admission_table(con)  # gives adm_draft (per-admission statin + reason flags)
    statin = C.STATIN_REGEX.replace("'", "''")

    con.execute("""
    CREATE OR REPLACE TEMP TABLE idx AS
    SELECT * FROM adm_draft
    WHERE NOT (died_in_hospital OR discharged_hospice OR palliative)
    QUALIFY row_number() OVER (PARTITION BY subject_id ORDER BY admittime DESC) = 1""")

    ascvd = " OR ".join(_dx_any(s) for s in C.ASCVD_DX.values())
    harm = " OR ".join(_dx_any(C.JUSTIFIED_DX[k])
                       for k in ("statin_adverse_effect", "drug_myopathy", "rhabdomyolysis"))
    # Diagnoses from every admission that started on or before the index admission.
    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE hist_dx AS
    SELECT i.subject_id,
           {ascvd} AS ascvd_dx_ever,
           {harm} AS statin_harm_dx_ever,
           {_dx_any(C.RISK_DX['diabetes'])} AS diabetes,
           {_dx_any(C.RISK_DX['hypertension'])} AS hypertension,
           {_dx_any(C.RISK_DX['current_smoker'])} AS smoker_code_ever,
           {_dx_any(C.RISK_DX['former_smoker'])} AS former_smoker_code_ever,
           count(DISTINCT a.hadm_id) AS n_admissions_to_index
    FROM idx i JOIN admissions a ON a.subject_id = i.subject_id AND a.admittime <= i.admittime
    JOIN diagnoses_icd d ON d.hadm_id = a.hadm_id
    GROUP BY i.subject_id""")
    # Current smoking = a current-use code on the index admission (latest status).
    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE idx_smoke AS
    SELECT i.subject_id, {_dx_any(C.RISK_DX['current_smoker'])} AS smoker
    FROM idx i JOIN diagnoses_icd d ON d.hadm_id = i.hadm_id GROUP BY i.subject_id""")
    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE hist_px AS
    SELECT i.subject_id,
           bool_or({C.icd_sql(C.REVASC_PROC['cabg'], 'p.icd_code', 'p.icd_version')}
                OR {C.icd_sql(C.REVASC_PROC['pci'], 'p.icd_code', 'p.icd_version')}) AS revasc_ever
    FROM idx i JOIN admissions a ON a.subject_id = i.subject_id AND a.admittime <= i.admittime
    JOIN procedures_icd p ON p.hadm_id = a.hadm_id GROUP BY i.subject_id""")
    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE hist_statin AS
    SELECT i.subject_id, count(DISTINCT p.hadm_id) AS n_adm_with_statin
    FROM idx i JOIN prescriptions p ON p.subject_id = i.subject_id
    JOIN admissions a ON a.hadm_id = p.hadm_id AND a.admittime <= i.admittime
    WHERE regexp_matches(lower(p.drug), '{statin}') GROUP BY i.subject_id""")
    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE idx_bpmed AS
    SELECT i.subject_id, bool_or(regexp_matches(lower(p.drug), '{C.ANTIHYPERTENSIVE_REGEX}')) AS bp_treated
    FROM idx i JOIN prescriptions p ON p.hadm_id = i.hadm_id
    WHERE p.starttime <= i.dischtime
      AND (p.stoptime IS NULL OR p.stoptime >= i.dischtime - INTERVAL 24 HOUR)
    GROUP BY i.subject_id""")

    window = "" if lookback_days is None else \
        f"AND l.charttime >= i.dischtime - INTERVAL {int(lookback_days)} DAY"
    lab_rows = []
    for key in ("ldl", "total_chol", "hdl"):
        lo, hi = PLAUSIBLE[key]
        lab_rows.append(f"""
        SELECT i.subject_id, '{key}' AS k, l.valuenum AS v, l.charttime AS t,
               l.hadm_id IS NULL AS outpatient
        FROM idx i JOIN labevents l ON l.subject_id = i.subject_id
        WHERE l.itemid IN ({LAB[key]}) AND l.valuenum BETWEEN {lo} AND {hi}
          AND l.charttime <= i.dischtime {window}""")
    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE lab_latest AS
    SELECT * FROM ({' UNION ALL '.join(lab_rows)})
    QUALIFY row_number() OVER (PARTITION BY subject_id, k ORDER BY t DESC) = 1""")
    con.execute("""
    CREATE OR REPLACE TEMP TABLE pt_labs AS
    SELECT subject_id,
           max(CASE WHEN k='ldl' THEN v END) AS ldl,
           max(CASE WHEN k='total_chol' THEN v END) AS total_chol,
           max(CASE WHEN k='hdl' THEN v END) AS hdl,
           bool_or(outpatient) AS latest_lipid_outpatient
    FROM lab_latest GROUP BY subject_id""")

    owindow = "" if lookback_days is None else \
        f"AND o.chartdate >= (i.dischtime - INTERVAL {int(lookback_days)} DAY)::DATE"
    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE pt_sbp AS
    SELECT subject_id, sbp FROM (
      SELECT i.subject_id,
             try_cast(split_part(o.result_value, '/', 1) AS DOUBLE) AS sbp,
             row_number() OVER (PARTITION BY i.subject_id ORDER BY o.chartdate DESC, o.seq_num) AS rn
      FROM idx i JOIN omr o ON o.subject_id = i.subject_id
      WHERE o.result_name LIKE 'Blood Pressure%' AND o.chartdate <= i.dischtime::DATE {owindow}
        AND try_cast(split_part(o.result_value, '/', 1) AS DOUBLE) BETWEEN 60 AND 300)
    WHERE rn = 1""")

    con.execute("""
    CREATE OR REPLACE TEMP TABLE pt_facts AS
    SELECT i.subject_id, i.hadm_id AS index_hadm_id, i.dischtime AS index_time,
           i.age_at_admit AS age, CASE WHEN i.gender = 'M' THEN 'male' ELSE 'female' END AS sex,
           CASE WHEN a.race ILIKE '%BLACK%' THEN 'black' ELSE 'white' END AS race,
           i.statin_at_discharge_proxy AS on_statin,
           coalesce(h.statin_harm_dx_ever, FALSE) OR i.liver_disease OR i.pregnancy AS statin_intolerance,
           coalesce(h.ascvd_dx_ever, FALSE) OR coalesce(px.revasc_ever, FALSE) AS ascvd,
           coalesce(h.diabetes, FALSE) AS diabetes,
           coalesce(s.smoker, FALSE) AS smoker,
           coalesce(b.bp_treated, FALSE) AS bp_treated,
           coalesce(h.hypertension, FALSE) AS hypertension,
           l.ldl, l.total_chol, l.hdl, sb.sbp,
           l.latest_lipid_outpatient,
           coalesce(h.n_admissions_to_index, 0) AS n_admissions_to_index,
           coalesce(hs.n_adm_with_statin, 0) AS n_adm_with_statin,
           i.died_in_hospital OR i.discharged_hospice OR i.palliative AS end_of_life
    FROM idx i JOIN admissions a ON a.hadm_id = i.hadm_id
    LEFT JOIN hist_dx h ON h.subject_id = i.subject_id
    LEFT JOIN idx_smoke s ON s.subject_id = i.subject_id
    LEFT JOIN hist_px px ON px.subject_id = i.subject_id
    LEFT JOIN hist_statin hs ON hs.subject_id = i.subject_id
    LEFT JOIN idx_bpmed b ON b.subject_id = i.subject_id
    LEFT JOIN pt_labs l ON l.subject_id = i.subject_id
    LEFT JOIN pt_sbp sb ON sb.subject_id = i.subject_id""")


def classify_patients(con) -> list[dict]:
    cur = con.execute("SELECT * FROM pt_facts ORDER BY subject_id")
    cols = [c[0] for c in cur.description]
    out = []
    for row in cur.fetchall():
        r = dict(zip(cols, row))
        if r["end_of_life"]:
            r["category"], r["pce_risk"] = "excluded_end_of_life", None
        else:
            f = Facts(age=r["age"], sex=r["sex"], race=r["race"], on_statin=bool(r["on_statin"]),
                      statin_intolerance=bool(r["statin_intolerance"]), ascvd=bool(r["ascvd"]),
                      diabetes=bool(r["diabetes"]), smoker=bool(r["smoker"]),
                      bp_treated=bool(r["bp_treated"]), hypertension=bool(r["hypertension"]),
                      ldl=r["ldl"], total_chol=r["total_chol"], hdl=r["hdl"], sbp=r["sbp"])
            r["category"], r["pce_risk"] = classify(f)
        out.append(r)
    return out


def summarize(rows: list[dict], lookback_label: str) -> tuple[str, dict]:
    n = len(rows)
    order = ["excluded_end_of_life"] + CATEGORIES
    counts = {c: sum(r["category"] == c for r in rows) for c in order}
    lines = [f"## Lookback for labs and blood pressure: {lookback_label}\n",
             f"Patients: {n}.\n"]
    lines.append(_md_table([(c, counts[c]) for c in order], ["category", "patients"]))
    # Where gaps and insufficient data come from
    def cnt(pred): return sum(1 for r in rows if pred(r))
    alive = [r for r in rows if r["category"] != "excluded_end_of_life"]
    fact_rows = [
        ("age 40-75 at index", cnt(lambda r: 40 <= (r["age"] or 0) <= 75)),
        ("ASCVD in any admission up to index", cnt(lambda r: r["ascvd"])),
        ("diabetes code", cnt(lambda r: r["diabetes"])),
        ("hypertension code", cnt(lambda r: r["hypertension"])),
        ("current smoking code on index admission", cnt(lambda r: r["smoker"])),
        ("LDL available", cnt(lambda r: r["ldl"] is not None)),
        ("total cholesterol and HDL available", cnt(lambda r: r["total_chol"] is not None and r["hdl"] is not None)),
        ("  latest lipid was an outpatient lab", cnt(lambda r: r["latest_lipid_outpatient"])),
        ("outpatient SBP available (omr)", cnt(lambda r: r["sbp"] is not None)),
        ("all four PCE inputs available", cnt(lambda r: None not in (r["ldl"], r["total_chol"], r["hdl"], r["sbp"]))),
        ("statin active at end of index admission", cnt(lambda r: r["on_statin"])),
        ("statin in any admission up to index", cnt(lambda r: r["n_adm_with_statin"] > 0)),
        ("more than one admission up to index", cnt(lambda r: r["n_admissions_to_index"] > 1)),
    ]
    lines.append("\nFacts available (all patients):\n")
    lines.append(_md_table(fact_rows, ["fact", "patients"]))
    gaps = [r for r in alive if r["category"] == "indicated"]
    why = [("ASCVD (secondary prevention)", sum(r["ascvd"] for r in gaps)),
           ("not ASCVD: diabetes / LDL>=190 / PCE", sum(not r["ascvd"] for r in gaps)),
           ("  of these, statin seen in an earlier admission", sum((not r["ascvd"]) and r["n_adm_with_statin"] > 0 for r in gaps)),
           ("any gap with a statin in an earlier admission", sum(r["n_adm_with_statin"] > 0 for r in gaps))]
    lines.append(f"\n`indicated` (care gap) patients: {len(gaps)}\n")
    lines.append(_md_table(why, ["", "patients"]))
    ins = [r for r in alive if r["category"] == "insufficient_data"]
    miss = [(k, sum(r[k] is None for r in ins)) for k in ("ldl", "total_chol", "hdl", "sbp")]
    lines.append(f"\n`insufficient_data` patients: {len(ins)}; missing input counts:\n")
    lines.append(_md_table(miss, ["input", "missing"]))
    risks = [r["pce_risk"] for r in alive if r["pce_risk"] is not None]
    if risks:
        lines.append(f"\nPCE computed for {len(risks)} patients; median 10-year risk "
                     f"{sorted(risks)[len(risks) // 2]:.1%}.")
    return "\n".join(lines) + "\n", {"n": n, "counts": counts, "facts": dict(fact_rows)}


def main():
    ap = argparse.ArgumentParser(description="Patient-level statin screening stats on MIMIC-IV")
    ap.add_argument("--db", default="data/mimic_demo.duckdb")
    ap.add_argument("--out-dir", default="results_mimic/patient_level")
    ap.add_argument("--lookbacks", default="365,1095,all",
                    help="comma list of lookback windows in days for labs/BP; 'all' = no limit")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(args.db, read_only=True)
    report = ["# MIMIC-IV patient-level statin screening (draft)\n",
              "Reference rules: the Synthea prototype's `statin_agent.guideline.classify`, unchanged. "
              "`statin_intolerance` here = statin-harm ICD code in any admission up to index, or "
              "active liver disease / pregnancy on the index admission. Index = discharge of the last "
              "admission that did not end in death, hospice or palliative care.\n"]
    stats = {}
    for lb in args.lookbacks.split(","):
        days = None if lb.strip() == "all" else int(lb)
        build_patient_table(con, days)
        rows = classify_patients(con)
        label = "no limit" if days is None else f"{days} days"
        text, st = summarize(rows, label)
        report.append(text)
        stats[label] = st
        safe = re.sub(r"\W+", "_", label)
        cols = list(rows[0].keys())
        with open(out / f"patients_{safe}.csv", "w") as fh:
            fh.write(",".join(cols) + "\n")
            for r in rows:
                fh.write(",".join("" if r[c] is None else str(r[c]) for c in cols) + "\n")
    text = "\n".join(report)
    (out / "summary.md").write_text(text)
    (out / "summary.json").write_text(json.dumps(stats, indent=2, default=str))
    print(text)
    print(f"Wrote {out}/summary.md, summary.json, patients_*.csv")


if __name__ == "__main__":
    main()
