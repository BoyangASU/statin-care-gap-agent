"""Descriptive statistics for the MIMIC-IV statin task (Phase 1 design input).

Builds one row per admission with ASCVD flags, statin exposure, and candidate
"justified no statin" signals, then prints a summary in Markdown. Nothing here
is a final label: the draft category at the end only shows how the proposed
categories would split the cohort, so the category design can be discussed.

Only aggregate counts are printed. The per-admission table is written to
--out-dir (keep it local for credentialed MIMIC-IV; the Demo is ODbL).

Usage (from the repository root):
    python -m mimic.describe --db data/mimic_demo.duckdb --out-dir results_mimic/describe
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb

from mimic import codes as C


def _flag_cols(groups: dict, table_alias: str = "d") -> str:
    return ",\n       ".join(
        f"bool_or({C.icd_sql(spec, f'{table_alias}.icd_code', f'{table_alias}.icd_version')}) AS {name}"
        for name, spec in groups.items())


def _principal_cols(groups: dict, which: tuple, table_alias: str = "d") -> str:
    return ",\n       ".join(
        f"bool_or(d.seq_num = 1 AND {C.icd_sql(groups[name], f'{table_alias}.icd_code', f'{table_alias}.icd_version')}) AS pdx_{name}"
        for name in which)


def _lab_ratio(name: str) -> str:
    ids = ", ".join(str(i) for i in C.LAB_ITEMS[name])
    uln = C.DEFAULT_ULN[name]
    return (f"max(CASE WHEN l.itemid IN ({ids}) "
            f"THEN l.valuenum / coalesce(nullif(l.ref_range_upper, 0), {uln}) END) AS {name}_max_xuln")


def build_admission_table(con) -> None:
    """Create temp table `adm_features`: one row per admission."""
    given = ", ".join(f"'{e}'" for e in C.EMAR_GIVEN)
    statin = C.STATIN_REGEX.replace("'", "''")
    nonstatin = C.NONSTATIN_LIPID_REGEX

    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE dx_flags AS
    SELECT d.hadm_id,
       {_flag_cols({**C.ASCVD_DX, **C.JUSTIFIED_DX})},
       {_principal_cols(C.ASCVD_DX, C.INDEX_EVENT_GROUPS)}
    FROM diagnoses_icd d GROUP BY d.hadm_id""")

    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE proc_flags AS
    SELECT d.hadm_id, {_flag_cols(C.REVASC_PROC)}
    FROM procedures_icd d GROUP BY d.hadm_id""")

    # Statin orders (prescriptions) and administrations (eMAR), per admission.
    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE rx AS
    SELECT p.hadm_id,
           count(*) AS n_rx_rows,
           bool_or(regexp_matches(lower(p.drug), '{statin}')) AS rx_statin,
           bool_or(regexp_matches(lower(p.drug), '{statin}')
                   AND p.starttime <= a.dischtime
                   AND (p.stoptime IS NULL OR p.stoptime >= a.dischtime - INTERVAL 24 HOUR))
               AS rx_statin_active_at_disch,
           bool_or(regexp_matches(lower(p.drug), '{nonstatin}')) AS rx_nonstatin_lipid,
           string_agg(DISTINCT CASE WHEN regexp_matches(lower(p.drug), '{statin}')
                                    THEN lower(p.drug) END, '; ') AS statin_names
    FROM prescriptions p JOIN admissions a USING (hadm_id)
    GROUP BY p.hadm_id""")

    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE em AS
    SELECT e.hadm_id,
           count(*) AS n_emar_rows,
           bool_or(regexp_matches(lower(e.medication), '{statin}') AND e.event_txt IN ({given}))
               AS emar_statin_given,
           bool_or(regexp_matches(lower(e.medication), '{statin}') AND e.event_txt IN ({given})
                   AND e.charttime >= a.dischtime - INTERVAL 48 HOUR)
               AS emar_statin_given_last48h,
           bool_or(regexp_matches(lower(e.medication), '{statin}')
                   AND e.event_txt IN ('Not Given', 'Hold Dose')) AS emar_statin_not_given
    FROM emar e JOIN admissions a USING (hadm_id)
    WHERE e.hadm_id IS NOT NULL
    GROUP BY e.hadm_id""")

    # Labs by subject and time window (outpatient labs have NULL hadm_id).
    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE labs AS
    SELECT a.hadm_id,
           {_lab_ratio('alt')}, {_lab_ratio('ast')}, {_lab_ratio('ck')}
    FROM admissions a JOIN labevents l
      ON l.subject_id = a.subject_id AND l.charttime BETWEEN a.admittime AND a.dischtime
    WHERE l.itemid IN ({', '.join(str(i) for k in ('alt', 'ast', 'ck') for i in C.LAB_ITEMS[k])})
    GROUP BY a.hadm_id""")

    ldl_ids = ", ".join(str(i) for i in C.LAB_ITEMS["ldl"])
    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE ldl AS
    SELECT hadm_id, ldl_last, ldl_days_before_disch FROM (
      SELECT a.hadm_id, l.valuenum AS ldl_last,
             date_diff('day', l.charttime, a.dischtime) AS ldl_days_before_disch,
             row_number() OVER (PARTITION BY a.hadm_id ORDER BY l.charttime DESC) AS rn
      FROM admissions a JOIN labevents l
        ON l.subject_id = a.subject_id
       AND l.charttime BETWEEN a.dischtime - INTERVAL 365 DAY AND a.dischtime
      WHERE l.itemid IN ({ldl_ids}) AND l.valuenum IS NOT NULL) WHERE rn = 1""")

    # Statin or statin-harm evidence in any EARLIER admission of the same patient.
    harm = " OR ".join(C.icd_sql(C.JUSTIFIED_DX[k], "d.icd_code", "d.icd_version")
                       for k in ("statin_adverse_effect", "drug_myopathy", "rhabdomyolysis"))
    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE prior AS
    SELECT a.hadm_id,
           bool_or(r.rx_statin OR coalesce(e.emar_statin_given, FALSE)) AS prior_statin,
           count(DISTINCT b.hadm_id) AS n_prior_admissions
    FROM admissions a JOIN admissions b
      ON b.subject_id = a.subject_id AND b.dischtime <= a.admittime
    LEFT JOIN rx r ON r.hadm_id = b.hadm_id
    LEFT JOIN em e ON e.hadm_id = b.hadm_id
    GROUP BY a.hadm_id""")
    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE prior_harm AS
    SELECT a.hadm_id, bool_or({harm}) AS prior_statin_harm_dx
    FROM admissions a JOIN admissions b
      ON b.subject_id = a.subject_id AND b.dischtime <= a.admittime
    JOIN diagnoses_icd d ON d.hadm_id = b.hadm_id
    GROUP BY a.hadm_id""")

    ascvd_any = " OR ".join(f"coalesce(x.{g}, FALSE)" for g in C.ASCVD_DX)
    pdx_any = " OR ".join(f"coalesce(x.pdx_{g}, FALSE)" for g in C.INDEX_EVENT_GROUPS)
    revasc = "coalesce(pr.cabg, FALSE) OR coalesce(pr.pci, FALSE)"
    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE adm_features AS
    SELECT a.subject_id, a.hadm_id, a.admittime, a.dischtime, a.admission_type,
           a.discharge_location, a.hospital_expire_flag, a.gender, a.age_at_admit, a.los_days,
           x.* EXCLUDE (hadm_id), pr.* EXCLUDE (hadm_id),
           ({pdx_any}) OR ({revasc}) AS index_ascvd_admission,
           ({ascvd_any}) OR ({revasc}) AS any_ascvd_admission,
           coalesce(r.n_rx_rows, 0) AS n_rx_rows, coalesce(e.n_emar_rows, 0) AS n_emar_rows,
           coalesce(r.n_rx_rows, 0) + coalesce(e.n_emar_rows, 0) > 0 AS has_med_data,
           coalesce(r.rx_statin, FALSE) AS rx_statin,
           coalesce(r.rx_statin_active_at_disch, FALSE) AS rx_statin_active_at_disch,
           coalesce(e.emar_statin_given, FALSE) AS emar_statin_given,
           coalesce(e.emar_statin_given_last48h, FALSE) AS emar_statin_given_last48h,
           coalesce(e.emar_statin_not_given, FALSE) AS emar_statin_not_given,
           coalesce(r.rx_statin, FALSE) OR coalesce(e.emar_statin_given, FALSE) AS statin_any_in_stay,
           coalesce(r.rx_statin_active_at_disch, FALSE) OR coalesce(e.emar_statin_given_last48h, FALSE)
               AS statin_at_discharge_proxy,
           coalesce(r.rx_nonstatin_lipid, FALSE) AS nonstatin_lipid_rx,
           r.statin_names,
           lb.alt_max_xuln, lb.ast_max_xuln, lb.ck_max_xuln,
           ld.ldl_last, ld.ldl_days_before_disch,
           coalesce(p.prior_statin, FALSE) AS prior_statin,
           coalesce(p.n_prior_admissions, 0) AS n_prior_admissions,
           coalesce(ph.prior_statin_harm_dx, FALSE) AS prior_statin_harm_dx,
           -- coalesce: discharge_location is NULL for many short stays
           coalesce(a.hospital_expire_flag = 1 OR a.discharge_location = 'DIED', FALSE) AS died_in_hospital,
           coalesce(a.discharge_location = 'HOSPICE', FALSE) AS discharged_hospice,
           coalesce(a.discharge_location = 'AGAINST ADVICE', FALSE) AS left_ama,
           coalesce(a.discharge_location = 'ACUTE HOSPITAL', FALSE) AS transferred_acute
    FROM admissions_x a
    LEFT JOIN dx_flags x USING (hadm_id) LEFT JOIN proc_flags pr USING (hadm_id)
    LEFT JOIN rx r USING (hadm_id) LEFT JOIN em e USING (hadm_id)
    LEFT JOIN labs lb USING (hadm_id) LEFT JOIN ldl ld USING (hadm_id)
    LEFT JOIN prior p USING (hadm_id) LEFT JOIN prior_harm ph USING (hadm_id)""")

    # Fill NULL diagnosis/procedure flags (admissions with no codes) with FALSE.
    bool_cols = list(C.ASCVD_DX) + list(C.JUSTIFIED_DX) + list(C.REVASC_PROC) + \
        [f"pdx_{g}" for g in C.INDEX_EVENT_GROUPS]
    for col in bool_cols:
        con.execute(f"UPDATE adm_features SET {col} = FALSE WHERE {col} IS NULL")

    # Candidate reasons (draft). Each is a separate column so the list can be edited.
    con.execute("""
    CREATE OR REPLACE TEMP TABLE adm_reasons AS
    SELECT *,
      -- Tier A: clinical reasons documented in this stay
      statin_adverse_effect OR drug_myopathy OR rhabdomyolysis       AS r_statin_harm_dx,
      prior_statin_harm_dx                                           AS r_prior_statin_harm_dx,
      coalesce(ck_max_xuln, 0) > 10                                  AS r_ck_gt10x,
      coalesce(alt_max_xuln, 0) > 3 OR coalesce(ast_max_xuln, 0) > 3 AS r_transaminase_gt3x,
      liver_disease                                                  AS r_liver_disease,
      pregnancy                                                      AS r_pregnancy,
      -- Tier B: care-process reasons (quality-measure style exclusions)
      died_in_hospital                                               AS r_died,
      discharged_hospice OR palliative                               AS r_hospice_palliative,
      left_ama                                                       AS r_left_ama,
      transferred_acute                                              AS r_transferred_acute,
      -- Tier C: debatable, report only
      esrd_dialysis                                                  AS c_esrd_dialysis,
      coalesce(age_at_admit, 0) > 75                                 AS c_age_gt75,
      ldl_last IS NOT NULL AND ldl_last < 70                         AS c_ldl_lt70,
      nonstatin_lipid_rx                                             AS c_nonstatin_lipid_only,
      prior_statin AND nonstatin_lipid_rx AND NOT statin_any_in_stay AS c_switched_to_nonstatin,
      myalgia                                                        AS c_myalgia_dx,
      dnr                                                            AS c_dnr
    FROM adm_features""")
    con.execute("""
    CREATE OR REPLACE TEMP TABLE adm_draft AS
    SELECT *,
      (r_statin_harm_dx OR r_prior_statin_harm_dx OR r_ck_gt10x OR r_transaminase_gt3x
       OR r_liver_disease OR r_pregnancy OR r_died OR r_hospice_palliative
       OR r_left_ama OR r_transferred_acute) AS any_justified_reason,
      CASE
        WHEN NOT has_med_data THEN 'insufficient_data'
        WHEN statin_at_discharge_proxy THEN 'on_statin'
        WHEN (r_statin_harm_dx OR r_prior_statin_harm_dx OR r_ck_gt10x OR r_transaminase_gt3x
              OR r_liver_disease OR r_pregnancy OR r_died OR r_hospice_palliative
              OR r_left_ama OR r_transferred_acute) THEN 'justified_no_statin'
        ELSE 'gap'
      END AS draft_category
    FROM adm_reasons""")


def _md_table(rows, headers) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join("" if v is None else str(v) for v in r) + " |" for r in rows]
    return "\n".join(out)


def summarize(con) -> tuple[str, dict]:
    q = lambda s: con.execute(s).fetchall()
    one = lambda s: con.execute(s).fetchone()
    lines, stats = [], {}

    n_pat, n_adm = one("SELECT count(DISTINCT subject_id), count(*) FROM adm_draft")
    lines.append(f"# MIMIC-IV statin task: descriptive statistics\n\nPatients: {n_pat}; admissions: {n_adm}.\n")
    stats["n_patients"], stats["n_admissions"] = n_pat, n_adm

    # Data coverage
    cov = one("""SELECT sum(n_rx_rows > 0::INT), sum(n_emar_rows > 0::INT), sum(has_med_data::INT),
                        sum(icd_any::INT)
                 FROM (SELECT *, hadm_id IN (SELECT hadm_id FROM diagnoses_icd) AS icd_any FROM adm_draft)""")
    lines.append("## Data coverage (all admissions)\n")
    lines.append(_md_table([("with prescriptions rows", cov[0]), ("with eMAR rows", cov[1]),
                            ("with either (has_med_data)", cov[2]), ("with any ICD diagnosis", cov[3])],
                           ["", "admissions"]))
    stats["coverage"] = dict(zip(["rx", "emar", "either", "icd"], cov))

    # Cohort definitions
    lines.append("\n## ASCVD cohort\n")
    lines.append("`index` = principal diagnosis (seq_num 1) is acute MI / CHD / ischemic stroke / TIA, "
                 "or CABG/PCI during the stay. `any` = any ASCVD code in any position "
                 "(including old MI, PAD, revascularization status) or CABG/PCI.\n")
    coh = q("""SELECT 'index', count(*), count(DISTINCT subject_id) FROM adm_draft WHERE index_ascvd_admission
               UNION ALL
               SELECT 'any', count(*), count(DISTINCT subject_id) FROM adm_draft WHERE any_ascvd_admission""")
    lines.append(_md_table(coh, ["cohort", "admissions", "patients"]))
    stats["cohort"] = {r[0]: {"admissions": r[1], "patients": r[2]} for r in coh}

    groups = list(C.ASCVD_DX) + list(C.REVASC_PROC)
    rows = []
    for g in groups:
        pdx = f"sum(pdx_{g}::INT)" if g in C.INDEX_EVENT_GROUPS else "NULL"
        r = one(f"SELECT sum({g}::INT), {pdx} FROM adm_draft")
        rows.append((g, r[0], r[1] if r[1] is not None else "-"))
    lines.append("\nBy component (admissions; a stay can have several):\n")
    lines.append(_md_table(rows, ["component", "any position", "principal dx"]))
    ver = q("""SELECT d.icd_version, count(DISTINCT d.hadm_id) FROM diagnoses_icd d
               JOIN adm_draft a USING (hadm_id) WHERE a.any_ascvd_admission GROUP BY 1 ORDER BY 1""")
    lines.append("\nICD version among `any` ASCVD admissions: " +
                 ", ".join(f"ICD-{v}: {n}" for v, n in ver))

    # Statin exposure by cohort
    for coh_name, where in (("index", "index_ascvd_admission"), ("any", "any_ascvd_admission")):
        r = one(f"""SELECT count(*), sum(has_med_data::INT), sum(statin_any_in_stay::INT),
                           sum(rx_statin::INT), sum(emar_statin_given::INT),
                           sum(statin_at_discharge_proxy::INT),
                           sum((statin_any_in_stay AND NOT statin_at_discharge_proxy)::INT),
                           sum(prior_statin::INT), sum(nonstatin_lipid_rx::INT)
                    FROM adm_draft WHERE {where}""")
        lines.append(f"\n## Statin exposure, `{coh_name}` cohort (n = {r[0]})\n")
        lines.append(_md_table([
            ("has any medication data", r[1]),
            ("statin ordered or given during stay", r[2]),
            ("  ordered (prescriptions)", r[3]),
            ("  given (eMAR, administered)", r[4]),
            ("statin at discharge (proxy: order active in last 24 h or dose in last 48 h)", r[5]),
            ("statin during stay but not near discharge", r[6]),
            ("statin in an earlier admission", r[7]),
            ("non-statin lipid therapy ordered (ezetimibe, PCSK9i, ...)", r[8]),
        ], ["", "admissions"]))
        stats[f"statin_{coh_name}"] = dict(zip(
            ["n", "has_med_data", "any_in_stay", "rx", "emar", "at_discharge_proxy",
             "stopped_before_discharge", "prior", "nonstatin"], r))

    names = q("""SELECT lower(drug), count(*) FROM prescriptions
                 WHERE regexp_matches(lower(drug), 'statin') GROUP BY 1 ORDER BY 2 DESC""")
    lines.append("\nDrug names containing 'statin' in prescriptions (all admissions) and whether the matcher accepts them:\n")
    lines.append(_md_table([(n, c, C.is_statin(n)) for n, c in names], ["drug", "rows", "is_statin"]))

    # Candidate justified reasons among ASCVD admissions WITHOUT statin at discharge
    reason_cols = ["r_statin_harm_dx", "r_prior_statin_harm_dx", "r_ck_gt10x", "r_transaminase_gt3x",
                   "r_liver_disease", "r_pregnancy", "r_died", "r_hospice_palliative",
                   "r_left_ama", "r_transferred_acute",
                   "c_esrd_dialysis", "c_age_gt75", "c_ldl_lt70", "c_nonstatin_lipid_only",
                   "c_switched_to_nonstatin", "c_myalgia_dx", "c_dnr"]
    for coh_name, where in (("index", "index_ascvd_admission"), ("any", "any_ascvd_admission")):
        base = f"{where} AND has_med_data AND NOT statin_at_discharge_proxy"
        n = one(f"SELECT count(*) FROM adm_draft WHERE {base}")[0]
        rows = [(c, one(f"SELECT sum({c}::INT) FROM adm_draft WHERE {base}")[0] or 0) for c in reason_cols]
        lines.append(f"\n## Candidate reasons, `{coh_name}` cohort, no statin at discharge (n = {n})\n")
        lines.append("`r_` = proposed justified reason; `c_` = debatable, reported only.\n")
        lines.append(_md_table(rows, ["signal", "admissions"]))
        stats[f"reasons_{coh_name}"] = {"n": n, **dict(rows)}

    # Same signals among those ON statin (how often would the rules fire anyway?)
    base = "any_ascvd_admission AND statin_at_discharge_proxy"
    n = one(f"SELECT count(*) FROM adm_draft WHERE {base}")[0]
    rows = [(c, one(f"SELECT sum({c}::INT) FROM adm_draft WHERE {base}")[0] or 0) for c in reason_cols]
    lines.append(f"\n## Same signals among `any` ASCVD admissions ON statin at discharge (n = {n})\n")
    lines.append(_md_table(rows, ["signal", "admissions"]))

    # Lab availability
    r = one("""SELECT count(*), sum((alt_max_xuln IS NOT NULL)::INT), sum((ck_max_xuln IS NOT NULL)::INT),
                      sum((ldl_last IS NOT NULL)::INT) FROM adm_draft WHERE any_ascvd_admission""")
    lines.append(f"\n## Lab availability, `any` ASCVD admissions (n = {r[0]})\n")
    lines.append(_md_table([("ALT measured during stay", r[1]), ("CK measured during stay", r[2]),
                            ("LDL within 365 days before discharge", r[3])], ["", "admissions"]))

    # Draft category split
    for coh_name, where in (("index", "index_ascvd_admission"), ("any", "any_ascvd_admission")):
        rows = q(f"""SELECT draft_category, count(*), count(DISTINCT subject_id) FROM adm_draft
                     WHERE {where} GROUP BY 1 ORDER BY 2 DESC""")
        lines.append(f"\n## Draft category split, `{coh_name}` cohort (proposal only)\n")
        lines.append(_md_table(rows, ["draft_category", "admissions", "patients"]))
        stats[f"draft_{coh_name}"] = {r[0]: r[1] for r in rows}

    # Patient-level view: one (latest) ASCVD admission per patient avoids letting
    # patients with many stays dominate the counts.
    rows = q("""SELECT draft_category, count(*) FROM (
                  SELECT * FROM adm_draft WHERE any_ascvd_admission
                  QUALIFY row_number() OVER (PARTITION BY subject_id ORDER BY admittime DESC) = 1)
                GROUP BY 1 ORDER BY 2 DESC""")
    lines.append("\n## Draft category split, `any` cohort, latest admission per patient\n")
    lines.append(_md_table(rows, ["draft_category", "patients"]))
    stats["draft_any_latest_per_patient"] = {r[0]: r[1] for r in rows}

    # Other descriptors
    r = one("""SELECT round(median(age_at_admit),1), round(median(los_days),1),
                      sum((gender='F')::INT), count(*) FROM adm_draft WHERE any_ascvd_admission""")
    lines.append(f"\n`any` ASCVD admissions: median age {r[0]}, median LOS {r[1]} days, "
                 f"{r[2]}/{r[3]} female.")
    dl = q("""SELECT coalesce(discharge_location, '(missing)'), count(*) FROM adm_draft
              WHERE any_ascvd_admission GROUP BY 1 ORDER BY 2 DESC""")
    lines.append("\nDischarge location, `any` ASCVD admissions:\n")
    lines.append(_md_table(dl, ["discharge_location", "admissions"]))
    return "\n".join(lines) + "\n", stats


def main():
    ap = argparse.ArgumentParser(description="Descriptive stats for the MIMIC-IV statin task")
    ap.add_argument("--db", default="data/mimic_demo.duckdb")
    ap.add_argument("--out-dir", default="results_mimic/describe")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(args.db, read_only=True)
    build_admission_table(con)
    report, stats = summarize(con)
    (out / "summary.md").write_text(report)
    (out / "summary.json").write_text(json.dumps(stats, indent=2, default=str))
    con.execute(f"COPY (SELECT * FROM adm_draft ORDER BY subject_id, admittime) "
                f"TO '{(out / 'admissions_features.csv').as_posix()}' (HEADER)")
    print(report)
    print(f"Wrote {out}/summary.md, summary.json, admissions_features.csv")


if __name__ == "__main__":
    main()
