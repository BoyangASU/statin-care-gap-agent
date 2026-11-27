"""Patient-scoped, time-limited view of MIMIC-IV, and the reference facts.

`MimicRecord` is what the agent's tools read. It is bound to ONE patient and
ONE index admission: nothing after the index admission's discharge time is
visible (later admissions, later labs, later blood pressure readings).

`reference_facts(rec)` is the correct reading of that record under the Phase-1
rules. It is used to build labels and by the verifier, never by the agent.
It matches `mimic/patient_level.py` (the SQL version used for descriptive
statistics); `mimic/tests/test_agent.py` checks that the two agree.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta

from mimic import codes as C
from mimic.guideline import Facts

PLAUSIBLE = {"ldl": (10, 600), "total_chol": (50, 1000), "hdl": (5, 200), "sbp": (60, 300)}
# Labs the agent can ask for by name (Blood, d_labitems itemids).
LAB_NAMES = {**C.LAB_ITEMS, "a1c": [50852]}
STATIN_ACTIVE_HOURS = 24      # order still active within this many hours of discharge
STATIN_GIVEN_HOURS = 48       # or a dose given within this many hours of discharge

_statin_re = re.compile(C.STATIN_REGEX)
_bp_re = re.compile(C.ANTIHYPERTENSIVE_REGEX)


def _iso(x):
    return x.isoformat(sep=" ", timespec="minutes") if isinstance(x, datetime) else (
        x.isoformat() if hasattr(x, "isoformat") else x)


class MimicRecord:
    """Read-only view of one patient up to the discharge of the index admission."""

    def __init__(self, con, subject_id: int, index_hadm_id: int):
        self.con = con
        self.subject_id = int(subject_id)
        self.index_hadm_id = int(index_hadm_id)
        row = con.execute("SELECT admittime, dischtime FROM admissions WHERE hadm_id = ? AND subject_id = ?",
                          [self.index_hadm_id, self.subject_id]).fetchone()
        if row is None:
            raise ValueError(f"admission {index_hadm_id} not found for patient {subject_id}")
        self.index_admit, self.index_time = row
        self.patient_id = str(self.subject_id)       # name used by the shared pipeline/evaluation

    # ------------------------------------------------------------ helpers
    def _q(self, sql: str, params=()):
        cur = self.con.execute(sql, list(params))
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]

    def _days_before(self, t) -> int:
        if t is None:
            return None
        if not isinstance(t, datetime):
            t = datetime(t.year, t.month, t.day)
        return (self.index_time - t).days

    def _hours_before_discharge(self, t) -> float | None:
        return None if t is None else round((self.index_time - t).total_seconds() / 3600, 1)

    def _visible_admissions_sql(self) -> str:
        return "SELECT hadm_id FROM admissions WHERE subject_id = ? AND admittime <= ?"

    # ------------------------------------------------------------ record views
    def demographics(self) -> dict:
        r = self._q("""SELECT p.gender, p.anchor_age, p.anchor_year, a.race
                       FROM patients p JOIN admissions a ON a.subject_id = p.subject_id
                       WHERE a.hadm_id = ?""", [self.index_hadm_id])[0]
        age = r["anchor_age"] + (self.index_admit.year - r["anchor_year"])   # same as admissions_x
        return {"age": int(age), "sex": "male" if r["gender"] == "M" else "female",
                "race": r["race"], "index_time": _iso(self.index_time),
                "index_admission": self.index_hadm_id,
                "note": "Dates are shifted into the future (21xx) by MIMIC; intervals are real."}

    def admissions(self) -> list[dict]:
        rows = self._q("""SELECT a.hadm_id, a.admittime, a.dischtime, a.admission_type,
                                 a.discharge_location, t.long_title AS principal_diagnosis
                          FROM admissions a
                          LEFT JOIN diagnoses_icd d ON d.hadm_id = a.hadm_id AND d.seq_num = 1
                          LEFT JOIN d_icd_diagnoses t ON t.icd_code = d.icd_code AND t.icd_version = d.icd_version
                          WHERE a.subject_id = ? AND a.admittime <= ? ORDER BY a.admittime""",
                       [self.subject_id, self.index_admit])
        return [{"hadm_id": r["hadm_id"], "admit": _iso(r["admittime"]), "discharge": _iso(r["dischtime"]),
                 "days_before_index": self._days_before(r["dischtime"]),
                 "is_index_admission": r["hadm_id"] == self.index_hadm_id,
                 "type": r["admission_type"], "discharge_location": r["discharge_location"],
                 "principal_diagnosis": r["principal_diagnosis"]} for r in rows]

    def diagnoses(self, keyword: str | None = None) -> list[dict]:
        """One row per distinct ICD code across all visible admissions."""
        sql = f"""SELECT d.icd_code, d.icd_version, t.long_title,
                         min(a.admittime) AS first_seen, max(a.admittime) AS last_seen,
                         count(DISTINCT d.hadm_id) AS n_admissions,
                         bool_or(d.hadm_id = ?) AS on_index_admission
                  FROM diagnoses_icd d JOIN admissions a ON a.hadm_id = d.hadm_id
                  LEFT JOIN d_icd_diagnoses t ON t.icd_code = d.icd_code AND t.icd_version = d.icd_version
                  WHERE d.hadm_id IN ({self._visible_admissions_sql()})"""
        params = [self.index_hadm_id, self.subject_id, self.index_admit]
        if keyword:
            sql += " AND (t.long_title ILIKE ? OR d.icd_code ILIKE ?)"
            params += [f"%{keyword}%", f"{keyword}%"]
        sql += " GROUP BY ALL ORDER BY last_seen DESC, d.icd_code"
        return [{"icd_code": r["icd_code"], "icd_version": r["icd_version"],
                 "title": (r["long_title"] or "")[:80], "last_seen": _iso(r["last_seen"])[:10],
                 "n_admissions": r["n_admissions"], "on_index_admission": bool(r["on_index_admission"])}
                for r in self._q(sql, params)]

    def procedures(self) -> list[dict]:
        rows = self._q(f"""SELECT p.chartdate, p.icd_code, p.icd_version, t.long_title, p.hadm_id
                           FROM procedures_icd p
                           LEFT JOIN d_icd_procedures t ON t.icd_code = p.icd_code AND t.icd_version = p.icd_version
                           WHERE p.hadm_id IN ({self._visible_admissions_sql()})
                           ORDER BY p.chartdate DESC""", [self.subject_id, self.index_admit])
        return [{"date": _iso(r["chartdate"]), "icd_code": r["icd_code"], "icd_version": r["icd_version"],
                 "title": r["long_title"], "hadm_id": r["hadm_id"]} for r in rows]

    def medication_orders(self, keyword: str | None = None, all_admissions: bool = False) -> list[dict]:
        """Prescription orders grouped by drug and admission."""
        where = (f"p.hadm_id IN ({self._visible_admissions_sql()})" if all_admissions else "p.hadm_id = ?")
        params = [self.subject_id, self.index_admit] if all_admissions else [self.index_hadm_id]
        sql = f"""SELECT p.hadm_id, p.drug, min(p.starttime) AS first_start, max(p.stoptime) AS last_stop,
                         count(*) AS n_orders, string_agg(DISTINCT p.prod_strength, '; ') AS strengths
                  FROM prescriptions p WHERE {where} AND p.drug_type = 'MAIN'"""
        if keyword:
            sql += " AND p.drug ILIKE ?"; params.append(f"%{keyword}%")
        sql += " GROUP BY ALL ORDER BY first_start DESC"
        out = []
        for r in self._q(sql, params):
            row = {"hadm_id": r["hadm_id"], "drug": r["drug"],
                   "first_start": _iso(r["first_start"]), "last_stop": _iso(r["last_stop"])}
            if keyword:                                 # strengths only on targeted queries
                row["strengths"] = r["strengths"]
            if r["hadm_id"] == self.index_hadm_id:
                row["last_stop_hours_before_discharge"] = self._hours_before_discharge(r["last_stop"])
            out.append(row)
        return out

    def statin_orders_index(self) -> list[dict]:
        """Raw statin order rows on the index admission (used by the reference and the verifier)."""
        rows = self._q("SELECT drug, starttime, stoptime FROM prescriptions WHERE hadm_id = ?",
                       [self.index_hadm_id])
        return [r for r in rows if _statin_re.search((r["drug"] or "").lower())]

    def administrations(self, keyword: str | None = None) -> list[dict]:
        """eMAR events on the index admission (what nurses actually gave)."""
        sql = "SELECT charttime, medication, event_txt FROM emar WHERE hadm_id = ?"
        params = [self.index_hadm_id]
        if keyword:
            sql += " AND medication ILIKE ?"; params.append(f"%{keyword}%")
        sql += " ORDER BY charttime DESC LIMIT 40"
        return [{"time": _iso(r["charttime"]), "medication": r["medication"], "event": r["event_txt"],
                 "hours_before_discharge": self._hours_before_discharge(r["charttime"])}
                for r in self._q(sql, params)]

    def labs(self, name: str | None = None, keyword: str | None = None, limit: int = 10) -> list[dict]:
        if name:
            ids = LAB_NAMES.get(name)
            if not ids:
                raise ValueError(f"unknown lab name {name!r}; use one of {sorted(LAB_NAMES)} or keyword=")
            cond, params = f"l.itemid IN ({', '.join(map(str, ids))})", []
        elif keyword:
            cond, params = "i.label ILIKE ? AND i.fluid = 'Blood'", [f"%{keyword}%"]
        else:
            raise ValueError("give name= or keyword=")
        rows = self._q(f"""SELECT l.charttime, i.label, l.valuenum, l.value, l.valueuom,
                                  l.ref_range_upper, l.hadm_id
                           FROM labevents l JOIN d_labitems i ON i.itemid = l.itemid
                           WHERE l.subject_id = ? AND l.charttime <= ? AND {cond}
                           ORDER BY l.charttime DESC, l.itemid DESC, l.valuenum LIMIT ?""",
                       [self.subject_id, self.index_time, *params, min(int(limit), 50)])
        return [{"time": _iso(r["charttime"]), "days_before_index": self._days_before(r["charttime"]),
                 "test": r["label"], "value": r["valuenum"] if r["valuenum"] is not None else r["value"],
                 "unit": r["valueuom"], "upper_limit_of_normal": r["ref_range_upper"],
                 "setting": "outpatient" if r["hadm_id"] is None else "inpatient"} for r in rows]

    def blood_pressure(self, limit: int = 10) -> list[dict]:
        rows = self._q("""SELECT chartdate, result_name, result_value FROM omr
                          WHERE subject_id = ? AND result_name LIKE 'Blood Pressure%' AND chartdate <= ?
                          ORDER BY chartdate DESC, seq_num, result_name, result_value LIMIT ?""",
                       [self.subject_id, self.index_time.date(), min(int(limit), 50)])
        return [{"date": _iso(r["chartdate"]), "days_before_index": self._days_before(r["chartdate"]),
                 "measurement": r["result_name"], "value": r["result_value"]} for r in rows]


# ---------------------------------------------------------------- reference facts
def _dx_any(rec: MimicRecord, spec, index_only=False) -> list[str]:
    where = "d.hadm_id = ?" if index_only else f"d.hadm_id IN ({rec._visible_admissions_sql()})"
    params = [rec.index_hadm_id] if index_only else [rec.subject_id, rec.index_admit]
    rows = rec._q(f"SELECT DISTINCT icd_code FROM diagnoses_icd d WHERE {where} AND {C.icd_sql(spec, 'd.icd_code', 'd.icd_version')}",
                  params)
    return [r["icd_code"] for r in rows]


def _px_any(rec: MimicRecord, spec) -> list[str]:
    rows = rec._q(f"SELECT DISTINCT icd_code FROM procedures_icd p WHERE p.hadm_id IN ({rec._visible_admissions_sql()}) "
                  f"AND {C.icd_sql(spec, 'p.icd_code', 'p.icd_version')}", [rec.subject_id, rec.index_admit])
    return [r["icd_code"] for r in rows]


def _active_at_discharge(rec: MimicRecord, regex: re.Pattern) -> bool:
    rows = rec._q("SELECT drug, starttime, stoptime FROM prescriptions WHERE hadm_id = ?", [rec.index_hadm_id])
    cutoff = rec.index_time - timedelta(hours=STATIN_ACTIVE_HOURS)
    return any(regex.search((r["drug"] or "").lower()) and r["starttime"] is not None
               and r["starttime"] <= rec.index_time and (r["stoptime"] is None or r["stoptime"] >= cutoff)
               for r in rows)


def reference_on_statin(rec: MimicRecord) -> bool:
    if _active_at_discharge(rec, _statin_re):
        return True
    cutoff = rec.index_time - timedelta(hours=STATIN_GIVEN_HOURS)
    rows = rec._q("SELECT medication, event_txt, charttime FROM emar WHERE hadm_id = ?", [rec.index_hadm_id])
    return any(_statin_re.search((r["medication"] or "").lower()) and r["event_txt"] in C.EMAR_GIVEN
               and r["charttime"] is not None and r["charttime"] >= cutoff for r in rows)


def latest_valid(rec: MimicRecord, key: str, lookback_days: int | None = 365):
    lo, hi = PLAUSIBLE[key]
    if key == "sbp":
        for r in rec.blood_pressure(limit=50):
            if lookback_days is not None and r["days_before_index"] > lookback_days:
                return None
            try:
                v = float(str(r["value"]).split("/")[0])
            except ValueError:
                continue
            if lo <= v <= hi:
                return v
        return None
    for r in rec.labs(name=key, limit=50):
        if lookback_days is not None and r["days_before_index"] > lookback_days:
            return None
        v = r["value"]
        if isinstance(v, (int, float)) and lo <= v <= hi:
            return float(v)
    return None


def reference_evidence(rec: MimicRecord) -> dict:
    """Codes behind each coded fact (the reference reading)."""
    harm = sum((_dx_any(rec, C.JUSTIFIED_DX[k]) for k in
                ("statin_adverse_effect", "drug_myopathy", "rhabdomyolysis")), [])
    return {
        "ascvd": sum((_dx_any(rec, s) for s in C.ASCVD_DX.values()), []) +
                 sum((_px_any(rec, s) for s in C.REVASC_PROC.values()), []),
        "diabetes": _dx_any(rec, C.RISK_DX["diabetes"]),
        "hypertension": _dx_any(rec, C.RISK_DX["hypertension"]),
        "smoker": _dx_any(rec, C.RISK_DX["current_smoker"], index_only=True),
        "statin_intolerance": harm + _dx_any(rec, C.JUSTIFIED_DX["liver_disease"], index_only=True)
                              + _dx_any(rec, C.JUSTIFIED_DX["pregnancy"], index_only=True),
    }


def reference_facts(rec: MimicRecord, lookback_days: int | None = 365) -> Facts:
    demo = rec.demographics()
    ev = reference_evidence(rec)
    return Facts(
        age=demo["age"], sex=demo["sex"],
        race="black" if "BLACK" in (demo["race"] or "").upper() else "white",
        on_statin=reference_on_statin(rec),
        statin_intolerance=bool(ev["statin_intolerance"]),
        ascvd=bool(ev["ascvd"]), diabetes=bool(ev["diabetes"]), smoker=bool(ev["smoker"]),
        bp_treated=_active_at_discharge(rec, _bp_re), hypertension=bool(ev["hypertension"]),
        ldl=latest_valid(rec, "ldl", lookback_days), total_chol=latest_valid(rec, "total_chol", lookback_days),
        hdl=latest_valid(rec, "hdl", lookback_days), sbp=latest_valid(rec, "sbp", lookback_days),
    )


def index_admissions(con, min_age: int = 40, max_age: int = 80) -> list[tuple[int, int]]:
    """(subject_id, index hadm_id): each patient's last admission that did not end in
    death, hospice or a palliative-care code, with age at index in [min_age, max_age]."""
    pal = C.icd_sql(C.JUSTIFIED_DX["palliative"], "d.icd_code", "d.icd_version")
    rows = con.execute(f"""
        SELECT subject_id, hadm_id FROM (
          SELECT a.subject_id, a.hadm_id, p.anchor_age + (year(a.admittime) - p.anchor_year) AS age
          FROM admissions a JOIN patients p USING (subject_id)
          WHERE coalesce(a.hospital_expire_flag, 0) = 0
            AND coalesce(a.discharge_location, '') NOT IN ('DIED', 'HOSPICE')
            AND a.hadm_id NOT IN (SELECT d.hadm_id FROM diagnoses_icd d WHERE {pal})
          QUALIFY row_number() OVER (PARTITION BY a.subject_id ORDER BY a.admittime DESC) = 1)
        WHERE age BETWEEN ? AND ? ORDER BY subject_id""", [min_age, max_age]).fetchall()
    return [(int(s), int(h)) for s, h in rows]
