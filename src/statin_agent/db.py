"""DuckDB data layer over Synthea CSV exports.

`build_database` loads the four tables the agent needs (patients, conditions,
medications, observations, allergies) with typed dates and indexes.
`PatientRecord` gives read-only, patient-scoped SQL access; every query is
restricted to one patient so an agent can never read another patient's data.
"""
from __future__ import annotations
from datetime import date
from pathlib import Path
import duckdb

TABLES = {
    "patients": "SELECT Id AS patient_id, CAST(BIRTHDATE AS DATE) AS birthdate, "
                "CAST(DEATHDATE AS DATE) AS deathdate, GENDER AS gender, RACE AS race "
                "FROM read_csv_auto('{d}/patients.csv')",
    "conditions": "SELECT PATIENT AS patient_id, CAST(\"START\" AS DATE) AS start, "
                  "CAST(STOP AS DATE) AS stop, CAST(CODE AS BIGINT) AS code, DESCRIPTION AS description "
                  "FROM read_csv_auto('{d}/conditions.csv')",
    "medications": "SELECT PATIENT AS patient_id, CAST(\"START\" AS DATE) AS start, "
                   "CAST(STOP AS DATE) AS stop, CAST(CODE AS VARCHAR) AS code, DESCRIPTION AS description "
                   "FROM read_csv_auto('{d}/medications.csv')",
    "observations": "SELECT PATIENT AS patient_id, CAST(\"DATE\" AS DATE) AS date, "
                    "CAST(CODE AS VARCHAR) AS loinc, DESCRIPTION AS description, "
                    "CAST(VALUE AS VARCHAR) AS value, UNITS AS units, TYPE AS type "
                    "FROM read_csv_auto('{d}/observations.csv', all_varchar=true)",
    "allergies": "SELECT PATIENT AS patient_id, CAST(\"START\" AS DATE) AS start, "
                 "CAST(STOP AS DATE) AS stop, DESCRIPTION AS description "
                 "FROM read_csv_auto('{d}/allergies.csv')",
}


def build_database(csv_dir: str, db_path: str) -> dict:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(db_path)
    counts = {}
    for name, sql in TABLES.items():
        con.execute(f"CREATE OR REPLACE TABLE {name} AS {sql.format(d=csv_dir)}")
        counts[name] = con.execute(f"SELECT count(*) FROM {name}").fetchone()[0]
    for t in ("conditions", "medications", "observations", "allergies"):
        con.execute(f"CREATE INDEX IF NOT EXISTS idx_{t} ON {t}(patient_id)")
    max_date = con.execute("SELECT max(date) FROM observations").fetchone()[0]
    con.execute("CREATE OR REPLACE TABLE meta AS SELECT ?::DATE AS data_end_date", [max_date])
    con.close()
    counts["data_end_date"] = str(max_date)
    return counts


def connect(db_path: str) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(db_path, read_only=True)


def data_end_date(con) -> date:
    return con.execute("SELECT data_end_date FROM meta").fetchone()[0]


class PatientRecord:
    """Read-only, patient-scoped view of the database."""

    def __init__(self, con, patient_id: str, index_date: date):
        self.con, self.patient_id, self.index_date = con, patient_id, index_date

    def _q(self, sql: str, *params):
        cur = self.con.execute(sql, [self.patient_id, *params])
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def demographics(self):
        r = self._q("SELECT birthdate, deathdate, gender, race FROM patients WHERE patient_id = ?")[0]
        b = r["birthdate"]; i = self.index_date
        age = i.year - b.year - ((i.month, i.day) < (b.month, b.day))
        return {"age": age, "sex": "male" if r["gender"] == "M" else "female",
                "race": r["race"], "index_date": i.isoformat()}

    def conditions(self):
        return self._q("SELECT code, description, start, stop FROM conditions "
                       "WHERE patient_id = ? AND start <= ? ORDER BY start", self.index_date)

    def medications(self):
        return self._q("SELECT code, description, start, stop FROM medications "
                       "WHERE patient_id = ? AND start <= ? ORDER BY start", self.index_date)

    def allergies(self):
        return self._q("SELECT description, start, stop FROM allergies "
                       "WHERE patient_id = ? AND start <= ? ORDER BY start", self.index_date)

    def observations(self, loinc: str | None = None, keyword: str | None = None, limit: int = 20):
        sql = ("SELECT date, loinc, description, value, units FROM observations "
               "WHERE patient_id = ? AND date <= ?")
        params = [self.index_date]
        if loinc:
            sql += " AND loinc = ?"; params.append(loinc)
        if keyword:
            sql += " AND description ILIKE ?"; params.append(f"%{keyword}%")
        sql += " ORDER BY date DESC LIMIT ?"; params.append(limit)
        return self._q(sql, *params)

    def days_before_index(self, d: date) -> int:
        return (self.index_date - d).days


def cohort(con, index_date: date, min_age=40, max_age=80, alive_only=True) -> list[str]:
    sql = ("SELECT patient_id FROM patients WHERE "
           "date_diff('year', birthdate, ?::DATE) BETWEEN ? AND ?")
    params = [index_date, min_age, max_age]
    if alive_only:
        sql += " AND (deathdate IS NULL OR deathdate > ?::DATE)"; params.append(index_date)
    return [r[0] for r in con.execute(sql + " ORDER BY patient_id", params).fetchall()]
