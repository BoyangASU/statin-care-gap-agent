"""Unit tests for the MIMIC code sets (no data needed). Run: python -m pytest mimic/tests"""
import duckdb

from mimic import codes as C


def test_statin_matcher_variants():
    for name in ["Atorvastatin", "atorvaSTATin", "Rosuvastatin Calcium", "Lipitor",
                 "Simvastatin 20mg Tab", "Ezetimibe-Simvastatin", "pravastatin"]:
        assert C.is_statin(name), name
    for name in ["Nystatin Oral Suspension", "nystatin-triamcinolone cream", "Nystatin",
                 "Ezetimibe", "", None]:
        assert not C.is_statin(name), name


def test_icd_match_prefix_and_exclusion():
    assert C.icd_match("I214", 10, C.ASCVD_DX["acute_mi"])
    assert C.icd_match("41071", 9, C.ASCVD_DX["acute_mi"])
    assert not C.icd_match("I214", 9, C.ASCVD_DX["acute_mi"])        # version matters
    assert C.icd_match("I2510", 10, C.ASCVD_DX["chd"])
    assert not C.icd_match("I252", 10, C.ASCVD_DX["chd"])             # old MI is its own group
    assert not C.icd_match("41412", 9, C.ASCVD_DX["chd"])             # coronary dissection excluded
    assert C.icd_match("43491", 9, C.ASCVD_DX["ischemic_stroke"])
    assert not C.icd_match("43490", 9, C.ASCVD_DX["ischemic_stroke"]) # without infarction
    assert not C.icd_match("G454", 10, C.ASCVD_DX["tia"])             # transient global amnesia
    assert C.icd_match("02703ZZ", 10, C.REVASC_PROC["pci"])
    assert C.icd_match("3615", 9, C.REVASC_PROC["cabg"])


def test_icd_sql_agrees_with_python():
    con = duckdb.connect()
    rows = [("I214", 10), ("41071", 9), ("I252", 10), ("41412", 9), ("G454", 10),
            ("G459", 10), ("43490", 9), ("43491", 9), ("4019", 9)]
    con.execute("CREATE TABLE t(icd_code VARCHAR, icd_version INTEGER)")
    con.executemany("INSERT INTO t VALUES (?, ?)", rows)
    for name, spec in C.ASCVD_DX.items():
        got = {r[0] for r in con.execute(f"SELECT icd_code FROM t WHERE {C.icd_sql(spec)}").fetchall()}
        want = {c for c, v in rows if C.icd_match(c, v, spec)}
        assert got == want, name
