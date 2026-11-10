"""End-to-end test on a tiny hand-made Synthea-format dataset (tests/fixtures)."""
from datetime import date
from pathlib import Path

import pytest
from statin_agent.db import build_database, connect, PatientRecord
from statin_agent.facts import reference_label
from statin_agent.agent.backends import Episode
from statin_agent.agent.simulated import SimulatedBackend
from statin_agent.agent.verifier import verify
from statin_agent.agent.pipeline import decide

FIX = Path(__file__).parent / "fixtures"
IDX = date(2026, 1, 1)


@pytest.fixture(scope="module")
def con(tmp_path_factory):
    db = tmp_path_factory.mktemp("db") / "t.duckdb"
    build_database(str(FIX), str(db))
    return connect(str(db))


def test_reference_labels(con):
    # P1: MI history, not on statin -> care gap. LDL 2 years old is ignored; -5 is implausible.
    cat, _, f = reference_label(PatientRecord(con, "P1", IDX))
    assert cat == "indicated" and f.ascvd and f.ldl == 150.0
    # P2: statin stopped in 2020, smoker per latest status, stale labs only -> insufficient data
    cat, _, f = reference_label(PatientRecord(con, "P2", IDX))
    assert not f.on_statin and f.smoker and f.ldl is None and cat == "insufficient_data"
    # P3: active statin
    assert reference_label(PatientRecord(con, "P3", IDX))[0] == "on_statin"


def test_verifier_catches_stale_lab(con):
    rec = PatientRecord(con, "P1", IDX)
    _, _, f = reference_label(rec)
    good = Episode("indicated", dict(vars(f)), trace=[{"tool": "list_allergies"}])
    assert verify(rec, good) == []
    bad = Episode("indicated", {**vars(f), "ldl": 210.0}, trace=[{"tool": "list_allergies"}])
    assert any("ldl" in p for p in verify(rec, bad))


def test_simulated_pipeline_runs(con):
    rec = PatientRecord(con, "P1", IDX)
    d = decide(rec, SimulatedBackend(seed=1), k=3, tau=0.8)
    assert d.category is not None and 0 < d.confidence <= 1
    assert d.samples[0].episodes[0]["trace"]
