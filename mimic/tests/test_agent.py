"""Tests for the MIMIC agent harness.

The data tests need a DuckDB built from MIMIC-IV Demo:
    MIMIC_DEMO_DB=data/mimic_demo.duckdb python -m pytest -q mimic/tests
Without MIMIC_DEMO_DB they are skipped; the backend test always runs.
"""
import json
import os
from types import SimpleNamespace

import pytest

DB = os.environ.get("MIMIC_DEMO_DB")
needs_db = pytest.mark.skipif(not DB or not os.path.exists(DB), reason="set MIMIC_DEMO_DB")


@pytest.fixture(scope="module")
def con():
    import duckdb
    return duckdb.connect(DB, read_only=True)


@needs_db
def test_reference_matches_patient_level_sql(con):
    """Python reference facts (used for labels/verifier) == SQL version (used for stats)."""
    from mimic.agent.record import MimicRecord, index_admissions, reference_facts
    from mimic.patient_level import build_patient_table
    build_patient_table(con, 365)
    cur = con.execute("SELECT * FROM pt_facts")
    cols = [c[0] for c in cur.description]
    sql = {r["subject_id"]: r for r in (dict(zip(cols, row)) for row in cur.fetchall())}
    ids = index_admissions(con, 0, 200)
    assert len(ids) == len(sql)
    for s, h in ids:
        assert sql[s]["index_hadm_id"] == h
        f = reference_facts(MimicRecord(con, s, h))
        for k in f.__dataclass_fields__:
            a, b = getattr(f, k), sql[s][k]
            if isinstance(a, float) or isinstance(b, float):
                assert (a is None and b is None) or abs(a - b) < 1e-6, (s, k, a, b)
            else:
                assert a == b, (s, k, a, b)


@needs_db
def test_record_hides_the_future(con):
    from mimic.agent.record import MimicRecord
    s, h = con.execute("""SELECT subject_id, hadm_id FROM admissions
                          QUALIFY row_number() OVER (PARTITION BY subject_id ORDER BY admittime) = 1
                          AND count(*) OVER (PARTITION BY subject_id) > 2 LIMIT 1""").fetchone()
    rec = MimicRecord(con, s, h)
    assert [a["hadm_id"] for a in rec.admissions()] == [h]
    assert all(r["days_before_index"] >= 0 for r in rec.labs(keyword="a", limit=50))


@needs_db
def test_perfect_agent_passes_verifier(con):
    from mimic.agent.record import MimicRecord, index_admissions
    from mimic.agent.simulated import SimulatedBackend
    from mimic.agent.verifier import verify
    b = SimulatedBackend(p_stale=0, p_history=0, p_invent=0, p_index=0, p_smoke=0, p_bp=0)
    for s, h in index_admissions(con):
        rec = MimicRecord(con, s, h)
        assert verify(rec, b.run(rec)) == [], s


def test_openai_loop_with_stub_client():
    """The tool loop parses tool calls, runs tools, and stops at submit_decision."""
    from mimic.agent import backend as B

    def call(name, args, i):
        return SimpleNamespace(id=f"c{i}", function=SimpleNamespace(name=name, arguments=json.dumps(args)))

    script = [[call("get_demographics", {}, 0)],
              [call("submit_decision", {"category": "not_indicated", "facts": {"age": 30},
                                        "evidence_codes": {}, "rationale": "young"}, 1)]]

    class Msg(SimpleNamespace):
        def model_dump(self, exclude_none=True):
            return {"role": "assistant", "content": None}

    class Stub:
        def __init__(self):
            self.i = 0
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kw):
            m = Msg(tool_calls=script[self.i]); self.i += 1
            return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
                                   choices=[SimpleNamespace(message=m)])

    be = B.OpenAIBackend.__new__(B.OpenAIBackend)
    be.client, be.model, be.temperature, be.max_steps = Stub(), "stub", None, 5
    rec = SimpleNamespace(patient_id="1", demographics=lambda: {"age": 30})
    ep = be.run(rec)
    assert ep.category == "not_indicated" and ep.trace[0]["output"] == {"age": 30}
    assert ep.input_tokens == 20


def test_backend_refuses_remote_server():
    from mimic.agent.backend import make_backend, _is_local
    assert _is_local("http://localhost:8000/v1") and _is_local("http://sg017:8000/v1")
    assert not _is_local("https://api.openai.com/v1")
    with pytest.raises(SystemExit):
        make_backend("openai", base_url="https://api.openai.com/v1")
