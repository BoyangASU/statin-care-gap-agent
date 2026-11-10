"""Interview scenarios built from Synthea patients.

Each scenario has a TRUE state (what the patient knows) and a RECORD (what the
chart shows). The record differs from the truth in one way that a clinician
would only learn by asking:

  outside_labs    lipids were measured elsewhere in the last few months; the chart has none
  stopped_statin  the chart lists an active statin; the patient stopped it months ago
  intolerance     the patient stopped a statin because of muscle pain; the chart has no allergy
  now_smoker      the chart says non-smoker; the patient has started smoking again
  control         the chart is correct (the agent should change nothing)
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, replace
import random

import pandas as pd

from ..guideline import Facts, classify

TYPES = ["outside_labs", "stopped_statin", "intolerance", "now_smoker", "control"]
FIELDS = ["on_statin", "statin_intolerance", "smoker", "ldl", "total_chol", "hdl"]


@dataclass
class Scenario:
    sid: str
    patient_id: str
    kind: str
    truth: Facts
    record: Facts
    statin_name: str
    detail: dict

    @property
    def truth_category(self):
        return classify(self.truth)[0]

    @property
    def record_category(self):
        return classify(self.record)[0]

    def changed_fields(self):
        return [f for f in FIELDS if getattr(self.truth, f) != getattr(self.record, f)]

    def to_dict(self):
        d = asdict(self)
        d.update(truth_category=self.truth_category, record_category=self.record_category,
                 changed_fields=self.changed_fields())
        return d


def _facts(row) -> Facts:
    val = lambda k: None if pd.isna(row[k]) else (float(row[k]) if k in ("ldl", "total_chol", "hdl", "sbp")
                                                  else bool(row[k]) if k not in ("age", "sex", "race") else row[k])
    return Facts(**{k: val(k) for k in Facts.__dataclass_fields__})


def build(labels: pd.DataFrame, n_per_type: int = 10, seed: int = 0) -> list[Scenario]:
    """Pick patients where each perturbation is meaningful and changes what the chart implies."""
    rng = random.Random(seed)
    df = labels[(labels.age >= 40) & (labels.age <= 75)]
    full = df.dropna(subset=["ldl", "total_chol", "hdl", "sbp"])
    out = []

    def take(pool, kind, make):
        rows = pool.sample(frac=1, random_state=seed).itertuples(index=False)
        n = 0
        for r in rows:
            if n >= n_per_type:
                break
            row = pd.Series(r._asdict())
            sc = make(row)
            if sc and sc.truth_category != sc.record_category or (sc and kind == "control"):
                out.append(sc); n += 1

    statins = ["simvastatin", "atorvastatin", "rosuvastatin", "pravastatin"]

    def outside_labs(row):
        truth = _facts(row)
        if truth.on_statin:
            return None
        record = replace(truth, ldl=None, total_chol=None, hdl=None)
        months = rng.choice([2, 3, 4, 5])
        return Scenario(f"labs-{row.patient_id[:8]}", row.patient_id, "outside_labs", truth, record,
                        "", {"months_ago": months, "where": rng.choice(["a pharmacy health screening",
                                                                         "an urgent care clinic", "a work wellness check"])})

    def stopped(row):
        record = _facts(row)
        if not record.on_statin:
            return None
        truth = replace(record, on_statin=False)
        return Scenario(f"stop-{row.patient_id[:8]}", row.patient_id, "stopped_statin", truth, record,
                        rng.choice(statins), {"months_ago": rng.choice([4, 6, 8, 10]),
                                              "reason": rng.choice(["kept forgetting to refill it",
                                                                    "the copay went up", "felt fine and thought it was not needed"])})

    def intolerance(row):
        record = _facts(row)
        if record.on_statin or classify(record)[0] != "indicated":
            return None
        truth = replace(record, statin_intolerance=True)
        return Scenario(f"intol-{row.patient_id[:8]}", row.patient_id, "intolerance", truth, record,
                        rng.choice(statins), {"months_ago": rng.choice([6, 9, 12, 18])})

    def now_smoker(row):
        record = _facts(row)
        if record.smoker or record.on_statin:
            return None
        truth = replace(record, smoker=True)
        return Scenario(f"smoke-{row.patient_id[:8]}", row.patient_id, "now_smoker", truth, record,
                        "", {"months_ago": rng.choice([3, 6, 9])})

    def control(row):
        truth = _facts(row)
        return Scenario(f"ctrl-{row.patient_id[:8]}", row.patient_id, "control", truth, truth,
                        rng.choice(statins) if truth.on_statin else "", {})

    take(full, "outside_labs", outside_labs)
    take(full, "stopped_statin", stopped)
    take(full, "intolerance", intolerance)
    take(full, "now_smoker", now_smoker)
    take(full, "control", control)
    return out
