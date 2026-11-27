"""A scripted stand-in for an LLM, to test the MIMIC harness without a GPU.

It follows the correct tool plan through the same tools an LLM uses, but makes
mistakes an LLM agent plausibly makes on MIMIC, each with a set probability:

  stale    - takes the newest lab / BP even if older than 365 days        (verifier sees)
  history  - counts a statin from ANY admission as on_statin               (verifier sees)
  invent   - cites an ASCVD code that is not in the record                 (verifier sees)
  index    - looks only at index-admission codes for ASCVD/diabetes        (verifier blind)
  smoke    - treats a history-of-smoking code as current smoking           (verifier blind)
  bp       - misses beta-blockers when setting bp_treated                  (verifier blind)

Its numbers demonstrate the evaluation harness only; they say nothing about a
real LLM. Run the openai backend against a local vLLM server for that.
"""
from __future__ import annotations

import random
import re
import time

from mimic import codes as C

from .backend import Episode
from .record import PLAUSIBLE
from .tools import execute, rows_as_dicts

_statin = re.compile(C.STATIN_REGEX)
_bp = re.compile(C.ANTIHYPERTENSIVE_REGEX)
_bp_narrow = re.compile(r"(lisinopril|amlodipine|hydrochlorothiazide|losartan)")


def _match(row, specs):
    return any(C.icd_match(row["icd_code"], row["icd_version"], s) for s in specs)


class SimulatedBackend:
    name = "simulated"

    def __init__(self, seed=0, p_stale=0.2, p_history=0.15, p_invent=0.1,
                 p_index=0.2, p_smoke=0.2, p_bp=0.2):
        self.p = dict(stale=p_stale, history=p_history, invent=p_invent,
                      index=p_index, smoke=p_smoke, bp=p_bp)
        self.rng = random.Random(seed)
        self._last = None

    def run(self, rec, feedback=None) -> Episode:
        if feedback and self._last is not None:
            err = dict(self._last)             # a retry fixes only what the verifier flagged
            if "365" in feedback or "implausible" in feedback:
                err["stale"] = False
            if "on_statin" in feedback:
                err["history"] = False
            if "not in this patient's record" in feedback:
                err["invent"] = False
        else:
            err = {k: self.rng.random() < v for k, v in self.p.items()}
        self._last = err
        trace, t0 = [], time.time()

        def call(tool, **args):
            out = execute(rec, tool, args)
            trace.append({"tool": tool, "args": args, "output": out})
            return out

        demo = call("get_demographics")
        dx = rows_as_dicts(call("list_diagnoses"))
        px = call("list_procedures")
        orders = call("list_medication_orders", keyword=None, all_admissions=err["history"])
        given = call("list_administrations", keyword="statin")
        pool = [d for d in dx if d["on_index_admission"]] if err["index"] else dx

        ascvd_codes = [d["icd_code"] for d in pool if _match(d, list(C.ASCVD_DX.values()))]
        ascvd_codes += [p["icd_code"] for p in px if _match(p, list(C.REVASC_PROC.values()))]
        if err["invent"]:
            ascvd_codes.append("I2510" if "I2510" not in {d["icd_code"] for d in dx} else "41401")
        dm_codes = [d["icd_code"] for d in pool if _match(d, [C.RISK_DX["diabetes"]])]
        htn_codes = [d["icd_code"] for d in dx if _match(d, [C.RISK_DX["hypertension"]])]
        smoke_specs = [C.RISK_DX["current_smoker"]] + ([C.RISK_DX["former_smoker"]] if err["smoke"] else [])
        smk_codes = [d["icd_code"] for d in dx if d["on_index_admission"] and _match(d, smoke_specs)]
        intol_codes = [d["icd_code"] for d in dx if _match(d, [C.JUSTIFIED_DX[k] for k in
                       ("statin_adverse_effect", "drug_myopathy", "rhabdomyolysis")])]
        intol_codes += [d["icd_code"] for d in dx if d["on_index_admission"] and
                        _match(d, [C.JUSTIFIED_DX["liver_disease"], C.JUSTIFIED_DX["pregnancy"]])]

        def active(o):
            h = o.get("last_stop_hours_before_discharge")
            return err["history"] or (o["hadm_id"] == rec.index_hadm_id and o["first_start"] is not None
                                      and o["first_start"] <= demo["index_time"] and (h is None or h <= 24))
        on_statin = any(_statin.search(o["drug"].lower()) and active(o) for o in orders) or any(
            _statin.search((g["medication"] or "").lower()) and g["event"] in C.EMAR_GIVEN
            and g["hours_before_discharge"] is not None and g["hours_before_discharge"] <= 48 for g in given)
        bp_re = _bp_narrow if err["bp"] else _bp
        bp_orders = orders if not err["history"] else call("list_medication_orders")
        bp_treated = any(bp_re.search(o["drug"].lower()) and o["hadm_id"] == rec.index_hadm_id and
                         o["first_start"] is not None and o["first_start"] <= demo["index_time"] and
                         (o.get("last_stop_hours_before_discharge") is None or
                          o["last_stop_hours_before_discharge"] <= 24) for o in bp_orders)

        def lab(key):
            lo, hi = PLAUSIBLE[key]
            rows = call("get_blood_pressure", limit=20) if key == "sbp" else call("get_labs", name=key, limit=20)
            for r in rows:
                if not err["stale"] and r["days_before_index"] > 365:
                    return None
                try:
                    v = float(str(r["value"]).split("/")[0])
                except ValueError:
                    continue
                if lo <= v <= hi:
                    return v
            return None

        facts = dict(age=demo["age"], sex=demo["sex"],
                     race="black" if "BLACK" in (demo["race"] or "").upper() else "white",
                     on_statin=on_statin, statin_intolerance=bool(intol_codes), ascvd=bool(ascvd_codes),
                     diabetes=bool(dm_codes), smoker=bool(smk_codes), bp_treated=bp_treated,
                     hypertension=bool(htn_codes), ldl=lab("ldl"), total_chol=lab("total_chol"),
                     hdl=lab("hdl"), sbp=lab("sbp"))
        g = call("check_statin_guideline", **facts)
        ev = {k: v for k, v in dict(ascvd=ascvd_codes, diabetes=dm_codes, hypertension=htn_codes,
                                     smoker=smk_codes, statin_intolerance=intol_codes).items() if v}
        return Episode(g["category"], facts, f"Guideline category: {g['category']}.", "", ev,
                       trace, seconds=time.time() - t0)
