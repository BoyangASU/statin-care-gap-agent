"""A scripted stand-in for an LLM, to test the harness without an API key.

It follows the correct tool plan through the same tools an LLM uses, but makes
mistakes LLM agents make on EHR data, each with a set probability per run:

  stale   - uses the newest lab even if it is older than 365 days   (verifier sees)
  invalid - accepts physiologically implausible values               (verifier sees)
  med     - treats any listed statin as active, ignoring stop dates  (verifier sees)
  allergy - skips the allergy check                                  (verifier sees)
  ascvd   - counts only myocardial-infarction codes as ASCVD         (verifier blind)
  smoke   - reads the oldest smoking status, not the latest          (verifier blind)
  bp      - misses beta-blockers and ARBs when setting bp_treated    (verifier blind)

Its numbers demonstrate the evaluation harness only. They say nothing about the
accuracy of a real LLM; run a real backend for that.
"""
from __future__ import annotations
import random
import time

from ..codes import LOINC, PLAUSIBLE, DIABETES_T2_SNOMED, is_statin, is_antihypertensive
from ..tools import execute
from .backends import Episode

MI_CODES = {"22298006", "401314000", "401303003", "399211009"}
ALL_ASCVD = MI_CODES | {"414545008", "399261000", "230690007"}
DIABETES = {str(c) for c in DIABETES_T2_SNOMED}
NARROW_BP = ("lisinopril", "amlodipine", "hydrochlorothiazide")


class SimulatedBackend:
    name = "simulated"

    def __init__(self, seed=0, p_stale=0.25, p_invalid=0.15, p_med=0.15, p_allergy=0.15,
                 p_ascvd=0.25, p_smoke=0.25, p_bp=0.25):
        self.p = dict(stale=p_stale, invalid=p_invalid, med=p_med, allergy=p_allergy,
                      ascvd=p_ascvd, smoke=p_smoke, bp=p_bp)
        self.rng = random.Random(seed)
        self._last = None

    def run(self, rec, feedback=None) -> Episode:
        if feedback and self._last is not None:
            err = dict(self._last)             # a retry fixes only what the verifier flagged
            if "365" in feedback or "implausible" in feedback:
                err["stale"] = err["invalid"] = False
            if "on_statin" in feedback:
                err["med"] = False
            if "allerg" in feedback:
                err["allergy"] = False
        else:
            err = {k: self.rng.random() < v for k, v in self.p.items()}
        self._last = err
        trace, t0 = [], time.time()

        def call(tool, **args):
            out = execute(rec, tool, args)
            trace.append({"tool": tool, "args": args, "output": out})
            return out

        demo = call("get_demographics")
        conds = call("list_conditions")
        meds = call("list_medications")
        idx = demo["index_date"]
        active = [m for m in meds if err["med"] or m["most_recent_stop"] is None
                  or m["most_recent_stop"] > idx]
        live_codes = {c["code"] for c in conds if c["resolved"] is None or c["resolved"] > idx
                      or c["code"] in ALL_ASCVD | DIABETES}

        def lab(key):
            lo, hi = PLAUSIBLE[key]
            for r in call("get_observations", loinc=LOINC[key], limit=10):
                if not err["stale"] and r["days_before_index"] > 365:
                    return None
                try:
                    v = float(r["value"])
                except (TypeError, ValueError):
                    continue
                if err["invalid"] or lo <= v <= hi:
                    return round(v, 1)
            return None

        intol = False
        if not err["allergy"]:
            intol = any(is_statin(a["description"]) for a in call("list_allergies"))
        smk = call("get_observations", loinc=LOINC["smoking"], limit=50)
        status = (smk[-1] if err["smoke"] else smk[0])["value"].lower() if smk else ""
        bp_drug = (lambda d: any(w in d.lower() for w in NARROW_BP)) if err["bp"] else is_antihypertensive
        facts = dict(
            age=demo["age"], sex=demo["sex"], race=demo["race"],
            on_statin=any(is_statin(m["description"]) for m in active),
            statin_intolerance=intol,
            ascvd=bool(live_codes & (MI_CODES if err["ascvd"] else ALL_ASCVD)),
            diabetes=bool(live_codes & DIABETES), hypertension="59621000" in live_codes,
            smoker="smokes" in status, bp_treated=any(bp_drug(m["description"]) for m in active),
            ldl=lab("ldl"), total_chol=lab("total_chol"), hdl=lab("hdl"), sbp=lab("sbp"))
        g = call("check_statin_guideline", **facts)
        return Episode(g["category"], facts, f"Guideline category: {g['category']}.", "",
                       trace, seconds=time.time() - t0)
