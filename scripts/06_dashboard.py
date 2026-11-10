"""Build a clinician-facing dashboard (one self-contained HTML file) for a run.

  python scripts/06_dashboard.py --run qwen8b_debug
  -> results/qwen8b_debug/dashboard.html   (open in a browser; no server needed)
"""
import json
from datetime import date

import pandas as pd
import _common
from statin_agent.codes import ASCVD_SNOMED, DIABETES_T2_SNOMED, HYPERTENSION_SNOMED, LOINC, is_statin
from statin_agent.config import load_config, index_date_for
from statin_agent.db import connect, PatientRecord
from statin_agent.evaluation import load_decisions

p = _common.base_parser(__doc__)
p.add_argument("--run", required=True)
a = p.parse_args()
cfg = load_config(a.config)
run_dir = _common.ROOT / "results" / a.run
labels = pd.read_csv(_common.ROOT / "results" / "labels.csv").set_index("patient_id")
decisions = load_decisions(run_dir / "decisions.jsonl")
run_cfg = json.loads((run_dir / "run_config.json").read_text()) if (run_dir / "run_config.json").exists() else {}
con = connect(cfg["database"])
idx = index_date_for(cfg, con)
LB = cfg["lookback_days"]
LAB_NAMES = {"ldl": "LDL", "total_chol": "Total cholesterol", "hdl": "HDL", "sbp": "Systolic BP"}


def iso(d):
    return d.isoformat() if isinstance(d, date) else d


def timeline(rec):
    ev = []
    for c in rec.conditions():
        code = c["code"]
        kind = ("ascvd" if code in ASCVD_SNOMED else "diabetes" if code in DIABETES_T2_SNOMED
                else "hypertension" if code in HYPERTENSION_SNOMED else None)
        if kind:
            ev.append({"date": iso(c["start"]), "kind": kind, "label": c["description"].split(" (")[0]})
    statins = {}
    for m in rec.medications():
        if is_statin(m["description"]):
            s = statins.setdefault(m["description"], {"start": m["start"], "stop": m["stop"]})
            s["stop"] = m["stop"]
    for name, s in statins.items():
        ev.append({"date": iso(s["start"]), "kind": "statin", "label": f"Started {name}"})
        if s["stop"] and s["stop"] <= idx:
            ev.append({"date": iso(s["stop"]), "kind": "statin_stop", "label": f"Stopped {name}"})
    ldl = []
    for r in rec.observations(loinc=LOINC["ldl"], limit=40):
        try:
            ldl.append({"date": iso(r["date"]), "value": round(float(r["value"]), 1)})
        except (TypeError, ValueError):
            pass
    return sorted(ev, key=lambda e: e["date"]), ldl[::-1]


patients = []
for d in decisions:
    pid = d["patient_id"]
    rec = PatientRecord(con, pid, idx)
    s0 = d["samples"][0]
    eps = s0.get("episodes") or [{}]
    final = eps[-1]
    facts = final.get("facts") or {}
    ref = labels.loc[pid] if pid in labels.index else None
    missing = [f"No {LAB_NAMES[k]} in the past {LB} days" for k in LAB_NAMES if facts.get(k) is None]
    events, ldl = timeline(rec)
    patients.append({
        "id": pid, "short": pid[:8], "age": facts.get("age"), "sex": facts.get("sex"),
        "category": d["category"], "confidence": d["confidence"], "verified": d["verified"],
        "deferred": d["deferred"], "samples": [s["final_category"] for s in d["samples"]],
        "facts": facts, "missing": missing, "rationale": final.get("rationale", ""),
        "evidence": final.get("evidence", ""),
        "problems": sorted({p for s in d["samples"] for p in (s["first_problems"] + s["final_problems"])}),
        "retried": any(s["retried"] for s in d["samples"]),
        "trace": [{"tool": t["tool"], "args": t.get("args", {})} for t in final.get("trace", [])],
        "risk": None if ref is None or pd.isna(ref["risk"]) else round(float(ref["risk"]), 4),
        "reference": None if ref is None else ref["label"],
        "events": events, "ldl": ldl,
    })

payload = {"run": a.run, "model": run_cfg.get("model") or run_cfg.get("backend", ""),
           "k": run_cfg.get("k"), "tau": run_cfg.get("tau"), "index_date": idx.isoformat(),
           "lookback": LB, "patients": patients}
tpl = (_common.ROOT / "src" / "statin_agent" / "templates" / "dashboard.html").read_text()
out = run_dir / "dashboard.html"
out.write_text(tpl.replace("/*__DATA__*/null", json.dumps(payload, default=str)))
print(f"{len(patients)} patients -> {out}")
