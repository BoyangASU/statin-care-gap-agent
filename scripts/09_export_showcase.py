"""Export a small JSON bundle of real results and example traces for the showcase page.

  python scripts/09_export_showcase.py
  -> results/showcase.json   (send this file)
"""
import json

import pandas as pd
import _common
from statin_agent.evaluation import load_decisions, summary_table, field_errors

R = _common.ROOT / "results"
RUNS = {"random": "qwen8b_rand300_k3", "stratified": "qwen8b_strat100_k3"}
INTERVIEWS = "interviews_openai_qwen3-8b_n10"
labels = pd.read_csv(R / "labels.csv")


def short(x, n=500):
    s = json.dumps(x, default=str)
    return s if len(s) <= n else s[:n] + " ..."


def trace(ep):
    return [{"tool": t["tool"], "args": t.get("args", {}), "output": short(t.get("output"))}
            for t in ep.get("trace", [])]


def example(d, lab, why):
    s0 = d["samples"][0]
    eps = s0.get("episodes") or [{}]
    return {"why": why, "patient": d["patient_id"][:8], "reference": lab, "final": d["category"],
            "confidence": d["confidence"], "deferred": d["deferred"], "verified": d["verified"],
            "votes": [s["final_category"] for s in d["samples"]],
            "first_attempt": {"category": s0["first_category"], "facts": eps[0].get("facts"),
                              "trace": trace(eps[0]), "problems": s0["first_problems"]},
            "retry": ({"category": s0["final_category"], "facts": eps[-1].get("facts"),
                       "trace": trace(eps[-1]), "problems": s0["final_problems"]} if len(eps) > 1 else None)}


out = {"agent": {}, "examples": [], "interviews": {}}
for key, run in RUNS.items():
    dec = load_decisions(R / run / "decisions.jsonl")
    lab = labels[labels.patient_id.isin({d["patient_id"] for d in dec})]
    out["agent"][key] = {"n": len(dec), "table": summary_table(lab, dec).reset_index().to_dict("records"),
                         "fields": field_errors(lab, dec).merge(field_errors(lab, dec, "final"), on="field",
                                                                suffixes=("_first", "_after")).to_dict("records"),
                         "labels": lab["label"].value_counts().to_dict()}

dec = load_decisions(R / RUNS["random"] / "decisions.jsonl")
ref = labels.set_index("patient_id")["label"]
picks = {
    "Care gap found automatically": lambda d, l: l == "indicated" and d["category"] == l and not d["deferred"]
                                                and not d["samples"][0]["first_problems"],
    "Self-check caught an error and the retry fixed it": lambda d, l: d["samples"][0]["first_problems"]
                                                and d["samples"][0]["final_category"] == l and not d["deferred"],
    "Wrong answer stopped by deferral": lambda d, l: d["deferred"] and d["category"] != l,
}
for why, test in picks.items():
    for d in dec:
        if test(d, ref[d["patient_id"]]):
            out["examples"].append(example(d, ref[d["patient_id"]], why))
            break

rows = [json.loads(l) for l in open(R / INTERVIEWS / "interviews.jsonl")]
out["interviews"]["summary"] = json.loads((R / INTERVIEWS / "summary.json").read_text())
out["interviews"]["by_type"] = pd.read_csv(R / INTERVIEWS / "summary_by_type.csv").to_dict("records")
want = [("stopped_statin", lambda s: s["hidden_gap_found"]),
        ("intolerance", lambda s: s["correct_after"] and s["escalated"]),
        ("outside_labs", lambda s: s["correct_after"]),
        ("intolerance", lambda s: not s["correct_after"])]
out["interviews"]["examples"] = []
for kind, test in want:
    for r in rows:
        if r.get("score", {}).get("kind") == kind and test(r["score"]):
            out["interviews"]["examples"].append({"scenario": r["scenario"], "transcript": r["result"]["transcript"],
                                                  "findings": r["result"]["findings"],
                                                  "escalations": r["result"]["escalations"],
                                                  "summary": r["result"]["summary"], "score": r["score"]})
            break

(R / "showcase.json").write_text(json.dumps(out, indent=1, default=str))
print(f"wrote {R / 'showcase.json'}: {len(out['examples'])} agent examples, "
      f"{len(out['interviews']['examples'])} interview examples")