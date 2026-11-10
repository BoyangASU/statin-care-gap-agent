"""Run interviews between the interviewer agent and simulated patients, then score them.

  python scripts/08_interviews.py --backend scripted --n 4                     # harness check, no model
  python scripts/08_interviews.py --base-url http://localhost:8000/v1 --model qwen3-8b --n 10 --workers 8
  python scripts/08_interviews.py --show <scenario id> --run-name <name>        # print one transcript
"""
import json
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import _common
from statin_agent.conversation.scenarios import build, TYPES
from statin_agent.conversation.patient import LLMPatient, ScriptedPatient
from statin_agent.conversation.interviewer import LLMInterviewer, ScriptedInterviewer
from statin_agent.conversation.evaluate import score, summarize

p = _common.base_parser(__doc__)
p.add_argument("--backend", choices=["openai", "scripted"], default="openai")
p.add_argument("--base-url"); p.add_argument("--model", default="qwen3-8b")
p.add_argument("--patient-model", help="model for the simulated patient (default: same as --model)")
p.add_argument("--n", type=int, default=10, help="scenarios per type")
p.add_argument("--workers", type=int, default=8)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--run-name")
p.add_argument("--show", help="print the transcript of this scenario id from an existing run")
a = p.parse_args()

name = a.run_name or f"interviews_{a.backend}{'_' + a.model.replace('/', '_') if a.backend == 'openai' else ''}_n{a.n}"
out = _common.ROOT / "results" / name
if a.show:
    for line in open(out / "interviews.jsonl"):
        d = json.loads(line)
        if d["scenario"]["sid"] == a.show:
            sc, r = d["scenario"], d["result"]
            print(f"{sc['sid']} ({sc['kind']}): record says {sc['record_category']}, truth is {sc['truth_category']}\n")
            for t in r["transcript"]:
                print(f"{'AGENT  ' if t['role'] == 'agent' else 'PATIENT'}: {t['text']}")
            print("\nFindings:", json.dumps(r["findings"], indent=1))
            print("Escalations:", r["escalations"], "\nSummary:", r["summary"])
            print("Score:", json.dumps(d["score"], indent=1))
    raise SystemExit

out.mkdir(parents=True, exist_ok=True)
labels = pd.read_csv(_common.ROOT / "results" / "labels.csv")
scenarios = build(labels, a.n, a.seed)
print(f"{len(scenarios)} scenarios: " + ", ".join(f"{t} {sum(s.kind == t for s in scenarios)}" for t in TYPES))

if a.backend == "scripted":
    make = lambda: (ScriptedInterviewer(), ScriptedPatient())
else:
    import os
    from openai import OpenAI
    kw = {"base_url": a.base_url, "api_key": os.environ.get("OPENAI_API_KEY", "EMPTY")} if a.base_url else {}
    client = OpenAI(**kw)
    make = lambda: (LLMInterviewer(client, a.model), LLMPatient(client, a.patient_model or a.model))


def one(s):
    agent, patient = make()
    r = agent.interview(s, patient)
    return {"scenario": s.to_dict(), "result": r, "score": score(s, r)}


rows = []
with open(out / "interviews.jsonl", "w") as fh, ThreadPoolExecutor(max(1, a.workers)) as ex:
    for i, d in enumerate(ex.map(one, scenarios), 1):
        fh.write(json.dumps(d, default=str) + "\n"); fh.flush()
        sc = d["score"]
        print(f"[{i}/{len(scenarios)}] {sc['sid']:<16} {sc['kind']:<15} {sc['record_category']:>17} -> "
              f"{sc['after_category']:<17} truth {sc['truth_category']:<17} q={sc['questions']} "
              f"{'ESC ' if sc['escalated'] else ''}{'unsupported=' + str(sc['unsupported_findings']) if sc['unsupported_findings'] else ''}")
        rows.append(sc)

by, overall = summarize(rows)
pd.set_option("display.width", 160)
print("\nBy scenario type:\n" + by.to_string())
print("\nOverall:\n" + "\n".join(f"  {k}: {v}" for k, v in overall.items()))
by.to_csv(out / "summary_by_type.csv"); (out / "summary.json").write_text(json.dumps(overall, indent=2))
print(f"\nSaved to {out}. Show a transcript: python scripts/08_interviews.py --run-name {name} --show <scenario id>")