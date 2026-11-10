"""Run the agent on one patient and print the full trace (good for a screen recording).

  python scripts/demo_patient.py --backend openai --model gpt-4o-mini            # a random care-gap patient
  python scripts/demo_patient.py --backend anthropic --patient <id> --k 3
"""
import json
import pandas as pd
import _common
from statin_agent.config import load_config, index_date_for
from statin_agent.db import connect, PatientRecord
from statin_agent.agent.backends import make_backend
from statin_agent.agent.pipeline import decide

p = _common.base_parser(__doc__)
p.add_argument("--backend", choices=["simulated", "openai", "anthropic"], default="simulated")
p.add_argument("--model"); p.add_argument("--base-url")
p.add_argument("--patient"); p.add_argument("--label", default="indicated")
p.add_argument("--k", type=int, default=1); p.add_argument("--seed", type=int, default=0)
a = p.parse_args()
cfg = load_config(a.config)
labels = pd.read_csv(_common.ROOT / "results" / "labels.csv")
pid = a.patient or labels[labels["label"] == a.label].sample(1, random_state=a.seed)["patient_id"].iloc[0]
con = connect(cfg["database"])
rec = PatientRecord(con, pid, index_date_for(cfg, con))
d = decide(rec, make_backend(a.backend, a.model, 0.7, cfg["agent"]["max_steps"], a.base_url, a.seed),
           k=a.k, tau=cfg["agent"]["defer_threshold"])

for si, s in enumerate(d.samples, 1):
    for ei, ep in enumerate(s.episodes):
        print(f"\n=== sample {si}, {'retry after self-check' if ei else 'attempt'} ===")
        for t in ep["trace"]:
            out = json.dumps(t["output"], default=str)
            print(f"-> {t['tool']}({json.dumps(t['args'])})\n   {out[:300]}{'...' if len(out) > 300 else ''}")
        print(f"SUBMIT {ep['category']}: {ep['rationale']}")
    if s.first_problems:
        print("Verifier found:\n  " + "\n  ".join(s.first_problems))
ref = labels.set_index("patient_id").loc[pid]
print(f"\nFINAL: {d.category} | confidence {d.confidence:.2f} | verified {d.verified} | "
      f"{'DEFER TO CLINICIAN' if d.deferred else 'automated'}")
print(f"Reference label: {ref['label']}")
