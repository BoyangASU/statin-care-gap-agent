"""Run the agent on a sample of the cohort and save decisions with full traces.

Examples
  python scripts/04_run_agent.py --backend simulated --n 400
  python scripts/04_run_agent.py --backend openai --model gpt-4o-mini --n 10 --k 1
  python scripts/04_run_agent.py --backend anthropic --model claude-sonnet-4-5 --n 100 --k 3 --workers 4
  python scripts/04_run_agent.py --backend openai --base-url http://localhost:8000/v1 --model Qwen/Qwen2.5-32B-Instruct
Re-running with the same --run-name resumes and skips finished patients.
"""
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import _common
from statin_agent.config import load_config, index_date_for
from statin_agent.db import connect, PatientRecord
from statin_agent.agent.backends import make_backend
from statin_agent.agent.pipeline import decide

p = _common.base_parser(__doc__)
p.add_argument("--backend", choices=["simulated", "openai", "anthropic"], default="simulated")
p.add_argument("--model")
p.add_argument("--base-url", help="OpenAI-compatible server (vLLM, Ollama, ...)")
p.add_argument("--temperature", type=float, default=0.7, help="use -1 to omit (for models without it)")
p.add_argument("--n", type=int, default=100, help="patients to sample from the cohort")
p.add_argument("--k", type=int, help="samples per patient (default from config)")
p.add_argument("--tau", type=float, help="deferral threshold (default from config)")
p.add_argument("--no-self-check", action="store_true")
p.add_argument("--workers", type=int, default=1, help="parallel patients (API backends)")
p.add_argument("--run-name")
p.add_argument("--seed", type=int)
p.add_argument("--stratify", action="store_true",
               help="sample about equally from each reference label (enriches care gaps)")
a = p.parse_args()

cfg = load_config(a.config)
ag = cfg["agent"]
k, tau = a.k or ag["k_samples"], a.tau if a.tau is not None else ag["defer_threshold"]
seed = cfg["seed"] if a.seed is None else a.seed
temp = None if a.temperature < 0 else a.temperature
labels = pd.read_csv(_common.ROOT / "results" / "labels.csv")
if a.stratify:
    per = -(-a.n // labels["label"].nunique())
    ids = (labels.groupby("label", group_keys=False)
           .apply(lambda g: g.sample(n=min(per, len(g)), random_state=seed))["patient_id"].tolist())[:a.n]
else:
    ids = labels.sample(n=min(a.n, len(labels)), random_state=seed)["patient_id"].tolist()

name = a.run_name or f"{'strat_' if a.stratify else ''}{a.backend}{'-' + a.model.replace('/', '_') if a.model else ''}_n{len(ids)}_k{k}"
out_dir = _common.ROOT / "results" / name
out_dir.mkdir(parents=True, exist_ok=True)
out = out_dir / "decisions.jsonl"
done = set()
if out.exists():
    done = {json.loads(l)["patient_id"] for l in out.open() if l.strip()}
todo = [i for i in ids if i not in done]
(out_dir / "run_config.json").write_text(json.dumps(
    {"backend": a.backend, "model": a.model, "base_url": a.base_url, "temperature": temp, "k": k,
     "tau": tau, "self_check": not a.no_self_check, "n": len(ids), "seed": seed}, indent=2))

con = connect(cfg["database"])
idx = index_date_for(cfg, con)
workers = 1 if a.backend == "simulated" else max(1, a.workers)
local = threading.local()
lock = threading.Lock()


def run_one(pid):
    if not hasattr(local, "backend"):
        local.backend = make_backend(a.backend, a.model, temp, ag["max_steps"], a.base_url, seed)
        local.cur = con.cursor()
    rec = PatientRecord(local.cur, pid, idx)
    return decide(rec, local.backend, k=k, tau=tau, self_check=not a.no_self_check,
                  lookback_days=cfg["lookback_days"])


print(f"{len(todo)} to run ({len(done)} already done) -> {out}")
with out.open("a") as fh, ThreadPoolExecutor(workers) as ex:
    futs = {ex.submit(run_one, pid): pid for pid in todo}
    for i, fut in enumerate(as_completed(futs), 1):
        pid = futs[fut]
        try:
            d = fut.result()
        except Exception as e:
            print(f"[{i}/{len(todo)}] {pid} FAILED: {type(e).__name__}: {e}")
            continue
        with lock:
            fh.write(json.dumps(d.to_dict(), default=str) + "\n"); fh.flush()
        flag = "DEFER" if d.deferred else "auto "
        print(f"[{i}/{len(todo)}] {pid[:8]} {flag} {d.category:<18} conf={d.confidence:.2f} "
              f"tokens={d.input_tokens + d.output_tokens}")
print(f"done. Evaluate with: python scripts/05_evaluate.py --run {out_dir.name}")
