"""Budget-aware review with the SDM 2026 cost-aware level-set method.

Simulates reviewing patients one at a time from a pool that the agent has
already processed, so no LLM is called: reviewing a patient reveals its label
(care gap or not) and costs what the agent spent on it (tokens).

  python scripts/07_active_review.py --run qwen8b_rand300_k3 --seeds 3
  python scripts/07_active_review.py --run qwen8b_rand300_k3 --steps 300 --iters 60   # quick check
"""
import json
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import _common
from statin_agent.config import load_config, index_date_for
from statin_agent.db import connect
from statin_agent.evaluation import load_decisions
from statin_agent.active.features import screening_features, minmax, FEATURES
from statin_agent.active.loop import run, STRATEGIES

p = _common.base_parser(__doc__)
p.add_argument("--run", required=True, help="agent run whose patients form the pool")
p.add_argument("--strategies", nargs="*", default=STRATEGIES)
p.add_argument("--seeds", type=int, default=3)
p.add_argument("--init", type=int, default=10, help="patients reviewed at random before the loop")
p.add_argument("--iters", type=int, default=100, help="patients reviewed by the strategy")
p.add_argument("--steps", type=int, default=1500, help="Adam steps per refit (SDM notebook: 1500)")
p.add_argument("--beta", type=float, default=0.2)
p.add_argument("--lam", type=float, default=0.5)
p.add_argument("--cost", choices=["tokens", "uniform"], default="tokens")
p.add_argument("--threads", type=int, default=2, help="torch threads per job")
p.add_argument("--jobs", type=int, default=1, help="strategy x seed runs in parallel")
a = p.parse_args()
torch.set_num_threads(a.threads)

cfg = load_config(a.config)
run_dir = _common.ROOT / "results" / a.run
out = run_dir / "active_review"; out.mkdir(exist_ok=True)
dec = {d["patient_id"]: d for d in load_decisions(run_dir / "decisions.jsonl")}
labels = pd.read_csv(_common.ROOT / "results" / "labels.csv").set_index("patient_id")
con = connect(cfg["database"])
feats = screening_features(con, list(dec), index_date_for(cfg, con))

y = (labels.loc[feats["patient_id"], "label"].to_numpy() == "indicated").astype(int)
tokens = np.array([dec[i]["input_tokens"] + dec[i]["output_tokens"] for i in feats["patient_id"]], float)
if a.cost == "uniform" or tokens.max() == 0:
    cost = np.ones(len(y)); cost_name = "patients reviewed"
else:
    cost = tokens; cost_name = "tokens spent"
X = minmax(feats[FEATURES].to_numpy(float))
print(f"pool {len(y)} patients, {y.sum()} care gaps; cost = {cost_name}; features: {', '.join(FEATURES)}")

def job(args):
    s, seed = args
    torch.set_num_threads(a.threads)
    t0 = time.time()
    r = run(X, y, cost, s, n_init=a.init, n_iter=a.iters, beta=a.beta, lam=a.lam,
            steps=a.steps, seed=seed, log=(print if a.jobs == 1 else (lambda *_: None)))
    print(f"{s} seed {seed} done in {time.time() - t0:.0f}s", flush=True)
    return r

tasks = [(s, seed) for s in a.strategies for seed in range(a.seeds)]
rows = []
if a.jobs > 1:
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    with ProcessPoolExecutor(a.jobs, mp_context=mp.get_context("fork")) as ex:
        for r in ex.map(job, tasks):
            rows += r
else:
    for t in tasks:
        rows += job(t)
df = pd.DataFrame(rows)
df.to_csv(out / "curves.csv", index=False)
(out / "config.json").write_text(json.dumps(vars(a) | {"pool": len(y), "care_gaps": int(y.sum()),
                                                        "cost": cost_name}, indent=2))

last = df[df.step == df.step.max()].groupby("strategy")[["cost_spent", "gaps_found", "gaps_identified",
                                                          "false_alarms_unreviewed", "f1_gap_unreviewed"]].agg(["mean", "std"])
print("\nAfter the last step (mean and std over seeds):\n" + last.round(3).to_string())
last.to_csv(out / "summary.csv")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for s, g in df.groupby("strategy"):
    m = g.groupby("step")[["cost_spent", "gaps_found", "f1_gap_unreviewed"]].mean()
    ax[0].plot(m["cost_spent"], m["gaps_found"], marker="o", ms=3, label=s)
    ax[1].plot(m["cost_spent"], m["f1_gap_unreviewed"], marker="o", ms=3, label=s)
ax[0].set(xlabel=cost_name, ylabel="care gaps found", title="Finding care gaps")
ax[1].set(xlabel=cost_name, ylabel="F1 (care gap) on unreviewed patients", title="Mapping the rest of the pool")
ax[0].legend(); fig.tight_layout(); fig.savefig(out / "active_review.png", dpi=150)
print(f"saved to {out}")
