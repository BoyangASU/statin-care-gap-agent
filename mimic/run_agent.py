"""Run the MIMIC record-review agent on patients from labels.csv and save decisions with traces.

Examples (from the repository root)
  # harness test, no GPU
  python -m mimic.run_agent --labels-dir results_mimic/demo --backend simulated --n 100
  # local Qwen3-8B on vLLM (start the server first, see mimic/README.md)
  python -m mimic.run_agent --labels-dir results_mimic/demo --backend openai \
      --base-url http://localhost:8000/v1 --model qwen3-8b --n 10 --k 1
Re-running with the same --run-name resumes and skips finished patients.
"""
from __future__ import annotations

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import duckdb
import pandas as pd

from mimic.agent.backend import make_backend
from mimic.agent.pipeline import decide
from mimic.agent.record import MimicRecord


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--db", default="data/mimic_demo.duckdb")
    p.add_argument("--labels-dir", default="results_mimic/demo", help="folder with labels.csv; runs go under it")
    p.add_argument("--backend", choices=["simulated", "openai"], default="simulated")
    p.add_argument("--model")
    p.add_argument("--base-url", help="local OpenAI-compatible server, e.g. http://localhost:8000/v1")
    p.add_argument("--allow-remote", action="store_true",
                   help="allow a non-local server. ONLY for the public MIMIC-IV Demo, never for credentialed data")
    p.add_argument("--temperature", type=float, default=0.7, help="use -1 to omit")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--tau", type=float, default=0.8)
    p.add_argument("--max-steps", type=int, default=20)
    p.add_argument("--lookback-days", type=int, default=365)
    p.add_argument("--no-self-check", action="store_true")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--stratify", action="store_true", help="sample about equally from each reference label")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-name")
    a = p.parse_args()

    base = Path(a.labels_dir)
    labels = pd.read_csv(base / "labels.csv", dtype={"patient_id": str})
    if a.stratify:
        per = -(-a.n // labels["label"].nunique())
        pick = labels.groupby("label", group_keys=False).apply(
            lambda g: g.sample(n=min(per, len(g)), random_state=a.seed)).head(a.n)
    else:
        pick = labels.sample(n=min(a.n, len(labels)), random_state=a.seed)
    todo_all = list(zip(pick["patient_id"], pick["index_hadm_id"]))

    name = a.run_name or f"{'strat_' if a.stratify else ''}{a.backend}{'-' + a.model.replace('/', '_') if a.model else ''}_n{len(todo_all)}_k{a.k}"
    out_dir = base / "runs" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "decisions.jsonl"
    done = {json.loads(l)["patient_id"] for l in out.open() if l.strip()} if out.exists() else set()
    todo = [(pid, h) for pid, h in todo_all if pid not in done]
    temp = None if a.temperature < 0 else a.temperature
    (out_dir / "run_config.json").write_text(json.dumps(
        {"backend": a.backend, "model": a.model, "base_url": a.base_url, "temperature": temp, "k": a.k,
         "tau": a.tau, "self_check": not a.no_self_check, "n": len(todo_all), "seed": a.seed,
         "lookback_days": a.lookback_days, "db": a.db}, indent=2))

    make_backend(a.backend, a.model, temp, a.max_steps, a.base_url, a.seed, a.allow_remote)  # fail fast
    con = duckdb.connect(a.db, read_only=True)
    workers = 1 if a.backend == "simulated" else max(1, a.workers)
    local, lock = threading.local(), threading.Lock()

    def run_one(pid, hadm):
        if not hasattr(local, "backend"):
            local.backend = make_backend(a.backend, a.model, temp, a.max_steps, a.base_url, a.seed, a.allow_remote)
            local.cur = con.cursor()
        rec = MimicRecord(local.cur, int(pid), int(hadm))
        return decide(rec, local.backend, k=a.k, tau=a.tau, self_check=not a.no_self_check,
                      lookback_days=a.lookback_days)

    print(f"{len(todo)} to run ({len(done)} already done) -> {out}")
    with out.open("a") as fh, ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(run_one, pid, h): pid for pid, h in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            pid = futs[fut]
            try:
                d = fut.result()
            except Exception as e:
                print(f"[{i}/{len(todo)}] {pid} FAILED: {type(e).__name__}: {e}")
                continue
            with lock:
                fh.write(json.dumps(d.to_dict(), default=str) + "\n"); fh.flush()
            print(f"[{i}/{len(todo)}] {pid} {'DEFER' if d.deferred else 'auto '} {str(d.category):<18} "
                  f"conf={d.confidence:.2f} tokens={d.input_tokens + d.output_tokens}")
    print(f"done. Evaluate with: python -m mimic.evaluate --labels-dir {base} --run {name}")


if __name__ == "__main__":
    main()
