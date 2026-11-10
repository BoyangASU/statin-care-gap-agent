"""Evaluate a run: configuration table, field errors, confusion matrix, plots."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import _common
from statin_agent import evaluation as ev

p = _common.base_parser(__doc__)
p.add_argument("--run", required=True, help="folder name under results/")
p.add_argument("--bootstrap", type=int, default=0, help="bootstrap resamples for 95%% CIs and a paired test (0 = off)")
p.add_argument("--exclude-runs", nargs="*", default=[],
               help="drop patients that appear in these runs (e.g. development runs)")
a = p.parse_args()
run = _common.ROOT / "results" / a.run
labels = pd.read_csv(_common.ROOT / "results" / "labels.csv")
dec = ev.load_decisions(run / "decisions.jsonl")
excluded = set()
for r in a.exclude_runs:
    excluded |= {d["patient_id"] for d in ev.load_decisions(_common.ROOT / "results" / r / "decisions.jsonl")}
dec = [d for d in dec if d["patient_id"] not in excluded]
labels = labels[labels["patient_id"].isin({d["patient_id"] for d in dec})]

table = ev.summary_table(labels, dec)
fe = ev.field_errors(labels, dec).merge(ev.field_errors(labels, dec, "final"), on="field",
                                        suffixes=("_first_attempt", "_after_self_check"))
conf = ev.confusion(labels, dec)
rc, cal = ev.risk_coverage(labels, dec), ev.calibration(labels, dec)
tokens = sum(d.get("input_tokens", 0) + d.get("output_tokens", 0) for d in dec)

pd.set_option("display.width", 160)
print(f"Run {a.run}: {len(dec)} patients, {tokens:,} tokens"
      + (f" (excluded {len(excluded)} patients from {a.exclude_runs})" if excluded else "") + "\n")
print("Label distribution:\n" + labels["label"].value_counts().to_string() + "\n")
print(table.to_string(), "\n")
print("Field-level extraction errors (sample 1):\n" + fe.to_string(index=False), "\n")
print("Confusion (rows = reference, cols = final decision):\n" + conf.to_string())
if a.bootstrap:
    ci = ev.bootstrap_table(labels, dec, B=a.bootstrap)
    wide = ci.assign(value=ci.apply(lambda r: f"{r.estimate:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}]", axis=1))\
             .pivot(index="configuration", columns="metric", values="value")[ev.CI_METRICS].reindex([n for _, n in ev.CONFIGS])
    print(f"\nEstimates with 95% bootstrap CIs ({a.bootstrap} patient-level resamples):\n" + wide.to_string())
    ci.to_csv(run / "summary_ci.csv", index=False)
    pt = ev.paired_test(labels, dec)
    print(f"\nPaired comparison of wrong automated decisions, single pass vs full pipeline: {pt}")

table.to_csv(run / "summary.csv"); fe.to_csv(run / "field_errors.csv", index=False)
conf.to_csv(run / "confusion.csv"); rc.to_csv(run / "risk_coverage.csv", index=False)
(run / "summary.json").write_text(json.dumps({"tokens": tokens, "table": table.to_dict("index")},
                                             indent=2, default=str))

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(rc["coverage"], rc["risk"], marker="o")
ax[0].set(xlabel="Coverage (share decided automatically)", ylabel="Error rate on automated",
          title="Risk-coverage")
ax[1].bar(cal["confidence"].astype(str), cal["accuracy"])
for i, r in cal.reset_index(drop=True).iterrows():
    ax[1].text(i, r["accuracy"], f"n={int(r['n'])}", ha="center", va="bottom", fontsize=8)
ax[1].set(xlabel="Self-consistency confidence", ylabel="Accuracy", ylim=(0, 1.08),
          title="Calibration")
fig.tight_layout(); fig.savefig(run / "risk_coverage_calibration.png", dpi=150)
print(f"\nSaved tables and figure to {run}")
