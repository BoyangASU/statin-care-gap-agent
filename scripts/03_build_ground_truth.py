"""Reference facts and statin category for every patient in the cohort."""
from dataclasses import asdict
import pandas as pd
import _common
from statin_agent.config import load_config, index_date_for
from statin_agent.db import connect, cohort, PatientRecord
from statin_agent.facts import reference_label

a = _common.base_parser(__doc__).parse_args()
cfg = load_config(a.config)
con = connect(cfg["database"])
idx = index_date_for(cfg, con)
c = cfg["cohort"]
ids = cohort(con, idx, c["min_age"], c["max_age"], c["alive_only"])
rows = []
for pid in ids:
    cat, risk, f = reference_label(PatientRecord(con, pid, idx), cfg["lookback_days"])
    rows.append({"patient_id": pid, "label": cat, "risk": risk, **asdict(f)})
df = pd.DataFrame(rows)
out = _common.ROOT / "results" / "labels.csv"
out.parent.mkdir(exist_ok=True)
df.to_csv(out, index=False)
print(f"index date {idx}; cohort {len(df)} patients -> {out}")
print(df["label"].value_counts().to_string())
print(f"\nLDL within {cfg['lookback_days']} days: {df['ldl'].notna().mean():.0%} | "
      f"ASCVD: {df['ascvd'].mean():.0%} | on statin: {df['on_statin'].mean():.0%} | "
      f"smokers: {df['smoker'].sum()}")
