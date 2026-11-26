# MIMIC-IV port (Phase 1, work in progress)

This folder ports the statin care-gap agent from Synthea to MIMIC-IV. The
Synthea version under `src/statin_agent/` is unchanged.

Phase-1 has two tasks (draft):

- Task A, patient-level screening (main): for a patient at an index date,
  use all earlier records (every prior admission, inpatient and outpatient
  labs, outpatient blood pressure) to fill the prototype's 14 facts and apply
  the prototype's rules unchanged (6 categories, PCE risk). This is the
  longitudinal-EHR setting closest to the Synthea prototype.
- Task B, statin at discharge after an ASCVD admission: a standard quality
  measure with clear rules, used for strict evaluation.

Status: data loading and descriptive statistics only. Category definitions,
guideline rules, tools, verifier and prompts come after the categories are
confirmed.

## Files

| file | what it does |
|---|---|
| `db.py` | Loads MIMIC-IV `hosp` CSVs (Demo 2.2 or full) into DuckDB with explicit column types; adds view `admissions_x` (age at admission, length of stay). |
| `codes.py` | Draft code sets: ASCVD ICD-9/10 diagnoses, CABG/PCI procedures, candidate "justified no statin" diagnoses, lab itemids, statin name matcher. `python -m mimic.codes --db ...` audits every prefix against the ICD dictionaries. |
| `describe.py` | One row per admission with ASCVD flags, statin exposure and candidate reasons; prints a Markdown summary and a draft category split. |
| `patient_level.py` | Task A: one row per patient at the index date, classified with `statin_agent.guideline.classify`; runs several lab/BP lookback windows. |
| `tests/test_codes.py` | Unit tests for the matchers (no data needed). |
| `docs/demo_summary.md` | Output of `describe.py` (task B) on MIMIC-IV Demo 2.2 (public, ODbL). |
| `docs/demo_patient_level.md` | Output of `patient_level.py` (task A) on the Demo. |

## Run on Sol (CPU only, login node is fine for the Demo)

```bash
cd ~/statin-care-gap-agent
git fetch origin
git checkout claude/project-thread-e5bipv
source /scratch/$USER/venvs/statin/bin/activate

# 1. Load the Demo hosp tables (about 5 seconds)
python -m mimic.db --hosp-dir /scratch/$USER/mimic-iv-demo/hosp --db data/mimic_demo.duckdb

# 2. Check that every code prefix exists in the ICD dictionaries (expect no "NO MATCH" lines)
python -m mimic.codes --db data/mimic_demo.duckdb | grep "NO MATCH"

# 3. Descriptive statistics
python -m mimic.describe --db data/mimic_demo.duckdb --out-dir results_mimic/describe

# 4. Patient-level screening (task A)
python -m mimic.patient_level --db data/mimic_demo.duckdb --out-dir results_mimic/patient_level

# 5. Unit tests
python -m pytest -q mimic/tests
```

`--hosp-dir` accepts either the `hosp` folder or its parent. For the full
MIMIC-IV, point it at the full `hosp` folder and use a different `--db`
(e.g. `/scratch/$USER/mimic/mimic_iv.duckdb`); `labevents` is large, so run
that on a compute node.

## Data rules

- MIMIC-IV Demo (100 patients, ODbL) can be shared and debugged with any tool.
- Credentialed MIMIC-IV: never send records to external LLM APIs; only local
  models. `describe.py` prints aggregate counts only, and the per-admission
  CSV it writes (`results_mimic/`, git-ignored) must stay on Sol.

## Known limitations of the structured data

- Discharge medications are only in discharge notes (MIMIC-IV-Note, separate
  DUA). "Statin at discharge" here is a proxy: a statin order still active in
  the last 24 h of the stay, or a statin dose given in the last 48 h.
- There is no allergy table; statin intolerance has to come from ICD codes,
  lab values, or (later) notes.
- Some prescription rows have `stoptime` before `starttime`, or start after
  discharge (orders discontinued right away). The discharge proxy requires
  `starttime <= dischtime`.
- `labevents.hadm_id` is NULL for outpatient labs, so labs are joined by
  `subject_id` and time window.
