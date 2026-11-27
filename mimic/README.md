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

Status: data loading, descriptive statistics, and the task-A record-review
agent (tools, reference labels, verifier, prompt, pipeline) are in place and
run end to end with the scripted backend. The LLM backend has not yet been run
on MIMIC (needs a GPU for local Qwen3-8B).

## Files

| file | what it does |
|---|---|
| `db.py` | Loads MIMIC-IV `hosp` CSVs (Demo 2.2 or full) into DuckDB with explicit column types; adds view `admissions_x` (age at admission, length of stay). |
| `codes.py` | Draft code sets: ASCVD ICD-9/10 diagnoses, CABG/PCI procedures, candidate "justified no statin" diagnoses, lab itemids, statin name matcher. `python -m mimic.codes --db ...` audits every prefix against the ICD dictionaries. |
| `describe.py` | One row per admission with ASCVD flags, statin exposure and candidate reasons; prints a Markdown summary and a draft category split. |
| `patient_level.py` | Task A: one row per patient at the index date, classified with `statin_agent.guideline.classify`; runs several lab/BP lookback windows. |
| `agent/record.py` | `MimicRecord`: one patient, nothing after the index admission's discharge. Reference facts (`reference_facts`) and index-admission selection. |
| `agent/tools.py` | The agent's 8 record tools plus the prototype's two calculator tools and `submit_decision` (with `evidence_codes`). |
| `agent/prompts.py` | System prompt with the MIMIC rules (timing of orders, 365-day window, ICD-9/10 code lists). |
| `agent/verifier.py` | Self-check: the four numbers, on_statin, cited codes exist and fit, diagnoses reviewed, category follows from facts. |
| `agent/backend.py` | OpenAI-compatible loop for a local vLLM server; refuses non-local servers unless `--allow-remote` (Demo only). |
| `agent/simulated.py` | Scripted agent with injected MIMIC-style mistakes, to test the harness without a GPU. |
| `agent/pipeline.py` | k samples, verify, one retry, majority vote, deferral (same logic as the prototype). |
| `build_labels.py`, `run_agent.py`, `evaluate.py` | Labels, runs, and the prototype's metrics on MIMIC. |
| `tests/` | `test_codes.py` (no data); `test_agent.py` (set `MIMIC_DEMO_DB` to run the data tests). |
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

# 5. Unit tests (the data tests run when MIMIC_DEMO_DB is set)
MIMIC_DEMO_DB=data/mimic_demo.duckdb python -m pytest -q mimic/tests

# 6. Record-review agent, harness test with the scripted backend (CPU)
python -m mimic.build_labels --db data/mimic_demo.duckdb --out-dir results_mimic/demo
python -m mimic.run_agent --db data/mimic_demo.duckdb --labels-dir results_mimic/demo --backend simulated --n 100
python -m mimic.evaluate --labels-dir results_mimic/demo --run simulated_n68_k3 --bootstrap 2000
```

### With local Qwen3-8B (when a GPU is free; do not use the GPUs reserved for other experiments)

On a GPU node, in one terminal:

```bash
module load cuda-13.0.1-gcc-13.2.0
source /scratch/$USER/venvs/statin/bin/activate
vllm serve Qwen/Qwen3-8B --served-model-name qwen3-8b --enable-auto-tool-choice \
    --tool-call-parser hermes --max-model-len 32768 --port 8000
```

In a second terminal on the same node:

```bash
cd ~/statin-care-gap-agent && source /scratch/$USER/venvs/statin/bin/activate
export OPENAI_API_KEY=EMPTY
# smoke test: 10 patients, 1 sample each
python -m mimic.run_agent --db data/mimic_demo.duckdb --labels-dir results_mimic/demo --backend openai \
    --base-url http://localhost:8000/v1 --model qwen3-8b --n 10 --k 1 --run-name qwen_smoke
python -m mimic.evaluate --labels-dir results_mimic/demo --run qwen_smoke
# full Demo run (68 patients, k=3)
python -m mimic.run_agent --db data/mimic_demo.duckdb --labels-dir results_mimic/demo --backend openai \
    --base-url http://localhost:8000/v1 --model qwen3-8b --n 100 --k 3 --workers 4 --run-name qwen_k3
python -m mimic.evaluate --labels-dir results_mimic/demo --run qwen_k3 --bootstrap 2000
```

The backend only talks to a local server (localhost or a bare node name such
as `sg017`). That is required for credentialed MIMIC-IV.

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
