# Budget-aware review with the SDM 2026 method: what was kept and what changed

## The question

Reviewing a patient is expensive: the agent spends tokens on every record, and a deferred
patient takes a clinician's time. With a fixed budget, **which patients should be reviewed
first?** This applies the cost-aware level-set Bayesian optimization method from
*Adaptive Sampling for Automated Post-Disaster Rapid Damage Assessment* (SDM 2026).

| Paper (damage assessment) | Here (care-gap review) |
|---|---|
| Pool of buildings | Pool of patients the agent already processed (`--run`) |
| Cheap features: location, stories, year built, area | Cheap SQL features: age, sex, race, any ASCVD / diabetes / hypertension code, statin ever listed, latest LDL and its age, counts of LDL tests, conditions, medications |
| Expensive observation: a UAV visit reveals the damage level | A review reveals whether the patient is a care gap |
| Cost: travel distance of the UAV | Cost: tokens the agent spent on that patient |
| Labels: 4 ordered damage levels | Labels: care gap vs not (2 levels) |

The screening features are deliberately the naive versions of what the agent must work out:
the latest LDL regardless of date or plausibility, any statin ever listed, any code ever
recorded. The review is what resolves dates, stop dates and implausible values.

Because the pool was already processed, the simulation calls no LLM: a review looks up the
patient's label and the tokens the agent actually spent.

## Kept unchanged

`src/statin_agent/active/sdm/` holds `model.py`, `likelihood.py` and `acquisition.py`
copied byte for byte (plus a one-line header). The loop follows the notebook
`tsunamis_DA.ipynb`:

- `DeepVariationalOrdinalGPModel` (MLP 25 → 30 features, RBF kernel, variational GP).
- `OrdinalRegressionLikelihood`, thresholds initialised with `linspace(-1, 1, K - 1)` and
  carried over between refits.
- A fresh model each step, Adam lr 1e-3, 1,500 steps (`--steps`).
- `straddle_acquisition` with β = 0.2; λ = 0.5 for the cost term.
- 10 initial random observations, 100 iterations; min-max scaled features.

## Adaptations (each marked `ADAPTATION` in `active/loop.py`)

1. **One threshold.** With two classes, the level set is at `thresholds[0]`
   (the notebook used `thresholds[1]`, between damage levels 1 and 2). With two levels the
   ordinal likelihood is a probit classifier with the same deep kernel.
2. **Cost term.** The paper penalises λ‖x − x_prev‖² on the first two columns
   (longitude, latitude): travel from the UAV's current position. A review has a fixed cost
   of its own, so the penalty is λ · cost(x), with cost scaled to [0, 1] by the pool maximum.
   The straddle term itself is the original function.
3. **First selection.** The model is fitted on the initial observations before the first
   selection (in the notebook the first point was chosen with an untrained model).
4. **Speed only.** The Monte Carlo expected log-likelihood is computed for all 100 samples at
   once instead of in a Python loop. The math is identical; `tests/test_active.py` checks the
   two give the same values. This made a refit about 5× faster.

## Things noticed while reading the original code

Worth a look in the original project; none of them is carried over here.

- `candidate_x = full_train_x[sample:]` keeps the 10 randomly chosen initial points in the
  candidate pool (they were drawn with `randperm`, not as the first 10 rows), and leaves the
  first 10 rows out of it.
- The main loop in `tsunamis_DA.ipynb` calls `straddle_acquisition`, not
  `modified_straddle_acquisition`; the travel cost is recorded but not used to choose.
- `OrdinalRegressionLikelihood` defines `self.sigma` but never uses it.
- `BatchNorm1d` in the feature extractor is trained on batches as small as 10 points.

## Baselines

- `random`: review in random order.
- `greedy`: review the patient most likely to be a care gap (pure exploitation).
- `straddle`: the original acquisition without cost.
- `cost_aware_straddle`: the original acquisition minus λ · cost.

All four start from the same 10 random patients for a given seed.

## Metrics (every 5 reviews)

- **Care gaps found** by review, against cumulative cost (the operational goal).
- **Care gaps identified**: found by review plus correctly flagged by the model among the
  unreviewed patients, and **false alarms** among the unreviewed.
- **F1 for care gaps on unreviewed patients** (the paper's goal: map the rest of the pool).

## Limitations to state

- Care gaps are about 10% of the pool; the classifier may predict "no gap" everywhere when
  few positives are observed, which makes F1 on the unreviewed patients unstable.
- In synthetic data the screening features carry most of the signal; real records, with
  free-text notes, would make the expensive review matter more.
- The level-set acquisition targets the decision boundary, while the operational goal is to
  find care gaps; both kinds of metric are reported for that reason.

## Running it

```bash
pip install gpytorch scikit-learn
# quick check (minutes)
python scripts/07_active_review.py --run qwen8b_rand300_k3 --seeds 1 --iters 40 --steps 300 --jobs 4
# full setting from the paper (4 strategies x 3 seeds, in parallel)
python scripts/07_active_review.py --run qwen8b_rand300_k3 --seeds 3 --jobs 12 --threads 2
```

Output: `results/<run>/active_review/` with `curves.csv`, `summary.csv`, `config.json` and
`active_review.png`.
