# MIMIC-IV patient-level statin screening (draft)

Reference rules: the Synthea prototype's `statin_agent.guideline.classify`, unchanged. `statin_intolerance` here = statin-harm ICD code in any admission up to index, or active liver disease / pregnancy on the index admission. Index = discharge of the last admission that did not end in death, hospice or palliative care.

## Lookback for labs and blood pressure: 365 days

Patients: 93.

| category | patients |
|---|---|
| excluded_end_of_life | 0 |
| on_statin | 41 |
| contraindicated | 6 |
| indicated | 4 |
| consider | 0 |
| not_indicated | 15 |
| insufficient_data | 27 |

Facts available (all patients):

| fact | patients |
|---|---|
| age 40-75 at index | 61 |
| ASCVD in any admission up to index | 37 |
| diabetes code | 33 |
| hypertension code | 62 |
| current smoking code on index admission | 18 |
| LDL available | 21 |
| total cholesterol and HDL available | 21 |
|   latest lipid was an outpatient lab | 13 |
| outpatient SBP available (omr) | 39 |
| all four PCE inputs available | 11 |
| statin active at end of index admission | 41 |
| statin in any admission up to index | 50 |
| more than one admission up to index | 41 |

`indicated` (care gap) patients: 4

|  | patients |
|---|---|
| ASCVD (secondary prevention) | 3 |
| not ASCVD: diabetes / LDL>=190 / PCE | 1 |
|   of these, statin seen in an earlier admission | 0 |
| any gap with a statin in an earlier admission | 2 |

`insufficient_data` patients: 27; missing input counts:

| input | missing |
|---|---|
| ldl | 24 |
| total_chol | 24 |
| hdl | 24 |
| sbp | 15 |

PCE computed for 1 patients; median 10-year risk 11.5%.

## Lookback for labs and blood pressure: 1095 days

Patients: 93.

| category | patients |
|---|---|
| excluded_end_of_life | 0 |
| on_statin | 41 |
| contraindicated | 6 |
| indicated | 4 |
| consider | 0 |
| not_indicated | 16 |
| insufficient_data | 26 |

Facts available (all patients):

| fact | patients |
|---|---|
| age 40-75 at index | 61 |
| ASCVD in any admission up to index | 37 |
| diabetes code | 33 |
| hypertension code | 62 |
| current smoking code on index admission | 18 |
| LDL available | 23 |
| total cholesterol and HDL available | 23 |
|   latest lipid was an outpatient lab | 16 |
| outpatient SBP available (omr) | 44 |
| all four PCE inputs available | 14 |
| statin active at end of index admission | 41 |
| statin in any admission up to index | 50 |
| more than one admission up to index | 41 |

`indicated` (care gap) patients: 4

|  | patients |
|---|---|
| ASCVD (secondary prevention) | 3 |
| not ASCVD: diabetes / LDL>=190 / PCE | 1 |
|   of these, statin seen in an earlier admission | 0 |
| any gap with a statin in an earlier admission | 2 |

`insufficient_data` patients: 26; missing input counts:

| input | missing |
|---|---|
| ldl | 23 |
| total_chol | 23 |
| hdl | 23 |
| sbp | 15 |

PCE computed for 1 patients; median 10-year risk 11.5%.

## Lookback for labs and blood pressure: no limit

Patients: 93.

| category | patients |
|---|---|
| excluded_end_of_life | 0 |
| on_statin | 41 |
| contraindicated | 6 |
| indicated | 5 |
| consider | 0 |
| not_indicated | 16 |
| insufficient_data | 25 |

Facts available (all patients):

| fact | patients |
|---|---|
| age 40-75 at index | 61 |
| ASCVD in any admission up to index | 37 |
| diabetes code | 33 |
| hypertension code | 62 |
| current smoking code on index admission | 18 |
| LDL available | 27 |
| total cholesterol and HDL available | 27 |
|   latest lipid was an outpatient lab | 19 |
| outpatient SBP available (omr) | 45 |
| all four PCE inputs available | 18 |
| statin active at end of index admission | 41 |
| statin in any admission up to index | 50 |
| more than one admission up to index | 41 |

`indicated` (care gap) patients: 5

|  | patients |
|---|---|
| ASCVD (secondary prevention) | 3 |
| not ASCVD: diabetes / LDL>=190 / PCE | 2 |
|   of these, statin seen in an earlier admission | 0 |
| any gap with a statin in an earlier admission | 2 |

`insufficient_data` patients: 25; missing input counts:

| input | missing |
|---|---|
| ldl | 22 |
| total_chol | 22 |
| hdl | 22 |
| sbp | 15 |

PCE computed for 2 patients; median 10-year risk 18.7%.
