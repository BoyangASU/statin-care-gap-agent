# MIMIC-IV statin task: descriptive statistics

Patients: 100; admissions: 275.

## Data coverage (all admissions)

|  | admissions |
|---|---|
| with prescriptions rows | 250 |
| with eMAR rows | 181 |
| with either (has_med_data) | 271 |
| with any ICD diagnosis | 275 |

## ASCVD cohort

`index` = principal diagnosis (seq_num 1) is acute MI / CHD / ischemic stroke / TIA, or CABG/PCI during the stay. `any` = any ASCVD code in any position (including old MI, PAD, revascularization status) or CABG/PCI.

| cohort | admissions | patients |
|---|---|---|
| index | 24 | 20 |
| any | 89 | 44 |

By component (admissions; a stay can have several):

| component | any position | principal dx |
|---|---|---|
| acute_mi | 12 | 6 |
| old_mi | 17 | - |
| chd | 74 | 14 |
| ischemic_stroke | 3 | 2 |
| tia | 0 | 0 |
| pad | 9 | - |
| revasc_history | 39 | - |
| cabg | 13 | - |
| pci | 5 | - |

ICD version among `any` ASCVD admissions: ICD-9: 40, ICD-10: 49

## Statin exposure, `index` cohort (n = 24)

|  | admissions |
|---|---|
| has any medication data | 24 |
| statin ordered or given during stay | 22 |
|   ordered (prescriptions) | 22 |
|   given (eMAR, administered) | 13 |
| statin at discharge (proxy: order active in last 24 h or dose in last 48 h) | 22 |
| statin during stay but not near discharge | 0 |
| statin in an earlier admission | 8 |
| non-statin lipid therapy ordered (ezetimibe, PCSK9i, ...) | 1 |

## Statin exposure, `any` cohort (n = 89)

|  | admissions |
|---|---|
| has any medication data | 88 |
| statin ordered or given during stay | 60 |
|   ordered (prescriptions) | 60 |
|   given (eMAR, administered) | 35 |
| statin at discharge (proxy: order active in last 24 h or dose in last 48 h) | 55 |
| statin during stay but not near discharge | 5 |
| statin in an earlier admission | 46 |
| non-statin lipid therapy ordered (ezetimibe, PCSK9i, ...) | 12 |

Drug names containing 'statin' in prescriptions (all admissions) and whether the matcher accepts them:

| drug | rows | is_statin |
|---|---|---|
| atorvastatin | 80 | True |
| simvastatin | 34 | True |
| rosuvastatin calcium | 15 | True |
| nystatin oral suspension | 8 | False |
| pravastatin | 7 | True |
| nystatin cream | 5 | False |
| nystatin-triamcinolone cream | 3 | False |
| nystatin | 1 | False |

## Candidate reasons, `index` cohort, no statin at discharge (n = 2)

`r_` = proposed justified reason; `c_` = debatable, reported only.

| signal | admissions |
|---|---|
| r_statin_harm_dx | 0 |
| r_prior_statin_harm_dx | 0 |
| r_ck_gt10x | 0 |
| r_transaminase_gt3x | 0 |
| r_liver_disease | 0 |
| r_pregnancy | 0 |
| r_died | 0 |
| r_hospice_palliative | 0 |
| r_left_ama | 0 |
| r_transferred_acute | 0 |
| c_esrd_dialysis | 0 |
| c_age_gt75 | 0 |
| c_ldl_lt70 | 0 |
| c_nonstatin_lipid_only | 0 |
| c_switched_to_nonstatin | 0 |
| c_myalgia_dx | 0 |
| c_dnr | 0 |

## Candidate reasons, `any` cohort, no statin at discharge (n = 33)

`r_` = proposed justified reason; `c_` = debatable, reported only.

| signal | admissions |
|---|---|
| r_statin_harm_dx | 0 |
| r_prior_statin_harm_dx | 0 |
| r_ck_gt10x | 0 |
| r_transaminase_gt3x | 7 |
| r_liver_disease | 3 |
| r_pregnancy | 0 |
| r_died | 7 |
| r_hospice_palliative | 4 |
| r_left_ama | 1 |
| r_transferred_acute | 0 |
| c_esrd_dialysis | 0 |
| c_age_gt75 | 5 |
| c_ldl_lt70 | 2 |
| c_nonstatin_lipid_only | 9 |
| c_switched_to_nonstatin | 9 |
| c_myalgia_dx | 0 |
| c_dnr | 6 |

## Same signals among `any` ASCVD admissions ON statin at discharge (n = 55)

| signal | admissions |
|---|---|
| r_statin_harm_dx | 0 |
| r_prior_statin_harm_dx | 0 |
| r_ck_gt10x | 1 |
| r_transaminase_gt3x | 4 |
| r_liver_disease | 0 |
| r_pregnancy | 0 |
| r_died | 3 |
| r_hospice_palliative | 4 |
| r_left_ama | 1 |
| r_transferred_acute | 0 |
| c_esrd_dialysis | 2 |
| c_age_gt75 | 22 |
| c_ldl_lt70 | 8 |
| c_nonstatin_lipid_only | 3 |
| c_switched_to_nonstatin | 0 |
| c_myalgia_dx | 0 |
| c_dnr | 3 |

## Lab availability, `any` ASCVD admissions (n = 89)

|  | admissions |
|---|---|
| ALT measured during stay | 45 |
| CK measured during stay | 31 |
| LDL within 365 days before discharge | 25 |

## Draft category split, `index` cohort (proposal only)

| draft_category | admissions | patients |
|---|---|---|
| on_statin | 22 | 20 |
| gap | 2 | 1 |

## Draft category split, `any` cohort (proposal only)

| draft_category | admissions | patients |
|---|---|---|
| on_statin | 55 | 35 |
| gap | 21 | 8 |
| justified_no_statin | 12 | 10 |
| insufficient_data | 1 | 1 |

## Draft category split, `any` cohort, latest admission per patient

| draft_category | patients |
|---|---|
| on_statin | 31 |
| justified_no_statin | 9 |
| gap | 4 |

`any` ASCVD admissions: median age 67.0, median LOS 5.3 days, 30/89 female.

Discharge location, `any` ASCVD admissions:

| discharge_location | admissions |
|---|---|
| HOME HEALTH CARE | 30 |
| SKILLED NURSING FACILITY | 14 |
| HOME | 13 |
| (missing) | 11 |
| DIED | 10 |
| REHAB | 4 |
| CHRONIC/LONG TERM ACUTE CARE | 3 |
| HOSPICE | 2 |
| AGAINST ADVICE | 2 |
