"""Reference clinical logic used as ground truth and as agent tools.

Sources (simplified for a research demo; NOT for clinical use):
  * Goff DC Jr, et al. 2013 ACC/AHA Guideline on the Assessment of
    Cardiovascular Risk (Pooled Cohort Equations, PCE).
  * Grundy SM, et al. 2018 AHA/ACC Guideline on the Management of Blood
    Cholesterol (statin benefit groups).
  * US Preventive Services Task Force (2022). Statin Use for the Primary
    Prevention of Cardiovascular Disease in Adults.
"""
from __future__ import annotations
import math
from dataclasses import dataclass

# Pooled Cohort Equation coefficients (Goff et al., 2013, Table A).
# Keys: ln_age, ln_age_sq, ln_tc, ln_age_ln_tc, ln_hdl, ln_age_ln_hdl,
#       ln_tsbp, ln_age_ln_tsbp, ln_usbp, ln_age_ln_usbp, smoker,
#       ln_age_smoker, diabetes; plus baseline survival s10 and mean.
PCE = {
    ("female", "white"): dict(ln_age=-29.799, ln_age_sq=4.884, ln_tc=13.540,
        ln_age_ln_tc=-3.114, ln_hdl=-13.578, ln_age_ln_hdl=3.149,
        ln_tsbp=2.019, ln_age_ln_tsbp=0.0, ln_usbp=1.957, ln_age_ln_usbp=0.0,
        smoker=7.574, ln_age_smoker=-1.665, diabetes=0.661,
        s10=0.9665, mean=-29.18),
    ("female", "black"): dict(ln_age=17.114, ln_age_sq=0.0, ln_tc=0.940,
        ln_age_ln_tc=0.0, ln_hdl=-18.920, ln_age_ln_hdl=4.475,
        ln_tsbp=29.291, ln_age_ln_tsbp=-6.432, ln_usbp=27.820,
        ln_age_ln_usbp=-6.087, smoker=0.691, ln_age_smoker=0.0,
        diabetes=0.874, s10=0.9533, mean=86.61),
    ("male", "white"): dict(ln_age=12.344, ln_age_sq=0.0, ln_tc=11.853,
        ln_age_ln_tc=-2.664, ln_hdl=-7.990, ln_age_ln_hdl=1.769,
        ln_tsbp=1.797, ln_age_ln_tsbp=0.0, ln_usbp=1.764, ln_age_ln_usbp=0.0,
        smoker=7.837, ln_age_smoker=-1.795, diabetes=0.658,
        s10=0.9144, mean=61.18),
    ("male", "black"): dict(ln_age=2.469, ln_age_sq=0.0, ln_tc=0.302,
        ln_age_ln_tc=0.0, ln_hdl=-0.307, ln_age_ln_hdl=0.0,
        ln_tsbp=1.916, ln_age_ln_tsbp=0.0, ln_usbp=1.809, ln_age_ln_usbp=0.0,
        smoker=0.549, ln_age_smoker=0.0, diabetes=0.645,
        s10=0.8954, mean=19.54),
}


def pce_10yr_risk(age: float, sex: str, race: str, total_chol: float,
                  hdl: float, sbp: float, bp_treated: bool, smoker: bool,
                  diabetes: bool) -> float:
    """10-year ASCVD risk (fraction) from the Pooled Cohort Equations.

    Races other than "black" use the white equations, as recommended by the
    2013 guideline. Valid for ages 40-79.
    """
    race_key = "black" if race == "black" else "white"
    c = PCE[(sex, race_key)]
    la, ltc, lh, ls = math.log(age), math.log(total_chol), math.log(hdl), math.log(sbp)
    s = (c["ln_age"] * la + c["ln_age_sq"] * la * la + c["ln_tc"] * ltc
         + c["ln_age_ln_tc"] * la * ltc + c["ln_hdl"] * lh
         + c["ln_age_ln_hdl"] * la * lh + c["diabetes"] * diabetes
         + (c["smoker"] + c["ln_age_smoker"] * la) * smoker)
    if bp_treated:
        s += c["ln_tsbp"] * ls + c["ln_age_ln_tsbp"] * la * ls
    else:
        s += c["ln_usbp"] * ls + c["ln_age_ln_usbp"] * la * ls
    return 1.0 - c["s10"] ** math.exp(s - c["mean"])


CATEGORIES = ["on_statin", "contraindicated", "indicated", "consider",
              "not_indicated", "insufficient_data"]


@dataclass
class Facts:
    age: float
    sex: str
    race: str
    on_statin: bool
    statin_intolerance: bool
    ascvd: bool
    diabetes: bool
    smoker: bool
    bp_treated: bool
    hypertension: bool
    ldl: float | None
    total_chol: float | None
    hdl: float | None
    sbp: float | None


def classify(f: Facts) -> tuple[str, float | None]:
    """Reference statin decision. Returns (category, PCE risk or None)."""
    if f.on_statin:
        return "on_statin", None
    if f.statin_intolerance:
        return "contraindicated", None
    if f.ascvd and f.age <= 75:
        return "indicated", None                     # secondary prevention
    if f.ldl is not None and f.ldl >= 190 and 20 <= f.age <= 75:
        return "indicated", None                     # severe hypercholesterolemia
    if f.diabetes and 40 <= f.age <= 75:
        if f.ldl is None:
            return "insufficient_data", None
        if f.ldl >= 70:
            return "indicated", None                 # diabetes, age 40-75
    if not (40 <= f.age <= 75):
        return "not_indicated", None                 # outside primary-prevention scope
    if None in (f.ldl, f.total_chol, f.hdl, f.sbp):
        return "insufficient_data", None
    if f.ldl < 70:
        return "not_indicated", None
    risk = pce_10yr_risk(f.age, f.sex, f.race, f.total_chol, f.hdl, f.sbp,
                         f.bp_treated, f.smoker, f.diabetes)
    has_risk_factor = f.diabetes or f.smoker or f.hypertension or f.ldl >= 130
    if risk >= 0.10 and has_risk_factor:
        return "indicated", risk                     # USPSTF grade B
    if risk >= 0.075:
        return "consider", risk                      # USPSTF C / ACC-AHA intermediate
    return "not_indicated", risk


def is_care_gap(category: str) -> bool:
    """A care gap: statin is indicated, the patient is not on one."""
    return category == "indicated"
