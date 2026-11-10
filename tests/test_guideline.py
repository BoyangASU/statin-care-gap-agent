from statin_agent.guideline import Facts, classify, pce_10yr_risk


def test_pce_matches_goff_2013_example():
    # Goff et al. 2013 worked example: age 55, TC 213, HDL 50, SBP 120 untreated, non-smoker, no diabetes
    expected = {("female", "white"): 0.021, ("female", "black"): 0.030,
                ("male", "white"): 0.054, ("male", "black"): 0.061}
    for (sex, race), r in expected.items():
        got = pce_10yr_risk(55, sex, race, 213, 50, 120, False, False, False)
        assert abs(got - r) < 0.002, (sex, race, got)


def base(**kw):
    f = dict(age=60, sex="male", race="white", on_statin=False, statin_intolerance=False,
             ascvd=False, diabetes=False, smoker=False, bp_treated=False, hypertension=False,
             ldl=120.0, total_chol=200.0, hdl=50.0, sbp=125.0)
    f.update(kw)
    return Facts(**f)


def test_priority_rules():
    assert classify(base(on_statin=True, ascvd=True))[0] == "on_statin"
    assert classify(base(statin_intolerance=True, ascvd=True))[0] == "contraindicated"
    assert classify(base(ascvd=True, ldl=None))[0] == "indicated"
    assert classify(base(ldl=195.0))[0] == "indicated"
    assert classify(base(diabetes=True, ldl=80.0))[0] == "indicated"
    assert classify(base(diabetes=True, ldl=None))[0] == "insufficient_data"
    assert classify(base(age=78))[0] == "not_indicated"
    assert classify(base(hdl=None))[0] == "insufficient_data"
    assert classify(base(ldl=65.0))[0] == "not_indicated"
