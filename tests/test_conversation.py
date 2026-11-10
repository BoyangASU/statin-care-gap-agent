from statin_agent.conversation.interviewer import supported, apply_findings
from statin_agent.guideline import Facts


def test_supported_requires_patient_words_and_numbers():
    said = ["The printout says total cholesterol 212, HDL 41, LDL 150."]
    assert supported({"field": "ldl", "value": 150, "evidence": "printout says LDL 150"}, said)
    assert not supported({"field": "ldl", "value": 160, "evidence": "printout says LDL 160"}, said)
    assert not supported({"field": "smoker", "value": True, "evidence": "I smoke every day"}, said)


def test_apply_findings_updates_only_known_fields():
    f = Facts(age=60, sex="male", race="white", on_statin=True, statin_intolerance=False, ascvd=False,
              diabetes=False, smoker=False, bp_treated=False, hypertension=False, ldl=None,
              total_chol=None, hdl=None, sbp=130.0)
    g = apply_findings(f, [{"field": "on_statin", "value": False}, {"field": "ldl", "value": "150"},
                           {"field": "age", "value": 20}])
    assert g.on_statin is False and g.ldl == 150.0 and g.age == 60