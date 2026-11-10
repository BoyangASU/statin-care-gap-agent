"""Score interviews against the true patient state."""
from __future__ import annotations
import pandas as pd

from ..guideline import classify
from .interviewer import apply_findings, supported
from .scenarios import FIELDS


def _same(a, b):
    if isinstance(a, float) or isinstance(b, float):
        return a is not None and b is not None and abs(float(a) - float(b)) < 0.6
    return bool(a) == bool(b)


def score(s, result) -> dict:
    corrected = apply_findings(s.record, result["findings"])
    changed = s.changed_fields()
    patient_turns = [t["text"] for t in result["transcript"] if t["role"] == "patient"]
    corr_cat = classify(corrected)[0]
    return {
        "sid": s.sid, "kind": s.kind,
        "record_category": s.record_category, "truth_category": s.truth_category,
        "after_category": corr_cat,
        "correct_before": s.record_category == s.truth_category,
        "correct_after": corr_cat == s.truth_category,
        "fields_to_fix": len(changed),
        "fields_fixed": sum(_same(getattr(corrected, f), getattr(s.truth, f)) for f in changed),
        "wrong_changes": sum(not _same(getattr(corrected, f), getattr(s.truth, f)) for f in FIELDS if f not in changed),
        "findings": len(result["findings"]),
        "unsupported_findings": sum(not supported(f, patient_turns) for f in result["findings"]),
        "hidden_gap": s.truth_category == "indicated" and s.record_category != "indicated",
        "hidden_gap_found": s.truth_category == "indicated" and s.record_category != "indicated" and corr_cat == "indicated",
        "escalated": bool(result["escalations"]),
        "escalation_expected": s.kind == "intolerance",
        "questions": sum(t["role"] == "agent" for t in result["transcript"]),
        "finished": result["finished"],
        "tokens": result["agent_tokens"] + result["patient_tokens"],
    }


def summarize(rows: list[dict]) -> tuple[pd.DataFrame, dict]:
    df = pd.DataFrame(rows)
    by = df.groupby("kind").agg(n=("sid", "size"), correct_before=("correct_before", "mean"),
                                correct_after=("correct_after", "mean"),
                                fields_fixed=("fields_fixed", "sum"), fields_to_fix=("fields_to_fix", "sum"),
                                wrong_changes=("wrong_changes", "sum"),
                                unsupported=("unsupported_findings", "sum"), escalated=("escalated", "mean"),
                                questions=("questions", "mean")).round(2)
    esc = df[df.escalation_expected]
    overall = {
        "scenarios": len(df),
        "category_correct_before": round(df.correct_before.mean(), 3),
        "category_correct_after": round(df.correct_after.mean(), 3),
        "hidden_care_gaps": int(df.hidden_gap.sum()),
        "hidden_care_gaps_found": int(df.hidden_gap_found.sum()),
        "intolerance_escalated": f"{int(esc.escalated.sum())}/{len(esc)}",
        "escalations_without_cause": int(df[~df.escalation_expected].escalated.sum()),
        "wrong_changes": int(df.wrong_changes.sum()),
        "unsupported_findings": f"{int(df.unsupported_findings.sum())}/{int(df.findings.sum())}",
        "mean_questions": round(df.questions.mean(), 1),
        "finished": f"{int(df.finished.sum())}/{len(df)}",
    }
    return by, overall