"""The interviewer agent: sees the (incomplete) record, talks to the patient,
records what it learns with the patient's own words as evidence, escalates
when a clinician is needed, and stops when it has what the decision needs."""
from __future__ import annotations
import json
import re
import time
from dataclasses import replace

from .patient import strip_think
from .scenarios import Scenario, FIELDS

SYSTEM = """You are a care-team assistant calling a patient before their clinician reviews cholesterol
treatment. Your job is to confirm and complete the facts that decide statin eligibility, as a
clinician would during a visit. You see the patient's record, which may be missing or out of date.

Check, one question at a time and in plain everyday language:
1. Whether the patient currently takes a cholesterol medicine (statin), and if they stopped one, when and why.
2. Whether they ever had side effects from a statin (for example muscle aches or weakness).
3. Whether they had a cholesterol test anywhere in the past year that is not in the record; if so, ask for
   the total cholesterol, HDL and LDL numbers.
4. Whether they currently smoke.

Rules:
- Ask one short question per turn with ask_patient. At most 8 questions.
- When the patient tells you something that differs from or adds to the record, call record_finding with the
  field, the value, and the patient's exact words as evidence. Record only what the patient actually said.
- Never give medical advice and never tell the patient to start, stop or change a medicine.
- If the patient reports side effects from a statin or anything a clinician should assess, call
  escalate_to_clinician with a short reason.
- When you have checked all four items, call finish with a one-sentence summary."""

TOOLS = [
    {"name": "get_record", "description": "The patient's record as the clinic currently has it.",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "ask_patient", "description": "Ask the patient one question and get their answer.",
     "parameters": {"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]}},
    {"name": "record_finding", "description": "Record a fact the patient stated that differs from or adds to the record.",
     "parameters": {"type": "object", "properties": {
         "field": {"type": "string", "enum": FIELDS},
         "value": {"type": ["boolean", "number"], "description": "true/false for on_statin, statin_intolerance, smoker; mg/dL for ldl, total_chol, hdl"},
         "evidence": {"type": "string", "description": "the patient's exact words"}},
         "required": ["field", "value", "evidence"]}},
    {"name": "escalate_to_clinician", "description": "Flag something a clinician must assess.",
     "parameters": {"type": "object", "properties": {"reason": {"type": "string"}}, "required": ["reason"]}},
    {"name": "finish", "description": "End the call.",
     "parameters": {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]}},
]


def record_view(s: Scenario) -> dict:
    r = s.record
    labs = ({"total_cholesterol": r.total_chol, "hdl": r.hdl, "ldl": r.ldl} if r.ldl is not None
            else "no lipid panel in the past 12 months")
    return {
        "age": int(r.age), "sex": r.sex,
        "diagnoses": [d for d, on in [("history of heart attack or stroke", r.ascvd), ("type 2 diabetes", r.diabetes),
                                      ("hypertension", r.hypertension)] if on] or ["none relevant"],
        "medications": ([f"{s.statin_name or 'a statin'} (active)"] if r.on_statin else ["no statin on file"])
                       + (["blood pressure medicine (active)"] if r.bp_treated else []),
        "allergies_intolerances": ["statin intolerance"] if r.statin_intolerance else ["none recorded"],
        "latest_lipids": labs,
        "smoking_status_on_file": "current smoker" if r.smoker else "non-smoker",
    }


def apply_findings(record, findings):
    upd = {}
    for f in findings:
        k, v = f.get("field"), f.get("value")
        if k not in FIELDS:
            continue
        try:
            upd[k] = bool(v) if k in ("on_statin", "statin_intolerance", "smoker") else float(v)
        except (TypeError, ValueError):
            continue
    return replace(record, **upd)


def supported(finding, patient_turns) -> bool:
    """A finding is supported if its evidence matches something the patient said
    and, for a number, the number itself was said by the patient."""
    said = " ".join(patient_turns).lower()
    ev = str(finding.get("evidence", "")).lower()
    words = [w for w in re.findall(r"[a-z]+", ev) if len(w) > 2]
    overlap = sum(w in said for w in words) / max(len(words), 1)
    if finding.get("field") in ("ldl", "total_chol", "hdl"):
        try:
            num = f"{float(finding['value']):g}"
        except (TypeError, ValueError, KeyError):
            return False
        return num in re.findall(r"\d+(?:\.\d+)?", said) and overlap >= 0.5
    return overlap >= 0.5


class LLMInterviewer:
    def __init__(self, client, model, temperature=0.7, max_steps=24):
        self.client, self.model, self.temperature, self.max_steps = client, model, temperature, max_steps

    def interview(self, s: Scenario, patient) -> dict:
        patient.start(s)
        tools = [{"type": "function", "function": t} for t in TOOLS]
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": "Start the call. Look at the record first."}]
        transcript, findings, escalations, tokens, t0, done = [], [], [], 0, time.time(), None
        extra = {} if self.temperature is None else {"temperature": self.temperature}
        for _ in range(self.max_steps):
            r = self.client.chat.completions.create(model=self.model, messages=msgs, tools=tools, **extra)
            tokens += r.usage.total_tokens if r.usage else 0
            m = r.choices[0].message
            msgs.append(m.model_dump(exclude_none=True))
            if not m.tool_calls:
                msgs.append({"role": "user", "content": "Use a tool: ask_patient, record_finding, escalate_to_clinician or finish."})
                continue
            for tc in m.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                name = tc.function.name
                if name == "get_record":
                    out = record_view(s)
                elif name == "ask_patient":
                    q = strip_think(args.get("question", ""))
                    if sum(1 for x in transcript if x["role"] == "agent") >= 8:
                        out = {"error": "question limit reached; record findings and finish"}
                    else:
                        a = patient.reply(q)
                        transcript += [{"role": "agent", "text": q}, {"role": "patient", "text": a}]
                        out = {"answer": a}
                elif name == "record_finding":
                    findings.append(args); out = {"ok": True}
                elif name == "escalate_to_clinician":
                    escalations.append(args.get("reason", "")); out = {"ok": True}
                elif name == "finish":
                    done = args.get("summary", ""); out = {"ok": True}
                else:
                    out = {"error": f"unknown tool {name}"}
                msgs.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(out, default=str)})
            if done is not None:
                break
        return {"transcript": transcript, "findings": findings, "escalations": escalations,
                "summary": done, "finished": done is not None, "agent_tokens": tokens,
                "patient_tokens": getattr(patient, "tokens", 0), "seconds": round(time.time() - t0, 1)}


class ScriptedInterviewer:
    """Fixed questions and regex parsing, to test the harness without a model."""

    def interview(self, s: Scenario, patient) -> dict:
        patient.start(s)
        qs = ["Have you had a cholesterol test anywhere in the past year?",
              "Are you taking any cholesterol medicine like a statin, and any side effects?",
              "Do you currently smoke?"]
        transcript, findings, esc = [], [], []
        for q in qs:
            a = patient.reply(q)
            transcript += [{"role": "agent", "text": q}, {"role": "patient", "text": a}]
            al = a.lower()
            nums = dict(re.findall(r"(total cholesterol|hdl|ldl) (\d+)", al))
            for k, f in (("total cholesterol", "total_chol"), ("hdl", "hdl"), ("ldl", "ldl")):
                if k in nums:
                    findings.append({"field": f, "value": float(nums[k]), "evidence": a})
            if "stopped" in al and "ache" not in al:
                findings.append({"field": "on_statin", "value": False, "evidence": a})
            if "ache" in al:
                findings.append({"field": "statin_intolerance", "value": True, "evidence": a})
                esc.append("statin side effects")
            if "started again" in al:
                findings.append({"field": "smoker", "value": True, "evidence": a})
        return {"transcript": transcript, "findings": findings, "escalations": esc, "summary": "scripted",
                "finished": True, "agent_tokens": 0, "patient_tokens": 0, "seconds": 0}