"""Simulated patients ("patient actors").

The persona tells an LLM what the patient truly knows, in plain language, and
how to behave: answer only what is asked, in everyday words, without medical
jargon and without volunteering everything at once.
"""
from __future__ import annotations
import re

from .scenarios import Scenario

PATIENT_RULES = """You are role-playing a patient in a short phone call with a member of your care team.
Stay in character. Answer in one to three short sentences, in everyday language.
Only share what you are asked about; do not volunteer your whole history at once.
You do not know medical terms like "ASCVD" or "LDL-C" unless they are on your printout.
If you are asked for a number that you have (for example on a lab printout), read it out exactly.
If you do not know something, say so. Never invent facts beyond what is written below.
Never say that you are an AI or a simulation."""


def persona(s: Scenario) -> str:
    t, d = s.truth, s.detail
    lines = [f"You are a {int(t.age)}-year-old {t.sex}."]
    if t.ascvd:
        lines.append("You had a heart attack or stroke in the past and know your heart health matters.")
    if t.diabetes:
        lines.append("You have type 2 diabetes.")
    if t.hypertension:
        lines.append("You have high blood pressure" + (" and take a pill for it." if t.bp_treated else "."))
    # statin story
    if s.kind == "stopped_statin":
        lines.append(f"You used to take {s.statin_name} for cholesterol, but you stopped about "
                     f"{d['months_ago']} months ago because you {d['reason']}. You have not restarted. "
                     "You had no side effects from it.")
    elif s.kind == "intolerance":
        lines.append(f"About {d['months_ago']} months ago a doctor at another clinic started you on "
                     f"{s.statin_name}. Within a few weeks you had bad aching and weakness in your thigh "
                     "and shoulder muscles, so you stopped it and the aches went away. You are worried "
                     "about taking any cholesterol pill like that again. Nobody here has written this down.")
    elif t.on_statin:
        lines.append(f"You take {s.statin_name or 'a cholesterol pill'} every day and have no side effects.")
    else:
        lines.append("You do not take any cholesterol medicine.")
    # labs
    if s.kind == "outside_labs":
        lines.append(f"About {d['months_ago']} months ago you had a cholesterol test at {d['where']}. "
                     f"You have the printout. It says: total cholesterol {t.total_chol:.0f}, "
                     f"HDL {t.hdl:.0f}, LDL {t.ldl:.0f}. Your regular doctor has not seen it.")
    # smoking
    if s.kind == "now_smoker":
        lines.append(f"You quit smoking years ago but started again about {d['months_ago']} months ago, "
                     "about half a pack a day.")
    elif t.smoker:
        lines.append("You smoke about half a pack of cigarettes a day.")
    else:
        lines.append("You do not smoke.")
    return PATIENT_RULES + "\n\nWhat you know about yourself:\n- " + "\n- ".join(lines)


def strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()


class LLMPatient:
    def __init__(self, client, model, temperature=0.7):
        self.client, self.model, self.temperature = client, model, temperature
        self.tokens = 0

    def start(self, scenario: Scenario):
        self.messages = [{"role": "system", "content": persona(scenario)}]

    def reply(self, question: str) -> str:
        self.messages.append({"role": "user", "content": question})
        extra = {} if self.temperature is None else {"temperature": self.temperature}
        r = self.client.chat.completions.create(model=self.model, messages=self.messages, max_tokens=800, **extra)
        if r.usage:
            self.tokens += r.usage.total_tokens
        text = strip_think(r.choices[0].message.content)
        self.messages.append({"role": "assistant", "content": text})
        return text


class ScriptedPatient:
    """Keyword-based patient for tests without a model."""

    def start(self, scenario: Scenario):
        self.s, self.tokens = scenario, 0

    def reply(self, q: str) -> str:
        s, t, q = self.s, self.s.truth, q.lower()
        if "cholesterol" in q and ("test" in q or "lab" in q or "check" in q):
            if s.kind == "outside_labs":
                return (f"Yes, I had one at {s.detail['where']} a few months ago. The printout says total "
                        f"cholesterol {t.total_chol:.0f}, HDL {t.hdl:.0f}, LDL {t.ldl:.0f}.")
            return "Not that I remember."
        if "smok" in q:
            return "I started again a few months ago, about half a pack a day." if s.kind == "now_smoker" else (
                "Yes, about half a pack a day." if t.smoker else "No, I don't smoke.")
        if "side effect" in q or "muscle" in q or "tried" in q or "medicine" in q or "statin" in q or "pill" in q:
            if s.kind == "stopped_statin":
                return f"I stopped the {s.statin_name} months ago, I {s.detail['reason']}."
            if s.kind == "intolerance":
                return f"I tried {s.statin_name} but my muscles ached badly, so I stopped it."
            return "I take it every day, no problems." if t.on_statin else "I don't take any cholesterol medicine."
        return "I'm not sure."
