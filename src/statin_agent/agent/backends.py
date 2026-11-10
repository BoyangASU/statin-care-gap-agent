"""LLM backends. Each runs one tool-use episode for one patient.

* OpenAIBackend    - OpenAI Chat Completions, or any OpenAI-compatible server
                     (vLLM, Ollama, Together, ...) via base_url.
* AnthropicBackend - Anthropic Messages API.
Both return an Episode with the full trace (tool, arguments, output), which
is what the verifier checks and what you show in a demo.
"""
from __future__ import annotations
import json
import os
import time
from dataclasses import dataclass, field

from ..tools import ALL_SPECS, execute
from .prompts import SYSTEM_PROMPT, user_prompt


@dataclass
class Episode:
    category: str | None
    facts: dict
    rationale: str = ""
    evidence: str = ""
    trace: list = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    error: str | None = None


def _short(obj, n=4000):
    s = json.dumps(obj, default=str)
    return s if len(s) <= n else s[:n] + f'... [truncated {len(s) - n} chars]'


def _finish(args, trace, t0, tin, tout):
    return Episode(args.get("category"), args.get("facts") or {}, args.get("rationale", ""),
                   args.get("evidence", ""), trace, tin, tout, time.time() - t0)


class OpenAIBackend:
    def __init__(self, model="gpt-4o-mini", temperature=0.7, max_steps=16, base_url=None):
        from openai import OpenAI
        kw = {}
        base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        if base_url:
            kw["base_url"] = base_url
            kw["api_key"] = os.environ.get("OPENAI_API_KEY", "EMPTY")
        self.client = OpenAI(**kw)
        self.model, self.temperature, self.max_steps = model, temperature, max_steps
        self.name = f"openai:{model}"

    def run(self, rec, feedback=None) -> Episode:
        tools = [{"type": "function", "function": s} for s in ALL_SPECS]
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt(rec.patient_id, feedback)}]
        trace, t0, tin, tout = [], time.time(), 0, 0
        extra = {} if self.temperature is None else {"temperature": self.temperature}
        for _ in range(self.max_steps):
            r = self.client.chat.completions.create(model=self.model, messages=msgs,
                                                    tools=tools, **extra)
            if r.usage:
                tin += r.usage.prompt_tokens; tout += r.usage.completion_tokens
            m = r.choices[0].message
            msgs.append(m.model_dump(exclude_none=True))
            if not m.tool_calls:
                msgs.append({"role": "user", "content": "Continue with tool calls; finish with submit_decision."})
                continue
            for tc in m.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError as e:
                    args, out = {}, {"error": f"Invalid JSON arguments: {e}"}
                else:
                    if tc.function.name == "submit_decision":
                        return _finish(args, trace, t0, tin, tout)
                    out = execute(rec, tc.function.name, args)
                trace.append({"tool": tc.function.name, "args": args, "output": out})
                msgs.append({"role": "tool", "tool_call_id": tc.id, "content": _short(out)})
        return Episode(None, {}, trace=trace, input_tokens=tin, output_tokens=tout,
                       seconds=time.time() - t0, error="step limit reached")


class AnthropicBackend:
    def __init__(self, model="claude-sonnet-4-5", temperature=0.7, max_steps=16):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model, self.temperature, self.max_steps = model, temperature, max_steps
        self.name = f"anthropic:{model}"

    def run(self, rec, feedback=None) -> Episode:
        tools = [{"name": s["name"], "description": s["description"], "input_schema": s["parameters"]}
                 for s in ALL_SPECS]
        msgs = [{"role": "user", "content": user_prompt(rec.patient_id, feedback)}]
        trace, t0, tin, tout = [], time.time(), 0, 0
        extra = {} if self.temperature is None else {"temperature": self.temperature}
        for _ in range(self.max_steps):
            r = self.client.messages.create(model=self.model, system=SYSTEM_PROMPT, max_tokens=2000,
                                            tools=tools, messages=msgs, **extra)
            tin += r.usage.input_tokens; tout += r.usage.output_tokens
            msgs.append({"role": "assistant", "content": r.content})
            uses = [b for b in r.content if b.type == "tool_use"]
            if not uses:
                msgs.append({"role": "user", "content": "Continue with tool calls; finish with submit_decision."})
                continue
            results = []
            for u in uses:
                if u.name == "submit_decision":
                    return _finish(u.input, trace, t0, tin, tout)
                out = execute(rec, u.name, u.input)
                trace.append({"tool": u.name, "args": u.input, "output": out})
                results.append({"type": "tool_result", "tool_use_id": u.id, "content": _short(out)})
            msgs.append({"role": "user", "content": results})
        return Episode(None, {}, trace=trace, input_tokens=tin, output_tokens=tout,
                       seconds=time.time() - t0, error="step limit reached")


def make_backend(kind: str, model: str | None = None, temperature: float | None = 0.7,
                 max_steps: int = 16, base_url: str | None = None, seed: int = 0):
    if kind == "openai":
        return OpenAIBackend(model or "gpt-4o-mini", temperature, max_steps, base_url)
    if kind == "anthropic":
        return AnthropicBackend(model or "claude-sonnet-4-5", temperature, max_steps)
    if kind == "simulated":
        from .simulated import SimulatedBackend
        return SimulatedBackend(seed=seed)
    raise ValueError(f"unknown backend {kind!r}")
