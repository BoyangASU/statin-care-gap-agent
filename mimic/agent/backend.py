"""LLM backend for the MIMIC agent: one tool-use episode for one patient.

Only an OpenAI-compatible backend is provided, meant for a LOCAL server
(vLLM serving Qwen3-8B on Sol). Credentialed MIMIC-IV must never be sent to an
external API; `make_backend` refuses a non-local base URL unless the caller
passes allow_remote=True, which is only acceptable for the public Demo.

The loop is the prototype's OpenAIBackend with the MIMIC tools and prompt, a
larger tool-output limit (MIMIC records are longer), and evidence_codes kept.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from urllib.parse import urlparse

from .prompts import SYSTEM_PROMPT, user_prompt
from .tools import ALL_SPECS, execute


@dataclass
class Episode:
    category: str | None
    facts: dict
    rationale: str = ""
    evidence: str = ""
    evidence_codes: dict = field(default_factory=dict)
    trace: list = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    error: str | None = None


def _short(obj, n=16000):
    s = json.dumps(obj, default=str)
    return s if len(s) <= n else s[:n] + f"... [truncated {len(s) - n} chars; narrow the query with keyword=]"


def _finish(args, trace, t0, tin, tout):
    return Episode(args.get("category"), args.get("facts") or {}, args.get("rationale", ""),
                   args.get("evidence", ""), args.get("evidence_codes") or {}, trace, tin, tout,
                   time.time() - t0)


class OpenAIBackend:
    def __init__(self, model="qwen3-8b", temperature=0.7, max_steps=20, base_url=None):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"))
        self.model, self.temperature, self.max_steps = model, temperature, max_steps
        self.name = f"openai:{model}"

    def run(self, rec, feedback=None) -> Episode:
        tools = [{"type": "function", "function": s} for s in ALL_SPECS]
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt(rec.patient_id, feedback)}]
        trace, t0, tin, tout = [], time.time(), 0, 0
        extra = {} if self.temperature is None else {"temperature": self.temperature}
        for _ in range(self.max_steps):
            r = self.client.chat.completions.create(model=self.model, messages=msgs, tools=tools, **extra)
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


def _is_local(url: str | None) -> bool:
    """localhost, or a bare cluster node name such as sg017 (no dots)."""
    host = (urlparse(url).hostname or "") if url else ""
    return host in ("localhost", "127.0.0.1", "0.0.0.0", "::1") or (host != "" and "." not in host)


def make_backend(kind: str, model: str | None = None, temperature: float | None = 0.7,
                 max_steps: int = 20, base_url: str | None = None, seed: int = 0,
                 allow_remote: bool = False):
    if kind == "simulated":
        from .simulated import SimulatedBackend
        return SimulatedBackend(seed=seed)
    if kind == "openai":
        base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        if not allow_remote and not _is_local(base_url):
            raise SystemExit(f"Refusing base_url={base_url!r}: MIMIC data may only go to a local model "
                             f"server (e.g. http://localhost:8000/v1). Use --allow-remote only for the public Demo.")
        return OpenAIBackend(model or "qwen3-8b", temperature, max_steps, base_url)
    raise ValueError(f"unknown backend {kind!r}")
