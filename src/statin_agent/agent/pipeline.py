"""Decision pipeline for one patient: k samples x (attempt -> verify -> one retry),
self-consistency confidence, and deferral to a clinician.

Each sample keeps its first attempt and its final attempt, so one run yields all
three evaluation configurations (single pass, + self-check, + consistency/deferral).
"""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass, asdict, field

from .verifier import verify


@dataclass
class Sample:
    first_category: str | None
    first_problems: list
    final_category: str | None
    final_problems: list
    retried: bool
    episodes: list = field(default_factory=list)   # serialized Episodes (with traces)


@dataclass
class Decision:
    patient_id: str
    category: str | None
    confidence: float
    verified: bool
    deferred: bool
    samples: list
    input_tokens: int = 0
    output_tokens: int = 0

    def to_dict(self, with_traces=True):
        d = asdict(self)
        if not with_traces:
            for s in d["samples"]:
                s.pop("episodes", None)
        return d


def _ep_dict(ep):
    return {"category": ep.category, "facts": ep.facts, "rationale": ep.rationale,
            "evidence": ep.evidence, "error": ep.error, "trace": ep.trace,
            "input_tokens": ep.input_tokens, "output_tokens": ep.output_tokens,
            "seconds": round(ep.seconds, 2)}


def decide(rec, backend, k=3, tau=0.8, self_check=True, lookback_days=365) -> Decision:
    samples, tin, tout = [], 0, 0
    for _ in range(k):
        ep = backend.run(rec)
        p1 = verify(rec, ep, lookback_days)
        eps = [ep]
        final, p2 = ep, p1
        if self_check and p1:
            final = backend.run(rec, feedback="\n".join(f"- {p}" for p in p1))
            p2 = verify(rec, final, lookback_days)
            eps.append(final)
        for e in eps:
            tin += e.input_tokens; tout += e.output_tokens
        samples.append(Sample(ep.category, p1, final.category, p2, len(eps) > 1,
                              [_ep_dict(e) for e in eps]))
        
    cats = [s.final_category for s in samples]
    top, n = Counter(cats).most_common(1)[0]
    conf = n / k
    verified = all(not s.final_problems for s in samples if s.final_category == top)
    deferred = conf < tau or (self_check and not verified) or top is None
    return Decision(rec.patient_id, top, conf, verified, deferred, samples, tin, tout)
