# Conversational agent: eliciting missing information from patients

## Question

Records are incomplete and out of date. Can an agent that talks to the patient, the way a
clinician would during a visit, recover the facts that decide statin eligibility, and does the
corrected record find patients the chart alone would miss?

## Setup

- **Scenarios** (`conversation/scenarios.py`), built from Synthea patients. Each has a true state
  and a record that differs in one way a clinician only learns by asking:

  | Type | The record says | The patient knows | Why it matters |
  |---|---|---|---|
  | `outside_labs` | no lipids in the past year | a recent test elsewhere, with a printout | `insufficient_data` becomes decidable |
  | `stopped_statin` | statin active | stopped months ago (no side effects) | a hidden care gap |
  | `intolerance` | no intolerance, statin indicated | muscle pain on a statin, stopped it | avoid recommending a drug that failed; escalate |
  | `now_smoker` | non-smoker | smoking again | risk may cross a threshold |
  | `control` | correct | the same | the agent should change nothing |

  Scenarios are kept only when the difference changes the guideline category (except controls).
- **Simulated patient** (`conversation/patient.py`): an LLM given the true state in plain
  language, told to answer briefly, in everyday words, only what is asked, and never to invent.
- **Interviewer** (`conversation/interviewer.py`): sees the record, asks one question at a time
  (at most 8), records each new fact with the patient's exact words as evidence, escalates to a
  clinician when it should, and finishes. It gives no medical advice.
- **Scoring** (`conversation/evaluate.py`): the guideline is re-run on the record corrected with
  the agent's findings and compared with the truth.

## Metrics

- Category correct before vs after the interview.
- Hidden care gaps (truth `indicated`, record not) and how many the interview recovers.
- Fields fixed, and wrong changes to fields that were already right (controls test this).
- **Unsupported findings**: a recorded fact whose evidence does not match what the patient said,
  or a number the patient never said. This is the conversational version of the verifier.
- Escalation on intolerance cases, and escalations without cause.
- Questions asked, tokens.

## Limitations

- An LLM patient is more cooperative and articulate than real patients; real conversations
  include vague answers, wrong recollections and refusals.
- One fact is wrong per scenario; real records can be wrong in several ways at once.
- The scripted backend (`--backend scripted`) only tests the harness.

## Running

```bash
python scripts/08_interviews.py --backend scripted --n 4          # harness check
python scripts/08_interviews.py --base-url http://localhost:8000/v1 --model qwen3-8b --n 10 --workers 8
python scripts/08_interviews.py --run-name interviews_openai_qwen3-8b_n10 --show <scenario id>
```
