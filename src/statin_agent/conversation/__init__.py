"""Conversational agent that elicits missing or outdated information from a patient.

A simulated patient (an LLM given the patient's true situation) talks to an
interviewer agent that only sees an incomplete or outdated record. After the
interview, the guideline is re-run on the corrected record.
"""
