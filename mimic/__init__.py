"""MIMIC-IV port of the statin care-gap agent.

Self-contained: everything it needs from the Synthea prototype (guideline rules,
evaluation metrics, pipeline records) is copied into this folder, so the folder
can be used on its own. Run modules from the folder that CONTAINS mimic/:
    python -m mimic.db --hosp-dir ... --db data/mimic_demo.duckdb
"""
