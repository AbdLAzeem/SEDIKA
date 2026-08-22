"""Centralized project paths.

All scripts derive their I/O locations from this module so the project is
relocatable. Override any directory by setting the matching env var
(e.g. SEDIKA_MODEL_DIR) before running.
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PROCESSED_DIR = os.environ.get("SEDIKA_PROCESSED_DIR", os.path.join(BASE_DIR, "processed_data"))
EXTERNAL_DIR  = os.environ.get("SEDIKA_EXTERNAL_DIR",  os.path.join(BASE_DIR, "processed_external"))
MODEL_DIR     = os.environ.get("SEDIKA_MODEL_DIR",     os.path.join(BASE_DIR, "models"))
OUTPUT_DIR    = os.environ.get("SEDIKA_OUTPUT_DIR",    os.path.join(BASE_DIR, "results"))
PLOT_DIR      = os.environ.get("SEDIKA_PLOT_DIR",      os.path.join(BASE_DIR, "plots"))

RAW_CSV = os.environ.get("SEDIKA_RAW_CSV", os.path.join(BASE_DIR, "RT_IOT2022.csv"))


def ensure_dirs(*dirs):
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d, exist_ok=True)
