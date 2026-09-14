"""Shared paths and constants for the TOMATO-Star staging package."""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(os.environ.get("SCIGRAPHIR_ROOT") or Path(__file__).resolve().parents[2])

# A checkout of TOMATO-Star (Yang and Bing, 2026): data/{train,test}.jsonl and the OpenAlex
# domain outputs. EXTERNAL_REPOS is the directory that holds the checkout.
TOMATO_STAR = Path(os.environ.get("EXTERNAL_REPOS") or REPO_ROOT / "external") / "TOMATO-Star"
DATA_DIR = TOMATO_STAR / "data"
TEST_JSONL = DATA_DIR / "test.jsonl"
TRAIN_JSONL = DATA_DIR / "train.jsonl"

DOMAIN_DIR = TOMATO_STAR / "outputs" / "domain_outputs"
SRC_DOMAINS = DOMAIN_DIR / "source_domains.jsonl"
INSP_DOMAINS = DOMAIN_DIR / "inspiration_domains.jsonl"

OUTPUTS = REPO_ROOT / "outputs"
CACHE = OUTPUTS / "caches"
RESULTS = OUTPUTS / "results"
FIGS = OUTPUTS / "figures"

# retrieval eval config (matches the TOMATO-Star analysis)
K_VALUES = [1, 5, 10, 25, 50, 100]
BIOMEDICAL = {"Life Sciences", "Health Sciences"}     # source field is always biomedical
DISTANT = {"Physical Sciences", "Social Sciences"}    # cross-domain gold

for _d in (CACHE, RESULTS, FIGS):
    _d.mkdir(parents=True, exist_ok=True)
