"""
dpo.py — Generate candidate responses and build DPO preference pairs.

DPO (Direct Preference Optimization) needs pairs of:
    { prompt, chosen (good response), rejected (bad response) }

Pipeline:
  1. generate() : run each prompt N times to get candidate responses
  2. rank()     : score each candidate (reuse eval.py scoring)
  3. build_pairs(): pick best as chosen, worst as rejected
  4. save()     : write preference pairs to jsonl

Usage:
    python -m pipeline.dpo generate --model vinrei:v1
    python -m pipeline.dpo pairs
"""

import json 
import random 
import sys
from pathlib import Path 

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "ai"))
from vinrei.ollama_client import DEFAULT_MODEL, complete 
from pipeline.eval import score as eval_score