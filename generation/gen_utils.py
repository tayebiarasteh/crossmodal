"""
generation/gen_utils.py
Created on July 20, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""
from typing import List, Optional, Set, Tuple


BASE_TRACE_FIELDS: List[str] = [
    "item_id", "model", "benchmark", "domain", "answer_type",
    "prompt_variant", "seed", "reasoning_setting",
    "raw_output", "cot", "answer_text",
    "pred_answer", "gold_answer", "base_correct",
    "length_tokens", "cot_length_tokens",
    "cot_perplexity",
    "self_consistency_answers",
    "verbalized_confidence",
]

BASE_TRACE_KEY_FIELDS: Tuple[str, ...] = (
    "item_id", "prompt_variant", "seed", "reasoning_setting",
)


_WARNED_DAMAGED: Set[str] = set()


def progress(iterable, desc: Optional[str] = None, total: Optional[int] = None,
             enable: bool = True, leave: bool = True):
    if not enable:
        return iterable
    try:
        from tqdm import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, desc=desc, total=total, leave=leave)
