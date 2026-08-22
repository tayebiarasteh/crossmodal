"""
generation/gen_utils.py
Created on July 20, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import json
import os
from typing import Iterable, List, Optional, Set, Tuple


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


def ensure_parent(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def append_jsonl(path: str, record: dict) -> None:
    ensure_parent(path)
    data = (json.dumps(record, ensure_ascii=True) + "\n").encode("ascii")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        while data:
            data = data[os.write(fd, data):]
    finally:
        os.close(fd)


_WARNED_DAMAGED: Set[str] = set()


def read_jsonl(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    records = []
    damaged = 0
    with open(path, "rb") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                records.append(json.loads(raw.decode("utf-8")))
            except (UnicodeDecodeError, json.JSONDecodeError):
                damaged += 1
                continue
    if damaged and path not in _WARNED_DAMAGED:
        _WARNED_DAMAGED.add(path)
        print(f"[gen_utils] {path}: skipped {damaged} damaged line(s) (torn "
              f"append); the stage that owns them will regenerate them.",
              flush=True)
    return records


def write_json_atomic(path: str, obj) -> None:
    ensure_parent(path)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_json_safe(path: str, owner: str = "gen_utils"):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        print(f"[{owner}] unreadable JSON at {path} ({type(e).__name__}); "
              f"it will be rebuilt by the stage that owns it.", flush=True)
        return None


def diagnose_missing_base(traces_dir: str, model_key: str, benchmark: str,
                          variant: str, setting) -> str:
    path = os.path.join(traces_dir, model_key, f"{benchmark}.jsonl")
    if not os.path.exists(path):
        return f"no trace file at {path}; run the trace generation stage"
    n = n_cell = n_correct = n_cot = 0
    for rec in read_jsonl(path):
        n += 1
        if rec.get("prompt_variant") != variant or rec.get("reasoning_setting") != setting:
            continue
        n_cell += 1
        if not rec.get("base_correct"):
            continue
        n_correct += 1
        if (rec.get("cot") or "").strip():
            n_cot += 1
    if n_cell == 0:
        return (f"{n} records in the file but none at (variant={variant}, "
                f"setting={setting}); run the trace generation stage for that condition")
    if n_correct == 0:
        return (f"{n_cell} records at (variant={variant}, setting={setting}), none "
                f"base_correct; there is nothing to intervene on")
    if n_cot == 0:
        return (f"{n_correct} of {n_cell} records are base_correct but EVERY CoT is "
                f"empty, so none is usable. Regenerating the traces cannot fix this: the cell is "
                f"complete and it skips. Run the trace repair stage instead "
                f"(main_repair_traces(cfg, model_names=['{model_key}']), CPU and "
                f"free) to re-derive the CoT from the stored raw_output, then re-run "
                f"this stage")
    return (f"{n_cot} usable records exist but the loader returned none; this should "
            f"not happen, inspect {path}")


def done_keys(path: str, key_fields: Iterable[str]) -> Set[Tuple]:
    kf = tuple(key_fields)
    keys: Set[Tuple] = set()
    for rec in read_jsonl(path):
        keys.add(tuple(rec.get(k) for k in kf))
    return keys


def record_key(record: dict, key_fields: Iterable[str]) -> Tuple:
    return tuple(record.get(k) for k in key_fields)


def reasoning_settings_for(reasoning_mode: str) -> List[Optional[str]]:
    mode = (reasoning_mode or "always_on").lower()
    if mode == "toggle":
        return ["on", "off"]
    if mode == "effort":
        return ["low", "medium", "high"]
    return [None]


def primary_setting(reasoning_mode: str) -> Optional[str]:
    mode = (reasoning_mode or "always_on").lower()
    if mode == "toggle":
        return "on"
    if mode == "effort":
        return "high"
    return None


def progress(iterable, desc: Optional[str] = None, total: Optional[int] = None,
             enable: bool = True, leave: bool = True):
    if not enable:
        return iterable
    try:
        from tqdm import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, desc=desc, total=total, leave=leave)
