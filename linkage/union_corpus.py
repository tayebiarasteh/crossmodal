"""
linkage/union_corpus.py
Created on October 8, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from config.serde import read_config
from data_loader.build_paired_pool import load_pool
from data_loader.build_utils import read_csv_defensively
from encoders.extract_embeddings import load_embeddings
from linkage.conditioning import _age_band, _match_mask, _study_attributes
from linkage.record_shards import (drop_shards, records_are_current, run_record_shards,
                                   split_kept, write_records)
from linkage.retrieval import (_dedupe_corpus, _digest, add_tied_at_top,
                               embedding_key)
from Inference.resume_utils import MissingInput


def _records_path(cfg: dict, model: str, cohort: str, scope: str, conditioning: str,
                  transform: str, variant: str, split: str) -> str:
    d = os.path.join(cfg["paths"]["linkage_dir"], model, cohort)
    return os.path.join(d, f"union_{scope}_{conditioning}_{variant}_{transform}_{split}.csv")


def _corpus(cfg: dict, model_key: str, cohorts: List[str], splits: List[str], transform: str,
            width: int) -> Tuple[List[str], np.ndarray, pd.DataFrame]:
    keys: List[str] = []
    blocks: List[np.ndarray] = []
    attrs: List[pd.DataFrame] = []
    for c in cohorts:
        pool = load_pool(cfg, c, usecols=["case_id", "split", "study_id", "view", "age",
                                          "sex", "study_year", "site"])
        c2s = dict(zip(pool["case_id"].astype(str), pool["study_id"].astype(str)))
        for sp in splits:
            txt = load_embeddings(cfg, model_key, c, sp, "txt", transform)
            sids, E = _dedupe_corpus(txt["case_ids"],
                                     np.asarray(txt["embeddings"], np.float32), c2s)
            if not sids:
                raise MissingInput(f"{c}/{sp}: the text cache holds no study of this split")
            sub = pool[pool["split"] == sp].reset_index(drop=True)
            at = _study_attributes(sub, width).set_index("study_id").reindex(sids).reset_index()
            at["study_id"] = [f"{c}::{s}" for s in sids]
            keys += list(at["study_id"])
            blocks.append(E)
            attrs.append(at)
    if len(set(keys)) != len(keys):
        raise ValueError("the pooled bank holds the same study twice; a study belongs to one "
                         "split, so a duplicate means two splits overlap")
    total = sum(b.shape[0] for b in blocks)
    out = np.empty((total, blocks[0].shape[1]), dtype=np.float32)
    at_row = 0
    for i, b in enumerate(blocks):
        out[at_row:at_row + b.shape[0]] = b
        at_row += b.shape[0]
        blocks[i] = None
    return keys, out, pd.concat(attrs, ignore_index=True)


def build_union_corpus_records(cfg_path: str, model: str, cohort: str, scope: str,
                               corpus_cohorts: Optional[List[str]] = None,
                               corpus_splits: Optional[List[str]] = None,
                               conditioning: str = "demographics", transform: str = "T0",
                               variant: str = "off_shelf", split: str = "test",
                               force: bool = False, device: str = "cuda") -> str:
    params = read_config(cfg_path)
    cfg = params["CrossModal"]
    from Inference.rows import append_status
    status_dir, stage = cfg["paths"]["status_dir"], f"union_{model}"

    pcfg = cfg["pool_scale"]
    if scope not in pcfg["scopes"]:
        raise KeyError(f"pool_scale.scopes has no entry for {scope!r}; "
                       f"it declares {sorted(pcfg['scopes'])}")
    spec = pcfg["scopes"][scope]
    corpus_cohorts = corpus_cohorts or list(spec["bank_cohorts"]) or [cohort]
    corpus_splits = corpus_splits or list(spec["splits"])
    ccfg = cfg["conditioning"]
    if conditioning not in ccfg["sets"]:
        raise KeyError(f"conditioning {conditioning!r} is not in config; "
                       f"available: {sorted(ccfg['sets'])}")
    attrs = list(ccfg["sets"][conditioning])
    if cohort not in corpus_cohorts or split not in corpus_splits:
        raise ValueError(f"the bank of scope {scope!r} covers {corpus_cohorts} over "
                         f"{corpus_splits} and the queries are {cohort}/{split}, so no query's "
                         f"own report is in it; that is the open-set measurement of E9 and not "
                         f"this one")
    for a in attrs:
        for c in corpus_cohorts:
            if a == "study_year" and not cfg["cohorts"][c].get("date_col"):
                raise ValueError(f"{c} publishes no study year, so the {conditioning!r} "
                                 f"conjunction cannot be applied across this bank")
            if a == "site" and not cfg["cohorts"][c].get("site_col"):
                raise ValueError(f"{c} publishes no site, so the {conditioning!r} conjunction "
                                 f"cannot be applied across this bank")

    width = int(ccfg.get("age_band_years", 5))
    min_pool = int(ccfg.get("min_pool_size", 2))
    emb_model = embedding_key(model, variant)
    device = device if torch.cuda.is_available() else "cpu"

    path = _records_path(cfg, emb_model, cohort, scope, conditioning, transform, variant, split)
    keys, C, sattr = _corpus(cfg, emb_model, corpus_cohorts, corpus_splits, transform, width)
    img = load_embeddings(cfg, emb_model, cohort, split, "img")
    q_ids = [str(x) for x in img["case_ids"]]
    want = {"model": emb_model, "cohort": cohort, "split": split, "transform": transform,
            "variant": variant, "conditioning": conditioning, "attributes": attrs,
            "scope": scope, "corpus_cohorts": corpus_cohorts, "corpus_splits": corpus_splits,
            "min_pool_size": min_pool, "age_band_years": width, "n_corpus_full": len(keys),
            "query_ids_digest": _digest(q_ids), "corpus_ids_digest": _digest(keys)}
    if not force and records_are_current(path, want):
        append_status(status_dir, stage, f"{cohort}/{scope}/{conditioning}: cached, skipping")
        return path

    pool = load_pool(cfg, cohort, usecols=["case_id", "split", "patient_id", "study_id",
                                           "view", "age", "sex", "study_year", "site"])
    c2s = dict(zip(pool["case_id"].astype(str), pool["study_id"].astype(str)))
    Q = torch.from_numpy(np.asarray(img["embeddings"], np.float32))
    qmeta = pool.set_index(pool["case_id"].astype(str)).reindex(q_ids)
    qmeta["age_band"] = qmeta["age"].map(lambda a: _age_band(a, width))
    pid = qmeta["patient_id"].astype(str).to_numpy()
    sid_to_row = {s: i for i, s in enumerate(keys)}
    Cd = torch.from_numpy(np.ascontiguousarray(C)).to(device)
    chunk = max(1, min(int(cfg["instrument"].get("score_chunk_rows", 2048)),
                       int(float(cfg["instrument"].get("score_chunk_max_gb", 4.0))
                           * 1e9 / (4.0 * max(len(keys), 1)))))
    ident = {"model": emb_model, "cohort": cohort, "transform": transform, "variant": variant,
             "split": split, "conditioning": conditioning, "corpus_scope": scope}
    append_status(status_dir, stage,
                  f"{cohort}/{scope}/{conditioning}: {len(q_ids)} queries against {len(keys)} "
                  f"studies from {corpus_cohorts} over {corpus_splits}")

    def build_chunk(lo: int, hi: int, Cd=Cd) -> List[Dict]:
        S = (Q[lo:hi].to(device) @ Cd.T).cpu().numpy()
        out = []
        for j in range(lo, hi):
            cid = q_ids[j]
            sid = c2s.get(cid)
            row = {"case_id": cid, "patient_id": pid[j], "study_id": f"{cohort}::{sid}",
                   **ident, "kept": 0, "drop_reason": ""}
            tr = sid_to_row.get(f"{cohort}::{sid}")
            if tr is None:
                out.append({**row, "drop_reason": "no_corpus_row"})
                continue
            mask = _match_mask(qmeta.iloc[j], sattr, attrs)
            mask[tr] = True
            n_i = int(mask.sum())
            if n_i <= 1:
                out.append({**row, "drop_reason": "singleton_pool"})
                continue
            if n_i < min_pool:
                out.append({**row, "drop_reason": "below_min_pool"})
                continue
            idx = np.flatnonzero(mask)
            sc = S[j - lo][idx]
            local_true = int(np.searchsorted(idx, tr))
            ts = float(sc[local_true])
            rank = int((sc > ts).sum()) + 1
            order = np.argsort(-sc, kind="stable")[:2]
            s1 = float(sc[order[0]])
            s2 = float(sc[order[1]]) if n_i > 1 else float("nan")
            out.append({**row, "kept": 1, "rank": rank, "true_score": ts, "s1": s1, "s2": s2,
                        "margin": s1 - s2, "top1_study_id": keys[int(idx[order[0]])],
                        "n_corpus": n_i, "correct": int(rank == 1),
                        "reciprocal_rank": 1.0 / rank})
        del S
        return add_tied_at_top(pd.DataFrame(out)).to_dict("records")

    evaluated = run_record_shards(path, want, q_ids, chunk, build_chunk,
                                  f"{model}:{cohort}/{scope}/{conditioning}",
                                  lambda m: append_status(status_dir, stage, m))
    del build_chunk, Cd
    if device == "cuda":
        torch.cuda.empty_cache()
    kept, counts = split_kept(evaluated)
    if kept.empty:
        raise ValueError(f"{cohort}/{scope}/{conditioning}: no query kept a candidate set of at "
                         f"least {min_pool} studies over a bank of {len(keys)}")
    counts["n_corpus_full"] = len(keys)
    write_records(path, kept, want, counts)
    drop_shards(path, want)
    append_status(status_dir, stage,
                  f"{cohort}/{scope}/{conditioning}: {len(kept)} queries kept, median candidates "
                  f"{int(kept['n_corpus'].median())}, Recall@1 "
                  f"{100.0 * kept['correct'].mean():.3f}%")
    return path


def load_union_corpus(cfg: dict, model: str, cohort: str, scope: str,
                      conditioning: str = "demographics", transform: str = "T0",
                      variant: str = "off_shelf", split: str = "test") -> pd.DataFrame:
    path = _records_path(cfg, embedding_key(model, variant), cohort, scope, conditioning,
                         transform, variant, split)
    if not os.path.exists(path):
        raise MissingInput(f"no pooled-bank records for {model}/{cohort}/{scope}. Run "
                           f"main_build_union_corpus_records(model='{model}', "
                           f"cohort='{cohort}', scope='{scope}').")
    return read_csv_defensively(path, dtype={"case_id": str, "patient_id": str,
                                             "study_id": str, "top1_study_id": str})


def load_union_corpus_counts(cfg: dict, model: str, cohort: str, scope: str,
                             conditioning: str = "demographics", transform: str = "T0",
                             variant: str = "off_shelf", split: str = "test") -> dict:
    path = _records_path(cfg, embedding_key(model, variant), cohort, scope, conditioning,
                         transform, variant, split).replace(".csv", ".counts.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)
