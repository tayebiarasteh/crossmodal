"""
linkage/openset.py
Created on October 8, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""
import json
import os
from typing import Dict, List, Optional

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


def _records_path(cfg: dict, model: str, cohort: str, transform: str, variant: str,
                  split: str, conditioning: str, level: str = "query") -> str:
    d = os.path.join(cfg["paths"]["linkage_dir"], model, cohort)
    stem = f"open_{conditioning}_{variant}_{transform}_{split}"
    return os.path.join(d, stem + (".csv" if level == "query" else f".{level}.csv"))


def _top2(s: np.ndarray):
    order = np.argsort(-s, kind="stable")[:2]
    s1 = float(s[order[0]])
    s2 = float(s[order[1]]) if len(order) > 1 else float("nan")
    return s1, s2, int(order[0])


def _setup(cfg: dict, model: str, cohort: str, conditioning: str, transform: str,
           variant: str, split: str, device: str) -> dict:
    ccfg = cfg["conditioning"]
    if conditioning not in ccfg["sets"]:
        raise KeyError(f"conditioning {conditioning!r} is not in config; "
                       f"available: {sorted(ccfg['sets'])}")
    width = int(ccfg.get("age_band_years", 5))
    emb_model = embedding_key(model, variant)

    pool = load_pool(cfg, cohort, usecols=["case_id", "split", "patient_id", "study_id",
                                           "view", "age", "sex", "study_year", "site"])
    pool_split = pool[pool["split"] == split].reset_index(drop=True)
    c2s = dict(zip(pool["case_id"].astype(str), pool["study_id"].astype(str)))

    img = load_embeddings(cfg, emb_model, cohort, split, "img")
    txt = load_embeddings(cfg, emb_model, cohort, split, "txt", transform)
    studies, C = _dedupe_corpus(txt["case_ids"], np.asarray(txt["embeddings"], np.float32), c2s)
    sattr = _study_attributes(pool_split, width).set_index("study_id").reindex(studies).reset_index()

    q_ids = [str(x) for x in img["case_ids"]]
    qmeta = pool.set_index(pool["case_id"].astype(str)).reindex(q_ids)
    qmeta["age_band"] = qmeta["age"].map(lambda a: _age_band(a, width))
    chunk = max(1, min(int(cfg["instrument"].get("score_chunk_rows", 2048)),
                       int(float(cfg["instrument"].get("score_chunk_max_gb", 4.0))
                           * 1e9 / (4.0 * max(len(studies), 1)))))
    return {"emb_model": emb_model, "attrs": list(ccfg["sets"][conditioning]),
            "min_pool": int(ccfg.get("min_pool_size", 2)),
            "studies": studies, "sid_to_row": {s: i for i, s in enumerate(studies)},
            "sattr": sattr, "c2s": c2s, "q_ids": q_ids, "qmeta": qmeta,
            "Q": torch.from_numpy(np.asarray(img["embeddings"], np.float32)),
            "Cd": torch.from_numpy(np.ascontiguousarray(C)).to(device),
            "pid": qmeta["patient_id"].astype(str).to_numpy(),
            "view": qmeta["view"].astype(str).to_numpy(), "chunk": chunk}


def _ident(s: dict, cohort: str, transform: str, variant: str, split: str,
           conditioning: str) -> dict:
    return {"model": s["emb_model"], "cohort": cohort, "transform": transform,
            "variant": variant, "split": split, "conditioning": conditioning}


def build_openset_records(cfg_path: str, model: str, conditioning: str = "demographics",
                          cohorts: Optional[List[str]] = None, transform: str = "T0",
                          variant: str = "off_shelf", split: str = "test",
                          force: bool = False, device: str = "cuda") -> List[str]:
    params = read_config(cfg_path)
    cfg = params["CrossModal"]
    from Inference.rows import append_status
    status_dir, stage = cfg["paths"]["status_dir"], f"open_{model}"
    cohorts = cohorts or [c for c, cc in cfg["cohorts"].items() if cc.get("enabled", True)]
    device = device if torch.cuda.is_available() else "cpu"

    written = []
    for cohort in cohorts:
        s = _setup(cfg, model, cohort, conditioning, transform, variant, split, device)
        path = _records_path(cfg, s["emb_model"], cohort, transform, variant, split, conditioning)
        want = {"model": s["emb_model"], "cohort": cohort, "transform": transform,
                "variant": variant, "split": split, "conditioning": conditioning,
                "attributes": s["attrs"], "min_pool_size": s["min_pool"],
                "n_corpus_full": len(s["studies"]), "emits": "closed_and_open_halves",
                "query_ids_digest": _digest(s["q_ids"]),
                "corpus_ids_digest": _digest(s["studies"])}
        if not force and records_are_current(path, want):
            append_status(status_dir, stage, f"{cohort}/{conditioning}: cached, skipping")
            written.append(path)
            del s
            continue

        ident = _ident(s, cohort, transform, variant, split, conditioning)
        append_status(status_dir, stage, f"{cohort}/{conditioning}: {len(s['q_ids'])} queries "
                                        f"against {len(s['studies'])} studies")

        def build_chunk(lo: int, hi: int, s=s, ident=ident) -> List[Dict]:
            S = (s["Q"][lo:hi].to(device) @ s["Cd"].T).cpu().numpy()
            out = []
            for j in range(lo, hi):
                cid = s["q_ids"][j]
                sid = s["c2s"].get(cid)
                row = {"case_id": cid, "patient_id": s["pid"][j], "study_id": sid,
                       "view": s["view"][j], **ident, "kept": 0, "drop_reason": ""}
                tr = s["sid_to_row"].get(sid)
                if tr is None:
                    out.append({**row, "drop_reason": "no_corpus_row"})
                    continue
                mask = _match_mask(s["qmeta"].iloc[j], s["sattr"], s["attrs"])
                mask[tr] = True
                n_i = int(mask.sum())
                if n_i <= 1:
                    out.append({**row, "drop_reason": "singleton_pool"})
                    continue
                if n_i < s["min_pool"]:
                    out.append({**row, "drop_reason": "below_min_pool"})
                    continue
                idx = np.flatnonzero(mask)
                sc = S[j - lo][idx]
                local_true = int(np.searchsorted(idx, tr))
                ts = float(sc[local_true])
                rank = int((sc > ts).sum()) + 1
                s1, s2, best = _top2(sc)

                keep = np.ones(len(sc), dtype=bool)
                keep[local_true] = False
                sc_open, idx_open = sc[keep], idx[keep]
                if len(sc_open) >= 2:
                    s1o, s2o, best_o = _top2(sc_open)
                    top1_open = s["studies"][int(idx_open[best_o])]
                else:
                    s1o = s2o = float("nan")
                    top1_open = ""
                out.append({**row, "kept": 1,
                            "rank": rank, "true_score": ts, "s1": s1, "s2": s2,
                            "margin": s1 - s2, "top1_study_id": s["studies"][int(idx[best])],
                            "n_corpus": n_i, "correct": int(rank == 1),
                            "reciprocal_rank": 1.0 / rank,
                            "s1_open": s1o, "s2_open": s2o, "margin_open": s1o - s2o,
                            "top1_open_study_id": top1_open, "n_corpus_open": n_i - 1,
                            "open_undefined": int(not np.isfinite(s1o))})
            del S
            return add_tied_at_top(pd.DataFrame(out)).to_dict("records")

        evaluated = run_record_shards(path, want, s["q_ids"], s["chunk"], build_chunk,
                                      f"{model}:{cohort}/{conditioning}/open",
                                      lambda m: append_status(status_dir, stage, m))
        min_pool, n_total = s["min_pool"], len(s["q_ids"])
        del s
        if device == "cuda":
            torch.cuda.empty_cache()
        kept, counts = split_kept(evaluated)
        if kept.empty:
            raise ValueError(
                f"{cohort}/{conditioning}: none of the {n_total} queries kept a candidate set of "
                f"at least {min_pool} studies, so the open-set measurement is not defined here.")
        counts["n_open_undefined"] = int(kept["open_undefined"].astype(int).sum())
        write_records(path, kept, want, counts)
        drop_shards(path, want)
        append_status(status_dir, stage,
                      f"{cohort}/{conditioning}: {len(kept)} queries kept, Recall@1 "
                      f"{100.0 * kept['correct'].mean():.3f}% with the true report present, "
                      f"{counts['n_open_undefined']} with no second candidate once it is removed")
        written.append(path)
    return written


def build_openset_study_level(cfg_path: str, model: str, conditioning: str = "demographics",
                              cohorts: Optional[List[str]] = None, transform: str = "T0",
                              variant: str = "off_shelf", split: str = "test",
                              force: bool = False, device: str = "cuda") -> List[str]:
    params = read_config(cfg_path)
    cfg = params["CrossModal"]
    from Inference.rows import append_status
    status_dir, stage = cfg["paths"]["status_dir"], f"open_{model}"
    cohorts = cohorts or [c for c, cc in cfg["cohorts"].items() if cc.get("enabled", True)]
    device = device if torch.cuda.is_available() else "cpu"

    written = []
    for cohort in cohorts:
        s = _setup(cfg, model, cohort, conditioning, transform, variant, split, device)
        path = _records_path(cfg, s["emb_model"], cohort, transform, variant, split,
                             conditioning, level="studylevel")
        groups: Dict[tuple, List[int]] = {}
        for j, cid in enumerate(s["q_ids"]):
            sid = s["c2s"].get(cid)
            if sid is None or s["sid_to_row"].get(sid) is None:
                continue
            groups.setdefault((sid, s["view"][j]), []).append(j)
        keys = [f"{a}|{b}" for (a, b) in sorted(groups) if len(groups[(a, b)]) > 1]
        want = {"model": s["emb_model"], "cohort": cohort, "transform": transform,
                "variant": variant, "split": split, "conditioning": conditioning,
                "attributes": s["attrs"], "min_pool_size": s["min_pool"],
                "n_corpus_full": len(s["studies"]), "n_groups": len(keys),
                "group_key": "study_and_view", "emits": "study_level_mean_score",
                "query_ids_digest": _digest(s["q_ids"]),
                "corpus_ids_digest": _digest(s["studies"])}
        if not keys:
            append_status(status_dir, stage,
                          f"{cohort}/{conditioning}: no study holds more than one image of one "
                          f"view, so there is no study-level attack to measure")
            del s
            continue
        if not force and records_are_current(path, want):
            append_status(status_dir, stage,
                          f"{cohort}/{conditioning}: study level cached, skipping")
            written.append(path)
            del s
            continue

        ident = _ident(s, cohort, transform, variant, split, conditioning)
        append_status(status_dir, stage, f"{cohort}/{conditioning}: study-level attack over "
                                        f"{len(keys)} groups holding more than one image")

        def build_chunk(lo: int, hi: int, s=s, ident=ident, keys=keys,
                        groups=groups) -> List[Dict]:
            out = []
            for key in keys[lo:hi]:
                sid, view = key.split("|", 1)
                rows_j = groups[(sid, view)]
                tr = s["sid_to_row"][sid]
                row = {"case_id": key, "study_id": sid, "view": view,
                       "patient_id": s["pid"][rows_j[0]], **ident,
                       "n_images": len(rows_j), "kept": 0, "drop_reason": ""}
                mask = _match_mask(s["qmeta"].iloc[rows_j[0]], s["sattr"], s["attrs"])
                mask[tr] = True
                n_i = int(mask.sum())
                if n_i <= 1:
                    out.append({**row, "drop_reason": "singleton_pool"})
                    continue
                if n_i < s["min_pool"]:
                    out.append({**row, "drop_reason": "below_min_pool"})
                    continue
                idx = np.flatnonzero(mask)
                S = (s["Q"][rows_j].to(device) @ s["Cd"].T).cpu().numpy()[:, idx]
                sc = S.mean(axis=0)
                local_true = int(np.searchsorted(idx, tr))
                ts = float(sc[local_true])
                rank = int((sc > ts).sum()) + 1
                s1, s2, best = _top2(sc)
                out.append({**row, "kept": 1, "rank": rank, "true_score": ts, "s1": s1,
                            "s2": s2, "margin": s1 - s2,
                            "top1_study_id": s["studies"][int(idx[best])], "n_corpus": n_i,
                            "correct": int(rank == 1), "reciprocal_rank": 1.0 / rank})
                del S
            return add_tied_at_top(pd.DataFrame(out)).to_dict("records")

        evaluated = run_record_shards(path, want, keys, max(1, s["chunk"] // 4), build_chunk,
                                      f"{model}:{cohort}/{conditioning}/studylevel",
                                      lambda m: append_status(status_dir, stage, m))
        del s
        if device == "cuda":
            torch.cuda.empty_cache()
        kept, counts = split_kept(evaluated)
        if kept.empty:
            raise ValueError(f"{cohort}/{conditioning}: no group of several images kept a "
                             f"candidate set, so the study-level attack is not defined here.")
        write_records(path, kept, want, counts)
        drop_shards(path, want)
        append_status(status_dir, stage,
                      f"{cohort}/{conditioning}: study-level Recall@1 "
                      f"{100.0 * kept['correct'].mean():.3f}% over {len(kept)} groups")
        written.append(path)
    return written


def _load(cfg: dict, model: str, cohort: str, conditioning: str, transform: str,
          variant: str, split: str, level: str) -> pd.DataFrame:
    path = _records_path(cfg, embedding_key(model, variant), cohort, transform, variant,
                         split, conditioning, level=level)
    if not os.path.exists(path):
        raise MissingInput(f"no open-set {level} records for {model}/{cohort}/{conditioning}. "
                           f"Run main_build_openset_records(model='{model}', "
                           f"conditioning='{conditioning}').")
    return read_csv_defensively(path, dtype={"case_id": str, "patient_id": str, "study_id": str,
                                             "top1_study_id": str, "top1_open_study_id": str,
                                             "view": str})


def load_openset(cfg: dict, model: str, cohort: str, conditioning: str = "demographics",
                 transform: str = "T0", variant: str = "off_shelf",
                 split: str = "test") -> pd.DataFrame:
    return _load(cfg, model, cohort, conditioning, transform, variant, split, "query")


def load_openset_study_level(cfg: dict, model: str, cohort: str,
                             conditioning: str = "demographics", transform: str = "T0",
                             variant: str = "off_shelf", split: str = "test") -> pd.DataFrame:
    return _load(cfg, model, cohort, conditioning, transform, variant, split, "studylevel")


def load_openset_counts(cfg: dict, model: str, cohort: str, conditioning: str = "demographics",
                        transform: str = "T0", variant: str = "off_shelf",
                        split: str = "test", level: str = "query") -> dict:
    path = _records_path(cfg, embedding_key(model, variant), cohort, transform, variant,
                         split, conditioning, level=level).replace(".csv", ".counts.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)
