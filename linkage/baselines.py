"""
linkage/baselines.py
Created on August 8, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""
import hashlib
import json
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from config.serde import read_config
from data_loader.build_paired_pool import load_pool
from data_loader.build_utils import write_manifest
from data_loader.cxr_harmonization import resolve_cxr_image_path
from linkage.conditioning import _age_band, _study_attributes
from linkage.retrieval import _digest, add_tied_at_top
from Inference.resume_utils import MissingInput


def _query_tiebreak(case_id: str, n: int) -> np.ndarray:
    seed = int.from_bytes(hashlib.blake2b(str(case_id).encode(), digest_size=4).digest(), "big")
    return np.random.default_rng(seed).random(n)


def _txrv_label_map(pathologies: List[str], findings: List[str]) -> Dict[int, int]:
    out = {}
    for i, name in enumerate(pathologies):
        key = str(name).strip().lower().replace(" ", "_")
        alias = {"effusion": "pleural_effusion", "lung_opacity": "lung_opacity",
                 "enlarged_cardiomediastinum": "enlarged_cardiomediastinum",
                 "pleural_thickening": "pleural_other", "mass": "lung_lesion",
                 "nodule": "lung_lesion"}.get(key, key)
        if alias in findings:
            out[i] = findings.index(alias)
    return out


def _records_path(cfg: dict, baseline: str, cohort: str, split: str,
                  conditioning: str = "none") -> str:
    d = os.path.join(cfg["paths"]["linkage_dir"], f"baseline_{baseline}", cohort)
    if conditioning == "none":
        return os.path.join(d, f"records_{split}.csv")
    return os.path.join(d, f"records_cond_{conditioning}_{split}.csv")


def _preds_path(cfg: dict, baseline: str, cohort: str, split: str) -> str:
    d = os.path.join(cfg["paths"]["linkage_dir"], f"baseline_{baseline}", cohort)
    return os.path.join(d, f"preds_{split}.npz")


def _txrv_predictions(cfg: dict, cfg_path: str, cohort: str, qmeta, findings: List[str],
                      bcfg: dict, status_dir: str, stage: str, device: str,
                      cache_path: str, force: bool):
    from Inference.rows import append_status
    want = {"checkpoint": str(bcfg.get("checkpoint")), "image_size": int(bcfg.get("image_size", 224)),
            "findings": ",".join(findings), "query_ids_digest": _digest(qmeta["case_id"].astype(str).tolist())}
    if not force and os.path.exists(cache_path):
        z = np.load(cache_path, allow_pickle=False)
        stored = {k: str(z[f"bp__{k}"]) if f"bp__{k}" in z else None for k in want}
        moved = [k for k, v in want.items() if stored.get(k) is not None and stored[k] != str(v)]
        if not moved:
            append_status(status_dir, stage, f"{cohort}: cached TorchXRayVision predictions, skipping")
            return z["preds"]

    import torch
    from PIL import Image
    from encoders.extract_embeddings import source_resolution
    from encoders.image_text_encoders import (encode_images, get_spec, load_model,
                                              txrv_pathologies)
    spec = get_spec("txrv_labels", cfg)
    dev = device if torch.cuda.is_available() else "cpu"
    mdl, proc, _ = load_model("txrv_labels", cfg_path, dev)
    imap = _txrv_label_map(txrv_pathologies(mdl), findings)
    if not imap:
        raise ValueError("no TorchXRayVision pathology maps onto the 14 findings")
    root = cfg["cohorts"][cohort]["image_root"]
    src_res = source_resolution(cfg, cohort)
    batch = int(cfg["extraction"].get("batch_size", 64))
    preds = np.full((len(qmeta), len(findings)), np.nan, dtype=float)
    keys = qmeta["image_key"].astype(str).tolist()
    for lo in tqdm(range(0, len(keys), batch), desc=f"txrv_labels:{cohort}"):
        hi = min(lo + batch, len(keys))
        paths = [resolve_cxr_image_path(dataset=cohort, image_root=root, image_key=k,
                                        resolution=src_res) for k in keys[lo:hi]]
        imgs = [Image.open(p) for p in paths]
        p18 = encode_images(mdl, proc, spec, imgs, dev, torch.float32)
        for dst in sorted(set(imap.values())):
            srcs = [s_ for s_, d_ in imap.items() if d_ == dst]
            preds[lo:hi, dst] = p18[:, srcs].max(axis=1)
    obs = np.isfinite(preds).any(axis=0)
    append_status(status_dir, stage,
                  f"{cohort}: TorchXRayVision covers {int(obs.sum())} of {len(findings)} findings")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    tmp = cache_path + ".tmp.npz"
    np.savez_compressed(tmp, preds=preds, **{f"bp__{k}": str(v) for k, v in want.items()})
    os.replace(tmp, cache_path)
    return preds


def build_baseline_records(cfg_path: str, baseline: str = "txrv_labels",
                           cohorts: Optional[List[str]] = None, split: str = "test",
                           conditioning: str = "none",
                           force: bool = False, device: str = "cuda") -> List[str]:
    params = read_config(cfg_path)
    cfg = params["CrossModal"]
    from Inference.rows import append_status
    status_dir, stage = cfg["paths"]["status_dir"], f"baseline_{baseline}"

    bcfg = cfg["baselines"].get(baseline)
    if bcfg is None:
        raise KeyError(f"baseline {baseline!r} is not in config; available: {sorted(cfg['baselines'])}")
    findings = list(cfg["labels"]["findings"])
    ccfg = cfg["conditioning"]
    width = int(ccfg.get("age_band_years", 5))
    min_pool = int(ccfg.get("min_pool_size", 2))
    cohorts = cohorts or [c for c, cc in cfg["cohorts"].items() if cc.get("enabled", True)]

    if conditioning != "none" and conditioning not in ccfg["sets"]:
        raise KeyError(f"conditioning {conditioning!r} is not in config; "
                       f"available: {sorted(ccfg['sets'])}")
    attrs = list(bcfg.get("attributes") or [])
    if baseline == "metadata_only" and conditioning != "none":
        attrs = list(ccfg["sets"][conditioning])
    cond_attrs = [] if conditioning == "none" or baseline == "metadata_only" else list(ccfg["sets"][conditioning])

    written = []
    for cohort in cohorts:
        if conditioning == "site" and not cfg["cohorts"][cohort].get("site_col"):
            append_status(status_dir, stage, f"{cohort}: no site column, skipping the site conditioning")
            continue
        path = _records_path(cfg, baseline, cohort, split, conditioning)
        meta = path.replace(".csv", ".buildparams.json")

        pool = load_pool(cfg, cohort, usecols=["case_id", "split", "patient_id", "study_id",
                                               "view", "age", "sex", "study_year", "site",
                                               "image_key"] + findings)
        sub = pool[pool["split"] == split].reset_index(drop=True)
        if sub.empty:
            raise MissingInput(f"{cohort}/{split} has no rows")
        studies = sub.drop_duplicates("study_id")["study_id"].astype(str).tolist()

        want = {"baseline": baseline, "cohort": cohort, "split": split,
                "attributes": attrs,
                "match": bcfg.get("match"), "checkpoint": bcfg.get("checkpoint"),
                "age_band_years": width, "findings": findings,
                "conditioning": conditioning, "conditioning_attributes": cond_attrs,
                "min_pool_size": min_pool,
                "n_query": len(sub), "n_corpus": len(studies),
                "query_ids_digest": _digest(sub["case_id"].astype(str).tolist()),
                "corpus_ids_digest": _digest(studies)}
        if not force and os.path.exists(path) and os.path.exists(meta):
            with open(meta) as f:
                stored = json.load(f)
            moved = [k for k, v in want.items() if k in stored and stored[k] != v]
            if not moved:
                append_status(status_dir, stage, f"{cohort}: cached, skipping")
                written.append(path)
                continue

        sid_to_row = {s: i for i, s in enumerate(studies)}
        slab = (sub.drop_duplicates("study_id").set_index("study_id")
                .reindex(studies)[findings].to_numpy(dtype=float))
        sattr = _study_attributes(sub, width).set_index("study_id").reindex(studies).reset_index()
        n_corpus = len(studies)

        qmeta = sub.copy()
        qmeta["age_band"] = qmeta["age"].map(lambda a: _age_band(a, width))
        append_status(status_dir, stage,
                      f"{cohort}: {len(qmeta)} queries against {n_corpus} distinct studies")

        preds = None
        if baseline == "txrv_labels":
            preds = _txrv_predictions(cfg, cfg_path, cohort, qmeta, findings, bcfg,
                                      status_dir, stage, device,
                                      _preds_path(cfg, baseline, cohort, split), force)

        need_views = ("view" in attrs) or ("view" in cond_attrs)
        view_sets = list(sattr["views"]) if need_views else None
        view_masks = ({view: np.fromiter((view in views for views in view_sets),
                                       dtype=bool, count=n_corpus)
                       for view in qmeta["view"].astype(str).unique()}
                      if need_views else {})
        attr_cols = {a: sattr[a].astype(str).to_numpy()
                     for a in set(attrs) | set(cond_attrs) if a != "view"}
        A = na = None
        if baseline == "txrv_labels":
            rule = str(bcfg.get("match", "label_vector_cosine"))
            if rule != "label_vector_cosine":
                raise KeyError(f"baselines.{baseline}.match={rule!r} is configured and not "
                               "implemented; a configured name that nothing implements must "
                               "fail loudly.")
            observed = np.isfinite(preds).any(axis=0)
            A = np.nan_to_num(slab[:, observed], nan=0.0)
            na = np.linalg.norm(A, axis=1) + 1e-9

        rows, dropped, singleton = [], 0, 0
        for j in tqdm(range(len(qmeta)), desc=f"{stage}:{cohort}/{conditioning}:rank"):
            q = qmeta.iloc[j]
            sid = str(q["study_id"])
            tr = sid_to_row.get(sid)
            if tr is None:
                continue
            tie = _query_tiebreak(str(q["case_id"]), n_corpus)
            if baseline == "metadata_only":
                mask = np.ones(n_corpus, dtype=bool)
                for a in attrs:
                    if a == "view":
                        mask &= view_masks[str(q["view"])]
                    else:
                        mask &= (attr_cols[a] == str(q[a]))
                mask[tr] = True
                score = np.where(mask, 1.0, 0.0) + tie * 1e-3
            else:
                pv = preds[j][observed]
                if not np.isfinite(pv).all():
                    continue
                score = (A @ pv) / (na * (np.linalg.norm(pv) + 1e-9))
                score = score + tie * 1e-9

            if cond_attrs:
                cmask = np.ones(n_corpus, dtype=bool)
                for a in cond_attrs:
                    if a == "view":
                        cmask &= view_masks[str(q["view"])]
                    else:
                        cmask &= (attr_cols[a] == str(q[a]))
                cmask[tr] = True
                n_i = int(cmask.sum())
                if n_i <= 1:
                    singleton += 1
                    dropped += 1
                    continue
                if n_i < min_pool:
                    dropped += 1
                    continue
                idx = np.flatnonzero(cmask)
                s = score[idx]
                local_true = int(np.searchsorted(idx, tr))
            else:
                n_i = n_corpus
                s = score
                idx = None
                local_true = tr

            ts = float(s[local_true])
            rank = int((s > ts).sum()) + 1
            order = np.argpartition(-s, min(1, len(s) - 1))[:2]
            order = order[np.argsort(-s[order])]
            top1_row = int(order[0]) if idx is None else int(idx[int(order[0])])
            s1 = float(s[order[0]])
            s2 = float(s[order[1]]) if n_i > 1 else float("nan")
            rows.append({
                "case_id": str(q["case_id"]), "patient_id": str(q["patient_id"]), "study_id": sid,
                "model": f"baseline_{baseline}", "cohort": cohort, "transform": "T0",
                "variant": "baseline", "split": split, "conditioning": conditioning,
                "rank": rank, "true_score": ts, "s1": s1, "s2": s2, "margin": s1 - s2,
                "top1_study_id": studies[top1_row], "n_corpus": n_i,
            })

        if not rows:
            raise MissingInput(f"{cohort}: the {baseline} baseline produced no rows")
        out = pd.DataFrame(rows)
        out["correct"] = (out["rank"] == 1).astype(int)
        out["reciprocal_rank"] = 1.0 / out["rank"]
        add_tied_at_top(out)
        write_manifest(out, path)
        with open(meta + ".tmp", "w") as f:
            json.dump(want, f, indent=2, sort_keys=True)
        os.replace(meta + ".tmp", meta)
        counts = path.replace(".csv", ".counts.json")
        with open(counts + ".tmp", "w") as f:
            json.dump({"n_query_total": int(len(qmeta)), "n_kept": int(len(out)),
                       "n_dropped": int(dropped), "n_singleton_pool": int(singleton),
                       "min_pool_size": min_pool}, f, indent=2, sort_keys=True)
        os.replace(counts + ".tmp", counts)
        chance = float(np.mean(1.0 / out["n_corpus"].to_numpy(dtype=float)))
        append_status(status_dir, stage,
                      f"{cohort}/{conditioning}: {len(out)} of {len(qmeta)} queries kept "
                      f"({singleton} uniquely identified), median candidates "
                      f"{int(out['n_corpus'].median())}, Recall@1 "
                      f"{100.0 * out['correct'].mean():.4f}% against a chance of "
                      f"{100.0 * chance:.4f}%")
        written.append(path)
    return written


def load_baseline(cfg: dict, baseline: str, cohort: str, split: str = "test",
                  conditioning: str = "none") -> pd.DataFrame:
    path = _records_path(cfg, baseline, cohort, split, conditioning)
    if not os.path.exists(path):
        raise MissingInput(f"no {baseline} baseline records for {cohort}/{conditioning}. "
                           f"Run main_build_baseline_records(baseline='{baseline}', "
                           f"conditioning='{conditioning}').")
    df = pd.read_csv(path, dtype={"case_id": str, "patient_id": str, "study_id": str,
                                  "top1_study_id": str})
    if "tied_at_top" not in df.columns:
        add_tied_at_top(df)
    return df


def load_baseline_counts(cfg: dict, baseline: str, cohort: str, split: str = "test",
                         conditioning: str = "none") -> dict:
    path = _records_path(cfg, baseline, cohort, split, conditioning).replace(".csv", ".counts.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)
