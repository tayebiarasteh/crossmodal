"""
mitigation/utility.py
Created on August 8, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from analysis._common import enabled_cohorts, run_analysis, unit_key
from config.serde import read_config
from data_loader.build_paired_pool import load_pool
from encoders.extract_embeddings import load_embeddings
from Inference import rows as R
from Inference.resume_utils import MissingInput


def _model_key(cfg: dict, variant: str) -> str:
    backbone = cfg["heads"]["backbone"]
    return backbone if variant == "off_shelf" else f"{backbone}_{variant}"


def _labels(cfg: dict, cohort: str, split: str, case_ids: List[str]) -> pd.DataFrame:
    findings = list(cfg["labels"]["findings"])
    pool = load_pool(cfg, cohort, split=split,
                     usecols=["case_id", "patient_id", "study_id"] + findings)
    pool = pool.set_index(pool["case_id"].astype(str)).reindex(case_ids)
    missing = int(pool["patient_id"].isna().sum())
    if missing:
        raise MissingInput(f"{cohort}/{split}: {missing} embedded cases are absent from the pool")
    return pool


def _prompt_scores(cfg: dict, cfg_path: str, variant: str, cohort: str, split: str
                   ) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
    from encoders.image_text_encoders import encode_texts, get_spec, load_model
    findings = list(cfg["labels"]["findings"])
    ucfg = cfg["utility"]["zero_shot"]
    model_key = _model_key(cfg, variant)
    img = load_embeddings(cfg, model_key, cohort, split, "img")
    Z = np.asarray(img["embeddings"], dtype=np.float32)

    backbone = cfg["heads"]["backbone"]
    spec = get_spec(backbone, cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mdl, proc, _ = load_model(backbone, cfg_path, device)
    pos = [ucfg["prompt_positive"].format(finding=f.replace("_", " ")) for f in findings]
    neg = [ucfg["prompt_negative"].format(finding=f.replace("_", " ")) for f in findings]
    Ep = encode_texts(mdl, proc, spec, pos, device, torch.float32)
    En = encode_texts(mdl, proc, spec, neg, device, torch.float32)

    if variant != "off_shelf":
        from mitigation.train_heads import load_heads
        heads, _, _ = load_heads(cfg, variant)
        heads = heads.to(device).eval()
        with torch.no_grad():
            Ep = heads.encode_text(torch.from_numpy(Ep).float().to(device)).cpu().numpy()
            En = heads.encode_text(torch.from_numpy(En).float().to(device)).cpu().numpy()

    if Z.shape[1] != Ep.shape[1]:
        raise ValueError(
            f"the image embeddings are {Z.shape[1]}-d and the projected prompts are "
            f"{Ep.shape[1]}-d, so the two sides of the cosine are not in one space. The zero-shot "
            "readout would still produce numbers, and they would mean nothing.")
    scores = Z @ Ep.T - Z @ En.T
    return scores, img["case_ids"], _labels(cfg, cohort, split, img["case_ids"])


def run_zero_shot(cfg_path: str, variant: str = "off_shelf", force: bool = False) -> pd.DataFrame:
    params = read_config(cfg_path)
    cfg = params["CrossModal"]
    findings = list(cfg["labels"]["findings"])
    units = [unit_key(variant=variant, cohort=c) for c in enabled_cohorts(cfg)]

    def compute(unit: str) -> List[Dict]:
        p = dict(x.split("-", 1) for x in unit.split("__"))
        cohort = p["cohort"]
        scores, ids, lab = _prompt_scores(cfg, cfg_path, variant, cohort, "test")
        clusters = lab["patient_id"].astype(str).to_numpy()
        base = {"model": _model_key(cfg, variant), "cohort": cohort, "variant": variant,
                "experiment": "E6", "analysis": "cross_modal_utility",
                "fdr_family": f"E6_zeroshot_{cohort}"}

        rows, aurocs = [], []
        for j, f in enumerate(findings):
            y = pd.to_numeric(lab[f], errors="coerce").to_numpy(dtype=float)
            ok = np.isfinite(y)
            if ok.sum() < 50 or len(np.unique(y[ok])) < 2:
                continue
            row = R.auroc(y[ok], scores[ok, j], name="task_auroc", cluster_ids=clusters[ok],
                          **{**base, "finding": f})
            rows.append(row)
            aurocs.append(row["mean_raw"] if np.isfinite(row.get("mean_raw", np.nan)) else np.nan)

        vals = np.array([a for a in aurocs if np.isfinite(a)], dtype=float)
        if vals.size:
            rows.append(R.metric(vals, name="task_auroc", **{**base, "finding": "macro"}))

        img = load_embeddings(cfg, _model_key(cfg, variant), cohort, "test", "img")
        txt = load_embeddings(cfg, _model_key(cfg, variant), cohort, "test", "txt", "T0")
        Zi = torch.from_numpy(np.asarray(img["embeddings"], np.float32))
        Zt = torch.from_numpy(np.asarray(txt["embeddings"], np.float32))
        first_of_study = np.sort(np.unique(lab["study_id"].astype(str).to_numpy(),
                                           return_index=True)[1])
        n = min(len(first_of_study), int(cfg["utility"].get("t2i_sample", 5000)))
        rng = np.random.RandomState(int(cfg["stats"]["boot_seed"]))
        sel = np.sort(rng.choice(first_of_study, size=n, replace=False))
        S = (Zt[sel] @ Zi[sel].T).numpy()
        hit = (S.argmax(axis=1) == np.arange(n)).astype(float)
        rows.append(R.metric(hit, name="recall_at_1", cluster_ids=clusters[sel],
                             **{**base, "analysis": "text_to_image", "case_set": "t2i"}))
        return rows

    return run_analysis(cfg, f"e6_zeroshot_{variant}", units, compute, force=force)


def run_linear_probe(cfg_path: str, variant: str = "off_shelf", force: bool = False) -> pd.DataFrame:
    params = read_config(cfg_path)
    cfg = params["CrossModal"]
    findings = list(cfg["labels"]["findings"])
    pcfg = cfg["utility"]["linear_probe"]
    scope = [c for c in cfg["utility"].get("probe_cohorts", [cfg["heads"]["train_cohort"]])
             if c in enabled_cohorts(cfg)]
    if not scope:
        raise MissingInput("utility.probe_cohorts names no enabled cohort")
    units = [unit_key(variant=variant, cohort=c) for c in scope]

    def compute(unit: str) -> List[Dict]:
        p = dict(x.split("-", 1) for x in unit.split("__"))
        cohort = p["cohort"]
        model_key = _model_key(cfg, variant)
        sets = {}
        for split in ("train", "val", "test"):
            emb = load_embeddings(cfg, model_key, cohort, split, "img")
            sets[split] = (np.asarray(emb["embeddings"], np.float32),
                           _labels(cfg, cohort, split, emb["case_ids"]))

        device = "cuda" if torch.cuda.is_available() else "cpu"
        base = {"model": model_key, "cohort": cohort, "variant": variant,
                "experiment": "E6", "analysis": "image_side_utility",
                "fdr_family": f"E6_probe_{cohort}"}
        rows, aurocs = [], []

        for f in findings:
            ys = {}
            for split, (X, lab) in sets.items():
                y = pd.to_numeric(lab[f], errors="coerce").to_numpy(dtype=float)
                ys[split] = (X[np.isfinite(y)], y[np.isfinite(y)],
                             lab["patient_id"].astype(str).to_numpy()[np.isfinite(y)])
            if min(len(ys[s][1]) for s in ys) < 50 or len(np.unique(ys["train"][1])) < 2:
                continue

            Xtr = torch.from_numpy(ys["train"][0]).to(device)
            ytr = torch.from_numpy(ys["train"][1]).float().to(device)
            Xva = torch.from_numpy(ys["val"][0]).to(device)
            yva = torch.from_numpy(ys["val"][1]).float().to(device)
            lin = torch.nn.Linear(Xtr.shape[1], 1).to(device)
            opt = torch.optim.AdamW(lin.parameters(), lr=float(pcfg["lr"]),
                                    weight_decay=float(pcfg["weight_decay"]))
            lossf = torch.nn.BCEWithLogitsLoss()
            bs = int(pcfg["batch_size"])
            best_state, best_val, patience = None, -1.0, 0

            for _ in range(int(pcfg["num_epochs"])):
                lin.train()
                perm = torch.randperm(len(Xtr), device=device)
                for lo in range(0, len(perm), bs):
                    idx = perm[lo:lo + bs]
                    opt.zero_grad(set_to_none=True)
                    lossf(lin(Xtr[idx]).squeeze(1), ytr[idx]).backward()
                    opt.step()
                lin.eval()
                with torch.no_grad():
                    s = lin(Xva).squeeze(1).cpu().numpy()
                yv = yva.cpu().numpy()
                order = np.argsort(s)
                r = np.empty_like(order, dtype=float); r[order] = np.arange(len(s))
                npos, nneg = float((yv == 1).sum()), float((yv == 0).sum())
                va = ((r[yv == 1].sum() - npos * (npos - 1) / 2) / (npos * nneg)
                      if npos > 0 and nneg > 0 else np.nan)
                if np.isfinite(va) and va > best_val:
                    best_val, patience = va, 0
                    best_state = {k: v.detach().clone() for k, v in lin.state_dict().items()}
                else:
                    patience += 1
                    if patience >= int(pcfg["early_stop_patience"]):
                        break

            if best_state is None:
                continue
            lin.load_state_dict(best_state)
            lin.eval()
            with torch.no_grad():
                st = lin(torch.from_numpy(ys["test"][0]).to(device)).squeeze(1).cpu().numpy()
            row = R.auroc(ys["test"][1], st, name="task_auroc", cluster_ids=ys["test"][2],
                          **{**base, "finding": f})
            rows.append(row)
            aurocs.append(row.get("mean_raw", np.nan))

        vals = np.array([a for a in aurocs if np.isfinite(a)], dtype=float)
        if vals.size:
            rows.append(R.metric(vals, name="task_auroc", **{**base, "finding": "macro"}))
        return rows

    return run_analysis(cfg, f"e6_probe_{variant}", units, compute, force=force)
