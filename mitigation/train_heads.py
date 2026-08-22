"""
mitigation/train_heads.py
Created on August 8, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import gc
import json
import os
from typing import List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from config.serde import read_config
from data_loader.build_paired_pool import load_pool
from encoders.extract_embeddings import load_embeddings
from Inference.resume_utils import MissingInput
from mitigation.projection_heads import (ProjectionHeads, fixed_bank_infonce,
                                         group_privacy_epsilon, symmetric_infonce)


def _head_dir(cfg: dict, variant: str) -> str:
    d = os.path.join(cfg["paths"]["heads_dir"], variant)
    os.makedirs(d, exist_ok=True)
    return d


def _replace_bn_with_gn(module, max_groups: int = 32):
    import torch.nn as nn
    for name, child in module.named_children():
        if isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            c = child.num_features
            g = max(1, min(int(max_groups), c))
            while c % g:
                g -= 1
            setattr(module, name, nn.GroupNorm(g, c, affine=child.affine))
        else:
            _replace_bn_with_gn(child, max_groups)
    return module


def _build_heads(cfg: dict, image_dim: int, text_dim: int) -> ProjectionHeads:
    hcfg = cfg["heads"]
    return ProjectionHeads(image_dim=image_dim, text_dim=text_dim,
                           out_dim=int(hcfg.get("out_dim", 128)),
                           hidden_dim=int(hcfg.get("hidden_dim", 0)))


def _random_orthogonal(dim: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(int(seed))
    q, r = torch.linalg.qr(torch.randn(dim, dim, generator=g, dtype=torch.float64))
    return (q * torch.sign(torch.diagonal(r))).float()


def _fit_random_disruption(cfg: dict, image_dim: int, text_dim: int,
                           seed: int) -> ProjectionHeads:
    if image_dim != text_dim:
        raise ValueError(
            f"the disruption control needs one square map per tower, and the caches are "
            f"{image_dim}-d on the image side against {text_dim}-d on the text side. A "
            "non-square map is not orthogonal, so it would change the within-modality geometry "
            "too and stop being the control it is meant to be.")
    heads = ProjectionHeads(image_dim=image_dim, text_dim=text_dim,
                            out_dim=image_dim, hidden_dim=0)
    with torch.no_grad():
        heads.image_projector.weight.copy_(_random_orthogonal(image_dim, seed))
        heads.image_projector.bias.zero_()
        heads.text_projector.weight.copy_(_random_orthogonal(text_dim, seed + 1))
        heads.text_projector.bias.zero_()
    return heads


def load_heads(cfg: dict, variant: str) -> Tuple[ProjectionHeads, dict, str]:
    d = _head_dir(cfg, variant)
    best_path, done_flag = os.path.join(d, "heads_best.pt"), os.path.join(d, "done.flag")
    if not os.path.exists(best_path) or not os.path.exists(done_flag):
        raise MissingInput(f"the {variant} heads are not trained; "
                           f"run main_train_heads(variant='{variant}')")
    with open(done_flag) as f:
        done_text = f.read()
    meta = json.loads(done_text)
    arch = meta.get("head_arch")
    if not arch:
        raise MissingInput(
            f"the {variant} done flag records no head_arch, so it predates the current head "
            f"definition and the widths cannot be reconstructed. Delete {d} and retrain.")
    ck = torch.load(best_path, map_location="cpu", weights_only=False)
    heads = ProjectionHeads(image_dim=int(arch["image_dim"]), text_dim=int(arch["text_dim"]),
                            out_dim=int(arch["out_dim"]), hidden_dim=int(arch["hidden_dim"]))
    heads.load_state_dict({k.replace("_module.", ""): v for k, v in ck["heads"].items()})
    return heads, meta, done_text


def _cached(cfg: dict, model: str, cohort: str, split: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    img = load_embeddings(cfg, model, cohort, split, "img")
    txt = load_embeddings(cfg, model, cohort, split, "txt", "T0")
    if img["case_ids"] != txt["case_ids"]:
        raise MissingInput(
            f"{cohort}/{split}: the image and text caches are not in the same case order. "
            "Re-extract one of them; a silent misalignment here would train the heads to "
            "match every image to the wrong report.")
    return np.asarray(img["embeddings"]), np.asarray(txt["embeddings"]), img["case_ids"]


def _selective_precision_val(heads: ProjectionHeads, vi: torch.Tensor, vt: torch.Tensor,
                             study_of: np.ndarray, k: int, device: str) -> float:
    heads.eval()
    with torch.no_grad():
        zi, zt = heads(vi.to(device), vt.to(device))
        keep = np.unique(study_of, return_index=True)[1]
        keep_t = torch.from_numpy(np.sort(keep)).to(device)
        C = zt[keep_t]
        studies = study_of[np.sort(keep)]
        row_of = {s: i for i, s in enumerate(studies)}
        true_row = torch.tensor([row_of[s] for s in study_of], device=device)
        S = zi @ C.T
        ts = S.gather(1, true_row.view(-1, 1)).squeeze(1)
        rank = (S > ts.view(-1, 1)).sum(dim=1) + 1
        top2 = torch.topk(S, k=min(2, S.shape[1]), dim=1).values
        margin = (top2[:, 0] - top2[:, 1]) if S.shape[1] > 1 else torch.zeros_like(ts)
        correct = (rank == 1).float()
        kk = min(int(k), len(correct))
        idx = torch.argsort(margin, descending=True)[:kk]
    heads.train()
    return float(correct[idx].mean().item())


def train_heads(cfg_path: str, variant: str = "plain", force: bool = False,
                device: str = "cuda") -> str:
    params = read_config(cfg_path)
    cfg = params["CrossModal"]
    from Inference.rows import append_status
    hcfg = cfg["heads"]
    vcfg = hcfg["variants"].get(variant)
    if vcfg is None:
        raise KeyError(f"head variant {variant!r} is not in config; available: {sorted(hcfg['variants'])}")

    backbone = hcfg["backbone"]
    cohort = hcfg["train_cohort"]
    tau = float(hcfg["temperature"])
    status_dir, stage = cfg["paths"]["status_dir"], f"heads_{variant}"
    out_dir = _head_dir(cfg, variant)
    done_flag = os.path.join(out_dir, "done.flag")
    best_path = os.path.join(out_dir, "heads_best.pt")
    resume_path = os.path.join(out_dir, "resume.pt")

    if os.path.exists(done_flag) and not force:
        append_status(status_dir, stage, "already trained; skipping")
        print(f"[{stage}] already trained. Use force=True to retrain.", flush=True)
        return best_path

    device = device if torch.cuda.is_available() else "cpu"
    ti, tt, t_ids = _cached(cfg, backbone, cohort, "train")
    vi, vt, v_ids = _cached(cfg, backbone, cohort, "val")
    pool = load_pool(cfg, cohort, usecols=["case_id", "study_id", "patient_id"])
    c2s = dict(zip(pool["case_id"].astype(str), pool["study_id"].astype(str)))
    v_studies = np.array([c2s.get(c, c) for c in v_ids])

    if str(vcfg.get("kind", "trained")) == "random_orthogonal":
        heads = _fit_random_disruption(cfg, ti.shape[1], tt.shape[1],
                                       int(cfg["seed"])).to(device)
        sel_k = int(cfg["instrument"]["budgets_k"][1]) if len(cfg["instrument"]["budgets_k"]) > 1 else 100
        val = _selective_precision_val(heads, torch.from_numpy(vi).float(),
                                       torch.from_numpy(vt).float(), v_studies, sel_k, device)
        torch.save({"heads": {k: v.detach().cpu() for k, v in heads.state_dict().items()},
                    "epoch": -1, "val_selective_precision": val, "epsilon": None,
                    "n_trainable": 0,
                    "regime": {"variant": variant, "kind": "random_orthogonal",
                               "private": False, "objective": "none_no_training",
                               "selected_on": "not_selected_nothing_is_trained"}},
                   best_path + ".tmp")
        os.replace(best_path + ".tmp", best_path)
        meta = {"variant": variant, "backbone": backbone, "cohort": cohort,
                "n_trainable": 0, "best_val_selective_precision": val, "best_epoch": -1,
                "epsilon_example_level": None, "delta": None,
                "max_images_per_patient": int(pool.groupby("patient_id").size().max()),
                "epsilon_patient_level_upper_bound": None,
                "objective": "none_no_training",
                "head_arch": {"image_dim": int(ti.shape[1]), "text_dim": int(tt.shape[1]),
                              "hidden_dim": 0, "out_dim": int(ti.shape[1])},
                "epsilon_is_valid": False,
                "note": ("the alignment-disruption control of protocol Section 12f: two independent "
                         "random orthogonal maps, no training, no privacy mechanism. It is the "
                         "frontier a real defense has to beat, never a defense.")}
        with open(done_flag + ".tmp", "w") as f:
            json.dump(meta, f, indent=2, sort_keys=True)
        os.replace(done_flag + ".tmp", done_flag)
        append_status(status_dir, stage,
                      f"disruption control built, no training; validation selective precision "
                      f"{100 * val:.2f}%")
        print(f"[{stage}] {best_path}", flush=True)
        return best_path

    heads = _build_heads(cfg, image_dim=ti.shape[1], text_dim=tt.shape[1])
    if bool(vcfg.get("private", False)):
        _replace_bn_with_gn(heads, int(vcfg.get("gn_num_groups", 32)))
    heads = heads.to(device)
    heads.train()
    n_train = heads.n_trainable()
    append_status(status_dir, stage,
                  f"{n_train:,} trainable head parameters on {ti.shape[1]}-d image and "
                  f"{tt.shape[1]}-d text features, {len(t_ids)} train and {len(v_ids)} val cases")

    Ti = torch.from_numpy(ti).float()
    Tt = torch.from_numpy(tt).float()
    Vi = torch.from_numpy(vi).float()
    Vt = torch.from_numpy(vt).float()

    private = bool(vcfg.get("private", False))
    epochs = int(vcfg["num_epochs"])
    lr = float(vcfg["lr"])
    wd = float(vcfg["weight_decay"])
    sel_k = int(cfg["instrument"]["budgets_k"][1]) if len(cfg["instrument"]["budgets_k"]) > 1 else 100

    bank_i = bank_t = None
    if private and str(vcfg.get("negatives", "fixed_bank")) == "fixed_bank":
        n_bank = min(int(vcfg.get("negative_bank_size", 4096)), len(Vi))
        rng = np.random.RandomState(int(cfg["seed"]))
        sel = np.sort(rng.choice(len(Vi), size=n_bank, replace=False))
        bank_i, bank_t = Vi[sel].to(device), Vt[sel].to(device)
        append_status(status_dir, stage,
                      f"fixed negative bank: {n_bank} held-out cases from the validation split")

    opt = torch.optim.AdamW([p for p in heads.parameters() if p.requires_grad], lr=lr, weight_decay=wd)
    batch = int(vcfg.get("logical_batch_size", vcfg.get("batch_size", 512)))
    ds = torch.utils.data.TensorDataset(Ti, Tt)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True, drop_last=True,
                                         num_workers=int(hcfg.get("num_workers", 0)))
    if len(loader) == 0:
        raise ValueError(
            f"the training split holds {len(Ti)} cases and the logical batch is {batch} with "
            f"drop_last, so the loader yields no batches at all and the heads would be written "
            f"untrained. Lower heads.variants.{variant}.batch_size / logical_batch_size.")

    privacy_engine = None
    if private:
        from opacus import PrivacyEngine
        from opacus.validators import ModuleValidator
        errs = ModuleValidator.validate(heads)
        if errs:
            raise ValueError(f"the heads are not DP-compatible: {errs}")
        privacy_engine = PrivacyEngine()
        heads, opt, loader = privacy_engine.make_private_with_epsilon(
            module=heads, optimizer=opt, data_loader=loader, epochs=epochs,
            target_epsilon=float(vcfg["target_epsilon"]), target_delta=float(vcfg["delta"]),
            max_grad_norm=float(vcfg["max_grad_norm"]))

    start_epoch, best = 0, -1.0
    if os.path.exists(resume_path) and not force:
        ck = torch.load(resume_path, map_location="cpu", weights_only=False)
        stored = ck.get("regime", {})
        want = {"variant": variant, "negatives": vcfg.get("negatives", "in_batch"),
                "private": private, "lr": lr, "epochs": epochs, "batch": batch}
        moved = [k for k, v in want.items() if k in stored and stored[k] != v]
        if moved:
            print(f"[{stage}] REFUSING the resume: the regime moved {moved}. Delete {out_dir} "
                  "to retrain from scratch.", flush=True)
            raise ValueError(f"incompatible checkpoint regime: {moved}")
        if privacy_engine is not None:
            acc_state = ck.get("accountant")
            if acc_state is None:
                raise ValueError(
                    "this checkpoint carries no privacy-accountant state, so the spent budget "
                    "cannot be reconstructed and any epsilon reported after the resume would be "
                    f"too small. Delete {out_dir} and retrain from scratch.")
            privacy_engine.accountant.load_state_dict(acc_state)
        heads.load_state_dict(ck["heads"])
        opt.load_state_dict(ck["opt"])
        start_epoch, best = int(ck["epoch"]) + 1, float(ck.get("best", -1.0))
        del ck
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        append_status(status_dir, stage, f"resumed at epoch {start_epoch}, best {best:.4f}")

    from contextlib import nullcontext
    phys = int(vcfg.get("physical_batch_size", 0) or 0)
    def _epoch_loader():
        if private and phys:
            from opacus.utils.batch_memory_manager import BatchMemoryManager
            return BatchMemoryManager(data_loader=loader, max_physical_batch_size=phys,
                                      optimizer=opt)
        return nullcontext(loader)

    for epoch in range(start_epoch, epochs):
        tot, nb = 0.0, 0
        with _epoch_loader() as epoch_loader:
          for bi_, bt_ in tqdm(epoch_loader, desc=f"{stage} epoch {epoch + 1}/{epochs}"):
            opt.zero_grad(set_to_none=True)
            zi, zt = heads(bi_.to(device), bt_.to(device))
            if private and bank_i is not None:
                with torch.no_grad():
                    bz_i, bz_t = heads(bank_i, bank_t)
                loss = fixed_bank_infonce(zi, zt, bz_t, bz_i, tau).mean()
            else:
                loss = symmetric_infonce(zi, zt, tau)
            loss.backward()
            opt.step()
            tot += float(loss.item()); nb += 1

        val = _selective_precision_val(heads, Vi, Vt, v_studies, sel_k, device)
        eps = privacy_engine.get_epsilon(float(vcfg["delta"])) if privacy_engine else None
        msg = (f"epoch {epoch + 1}/{epochs} loss {tot / max(nb, 1):.4f} "
               f"val selective precision@{sel_k} {100 * val:.2f}%")
        if eps is not None:
            msg += f" (epsilon {eps:.3f})"
        append_status(status_dir, stage, msg)

        if str(vcfg.get("select_on", "valid_selective_precision")) != "valid_selective_precision":
            raise KeyError(f"select_on={vcfg.get('select_on')!r} is configured and not implemented; "
                           "a configured name that nothing implements must fail loudly.")
        better = (val < best or best < 0) if private else (val > best)
        if better:
            best = val
            state = {k: v.detach().cpu() for k, v in heads.state_dict().items()}
            torch.save({"heads": state, "epoch": epoch, "val_selective_precision": val,
                        "epsilon": eps, "n_trainable": n_train,
                        "regime": {"variant": variant, "negatives": vcfg.get("negatives", "in_batch"),
                                   "private": private, "objective": (
                                       "fixed_bank_decomposable" if (private and bank_i is not None)
                                       else "in_batch_not_decomposable"),
                                   "lr": lr, "epochs": epochs, "batch": batch,
                                   "selected_on": "valid_selective_precision"}},
                       best_path + ".tmp")
            os.replace(best_path + ".tmp", best_path)

        torch.save({"heads": heads.state_dict(), "opt": opt.state_dict(), "epoch": epoch,
                    "best": best,
                    "accountant": (privacy_engine.accountant.state_dict()
                                   if privacy_engine is not None else None),
                    "regime": {"variant": variant, "negatives": vcfg.get("negatives", "in_batch"),
                               "private": private, "lr": lr, "epochs": epochs, "batch": batch}},
                   resume_path + ".tmp")
        os.replace(resume_path + ".tmp", resume_path)

    max_per_patient = int(pool.groupby("patient_id").size().max())
    best_epoch = -1
    if os.path.exists(best_path):
        best_epoch = int(torch.load(best_path, map_location="cpu",
                                    weights_only=False).get("epoch", -1))
    meta = {"variant": variant, "backbone": backbone, "cohort": cohort,
            "n_trainable": n_train, "best_val_selective_precision": best,
            "best_epoch": best_epoch,
            "epsilon_example_level": (privacy_engine.get_epsilon(float(vcfg["delta"]))
                                      if privacy_engine else None),
            "delta": float(vcfg["delta"]) if private else None,
            "max_images_per_patient": max_per_patient,
            "epsilon_patient_level_upper_bound": (
                group_privacy_epsilon(privacy_engine.get_epsilon(float(vcfg["delta"])), max_per_patient)
                if (privacy_engine and bool(vcfg.get("report_patient_level", True))) else None),
            "objective": ("fixed_bank_decomposable" if (private and bank_i is not None)
                          else "in_batch_not_decomposable"),
            "head_arch": {"image_dim": int(ti.shape[1]), "text_dim": int(tt.shape[1]),
                          "hidden_dim": int(hcfg.get("hidden_dim", 0)),
                          "out_dim": int(hcfg.get("out_dim", 128))},
            "epsilon_is_valid": bool(private) and bool(vcfg.get("epsilon_is_valid", True))
                                and bank_i is not None}
    with open(done_flag + ".tmp", "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    os.replace(done_flag + ".tmp", done_flag)
    append_status(status_dir, stage, f"done: {json.dumps(meta)}")
    print(f"[{stage}] {best_path}", flush=True)
    return best_path


def reembed_with_heads(cfg_path: str, variant: str = "plain", cohorts: Optional[List[str]] = None,
                       splits: Optional[List[str]] = None, force: bool = False,
                       device: str = "cuda") -> List[str]:
    params = read_config(cfg_path)
    cfg = params["CrossModal"]
    from encoders.extract_embeddings import _build_params, _cache_path, _save_cache
    from Inference.rows import append_status
    hcfg = cfg["heads"]
    backbone = hcfg["backbone"]
    status_dir, stage = cfg["paths"]["status_dir"], f"reembed_{variant}"

    heads, meta, done_text = load_heads(cfg, variant)
    device = device if torch.cuda.is_available() else "cpu"
    heads = heads.to(device).eval()

    cohorts = cohorts or [c for c, cc in cfg["cohorts"].items() if cc.get("enabled", True)]
    spec = dict(cfg["models"][backbone])
    spec["embed_dim"] = int(meta["head_arch"]["out_dim"])

    eval_splits = list(cfg["extraction"].get("splits", ["test"]))
    probe_cohorts = set(cfg["utility"].get("probe_cohorts", [cfg["heads"]["train_cohort"]]))

    def _splits_for(cohort: str) -> List[str]:
        if splits is not None:
            return list(splits)
        if cohort in probe_cohorts:
            return sorted(set(eval_splits) | {"train", "val", "test"})
        return eval_splits

    written = []
    for cohort in cohorts:
        for split in _splits_for(cohort):
            img, txt, ids = _cached(cfg, backbone, cohort, split)
            with torch.no_grad():
                zi, zt = heads(torch.from_numpy(img).float().to(device),
                               torch.from_numpy(txt).float().to(device))
            for tower, arr in (("img", zi), ("txt", zt)):
                path = _cache_path(cfg, f"{backbone}_{variant}", cohort, split, tower,
                                   None if tower == "img" else "T0")
                p = _build_params(cfg, f"{backbone}_{variant}", spec, cohort, split, tower,
                                  None if tower == "img" else "T0")
                p["head_variant"] = variant
                p["head_epoch"] = int(meta.get("best_epoch", -1))
                p["head_done_flag"] = done_text
                _save_cache(path, arr.cpu().numpy(), ids, p)
                written.append(path)
            append_status(status_dir, stage, f"{cohort}/{split}: re-embedded {len(ids)} cases")
    print(f"[{stage}] wrote {len(written)} caches", flush=True)
    return written
