"""
Created on March 27, 2026.
mitigations_test.py

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh/crossmodal/
"""

import os
import math
import json
import pdb
from pathlib import Path
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor, AutoTokenizer
from PIL import Image
from health_multimodal.image.utils import get_image_inference, ImageModelType
from torch.utils.data import Dataset, DataLoader
from sklearn import metrics

from config.serde import read_config
from data.data_provider import mimic_retrieval_test_data_loader_2D, mimic_retrieval_collate_fn, chexpert_retrieval_test_data_loader_2D, chexpert_retrieval_collate_fn, mimic_biovilt_dp_data_loader_2D, mimic_biovilt_dp_collate_fn, CachedProjectionDataset, cached_projection_collate_fn
from Evaluations_crossmodal import PredictionCrossModalSettingA, PredictionCrossModalSettingA_BioViLT, PredictionCrossModalSettingA_BiomedCLIP, PredictionCrossModalSettingB, PredictionCrossModalSettingB_BiomedCLIP, PredictionCrossModalSettingB_BioViLT


epsilon = 1e-15




def get_crossmodal_label_names():
    return [
        "atelectasis",
        "cardiomegaly",
        "consolidation",
        "edema",
        "enlarged_cardiomediastinum",
        "fracture",
        "lung_lesion",
        "lung_opacity",
        "no_finding",
        "pleural_effusion",
        "pleural_other",
        "pneumonia",
        "pneumothorax",
        "support_devices",
    ]


def get_filtered_split_df_mimic(csv_path, split_name):
    df = pd.read_csv(csv_path)

    df["split"] = df["split"].astype(str).str.strip().str.lower()
    df["view"] = df["view"].astype(str).str.strip().str.upper()

    split_name = str(split_name).lower()
    if split_name not in ["train", "valid", "test"]:
        raise ValueError(f"Unsupported split_name: {split_name}")

    df = df[df["split"] == split_name]

    pa_df = df[df["view"] == "PA"]
    ap_df = df[df["view"] == "AP"]
    df = pd.concat([pa_df, ap_df], ignore_index=True)
    df = df.reset_index(drop=True)

    return df


def build_binary_labels_and_mask_from_df(df, label_names):
    """
    Rule:
        1 -> positive
        0 -> negative
        2 -> negative
        missing -> excluded from loss
    """
    n = len(df)
    c = len(label_names)

    labels = np.zeros((n, c), dtype=np.float32)
    mask = np.zeros((n, c), dtype=np.float32)

    for i, label_name in enumerate(label_names):
        col = df[label_name].values

        for j in range(n):
            val = col[j]

            if pd.isna(val):
                labels[j, i] = 0.0
                mask[j, i] = 0.0
                continue

            try:
                val = int(val)
            except Exception:
                labels[j, i] = 0.0
                mask[j, i] = 0.0
                continue

            if val == 1:
                labels[j, i] = 1.0
                mask[j, i] = 1.0
            elif val == 0 or val == 2:
                labels[j, i] = 0.0
                mask[j, i] = 1.0
            else:
                labels[j, i] = 0.0
                mask[j, i] = 0.0

    return labels, mask



class MimicImageEmbeddingDataset(Dataset):
    """
    MIMIC image-only dataset for utility extraction.
    Uses the exact MIMIC CSV and split filtering.
    """

    def __init__(self, cfg_path, split_name):
        self.cfg_path = cfg_path
        self.params = read_config(cfg_path)
        self.split_name = str(split_name).lower()

        self.csv_path = self.params["CrossModal"]["mimic_master_csv"]
        self.image_root = self.params["CrossModal"]["mimic_image_root"]
        self.label_names = get_crossmodal_label_names()

        self.df = get_filtered_split_df_mimic(self.csv_path, self.split_name)
        self.labels, self.mask = build_binary_labels_and_mask_from_df(self.df, self.label_names)

        image_engine = get_image_inference(ImageModelType.BIOVIL_T)
        self.image_transform = image_engine.transform

    def __len__(self):
        return len(self.df)

    def _resolve_image_path(self, rel_path):
        rel_path = str(rel_path).replace("/files/", "/preprocessed224/")
        return os.path.join(self.image_root, rel_path)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self._resolve_image_path(row["jpg_rel_path"])

        img = Image.open(Path(img_path))
        img_tensor = self.image_transform(img)

        subject_id = row["subject_id"] if "subject_id" in row.index else -1
        study_id = row["study_id"] if "study_id" in row.index else -1

        try:
            subject_id = int(subject_id)
        except Exception:
            subject_id = -1

        try:
            study_id = int(study_id)
        except Exception:
            study_id = -1

        return {
            "row_id": int(idx),
            "subject_id": subject_id,
            "study_id": study_id,
            "image_tensor": img_tensor,
            "label": torch.from_numpy(self.labels[idx]).float(),
            "label_mask": torch.from_numpy(self.mask[idx]).float(),
        }


def mimic_image_embedding_collate_fn(batch):
    return {
        "row_ids": [x["row_id"] for x in batch],
        "subject_ids": [x["subject_id"] for x in batch],
        "study_ids": [x["study_id"] for x in batch],
        "images": torch.stack([x["image_tensor"] for x in batch], dim=0),
        "labels": torch.stack([x["label"] for x in batch], dim=0),
        "label_masks": torch.stack([x["label_mask"] for x in batch], dim=0),
    }





class BioViLTImageEmbeddingExtractor:
    """
    Image-only BioViL-T extractor.
    Supports:
        - off_shelf
        - dp_head
        - any projector-updated variant via checkpoint
    """

    def __init__(self, variant_name="off_shelf", trained_model_path=None, device=None):
        self.variant_name = str(variant_name)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        image_engine = get_image_inference(ImageModelType.BIOVIL_T)
        self.image_model = image_engine.model.to(self.device)
        self.image_model.eval()

        if trained_model_path is not None:
            self._load_projector_weights(trained_model_path)

    def _load_projector_weights(self, trained_model_path):
        ckpt = torch.load(trained_model_path, map_location="cpu", weights_only=False)

        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        else:
            state = ckpt

        projector_state = {}

        # support cached-head DP checkpoints
        for k, v in state.items():
            if k.startswith("image_projector."):
                new_k = k.replace("image_projector.", "")
                projector_state[new_k] = v

        # support full model checkpoints
        for k, v in state.items():
            if k.startswith("image_model.projector."):
                new_k = k.replace("image_model.projector.", "")
                projector_state[new_k] = v

        # fallback if keys are already projector-only
        if len(projector_state) == 0:
            for k, v in state.items():
                if k.startswith("projector."):
                    new_k = k.replace("projector.", "")
                    projector_state[new_k] = v

        if len(projector_state) == 0:
            raise ValueError("Could not find BioViL-T image projector weights in checkpoint.")

        missing, unexpected = self.image_model.projector.load_state_dict(projector_state, strict=False)
        print(f"Loaded projector weights from: {trained_model_path}")
        print("Missing projector keys:", missing)
        print("Unexpected projector keys:", unexpected)

    def extract_embeddings(self, data_loader, use_amp=True):
        z_img_cache = []
        row_ids_cache = []
        subject_ids_cache = []
        study_ids_cache = []
        labels_cache = []
        masks_cache = []

        amp_enabled = bool(use_amp and self.device.type == "cuda")

        with torch.inference_mode():
            for batch in tqdm(data_loader):
                x = batch["images"].to(self.device, non_blocking=True)

                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                    out = self.image_model(x)
                    z_img = out.projected_global_embedding
                    z_img = F.normalize(z_img.float(), p=2, dim=-1)

                z_img_cache.append(z_img.cpu())
                row_ids_cache.extend(batch["row_ids"])
                subject_ids_cache.extend(batch["subject_ids"])
                study_ids_cache.extend(batch["study_ids"])
                labels_cache.append(batch["labels"].cpu())
                masks_cache.append(batch["label_masks"].cpu())

        Z_img = torch.cat(z_img_cache, dim=0).to(torch.float16)
        labels = torch.cat(labels_cache, dim=0).numpy().astype(np.float32)
        masks = torch.cat(masks_cache, dim=0).numpy().astype(np.float32)

        return {
            "Z_img": Z_img,
            "row_ids": np.asarray(row_ids_cache),
            "subject_ids": np.asarray(subject_ids_cache),
            "study_ids": np.asarray(study_ids_cache),
            "labels": labels,
            "label_masks": masks,
        }


def extract_and_save_image_embeddings(
    cfg_path,
    split_name,
    out_path,
    variant_name="off_shelf",
    trained_model_path=None,
    batch_size=256,
    num_workers=8,
    use_amp=True,
):
    dataset = MimicImageEmbeddingDataset(cfg_path=cfg_path, split_name=split_name)

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        collate_fn=mimic_image_embedding_collate_fn,
    )

    extractor = BioViLTImageEmbeddingExtractor(
        variant_name=variant_name,
        trained_model_path=trained_model_path,
    )

    emb_dict = extractor.extract_embeddings(loader, use_amp=use_amp)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(emb_dict, out_path)
    print(f"Saved embeddings to: {out_path}")





class EmbeddingLabelDataset(Dataset):
    def __init__(self, embeddings, labels, label_mask):
        self.embeddings = torch.as_tensor(embeddings, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.float32)
        self.label_mask = torch.as_tensor(label_mask, dtype=torch.float32)

    def __len__(self):
        return self.embeddings.shape[0]

    def __getitem__(self, idx):
        return {
            "embedding": self.embeddings[idx],
            "label": self.labels[idx],
            "label_mask": self.label_mask[idx],
        }


class LinearProbe(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.classifier = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.classifier(x)


def _masked_bce_with_logits(logits, targets, mask):
    loss_raw = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    loss = (loss_raw * mask).sum() / (mask.sum() + epsilon)
    return loss


def _compute_multilabel_metrics_from_logits(logits, targets, mask, threshold=0.5):
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    mask = mask.detach().cpu().numpy()

    num_labels = targets.shape[1]

    per_label_auc = []
    per_label_acc = []
    per_label_sens = []
    per_label_spec = []

    for j in range(num_labels):
        valid_idx = mask[:, j] > 0.5
        y_true = targets[valid_idx, j]
        y_prob = probs[valid_idx, j]

        if len(y_true) == 0:
            per_label_auc.append(np.nan)
            per_label_acc.append(np.nan)
            per_label_sens.append(np.nan)
            per_label_spec.append(np.nan)
            continue

        y_pred = (y_prob >= threshold).astype(np.int32)

        try:
            auc_val = metrics.roc_auc_score(y_true, y_prob)
        except Exception:
            auc_val = np.nan

        try:
            acc_val = metrics.accuracy_score(y_true, y_pred)
        except Exception:
            acc_val = np.nan

        try:
            sens_val = metrics.recall_score(y_true, y_pred, zero_division=0)
        except Exception:
            sens_val = np.nan

        try:
            tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            spec_val = tn / (tn + fp + epsilon)
        except Exception:
            spec_val = np.nan

        per_label_auc.append(auc_val)
        per_label_acc.append(acc_val)
        per_label_sens.append(sens_val)
        per_label_spec.append(spec_val)

    return {
        "macro_auroc": float(np.nanmean(per_label_auc)),
        "macro_accuracy": float(np.nanmean(per_label_acc)),
        "macro_sensitivity": float(np.nanmean(per_label_sens)),
        "macro_specificity": float(np.nanmean(per_label_spec)),
        "per_label_auroc": np.asarray(per_label_auc, dtype=np.float32),
        "per_label_accuracy": np.asarray(per_label_acc, dtype=np.float32),
        "per_label_sensitivity": np.asarray(per_label_sens, dtype=np.float32),
        "per_label_specificity": np.asarray(per_label_spec, dtype=np.float32),
        "probs": probs,
    }


def load_saved_embedding_pack(emb_path):
    pack = torch.load(emb_path, map_location="cpu", weights_only=False)

    if isinstance(pack["Z_img"], torch.Tensor):
        embeddings = pack["Z_img"].cpu().numpy().astype(np.float32)
    else:
        embeddings = np.asarray(pack["Z_img"], dtype=np.float32)

    labels = np.asarray(pack["labels"], dtype=np.float32)
    masks = np.asarray(pack["label_masks"], dtype=np.float32)

    return {
        "embeddings": embeddings,
        "labels": labels,
        "masks": masks,
    }


def train_linear_probe_from_saved_embeddings(
    train_emb_path,
    val_emb_path,
    classifier_save_path,
    batch_size=512,
    lr=1e-3,
    num_epochs=20,
    device=None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    train_pack = load_saved_embedding_pack(train_emb_path)
    val_pack = load_saved_embedding_pack(val_emb_path)

    train_dataset = EmbeddingLabelDataset(
        embeddings=train_pack["embeddings"],
        labels=train_pack["labels"],
        label_mask=train_pack["masks"],
    )
    val_dataset = EmbeddingLabelDataset(
        embeddings=val_pack["embeddings"],
        labels=val_pack["labels"],
        label_mask=val_pack["masks"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        pin_memory=True,
    )

    in_dim = train_pack["embeddings"].shape[1]
    out_dim = train_pack["labels"].shape[1]

    model = LinearProbe(in_dim=in_dim, out_dim=out_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_macro_auc = -np.inf
    best_epoch = -1
    history = []

    os.makedirs(os.path.dirname(classifier_save_path), exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        train_loss_sum = 0.0

        for batch in train_loader:
            x = batch["embedding"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            m = batch["label_mask"].to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(x)
            loss = _masked_bce_with_logits(logits, y, m)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()

        train_loss = train_loss_sum / max(1, len(train_loader))

        model.eval()
        val_logits_all = []
        val_targets_all = []
        val_masks_all = []
        val_loss_sum = 0.0

        with torch.no_grad():
            for batch in val_loader:
                x = batch["embedding"].to(device, non_blocking=True)
                y = batch["label"].to(device, non_blocking=True)
                m = batch["label_mask"].to(device, non_blocking=True)

                logits = model(x)
                loss = _masked_bce_with_logits(logits, y, m)

                val_loss_sum += loss.item()
                val_logits_all.append(logits)
                val_targets_all.append(y)
                val_masks_all.append(m)

        val_logits_all = torch.cat(val_logits_all, dim=0)
        val_targets_all = torch.cat(val_targets_all, dim=0)
        val_masks_all = torch.cat(val_masks_all, dim=0)

        val_metrics = _compute_multilabel_metrics_from_logits(
            logits=val_logits_all,
            targets=val_targets_all,
            mask=val_masks_all,
            threshold=0.5,
        )

        val_loss = val_loss_sum / max(1, len(val_loader))
        val_macro_auc = val_metrics["macro_auroc"]

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_macro_auroc": val_macro_auc,
        })

        print(
            f"Epoch {epoch + 1:02d}/{num_epochs} | "
            f"train loss: {train_loss:.6f} | "
            f"val loss: {val_loss:.6f} | "
            f"val macro AUROC: {val_macro_auc * 100:.4f}%"
        )

        if val_macro_auc > best_val_macro_auc:
            best_val_macro_auc = val_macro_auc
            best_epoch = epoch + 1

            torch.save(
                {
                    "model_state_dict": copy.deepcopy(model.state_dict()),
                    "best_val_macro_auroc": best_val_macro_auc,
                    "best_epoch": best_epoch,
                    "history": history,
                    "in_dim": in_dim,
                    "out_dim": out_dim,
                    "label_names": get_crossmodal_label_names(),
                },
                classifier_save_path,
            )

    print(f"Saved best classifier to: {classifier_save_path}")
    print(f"Best epoch: {best_epoch} | Best val macro AUROC: {best_val_macro_auc * 100:.4f}%")





def test_linear_probe_from_saved_embeddings(
    test_emb_path,
    classifier_ckpt_path,
    batch_size=512,
    device=None,
    save_predictions_csv_path=None,
    save_bootstrap_csv_path=None,
    save_summary_txt_path=None,
    n_bootstrap=1000,
    bootstrap_seed=13,
):
    def _ci95(x):
        lo, hi = np.percentile(x, [2.5, 97.5])
        return float(lo), float(hi)

    def _summary_str_pct(arr):
        arr = np.asarray(arr, dtype=np.float64)
        mean = float(arr.mean()) * 100.0
        std = float(arr.std()) * 100.0
        lo, hi = _ci95(arr)
        lo *= 100.0
        hi *= 100.0
        return f"{mean:.4f} ± {std:.4f} [{lo:.4f}, {hi:.4f}]"

    def _compute_metrics_from_probs_targets_mask(probs, targets, mask, threshold=0.5):
        probs = np.asarray(probs, dtype=np.float32)
        targets = np.asarray(targets, dtype=np.float32)
        mask = np.asarray(mask, dtype=np.float32)

        num_labels = targets.shape[1]

        per_label_auc = []
        per_label_acc = []
        per_label_sens = []
        per_label_spec = []

        for j in range(num_labels):
            valid_idx = mask[:, j] > 0.5
            y_true = targets[valid_idx, j]
            y_prob = probs[valid_idx, j]

            if len(y_true) == 0:
                per_label_auc.append(np.nan)
                per_label_acc.append(np.nan)
                per_label_sens.append(np.nan)
                per_label_spec.append(np.nan)
                continue

            y_pred = (y_prob >= threshold).astype(np.int32)

            try:
                auc_val = metrics.roc_auc_score(y_true, y_prob)
            except Exception:
                auc_val = np.nan

            try:
                acc_val = metrics.accuracy_score(y_true, y_pred)
            except Exception:
                acc_val = np.nan

            try:
                sens_val = metrics.recall_score(y_true, y_pred, zero_division=0)
            except Exception:
                sens_val = np.nan

            try:
                tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
                spec_val = tn / (tn + fp + epsilon)
            except Exception:
                spec_val = np.nan

            per_label_auc.append(auc_val)
            per_label_acc.append(acc_val)
            per_label_sens.append(sens_val)
            per_label_spec.append(spec_val)

        return {
            "macro_auroc": float(np.nanmean(per_label_auc)),
            "macro_accuracy": float(np.nanmean(per_label_acc)),
            "macro_sensitivity": float(np.nanmean(per_label_sens)),
            "macro_specificity": float(np.nanmean(per_label_spec)),
            "per_label_auroc": np.asarray(per_label_auc, dtype=np.float32),
            "per_label_accuracy": np.asarray(per_label_acc, dtype=np.float32),
            "per_label_sensitivity": np.asarray(per_label_sens, dtype=np.float32),
            "per_label_specificity": np.asarray(per_label_spec, dtype=np.float32),
        }

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    test_pack = load_saved_embedding_pack(test_emb_path)
    ckpt = torch.load(classifier_ckpt_path, map_location="cpu", weights_only=False)

    model = LinearProbe(
        in_dim=int(ckpt["in_dim"]),
        out_dim=int(ckpt["out_dim"]),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    test_dataset = EmbeddingLabelDataset(
        embeddings=test_pack["embeddings"],
        labels=test_pack["labels"],
        label_mask=test_pack["masks"],
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        pin_memory=True,
    )

    logits_all = []
    targets_all = []
    masks_all = []

    with torch.no_grad():
        for batch in test_loader:
            x = batch["embedding"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            m = batch["label_mask"].to(device, non_blocking=True)

            logits = model(x)

            logits_all.append(logits.cpu())
            targets_all.append(y.cpu())
            masks_all.append(m.cpu())

    logits_all = torch.cat(logits_all, dim=0)
    targets_all = torch.cat(targets_all, dim=0)
    masks_all = torch.cat(masks_all, dim=0)

    probs_all = torch.sigmoid(logits_all).numpy().astype(np.float32)
    targets_np = targets_all.numpy().astype(np.float32)
    masks_np = masks_all.numpy().astype(np.float32)

    point_metrics = _compute_metrics_from_probs_targets_mask(
        probs=probs_all,
        targets=targets_np,
        mask=masks_np,
        threshold=0.5,
    )

    n_samples = len(probs_all)
    rng = np.random.RandomState(bootstrap_seed)

    macro_auroc_bs = []
    macro_accuracy_bs = []
    macro_sensitivity_bs = []
    macro_specificity_bs = []

    per_label_auroc_bs = []
    per_label_accuracy_bs = []
    per_label_sensitivity_bs = []
    per_label_specificity_bs = []

    for _ in tqdm(range(n_bootstrap)):
        idx = rng.choice(n_samples, n_samples, replace=True)

        bs_metrics = _compute_metrics_from_probs_targets_mask(
            probs=probs_all[idx],
            targets=targets_np[idx],
            mask=masks_np[idx],
            threshold=0.5,
        )

        macro_auroc_bs.append(bs_metrics["macro_auroc"])
        macro_accuracy_bs.append(bs_metrics["macro_accuracy"])
        macro_sensitivity_bs.append(bs_metrics["macro_sensitivity"])
        macro_specificity_bs.append(bs_metrics["macro_specificity"])

        per_label_auroc_bs.append(bs_metrics["per_label_auroc"])
        per_label_accuracy_bs.append(bs_metrics["per_label_accuracy"])
        per_label_sensitivity_bs.append(bs_metrics["per_label_sensitivity"])
        per_label_specificity_bs.append(bs_metrics["per_label_specificity"])

    macro_auroc_bs = np.asarray(macro_auroc_bs, dtype=np.float64)
    macro_accuracy_bs = np.asarray(macro_accuracy_bs, dtype=np.float64)
    macro_sensitivity_bs = np.asarray(macro_sensitivity_bs, dtype=np.float64)
    macro_specificity_bs = np.asarray(macro_specificity_bs, dtype=np.float64)

    per_label_auroc_bs = np.asarray(per_label_auroc_bs, dtype=np.float64)
    per_label_accuracy_bs = np.asarray(per_label_accuracy_bs, dtype=np.float64)
    per_label_sensitivity_bs = np.asarray(per_label_sensitivity_bs, dtype=np.float64)
    per_label_specificity_bs = np.asarray(per_label_specificity_bs, dtype=np.float64)

    label_names = ckpt.get("label_names", get_crossmodal_label_names())

    print("\nBootstrapped test results")
    print(f"Macro AUROC      : {_summary_str_pct(macro_auroc_bs)}")
    print(f"Macro Accuracy   : {_summary_str_pct(macro_accuracy_bs)}")
    print(f"Macro Sensitivity: {_summary_str_pct(macro_sensitivity_bs)}")
    print(f"Macro Specificity: {_summary_str_pct(macro_specificity_bs)}")

    print("\nPer-label AUROC")
    for idx, name in enumerate(label_names):
        arr = per_label_auroc_bs[:, idx]
        arr = arr[np.isfinite(arr)]
        print(f"{name}: {'nan' if len(arr) == 0 else _summary_str_pct(arr)}")

    print("\nPer-label Accuracy")
    for idx, name in enumerate(label_names):
        arr = per_label_accuracy_bs[:, idx]
        arr = arr[np.isfinite(arr)]
        print(f"{name}: {'nan' if len(arr) == 0 else _summary_str_pct(arr)}")

    print("\nPer-label Sensitivity")
    for idx, name in enumerate(label_names):
        arr = per_label_sensitivity_bs[:, idx]
        arr = arr[np.isfinite(arr)]
        print(f"{name}: {'nan' if len(arr) == 0 else _summary_str_pct(arr)}")

    print("\nPer-label Specificity")
    for idx, name in enumerate(label_names):
        arr = per_label_specificity_bs[:, idx]
        arr = arr[np.isfinite(arr)]
        print(f"{name}: {'nan' if len(arr) == 0 else _summary_str_pct(arr)}")

    if save_predictions_csv_path is not None:
        out_df = pd.DataFrame()

        for j in range(probs_all.shape[1]):
            col_name = label_names[j]
            out_df[f"{col_name}_target"] = targets_np[:, j]
            out_df[f"{col_name}_mask"] = masks_np[:, j]
            out_df[f"{col_name}_prob"] = probs_all[:, j]

        os.makedirs(os.path.dirname(save_predictions_csv_path), exist_ok=True)
        out_df.to_csv(save_predictions_csv_path, index=False)
        print(f"Saved predictions to: {save_predictions_csv_path}")

    if save_bootstrap_csv_path is not None:
        boot_df = pd.DataFrame({
            "macro_auroc": macro_auroc_bs,
            "macro_accuracy": macro_accuracy_bs,
            "macro_sensitivity": macro_sensitivity_bs,
            "macro_specificity": macro_specificity_bs,
        })

        for j, name in enumerate(label_names):
            boot_df[f"{name}_auroc"] = per_label_auroc_bs[:, j]
            boot_df[f"{name}_accuracy"] = per_label_accuracy_bs[:, j]
            boot_df[f"{name}_sensitivity"] = per_label_sensitivity_bs[:, j]
            boot_df[f"{name}_specificity"] = per_label_specificity_bs[:, j]

        os.makedirs(os.path.dirname(save_bootstrap_csv_path), exist_ok=True)
        boot_df.to_csv(save_bootstrap_csv_path, index=False)
        print(f"Saved bootstrap metrics to: {save_bootstrap_csv_path}")

    summary_dict = {
        "macro_auroc": _summary_str_pct(macro_auroc_bs),
        "macro_accuracy": _summary_str_pct(macro_accuracy_bs),
        "macro_sensitivity": _summary_str_pct(macro_sensitivity_bs),
        "macro_specificity": _summary_str_pct(macro_specificity_bs),
        "per_label_auroc": {},
        "per_label_accuracy": {},
        "per_label_sensitivity": {},
        "per_label_specificity": {},
    }

    for idx, name in enumerate(label_names):
        for metric_name, metric_arr, summary_key in [
            ("auroc", per_label_auroc_bs[:, idx], "per_label_auroc"),
            ("accuracy", per_label_accuracy_bs[:, idx], "per_label_accuracy"),
            ("sensitivity", per_label_sensitivity_bs[:, idx], "per_label_sensitivity"),
            ("specificity", per_label_specificity_bs[:, idx], "per_label_specificity"),
        ]:
            arr = metric_arr[np.isfinite(metric_arr)]
            summary_dict[summary_key][name] = "nan" if len(arr) == 0 else _summary_str_pct(arr)

    if save_summary_txt_path is None and save_predictions_csv_path is not None:
        base_dir = os.path.dirname(save_predictions_csv_path)
        base_name = os.path.splitext(os.path.basename(save_predictions_csv_path))[0]
        save_summary_txt_path = os.path.join(base_dir, f"{base_name}_bootstrap_summary.txt")

    formatted_summary_csv_path = None
    if save_predictions_csv_path is not None:
        base_dir = os.path.dirname(save_predictions_csv_path)
        base_name = os.path.splitext(os.path.basename(save_predictions_csv_path))[0]
        formatted_summary_csv_path = os.path.join(base_dir, f"{base_name}_bootstrap_summary.csv")

    if save_summary_txt_path is not None:
        msg = (
            f"Bootstrapped test results\n"
            f"Macro AUROC      : {summary_dict['macro_auroc']}\n"
            f"Macro Accuracy   : {summary_dict['macro_accuracy']}\n"
            f"Macro Sensitivity: {summary_dict['macro_sensitivity']}\n"
            f"Macro Specificity: {summary_dict['macro_specificity']}\n\n"
            f"Per-label AUROC\n"
        )
        for name in label_names:
            msg += f"{name}: {summary_dict['per_label_auroc'][name]}\n"

        msg += "\nPer-label Accuracy\n"
        for name in label_names:
            msg += f"{name}: {summary_dict['per_label_accuracy'][name]}\n"

        msg += "\nPer-label Sensitivity\n"
        for name in label_names:
            msg += f"{name}: {summary_dict['per_label_sensitivity'][name]}\n"

        msg += "\nPer-label Specificity\n"
        for name in label_names:
            msg += f"{name}: {summary_dict['per_label_specificity'][name]}\n"

        os.makedirs(os.path.dirname(save_summary_txt_path), exist_ok=True)
        with open(save_summary_txt_path, "w") as f:
            f.write(msg)
        print(f"Saved formatted bootstrap summary to: {save_summary_txt_path}")

    if formatted_summary_csv_path is not None:
        rows = [
            {"metric": "macro_auroc", "summary": summary_dict["macro_auroc"]},
            {"metric": "macro_accuracy", "summary": summary_dict["macro_accuracy"]},
            {"metric": "macro_sensitivity", "summary": summary_dict["macro_sensitivity"]},
            {"metric": "macro_specificity", "summary": summary_dict["macro_specificity"]},
        ]

        for name in label_names:
            rows.append({"metric": f"{name}_auroc", "summary": summary_dict["per_label_auroc"][name]})
            rows.append({"metric": f"{name}_accuracy", "summary": summary_dict["per_label_accuracy"][name]})
            rows.append({"metric": f"{name}_sensitivity", "summary": summary_dict["per_label_sensitivity"][name]})
            rows.append({"metric": f"{name}_specificity", "summary": summary_dict["per_label_specificity"][name]})

        summary_df = pd.DataFrame(rows)
        summary_df.to_csv(formatted_summary_csv_path, index=False)
        print(f"Saved formatted bootstrap summary csv to: {formatted_summary_csv_path}")

    return {
        "point_metrics": point_metrics,
        "bootstrap": {
            "macro_auroc": macro_auroc_bs,
            "macro_accuracy": macro_accuracy_bs,
            "macro_sensitivity": macro_sensitivity_bs,
            "macro_specificity": macro_specificity_bs,
            "per_label_auroc": per_label_auroc_bs,
            "per_label_accuracy": per_label_accuracy_bs,
            "per_label_sensitivity": per_label_sensitivity_bs,
            "per_label_specificity": per_label_specificity_bs,
        },
        "summary": summary_dict,
    }





def extract_and_save_dp_projected_image_embeddings(
    cfg_path,
    split_name,
    initial_feature_path,
    dp_checkpoint_path,
    out_path,
    device=None,
):
    """
    Correct DP utility embedding extraction:
    1) load cached initial image features
    2) load trained DP image projector
    3) build final normalized image embeddings
    4) attach labels + masks for this split
    5) save utility-ready pack
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    params = read_config(cfg_path)
    csv_path = params["CrossModal"]["mimic_master_csv"]

    split_df = get_filtered_split_df_mimic(csv_path=csv_path, split_name=split_name)
    label_names = get_crossmodal_label_names()
    labels_np, masks_np = build_binary_labels_and_mask_from_df(split_df, label_names)

    img_features = torch.load(initial_feature_path, map_location="cpu", weights_only=False)

    if not isinstance(img_features, torch.Tensor):
        raise ValueError(f"Expected tensor at {initial_feature_path}, got {type(img_features)}")

    if len(img_features) != len(split_df):
        raise ValueError(
            f"Length mismatch for split={split_name}: "
            f"len(cached img features)={len(img_features)} vs len(filtered df)={len(split_df)}"
        )

    image_engine = get_image_inference(ImageModelType.BIOVIL_T)
    image_projector = image_engine.model.projector.to(device)
    image_projector.eval()

    ckpt = torch.load(dp_checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

    projector_state = {}
    prefix = "_module.image_projector."

    for k, v in state.items():
        if k.startswith(prefix):
            projector_state[k.replace(prefix, "")] = v

    if len(projector_state) == 0:
        raise ValueError("Could not find image projector weights in DP checkpoint under '_module.image_projector.'")

    missing, unexpected = image_projector.load_state_dict(projector_state, strict=False)
    print(f"Loaded DP projector from: {dp_checkpoint_path}")
    print("Missing projector keys:", missing)
    print("Unexpected projector keys:", unexpected)

    batch_size = 1024
    z_img_all = []

    with torch.inference_mode():
        for start in tqdm(range(0, len(img_features), batch_size), desc=f"Projecting {split_name} cached features"):
            end = min(start + batch_size, len(img_features))
            x = img_features[start:end].to(device, non_blocking=True)

            if x.ndim == 2:
                x = x.unsqueeze(-1).unsqueeze(-1)

            z_img = image_projector(x)
            z_img = z_img.flatten(1)
            z_img = F.normalize(z_img.float(), p=2, dim=-1)

            z_img_all.append(z_img.cpu())


    Z_img = torch.cat(z_img_all, dim=0).to(torch.float16)

    save_obj = {
        "Z_img": Z_img,
        "labels": labels_np.astype(np.float32),
        "label_masks": masks_np.astype(np.float32),
        "row_ids": np.arange(len(split_df)),
        "subject_ids": split_df["subject_id"].values if "subject_id" in split_df.columns else np.full(len(split_df), -1),
        "study_ids": split_df["study_id"].values if "study_id" in split_df.columns else np.full(len(split_df), -1),
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(save_obj, out_path)
    print(f"Saved DP utility-ready embeddings to: {out_path}")





if __name__ == '__main__':
    cfg_path = "/PATH/config/config.yaml"
    out_base_path = "/PATH/embeddings"
    trained_model_path = "/PATH/epoch1_trained_model.pth"
    params = read_config(cfg_path)


    ''' off the shelf'''
    extract_and_save_image_embeddings(
        cfg_path=cfg_path,
        split_name="train",
        out_path=os.path.join(out_base_path, "biovilt_offshelf_train.pt"),
        variant_name="off_shelf",
        trained_model_path=None,
        batch_size=128,
        num_workers=8,
        use_amp=True,
    )
    extract_and_save_image_embeddings(
        cfg_path=cfg_path,
        split_name="test",
        out_path=os.path.join(out_base_path, "biovilt_offshelf_test.pt"),
        variant_name="off_shelf",
        trained_model_path=None,
        batch_size=128,
        num_workers=8,
        use_amp=True,
    )

    # train linear probe
    train_linear_probe_from_saved_embeddings(
        train_emb_path=os.path.join(out_base_path, "biovilt_offshelf_train.pt"),
        val_emb_path=os.path.join(out_base_path, "biovilt_offshelf_test.pt"),
        classifier_save_path=os.path.join(out_base_path, "linear_probe_biovilt_offshelf.pth"),
        batch_size=512,
        lr=1e-3,
        num_epochs=100,
    )

    # test linear probe
    test_linear_probe_from_saved_embeddings(
        test_emb_path=os.path.join(out_base_path, "biovilt_offshelf_test.pt"),
        classifier_ckpt_path=os.path.join(out_base_path, "linear_probe_biovilt_offshelf.pth"),
        batch_size=512,
        save_predictions_csv_path=os.path.join(out_base_path, "biovilt_offshelf_predictions.csv"),
        save_bootstrap_csv_path=os.path.join(out_base_path, "biovilt_offshelf_bootstrap.csv"),
        save_summary_txt_path=os.path.join(out_base_path, "biovilt_offshelf_bootstraptext.text"),
    )








    ''' DP version'''
    cache_dir = os.path.join(params["target_dir"], 'cached_biovilt_features')
    train_img_path = os.path.join(cache_dir, "train_img.pt")
    train_txt_path = os.path.join(cache_dir, "train_txt.pt")
    test_img_path = os.path.join(cache_dir, "test_img.pt")
    test_txt_path = os.path.join(cache_dir, "test_txt.pt")

    extract_and_save_dp_projected_image_embeddings(
        cfg_path=cfg_path,
        split_name="train",
        initial_feature_path=train_img_path,
        dp_checkpoint_path=trained_model_path,
        out_path=os.path.join(out_base_path, "biovilt_dp_train.pt"),
    )

    extract_and_save_dp_projected_image_embeddings(
        cfg_path=cfg_path,
        split_name="test",
        initial_feature_path=test_img_path,
        dp_checkpoint_path=trained_model_path,
        out_path=os.path.join(out_base_path, "biovilt_dp_test.pt"),
    )


    # train linear probe
    train_linear_probe_from_saved_embeddings(
        train_emb_path=os.path.join(out_base_path, "biovilt_dp_train.pt"),
        val_emb_path=os.path.join(out_base_path, "biovilt_dp_est.pt"),
        classifier_save_path=os.path.join(out_base_path, "linear_probe_biovilt_DP.pth"),
        batch_size=512,
        lr=1e-3,
        num_epochs=100,
    )

    # test linear probe
    test_linear_probe_from_saved_embeddings(
        test_emb_path=os.path.join(out_base_path, "biovilt_dp_est.pt"),
        classifier_ckpt_path=os.path.join(out_base_path, "linear_probe_biovilt_DP.pth"),
        batch_size=512,
        save_predictions_csv_path=os.path.join(out_base_path, "biovilt_DP_predictions.csv"),
        save_bootstrap_csv_path=os.path.join(out_base_path, "biovilt_DP_bootstrap.csv"),
        save_summary_txt_path=os.path.join(out_base_path, "biovilt_DP_bootstraptext.text"),
    )

