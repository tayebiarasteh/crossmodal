"""
Created on March 26, 2026.
Evaluations_crossmodal.py

https://github.com/tayebiarasteh/crossmodal/
"""

import os
import math
import json
import pdb
from pathlib import Path
import torch.nn as nn

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor, AutoTokenizer
from PIL import Image
from health_multimodal.image.utils import get_image_inference, ImageModelType
import open_clip
from torch.utils.data import DataLoader

from config.serde import read_config
from data.data_provider import mimic_retrieval_test_data_loader_2D, mimic_retrieval_collate_fn, chexpert_retrieval_test_data_loader_2D, chexpert_retrieval_collate_fn


epsilon = 1e-15






class PredictionCrossModalSettingA:
    def __init__(self, cfg_path: str, split_name: str, dataset_name: str, model_name: str, variant_name: str):
        self.params = read_config(cfg_path)
        self.cfg_path = cfg_path
        self.split_name = str(split_name)
        self.dataset_name = str(dataset_name).lower()
        self.model_name = str(model_name)
        self.variant_name = str(variant_name)

        self.setup_cuda()

        self.batch_size = int(self.params["CrossModal"]["batch_size"])
        self.max_text_tokens = int(self.params["CrossModal"]["max_text_tokens_clip_pubmed"])
        self.base_seed = int(self.params["CrossModal"]["base_seed"])
        self.n_trials = int(self.params["CrossModal"]["n_trials_settingA"])
        self.n_bootstrap = int(self.params["CrossModal"]["n_bootstrap"])

        self.out_dir = os.path.join(
            self.params["target_dir"],
            self.params["stat_log_path"],
            f"{self.dataset_name}_{self.variant_name}_{self.model_name.replace('/', '_')}_split_{self.split_name}",
        )
        os.makedirs(self.out_dir, exist_ok=True)

        self.bootstrap_dir = os.path.join(
            self.params["target_dir"],
            self.params["stat_log_path"],
            "fixed_bootstrap_indices",
        )
        os.makedirs(self.bootstrap_dir, exist_ok=True)

    def setup_cuda(self, cuda_device_id: int = 0):
        if torch.cuda.is_available():
            torch.backends.cudnn.fastest = True
            torch.cuda.set_device(cuda_device_id)
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def setup_model(self, trained_model_path=None):
        self.processor = None
        self.tokenizer = None

        try:
            self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
        except Exception:
            self.processor = None

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                use_fast=True,
            )
        except Exception:
            self.tokenizer = None

        self.model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            use_safetensors=False,
        ).to(self.device)

        if trained_model_path is not None:
            state = torch.load(trained_model_path, map_location="cpu")
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]

            missing, unexpected = self.model.load_state_dict(state, strict=False)
            print("Loaded trained checkpoint")
            print("Missing keys:", missing)
            print("Unexpected keys:", unexpected)

        self.model.eval()

    def _unwrap_to_tensor(self, x):
        if isinstance(x, torch.Tensor):
            return x

        if hasattr(x, "image_embeds") and x.image_embeds is not None:
            return x.image_embeds

        if hasattr(x, "text_embeds") and x.text_embeds is not None:
            return x.text_embeds

        if hasattr(x, "pooler_output") and x.pooler_output is not None:
            return x.pooler_output

        if hasattr(x, "last_hidden_state") and x.last_hidden_state is not None:
            return x.last_hidden_state[:, 0]

        if isinstance(x, (tuple, list)) and len(x) > 0:
            first = x[0]
            if isinstance(first, torch.Tensor):
                if first.ndim == 3:
                    return first[:, 0]
                return first

        raise TypeError(f"Could not convert output of type {type(x)} to embedding tensor.")

    def _manual_image_preprocess(self, images):
        """
        Fallback for models where AutoProcessor does not support images.
        Produces CLIP-style pixel_values with size 224.
        """
        pixel_tensors = []

        for img in images:
            img = img.convert("RGB")
            img = img.resize((224, 224))
            arr = np.asarray(img).astype(np.float32) / 255.0
            arr = (arr - 0.5) / 0.5
            arr = np.transpose(arr, (2, 0, 1))
            pixel_tensors.append(torch.tensor(arr, dtype=torch.float32))

        pixel_values = torch.stack(pixel_tensors, dim=0)
        return {"pixel_values": pixel_values.to(self.device, non_blocking=True)}

    def _encode_images_batch(self, image_paths):
        images = [Image.open(p).convert("RGB") for p in image_paths]

        inputs = None

        if self.processor is not None:
            try:
                inputs = self.processor(images=images, return_tensors="pt")
                inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}
            except Exception:
                inputs = None

        if inputs is None:
            inputs = self._manual_image_preprocess(images)

        with torch.no_grad():
            if hasattr(self.model, "get_image_features"):
                z = self.model.get_image_features(**inputs)
            elif hasattr(self.model, "get_projected_image_embeddings"):
                z = self.model.get_projected_image_embeddings(**inputs)
            elif hasattr(self.model, "get_vision_features"):
                z = self.model.get_vision_features(**inputs)
            else:
                z = self.model(**inputs)

        z = self._unwrap_to_tensor(z)
        z = F.normalize(z.float(), p=2, dim=-1)
        return z

    def _encode_texts_batch(self, texts):
        tok = None

        max_len = int(self.max_text_tokens)

        if self.processor is not None:
            try:
                tok = self.processor(
                    text=texts,
                    padding=True,
                    truncation=True,
                    max_length=max_len,
                    return_tensors="pt",
                )
            except Exception:
                tok = None

        if tok is None:
            if self.tokenizer is None:
                raise ValueError(f"No tokenizer available for model {self.model_name}")
            tok = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            )

        tok = {k: v.to(self.device, non_blocking=True) for k, v in tok.items()}

        with torch.no_grad():
            if hasattr(self.model, "get_text_features"):
                z = self.model.get_text_features(**tok)
            elif hasattr(self.model, "get_text_embeddings"):
                z = self.model.get_text_embeddings(**tok)
            elif hasattr(self.model, "get_projected_text_embeddings"):
                z = self.model.get_projected_text_embeddings(
                    input_ids=tok["input_ids"],
                    attention_mask=tok["attention_mask"],
                )
            else:
                z = self.model(**tok)

        z = self._unwrap_to_tensor(z)
        z = F.normalize(z.float(), p=2, dim=-1)
        return z

    def extract_embeddings(self, test_loader):
        self.model.eval()

        z_img_cache = []
        z_txt_cache = []
        row_ids_cache = []
        subject_ids_cache = []
        study_ids_cache = []
        labels_cache = []

        with torch.no_grad():
            for batch in tqdm(test_loader):
                z_img = self._encode_images_batch(batch["image_paths"])
                z_txt = self._encode_texts_batch(batch["report_texts"])

                z_img_cache.append(z_img.detach().cpu())
                z_txt_cache.append(z_txt.detach().cpu())

                row_ids_cache.extend(batch["row_ids"])
                subject_ids_cache.extend(batch["subject_ids"])
                study_ids_cache.extend(batch["study_ids"])
                labels_cache.append(batch["labels"].detach().cpu())

        Z_img = torch.cat(z_img_cache, dim=0).to(torch.float16)
        Z_txt = torch.cat(z_txt_cache, dim=0).to(torch.float16)
        labels_all = torch.cat(labels_cache, dim=0).numpy()

        return {
            "Z_img": Z_img,
            "Z_txt": Z_txt,
            "row_ids": np.asarray(row_ids_cache),
            "subject_ids": np.asarray(subject_ids_cache),
            "study_ids": np.asarray(study_ids_cache),
            "labels": labels_all,
        }

    def _get_candidate_sizes(self, M):
        out = [100, 1000, 10000]
        out = [n for n in out if n <= M]
        if M not in out:
            out.append(M)
        return sorted(out)

    def _seed_for_setting_a(self, N, trial_id, max_test_size):
        if N in [100, 1000, 10000]:
            return self.base_seed + int(1000000 * math.log10(N)) + trial_id
        if N == max_test_size:
            return self.base_seed + 9999999 + trial_id
        raise ValueError(f"Unsupported N: {N}")

    def _compute_metrics_from_rank(self, rank_1based):
        return {
            "recall@1": 1.0 if rank_1based <= 1 else 0.0,
            "recall@5": 1.0 if rank_1based <= 5 else 0.0,
            "recall@10": 1.0 if rank_1based <= 10 else 0.0,
            "mrr": 1.0 / float(rank_1based),
        }

    def _evaluate_one_query_one_trial(self, z_img_i, Z_txt_device, query_idx, cand_idx):
        cand_idx_t = torch.as_tensor(cand_idx, device=Z_txt_device.device, dtype=torch.long)
        cand_txt = Z_txt_device.index_select(0, cand_idx_t)
        scores = torch.matmul(cand_txt, z_img_i)

        order = torch.argsort(scores, descending=True)

        true_pos = np.where(np.asarray(cand_idx) == query_idx)[0]
        if len(true_pos) != 1:
            raise ValueError("True report missing or duplicated in candidate pool.")
        true_pos = int(true_pos[0])

        rank_1based = int((order == true_pos).nonzero(as_tuple=False).item()) + 1

        out = self._compute_metrics_from_rank(rank_1based)
        out["rank"] = rank_1based
        return out

    def _get_or_create_bootstrap_indices(self, n_queries, seed=13):
        bs_path = os.path.join(
            self.bootstrap_dir,
            f"bootstrap_indices_{self.dataset_name}_{self.split_name}_n{n_queries}_b{self.n_bootstrap}_seed{seed}.npy",
        )

        if os.path.exists(bs_path):
            return np.load(bs_path)

        rng = np.random.RandomState(seed)
        indices = np.zeros((self.n_bootstrap, n_queries), dtype=np.int64)

        for i in range(self.n_bootstrap):
            indices[i] = rng.choice(n_queries, n_queries, replace=True)

        np.save(bs_path, indices)
        return indices

    def bootstrap_metrics(self, per_query_df, bootstrap_indices):
        metric_names = ["recall@1", "recall@5", "recall@10", "mrr"]
        boot_dict = {}

        for metric in metric_names:
            vals = per_query_df[metric].to_numpy(dtype=np.float64)
            arr = np.zeros((len(bootstrap_indices),), dtype=np.float32)

            for i, idx in enumerate(bootstrap_indices):
                arr[i] = float(vals[idx].mean())

            boot_dict[metric] = arr

        return boot_dict

    def ci95(self, x):
        lo, hi = np.percentile(x, [2.5, 97.5])
        return lo, hi

    def _summary_str_pct(self, arr: np.ndarray) -> str:
        arr = np.asarray(arr, dtype=np.float32)
        mean = float(arr.mean()) * 100.0
        std = float(arr.std()) * 100.0
        lo, hi = self.ci95(arr)
        lo *= 100.0
        hi *= 100.0
        return f"{mean:.4f} ± {std:.4f} [{lo:.4f}, {hi:.4f}]"

    def run_settingA(self, emb_dict):
        Z_img = emb_dict["Z_img"].float().to(self.device)
        Z_txt = emb_dict["Z_txt"].float().to(self.device)

        row_ids = emb_dict["row_ids"]
        subject_ids = emb_dict["subject_ids"]
        study_ids = emb_dict["study_ids"]

        M = len(row_ids)
        candidate_sizes = self._get_candidate_sizes(M)
        all_indices = np.arange(M)

        bootstrap_indices = self._get_or_create_bootstrap_indices(n_queries=M, seed=13)

        all_summary_rows = []
        raw_trial_rows = []
        all_boot_rows = []

        for N in candidate_sizes:
            print(f"\nRunning Setting A | dataset={self.dataset_name} | model={self.model_name} | variant={self.variant_name} | N={N}")

            per_query_rows = []

            for q_idx in tqdm(range(M)):
                z_img_i = Z_img[q_idx]
                neg_pool = all_indices[all_indices != q_idx]

                trial_metrics = []

                if N == M:
                    cand_idx = all_indices.copy()

                    for trial_id in range(self.n_trials):
                        m = self._evaluate_one_query_one_trial(
                            z_img_i=z_img_i,
                            Z_txt_device=Z_txt,
                            query_idx=q_idx,
                            cand_idx=cand_idx,
                        )
                        trial_metrics.append(m)

                        raw_trial_rows.append({
                            "dataset_name": self.dataset_name,
                            "model_name": self.model_name,
                            "variant_name": self.variant_name,
                            "N": N,
                            "query_row_id": int(row_ids[q_idx]),
                            "subject_id": int(subject_ids[q_idx]),
                            "study_id": int(study_ids[q_idx]),
                            "trial_id": int(trial_id),
                            "rank": int(m["rank"]),
                            "recall@1": float(m["recall@1"]),
                            "recall@5": float(m["recall@5"]),
                            "recall@10": float(m["recall@10"]),
                            "mrr": float(m["mrr"]),
                        })
                else:
                    for trial_id in range(self.n_trials):
                        seed = self._seed_for_setting_a(N, trial_id, M)
                        rng = np.random.RandomState(seed)

                        neg_idx = rng.choice(neg_pool, size=N - 1, replace=False)
                        cand_idx = np.concatenate([[q_idx], neg_idx])
                        rng.shuffle(cand_idx)

                        m = self._evaluate_one_query_one_trial(
                            z_img_i=z_img_i,
                            Z_txt_device=Z_txt,
                            query_idx=q_idx,
                            cand_idx=cand_idx,
                        )
                        trial_metrics.append(m)

                        raw_trial_rows.append({
                            "dataset_name": self.dataset_name,
                            "model_name": self.model_name,
                            "variant_name": self.variant_name,
                            "N": N,
                            "query_row_id": int(row_ids[q_idx]),
                            "subject_id": int(subject_ids[q_idx]),
                            "study_id": int(study_ids[q_idx]),
                            "trial_id": int(trial_id),
                            "rank": int(m["rank"]),
                            "recall@1": float(m["recall@1"]),
                            "recall@5": float(m["recall@5"]),
                            "recall@10": float(m["recall@10"]),
                            "mrr": float(m["mrr"]),
                        })

                per_query_rows.append({
                    "dataset_name": self.dataset_name,
                    "model_name": self.model_name,
                    "variant_name": self.variant_name,
                    "N": N,
                    "query_row_id": int(row_ids[q_idx]),
                    "subject_id": int(subject_ids[q_idx]),
                    "study_id": int(study_ids[q_idx]),
                    "recall@1": float(np.mean([m["recall@1"] for m in trial_metrics])),
                    "recall@5": float(np.mean([m["recall@5"] for m in trial_metrics])),
                    "recall@10": float(np.mean([m["recall@10"] for m in trial_metrics])),
                    "mrr": float(np.mean([m["mrr"] for m in trial_metrics])),
                })

            per_query_df = pd.DataFrame(per_query_rows)
            boot_dict = self.bootstrap_metrics(per_query_df=per_query_df, bootstrap_indices=bootstrap_indices)

            for i in range(len(boot_dict["recall@1"])):
                all_boot_rows.append({
                    "dataset_name": self.dataset_name,
                    "model_name": self.model_name,
                    "variant_name": self.variant_name,
                    "N": N,
                    "bootstrap_id": i,
                    "recall@1": float(boot_dict["recall@1"][i]),
                    "recall@5": float(boot_dict["recall@5"][i]),
                    "recall@10": float(boot_dict["recall@10"][i]),
                    "mrr": float(boot_dict["mrr"][i]),
                })

            all_summary_rows.append({
                "dataset_name": self.dataset_name,
                "model_name": self.model_name,
                "variant_name": self.variant_name,
                "split_name": self.split_name,
                "N": int(N),
                "num_queries": int(len(per_query_df)),
                "num_trials_per_query": int(self.n_trials),
                "recall@1": self._summary_str_pct(boot_dict["recall@1"]),
                "recall@5": self._summary_str_pct(boot_dict["recall@5"]),
                "recall@10": self._summary_str_pct(boot_dict["recall@10"]),
                "mrr": self._summary_str_pct(boot_dict["mrr"]),
            })

        return {
            "summary_df": pd.DataFrame(all_summary_rows),
            "raw_trial_df": pd.DataFrame(raw_trial_rows),
            "bootstrap_df": pd.DataFrame(all_boot_rows),
        }

    def save_predictions_csv(self, raw_trial_df, setting_name="A"):
        setting_name = str(setting_name).upper()
        out_path = os.path.join(
            self.out_dir,
            f"predictions_setting{setting_name}_{self.dataset_name}_{self.variant_name}_{self.model_name.replace('/', '_')}_split_{self.split_name}.csv",
        )
        raw_trial_df.to_csv(out_path, index=False)

    def save_bootstrap_csv(self, bootstrap_df, setting_name="A"):
        setting_name = str(setting_name).upper()
        out_path = os.path.join(
            self.out_dir,
            f"bootstrap_setting{setting_name}_{self.dataset_name}_{self.variant_name}_{self.model_name.replace('/', '_')}_split_{self.split_name}.csv",
        )
        bootstrap_df.to_csv(out_path, index=False)

    def save_summary_csv(self, summary_df, setting_name="A"):
        setting_name = str(setting_name).upper()
        out_path = os.path.join(
            self.out_dir,
            f"summary_setting{setting_name}_{self.dataset_name}_{self.variant_name}_{self.model_name.replace('/', '_')}_split_{self.split_name}.csv",
        )
        summary_df.to_csv(out_path, index=False)

    def print_and_save_stats(self, summary_df, setting_name="A"):
        setting_name = str(setting_name).upper()

        msg = (
            f"\n\n----------------------------------------------------------------------------------------\n"
            f"setting: {setting_name}\n"
            f"dataset: {self.dataset_name}\n"
            f"model: {self.model_name}\n"
            f"variant: {self.variant_name}\n"
            f"split: {self.split_name}\n\n"
        )

        for _, row in summary_df.iterrows():
            msg += (
                f"N={int(row['N'])}\n"
                f"Recall@1: {row['recall@1']}\n"
                f"Recall@5: {row['recall@5']}\n"
                f"Recall@10: {row['recall@10']}\n"
                f"MRR: {row['mrr']}\n\n"
            )

        print(msg)

        out_path = os.path.join(
            self.out_dir,
            f"Test_on_setting{setting_name}_{self.dataset_name}_{self.variant_name}_{self.model_name.replace('/', '_')}_split_{self.split_name}.txt",
        )
        with open(out_path, "a") as f:
            f.write(msg)






class PredictionCrossModalSettingA_BioViLT:
    def __init__(self, cfg_path: str, split_name: str, dataset_name: str, variant_name: str):
        self.params = read_config(cfg_path)
        self.cfg_path = cfg_path
        self.split_name = str(split_name)
        self.dataset_name = str(dataset_name).lower()
        self.model_name = "microsoft/BiomedVLP-BioViL-T"
        self.variant_name = str(variant_name)

        self.setup_cuda()

        self.batch_size = int(self.params["CrossModal"]["batch_size"])
        self.max_text_tokens = int(self.params["CrossModal"]["max_text_tokens_biovilt"])
        self.base_seed = int(self.params["CrossModal"]["base_seed"])
        self.n_trials = int(self.params["CrossModal"]["n_trials_settingA"])
        self.n_bootstrap = int(self.params["CrossModal"]["n_bootstrap"])

        self.out_dir = os.path.join(
            self.params["target_dir"],
            self.params["stat_log_path"],
            f"{self.dataset_name}_{self.variant_name}_biovilt_split_{self.split_name}",
        )
        os.makedirs(self.out_dir, exist_ok=True)

        self.bootstrap_dir = os.path.join(
            self.params["target_dir"],
            self.params["stat_log_path"],
            "fixed_bootstrap_indices",
        )
        os.makedirs(self.bootstrap_dir, exist_ok=True)

    def setup_cuda(self, cuda_device_id: int = 0):
        if torch.cuda.is_available():
            torch.backends.cudnn.fastest = True
            torch.cuda.set_device(cuda_device_id)
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def setup_model(self, trained_model_path=None):
        self.image_engine = get_image_inference(ImageModelType.BIOVIL_T)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            use_fast=True,
        )

        self.text_model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            use_safetensors=False,
        ).to(self.device)

        if trained_model_path is not None:
            state = torch.load(trained_model_path, map_location="cpu")
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]

            missing, unexpected = self.text_model.load_state_dict(state, strict=False)
            print("Loaded trained checkpoint")
            print("Missing keys:", missing)
            print("Unexpected keys:", unexpected)

        self.text_model.eval()

    def _encode_images_batch(self, image_paths):
        z_list = []

        for p in image_paths:
            p = Path(p)
            z = self.image_engine.get_projected_global_embedding(p)

            if z.ndim == 1:
                z = z.unsqueeze(0)

            z = F.normalize(z.float(), p=2, dim=-1)
            z_list.append(z.cpu())

        z = torch.cat(z_list, dim=0).to(self.device)
        return z


    def _encode_texts_batch(self, texts):
        max_len = int(self.max_text_tokens)

        tok = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        tok = {k: v.to(self.device, non_blocking=True) for k, v in tok.items()}

        with torch.no_grad():
            if hasattr(self.text_model, "get_projected_text_embeddings"):
                z = self.text_model.get_projected_text_embeddings(
                    input_ids=tok["input_ids"],
                    attention_mask=tok["attention_mask"],
                )
            elif hasattr(self.text_model, "get_text_features"):
                z = self.text_model.get_text_features(**tok)
            elif hasattr(self.text_model, "get_text_embeddings"):
                z = self.text_model.get_text_embeddings(**tok)
            else:
                raise AttributeError("BioViL-T text model does not expose a supported text embedding method.")

        z = F.normalize(z.float(), p=2, dim=-1)
        return z

    def extract_embeddings(self, test_loader):
        self.text_model.eval()

        z_img_cache = []
        z_txt_cache = []
        row_ids_cache = []
        subject_ids_cache = []
        study_ids_cache = []
        labels_cache = []

        with torch.no_grad():
            for batch in tqdm(test_loader):
                z_img = self._encode_images_batch(batch["image_paths"])
                z_txt = self._encode_texts_batch(batch["report_texts"])

                z_img_cache.append(z_img.detach().cpu())
                z_txt_cache.append(z_txt.detach().cpu())

                row_ids_cache.extend(batch["row_ids"])
                subject_ids_cache.extend(batch["subject_ids"])
                study_ids_cache.extend(batch["study_ids"])
                labels_cache.append(batch["labels"].detach().cpu())

        Z_img = torch.cat(z_img_cache, dim=0).to(torch.float16)
        Z_txt = torch.cat(z_txt_cache, dim=0).to(torch.float16)
        labels_all = torch.cat(labels_cache, dim=0).numpy()

        return {
            "Z_img": Z_img,
            "Z_txt": Z_txt,
            "row_ids": np.asarray(row_ids_cache),
            "subject_ids": np.asarray(subject_ids_cache),
            "study_ids": np.asarray(study_ids_cache),
            "labels": labels_all,
        }

    def _get_candidate_sizes(self, M):
        out = [100, 1000, 10000]
        out = [n for n in out if n <= M]
        if M not in out:
            out.append(M)
        return sorted(out)

    def _seed_for_setting_a(self, N, trial_id, max_test_size):
        if N in [100, 1000, 10000]:
            return self.base_seed + int(1000000 * math.log10(N)) + trial_id
        if N == max_test_size:
            return self.base_seed + 9999999 + trial_id
        raise ValueError(f"Unsupported N: {N}")

    def _compute_metrics_from_rank(self, rank_1based):
        return {
            "recall@1": 1.0 if rank_1based <= 1 else 0.0,
            "recall@5": 1.0 if rank_1based <= 5 else 0.0,
            "recall@10": 1.0 if rank_1based <= 10 else 0.0,
            "mrr": 1.0 / float(rank_1based),
        }

    def _evaluate_one_query_one_trial(self, z_img_i, Z_txt_device, query_idx, cand_idx):
        cand_idx_t = torch.as_tensor(cand_idx, device=Z_txt_device.device, dtype=torch.long)
        cand_txt = Z_txt_device.index_select(0, cand_idx_t)
        scores = torch.matmul(cand_txt, z_img_i)

        order = torch.argsort(scores, descending=True)

        true_pos = np.where(np.asarray(cand_idx) == query_idx)[0]
        if len(true_pos) != 1:
            raise ValueError("True report missing or duplicated in candidate pool.")
        true_pos = int(true_pos[0])

        rank_1based = int((order == true_pos).nonzero(as_tuple=False).item()) + 1

        out = self._compute_metrics_from_rank(rank_1based)
        out["rank"] = rank_1based
        return out

    def _get_or_create_bootstrap_indices(self, n_queries, seed=13):
        bs_path = os.path.join(
            self.bootstrap_dir,
            f"bootstrap_indices_{self.dataset_name}_{self.split_name}_n{n_queries}_b{self.n_bootstrap}_seed{seed}.npy",
        )

        if os.path.exists(bs_path):
            return np.load(bs_path)

        rng = np.random.RandomState(seed)
        indices = np.zeros((self.n_bootstrap, n_queries), dtype=np.int64)

        for i in range(self.n_bootstrap):
            indices[i] = rng.choice(n_queries, n_queries, replace=True)

        np.save(bs_path, indices)
        return indices

    def bootstrap_metrics(self, per_query_df, bootstrap_indices):
        metric_names = ["recall@1", "recall@5", "recall@10", "mrr"]
        boot_dict = {}

        for metric in metric_names:
            vals = per_query_df[metric].to_numpy(dtype=np.float64)
            arr = np.zeros((len(bootstrap_indices),), dtype=np.float32)

            for i, idx in enumerate(bootstrap_indices):
                arr[i] = float(vals[idx].mean())

            boot_dict[metric] = arr

        return boot_dict

    def ci95(self, x):
        lo, hi = np.percentile(x, [2.5, 97.5])
        return lo, hi

    def _summary_str_pct(self, arr: np.ndarray) -> str:
        arr = np.asarray(arr, dtype=np.float32)
        mean = float(arr.mean()) * 100.0
        std = float(arr.std()) * 100.0
        lo, hi = self.ci95(arr)
        lo *= 100.0
        hi *= 100.0
        return f"{mean:.4f} ± {std:.4f} [{lo:.4f}, {hi:.4f}]"

    def run_settingA(self, emb_dict):
        Z_img = emb_dict["Z_img"].float().to(self.device)
        Z_txt = emb_dict["Z_txt"].float().to(self.device)

        row_ids = emb_dict["row_ids"]
        subject_ids = emb_dict["subject_ids"]
        study_ids = emb_dict["study_ids"]

        M = len(row_ids)
        candidate_sizes = self._get_candidate_sizes(M)
        all_indices = np.arange(M)

        bootstrap_indices = self._get_or_create_bootstrap_indices(n_queries=M, seed=13)

        all_summary_rows = []
        raw_trial_rows = []
        all_boot_rows = []

        for N in candidate_sizes:
            print(f"\nRunning Setting A | dataset={self.dataset_name} | model=BioViL-T | variant={self.variant_name} | N={N}")

            per_query_rows = []

            for q_idx in tqdm(range(M)):
                z_img_i = Z_img[q_idx]
                neg_pool = all_indices[all_indices != q_idx]

                trial_metrics = []

                if N == M:
                    cand_idx = all_indices.copy()

                    for trial_id in range(self.n_trials):
                        m = self._evaluate_one_query_one_trial(
                            z_img_i=z_img_i,
                            Z_txt_device=Z_txt,
                            query_idx=q_idx,
                            cand_idx=cand_idx,
                        )
                        trial_metrics.append(m)

                        raw_trial_rows.append({
                            "dataset_name": self.dataset_name,
                            "model_name": "microsoft/BiomedVLP-BioViL-T",
                            "variant_name": self.variant_name,
                            "N": N,
                            "query_row_id": int(row_ids[q_idx]),
                            "subject_id": int(subject_ids[q_idx]),
                            "study_id": int(study_ids[q_idx]),
                            "trial_id": int(trial_id),
                            "rank": int(m["rank"]),
                            "recall@1": float(m["recall@1"]),
                            "recall@5": float(m["recall@5"]),
                            "recall@10": float(m["recall@10"]),
                            "mrr": float(m["mrr"]),
                        })
                else:
                    for trial_id in range(self.n_trials):
                        seed = self._seed_for_setting_a(N, trial_id, M)
                        rng = np.random.RandomState(seed)

                        neg_idx = rng.choice(neg_pool, size=N - 1, replace=False)
                        cand_idx = np.concatenate([[q_idx], neg_idx])
                        rng.shuffle(cand_idx)

                        m = self._evaluate_one_query_one_trial(
                            z_img_i=z_img_i,
                            Z_txt_device=Z_txt,
                            query_idx=q_idx,
                            cand_idx=cand_idx,
                        )
                        trial_metrics.append(m)

                        raw_trial_rows.append({
                            "dataset_name": self.dataset_name,
                            "model_name": "microsoft/BiomedVLP-BioViL-T",
                            "variant_name": self.variant_name,
                            "N": N,
                            "query_row_id": int(row_ids[q_idx]),
                            "subject_id": int(subject_ids[q_idx]),
                            "study_id": int(study_ids[q_idx]),
                            "trial_id": int(trial_id),
                            "rank": int(m["rank"]),
                            "recall@1": float(m["recall@1"]),
                            "recall@5": float(m["recall@5"]),
                            "recall@10": float(m["recall@10"]),
                            "mrr": float(m["mrr"]),
                        })

                per_query_rows.append({
                    "dataset_name": self.dataset_name,
                    "model_name": "microsoft/BiomedVLP-BioViL-T",
                    "variant_name": self.variant_name,
                    "N": N,
                    "query_row_id": int(row_ids[q_idx]),
                    "subject_id": int(subject_ids[q_idx]),
                    "study_id": int(study_ids[q_idx]),
                    "recall@1": float(np.mean([m["recall@1"] for m in trial_metrics])),
                    "recall@5": float(np.mean([m["recall@5"] for m in trial_metrics])),
                    "recall@10": float(np.mean([m["recall@10"] for m in trial_metrics])),
                    "mrr": float(np.mean([m["mrr"] for m in trial_metrics])),
                })

            per_query_df = pd.DataFrame(per_query_rows)
            boot_dict = self.bootstrap_metrics(per_query_df=per_query_df, bootstrap_indices=bootstrap_indices)

            for i in range(len(boot_dict["recall@1"])):
                all_boot_rows.append({
                    "dataset_name": self.dataset_name,
                    "model_name": "microsoft/BiomedVLP-BioViL-T",
                    "variant_name": self.variant_name,
                    "N": N,
                    "bootstrap_id": i,
                    "recall@1": float(boot_dict["recall@1"][i]),
                    "recall@5": float(boot_dict["recall@5"][i]),
                    "recall@10": float(boot_dict["recall@10"][i]),
                    "mrr": float(boot_dict["mrr"][i]),
                })

            all_summary_rows.append({
                "dataset_name": self.dataset_name,
                "model_name": "microsoft/BiomedVLP-BioViL-T",
                "variant_name": self.variant_name,
                "split_name": self.split_name,
                "N": int(N),
                "num_queries": int(len(per_query_df)),
                "num_trials_per_query": int(self.n_trials),
                "recall@1": self._summary_str_pct(boot_dict["recall@1"]),
                "recall@5": self._summary_str_pct(boot_dict["recall@5"]),
                "recall@10": self._summary_str_pct(boot_dict["recall@10"]),
                "mrr": self._summary_str_pct(boot_dict["mrr"]),
            })

        return {
            "summary_df": pd.DataFrame(all_summary_rows),
            "raw_trial_df": pd.DataFrame(raw_trial_rows),
            "bootstrap_df": pd.DataFrame(all_boot_rows),
        }

    def save_predictions_csv(self, raw_trial_df, setting_name="A"):
        setting_name = str(setting_name).upper()
        out_path = os.path.join(
            self.out_dir,
            f"predictions_setting{setting_name}_{self.dataset_name}_{self.variant_name}_{self.model_name.replace('/', '_')}_split_{self.split_name}.csv",
        )
        raw_trial_df.to_csv(out_path, index=False)

    def save_bootstrap_csv(self, bootstrap_df, setting_name="A"):
        setting_name = str(setting_name).upper()
        out_path = os.path.join(
            self.out_dir,
            f"bootstrap_setting{setting_name}_{self.dataset_name}_{self.variant_name}_{self.model_name.replace('/', '_')}_split_{self.split_name}.csv",
        )
        bootstrap_df.to_csv(out_path, index=False)

    def save_summary_csv(self, summary_df, setting_name="A"):
        setting_name = str(setting_name).upper()
        out_path = os.path.join(
            self.out_dir,
            f"summary_setting{setting_name}_{self.dataset_name}_{self.variant_name}_{self.model_name.replace('/', '_')}_split_{self.split_name}.csv",
        )
        summary_df.to_csv(out_path, index=False)

    def print_and_save_stats(self, summary_df, setting_name="A"):
        setting_name = str(setting_name).upper()

        msg = (
            f"\n\n----------------------------------------------------------------------------------------\n"
            f"setting: {setting_name}\n"
            f"dataset: {self.dataset_name}\n"
            f"model: {self.model_name}\n"
            f"variant: {self.variant_name}\n"
            f"split: {self.split_name}\n\n"
        )

        for _, row in summary_df.iterrows():
            msg += (
                f"N={int(row['N'])}\n"
                f"Recall@1: {row['recall@1']}\n"
                f"Recall@5: {row['recall@5']}\n"
                f"Recall@10: {row['recall@10']}\n"
                f"MRR: {row['mrr']}\n\n"
            )

        print(msg)

        out_path = os.path.join(
            self.out_dir,
            f"Test_on_setting{setting_name}_{self.dataset_name}_{self.variant_name}_{self.model_name.replace('/', '_')}_split_{self.split_name}.txt",
        )
        with open(out_path, "a") as f:
            f.write(msg)






class PredictionCrossModalSettingA_BiomedCLIP:
    def __init__(self, cfg_path: str, split_name: str, dataset_name: str, variant_name: str):
        self.params = read_config(cfg_path)
        self.cfg_path = cfg_path
        self.split_name = str(split_name)
        self.dataset_name = str(dataset_name).lower()
        self.model_name = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        self.variant_name = str(variant_name)

        self.setup_cuda()

        self.batch_size = int(self.params["CrossModal"]["batch_size"])
        self.max_text_tokens = int(self.params["CrossModal"]["max_text_tokens_biovilt"])
        self.base_seed = int(self.params["CrossModal"]["base_seed"])
        self.n_trials = int(self.params["CrossModal"]["n_trials_settingA"])
        self.n_bootstrap = int(self.params["CrossModal"]["n_bootstrap"])

        self.out_dir = os.path.join(
            self.params["target_dir"],
            self.params["stat_log_path"],
            f"{self.dataset_name}_{self.variant_name}_biomedclip_split_{self.split_name}",
        )
        os.makedirs(self.out_dir, exist_ok=True)

        self.bootstrap_dir = os.path.join(
            self.params["target_dir"],
            self.params["stat_log_path"],
            "fixed_bootstrap_indices",
        )
        os.makedirs(self.bootstrap_dir, exist_ok=True)

    def setup_cuda(self, cuda_device_id: int = 0):
        if torch.cuda.is_available():
            torch.backends.cudnn.fastest = True
            torch.cuda.set_device(cuda_device_id)
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def setup_model(self, trained_model_path=None):
        model_id = f"hf-hub:{self.model_name}"

        self.model, self.preprocess = open_clip.create_model_from_pretrained(model_id)
        self.tokenizer = open_clip.get_tokenizer(model_id)
        self.model = self.model.to(self.device)

        if trained_model_path is not None:
            state = torch.load(trained_model_path, map_location="cpu")
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]

            missing, unexpected = self.model.load_state_dict(state, strict=False)
            print("Loaded trained checkpoint")
            print("Missing keys:", missing)
            print("Unexpected keys:", unexpected)

        self.model.eval()

    def _encode_images_batch(self, image_paths):
        images = [self.preprocess(Image.open(p).convert("RGB")) for p in image_paths]
        images = torch.stack(images, dim=0).to(self.device, non_blocking=True)

        with torch.no_grad():
            z = self.model.encode_image(images)

        z = F.normalize(z.float(), p=2, dim=-1)
        return z

    def _encode_texts_batch(self, texts):
        # official tokenizer path for BiomedCLIP
        tok = self.tokenizer(texts, context_length=int(self.max_text_tokens))
        tok = tok.to(self.device, non_blocking=True)

        with torch.no_grad():
            z = self.model.encode_text(tok)

        z = F.normalize(z.float(), p=2, dim=-1)
        return z

    def extract_embeddings(self, test_loader):
        self.model.eval()

        z_img_cache = []
        z_txt_cache = []
        row_ids_cache = []
        subject_ids_cache = []
        study_ids_cache = []
        labels_cache = []

        with torch.no_grad():
            for batch in tqdm(test_loader):
                z_img = self._encode_images_batch(batch["image_paths"])
                z_txt = self._encode_texts_batch(batch["report_texts"])

                z_img_cache.append(z_img.detach().cpu())
                z_txt_cache.append(z_txt.detach().cpu())

                row_ids_cache.extend(batch["row_ids"])
                subject_ids_cache.extend(batch["subject_ids"])
                study_ids_cache.extend(batch["study_ids"])
                labels_cache.append(batch["labels"].detach().cpu())

        Z_img = torch.cat(z_img_cache, dim=0).to(torch.float16)
        Z_txt = torch.cat(z_txt_cache, dim=0).to(torch.float16)
        labels_all = torch.cat(labels_cache, dim=0).numpy()

        return {
            "Z_img": Z_img,
            "Z_txt": Z_txt,
            "row_ids": np.asarray(row_ids_cache),
            "subject_ids": np.asarray(subject_ids_cache),
            "study_ids": np.asarray(study_ids_cache),
            "labels": labels_all,
        }

    def _get_candidate_sizes(self, M):
        out = [100, 1000, 10000]
        out = [n for n in out if n <= M]
        if M not in out:
            out.append(M)
        return sorted(out)

    def _seed_for_setting_a(self, N, trial_id, max_test_size):
        if N in [100, 1000, 10000]:
            return self.base_seed + int(1000000 * math.log10(N)) + trial_id
        if N == max_test_size:
            return self.base_seed + 9999999 + trial_id
        raise ValueError(f"Unsupported N: {N}")

    def _compute_metrics_from_rank(self, rank_1based):
        return {
            "recall@1": 1.0 if rank_1based <= 1 else 0.0,
            "recall@5": 1.0 if rank_1based <= 5 else 0.0,
            "recall@10": 1.0 if rank_1based <= 10 else 0.0,
            "mrr": 1.0 / float(rank_1based),
        }

    def _evaluate_one_query_one_trial(self, z_img_i, Z_txt_device, query_idx, cand_idx):
        cand_idx_t = torch.as_tensor(cand_idx, device=Z_txt_device.device, dtype=torch.long)
        cand_txt = Z_txt_device.index_select(0, cand_idx_t)
        scores = torch.matmul(cand_txt, z_img_i)

        order = torch.argsort(scores, descending=True)

        true_pos = np.where(np.asarray(cand_idx) == query_idx)[0]
        if len(true_pos) != 1:
            raise ValueError("True report missing or duplicated in candidate pool.")
        true_pos = int(true_pos[0])

        rank_1based = int((order == true_pos).nonzero(as_tuple=False).item()) + 1

        out = self._compute_metrics_from_rank(rank_1based)
        out["rank"] = rank_1based
        return out

    def _get_or_create_bootstrap_indices(self, n_queries, seed=13):
        bs_path = os.path.join(
            self.bootstrap_dir,
            f"bootstrap_indices_{self.dataset_name}_{self.split_name}_n{n_queries}_b{self.n_bootstrap}_seed{seed}.npy",
        )

        if os.path.exists(bs_path):
            return np.load(bs_path)

        rng = np.random.RandomState(seed)
        indices = np.zeros((self.n_bootstrap, n_queries), dtype=np.int64)

        for i in range(self.n_bootstrap):
            indices[i] = rng.choice(n_queries, n_queries, replace=True)

        np.save(bs_path, indices)
        return indices

    def bootstrap_metrics(self, per_query_df, bootstrap_indices):
        metric_names = ["recall@1", "recall@5", "recall@10", "mrr"]
        boot_dict = {}

        for metric in metric_names:
            vals = per_query_df[metric].to_numpy(dtype=np.float64)
            arr = np.zeros((len(bootstrap_indices),), dtype=np.float32)

            for i, idx in enumerate(bootstrap_indices):
                arr[i] = float(vals[idx].mean())

            boot_dict[metric] = arr

        return boot_dict

    def ci95(self, x):
        lo, hi = np.percentile(x, [2.5, 97.5])
        return lo, hi

    def _summary_str_pct(self, arr: np.ndarray) -> str:
        arr = np.asarray(arr, dtype=np.float32)
        mean = float(arr.mean()) * 100.0
        std = float(arr.std()) * 100.0
        lo, hi = self.ci95(arr)
        lo *= 100.0
        hi *= 100.0
        return f"{mean:.4f} ± {std:.4f} [{lo:.4f}, {hi:.4f}]"

    def run_settingA(self, emb_dict):
        Z_img = emb_dict["Z_img"].float().to(self.device)
        Z_txt = emb_dict["Z_txt"].float().to(self.device)

        row_ids = emb_dict["row_ids"]
        subject_ids = emb_dict["subject_ids"]
        study_ids = emb_dict["study_ids"]

        M = len(row_ids)
        candidate_sizes = self._get_candidate_sizes(M)
        all_indices = np.arange(M)

        bootstrap_indices = self._get_or_create_bootstrap_indices(n_queries=M, seed=13)

        all_summary_rows = []
        raw_trial_rows = []
        all_boot_rows = []

        for N in candidate_sizes:
            print(f"\nRunning Setting A | dataset={self.dataset_name} | model={self.model_name} | variant={self.variant_name} | N={N}")

            per_query_rows = []

            for q_idx in tqdm(range(M)):
                z_img_i = Z_img[q_idx]
                neg_pool = all_indices[all_indices != q_idx]

                trial_metrics = []

                if N == M:
                    cand_idx = all_indices.copy()
                    for trial_id in range(self.n_trials):
                        m = self._evaluate_one_query_one_trial(z_img_i, Z_txt, q_idx, cand_idx)
                        trial_metrics.append(m)
                        raw_trial_rows.append({
                            "dataset_name": self.dataset_name,
                            "model_name": self.model_name,
                            "variant_name": self.variant_name,
                            "N": N,
                            "query_row_id": int(row_ids[q_idx]),
                            "subject_id": int(subject_ids[q_idx]),
                            "study_id": int(study_ids[q_idx]),
                            "trial_id": int(trial_id),
                            "rank": int(m["rank"]),
                            "recall@1": float(m["recall@1"]),
                            "recall@5": float(m["recall@5"]),
                            "recall@10": float(m["recall@10"]),
                            "mrr": float(m["mrr"]),
                        })
                else:
                    for trial_id in range(self.n_trials):
                        seed = self._seed_for_setting_a(N, trial_id, M)
                        rng = np.random.RandomState(seed)
                        neg_idx = rng.choice(neg_pool, size=N - 1, replace=False)
                        cand_idx = np.concatenate([[q_idx], neg_idx])
                        rng.shuffle(cand_idx)

                        m = self._evaluate_one_query_one_trial(z_img_i, Z_txt, q_idx, cand_idx)
                        trial_metrics.append(m)
                        raw_trial_rows.append({
                            "dataset_name": self.dataset_name,
                            "model_name": self.model_name,
                            "variant_name": self.variant_name,
                            "N": N,
                            "query_row_id": int(row_ids[q_idx]),
                            "subject_id": int(subject_ids[q_idx]),
                            "study_id": int(study_ids[q_idx]),
                            "trial_id": int(trial_id),
                            "rank": int(m["rank"]),
                            "recall@1": float(m["recall@1"]),
                            "recall@5": float(m["recall@5"]),
                            "recall@10": float(m["recall@10"]),
                            "mrr": float(m["mrr"]),
                        })

                per_query_rows.append({
                    "dataset_name": self.dataset_name,
                    "model_name": self.model_name,
                    "variant_name": self.variant_name,
                    "N": N,
                    "query_row_id": int(row_ids[q_idx]),
                    "subject_id": int(subject_ids[q_idx]),
                    "study_id": int(study_ids[q_idx]),
                    "recall@1": float(np.mean([m["recall@1"] for m in trial_metrics])),
                    "recall@5": float(np.mean([m["recall@5"] for m in trial_metrics])),
                    "recall@10": float(np.mean([m["recall@10"] for m in trial_metrics])),
                    "mrr": float(np.mean([m["mrr"] for m in trial_metrics])),
                })

            per_query_df = pd.DataFrame(per_query_rows)
            boot_dict = self.bootstrap_metrics(per_query_df, bootstrap_indices)

            for i in range(len(boot_dict["recall@1"])):
                all_boot_rows.append({
                    "dataset_name": self.dataset_name,
                    "model_name": self.model_name,
                    "variant_name": self.variant_name,
                    "N": N,
                    "bootstrap_id": i,
                    "recall@1": float(boot_dict["recall@1"][i]),
                    "recall@5": float(boot_dict["recall@5"][i]),
                    "recall@10": float(boot_dict["recall@10"][i]),
                    "mrr": float(boot_dict["mrr"][i]),
                })

            all_summary_rows.append({
                "dataset_name": self.dataset_name,
                "model_name": self.model_name,
                "variant_name": self.variant_name,
                "split_name": self.split_name,
                "N": int(N),
                "num_queries": int(len(per_query_df)),
                "num_trials_per_query": int(self.n_trials),
                "recall@1": self._summary_str_pct(boot_dict["recall@1"]),
                "recall@5": self._summary_str_pct(boot_dict["recall@5"]),
                "recall@10": self._summary_str_pct(boot_dict["recall@10"]),
                "mrr": self._summary_str_pct(boot_dict["mrr"]),
            })

        return {
            "summary_df": pd.DataFrame(all_summary_rows),
            "raw_trial_df": pd.DataFrame(raw_trial_rows),
            "bootstrap_df": pd.DataFrame(all_boot_rows),
        }

    def save_predictions_csv(self, raw_trial_df, setting_name="A"):
        setting_name = str(setting_name).upper()
        out_path = os.path.join(
            self.out_dir,
            f"predictions_setting{setting_name}_{self.dataset_name}_{self.variant_name}_{self.model_name.replace('/', '_')}_split_{self.split_name}.csv",
        )
        raw_trial_df.to_csv(out_path, index=False)

    def save_bootstrap_csv(self, bootstrap_df, setting_name="A"):
        setting_name = str(setting_name).upper()
        out_path = os.path.join(
            self.out_dir,
            f"bootstrap_setting{setting_name}_{self.dataset_name}_{self.variant_name}_{self.model_name.replace('/', '_')}_split_{self.split_name}.csv",
        )
        bootstrap_df.to_csv(out_path, index=False)

    def save_summary_csv(self, summary_df, setting_name="A"):
        setting_name = str(setting_name).upper()
        out_path = os.path.join(
            self.out_dir,
            f"summary_setting{setting_name}_{self.dataset_name}_{self.variant_name}_{self.model_name.replace('/', '_')}_split_{self.split_name}.csv",
        )
        summary_df.to_csv(out_path, index=False)

    def print_and_save_stats(self, summary_df, setting_name="A"):
        setting_name = str(setting_name).upper()

        msg = (
            f"\n\n----------------------------------------------------------------------------------------\n"
            f"setting: {setting_name}\n"
            f"dataset: {self.dataset_name}\n"
            f"model: {self.model_name}\n"
            f"variant: {self.variant_name}\n"
            f"split: {self.split_name}\n\n"
        )

        for _, row in summary_df.iterrows():
            msg += (
                f"N={int(row['N'])}\n"
                f"Recall@1: {row['recall@1']}\n"
                f"Recall@5: {row['recall@5']}\n"
                f"Recall@10: {row['recall@10']}\n"
                f"MRR: {row['mrr']}\n\n"
            )

        print(msg)

        out_path = os.path.join(
            self.out_dir,
            f"Test_on_setting{setting_name}_{self.dataset_name}_{self.variant_name}_{self.model_name.replace('/', '_')}_split_{self.split_name}.txt",
        )
        with open(out_path, "a") as f:
            f.write(msg)






class PredictionCrossModalSettingB:
    def __init__(self, cfg_path: str, split_name: str, dataset_name: str, model_name: str, variant_name: str):
        self.params = read_config(cfg_path)
        self.cfg_path = cfg_path
        self.split_name = str(split_name)
        self.dataset_name = str(dataset_name).lower()
        self.model_name = str(model_name)
        self.variant_name = str(variant_name)

        self.setup_cuda()

        self.batch_size = int(self.params["CrossModal"]["batch_size"])
        self.max_text_tokens = int(self.params["CrossModal"]["max_text_tokens_clip_pubmed"])
        self.base_seed = int(self.params["CrossModal"]["base_seed"])
        self.n_trials = int(self.params["CrossModal"]["n_trials_settingB"])
        self.n_bootstrap = int(self.params["CrossModal"]["n_bootstrap"])
        self.settingB_N = int(self.params["CrossModal"].get("settingB_N", 10000))

        self.out_dir = os.path.join(
            self.params["target_dir"],
            self.params["stat_log_path"],
            f"{self.dataset_name}_{self.variant_name}_{self.model_name.replace('/', '_')}_split_{self.split_name}",
        )
        os.makedirs(self.out_dir, exist_ok=True)

        self.bootstrap_dir = os.path.join(
            self.params["target_dir"],
            self.params["stat_log_path"],
            "fixed_bootstrap_indices",
        )
        os.makedirs(self.bootstrap_dir, exist_ok=True)

    def setup_cuda(self, cuda_device_id: int = 0):
        if torch.cuda.is_available():
            torch.backends.cudnn.fastest = True
            torch.cuda.set_device(cuda_device_id)
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def setup_model(self, trained_model_path=None):
        self.processor = None
        self.tokenizer = None

        try:
            self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
        except Exception:
            self.processor = None

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                use_fast=True,
            )
        except Exception:
            self.tokenizer = None

        self.model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            use_safetensors=False,
        ).to(self.device)

        if trained_model_path is not None:
            state = torch.load(trained_model_path, map_location="cpu")
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]

            missing, unexpected = self.model.load_state_dict(state, strict=False)
            print("Loaded trained checkpoint")
            print("Missing keys:", missing)
            print("Unexpected keys:", unexpected)

        self.model.eval()

    def _unwrap_to_tensor(self, x):
        if isinstance(x, torch.Tensor):
            return x

        if hasattr(x, "image_embeds") and x.image_embeds is not None:
            return x.image_embeds

        if hasattr(x, "text_embeds") and x.text_embeds is not None:
            return x.text_embeds

        if hasattr(x, "pooler_output") and x.pooler_output is not None:
            return x.pooler_output

        if hasattr(x, "last_hidden_state") and x.last_hidden_state is not None:
            return x.last_hidden_state[:, 0]

        if isinstance(x, (tuple, list)) and len(x) > 0:
            first = x[0]
            if isinstance(first, torch.Tensor):
                if first.ndim == 3:
                    return first[:, 0]
                return first

        raise TypeError(f"Could not convert output of type {type(x)} to embedding tensor.")

    def _manual_image_preprocess(self, images):
        pixel_tensors = []

        for img in images:
            img = img.convert("RGB")
            img = img.resize((224, 224))
            arr = np.asarray(img).astype(np.float32) / 255.0
            arr = (arr - 0.5) / 0.5
            arr = np.transpose(arr, (2, 0, 1))
            pixel_tensors.append(torch.tensor(arr, dtype=torch.float32))

        pixel_values = torch.stack(pixel_tensors, dim=0)
        return {"pixel_values": pixel_values.to(self.device, non_blocking=True)}

    def _encode_images_batch(self, image_paths):
        images = [Image.open(p).convert("RGB") for p in image_paths]

        inputs = None

        if self.processor is not None:
            try:
                inputs = self.processor(images=images, return_tensors="pt")
                inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}
            except Exception:
                inputs = None

        if inputs is None:
            inputs = self._manual_image_preprocess(images)

        with torch.no_grad():
            if hasattr(self.model, "get_image_features"):
                z = self.model.get_image_features(**inputs)
            elif hasattr(self.model, "get_projected_image_embeddings"):
                z = self.model.get_projected_image_embeddings(**inputs)
            elif hasattr(self.model, "get_vision_features"):
                z = self.model.get_vision_features(**inputs)
            else:
                z = self.model(**inputs)

        z = self._unwrap_to_tensor(z)
        z = F.normalize(z.float(), p=2, dim=-1)
        return z

    def _encode_texts_batch(self, texts):
        tok = None
        max_len = int(self.max_text_tokens)

        if self.processor is not None:
            try:
                tok = self.processor(
                    text=texts,
                    padding=True,
                    truncation=True,
                    max_length=max_len,
                    return_tensors="pt",
                )
            except Exception:
                tok = None

        if tok is None:
            if self.tokenizer is None:
                raise ValueError(f"No tokenizer available for model {self.model_name}")
            tok = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            )

        tok = {k: v.to(self.device, non_blocking=True) for k, v in tok.items()}

        with torch.no_grad():
            if hasattr(self.model, "get_text_features"):
                z = self.model.get_text_features(**tok)
            elif hasattr(self.model, "get_text_embeddings"):
                z = self.model.get_text_embeddings(**tok)
            elif hasattr(self.model, "get_projected_text_embeddings"):
                z = self.model.get_projected_text_embeddings(
                    input_ids=tok["input_ids"],
                    attention_mask=tok["attention_mask"],
                )
            else:
                z = self.model(**tok)

        z = self._unwrap_to_tensor(z)
        z = F.normalize(z.float(), p=2, dim=-1)
        return z

    def extract_embeddings(self, test_loader):
        self.model.eval()

        z_img_cache = []
        z_txt_cache = []
        row_ids_cache = []
        subject_ids_cache = []
        study_ids_cache = []
        labels_cache = []

        with torch.no_grad():
            for batch in tqdm(test_loader):
                z_img = self._encode_images_batch(batch["image_paths"])
                z_txt = self._encode_texts_batch(batch["report_texts"])

                z_img_cache.append(z_img.detach().cpu())
                z_txt_cache.append(z_txt.detach().cpu())

                row_ids_cache.extend(batch["row_ids"])
                subject_ids_cache.extend(batch["subject_ids"])
                study_ids_cache.extend(batch["study_ids"])
                labels_cache.append(batch["labels"].detach().cpu())

        Z_img = torch.cat(z_img_cache, dim=0).to(torch.float16)
        Z_txt = torch.cat(z_txt_cache, dim=0).to(torch.float16)
        labels_all = torch.cat(labels_cache, dim=0).numpy().astype(np.int8)

        return {
            "Z_img": Z_img,
            "Z_txt": Z_txt,
            "row_ids": np.asarray(row_ids_cache),
            "subject_ids": np.asarray(subject_ids_cache),
            "study_ids": np.asarray(study_ids_cache),
            "labels": labels_all,
        }

    def _seed_for_setting_b(self, trial_id):
        return self.base_seed + 777 + trial_id

    def _compute_metrics_from_rank(self, rank_1based):
        return {
            "recall@1": 1.0 if rank_1based <= 1 else 0.0,
            "recall@5": 1.0 if rank_1based <= 5 else 0.0,
            "recall@10": 1.0 if rank_1based <= 10 else 0.0,
            "mrr": 1.0 / float(rank_1based),
        }

    def _evaluate_one_query_one_trial(self, z_img_i, Z_txt_device, query_idx, cand_idx):
        cand_idx_t = torch.as_tensor(cand_idx, device=Z_txt_device.device, dtype=torch.long)
        cand_txt = Z_txt_device.index_select(0, cand_idx_t)
        scores = torch.matmul(cand_txt, z_img_i)

        order = torch.argsort(scores, descending=True)

        true_pos = np.where(np.asarray(cand_idx) == query_idx)[0]
        if len(true_pos) != 1:
            raise ValueError("True report missing or duplicated in candidate pool.")
        true_pos = int(true_pos[0])

        rank_1based = int((order == true_pos).nonzero(as_tuple=False).item()) + 1

        out = self._compute_metrics_from_rank(rank_1based)
        out["rank"] = rank_1based
        return out

    def _get_or_create_bootstrap_indices(self, n_queries, seed=13):
        bs_path = os.path.join(
            self.bootstrap_dir,
            f"bootstrap_indices_{self.dataset_name}_{self.split_name}_n{n_queries}_b{self.n_bootstrap}_seed{seed}.npy",
        )

        if os.path.exists(bs_path):
            return np.load(bs_path)

        rng = np.random.RandomState(seed)
        indices = np.zeros((self.n_bootstrap, n_queries), dtype=np.int64)

        for i in range(self.n_bootstrap):
            indices[i] = rng.choice(n_queries, n_queries, replace=True)

        np.save(bs_path, indices)
        return indices

    def bootstrap_metrics(self, per_query_df, bootstrap_indices):
        metric_names = ["recall@1", "recall@5", "recall@10", "mrr"]
        boot_dict = {}

        for metric in metric_names:
            vals = per_query_df[metric].to_numpy(dtype=np.float64)
            arr = np.zeros((len(bootstrap_indices),), dtype=np.float32)

            for i, idx in enumerate(bootstrap_indices):
                arr[i] = float(vals[idx].mean())

            boot_dict[metric] = arr

        return boot_dict

    def ci95(self, x):
        lo, hi = np.percentile(x, [2.5, 97.5])
        return lo, hi

    def _summary_str_pct(self, arr: np.ndarray) -> str:
        arr = np.asarray(arr, dtype=np.float32)
        mean = float(arr.mean()) * 100.0
        std = float(arr.std()) * 100.0
        lo, hi = self.ci95(arr)
        lo *= 100.0
        hi *= 100.0
        return f"{mean:.4f} ± {std:.4f} [{lo:.4f}, {hi:.4f}]"

    def _build_label_buckets(self, labels_np):
        vector_to_indices = {}
        for idx, vec in enumerate(labels_np):
            key = tuple(vec.tolist())
            if key not in vector_to_indices:
                vector_to_indices[key] = []
            vector_to_indices[key].append(idx)
        return vector_to_indices

    def _sample_from_pool(self, pool, n_needed, rng):
        if n_needed <= 0 or len(pool) == 0:
            return []

        pool = np.asarray(pool, dtype=np.int64)
        if len(pool) <= n_needed:
            out = pool.copy()
            rng.shuffle(out)
            return out.tolist()

        chosen = rng.choice(pool, size=n_needed, replace=False)
        return chosen.tolist()

    def _build_setting_b_candidate_pool(self, query_idx, labels_np, vector_to_indices, rng):
        """
        Setting B:
        - include true report
        - fill negatives from exact same label vector first
        - if insufficient, fill by ascending Hamming distance
        - N fixed to self.settingB_N
        """
        M = len(labels_np)
        if self.settingB_N > M:
            raise ValueError(f"Setting B requires N={self.settingB_N}, but test set has only M={M} rows.")

        query_vec = labels_np[query_idx]
        query_key = tuple(query_vec.tolist())

        negatives = []

        # exact-match negatives
        exact_pool = [idx for idx in vector_to_indices.get(query_key, []) if idx != query_idx]
        negatives.extend(self._sample_from_pool(exact_pool, self.settingB_N - 1, rng))

        remaining = (self.settingB_N - 1) - len(negatives)
        if remaining <= 0:
            cand_idx = np.asarray([query_idx] + negatives, dtype=np.int64)
            rng.shuffle(cand_idx)
            return cand_idx

        # fill by increasing Hamming distance
        all_keys = list(vector_to_indices.keys())
        used = set(negatives)
        used.add(query_idx)

        for dist in range(1, labels_np.shape[1] + 1):
            bucket = []

            for key in all_keys:
                if key == query_key:
                    continue

                key_arr = np.asarray(key, dtype=np.int8)
                hamming_dist = int(np.sum(key_arr != query_vec))

                if hamming_dist == dist:
                    for idx in vector_to_indices[key]:
                        if idx not in used:
                            bucket.append(idx)

            chosen = self._sample_from_pool(bucket, remaining, rng)
            negatives.extend(chosen)
            used.update(chosen)

            remaining = (self.settingB_N - 1) - len(negatives)
            if remaining <= 0:
                break

        if len(negatives) != self.settingB_N - 1:
            raise ValueError(
                f"Could not build Setting B pool for query {query_idx}. "
                f"Needed {self.settingB_N - 1} negatives, got {len(negatives)}."
            )

        cand_idx = np.asarray([query_idx] + negatives, dtype=np.int64)
        rng.shuffle(cand_idx)
        return cand_idx

    def _make_numpy_rng(self, seed_val):
        seed_val = int(seed_val) % (2 ** 32 - 1)
        return np.random.RandomState(seed_val)

    def run_settingB(self, emb_dict):
        Z_img = emb_dict["Z_img"].float().to(self.device)
        Z_txt = emb_dict["Z_txt"].float().to(self.device)

        row_ids = emb_dict["row_ids"]
        subject_ids = emb_dict["subject_ids"]
        study_ids = emb_dict["study_ids"]
        labels_np = emb_dict["labels"].astype(np.int8)

        M = len(row_ids)
        if self.settingB_N > M:
            raise ValueError(f"Setting B requires N={self.settingB_N}, but test set size is only {M}.")

        vector_to_indices = self._build_label_buckets(labels_np)
        bootstrap_indices = self._get_or_create_bootstrap_indices(n_queries=M, seed=13)

        all_summary_rows = []
        raw_trial_rows = []
        all_boot_rows = []

        N = self.settingB_N
        print(f"\nRunning Setting B | dataset={self.dataset_name} | model={self.model_name} | variant={self.variant_name} | N={N}")

        per_query_rows = []

        for q_idx in tqdm(range(M)):
            z_img_i = Z_img[q_idx]
            trial_metrics = []

            for trial_id in range(self.n_trials):
                seed = self._seed_for_setting_b(trial_id)
                rng = self._make_numpy_rng(seed + q_idx * 100003)

                cand_idx = self._build_setting_b_candidate_pool(
                    query_idx=q_idx,
                    labels_np=labels_np,
                    vector_to_indices=vector_to_indices,
                    rng=rng,
                )

                m = self._evaluate_one_query_one_trial(
                    z_img_i=z_img_i,
                    Z_txt_device=Z_txt,
                    query_idx=q_idx,
                    cand_idx=cand_idx,
                )
                trial_metrics.append(m)

                raw_trial_rows.append({
                    "dataset_name": self.dataset_name,
                    "model_name": self.model_name,
                    "variant_name": self.variant_name,
                    "N": N,
                    "query_row_id": int(row_ids[q_idx]),
                    "subject_id": int(subject_ids[q_idx]),
                    "study_id": int(study_ids[q_idx]),
                    "trial_id": int(trial_id),
                    "rank": int(m["rank"]),
                    "recall@1": float(m["recall@1"]),
                    "recall@5": float(m["recall@5"]),
                    "recall@10": float(m["recall@10"]),
                    "mrr": float(m["mrr"]),
                })

            per_query_rows.append({
                "dataset_name": self.dataset_name,
                "model_name": self.model_name,
                "variant_name": self.variant_name,
                "N": N,
                "query_row_id": int(row_ids[q_idx]),
                "subject_id": int(subject_ids[q_idx]),
                "study_id": int(study_ids[q_idx]),
                "recall@1": float(np.mean([m["recall@1"] for m in trial_metrics])),
                "recall@5": float(np.mean([m["recall@5"] for m in trial_metrics])),
                "recall@10": float(np.mean([m["recall@10"] for m in trial_metrics])),
                "mrr": float(np.mean([m["mrr"] for m in trial_metrics])),
            })

        per_query_df = pd.DataFrame(per_query_rows)
        boot_dict = self.bootstrap_metrics(per_query_df=per_query_df, bootstrap_indices=bootstrap_indices)

        for i in range(len(boot_dict["recall@1"])):
            all_boot_rows.append({
                "dataset_name": self.dataset_name,
                "model_name": self.model_name,
                "variant_name": self.variant_name,
                "N": N,
                "bootstrap_id": i,
                "recall@1": float(boot_dict["recall@1"][i]),
                "recall@5": float(boot_dict["recall@5"][i]),
                "recall@10": float(boot_dict["recall@10"][i]),
                "mrr": float(boot_dict["mrr"][i]),
            })

        all_summary_rows.append({
            "dataset_name": self.dataset_name,
            "model_name": self.model_name,
            "variant_name": self.variant_name,
            "split_name": self.split_name,
            "N": int(N),
            "num_queries": int(len(per_query_df)),
            "num_trials_per_query": int(self.n_trials),
            "recall@1": self._summary_str_pct(boot_dict["recall@1"]),
            "recall@5": self._summary_str_pct(boot_dict["recall@5"]),
            "recall@10": self._summary_str_pct(boot_dict["recall@10"]),
            "mrr": self._summary_str_pct(boot_dict["mrr"]),
        })

        return {
            "summary_df": pd.DataFrame(all_summary_rows),
            "raw_trial_df": pd.DataFrame(raw_trial_rows),
            "bootstrap_df": pd.DataFrame(all_boot_rows),
        }

    def save_predictions_csv(self, raw_trial_df, setting_name="B"):
        setting_name = str(setting_name).upper()
        out_path = os.path.join(
            self.out_dir,
            f"predictions_setting{setting_name}_{self.dataset_name}_{self.variant_name}_{self.model_name.replace('/', '_')}_split_{self.split_name}.csv",
        )
        raw_trial_df.to_csv(out_path, index=False)

    def save_bootstrap_csv(self, bootstrap_df, setting_name="B"):
        setting_name = str(setting_name).upper()
        out_path = os.path.join(
            self.out_dir,
            f"bootstrap_setting{setting_name}_{self.dataset_name}_{self.variant_name}_{self.model_name.replace('/', '_')}_split_{self.split_name}.csv",
        )
        bootstrap_df.to_csv(out_path, index=False)

    def save_summary_csv(self, summary_df, setting_name="B"):
        setting_name = str(setting_name).upper()
        out_path = os.path.join(
            self.out_dir,
            f"summary_setting{setting_name}_{self.dataset_name}_{self.variant_name}_{self.model_name.replace('/', '_')}_split_{self.split_name}.csv",
        )
        summary_df.to_csv(out_path, index=False)

    def print_and_save_stats(self, summary_df, setting_name="B"):
        setting_name = str(setting_name).upper()

        msg = (
            f"\n\n----------------------------------------------------------------------------------------\n"
            f"setting: {setting_name}\n"
            f"dataset: {self.dataset_name}\n"
            f"model: {self.model_name}\n"
            f"variant: {self.variant_name}\n"
            f"split: {self.split_name}\n\n"
        )

        for _, row in summary_df.iterrows():
            msg += (
                f"N={int(row['N'])}\n"
                f"Recall@1: {row['recall@1']}\n"
                f"Recall@5: {row['recall@5']}\n"
                f"Recall@10: {row['recall@10']}\n"
                f"MRR: {row['mrr']}\n\n"
            )

        print(msg)

        out_path = os.path.join(
            self.out_dir,
            f"Test_on_setting{setting_name}_{self.dataset_name}_{self.variant_name}_{self.model_name.replace('/', '_')}_split_{self.split_name}.txt",
        )
        with open(out_path, "a") as f:
            f.write(msg)







class PredictionCrossModalSettingB_BiomedCLIP:
    def __init__(self, cfg_path: str, split_name: str, dataset_name: str, variant_name: str):
        self.params = read_config(cfg_path)
        self.cfg_path = cfg_path
        self.split_name = str(split_name)
        self.dataset_name = str(dataset_name).lower()
        self.model_name = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        self.variant_name = str(variant_name)

        self.setup_cuda()

        self.batch_size = int(self.params["CrossModal"]["batch_size"])
        self.max_text_tokens = int(self.params["CrossModal"]["max_text_tokens_biovilt"])
        self.base_seed = int(self.params["CrossModal"]["base_seed"])
        self.n_trials = int(self.params["CrossModal"]["n_trials_settingB"])
        self.n_bootstrap = int(self.params["CrossModal"]["n_bootstrap"])
        self.settingB_N = int(self.params["CrossModal"].get("settingB_N", 10000))

        self.out_dir = os.path.join(
            self.params["target_dir"],
            self.params["stat_log_path"],
            f"{self.dataset_name}_{self.variant_name}_biomedclip_split_{self.split_name}",
        )
        os.makedirs(self.out_dir, exist_ok=True)

        self.bootstrap_dir = os.path.join(
            self.params["target_dir"],
            self.params["stat_log_path"],
            "fixed_bootstrap_indices",
        )
        os.makedirs(self.bootstrap_dir, exist_ok=True)

    def setup_cuda(self, cuda_device_id: int = 0):
        if torch.cuda.is_available():
            torch.backends.cudnn.fastest = True
            torch.cuda.set_device(cuda_device_id)
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def setup_model(self, trained_model_path=None):
        model_id = f"hf-hub:{self.model_name}"

        self.model, self.preprocess = open_clip.create_model_from_pretrained(model_id)
        self.tokenizer = open_clip.get_tokenizer(model_id)
        self.model = self.model.to(self.device)

        if trained_model_path is not None:
            state = torch.load(trained_model_path, map_location="cpu")
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]

            missing, unexpected = self.model.load_state_dict(state, strict=False)
            print("Loaded trained checkpoint")
            print("Missing keys:", missing)
            print("Unexpected keys:", unexpected)

        self.model.eval()

    def _encode_images_batch(self, image_paths):
        images = [self.preprocess(Image.open(p).convert("RGB")) for p in image_paths]
        images = torch.stack(images, dim=0).to(self.device, non_blocking=True)

        with torch.no_grad():
            z = self.model.encode_image(images)

        z = F.normalize(z.float(), p=2, dim=-1)
        return z

    def _encode_texts_batch(self, texts):
        tok = self.tokenizer(texts, context_length=int(self.max_text_tokens))
        tok = tok.to(self.device, non_blocking=True)

        with torch.no_grad():
            z = self.model.encode_text(tok)

        z = F.normalize(z.float(), p=2, dim=-1)
        return z

    def extract_embeddings(self, test_loader):
        self.model.eval()

        z_img_cache = []
        z_txt_cache = []
        row_ids_cache = []
        subject_ids_cache = []
        study_ids_cache = []
        labels_cache = []

        with torch.no_grad():
            for batch in tqdm(test_loader):
                z_img = self._encode_images_batch(batch["image_paths"])
                z_txt = self._encode_texts_batch(batch["report_texts"])

                z_img_cache.append(z_img.detach().cpu())
                z_txt_cache.append(z_txt.detach().cpu())

                row_ids_cache.extend(batch["row_ids"])
                subject_ids_cache.extend(batch["subject_ids"])
                study_ids_cache.extend(batch["study_ids"])
                labels_cache.append(batch["labels"].detach().cpu())

        Z_img = torch.cat(z_img_cache, dim=0).to(torch.float16)
        Z_txt = torch.cat(z_txt_cache, dim=0).to(torch.float16)
        labels_all = torch.cat(labels_cache, dim=0).numpy().astype(np.int8)

        return {
            "Z_img": Z_img,
            "Z_txt": Z_txt,
            "row_ids": np.asarray(row_ids_cache),
            "subject_ids": np.asarray(subject_ids_cache),
            "study_ids": np.asarray(study_ids_cache),
            "labels": labels_all,
        }

    def _seed_for_setting_b(self, trial_id):
        return self.base_seed + 777 + trial_id

    def _compute_metrics_from_rank(self, rank_1based):
        return {
            "recall@1": 1.0 if rank_1based <= 1 else 0.0,
            "recall@5": 1.0 if rank_1based <= 5 else 0.0,
            "recall@10": 1.0 if rank_1based <= 10 else 0.0,
            "mrr": 1.0 / float(rank_1based),
        }

    def _evaluate_one_query_one_trial(self, z_img_i, Z_txt_device, query_idx, cand_idx):
        cand_idx_t = torch.as_tensor(cand_idx, device=Z_txt_device.device, dtype=torch.long)
        cand_txt = Z_txt_device.index_select(0, cand_idx_t)
        scores = torch.matmul(cand_txt, z_img_i)

        order = torch.argsort(scores, descending=True)

        true_pos = np.where(np.asarray(cand_idx) == query_idx)[0]
        if len(true_pos) != 1:
            raise ValueError("True report missing or duplicated in candidate pool.")
        true_pos = int(true_pos[0])

        rank_1based = int((order == true_pos).nonzero(as_tuple=False).item()) + 1

        out = self._compute_metrics_from_rank(rank_1based)
        out["rank"] = rank_1based
        return out

    def _get_or_create_bootstrap_indices(self, n_queries, seed=13):
        bs_path = os.path.join(
            self.bootstrap_dir,
            f"bootstrap_indices_{self.dataset_name}_{self.split_name}_n{n_queries}_b{self.n_bootstrap}_seed{seed}.npy",
        )

        if os.path.exists(bs_path):
            return np.load(bs_path)

        rng = np.random.RandomState(seed)
        indices = np.zeros((self.n_bootstrap, n_queries), dtype=np.int64)

        for i in range(self.n_bootstrap):
            indices[i] = rng.choice(n_queries, n_queries, replace=True)

        np.save(bs_path, indices)
        return indices

    def bootstrap_metrics(self, per_query_df, bootstrap_indices):
        metric_names = ["recall@1", "recall@5", "recall@10", "mrr"]
        boot_dict = {}

        for metric in metric_names:
            vals = per_query_df[metric].to_numpy(dtype=np.float64)
            arr = np.zeros((len(bootstrap_indices),), dtype=np.float32)

            for i, idx in enumerate(bootstrap_indices):
                arr[i] = float(vals[idx].mean())

            boot_dict[metric] = arr

        return boot_dict

    def ci95(self, x):
        lo, hi = np.percentile(x, [2.5, 97.5])
        return lo, hi

    def _summary_str_pct(self, arr: np.ndarray) -> str:
        arr = np.asarray(arr, dtype=np.float32)
        mean = float(arr.mean()) * 100.0
        std = float(arr.std()) * 100.0
        lo, hi = self.ci95(arr)
        lo *= 100.0
        hi *= 100.0
        return f"{mean:.4f} ± {std:.4f} [{lo:.4f}, {hi:.4f}]"

    def _build_label_buckets(self, labels_np):
        vector_to_indices = {}
        for idx, vec in enumerate(labels_np):
            key = tuple(vec.tolist())
            if key not in vector_to_indices:
                vector_to_indices[key] = []
            vector_to_indices[key].append(idx)
        return vector_to_indices

    def _sample_from_pool(self, pool, n_needed, rng):
        if n_needed <= 0 or len(pool) == 0:
            return []

        pool = np.asarray(pool, dtype=np.int64)
        if len(pool) <= n_needed:
            out = pool.copy()
            rng.shuffle(out)
            return out.tolist()

        chosen = rng.choice(pool, size=n_needed, replace=False)
        return chosen.tolist()

    def _build_setting_b_candidate_pool(self, query_idx, labels_np, vector_to_indices, rng):
        M = len(labels_np)
        if self.settingB_N > M:
            raise ValueError(f"Setting B requires N={self.settingB_N}, but test set has only M={M} rows.")

        query_vec = labels_np[query_idx]
        query_key = tuple(query_vec.tolist())

        negatives = []

        exact_pool = [idx for idx in vector_to_indices.get(query_key, []) if idx != query_idx]
        negatives.extend(self._sample_from_pool(exact_pool, self.settingB_N - 1, rng))

        remaining = (self.settingB_N - 1) - len(negatives)
        if remaining <= 0:
            cand_idx = np.asarray([query_idx] + negatives, dtype=np.int64)
            rng.shuffle(cand_idx)
            return cand_idx

        all_keys = list(vector_to_indices.keys())
        used = set(negatives)
        used.add(query_idx)

        for dist in range(1, labels_np.shape[1] + 1):
            bucket = []

            for key in all_keys:
                if key == query_key:
                    continue

                key_arr = np.asarray(key, dtype=np.int8)
                hamming_dist = int(np.sum(key_arr != query_vec))

                if hamming_dist == dist:
                    for idx in vector_to_indices[key]:
                        if idx not in used:
                            bucket.append(idx)

            chosen = self._sample_from_pool(bucket, remaining, rng)
            negatives.extend(chosen)
            used.update(chosen)

            remaining = (self.settingB_N - 1) - len(negatives)
            if remaining <= 0:
                break

        if len(negatives) != self.settingB_N - 1:
            raise ValueError(
                f"Could not build Setting B pool for query {query_idx}. "
                f"Needed {self.settingB_N - 1} negatives, got {len(negatives)}."
            )

        cand_idx = np.asarray([query_idx] + negatives, dtype=np.int64)
        rng.shuffle(cand_idx)
        return cand_idx

    def _make_numpy_rng(self, seed_val):
        seed_val = int(seed_val) % (2 ** 32 - 1)
        return np.random.RandomState(seed_val)

    def run_settingB(self, emb_dict):
        Z_img = emb_dict["Z_img"].float().to(self.device)
        Z_txt = emb_dict["Z_txt"].float().to(self.device)

        row_ids = emb_dict["row_ids"]
        subject_ids = emb_dict["subject_ids"]
        study_ids = emb_dict["study_ids"]
        labels_np = emb_dict["labels"].astype(np.int8)

        M = len(row_ids)
        if self.settingB_N > M:
            raise ValueError(f"Setting B requires N={self.settingB_N}, but test set size is only {M}.")

        vector_to_indices = self._build_label_buckets(labels_np)
        bootstrap_indices = self._get_or_create_bootstrap_indices(n_queries=M, seed=13)

        all_summary_rows = []
        raw_trial_rows = []
        all_boot_rows = []

        N = self.settingB_N
        print(f"\nRunning Setting B | dataset={self.dataset_name} | model={self.model_name} | variant={self.variant_name} | N={N}")

        per_query_rows = []

        for q_idx in tqdm(range(M)):
            z_img_i = Z_img[q_idx]
            trial_metrics = []

            for trial_id in range(self.n_trials):
                seed = self._seed_for_setting_b(trial_id)
                rng = self._make_numpy_rng(seed + q_idx * 100003)

                cand_idx = self._build_setting_b_candidate_pool(
                    query_idx=q_idx,
                    labels_np=labels_np,
                    vector_to_indices=vector_to_indices,
                    rng=rng,
                )

                m = self._evaluate_one_query_one_trial(
                    z_img_i=z_img_i,
                    Z_txt_device=Z_txt,
                    query_idx=q_idx,
                    cand_idx=cand_idx,
                )
                trial_metrics.append(m)

                raw_trial_rows.append({
                    "dataset_name": self.dataset_name,
                    "model_name": self.model_name,
                    "variant_name": self.variant_name,
                    "N": N,
                    "query_row_id": int(row_ids[q_idx]),
                    "subject_id": int(subject_ids[q_idx]),
                    "study_id": int(study_ids[q_idx]),
                    "trial_id": int(trial_id),
                    "rank": int(m["rank"]),
                    "recall@1": float(m["recall@1"]),
                    "recall@5": float(m["recall@5"]),
                    "recall@10": float(m["recall@10"]),
                    "mrr": float(m["mrr"]),
                })

            per_query_rows.append({
                "dataset_name": self.dataset_name,
                "model_name": self.model_name,
                "variant_name": self.variant_name,
                "N": N,
                "query_row_id": int(row_ids[q_idx]),
                "subject_id": int(subject_ids[q_idx]),
                "study_id": int(study_ids[q_idx]),
                "recall@1": float(np.mean([m["recall@1"] for m in trial_metrics])),
                "recall@5": float(np.mean([m["recall@5"] for m in trial_metrics])),
                "recall@10": float(np.mean([m["recall@10"] for m in trial_metrics])),
                "mrr": float(np.mean([m["mrr"] for m in trial_metrics])),
            })

        per_query_df = pd.DataFrame(per_query_rows)
        boot_dict = self.bootstrap_metrics(per_query_df=per_query_df, bootstrap_indices=bootstrap_indices)

        for i in range(len(boot_dict["recall@1"])):
            all_boot_rows.append({
                "dataset_name": self.dataset_name,
                "model_name": self.model_name,
                "variant_name": self.variant_name,
                "N": N,
                "bootstrap_id": i,
                "recall@1": float(boot_dict["recall@1"][i]),
                "recall@5": float(boot_dict["recall@5"][i]),
                "recall@10": float(boot_dict["recall@10"][i]),
                "mrr": float(boot_dict["mrr"][i]),
            })

        all_summary_rows.append({
            "dataset_name": self.dataset_name,
            "model_name": self.model_name,
            "variant_name": self.variant_name,
            "split_name": self.split_name,
            "N": int(N),
            "num_queries": int(len(per_query_df)),
            "num_trials_per_query": int(self.n_trials),
            "recall@1": self._summary_str_pct(boot_dict["recall@1"]),
            "recall@5": self._summary_str_pct(boot_dict["recall@5"]),
            "recall@10": self._summary_str_pct(boot_dict["recall@10"]),
            "mrr": self._summary_str_pct(boot_dict["mrr"]),
        })

        return {
            "summary_df": pd.DataFrame(all_summary_rows),
            "raw_trial_df": pd.DataFrame(raw_trial_rows),
            "bootstrap_df": pd.DataFrame(all_boot_rows),
        }

    def save_predictions_csv(self, raw_trial_df, setting_name="B"):
        setting_name = str(setting_name).upper()
        out_path = os.path.join(
            self.out_dir,
            f"predictions_setting{setting_name}_{self.dataset_name}_{self.variant_name}_biomedclip_split_{self.split_name}.csv",
        )
        raw_trial_df.to_csv(out_path, index=False)

    def save_bootstrap_csv(self, bootstrap_df, setting_name="B"):
        setting_name = str(setting_name).upper()
        out_path = os.path.join(
            self.out_dir,
            f"bootstrap_setting{setting_name}_{self.dataset_name}_{self.variant_name}_biomedclip_split_{self.split_name}.csv",
        )
        bootstrap_df.to_csv(out_path, index=False)

    def save_summary_csv(self, summary_df, setting_name="B"):
        setting_name = str(setting_name).upper()
        out_path = os.path.join(
            self.out_dir,
            f"summary_setting{setting_name}_{self.dataset_name}_{self.variant_name}_biomedclip_split_{self.split_name}.csv",
        )
        summary_df.to_csv(out_path, index=False)

    def print_and_save_stats(self, summary_df, setting_name="B"):
        setting_name = str(setting_name).upper()

        msg = (
            f"\n\n----------------------------------------------------------------------------------------\n"
            f"setting: {setting_name}\n"
            f"dataset: {self.dataset_name}\n"
            f"model: {self.model_name}\n"
            f"variant: {self.variant_name}\n"
            f"split: {self.split_name}\n\n"
        )

        for _, row in summary_df.iterrows():
            msg += (
                f"N={int(row['N'])}\n"
                f"Recall@1: {row['recall@1']}\n"
                f"Recall@5: {row['recall@5']}\n"
                f"Recall@10: {row['recall@10']}\n"
                f"MRR: {row['mrr']}\n\n"
            )

        print(msg)

        out_path = os.path.join(
            self.out_dir,
            f"Test_on_setting{setting_name}_{self.dataset_name}_{self.variant_name}_biomedclip_split_{self.split_name}.txt",
        )
        with open(out_path, "a") as f:
            f.write(msg)







class PredictionCrossModalSettingB_BioViLT:
    def __init__(self, cfg_path: str, split_name: str, dataset_name: str, variant_name: str):
        self.params = read_config(cfg_path)
        self.cfg_path = cfg_path
        self.split_name = str(split_name)
        self.dataset_name = str(dataset_name).lower()
        self.model_name = "microsoft/BiomedVLP-BioViL-T"
        self.variant_name = str(variant_name)

        self.setup_cuda()

        self.batch_size = int(self.params["CrossModal"]["batch_size"])
        self.max_text_tokens = int(self.params["CrossModal"]["max_text_tokens_biovilt"])
        self.base_seed = int(self.params["CrossModal"]["base_seed"])
        self.n_trials = int(self.params["CrossModal"]["n_trials_settingB"])
        self.n_bootstrap = int(self.params["CrossModal"]["n_bootstrap"])
        self.settingB_N = int(self.params["CrossModal"].get("settingB_N", 10000))

        self.out_dir = os.path.join(
            self.params["target_dir"],
            self.params["stat_log_path"],
            f"{self.dataset_name}_{self.variant_name}_biovilt_split_{self.split_name}",
        )
        os.makedirs(self.out_dir, exist_ok=True)

        self.bootstrap_dir = os.path.join(
            self.params["target_dir"],
            self.params["stat_log_path"],
            "fixed_bootstrap_indices",
        )
        os.makedirs(self.bootstrap_dir, exist_ok=True)

    def setup_cuda(self, cuda_device_id: int = 0):
        if torch.cuda.is_available():
            torch.backends.cudnn.fastest = True
            torch.cuda.set_device(cuda_device_id)
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def setup_model(self, trained_model_path=None):
        self.image_engine = get_image_inference(ImageModelType.BIOVIL_T)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            use_fast=True,
        )

        self.text_model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            use_safetensors=False,
        ).to(self.device)

        if trained_model_path is not None:
            state = torch.load(trained_model_path, map_location="cpu")
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]

            missing, unexpected = self.text_model.load_state_dict(state, strict=False)
            print("Loaded trained checkpoint")
            print("Missing keys:", missing)
            print("Unexpected keys:", unexpected)

        self.text_model.eval()

    def _encode_images_batch(self, image_paths):
        z_list = []

        for p in image_paths:
            p = Path(p)
            z = self.image_engine.get_projected_global_embedding(p)

            if z.ndim == 1:
                z = z.unsqueeze(0)

            z = F.normalize(z.float(), p=2, dim=-1)
            z_list.append(z.cpu())

        z = torch.cat(z_list, dim=0).to(self.device)
        return z

    def _encode_texts_batch(self, texts):
        max_len = int(self.max_text_tokens)

        tok = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        tok = {k: v.to(self.device, non_blocking=True) for k, v in tok.items()}

        with torch.no_grad():
            if hasattr(self.text_model, "get_projected_text_embeddings"):
                z = self.text_model.get_projected_text_embeddings(
                    input_ids=tok["input_ids"],
                    attention_mask=tok["attention_mask"],
                )
            elif hasattr(self.text_model, "get_text_features"):
                z = self.text_model.get_text_features(**tok)
            elif hasattr(self.text_model, "get_text_embeddings"):
                z = self.text_model.get_text_embeddings(**tok)
            else:
                raise AttributeError("BioViL-T text model does not expose a supported text embedding method.")

        z = F.normalize(z.float(), p=2, dim=-1)
        return z

    def extract_embeddings(self, test_loader):
        self.text_model.eval()

        z_img_cache = []
        z_txt_cache = []
        row_ids_cache = []
        subject_ids_cache = []
        study_ids_cache = []
        labels_cache = []

        with torch.no_grad():
            for batch in tqdm(test_loader):
                z_img = self._encode_images_batch(batch["image_paths"])
                z_txt = self._encode_texts_batch(batch["report_texts"])

                z_img_cache.append(z_img.detach().cpu())
                z_txt_cache.append(z_txt.detach().cpu())

                row_ids_cache.extend(batch["row_ids"])
                subject_ids_cache.extend(batch["subject_ids"])
                study_ids_cache.extend(batch["study_ids"])
                labels_cache.append(batch["labels"].detach().cpu())

        Z_img = torch.cat(z_img_cache, dim=0).to(torch.float16)
        Z_txt = torch.cat(z_txt_cache, dim=0).to(torch.float16)
        labels_all = torch.cat(labels_cache, dim=0).numpy().astype(np.int8)

        return {
            "Z_img": Z_img,
            "Z_txt": Z_txt,
            "row_ids": np.asarray(row_ids_cache),
            "subject_ids": np.asarray(subject_ids_cache),
            "study_ids": np.asarray(study_ids_cache),
            "labels": labels_all,
        }

    def _seed_for_setting_b(self, trial_id):
        return self.base_seed + 777 + trial_id

    def _compute_metrics_from_rank(self, rank_1based):
        return {
            "recall@1": 1.0 if rank_1based <= 1 else 0.0,
            "recall@5": 1.0 if rank_1based <= 5 else 0.0,
            "recall@10": 1.0 if rank_1based <= 10 else 0.0,
            "mrr": 1.0 / float(rank_1based),
        }

    def _evaluate_one_query_one_trial(self, z_img_i, Z_txt_device, query_idx, cand_idx):
        cand_idx_t = torch.as_tensor(cand_idx, device=Z_txt_device.device, dtype=torch.long)
        cand_txt = Z_txt_device.index_select(0, cand_idx_t)
        scores = torch.matmul(cand_txt, z_img_i)

        order = torch.argsort(scores, descending=True)

        true_pos = np.where(np.asarray(cand_idx) == query_idx)[0]
        if len(true_pos) != 1:
            raise ValueError("True report missing or duplicated in candidate pool.")
        true_pos = int(true_pos[0])

        rank_1based = int((order == true_pos).nonzero(as_tuple=False).item()) + 1

        out = self._compute_metrics_from_rank(rank_1based)
        out["rank"] = rank_1based
        return out

    def _get_or_create_bootstrap_indices(self, n_queries, seed=13):
        bs_path = os.path.join(
            self.bootstrap_dir,
            f"bootstrap_indices_{self.dataset_name}_{self.split_name}_n{n_queries}_b{self.n_bootstrap}_seed{seed}.npy",
        )

        if os.path.exists(bs_path):
            return np.load(bs_path)

        rng = np.random.RandomState(seed)
        indices = np.zeros((self.n_bootstrap, n_queries), dtype=np.int64)

        for i in range(self.n_bootstrap):
            indices[i] = rng.choice(n_queries, n_queries, replace=True)

        np.save(bs_path, indices)
        return indices

    def bootstrap_metrics(self, per_query_df, bootstrap_indices):
        metric_names = ["recall@1", "recall@5", "recall@10", "mrr"]
        boot_dict = {}

        for metric in metric_names:
            vals = per_query_df[metric].to_numpy(dtype=np.float64)
            arr = np.zeros((len(bootstrap_indices),), dtype=np.float32)

            for i, idx in enumerate(bootstrap_indices):
                arr[i] = float(vals[idx].mean())

            boot_dict[metric] = arr

        return boot_dict

    def ci95(self, x):
        lo, hi = np.percentile(x, [2.5, 97.5])
        return lo, hi

    def _summary_str_pct(self, arr: np.ndarray) -> str:
        arr = np.asarray(arr, dtype=np.float32)
        mean = float(arr.mean()) * 100.0
        std = float(arr.std()) * 100.0
        lo, hi = self.ci95(arr)
        lo *= 100.0
        hi *= 100.0
        return f"{mean:.4f} ± {std:.4f} [{lo:.4f}, {hi:.4f}]"

    def _build_label_buckets(self, labels_np):
        vector_to_indices = {}
        for idx, vec in enumerate(labels_np):
            key = tuple(vec.tolist())
            if key not in vector_to_indices:
                vector_to_indices[key] = []
            vector_to_indices[key].append(idx)
        return vector_to_indices

    def _sample_from_pool(self, pool, n_needed, rng):
        if n_needed <= 0 or len(pool) == 0:
            return []

        pool = np.asarray(pool, dtype=np.int64)
        if len(pool) <= n_needed:
            out = pool.copy()
            rng.shuffle(out)
            return out.tolist()

        chosen = rng.choice(pool, size=n_needed, replace=False)
        return chosen.tolist()

    def _build_setting_b_candidate_pool(self, query_idx, labels_np, vector_to_indices, rng):
        M = len(labels_np)
        if self.settingB_N > M:
            raise ValueError(f"Setting B requires N={self.settingB_N}, but test set has only M={M} rows.")

        query_vec = labels_np[query_idx]
        query_key = tuple(query_vec.tolist())

        negatives = []

        exact_pool = [idx for idx in vector_to_indices.get(query_key, []) if idx != query_idx]
        negatives.extend(self._sample_from_pool(exact_pool, self.settingB_N - 1, rng))

        remaining = (self.settingB_N - 1) - len(negatives)
        if remaining <= 0:
            cand_idx = np.asarray([query_idx] + negatives, dtype=np.int64)
            rng.shuffle(cand_idx)
            return cand_idx

        all_keys = list(vector_to_indices.keys())
        used = set(negatives)
        used.add(query_idx)

        for dist in range(1, labels_np.shape[1] + 1):
            bucket = []

            for key in all_keys:
                if key == query_key:
                    continue

                key_arr = np.asarray(key, dtype=np.int8)
                hamming_dist = int(np.sum(key_arr != query_vec))

                if hamming_dist == dist:
                    for idx in vector_to_indices[key]:
                        if idx not in used:
                            bucket.append(idx)

            chosen = self._sample_from_pool(bucket, remaining, rng)
            negatives.extend(chosen)
            used.update(chosen)

            remaining = (self.settingB_N - 1) - len(negatives)
            if remaining <= 0:
                break

        if len(negatives) != self.settingB_N - 1:
            raise ValueError(
                f"Could not build Setting B pool for query {query_idx}. "
                f"Needed {self.settingB_N - 1} negatives, got {len(negatives)}."
            )

        cand_idx = np.asarray([query_idx] + negatives, dtype=np.int64)
        rng.shuffle(cand_idx)
        return cand_idx

    def _make_numpy_rng(self, seed_val):
        seed_val = int(seed_val) % (2 ** 32 - 1)
        return np.random.RandomState(seed_val)

    def run_settingB(self, emb_dict):
        Z_img = emb_dict["Z_img"].float().to(self.device)
        Z_txt = emb_dict["Z_txt"].float().to(self.device)

        row_ids = emb_dict["row_ids"]
        subject_ids = emb_dict["subject_ids"]
        study_ids = emb_dict["study_ids"]
        labels_np = emb_dict["labels"].astype(np.int8)

        M = len(row_ids)
        if self.settingB_N > M:
            raise ValueError(f"Setting B requires N={self.settingB_N}, but test set size is only {M}.")

        vector_to_indices = self._build_label_buckets(labels_np)
        bootstrap_indices = self._get_or_create_bootstrap_indices(n_queries=M, seed=13)

        all_summary_rows = []
        raw_trial_rows = []
        all_boot_rows = []

        N = self.settingB_N
        print(f"\nRunning Setting B | dataset={self.dataset_name} | model=BioViL-T | variant={self.variant_name} | N={N}")

        per_query_rows = []

        for q_idx in tqdm(range(M)):
            z_img_i = Z_img[q_idx]
            trial_metrics = []

            for trial_id in range(self.n_trials):
                seed = self._seed_for_setting_b(trial_id)
                rng = self._make_numpy_rng(seed + q_idx * 100003)

                cand_idx = self._build_setting_b_candidate_pool(
                    query_idx=q_idx,
                    labels_np=labels_np,
                    vector_to_indices=vector_to_indices,
                    rng=rng,
                )

                m = self._evaluate_one_query_one_trial(
                    z_img_i=z_img_i,
                    Z_txt_device=Z_txt,
                    query_idx=q_idx,
                    cand_idx=cand_idx,
                )
                trial_metrics.append(m)

                raw_trial_rows.append({
                    "dataset_name": self.dataset_name,
                    "model_name": "microsoft/BiomedVLP-BioViL-T",
                    "variant_name": self.variant_name,
                    "N": N,
                    "query_row_id": int(row_ids[q_idx]),
                    "subject_id": int(subject_ids[q_idx]),
                    "study_id": int(study_ids[q_idx]),
                    "trial_id": int(trial_id),
                    "rank": int(m["rank"]),
                    "recall@1": float(m["recall@1"]),
                    "recall@5": float(m["recall@5"]),
                    "recall@10": float(m["recall@10"]),
                    "mrr": float(m["mrr"]),
                })

            per_query_rows.append({
                "dataset_name": self.dataset_name,
                "model_name": "microsoft/BiomedVLP-BioViL-T",
                "variant_name": self.variant_name,
                "N": N,
                "query_row_id": int(row_ids[q_idx]),
                "subject_id": int(subject_ids[q_idx]),
                "study_id": int(study_ids[q_idx]),
                "recall@1": float(np.mean([m["recall@1"] for m in trial_metrics])),
                "recall@5": float(np.mean([m["recall@5"] for m in trial_metrics])),
                "recall@10": float(np.mean([m["recall@10"] for m in trial_metrics])),
                "mrr": float(np.mean([m["mrr"] for m in trial_metrics])),
            })

        per_query_df = pd.DataFrame(per_query_rows)
        boot_dict = self.bootstrap_metrics(per_query_df=per_query_df, bootstrap_indices=bootstrap_indices)

        for i in range(len(boot_dict["recall@1"])):
            all_boot_rows.append({
                "dataset_name": self.dataset_name,
                "model_name": "microsoft/BiomedVLP-BioViL-T",
                "variant_name": self.variant_name,
                "N": N,
                "bootstrap_id": i,
                "recall@1": float(boot_dict["recall@1"][i]),
                "recall@5": float(boot_dict["recall@5"][i]),
                "recall@10": float(boot_dict["recall@10"][i]),
                "mrr": float(boot_dict["mrr"][i]),
            })

        all_summary_rows.append({
            "dataset_name": self.dataset_name,
            "model_name": "microsoft/BiomedVLP-BioViL-T",
            "variant_name": self.variant_name,
            "split_name": self.split_name,
            "N": int(N),
            "num_queries": int(len(per_query_df)),
            "num_trials_per_query": int(self.n_trials),
            "recall@1": self._summary_str_pct(boot_dict["recall@1"]),
            "recall@5": self._summary_str_pct(boot_dict["recall@5"]),
            "recall@10": self._summary_str_pct(boot_dict["recall@10"]),
            "mrr": self._summary_str_pct(boot_dict["mrr"]),
        })

        return {
            "summary_df": pd.DataFrame(all_summary_rows),
            "raw_trial_df": pd.DataFrame(raw_trial_rows),
            "bootstrap_df": pd.DataFrame(all_boot_rows),
        }

    def save_predictions_csv(self, raw_trial_df, setting_name="B"):
        setting_name = str(setting_name).upper()
        out_path = os.path.join(
            self.out_dir,
            f"predictions_setting{setting_name}_{self.dataset_name}_{self.variant_name}_biovilt_split_{self.split_name}.csv",
        )
        raw_trial_df.to_csv(out_path, index=False)

    def save_bootstrap_csv(self, bootstrap_df, setting_name="B"):
        setting_name = str(setting_name).upper()
        out_path = os.path.join(
            self.out_dir,
            f"bootstrap_setting{setting_name}_{self.dataset_name}_{self.variant_name}_biovilt_split_{self.split_name}.csv",
        )
        bootstrap_df.to_csv(out_path, index=False)

    def save_summary_csv(self, summary_df, setting_name="B"):
        setting_name = str(setting_name).upper()
        out_path = os.path.join(
            self.out_dir,
            f"summary_setting{setting_name}_{self.dataset_name}_{self.variant_name}_biovilt_split_{self.split_name}.csv",
        )
        summary_df.to_csv(out_path, index=False)

    def print_and_save_stats(self, summary_df, setting_name="B"):
        setting_name = str(setting_name).upper()

        msg = (
            f"\n\n----------------------------------------------------------------------------------------\n"
            f"setting: {setting_name}\n"
            f"dataset: {self.dataset_name}\n"
            f"model: microsoft/BiomedVLP-BioViL-T\n"
            f"variant: {self.variant_name}\n"
            f"split: {self.split_name}\n\n"
        )

        for _, row in summary_df.iterrows():
            msg += (
                f"N={int(row['N'])}\n"
                f"Recall@1: {row['recall@1']}\n"
                f"Recall@5: {row['recall@5']}\n"
                f"Recall@10: {row['recall@10']}\n"
                f"MRR: {row['mrr']}\n\n"
            )

        print(msg)

        out_path = os.path.join(
            self.out_dir,
            f"Test_on_setting{setting_name}_{self.dataset_name}_{self.variant_name}_biovilt_split_{self.split_name}.txt",
        )
        with open(out_path, "a") as f:
            f.write(msg)






class PredictionCrossModalDP_BioViLT:
    def __init__(self, cfg_path: str, split_name: str, dataset_name: str, variant_name: str):
        self.params = read_config(cfg_path)
        self.cfg_path = cfg_path
        self.split_name = str(split_name).lower()
        self.dataset_name = str(dataset_name).lower()
        self.model_name = "microsoft/BiomedVLP-BioViL-T"
        self.variant_name = str(variant_name)

        self.setup_cuda()

        self.base_seed = int(self.params["CrossModal"]["base_seed"])
        self.n_trials_settingA = int(self.params["CrossModal"]["n_trials_settingA"])
        self.n_trials_settingB = int(self.params["CrossModal"].get("n_trials_settingB", 10))
        self.n_bootstrap = int(self.params["CrossModal"]["n_bootstrap"])
        self.max_text_tokens = int(self.params["CrossModal"]["max_text_tokens_biovilt"])
        self.feature_cache_batch_size = int(self.params["CrossModal"]["feature_cache_batch_size"])

        self.out_dir = os.path.join(
            self.params["target_dir"],
            self.params["stat_log_path"],
            f"{self.dataset_name}_{self.variant_name}_biovilt_dp_split_{self.split_name}",
        )
        os.makedirs(self.out_dir, exist_ok=True)

        self.bootstrap_dir = os.path.join(
            self.params["target_dir"],
            self.params["stat_log_path"],
            "fixed_bootstrap_indices",
        )
        os.makedirs(self.bootstrap_dir, exist_ok=True)



    def setup_cuda(self, cuda_device_id: int = 0):
        if torch.cuda.is_available():
            torch.backends.cudnn.fastest = True
            torch.cuda.set_device(cuda_device_id)
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")



    def setup_model(self, trained_model_path=None):
        if trained_model_path is None:
            raise ValueError("For DP privacy evaluation, trained_model_path must be provided.")

        self.trained_model_path = trained_model_path

        image_engine = get_image_inference(ImageModelType.BIOVIL_T)
        self.image_projector = image_engine.model.projector.to(self.device)
        self.image_projector.eval()

        text_model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            use_safetensors=False,
        )
        self.text_projector = text_model.cls_projection_head.to(self.device)
        self.text_projector.eval()

        ckpt = torch.load(trained_model_path, map_location="cpu", weights_only=False)
        state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

        image_prefix = "_module.image_projector."
        text_prefix = "_module.text_projector."

        image_state = {}
        text_state = {}

        for k, v in state.items():
            if k.startswith(image_prefix):
                image_state[k.replace(image_prefix, "")] = v
            if k.startswith(text_prefix):
                text_state[k.replace(text_prefix, "")] = v

        if len(image_state) == 0:
            raise ValueError("Could not find image projector weights in DP checkpoint.")
        if len(text_state) == 0:
            raise ValueError("Could not find text projector weights in DP checkpoint.")

        missing_i, unexpected_i = self.image_projector.load_state_dict(image_state, strict=False)
        missing_t, unexpected_t = self.text_projector.load_state_dict(text_state, strict=False)

        print(f"Loaded DP checkpoint: {trained_model_path}")
        print("Image projector missing keys:", missing_i)
        print("Image projector unexpected keys:", unexpected_i)
        print("Text projector missing keys:", missing_t)
        print("Text projector unexpected keys:", unexpected_t)



    def _get_test_loader(self):
        if self.dataset_name == "mimic":
            dataset = mimic_retrieval_test_data_loader_2D(
                cfg_path=self.cfg_path,
                mode=self.split_name,
            )
            loader = DataLoader(
                dataset=dataset,
                batch_size=self.feature_cache_batch_size,
                pin_memory=True,
                drop_last=False,
                shuffle=False,
                num_workers=int(self.params["CrossModal"]["num_workers"]),
                collate_fn=mimic_retrieval_collate_fn,
                persistent_workers=True,
            )
            return dataset, loader

        elif self.dataset_name == "chexpert":
            dataset = chexpert_retrieval_test_data_loader_2D(
                cfg_path=self.cfg_path,
                mode=self.split_name,
            )
            loader = DataLoader(
                dataset=dataset,
                batch_size=self.feature_cache_batch_size,
                pin_memory=True,
                drop_last=False,
                shuffle=False,
                num_workers=int(self.params["CrossModal"]["num_workers"]),
                collate_fn=chexpert_retrieval_collate_fn,
                persistent_workers=True,
            )
            return dataset, loader

        else:
            raise ValueError("Unsupported dataset_name.")

    def _get_cache_dir(self):
        if self.dataset_name == "mimic":
            cache_dir = os.path.join(self.params["target_dir"], "cached_biovilt_features")
        elif self.dataset_name == "chexpert":
            cache_dir = os.path.join(self.params["target_dir"], "cached_biovilt_features_chexpert")
        else:
            raise ValueError("Unsupported dataset_name.")

        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir

    def _get_cache_paths(self):
        cache_dir = self._get_cache_dir()
        img_path = os.path.join(cache_dir, f"{self.split_name}_img.pt")
        txt_path = os.path.join(cache_dir, f"{self.split_name}_txt.pt")
        return img_path, txt_path



    def precompute_cached_features_for_test_if_needed(self):
        img_path, txt_path = self._get_cache_paths()

        if os.path.exists(img_path) and os.path.exists(txt_path):
            print(f"Using existing cached features:\n{img_path}\n{txt_path}")
            return img_path, txt_path

        _, test_loader = self._get_test_loader()

        feature_extractor = BioViLTFeatureExtractor(
            model_name=self.model_name,
            max_text_tokens=self.max_text_tokens,
            device=self.device,
        )

        all_img = []
        all_txt = []

        for batch in tqdm(test_loader, desc=f"Precomputing cached test features | {self.dataset_name}"):
            img_features = feature_extractor.extract_image_backbone_features(batch["image_paths"])
            txt_features = feature_extractor.extract_text_backbone_features(batch["report_texts"])
            all_img.append(img_features)
            all_txt.append(txt_features)

        img_tensor = torch.cat(all_img, dim=0)
        txt_tensor = torch.cat(all_txt, dim=0)

        torch.save(img_tensor, img_path)
        torch.save(txt_tensor, txt_path)

        print(f"Saved cached test image features to: {img_path}")
        print(f"Saved cached test text features to: {txt_path}")

        return img_path, txt_path



    def _load_test_metadata(self):
        dataset, _ = self._get_test_loader()

        row_ids = []
        subject_ids = []
        study_ids = []
        labels = []

        for i in range(len(dataset)):
            item = dataset[i]
            row_ids.append(item["row_id"])
            subject_ids.append(item["subject_id"])
            study_ids.append(item["study_id"])
            labels.append(item["label"].numpy())

        return {
            "row_ids": np.asarray(row_ids),
            "subject_ids": np.asarray(subject_ids),
            "study_ids": np.asarray(study_ids),
            "labels": np.asarray(labels, dtype=np.float32),
        }



    def extract_embeddings(self):
        img_path, txt_path = self.precompute_cached_features_for_test_if_needed()

        img_features = torch.load(img_path, map_location="cpu", weights_only=False)
        txt_features = torch.load(txt_path, map_location="cpu", weights_only=False)

        if not isinstance(img_features, torch.Tensor):
            raise ValueError(f"Expected tensor at {img_path}, got {type(img_features)}")
        if not isinstance(txt_features, torch.Tensor):
            raise ValueError(f"Expected tensor at {txt_path}, got {type(txt_features)}")

        z_img_all = []
        z_txt_all = []
        batch_size = 1024

        with torch.inference_mode():
            for start in tqdm(range(0, len(img_features), batch_size), desc=f"Projecting image features | {self.dataset_name}"):
                end = min(start + batch_size, len(img_features))
                x = img_features[start:end].to(self.device, non_blocking=True)

                if x.ndim == 2:
                    x = x.unsqueeze(-1).unsqueeze(-1)

                z_img = self.image_projector(x)
                z_img = z_img.flatten(1)
                z_img = F.normalize(z_img.float(), p=2, dim=-1)
                z_img_all.append(z_img.cpu())

            for start in tqdm(range(0, len(txt_features), batch_size), desc=f"Projecting text features | {self.dataset_name}"):
                end = min(start + batch_size, len(txt_features))
                x = txt_features[start:end].to(self.device, non_blocking=True)

                z_txt = self.text_projector(x)
                z_txt = F.normalize(z_txt.float(), p=2, dim=-1)
                z_txt_all.append(z_txt.cpu())

        Z_img = torch.cat(z_img_all, dim=0).to(torch.float16)
        Z_txt = torch.cat(z_txt_all, dim=0).to(torch.float16)

        meta = self._load_test_metadata()

        if len(Z_img) != len(meta["row_ids"]) or len(Z_txt) != len(meta["row_ids"]):
            raise ValueError(
                f"Length mismatch: Z_img={len(Z_img)}, Z_txt={len(Z_txt)}, meta={len(meta['row_ids'])}"
            )

        return {
            "Z_img": Z_img,
            "Z_txt": Z_txt,
            "row_ids": meta["row_ids"],
            "subject_ids": meta["subject_ids"],
            "study_ids": meta["study_ids"],
            "labels": meta["labels"],
        }



    def _get_candidate_sizes(self, M):
        out = [100, 1000, 10000]
        out = [n for n in out if n <= M]
        if M not in out:
            out.append(M)
        return sorted(out)

    def _seed_for_setting_a(self, N, trial_id, max_test_size):
        if N in [100, 1000, 10000]:
            return self.base_seed + int(1000000 * math.log10(N)) + trial_id
        if N == max_test_size:
            return self.base_seed + 9999999 + trial_id
        raise ValueError(f"Unsupported N: {N}")

    def _seed_for_setting_b(self, trial_id):
        return self.base_seed + 777 + trial_id

    def _safe_seed(self, seed):
        return int(seed % (2**32 - 1))

    def _compute_metrics_from_rank(self, rank_1based):
        return {
            "recall@1": 1.0 if rank_1based <= 1 else 0.0,
            "recall@5": 1.0 if rank_1based <= 5 else 0.0,
            "recall@10": 1.0 if rank_1based <= 10 else 0.0,
            "mrr": 1.0 / float(rank_1based),
        }

    def _evaluate_one_query_one_trial(self, z_img_i, Z_txt_device, query_idx, cand_idx):
        cand_idx_t = torch.as_tensor(cand_idx, device=Z_txt_device.device, dtype=torch.long)
        cand_txt = Z_txt_device.index_select(0, cand_idx_t)
        scores = torch.matmul(cand_txt, z_img_i)

        order = torch.argsort(scores, descending=True)

        true_pos = np.where(np.asarray(cand_idx) == query_idx)[0]
        if len(true_pos) != 1:
            raise ValueError("True report missing or duplicated in candidate pool.")
        true_pos = int(true_pos[0])

        rank_1based = int((order == true_pos).nonzero(as_tuple=False).item()) + 1

        out = self._compute_metrics_from_rank(rank_1based)
        out["rank"] = rank_1based
        return out

    def _get_or_create_bootstrap_indices(self, n_queries, seed=13):
        bs_path = os.path.join(
            self.bootstrap_dir,
            f"bootstrap_indices_{self.dataset_name}_{self.split_name}_n{n_queries}_b{self.n_bootstrap}_seed{seed}.npy",
        )

        if os.path.exists(bs_path):
            return np.load(bs_path)

        rng = np.random.RandomState(seed)
        indices = np.zeros((self.n_bootstrap, n_queries), dtype=np.int64)

        for i in range(self.n_bootstrap):
            indices[i] = rng.choice(n_queries, n_queries, replace=True)

        np.save(bs_path, indices)
        return indices

    def bootstrap_metrics(self, per_query_df, bootstrap_indices):
        metric_names = ["recall@1", "recall@5", "recall@10", "mrr"]
        boot_dict = {}

        for metric in metric_names:
            vals = per_query_df[metric].to_numpy(dtype=np.float64)
            arr = np.zeros((len(bootstrap_indices),), dtype=np.float32)

            for i, idx in enumerate(bootstrap_indices):
                arr[i] = float(vals[idx].mean())

            boot_dict[metric] = arr

        return boot_dict

    def ci95(self, x):
        lo, hi = np.percentile(x, [2.5, 97.5])
        return lo, hi

    def _summary_str_pct(self, arr: np.ndarray) -> str:
        arr = np.asarray(arr, dtype=np.float32)
        mean = float(arr.mean()) * 100.0
        std = float(arr.std()) * 100.0
        lo, hi = self.ci95(arr)
        lo *= 100.0
        hi *= 100.0
        return f"{mean:.4f} ± {std:.4f} [{lo:.4f}, {hi:.4f}]"



    def run_settingA(self, emb_dict):
        Z_img = emb_dict["Z_img"].float().to(self.device)
        Z_txt = emb_dict["Z_txt"].float().to(self.device)

        row_ids = emb_dict["row_ids"]
        subject_ids = emb_dict["subject_ids"]
        study_ids = emb_dict["study_ids"]

        M = len(row_ids)
        candidate_sizes = self._get_candidate_sizes(M)
        all_indices = np.arange(M)

        bootstrap_indices = self._get_or_create_bootstrap_indices(n_queries=M, seed=13)

        all_summary_rows = []
        raw_trial_rows = []
        all_boot_rows = []

        for N in candidate_sizes:
            print(f"\nRunning Setting A | dataset={self.dataset_name} | model=BioViL-T DP | variant={self.variant_name} | N={N}")

            per_query_rows = []

            for q_idx in tqdm(range(M)):
                z_img_i = Z_img[q_idx]
                neg_pool = all_indices[all_indices != q_idx]
                trial_metrics = []

                if N == M:
                    cand_idx = all_indices.copy()

                    for trial_id in range(self.n_trials_settingA):
                        m = self._evaluate_one_query_one_trial(z_img_i, Z_txt, q_idx, cand_idx)
                        trial_metrics.append(m)

                        raw_trial_rows.append({
                            "setting": "A",
                            "dataset_name": self.dataset_name,
                            "model_name": self.model_name,
                            "variant_name": self.variant_name,
                            "N": N,
                            "query_row_id": int(row_ids[q_idx]),
                            "subject_id": int(subject_ids[q_idx]),
                            "study_id": int(study_ids[q_idx]),
                            "trial_id": int(trial_id),
                            "rank": int(m["rank"]),
                            "recall@1": float(m["recall@1"]),
                            "recall@5": float(m["recall@5"]),
                            "recall@10": float(m["recall@10"]),
                            "mrr": float(m["mrr"]),
                        })
                else:
                    for trial_id in range(self.n_trials_settingA):
                        seed = self._seed_for_setting_a(N, trial_id, M)
                        rng = np.random.RandomState(self._safe_seed(seed + q_idx * 100003))

                        neg_idx = rng.choice(neg_pool, size=N - 1, replace=False)
                        cand_idx = np.concatenate([[q_idx], neg_idx])
                        rng.shuffle(cand_idx)

                        m = self._evaluate_one_query_one_trial(z_img_i, Z_txt, q_idx, cand_idx)
                        trial_metrics.append(m)

                        raw_trial_rows.append({
                            "setting": "A",
                            "dataset_name": self.dataset_name,
                            "model_name": self.model_name,
                            "variant_name": self.variant_name,
                            "N": N,
                            "query_row_id": int(row_ids[q_idx]),
                            "subject_id": int(subject_ids[q_idx]),
                            "study_id": int(study_ids[q_idx]),
                            "trial_id": int(trial_id),
                            "rank": int(m["rank"]),
                            "recall@1": float(m["recall@1"]),
                            "recall@5": float(m["recall@5"]),
                            "recall@10": float(m["recall@10"]),
                            "mrr": float(m["mrr"]),
                        })

                per_query_rows.append({
                    "dataset_name": self.dataset_name,
                    "model_name": self.model_name,
                    "variant_name": self.variant_name,
                    "N": N,
                    "query_row_id": int(row_ids[q_idx]),
                    "subject_id": int(subject_ids[q_idx]),
                    "study_id": int(study_ids[q_idx]),
                    "recall@1": float(np.mean([m["recall@1"] for m in trial_metrics])),
                    "recall@5": float(np.mean([m["recall@5"] for m in trial_metrics])),
                    "recall@10": float(np.mean([m["recall@10"] for m in trial_metrics])),
                    "mrr": float(np.mean([m["mrr"] for m in trial_metrics])),
                })

            per_query_df = pd.DataFrame(per_query_rows)
            boot_dict = self.bootstrap_metrics(per_query_df=per_query_df, bootstrap_indices=bootstrap_indices)

            for i in range(len(boot_dict["recall@1"])):
                all_boot_rows.append({
                    "setting": "A",
                    "dataset_name": self.dataset_name,
                    "model_name": self.model_name,
                    "variant_name": self.variant_name,
                    "N": N,
                    "bootstrap_id": i,
                    "recall@1": float(boot_dict["recall@1"][i]),
                    "recall@5": float(boot_dict["recall@5"][i]),
                    "recall@10": float(boot_dict["recall@10"][i]),
                    "mrr": float(boot_dict["mrr"][i]),
                })

            all_summary_rows.append({
                "setting": "A",
                "dataset_name": self.dataset_name,
                "model_name": self.model_name,
                "variant_name": self.variant_name,
                "split_name": self.split_name,
                "N": int(N),
                "num_queries": int(len(per_query_df)),
                "num_trials_per_query": int(self.n_trials_settingA),
                "recall@1": self._summary_str_pct(boot_dict["recall@1"]),
                "recall@5": self._summary_str_pct(boot_dict["recall@5"]),
                "recall@10": self._summary_str_pct(boot_dict["recall@10"]),
                "mrr": self._summary_str_pct(boot_dict["mrr"]),
            })
            print(all_summary_rows)

        return {
            "summary_df": pd.DataFrame(all_summary_rows),
            "raw_trial_df": pd.DataFrame(raw_trial_rows),
            "bootstrap_df": pd.DataFrame(all_boot_rows),
        }

    # setting B
    def _build_label_index_structures(self, labels):
        labels = np.asarray(labels, dtype=np.int32)
        label_tuples = [tuple(x.tolist()) for x in labels]

        exact_map = {}
        for idx, t in enumerate(label_tuples):
            exact_map.setdefault(t, []).append(idx)

        return labels, label_tuples, exact_map

    def _sample_settingB_candidates(self, q_idx, labels, exact_map, N, rng):
        query_label = labels[q_idx]
        query_tuple = tuple(query_label.tolist())

        same_exact = [idx for idx in exact_map.get(query_tuple, []) if idx != q_idx]

        selected = []

        if len(same_exact) >= (N - 1):
            selected = rng.choice(np.asarray(same_exact), size=N - 1, replace=False).tolist()
        else:
            selected.extend(same_exact)

            remaining = (N - 1) - len(selected)
            all_indices = np.arange(len(labels))
            valid_pool = all_indices[all_indices != q_idx]
            used = set(selected)

            not_used = [idx for idx in valid_pool.tolist() if idx not in used]

            hamming_buckets = {}
            for idx in not_used:
                dist = int(np.sum(np.abs(labels[idx] - query_label)))
                hamming_buckets.setdefault(dist, []).append(idx)

            for dist in sorted(hamming_buckets.keys()):
                bucket = hamming_buckets[dist]
                if len(bucket) <= remaining:
                    rng.shuffle(bucket)
                    selected.extend(bucket)
                    remaining -= len(bucket)
                else:
                    chosen = rng.choice(np.asarray(bucket), size=remaining, replace=False).tolist()
                    selected.extend(chosen)
                    remaining = 0

                if remaining == 0:
                    break

        if len(selected) != (N - 1):
            raise ValueError(f"Could not build Setting B candidate pool for query {q_idx}. Got {len(selected)} negatives.")

        cand_idx = np.concatenate([[q_idx], np.asarray(selected, dtype=np.int64)])
        rng.shuffle(cand_idx)
        return cand_idx

    def run_settingB(self, emb_dict):
        Z_img = emb_dict["Z_img"].float().to(self.device)
        Z_txt = emb_dict["Z_txt"].float().to(self.device)

        row_ids = emb_dict["row_ids"]
        subject_ids = emb_dict["subject_ids"]
        study_ids = emb_dict["study_ids"]
        labels = emb_dict["labels"].astype(np.int32)

        M = len(row_ids)
        N = min(10000, M)

        labels_arr, _, exact_map = self._build_label_index_structures(labels)
        bootstrap_indices = self._get_or_create_bootstrap_indices(n_queries=M, seed=13)

        all_summary_rows = []
        raw_trial_rows = []
        all_boot_rows = []
        per_query_rows = []

        print(f"\nRunning Setting B | dataset={self.dataset_name} | model=BioViL-T DP | variant={self.variant_name} | N={N}")

        for q_idx in tqdm(range(M)):
            z_img_i = Z_img[q_idx]
            trial_metrics = []

            for trial_id in range(self.n_trials_settingB):
                seed = self._seed_for_setting_b(trial_id)
                rng = np.random.RandomState(self._safe_seed(seed + q_idx * 100003))

                cand_idx = self._sample_settingB_candidates(
                    q_idx=q_idx,
                    labels=labels_arr,
                    exact_map=exact_map,
                    N=N,
                    rng=rng,
                )

                m = self._evaluate_one_query_one_trial(z_img_i, Z_txt, q_idx, cand_idx)
                trial_metrics.append(m)

                raw_trial_rows.append({
                    "setting": "B",
                    "dataset_name": self.dataset_name,
                    "model_name": self.model_name,
                    "variant_name": self.variant_name,
                    "N": N,
                    "query_row_id": int(row_ids[q_idx]),
                    "subject_id": int(subject_ids[q_idx]),
                    "study_id": int(study_ids[q_idx]),
                    "trial_id": int(trial_id),
                    "rank": int(m["rank"]),
                    "recall@1": float(m["recall@1"]),
                    "recall@5": float(m["recall@5"]),
                    "recall@10": float(m["recall@10"]),
                    "mrr": float(m["mrr"]),
                })

            per_query_rows.append({
                "dataset_name": self.dataset_name,
                "model_name": self.model_name,
                "variant_name": self.variant_name,
                "N": N,
                "query_row_id": int(row_ids[q_idx]),
                "subject_id": int(subject_ids[q_idx]),
                "study_id": int(study_ids[q_idx]),
                "recall@1": float(np.mean([m["recall@1"] for m in trial_metrics])),
                "recall@5": float(np.mean([m["recall@5"] for m in trial_metrics])),
                "recall@10": float(np.mean([m["recall@10"] for m in trial_metrics])),
                "mrr": float(np.mean([m["mrr"] for m in trial_metrics])),
            })

        per_query_df = pd.DataFrame(per_query_rows)
        boot_dict = self.bootstrap_metrics(per_query_df=per_query_df, bootstrap_indices=bootstrap_indices)

        for i in range(len(boot_dict["recall@1"])):
            all_boot_rows.append({
                "setting": "B",
                "dataset_name": self.dataset_name,
                "model_name": self.model_name,
                "variant_name": self.variant_name,
                "N": N,
                "bootstrap_id": i,
                "recall@1": float(boot_dict["recall@1"][i]),
                "recall@5": float(boot_dict["recall@5"][i]),
                "recall@10": float(boot_dict["recall@10"][i]),
                "mrr": float(boot_dict["mrr"][i]),
            })

        all_summary_rows.append({
            "setting": "B",
            "dataset_name": self.dataset_name,
            "model_name": self.model_name,
            "variant_name": self.variant_name,
            "split_name": self.split_name,
            "N": int(N),
            "num_queries": int(len(per_query_df)),
            "num_trials_per_query": int(self.n_trials_settingB),
            "recall@1": self._summary_str_pct(boot_dict["recall@1"]),
            "recall@5": self._summary_str_pct(boot_dict["recall@5"]),
            "recall@10": self._summary_str_pct(boot_dict["recall@10"]),
            "mrr": self._summary_str_pct(boot_dict["mrr"]),
        })

        return {
            "summary_df": pd.DataFrame(all_summary_rows),
            "raw_trial_df": pd.DataFrame(raw_trial_rows),
            "bootstrap_df": pd.DataFrame(all_boot_rows),
        }



    def save_predictions_csv(self, raw_trial_df, setting_name="A"):
        setting_name = str(setting_name).upper()
        out_path = os.path.join(
            self.out_dir,
            f"predictions_setting{setting_name}_{self.dataset_name}_{self.variant_name}_{self.model_name.replace('/', '_')}_split_{self.split_name}.csv",
        )
        raw_trial_df.to_csv(out_path, index=False)

    def save_bootstrap_csv(self, bootstrap_df, setting_name="A"):
        setting_name = str(setting_name).upper()
        out_path = os.path.join(
            self.out_dir,
            f"bootstrap_setting{setting_name}_{self.dataset_name}_{self.variant_name}_{self.model_name.replace('/', '_')}_split_{self.split_name}.csv",
        )
        bootstrap_df.to_csv(out_path, index=False)

    def save_summary_csv(self, summary_df, setting_name="A"):
        setting_name = str(setting_name).upper()
        out_path = os.path.join(
            self.out_dir,
            f"summary_setting{setting_name}_{self.dataset_name}_{self.variant_name}_{self.model_name.replace('/', '_')}_split_{self.split_name}.csv",
        )
        summary_df.to_csv(out_path, index=False)

    def print_and_save_stats(self, summary_df, setting_name="A"):
        setting_name = str(setting_name).upper()

        msg = (
            f"\n\n----------------------------------------------------------------------------------------\n"
            f"setting: {setting_name}\n"
            f"dataset: {self.dataset_name}\n"
            f"model: {self.model_name}\n"
            f"variant: {self.variant_name}\n"
            f"split: {self.split_name}\n\n"
        )

        for _, row in summary_df.iterrows():
            msg += (
                f"N={int(row['N'])}\n"
                f"Recall@1: {row['recall@1']}\n"
                f"Recall@5: {row['recall@5']}\n"
                f"Recall@10: {row['recall@10']}\n"
                f"MRR: {row['mrr']}\n\n"
            )

        print(msg)

        out_path = os.path.join(
            self.out_dir,
            f"Test_on_setting{setting_name}_{self.dataset_name}_{self.variant_name}_{self.model_name.replace('/', '_')}_split_{self.split_name}.txt",
        )
        with open(out_path, "a") as f:
            f.write(msg)





class BioViLTFeatureExtractor:
    """
    Extracts pre-projection frozen backbone features:
        - image backbone feature: 512-d
        - text backbone feature: 768-d
    """

    def __init__(self, model_name="microsoft/BiomedVLP-BioViL-T", max_text_tokens=256, device=None):
        self.model_name = model_name
        self.max_text_tokens = int(max_text_tokens)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.image_engine = get_image_inference(ImageModelType.BIOVIL_T)
        self.image_transform = self.image_engine.transform
        self.image_model = self.image_engine.model.to(self.device)
        self.image_model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=True,
        )
        self.text_model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_safetensors=False,
        ).to(self.device)
        self.text_model.eval()

    def extract_image_backbone_features(self, image_paths):
        image_tensors = []
        for p in image_paths:
            img = Image.open(Path(p))
            x = self.image_transform(img)
            image_tensors.append(x)

        x = torch.stack(image_tensors, dim=0).to(self.device, non_blocking=True)

        with torch.inference_mode():
            out = self.image_model(x)

            if hasattr(out, "img_embedding"):
                img_features = out.img_embedding
            elif isinstance(out, torch.Tensor):
                img_features = out
            else:
                raise ValueError(f"Unsupported image backbone output type: {type(out)}")

        return img_features.detach().cpu()

    def extract_text_backbone_features(self, report_texts):
        tok = self.tokenizer(
            report_texts,
            padding=True,
            truncation=True,
            max_length=self.max_text_tokens,
            return_tensors="pt",
        )
        tok = {k: v.to(self.device, non_blocking=True) for k, v in tok.items()}

        with torch.inference_mode():
            if hasattr(self.text_model, "bert"):
                out = self.text_model.bert(
                    input_ids=tok["input_ids"],
                    attention_mask=tok["attention_mask"],
                )
                txt_features = out.last_hidden_state[:, 0]
            else:
                out = self.text_model(
                    input_ids=tok["input_ids"],
                    attention_mask=tok["attention_mask"],
                )
                if hasattr(out, "last_hidden_state"):
                    txt_features = out.last_hidden_state[:, 0]
                else:
                    raise ValueError(f"Unsupported text backbone output type: {type(out)}")

        return txt_features.detach().cpu()





class PredictionCrossModalEntropy_BioViLT:
    def __init__(self, cfg_path: str, split_name: str, dataset_name: str, variant_name: str):
        self.params = read_config(cfg_path)
        self.cfg_path = cfg_path
        self.split_name = str(split_name).lower()
        self.dataset_name = str(dataset_name).lower()
        self.model_name = "microsoft/BiomedVLP-BioViL-T"
        self.variant_name = str(variant_name)

        self.setup_cuda()

        self.base_seed = int(self.params["CrossModal"]["base_seed"])
        self.n_trials_settingA = int(self.params["CrossModal"]["n_trials_settingA"])
        self.n_trials_settingB = int(self.params["CrossModal"].get("n_trials_settingB", 10))
        self.n_bootstrap = int(self.params["CrossModal"]["n_bootstrap"])
        self.max_text_tokens = int(self.params["CrossModal"]["max_text_tokens_biovilt"])
        self.feature_cache_batch_size = int(self.params["CrossModal"]["feature_cache_batch_size"])

        self.out_dir = os.path.join(
            self.params["target_dir"],
            self.params["stat_log_path"],
            f"{self.dataset_name}_{self.variant_name}_biovilt_entropy_split_{self.split_name}",
        )
        os.makedirs(self.out_dir, exist_ok=True)

        self.bootstrap_dir = os.path.join(
            self.params["target_dir"],
            self.params["stat_log_path"],
            "fixed_bootstrap_indices",
        )
        os.makedirs(self.bootstrap_dir, exist_ok=True)

    def setup_cuda(self, cuda_device_id: int = 0):
        if torch.cuda.is_available():
            torch.backends.cudnn.fastest = True
            torch.cuda.set_device(cuda_device_id)
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def setup_model(self, trained_model_path=None):
        if trained_model_path is None:
            raise ValueError("For entropy privacy evaluation, trained_model_path must be provided.")

        self.trained_model_path = trained_model_path

        image_engine = get_image_inference(ImageModelType.BIOVIL_T)
        self.image_projector = image_engine.model.projector.to(self.device)
        self.image_projector.eval()

        text_model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            use_safetensors=False,
        )
        self.text_projector = text_model.cls_projection_head.to(self.device)
        self.text_projector.eval()

        ckpt = torch.load(trained_model_path, map_location="cpu", weights_only=False)
        state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

        image_prefixes = [
            "_module.image_projector.",
            "image_projector.",
        ]
        text_prefixes = [
            "_module.text_projector.",
            "text_projector.",
        ]

        image_state = {}
        text_state = {}

        for k, v in state.items():
            for prefix in image_prefixes:
                if k.startswith(prefix):
                    image_state[k.replace(prefix, "")] = v
                    break

            for prefix in text_prefixes:
                if k.startswith(prefix):
                    text_state[k.replace(prefix, "")] = v
                    break

        if len(image_state) == 0:
            raise ValueError("Could not find image projector weights in entropy checkpoint.")
        if len(text_state) == 0:
            raise ValueError("Could not find text projector weights in entropy checkpoint.")

        missing_i, unexpected_i = self.image_projector.load_state_dict(image_state, strict=False)
        missing_t, unexpected_t = self.text_projector.load_state_dict(text_state, strict=False)

        print(f"Loaded entropy checkpoint: {trained_model_path}")
        print("Image projector missing keys:", missing_i)
        print("Image projector unexpected keys:", unexpected_i)
        print("Text projector missing keys:", missing_t)
        print("Text projector unexpected keys:", unexpected_t)

    def _get_test_loader(self):
        if self.dataset_name == "mimic":
            dataset = mimic_retrieval_test_data_loader_2D(
                cfg_path=self.cfg_path,
                mode=self.split_name,
            )
            loader = DataLoader(
                dataset=dataset,
                batch_size=self.feature_cache_batch_size,
                pin_memory=True,
                drop_last=False,
                shuffle=False,
                num_workers=int(self.params["CrossModal"]["num_workers"]),
                collate_fn=mimic_retrieval_collate_fn,
                persistent_workers=True,
            )
            return dataset, loader

        elif self.dataset_name == "chexpert":
            dataset = chexpert_retrieval_test_data_loader_2D(
                cfg_path=self.cfg_path,
                mode=self.split_name,
            )
            loader = DataLoader(
                dataset=dataset,
                batch_size=self.feature_cache_batch_size,
                pin_memory=True,
                drop_last=False,
                shuffle=False,
                num_workers=int(self.params["CrossModal"]["num_workers"]),
                collate_fn=chexpert_retrieval_collate_fn,
                persistent_workers=True,
            )
            return dataset, loader

        else:
            raise ValueError("Unsupported dataset_name.")

    def _get_cache_dir(self):
        if self.dataset_name == "mimic":
            cache_dir = os.path.join(self.params["target_dir"], "cached_biovilt_features")
        elif self.dataset_name == "chexpert":
            cache_dir = os.path.join(self.params["target_dir"], "cached_biovilt_features_chexpert")
        else:
            raise ValueError("Unsupported dataset_name.")

        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir

    def _get_cache_paths(self):
        cache_dir = self._get_cache_dir()
        img_path = os.path.join(cache_dir, f"{self.split_name}_img.pt")
        txt_path = os.path.join(cache_dir, f"{self.split_name}_txt.pt")
        return img_path, txt_path

    def precompute_cached_features_for_test_if_needed(self):
        img_path, txt_path = self._get_cache_paths()

        if os.path.exists(img_path) and os.path.exists(txt_path):
            print(f"Using existing cached features:\n{img_path}\n{txt_path}")
            return img_path, txt_path

        _, test_loader = self._get_test_loader()

        feature_extractor = BioViLTFeatureExtractor(
            model_name=self.model_name,
            max_text_tokens=self.max_text_tokens,
            device=self.device,
        )

        all_img = []
        all_txt = []

        for batch in tqdm(test_loader, desc=f"Precomputing cached test features | {self.dataset_name}"):
            img_features = feature_extractor.extract_image_backbone_features(batch["image_paths"])
            txt_features = feature_extractor.extract_text_backbone_features(batch["report_texts"])
            all_img.append(img_features)
            all_txt.append(txt_features)

        img_tensor = torch.cat(all_img, dim=0)
        txt_tensor = torch.cat(all_txt, dim=0)

        torch.save(img_tensor, img_path)
        torch.save(txt_tensor, txt_path)

        print(f"Saved cached test image features to: {img_path}")
        print(f"Saved cached test text features to: {txt_path}")

        return img_path, txt_path

    def _load_test_metadata(self):
        dataset, _ = self._get_test_loader()

        row_ids = []
        subject_ids = []
        study_ids = []
        labels = []

        for i in range(len(dataset)):
            item = dataset[i]
            row_ids.append(item["row_id"])
            subject_ids.append(item["subject_id"])
            study_ids.append(item["study_id"])
            labels.append(item["label"].numpy())

        return {
            "row_ids": np.asarray(row_ids),
            "subject_ids": np.asarray(subject_ids),
            "study_ids": np.asarray(study_ids),
            "labels": np.asarray(labels, dtype=np.float32),
        }

    def extract_embeddings(self):
        img_path, txt_path = self.precompute_cached_features_for_test_if_needed()

        img_features = torch.load(img_path, map_location="cpu", weights_only=False)
        txt_features = torch.load(txt_path, map_location="cpu", weights_only=False)

        if not isinstance(img_features, torch.Tensor):
            raise ValueError(f"Expected tensor at {img_path}, got {type(img_features)}")
        if not isinstance(txt_features, torch.Tensor):
            raise ValueError(f"Expected tensor at {txt_path}, got {type(txt_features)}")

        z_img_all = []
        z_txt_all = []
        batch_size = 1024

        with torch.inference_mode():
            for start in tqdm(range(0, len(img_features), batch_size), desc=f"Projecting image features | {self.dataset_name}"):
                end = min(start + batch_size, len(img_features))
                x = img_features[start:end].to(self.device, non_blocking=True)

                if x.ndim == 2:
                    x = x.unsqueeze(-1).unsqueeze(-1)

                z_img = self.image_projector(x)
                z_img = z_img.flatten(1)
                z_img = F.normalize(z_img.float(), p=2, dim=-1)
                z_img_all.append(z_img.cpu())

            for start in tqdm(range(0, len(txt_features), batch_size), desc=f"Projecting text features | {self.dataset_name}"):
                end = min(start + batch_size, len(txt_features))
                x = txt_features[start:end].to(self.device, non_blocking=True)

                z_txt = self.text_projector(x)
                z_txt = F.normalize(z_txt.float(), p=2, dim=-1)
                z_txt_all.append(z_txt.cpu())

        Z_img = torch.cat(z_img_all, dim=0).to(torch.float16)
        Z_txt = torch.cat(z_txt_all, dim=0).to(torch.float16)

        meta = self._load_test_metadata()

        if len(Z_img) != len(meta["row_ids"]) or len(Z_txt) != len(meta["row_ids"]):
            raise ValueError(
                f"Length mismatch: Z_img={len(Z_img)}, Z_txt={len(Z_txt)}, meta={len(meta['row_ids'])}"
            )

        return {
            "Z_img": Z_img,
            "Z_txt": Z_txt,
            "row_ids": meta["row_ids"],
            "subject_ids": meta["subject_ids"],
            "study_ids": meta["study_ids"],
            "labels": meta["labels"],
        }

    def _get_candidate_sizes(self, M):
        out = [100, 1000, 10000]
        out = [n for n in out if n <= M]
        if M not in out:
            out.append(M)
        return sorted(out)

    def _seed_for_setting_a(self, N, trial_id, max_test_size):
        if N in [100, 1000, 10000]:
            return self.base_seed + int(1000000 * math.log10(N)) + trial_id
        if N == max_test_size:
            return self.base_seed + 9999999 + trial_id
        raise ValueError(f"Unsupported N: {N}")

    def _seed_for_setting_b(self, trial_id):
        return self.base_seed + 777 + trial_id

    def _safe_seed(self, seed):
        return int(seed % (2**32 - 1))

    def _compute_metrics_from_rank(self, rank_1based):
        return {
            "recall@1": 1.0 if rank_1based <= 1 else 0.0,
            "recall@5": 1.0 if rank_1based <= 5 else 0.0,
            "recall@10": 1.0 if rank_1based <= 10 else 0.0,
            "mrr": 1.0 / float(rank_1based),
        }

    def _evaluate_one_query_one_trial(self, z_img_i, Z_txt_device, query_idx, cand_idx):
        cand_idx_t = torch.as_tensor(cand_idx, device=Z_txt_device.device, dtype=torch.long)
        cand_txt = Z_txt_device.index_select(0, cand_idx_t)
        scores = torch.matmul(cand_txt, z_img_i)

        order = torch.argsort(scores, descending=True)

        true_pos = np.where(np.asarray(cand_idx) == query_idx)[0]
        if len(true_pos) != 1:
            raise ValueError("True report missing or duplicated in candidate pool.")
        true_pos = int(true_pos[0])

        rank_1based = int((order == true_pos).nonzero(as_tuple=False).item()) + 1

        out = self._compute_metrics_from_rank(rank_1based)
        out["rank"] = rank_1based
        return out

    def _get_or_create_bootstrap_indices(self, n_queries, seed=13):
        bs_path = os.path.join(
            self.bootstrap_dir,
            f"bootstrap_indices_{self.dataset_name}_{self.split_name}_n{n_queries}_b{self.n_bootstrap}_seed{seed}.npy",
        )

        if os.path.exists(bs_path):
            return np.load(bs_path)

        rng = np.random.RandomState(seed)
        indices = np.zeros((self.n_bootstrap, n_queries), dtype=np.int64)

        for i in range(self.n_bootstrap):
            indices[i] = rng.choice(n_queries, n_queries, replace=True)

        np.save(bs_path, indices)
        return indices

    def bootstrap_metrics(self, per_query_df, bootstrap_indices):
        metric_names = ["recall@1", "recall@5", "recall@10", "mrr"]
        boot_dict = {}

        for metric in metric_names:
            vals = per_query_df[metric].to_numpy(dtype=np.float64)
            arr = np.zeros((len(bootstrap_indices),), dtype=np.float32)

            for i, idx in enumerate(bootstrap_indices):
                arr[i] = float(vals[idx].mean())

            boot_dict[metric] = arr

        return boot_dict

    def ci95(self, x):
        lo, hi = np.percentile(x, [2.5, 97.5])
        return lo, hi

    def _summary_str_pct(self, arr: np.ndarray) -> str:
        arr = np.asarray(arr, dtype=np.float32)
        mean = float(arr.mean()) * 100.0
        std = float(arr.std()) * 100.0
        lo, hi = self.ci95(arr)
        lo *= 100.0
        hi *= 100.0
        return f"{mean:.4f} ± {std:.4f} [{lo:.4f}, {hi:.4f}]"

    def run_settingA(self, emb_dict):
        Z_img = emb_dict["Z_img"].float().to(self.device)
        Z_txt = emb_dict["Z_txt"].float().to(self.device)

        row_ids = emb_dict["row_ids"]
        subject_ids = emb_dict["subject_ids"]
        study_ids = emb_dict["study_ids"]

        M = len(row_ids)
        candidate_sizes = self._get_candidate_sizes(M)
        all_indices = np.arange(M)

        bootstrap_indices = self._get_or_create_bootstrap_indices(n_queries=M, seed=13)

        all_summary_rows = []
        raw_trial_rows = []
        all_boot_rows = []

        for N in candidate_sizes:
            print(f"\nRunning Setting A | dataset={self.dataset_name} | model=BioViL-T Entropy | variant={self.variant_name} | N={N}")

            per_query_rows = []

            for q_idx in tqdm(range(M)):
                z_img_i = Z_img[q_idx]
                neg_pool = all_indices[all_indices != q_idx]
                trial_metrics = []

                if N == M:
                    cand_idx = all_indices.copy()

                    for trial_id in range(self.n_trials_settingA):
                        m = self._evaluate_one_query_one_trial(z_img_i, Z_txt, q_idx, cand_idx)
                        trial_metrics.append(m)

                        raw_trial_rows.append({
                            "setting": "A",
                            "dataset_name": self.dataset_name,
                            "model_name": self.model_name,
                            "variant_name": self.variant_name,
                            "N": N,
                            "query_row_id": int(row_ids[q_idx]),
                            "subject_id": int(subject_ids[q_idx]),
                            "study_id": int(study_ids[q_idx]),
                            "trial_id": int(trial_id),
                            "rank": int(m["rank"]),
                            "recall@1": float(m["recall@1"]),
                            "recall@5": float(m["recall@5"]),
                            "recall@10": float(m["recall@10"]),
                            "mrr": float(m["mrr"]),
                        })
                else:
                    for trial_id in range(self.n_trials_settingA):
                        seed = self._seed_for_setting_a(N, trial_id, M)
                        rng = np.random.RandomState(self._safe_seed(seed + q_idx * 100003))

                        neg_idx = rng.choice(neg_pool, size=N - 1, replace=False)
                        cand_idx = np.concatenate([[q_idx], neg_idx])
                        rng.shuffle(cand_idx)

                        m = self._evaluate_one_query_one_trial(z_img_i, Z_txt, q_idx, cand_idx)
                        trial_metrics.append(m)

                        raw_trial_rows.append({
                            "setting": "A",
                            "dataset_name": self.dataset_name,
                            "model_name": self.model_name,
                            "variant_name": self.variant_name,
                            "N": N,
                            "query_row_id": int(row_ids[q_idx]),
                            "subject_id": int(subject_ids[q_idx]),
                            "study_id": int(study_ids[q_idx]),
                            "trial_id": int(trial_id),
                            "rank": int(m["rank"]),
                            "recall@1": float(m["recall@1"]),
                            "recall@5": float(m["recall@5"]),
                            "recall@10": float(m["recall@10"]),
                            "mrr": float(m["mrr"]),
                        })

                per_query_rows.append({
                    "dataset_name": self.dataset_name,
                    "model_name": self.model_name,
                    "variant_name": self.variant_name,
                    "N": N,
                    "query_row_id": int(row_ids[q_idx]),
                    "subject_id": int(subject_ids[q_idx]),
                    "study_id": int(study_ids[q_idx]),
                    "recall@1": float(np.mean([m["recall@1"] for m in trial_metrics])),
                    "recall@5": float(np.mean([m["recall@5"] for m in trial_metrics])),
                    "recall@10": float(np.mean([m["recall@10"] for m in trial_metrics])),
                    "mrr": float(np.mean([m["mrr"] for m in trial_metrics])),
                })

            per_query_df = pd.DataFrame(per_query_rows)
            boot_dict = self.bootstrap_metrics(per_query_df=per_query_df, bootstrap_indices=bootstrap_indices)

            for i in range(len(boot_dict["recall@1"])):
                all_boot_rows.append({
                    "setting": "A",
                    "dataset_name": self.dataset_name,
                    "model_name": self.model_name,
                    "variant_name": self.variant_name,
                    "N": N,
                    "bootstrap_id": i,
                    "recall@1": float(boot_dict["recall@1"][i]),
                    "recall@5": float(boot_dict["recall@5"][i]),
                    "recall@10": float(boot_dict["recall@10"][i]),
                    "mrr": float(boot_dict["mrr"][i]),
                })

            all_summary_rows.append({
                "setting": "A",
                "dataset_name": self.dataset_name,
                "model_name": self.model_name,
                "variant_name": self.variant_name,
                "split_name": self.split_name,
                "N": int(N),
                "num_queries": int(len(per_query_df)),
                "num_trials_per_query": int(self.n_trials_settingA),
                "recall@1": self._summary_str_pct(boot_dict["recall@1"]),
                "recall@5": self._summary_str_pct(boot_dict["recall@5"]),
                "recall@10": self._summary_str_pct(boot_dict["recall@10"]),
                "mrr": self._summary_str_pct(boot_dict["mrr"]),
            })
            print(all_summary_rows)

        return {
            "summary_df": pd.DataFrame(all_summary_rows),
            "raw_trial_df": pd.DataFrame(raw_trial_rows),
            "bootstrap_df": pd.DataFrame(all_boot_rows),
        }

    def _build_label_index_structures(self, labels):
        labels = np.asarray(labels, dtype=np.int32)
        label_tuples = [tuple(x.tolist()) for x in labels]

        exact_map = {}
        for idx, t in enumerate(label_tuples):
            exact_map.setdefault(t, []).append(idx)

        return labels, label_tuples, exact_map

    def _sample_settingB_candidates(self, q_idx, labels, exact_map, N, rng):
        query_label = labels[q_idx]
        query_tuple = tuple(query_label.tolist())

        same_exact = [idx for idx in exact_map.get(query_tuple, []) if idx != q_idx]

        selected = []

        if len(same_exact) >= (N - 1):
            selected = rng.choice(np.asarray(same_exact), size=N - 1, replace=False).tolist()
        else:
            selected.extend(same_exact)

            remaining = (N - 1) - len(selected)
            all_indices = np.arange(len(labels))
            valid_pool = all_indices[all_indices != q_idx]
            used = set(selected)

            not_used = [idx for idx in valid_pool.tolist() if idx not in used]

            hamming_buckets = {}
            for idx in not_used:
                dist = int(np.sum(np.abs(labels[idx] - query_label)))
                hamming_buckets.setdefault(dist, []).append(idx)

            for dist in sorted(hamming_buckets.keys()):
                bucket = hamming_buckets[dist]
                if len(bucket) <= remaining:
                    rng.shuffle(bucket)
                    selected.extend(bucket)
                    remaining -= len(bucket)
                else:
                    chosen = rng.choice(np.asarray(bucket), size=remaining, replace=False).tolist()
                    selected.extend(chosen)
                    remaining = 0

                if remaining == 0:
                    break

        if len(selected) != (N - 1):
            raise ValueError(f"Could not build Setting B candidate pool for query {q_idx}. Got {len(selected)} negatives.")

        cand_idx = np.concatenate([[q_idx], np.asarray(selected, dtype=np.int64)])
        rng.shuffle(cand_idx)
        return cand_idx



    def run_settingB(self, emb_dict):
        Z_img = emb_dict["Z_img"].float().to(self.device)
        Z_txt = emb_dict["Z_txt"].float().to(self.device)

        row_ids = emb_dict["row_ids"]
        subject_ids = emb_dict["subject_ids"]
        study_ids = emb_dict["study_ids"]
        labels = emb_dict["labels"].astype(np.int32)

        M = len(row_ids)
        N = min(10000, M)

        labels_arr, _, exact_map = self._build_label_index_structures(labels)
        bootstrap_indices = self._get_or_create_bootstrap_indices(n_queries=M, seed=13)

        all_summary_rows = []
        raw_trial_rows = []
        all_boot_rows = []
        per_query_rows = []

        print(f"\nRunning Setting B | dataset={self.dataset_name} | model=BioViL-T Entropy | variant={self.variant_name} | N={N}")

        for q_idx in tqdm(range(M)):
            z_img_i = Z_img[q_idx]
            trial_metrics = []

            for trial_id in range(self.n_trials_settingB):
                seed = self._seed_for_setting_b(trial_id)
                rng = np.random.RandomState(self._safe_seed(seed + q_idx * 100003))

                cand_idx = self._sample_settingB_candidates(
                    q_idx=q_idx,
                    labels=labels_arr,
                    exact_map=exact_map,
                    N=N,
                    rng=rng,
                )

                m = self._evaluate_one_query_one_trial(z_img_i, Z_txt, q_idx, cand_idx)
                trial_metrics.append(m)

                raw_trial_rows.append({
                    "setting": "B",
                    "dataset_name": self.dataset_name,
                    "model_name": self.model_name,
                    "variant_name": self.variant_name,
                    "N": N,
                    "query_row_id": int(row_ids[q_idx]),
                    "subject_id": int(subject_ids[q_idx]),
                    "study_id": int(study_ids[q_idx]),
                    "trial_id": int(trial_id),
                    "rank": int(m["rank"]),
                    "recall@1": float(m["recall@1"]),
                    "recall@5": float(m["recall@5"]),
                    "recall@10": float(m["recall@10"]),
                    "mrr": float(m["mrr"]),
                })

            per_query_rows.append({
                "dataset_name": self.dataset_name,
                "model_name": self.model_name,
                "variant_name": self.variant_name,
                "N": N,
                "query_row_id": int(row_ids[q_idx]),
                "subject_id": int(subject_ids[q_idx]),
                "study_id": int(study_ids[q_idx]),
                "recall@1": float(np.mean([m["recall@1"] for m in trial_metrics])),
                "recall@5": float(np.mean([m["recall@5"] for m in trial_metrics])),
                "recall@10": float(np.mean([m["recall@10"] for m in trial_metrics])),
                "mrr": float(np.mean([m["mrr"] for m in trial_metrics])),
            })

        per_query_df = pd.DataFrame(per_query_rows)
        boot_dict = self.bootstrap_metrics(per_query_df=per_query_df, bootstrap_indices=bootstrap_indices)

        for i in range(len(boot_dict["recall@1"])):
            all_boot_rows.append({
                "setting": "B",
                "dataset_name": self.dataset_name,
                "model_name": self.model_name,
                "variant_name": self.variant_name,
                "N": N,
                "bootstrap_id": i,
                "recall@1": float(boot_dict["recall@1"][i]),
                "recall@5": float(boot_dict["recall@5"][i]),
                "recall@10": float(boot_dict["recall@10"][i]),
                "mrr": float(boot_dict["mrr"][i]),
            })

        all_summary_rows.append({
            "setting": "B",
            "dataset_name": self.dataset_name,
            "model_name": self.model_name,
            "variant_name": self.variant_name,
            "split_name": self.split_name,
            "N": int(N),
            "num_queries": int(len(per_query_df)),
            "num_trials_per_query": int(self.n_trials_settingB),
            "recall@1": self._summary_str_pct(boot_dict["recall@1"]),
            "recall@5": self._summary_str_pct(boot_dict["recall@5"]),
            "recall@10": self._summary_str_pct(boot_dict["recall@10"]),
            "mrr": self._summary_str_pct(boot_dict["mrr"]),
        })

        return {
            "summary_df": pd.DataFrame(all_summary_rows),
            "raw_trial_df": pd.DataFrame(raw_trial_rows),
            "bootstrap_df": pd.DataFrame(all_boot_rows),
        }

    def save_predictions_csv(self, raw_trial_df, setting_name="A"):
        setting_name = str(setting_name).upper()
        out_path = os.path.join(
            self.out_dir,
            f"predictions_setting{setting_name}_{self.dataset_name}_{self.variant_name}_{self.model_name.replace('/', '_')}_split_{self.split_name}.csv",
        )
        raw_trial_df.to_csv(out_path, index=False)

    def save_bootstrap_csv(self, bootstrap_df, setting_name="A"):
        setting_name = str(setting_name).upper()
        out_path = os.path.join(
            self.out_dir,
            f"bootstrap_setting{setting_name}_{self.dataset_name}_{self.variant_name}_{self.model_name.replace('/', '_')}_split_{self.split_name}.csv",
        )
        bootstrap_df.to_csv(out_path, index=False)

    def save_summary_csv(self, summary_df, setting_name="A"):
        setting_name = str(setting_name).upper()
        out_path = os.path.join(
            self.out_dir,
            f"summary_setting{setting_name}_{self.dataset_name}_{self.variant_name}_{self.model_name.replace('/', '_')}_split_{self.split_name}.csv",
        )
        summary_df.to_csv(out_path, index=False)

    def print_and_save_stats(self, summary_df, setting_name="A"):
        setting_name = str(setting_name).upper()

        msg = (
            f"\n\n----------------------------------------------------------------------------------------\n"
            f"setting: {setting_name}\n"
            f"dataset: {self.dataset_name}\n"
            f"model: {self.model_name}\n"
            f"variant: {self.variant_name}\n"
            f"split: {self.split_name}\n\n"
        )

        for _, row in summary_df.iterrows():
            msg += (
                f"N={int(row['N'])}\n"
                f"Recall@1: {row['recall@1']}\n"
                f"Recall@5: {row['recall@5']}\n"
                f"Recall@10: {row['recall@10']}\n"
                f"MRR: {row['mrr']}\n\n"
            )

        print(msg)

        out_path = os.path.join(
            self.out_dir,
            f"Test_on_setting{setting_name}_{self.dataset_name}_{self.variant_name}_{self.model_name.replace('/', '_')}_split_{self.split_name}.txt",
        )
        with open(out_path, "a") as f:
            f.write(msg)