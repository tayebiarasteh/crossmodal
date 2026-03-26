"""
Created on Feb 1, 2022.
data_provider.py

@author: Soroosh Tayebi Arasteh <soroosh.arasteh@rwth-aachen.de>
https://github.com/tayebiarasteh/crossmodal/
"""

import os

import matplotlib.pyplot as plt
import torch
import pdb
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import cv2

from config.serde import read_config



epsilon = 1e-15






class mimic_retrieval_test_data_loader_2D(Dataset):
    """
    MIMIC test loader for cross-modal image-report retrieval.

    Uses:
        - external prepared CSV for this project
        - AP / PA only
        - all 14 labels
        - report text loaded from report_rel_path
        - image path resolved from jpg_rel_path

    Returned item:
        {
            "row_id": int,
            "subject_id": int,
            "study_id": int,
            "image_path": str,
            "report_text": str,
            "label": torch.FloatTensor [14]
        }
    """

    def __init__(self, cfg_path, csv_path=None, mode='test'):
        self.cfg_path = cfg_path
        self.mode = str(mode).lower()

        from config.serde import read_config
        self.params = read_config(cfg_path)

        if csv_path is None:
            csv_path = self.params["CrossModal"]["mimic_master_csv"]

        self.csv_path = csv_path
        self.image_root = self.params["CrossModal"]["mimic_image_root"]
        self.report_root = self.params["CrossModal"]["mimic_reports_root"]

        self.chosen_labels = [
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

        self.org_df = pd.read_csv(self.csv_path)

        self.org_df["split"] = self.org_df["split"].astype(str).str.strip().str.lower()
        self.org_df["view"] = self.org_df["view"].astype(str).str.strip().str.upper()

        if self.mode == 'train':
            self.subset_df = self.org_df[self.org_df['split'] == 'train'].copy()
        elif self.mode == 'valid':
            self.subset_df = self.org_df[self.org_df['split'] == 'valid'].copy()
        elif self.mode == 'test':
            self.subset_df = self.org_df[self.org_df['split'] == 'test'].copy()
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        PAview = self.subset_df[self.subset_df['view'] == 'PA']
        APview = self.subset_df[self.subset_df['view'] == 'AP']
        self.subset_df = pd.concat([PAview, APview], ignore_index=True)
        self.subset_df = self.subset_df.reset_index(drop=True)

        self.file_path_list = list(self.subset_df['jpg_rel_path'])
        self.report_path_list = list(self.subset_df['report_rel_path'])

    def __len__(self):
        return len(self.subset_df)

    def _resolve_image_path(self, rel_path):
        rel_path = str(rel_path)
        rel_path = rel_path.replace("/files/", "/preprocessed224/")
        return os.path.join(self.image_root, rel_path)

    def _resolve_report_path(self, rel_path):
        return os.path.join(self.report_root, str(rel_path))

    def _read_report(self, report_path):
        with open(report_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = "\n".join([line.strip() for line in text.split("\n") if line.strip() != ""])
        return text

    def __getitem__(self, idx):
        row = self.subset_df.iloc[idx]

        img_path = self._resolve_image_path(row["jpg_rel_path"])
        report_path = self._resolve_report_path(row["report_rel_path"])
        report_text = self._read_report(report_path)

        label = np.zeros((len(self.chosen_labels),), dtype=np.float32)

        for label_idx, disease in enumerate(self.chosen_labels):
            val = row[disease]
            try:
                val = int(val)
            except Exception:
                val = 0

            # exact requested behavior
            if val == 1:
                label[label_idx] = 1.0
            else:
                label[label_idx] = 0.0

        label = torch.from_numpy(label).float()

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
            "image_path": img_path,
            "report_text": report_text,
            "label": label,
        }


def mimic_retrieval_collate_fn(batch):
    return {
        "row_ids": [item["row_id"] for item in batch],
        "subject_ids": [item["subject_id"] for item in batch],
        "study_ids": [item["study_id"] for item in batch],
        "image_paths": [item["image_path"] for item in batch],
        "report_texts": [item["report_text"] for item in batch],
        "labels": torch.stack([item["label"] for item in batch], dim=0),
    }

