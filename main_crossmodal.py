"""
Created on March 26, 2026.
main_groundingcrossmodal

https://github.com/tayebiarasteh/crossmodal/
"""

import pdb
import torch
import os
from torch.utils.data import Dataset
from torchvision import transforms, models
import numpy as np
from torch import nn
import pandas as pd
import matplotlib
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoImageProcessor, AutoModel
matplotlib.use('Agg')
import yaml
import json
from torch.utils.data import DataLoader

from config.serde import open_experiment, create_experiment, delete_experiment, write_config
from data.data_provider import mimic_retrieval_test_data_loader_2D, mimic_retrieval_collate_fn
from Evaluations_crossmodal import PredictionCrossModalSettingA, PredictionCrossModalSettingA_BioViLT, PredictionCrossModalSettingA_BiomedCLIP, PredictionCrossModalSettingB, PredictionCrossModalSettingB_BiomedCLIP, PredictionCrossModalSettingB_BioViLT
from config.serde import read_config

import warnings
warnings.filterwarnings('ignore')






def main_test_setting_bootstrap(global_config_path="/PATH/config/config.yaml", model_name="openai/clip-vit-base-patch32",
                                use_trained_version=False, trained_model_path=None, dataset_name="mimic", split_name="test",
                                csv_path=None, setting_name="A"):
    if str(dataset_name).lower() != "mimic":
        raise ValueError("This version is MIMIC-only. Use dataset_name='mimic'.")

    setting_name = str(setting_name).upper()
    if setting_name not in ["A", "B"]:
        raise ValueError("setting_name must be 'A' or 'B'.")

    test_dataset = mimic_retrieval_test_data_loader_2D(
        cfg_path=global_config_path,
        csv_path=csv_path,
        mode=split_name,
    )

    params = read_config(global_config_path)

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=int(params["CrossModal"]["batch_size"]),
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        num_workers=int(params["CrossModal"]["num_workers"]),
        collate_fn=mimic_retrieval_collate_fn,
        persistent_workers=True,
    )

    variant_name = "trained" if use_trained_version else "off_shelf"

    if setting_name == "A":
        if model_name == "microsoft/BiomedVLP-BioViL-T":
            predictor = PredictionCrossModalSettingA_BioViLT(
                cfg_path=global_config_path,
                split_name=split_name,
                dataset_name=dataset_name,
                variant_name=variant_name,
            )
        elif model_name == "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224":
            predictor = PredictionCrossModalSettingA_BiomedCLIP(
                cfg_path=global_config_path,
                split_name=split_name,
                dataset_name=dataset_name,
                variant_name=variant_name,
            )
        else:
            predictor = PredictionCrossModalSettingA(
                cfg_path=global_config_path,
                split_name=split_name,
                dataset_name=dataset_name,
                model_name=model_name,
                variant_name=variant_name,
            )
    else:
        if model_name == "microsoft/BiomedVLP-BioViL-T":
            predictor = PredictionCrossModalSettingB_BioViLT(
                cfg_path=global_config_path,
                split_name=split_name,
                dataset_name=dataset_name,
                variant_name=variant_name,
            )
        elif model_name == "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224":
            predictor = PredictionCrossModalSettingB_BiomedCLIP(
                cfg_path=global_config_path,
                split_name=split_name,
                dataset_name=dataset_name,
                variant_name=variant_name,
            )
        else:
            predictor = PredictionCrossModalSettingB(
                cfg_path=global_config_path,
                split_name=split_name,
                dataset_name=dataset_name,
                model_name=model_name,
                variant_name=variant_name,
            )

    predictor.setup_model(
        trained_model_path=trained_model_path if use_trained_version else None
    )

    emb_dict = predictor.extract_embeddings(test_loader)

    if setting_name == "A":
        result_dict = predictor.run_settingA(emb_dict)
    else:
        result_dict = predictor.run_settingB(emb_dict)

    predictor.save_predictions_csv(result_dict["raw_trial_df"], setting_name=setting_name)
    predictor.save_bootstrap_csv(result_dict["bootstrap_df"], setting_name=setting_name)
    predictor.save_summary_csv(result_dict["summary_df"], setting_name=setting_name)
    predictor.print_and_save_stats(result_dict["summary_df"], setting_name=setting_name)

    return result_dict["summary_df"], result_dict["bootstrap_df"]





if __name__ == '__main__':
    global_config_path = "/PATH/crossmodal/config/config.yaml"

    model_name = "microsoft/BiomedVLP-BioViL-T"
    # model_name = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    # model_name = "flaviagiammarino/pubmed-clip-vit-base-patch32"
    # model_name = "openai/clip-vit-base-patch32"

    main_test_setting_bootstrap(global_config_path=global_config_path, model_name=model_name,
                                 use_trained_version=False, trained_model_path=None, dataset_name="mimic", setting_name="B")

    # model_name = "microsoft/BiomedVLP-BioViL-T"
    # model_name = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    model_name = "flaviagiammarino/pubmed-clip-vit-base-patch32"
    # model_name = "openai/clip-vit-base-patch32"

    main_test_setting_bootstrap(global_config_path=global_config_path, model_name=model_name,
                                 use_trained_version=False, trained_model_path=None, dataset_name="mimic", setting_name="A")

