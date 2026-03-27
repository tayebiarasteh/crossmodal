"""
Created on March 26, 2026.
main_crossmodal.py

https://github.com/tayebiarasteh/crossmodal/
"""

import pdb
import torch
import os
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
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from tqdm import tqdm

from config.serde import open_experiment, create_experiment, delete_experiment, write_config, read_config
from data.data_provider import mimic_retrieval_test_data_loader_2D, mimic_retrieval_collate_fn, chexpert_retrieval_test_data_loader_2D, chexpert_retrieval_collate_fn, mimic_biovilt_dp_data_loader_2D, mimic_biovilt_dp_collate_fn, CachedProjectionDataset, cached_projection_collate_fn
from Evaluations_crossmodal import PredictionCrossModalSettingA, PredictionCrossModalSettingA_BioViLT, PredictionCrossModalSettingA_BiomedCLIP, PredictionCrossModalSettingB, PredictionCrossModalSettingB_BiomedCLIP, PredictionCrossModalSettingB_BioViLT
from Train_Valid_crossmodal import TrainingCrossModalDP, BioViLTProjectionHeadDPModel, BioViLTFeatureExtractor

import warnings
warnings.filterwarnings('ignore')






def main_test_setting_bootstrap(global_config_path="/PATH/config/config.yaml", model_name="openai/clip-vit-base-patch32",
                                use_trained_version=False, trained_model_path=None, dataset_name="mimic", split_name="test", setting_name="A"):

    setting_name = str(setting_name).upper()
    if setting_name not in ["A", "B"]:
        raise ValueError("setting_name must be 'A' or 'B'.")

    params = read_config(global_config_path)

    if dataset_name == 'mimic':
        test_dataset = mimic_retrieval_test_data_loader_2D(cfg_path=global_config_path, mode=split_name)
        test_loader = DataLoader(dataset=test_dataset, batch_size=int(params["CrossModal"]["batch_size"]),
                                 pin_memory=True, drop_last=False, shuffle=False,
                                 num_workers=int(params["CrossModal"]["num_workers"]),
                                 collate_fn=mimic_retrieval_collate_fn, persistent_workers=True)

    elif dataset_name == 'chexpert':
        test_dataset = chexpert_retrieval_test_data_loader_2D(cfg_path=global_config_path, mode=split_name)
        test_loader = DataLoader(dataset=test_dataset, batch_size=int(params["CrossModal"]["batch_size"]),
                                 pin_memory=True, drop_last=False, shuffle=False,
                                 num_workers=int(params["CrossModal"]["num_workers"]),
                                 collate_fn=chexpert_retrieval_collate_fn, persistent_workers=True)


    variant_name = "trained" if use_trained_version else "off_shelf"

    if setting_name == "A":
        if model_name == "microsoft/BiomedVLP-BioViL-T":
            predictor = PredictionCrossModalSettingA_BioViLT(cfg_path=global_config_path, split_name=split_name,
                                                             dataset_name=dataset_name, variant_name=variant_name)

        elif model_name == "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224":
            predictor = PredictionCrossModalSettingA_BiomedCLIP(cfg_path=global_config_path, split_name=split_name,
                                                                dataset_name=dataset_name, variant_name=variant_name)

        else:
            predictor = PredictionCrossModalSettingA(cfg_path=global_config_path, split_name=split_name,
                                                     dataset_name=dataset_name, model_name=model_name,
                                                     variant_name=variant_name)

    elif setting_name == "B":
        if model_name == "microsoft/BiomedVLP-BioViL-T":
            predictor = PredictionCrossModalSettingB_BioViLT(cfg_path=global_config_path, split_name=split_name,
                                                             dataset_name=dataset_name, variant_name=variant_name)

        elif model_name == "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224":
            predictor = PredictionCrossModalSettingB_BiomedCLIP(cfg_path=global_config_path, split_name=split_name,
                                                                dataset_name=dataset_name, variant_name=variant_name)

        else:
            predictor = PredictionCrossModalSettingB(cfg_path=global_config_path, split_name=split_name,
                                                     dataset_name=dataset_name, model_name=model_name,
                                                     variant_name=variant_name)

    predictor.setup_model(trained_model_path=trained_model_path if use_trained_version else None)

    emb_dict = predictor.extract_embeddings(test_loader)

    if setting_name == "A":
        result_dict = predictor.run_settingA(emb_dict)
    elif setting_name == "B":
        result_dict = predictor.run_settingB(emb_dict)

    predictor.save_predictions_csv(result_dict["raw_trial_df"], setting_name=setting_name)
    predictor.save_bootstrap_csv(result_dict["bootstrap_df"], setting_name=setting_name)
    predictor.save_summary_csv(result_dict["summary_df"], setting_name=setting_name)
    predictor.print_and_save_stats(result_dict["summary_df"], setting_name=setting_name)

    return result_dict["summary_df"], result_dict["bootstrap_df"]





def precompute_cached_features(feature_extractor, data_loader, save_prefix):
    all_img = []
    all_txt = []

    for batch in tqdm(data_loader):
        img_features = feature_extractor.extract_image_backbone_features(batch["image_paths"])
        txt_features = feature_extractor.extract_text_backbone_features(batch["report_texts"])
        all_img.append(img_features)
        all_txt.append(txt_features)

    img_tensor = torch.cat(all_img, dim=0)
    txt_tensor = torch.cat(all_txt, dim=0)

    torch.save(img_tensor, save_prefix + "_img.pt")
    torch.save(txt_tensor, save_prefix + "_txt.pt")

    return img_tensor, txt_tensor




def main_train_biovilt_DP(global_config_path="/PATH/config/config.yaml", valid=True, resume=False,
                          experiment_name='biovilt_dp_head', lr=1e-3):

    if resume:
        params = open_experiment(experiment_name, global_config_path)
    else:
        params = create_experiment(experiment_name, global_config_path)

    cfg_path = params["cfg_path"]

    # raw paired dataset for feature caching
    train_dataset_raw = mimic_biovilt_dp_data_loader_2D(cfg_path=cfg_path, mode='train')
    valid_dataset_raw = mimic_biovilt_dp_data_loader_2D(cfg_path=cfg_path, mode='test')

    raw_train_loader = DataLoader(
        dataset=train_dataset_raw,
        batch_size=int(params["CrossModal"]["feature_cache_batch_size"]),
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        num_workers=int(params["CrossModal"]["num_workers"]),
        collate_fn=mimic_biovilt_dp_collate_fn,
        persistent_workers=True,
    )

    raw_valid_loader = DataLoader(
        dataset=valid_dataset_raw,
        batch_size=int(params["CrossModal"]["feature_cache_batch_size"]),
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        num_workers=8,
        collate_fn=mimic_biovilt_dp_collate_fn,
        persistent_workers=True,
    )

    # cache frozen features
    cache_dir = os.path.join(params["target_dir"], 'cached_biovilt_features')
    os.makedirs(cache_dir, exist_ok=True)

    train_img_path = os.path.join(cache_dir, "train_img.pt")
    train_txt_path = os.path.join(cache_dir, "train_txt.pt")
    valid_img_path = os.path.join(cache_dir, "test_img.pt")
    valid_txt_path = os.path.join(cache_dir, "test_txt.pt")

    if all(os.path.exists(p) for p in [train_img_path, train_txt_path, valid_img_path, valid_txt_path]):
        train_img = torch.load(train_img_path, map_location="cpu")
        train_txt = torch.load(train_txt_path, map_location="cpu")
        valid_img = torch.load(valid_img_path, map_location="cpu")
        valid_txt = torch.load(valid_txt_path, map_location="cpu")
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        feature_extractor = BioViLTFeatureExtractor(
            model_name="microsoft/BiomedVLP-BioViL-T",
            max_text_tokens=int(params["CrossModal"]["max_text_tokens_biovilt"]),
            device=device,
        )

        train_img, train_txt = precompute_cached_features(
            feature_extractor=feature_extractor,
            data_loader=raw_train_loader,
            save_prefix=os.path.join(cache_dir, "train"),
        )
        valid_img, valid_txt = precompute_cached_features(
            feature_extractor=feature_extractor,
            data_loader=raw_valid_loader,
            save_prefix=os.path.join(cache_dir, "test"),
        )

    # cached tiny-head dataset
    train_dataset = CachedProjectionDataset(train_img, train_txt)
    valid_dataset = CachedProjectionDataset(valid_img, valid_txt)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=int(params['DP']['logical_batch_size']),
        pin_memory=True,
        drop_last=True,
        shuffle=True,
        num_workers=4,
        collate_fn=cached_projection_collate_fn,
        persistent_workers=True,
    )

    valid_loader = None
    if valid:
        valid_loader = DataLoader(
            dataset=valid_dataset,
            batch_size=int(params['DP']['logical_batch_size']),
            pin_memory=True,
            drop_last=False,
            shuffle=False,
            num_workers=2,
            collate_fn=cached_projection_collate_fn,
            persistent_workers=True,
        )

    # tiny cached-head model
    model = BioViLTProjectionHeadDPModel(model_name="microsoft/BiomedVLP-BioViL-T")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.train()

    errors = ModuleValidator.validate(model)
    if len(errors) != 0:
        print("Opacus validation warnings/errors:")
        for err in errors:
            print(err)
        raise ValueError("Cached-head model is still not DP-compatible.")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=lr,
        weight_decay=float(params['Network']['weight_decay']),
    )

    model_info = params['Network']
    model_info['lr'] = lr
    model_info['batch_size'] = int(params['DP']['logical_batch_size'])
    params['Network'] = model_info
    write_config(params, cfg_path, sort_keys=True)

    privacy_engine = PrivacyEngine()

    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        epochs=int(params['Network']['num_epochs']),
        target_epsilon=float(params['DP']['epsilon']),
        target_delta=float(params['DP']['delta']),
        max_grad_norm=float(params['DP']['max_grad_norm']),
    )

    trainer = TrainingCrossModalDP(
        cfg_path=cfg_path,
        resume=resume,
        physical_batch_size=int(params['DP']['physical_batch_size']),
    )

    if resume:
        trainer.load_checkpoint_DP(
            model=model,
            optimiser=optimizer,
            privacy_engine=privacy_engine,
        )
    else:
        trainer.setup_model(
            model=model,
            optimiser=optimizer,
            privacy_engine=privacy_engine,
        )

    trainer.train_DP_epoch(
        train_loader=train_loader,
        valid_loader=valid_loader,
        num_epochs=int(params['Network']['num_epochs']),
    )










if __name__ == '__main__':
    global_config_path = "/PATH/crossmodal/config/config.yaml"

    delete_experiment(global_config_path=global_config_path, experiment_name='biovilt_dp_head_temp')

    main_train_biovilt_DP(global_config_path=global_config_path, valid=True, resume=False,
                          experiment_name='biovilt_dp_head_temp', lr=5e-4)


    # model_name = "microsoft/BiomedVLP-BioViL-T"
    # # model_name = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    # # model_name = "flaviagiammarino/pubmed-clip-vit-base-patch32"
    # # model_name = "openai/clip-vit-base-patch32"
    #
    # dataset_name = 'mimic'
    # # dataset_name = 'chexpert'
    #
    # main_test_setting_bootstrap(global_config_path=global_config_path, model_name=model_name,
    #                              use_trained_version=False, trained_model_path=None, dataset_name=dataset_name, setting_name="A")
