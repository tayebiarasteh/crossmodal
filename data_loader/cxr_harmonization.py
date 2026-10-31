"""
data_loader/cxr_harmonization.py
Created on July 14, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""
import os
from typing import Optional


RESOLUTION_DIRS = {224: "preprocessed224", 512: "preprocessed"}


def resolution_dirname(resolution: int) -> str:
    try:
        return RESOLUTION_DIRS[int(resolution)]
    except KeyError:
        raise KeyError(
            f"source resolution {resolution!r} has no folder on disk; the available ones are "
            f"{sorted(RESOLUTION_DIRS)}. Set cohorts.<cohort>.source_resolution to one of them.")


def resolve_cxr_image_path(
    dataset: str,
    image_root: str,
    image_key: str,
    resolution: int = 224,
    split: Optional[str] = None,
    image_subdir: Optional[str] = None,
) -> str:
    res = resolution_dirname(resolution)
    key = str(image_key)

    if dataset == "mimic":
        return os.path.join(image_root, key.replace("files/", res + "/"))

    if dataset == "chexpert":
        return os.path.join(image_root, key.replace("CheXpert-v1.0/", f"CheXpert-v1.0/{res}/", 1))

    if dataset == "rexgradient":
        return os.path.join(image_root, res, key.lstrip("./"))

    raise KeyError(f"unknown cohort {dataset!r}; this project builds mimic, chexpert and rexgradient")
