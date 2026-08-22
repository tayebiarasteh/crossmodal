"""
main_crossmodal.py
Created on August 8, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import functools
import hashlib
import inspect
import os
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(line_buffering=True)
    except AttributeError:
        pass

GLOBAL_CONFIG_PATH = "/PATH/config/config.yaml"


_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

PENDING_BLOCKS: list = []
PENDING_PATHS: list = []
_SOURCE_FINGERPRINT = {}


def _fingerprint_sources():
    out = {}
    for root, dirs, files in os.walk(_REPO_ROOT):
        dirs[:] = [d for d in dirs if d not in (".git", "__pycache__", "logs", ".idea")]
        for name in sorted(files):
            if not name.endswith(".py"):
                continue
            path = os.path.join(root, name)
            with open(path, "rb") as f:
                out[path] = hashlib.blake2b(f.read(), digest_size=16).hexdigest()
    return out


def _guard(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        current = _fingerprint_sources()
        me = os.path.abspath(__file__)
        changed = [p for p, h in current.items()
                   if p != me and _SOURCE_FINGERPRINT.get(p) != h]
        gone = [p for p in _SOURCE_FINGERPRINT
                if p != me and p not in current]
        if changed or gone:
            raise RuntimeError(
                "repository source changed under a running process; this run is now "
                f"mixed-version. changed={changed} removed={gone}")
        if _SOURCE_FINGERPRINT.get(me) != current.get(me):
            print("[guard] the controller file changed on disk; the running process keeps "
                  "the version it started with.", flush=True)
            _SOURCE_FINGERPRINT[me] = current.get(me)
        bound = inspect.signature(fn).bind_partial(*args, **kwargs)
        bound.apply_defaults()
        _assert_roots(bound.arguments.get("cfg_path", GLOBAL_CONFIG_PATH))
        return fn(*args, **kwargs)
    return wrapper


def _assert_roots(cfg_path: str):
    from config.serde import read_config
    cfg = read_config(cfg_path)
    for key in ("datasets_root", "xray_root", "repo_root"):
        root = cfg.get(key)
        if not root or not os.path.isdir(root):
            raise FileNotFoundError(
                f"config root '{key}' resolves to {root!r}, which does not exist. "
                "Set ocean_root / home_root / user_home for this machine.")
    for path in cfg["CrossModal"]["paths"].values():
        os.makedirs(path, exist_ok=True)


def main_build_paired_pool(cfg_path: str = GLOBAL_CONFIG_PATH, cohort: str = "mimic",
                           force: bool = False):
    from data_loader.build_paired_pool import build_paired_pool
    return build_paired_pool(cfg_path, cohort, force=force)


def main_label_reports_llm(cfg_path: str = GLOBAL_CONFIG_PATH, cohort: str = "rexgradient",
                           force: bool = False):
    from data_loader.report_labeler import label_reports
    return label_reports(cfg_path, cohort, force=force)


def main_build_report_transforms(cfg_path: str = GLOBAL_CONFIG_PATH, cohort: str = "mimic",
                                 transforms=None, force: bool = False):
    from data_loader.report_transforms import build_report_transforms
    return build_report_transforms(cfg_path, cohort, transforms=transforms, force=force)


def main_extract_image_embeddings(cfg_path: str = GLOBAL_CONFIG_PATH, model: str = "biovilt",
                                  cohorts=None, splits=None, force: bool = False):
    from encoders.extract_embeddings import extract_image_embeddings
    return extract_image_embeddings(cfg_path, model, cohorts=cohorts, splits=splits, force=force)


def main_extract_text_embeddings(cfg_path: str = GLOBAL_CONFIG_PATH, model: str = "biovilt",
                                 cohorts=None, transforms=None, splits=None,
                                 force: bool = False):
    from encoders.extract_embeddings import extract_text_embeddings
    return extract_text_embeddings(cfg_path, model, cohorts=cohorts, transforms=transforms,
                                   splits=splits, force=force)


def main_purge_model_weights(cfg_path: str = GLOBAL_CONFIG_PATH, model: str = "clip_b32"):
    from encoders.extract_embeddings import purge_hub_cache
    return purge_hub_cache(cfg_path, model)


def main_build_linkage_records(cfg_path: str = GLOBAL_CONFIG_PATH, model: str = "biovilt",
                               cohorts=None, transforms=None, variant: str = "off_shelf",
                               splits=None, force: bool = False):
    from linkage.retrieval import build_linkage_records
    return build_linkage_records(cfg_path, model, cohorts=cohorts, transforms=transforms,
                                 variant=variant, splits=splits, force=force)


def main_e1_linkage(cfg_path: str = GLOBAL_CONFIG_PATH, force: bool = False):
    from analysis.e1_linkage import run_e1
    return run_e1(cfg_path, force=force)


def main_build_conditioned_records(cfg_path: str = GLOBAL_CONFIG_PATH, model: str = "biovilt",
                                   conditioning: str = "all", cohorts=None, transform: str = "T0",
                                   variant: str = "off_shelf", force: bool = False):
    from linkage.conditioning import build_conditioned_records
    return build_conditioned_records(cfg_path, model, conditioning, cohorts=cohorts,
                                     transform=transform, variant=variant, force=force)


def main_e2_conditioning(cfg_path: str = GLOBAL_CONFIG_PATH, force: bool = False):
    from analysis.e2_conditioning import run_e2
    return run_e2(cfg_path, force=force)


def main_e3_report_map(cfg_path: str = GLOBAL_CONFIG_PATH, force: bool = False):
    from analysis.e3_report_map import run_e3
    return run_e3(cfg_path, force=force)


def main_e4_exposure(cfg_path: str = GLOBAL_CONFIG_PATH, force: bool = False):
    from analysis.e4_exposure import run_e4
    return run_e4(cfg_path, force=force)


def main_e5_transform_curve(cfg_path: str = GLOBAL_CONFIG_PATH, force: bool = False):
    from analysis.e5_transform_curve import run_e5
    return run_e5(cfg_path, force=force)


def main_e6_model_defense(cfg_path: str = GLOBAL_CONFIG_PATH, force: bool = False):
    from analysis.e6_model_defense import run_e6
    return run_e6(cfg_path, force=force)


def main_e7_metric_disagreement(cfg_path: str = GLOBAL_CONFIG_PATH, force: bool = False):
    from analysis.e7_metric_disagreement import run_e7
    return run_e7(cfg_path, force=force)


def main_build_baseline_records(cfg_path: str = GLOBAL_CONFIG_PATH, baseline: str = "txrv_labels",
                                cohorts=None, conditioning: str = "none", force: bool = False):
    from linkage.baselines import build_baseline_records
    return build_baseline_records(cfg_path, baseline, cohorts=cohorts,
                                  conditioning=conditioning, force=force)


def main_e8_no_model_adversary(cfg_path: str = GLOBAL_CONFIG_PATH, force: bool = False):
    from analysis.e8_no_model_adversary import run_e8
    return run_e8(cfg_path, force=force)


def main_train_heads(cfg_path: str = GLOBAL_CONFIG_PATH, variant: str = "plain",
                     force: bool = False):
    from mitigation.train_heads import train_heads
    return train_heads(cfg_path, variant, force=force)


def main_reembed_heads(cfg_path: str = GLOBAL_CONFIG_PATH, variant: str = "plain",
                       cohorts=None, force: bool = False):
    from mitigation.train_heads import reembed_with_heads
    return reembed_with_heads(cfg_path, variant, cohorts=cohorts, force=force)


def main_utility_zero_shot(cfg_path: str = GLOBAL_CONFIG_PATH, variant: str = "off_shelf",
                           force: bool = False):
    from mitigation.utility import run_zero_shot
    return run_zero_shot(cfg_path, variant, force=force)


def main_utility_linear_probe(cfg_path: str = GLOBAL_CONFIG_PATH, variant: str = "off_shelf",
                              force: bool = False):
    from mitigation.utility import run_linear_probe
    return run_linear_probe(cfg_path, variant, force=force)


def main_audit_wiring(cfg_path: str = GLOBAL_CONFIG_PATH, strict: bool = False):
    from Inference.wiring_audit import audit_wiring
    return audit_wiring(cfg_path, pending_blocks=PENDING_BLOCKS,
                        pending_paths=PENDING_PATHS, strict=strict)


def main_check_environment(cfg_path: str = GLOBAL_CONFIG_PATH):
    from Inference.wiring_audit import audit_environment
    return audit_environment(cfg_path)


def main_invalidate_partials(cfg_path: str = GLOBAL_CONFIG_PATH, analysis: str = "",
                             contains: str = ""):
    from analysis._common import invalidate_partials
    from config.serde import read_config
    if not analysis:
        raise ValueError("main_invalidate_partials needs the analysis name")
    return invalidate_partials(read_config(cfg_path)["CrossModal"], analysis, contains)


def main_build_final_tables(cfg_path: str = GLOBAL_CONFIG_PATH):
    from aggregate.build_final_tables import build_final_tables
    return build_final_tables(cfg_path)


_SOURCE_FINGERPRINT.update(_fingerprint_sources())
for _n, _f in list(globals().items()):
    if _n.startswith("main_") and callable(_f) and getattr(_f, "__module__", None) == __name__:
        globals()[_n] = _guard(_f)
del _n, _f
