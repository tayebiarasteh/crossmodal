"""
encoders/extract_embeddings.py
Created on August 8, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import hashlib
import json
import os
import shutil
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from config.serde import read_config
from data_loader.build_paired_pool import load_pool, load_reports
from data_loader.cxr_harmonization import resolve_cxr_image_path
from encoders.image_text_encoders import (assert_dim, encode_images,
                                          encode_images_biovilt, encode_texts,
                                          get_spec, load_model, unload)
from Inference.resume_utils import MissingInput


def _cache_path(cfg: dict, model: str, cohort: str, split: str,
                tower: str, transform: Optional[str] = None) -> str:
    d = os.path.join(cfg["paths"]["embeddings_dir"], model, cohort)
    stem = f"{tower}_{split}" if transform is None else f"{tower}_{transform}_{split}"
    return os.path.join(d, stem + ".npz")


def source_resolution(cfg: dict, cohort: str) -> int:
    return int(cfg["cohorts"][cohort].get("source_resolution", 224))


def _build_params(cfg: dict, model: str, spec: dict, cohort: str, split: str,
                  tower: str, transform: Optional[str]) -> dict:
    p = {
        "model": model, "hf_id": spec["hf_id"], "type": spec["type"], "tower": tower,
        "cohort": cohort, "split": split, "transform": transform,
        "image_size": spec.get("image_size"), "max_text_tokens": spec.get("max_text_tokens"),
        "dtype": cfg["extraction"].get("dtype", "float32"),
    }
    if tower == "img" and cohort in cfg.get("cohorts", {}):
        p["source_resolution"] = source_resolution(cfg, cohort)
    return p


def _cache_is_current(path: str, want_params: dict, want_ids: List[str]) -> bool:
    if not os.path.exists(path):
        return False
    try:
        z = np.load(path, allow_pickle=True)
    except Exception as exc:
        print(f"  [cache] unreadable {os.path.basename(path)} ({exc}); rebuilding", flush=True)
        return False
    stored = {}
    if "_build_params" in z.files:
        try:
            stored = json.loads(str(z["_build_params"].item()))
        except Exception:
            stored = {}
    moved = [k for k, v in want_params.items() if k in stored and stored[k] != v]
    if moved:
        print(f"  [cache] REBUILD {os.path.basename(path)}: parameters moved: "
              f"{[(k, stored[k], want_params[k]) for k in moved]}", flush=True)
        return False
    ids = [str(x) for x in z["case_ids"]]
    if ids != list(want_ids):
        print(f"  [cache] REBUILD {os.path.basename(path)}: the cached case_id list is not the "
              f"manifest's ({len(ids)} cached against {len(want_ids)} wanted, order compared)",
              flush=True)
        return False
    return True


def _save_cache(path: str, emb: np.ndarray, ids: List[str], params: dict) -> None:
    if emb.shape[0] == 0:
        raise ValueError(f"refusing to write an empty cache to {path}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez(tmp, embeddings=emb.astype(np.float32),
             case_ids=np.array([str(i) for i in ids]),
             _build_params=np.array(json.dumps(params, sort_keys=True)))
    fd = os.open(tmp, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, path)


def _shard_dir(path: str, params: Optional[dict] = None) -> str:
    if not params:
        return path.replace(".npz", "_shards")
    sha = hashlib.blake2b(json.dumps(params, sort_keys=True).encode(), digest_size=6).hexdigest()
    return path.replace(".npz", f"_shards_{sha}")


def _run_sharded(path: str, n: int, shard_rows: int, encode_block, desc: str,
                 params: Optional[dict] = None) -> np.ndarray:
    sd = _shard_dir(path, params)
    os.makedirs(sd, exist_ok=True)
    n_shards = int(np.ceil(n / shard_rows)) if n else 0
    out = None
    for k in tqdm(range(n_shards), desc=f"{desc}:shards", unit="shard"):
        spath = os.path.join(sd, f"shard{k:05d}.npy")
        lo, hi = k * shard_rows, min((k + 1) * shard_rows, n)
        if os.path.exists(spath):
            block = np.load(spath)
        else:
            block = encode_block(lo, hi, f"{desc} shard {k + 1}/{n_shards}")
            if block is None or len(block) == 0:
                raise RuntimeError(f"{desc}: shard {k} produced nothing")
            block = np.asarray(block, dtype=np.float32)
            tmp = spath + ".tmp.npy"
            np.save(tmp, block)
            os.replace(tmp, spath)
        if out is None:
            out = np.empty((n, block.shape[1]), dtype=np.float32)
            print(f"  [{desc}] output block {out.nbytes / 1e9:.2f} GB "
                  f"({n} x {block.shape[1]})", flush=True)
        if block.shape[0] != hi - lo:
            raise RuntimeError(f"{desc}: shard {k} has {block.shape[0]} rows for {hi - lo} cases")
        out[lo:hi] = block
        del block
    if out is None:
        raise RuntimeError(f"{desc}: no shard produced anything")
    return out


def _drop_shards(path: str, params: Optional[dict] = None) -> None:
    shutil.rmtree(_shard_dir(path, params), ignore_errors=True)


def _batch_for(cfg: dict, spec: dict) -> int:
    if spec.get("batch_size"):
        return int(spec["batch_size"])
    base = int(cfg["extraction"].get("batch_size", 64))
    px = int(spec.get("image_size", 224))
    return max(4, int(base * (224.0 / max(px, 1)) ** 2))


def _encode_with_oom_recovery(fn, items, batch: int, desc: str):
    out, i = [], 0
    b = int(batch)
    while i < len(items):
        chunk = items[i:i + b]
        try:
            out.append(fn(chunk))
            i += len(chunk)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if b > 1:
                b = max(1, b // 2)
                print(f"  [{desc}] CUDA OOM; halving the batch to {b}", flush=True)
                continue
            free, total = (torch.cuda.mem_get_info() if torch.cuda.is_available() else (0, 0))
            raise RuntimeError(
                f"{desc}: out of memory at BATCH SIZE 1, so no batch knob can fix it. "
                f"The card reports {free / 1e9:.1f} GB free of {total / 1e9:.1f} GB. "
                f"Lower the model's image_size or run it on a larger GPU.")
    return out


def _width_checked(fn, model: str, spec: dict, tower: str):
    state = {"checked": False}

    def wrapped(chunk):
        out = fn(chunk)
        if not state["checked"] and out is not None and len(out):
            assert_dim(model, np.asarray(out), spec, tower)
            state["checked"] = True
        return out

    return wrapped


def purge_hub_cache(cfg_path: str, model: str) -> str:
    params = read_config(cfg_path)
    cfg = params["CrossModal"]
    hf_id = get_spec(model, cfg)["hf_id"]
    home = os.environ.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    d = os.path.join(home, "hub", "models--" + hf_id.replace("/", "--"))
    if not os.path.isdir(d):
        print(f"[purge] {model}: nothing cached at {d}", flush=True)
        return d
    size = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(d) for f in fs)
    shutil.rmtree(d, ignore_errors=True)
    print(f"[purge] {model}: removed {d} ({size / 1e9:.1f} GB). The .npz caches are untouched.",
          flush=True)
    return d


def extract_image_embeddings(cfg_path: str, model: str, cohorts: Optional[List[str]] = None,
                             splits: Optional[List[str]] = None, force: bool = False,
                             device: str = "cuda") -> List[str]:
    params = read_config(cfg_path)
    cfg = params["CrossModal"]
    from Inference.rows import append_status
    status_dir, stage = cfg["paths"]["status_dir"], f"embed_img_{model}"

    spec = get_spec(model, cfg)
    cohorts = cohorts or [c for c, cc in cfg["cohorts"].items() if cc.get("enabled", True)]
    splits = splits or list(cfg["extraction"].get("splits", ["test"]))
    shard_rows = int(cfg["extraction"].get("shard_rows", 20000))
    batch = _batch_for(cfg, spec)
    device = device if torch.cuda.is_available() else "cpu"
    append_status(status_dir, stage, f"batch size {batch} at {spec.get('image_size')} px")

    written = []
    for cohort in cohorts:
        image_root = cfg["cohorts"][cohort]["image_root"]
        for split in splits:
            path = _cache_path(cfg, model, cohort, split, "img")
            pool = load_pool(cfg, cohort, split=split,
                             usecols=["case_id", "dataset", "split", "image_key", "image_subdir"])
            if pool.empty:
                raise MissingInput(f"{cohort}/{split} has no rows in the pool manifest")
            ids = pool["case_id"].astype(str).tolist()
            want = _build_params(cfg, model, spec, cohort, split, "img", None)

            if not force and _cache_is_current(path, want, ids):
                append_status(status_dir, stage, f"{cohort}/{split}: cached, skipping")
                written.append(path)
                continue

            src_res = source_resolution(cfg, cohort)
            append_status(status_dir, stage,
                          f"{cohort}/{split}: encoding {len(ids)} images from the "
                          f"{src_res} px source")
            mdl, proc, _ = load_model(model, cfg_path, device)
            paths = [resolve_cxr_image_path(dataset=cohort, image_root=image_root,
                                            image_key=str(k), resolution=src_res,
                                            split=split, image_subdir=sd)
                     for k, sd in zip(pool["image_key"], pool.get("image_subdir", pd.Series([None] * len(pool))))]
            if not os.path.exists(paths[0]):
                raise MissingInput(
                    f"{cohort}/{split}: the first image resolves to {paths[0]}, which does not "
                    f"exist. Check cohorts.{cohort}.image_root and .source_resolution before "
                    "spending a GPU pass on a path that cannot be read.")

            dtype = (torch.float32 if cfg["extraction"].get("dtype") == "float32"
                     else torch.float16)

            def _one(chunk, _mdl=mdl, _proc=proc):
                if spec["type"] == "biovilt":
                    return encode_images_biovilt(_mdl[0], chunk)
                imgs = [Image.open(p).convert("RGB") for p in chunk]
                return encode_images(_mdl, _proc, spec, imgs, device, dtype)

            _one = _width_checked(_one, model, spec, "image")

            def encode_block(lo, hi, desc, _paths=paths):
                parts = _encode_with_oom_recovery(_one, _paths[lo:hi], batch, desc)
                return np.concatenate(parts, axis=0) if parts else None

            emb = _run_sharded(path, len(ids), shard_rows, encode_block, f"{model}:{cohort}/{split}",
                               params=want)
            assert_dim(model, emb, spec, "image")
            _save_cache(path, emb, ids, want)
            _drop_shards(path, want)
            append_status(status_dir, stage,
                          f"{cohort}/{split}: wrote {emb.shape[0]} x {emb.shape[1]}")
            written.append(path)
            print(f"[{stage}] {path}", flush=True)

    unload()
    return written


def extract_text_embeddings(cfg_path: str, model: str, cohorts: Optional[List[str]] = None,
                            transforms: Optional[List[str]] = None,
                            splits: Optional[List[str]] = None, force: bool = False,
                            device: str = "cuda") -> List[str]:
    params = read_config(cfg_path)
    cfg = params["CrossModal"]
    from Inference.rows import append_status
    status_dir, stage = cfg["paths"]["status_dir"], f"embed_txt_{model}"

    spec = get_spec(model, cfg)
    if spec["type"] == "txrv":
        raise MissingInput("the TorchXRayVision baseline has no text tower; it never runs text extraction.")
    cohorts = cohorts or [c for c, cc in cfg["cohorts"].items() if cc.get("enabled", True)]
    transforms = transforms or list(cfg["report_transforms"]["order"])
    splits = splits or list(cfg["extraction"].get("splits", ["test"]))
    shard_rows = int(cfg["extraction"].get("shard_rows", 20000))
    batch = _batch_for(cfg, spec)
    device = device if torch.cuda.is_available() else "cpu"

    written = []
    for cohort in cohorts:
        for transform in transforms:
            reports = load_reports(cfg, cohort, transform).set_index("case_id")
            for split in splits:
                path = _cache_path(cfg, model, cohort, split, "txt", transform)
                pool = load_pool(cfg, cohort, split=split, usecols=["case_id", "split"])
                ids = pool["case_id"].astype(str).tolist()
                missing = [i for i in ids if i not in reports.index]
                if missing:
                    raise MissingInput(
                        f"{cohort}/{transform}: {len(missing)} case_id values in the pool have no "
                        f"report row. Rebuild the transform before encoding it.")
                want = _build_params(cfg, model, spec, cohort, split, "txt", transform)

                if not force and _cache_is_current(path, want, ids):
                    append_status(status_dir, stage, f"{cohort}/{transform}/{split}: cached, skipping")
                    written.append(path)
                    continue

                texts = reports.loc[ids, "report_text"].fillna("").astype(str).tolist()
                index_of, uniq_texts = {}, []
                inverse = np.empty(len(texts), dtype=np.int64)
                for i, t in enumerate(texts):
                    j = index_of.get(t)
                    if j is None:
                        j = len(uniq_texts)
                        index_of[t] = j
                        uniq_texts.append(t)
                    inverse[i] = j
                append_status(status_dir, stage,
                              f"{cohort}/{transform}/{split}: encoding {len(uniq_texts)} distinct "
                              f"reports for {len(texts)} cases")
                mdl, proc, _ = load_model(model, cfg_path, device)

                dtype = (torch.float32 if cfg["extraction"].get("dtype") == "float32"
                         else torch.float16)

                def _one(chunk, _mdl=mdl, _proc=proc):
                    return encode_texts(_mdl, _proc, spec, chunk, device, dtype)

                _one = _width_checked(_one, model, spec, "text")

                def encode_block(lo, hi, desc, _texts=uniq_texts):
                    parts = _encode_with_oom_recovery(_one, _texts[lo:hi], batch, desc)
                    return np.concatenate(parts, axis=0) if parts else None

                uniq_emb = _run_sharded(path, len(uniq_texts), shard_rows, encode_block,
                                        f"{model}:{cohort}/{transform}/{split}", params=want)
                emb = uniq_emb[inverse]
                del uniq_emb
                assert_dim(model, emb, spec, "text")
                _save_cache(path, emb, ids, want)
                _drop_shards(path, want)
                append_status(status_dir, stage,
                              f"{cohort}/{transform}/{split}: wrote {emb.shape[0]} x {emb.shape[1]}")
                written.append(path)
                print(f"[{stage}] {path}", flush=True)

    unload()
    return written


def load_embeddings(cfg: dict, model: str, cohort: str, split: str, tower: str,
                    transform: Optional[str] = None) -> Dict[str, np.ndarray]:
    path = _cache_path(cfg, model, cohort, split, tower, transform)
    if not os.path.exists(path):
        raise MissingInput(
            f"no {tower} embeddings for {model}/{cohort}/{transform or '-'}/{split}. "
            f"Run main_extract_{'image' if tower == 'img' else 'text'}_embeddings(model='{model}').")
    z = np.load(path, allow_pickle=True)
    return {"embeddings": z["embeddings"], "case_ids": [str(x) for x in z["case_ids"]]}
