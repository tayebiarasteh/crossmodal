"""
encoders/image_text_encoders.py
Created on August 8, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""
from functools import lru_cache
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from config.serde import read_config


def get_spec(model_name: str, cfg: dict) -> dict:
    panel = cfg["models"]
    if model_name in panel:
        return panel[model_name]
    base = cfg.get("baselines", {})
    if model_name in base:
        return base[model_name]
    raise KeyError(f"[encoders] {model_name!r} is in neither CrossModal.models nor "
                   f"CrossModal.baselines. Available: {sorted(panel) + sorted(base)}")


def _dtype_for(cfg: dict) -> torch.dtype:
    return torch.float32 if str(cfg["extraction"].get("dtype", "float32")) == "float32" else torch.float16


def _l2_norm(x: torch.Tensor) -> np.ndarray:
    return F.normalize(x.float(), p=2, dim=-1).cpu().numpy()


@lru_cache(maxsize=3)
def load_model(model_name: str, cfg_path: str, device: str = "cuda"):
    params = read_config(cfg_path)
    cfg = params["CrossModal"]
    spec = get_spec(model_name, cfg)
    hf_id, etype = spec["hf_id"], spec["type"]
    token = cfg.get("hf_token")
    if token and "YOUR_" in str(token):
        token = None
    dtype = _dtype_for(cfg)

    if etype == "clip_pooled":
        from transformers import AutoModel, AutoProcessor
        model = AutoModel.from_pretrained(hf_id, token=token)
        proc = AutoProcessor.from_pretrained(hf_id, token=token)
        want = int(spec.get("image_size", 224))
        ip = getattr(proc, "image_processor", proc)
        for attr, val in (("size", {"height": want, "width": want}),
                          ("crop_size", {"height": want, "width": want})):
            if hasattr(ip, attr) and getattr(ip, attr) is not None:
                setattr(ip, attr, val)
        model = model.to(device=device, dtype=dtype).eval()
        return model, proc, spec

    if etype == "open_clip":
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(f"hf-hub:{hf_id}")
        tokenizer = open_clip.get_tokenizer(f"hf-hub:{hf_id}")
        model = model.to(device=device, dtype=dtype).eval()
        model._oc_preprocess = preprocess
        model._oc_tokenizer = tokenizer
        return model, "open_clip", spec

    if etype == "biovilt":
        from health_multimodal.image.utils import ImageModelType, get_image_inference
        from transformers import AutoModel, AutoTokenizer
        img_engine = get_image_inference(ImageModelType.BIOVIL_T)
        img_engine.to(device)
        img_engine.model.freeze_encoder = True
        txt_model = AutoModel.from_pretrained(hf_id, trust_remote_code=True,
                                              use_safetensors=False, token=token)
        txt_tok = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True, token=token)
        txt_model = txt_model.to(device=device).eval()
        return (img_engine, txt_model, txt_tok), "biovilt", spec

    if etype == "txrv":
        import functools
        import torchxrayvision as xrv
        _orig_load = torch.load
        torch.load = functools.partial(_orig_load, weights_only=False)
        try:
            model = xrv.models.DenseNet(weights=spec.get("checkpoint", "densenet121-res224-all"))
        finally:
            torch.load = _orig_load
        model = model.to(device=device).eval()
        return model, "txrv", spec

    raise ValueError(f"[encoders] unknown type {etype!r} for {model_name!r}")


def txrv_pathologies(model) -> List[str]:
    return [str(p) for p in model.pathologies]


def unload():
    load_model.cache_clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _prep_images_hf(images: List[Image.Image], processor, device: str, dtype: torch.dtype) -> dict:
    ip = getattr(processor, "image_processor", None)
    attempts = []
    if ip is not None:
        attempts.append(("image_processor", lambda: ip(images=images, return_tensors="pt")))
    attempts.append(("processor", lambda: processor(images=images, return_tensors="pt")))
    attempts.append(("processor+empty text",
                     lambda: processor(images=images, text=[""] * len(images), return_tensors="pt")))

    errs, inputs = [], None
    for label, fn in attempts:
        try:
            inputs = fn()
            break
        except Exception as e:
            errs.append(f"{label}: {type(e).__name__}: {e}")
    if inputs is None:
        raise RuntimeError("image preprocessing failed. Tried:\n  " + "\n  ".join(errs))
    return {k: (v.to(device=device, dtype=dtype if v.is_floating_point() else v.dtype)
                if hasattr(v, "to") else v) for k, v in inputs.items()}


def _resolve_vision_output(out):
    if isinstance(out, torch.Tensor):
        return out[:, 0, :] if out.dim() == 3 else out
    for attr in ("image_embeds", "pooler_output"):
        v = getattr(out, attr, None)
        if v is not None:
            return v
    lhs = getattr(out, "last_hidden_state", None)
    if lhs is not None:
        return lhs[:, 0, :] if lhs.dim() == 3 else lhs
    if isinstance(out, dict):
        for k in ("image_embeds", "pooler_output", "last_hidden_state"):
            v = out.get(k)
            if v is not None:
                return v[:, 0, :] if hasattr(v, "dim") and v.dim() == 3 else v
    if isinstance(out, (tuple, list)) and out and isinstance(out[0], torch.Tensor):
        return out[0][:, 0, :] if out[0].dim() == 3 else out[0]
    return out


def _project(model, pooled, attr: str):
    if pooled is None:
        return None
    proj = getattr(model, attr, None)
    if proj is None:
        return pooled
    want = getattr(getattr(proj, "weight", None), "shape", (None, None))[1]
    if want is not None and pooled.shape[-1] != want:
        return pooled
    return proj(pooled)


def encode_images(model, processor, spec: dict, images: List[Image.Image],
                  device: str, dtype: torch.dtype) -> np.ndarray:
    etype = spec["type"]

    if etype == "open_clip":
        pre = model._oc_preprocess
        batch = torch.stack([pre(im.convert("RGB")) for im in images]).to(device=device, dtype=dtype)
        with torch.no_grad():
            return _l2_norm(model.encode_image(batch))

    if etype == "biovilt":
        raise RuntimeError("BioViL-T images are encoded from file paths, not PIL objects; "
                           "call encode_images_biovilt.")

    if etype == "txrv":
        import torchxrayvision as xrv
        arrs = []
        for im in images:
            a = np.asarray(im.convert("L"), dtype=np.float32)
            a = xrv.datasets.normalize(a, 255)
            a = a[None, ...]
            arrs.append(torch.from_numpy(a))
        batch = torch.stack(arrs).to(device=device)
        with torch.no_grad():
            out = model(batch)
        out = out.float().cpu().numpy()
        if np.nanmin(out) < 0.0 or np.nanmax(out) > 1.0:
            raise ValueError(
                f"the TorchXRayVision checkpoint returned values on "
                f"[{np.nanmin(out):.3f}, {np.nanmax(out):.3f}], which is not a probability. "
                "A checkpoint with no op_threshs returns logits and needs a sigmoid here; "
                "add the branch rather than letting an unnormalized vector reach the ranking.")
        return out

    inputs = _prep_images_hf(images, processor, device, dtype)
    key = "pixel_values" if "pixel_values" in inputs else list(inputs)[0]
    with torch.no_grad():
        emb = None
        if not getattr(model, "_cm_no_get_image_features", False) and hasattr(model, "get_image_features"):
            emb = model.get_image_features(pixel_values=inputs[key])
            if not isinstance(emb, torch.Tensor):
                model._cm_no_get_image_features = True
                emb = None
        if emb is None:
            vision = getattr(model, "vision_model", model)
            pooled = _resolve_vision_output(vision(pixel_values=inputs[key]))
            emb = _project(model, pooled, "visual_projection")
    if not isinstance(emb, torch.Tensor):
        emb = _resolve_vision_output(emb)
    if not isinstance(emb, torch.Tensor):
        raise RuntimeError(f"image extraction did not yield a tensor (got {type(emb).__name__})")
    return _l2_norm(emb)


def encode_images_biovilt(engine, paths: List[str]) -> np.ndarray:
    from pathlib import Path
    embs = [engine.get_projected_global_embedding(Path(p)) for p in paths]
    return _l2_norm(torch.stack([e.detach().cpu() for e in embs]))


def encode_texts(model, processor, spec: dict, texts: List[str],
                 device: str, dtype: torch.dtype) -> np.ndarray:
    etype = spec["type"]
    max_len = int(spec.get("max_text_tokens", 77))

    if etype == "open_clip":
        toks = model._oc_tokenizer(texts, context_length=max_len).to(device)
        with torch.no_grad():
            return _l2_norm(model.encode_text(toks))

    if etype == "biovilt":
        _, txt_model, txt_tok = model
        enc = txt_tok(texts, padding="max_length", truncation=True,
                      max_length=max_len, return_tensors="pt").to(device)
        with torch.no_grad():
            out = txt_model.get_projected_text_embeddings(
                input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        return _l2_norm(out)

    if etype == "txrv":
        raise RuntimeError("the TorchXRayVision baseline has no text tower; it ranks reports "
                           "on their own label vector, not on a text embedding.")

    tok = getattr(processor, "tokenizer", processor)
    enc = tok(texts, padding="max_length", truncation=True, max_length=max_len,
              return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items() if k in ("input_ids", "attention_mask")}
    with torch.no_grad():
        emb = None
        if not getattr(model, "_cm_no_get_text_features", False) and hasattr(model, "get_text_features"):
            emb = model.get_text_features(**enc)
            if not isinstance(emb, torch.Tensor):
                model._cm_no_get_text_features = True
                emb = None
        if emb is None:
            text_model = getattr(model, "text_model", model)
            out = text_model(**enc)
            emb = getattr(out, "text_embeds", None)
            if emb is None:
                pooled = getattr(out, "pooler_output", None)
                if pooled is None:
                    lhs = getattr(out, "last_hidden_state", None)
                    pooled = lhs[:, 0, :] if lhs is not None else None
                emb = _project(model, pooled, "text_projection")
    if not isinstance(emb, torch.Tensor):
        raise RuntimeError(f"text extraction did not yield a tensor (got {type(emb).__name__})")
    return _l2_norm(emb)


def assert_dim(name: str, arr: np.ndarray, spec: dict, tower: str) -> None:
    declared = spec.get("embed_dim")
    if declared is None:
        return
    got = int(arr.shape[1])
    if got != int(declared):
        raise ValueError(
            f"[{name}] the {tower} tower produced {got} dimensions and the config declares "
            f"{declared}. A wrong width silently breaks every cosine in the project, so fix "
            "the config and not the assertion.")
