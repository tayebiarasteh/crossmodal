"""
mitigation/projection_heads.py
Created on August 8, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _projector(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Module:
    if not hidden_dim:
        return nn.Linear(in_dim, out_dim)
    return nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.GELU(),
                         nn.Linear(hidden_dim, out_dim))


class ProjectionHeads(nn.Module):

    def __init__(self, image_dim: int, text_dim: int, out_dim: int,
                 hidden_dim: int = 0):
        super().__init__()
        self.image_dim = int(image_dim)
        self.text_dim = int(text_dim)
        self.out_dim = int(out_dim)
        self.hidden_dim = int(hidden_dim)
        self.image_projector = _projector(self.image_dim, self.hidden_dim, self.out_dim)
        self.text_projector = _projector(self.text_dim, self.hidden_dim, self.out_dim)
        for p in self.parameters():
            p.requires_grad = True

    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def encode_image(self, img_features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.image_projector(img_features).float(), p=2, dim=-1)

    def encode_text(self, txt_features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.text_projector(txt_features).float(), p=2, dim=-1)

    def forward(self, img_features: torch.Tensor, txt_features: torch.Tensor):
        return self.encode_image(img_features), self.encode_text(txt_features)


def symmetric_infonce(z_img: torch.Tensor, z_txt: torch.Tensor, tau: float) -> torch.Tensor:
    logits = z_img @ z_txt.T / tau
    target = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target))


def fixed_bank_infonce(z_img: torch.Tensor, z_txt: torch.Tensor,
                       bank_txt: torch.Tensor, bank_img: torch.Tensor,
                       tau: float) -> torch.Tensor:
    pos_i2t = (z_img * z_txt).sum(dim=-1, keepdim=True) / tau
    neg_i2t = z_img @ bank_txt.T / tau
    loss_i2t = -pos_i2t.squeeze(1) + torch.logsumexp(
        torch.cat([pos_i2t, neg_i2t], dim=1), dim=1)

    pos_t2i = (z_txt * z_img).sum(dim=-1, keepdim=True) / tau
    neg_t2i = z_txt @ bank_img.T / tau
    loss_t2i = -pos_t2i.squeeze(1) + torch.logsumexp(
        torch.cat([pos_t2i, neg_t2i], dim=1), dim=1)

    return 0.5 * (loss_i2t + loss_t2i)


def group_privacy_epsilon(epsilon: float, max_records_per_patient: int) -> float:
    return float(epsilon) * int(max(1, max_records_per_patient))
