"""
Created on March 27, 2026.
Train_Valid_crossmodal.py

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh/crossmodal/
"""

import os.path
import pdb
import numpy as np
import torch
from sklearn import metrics
import copy
from PIL import Image
import os
import time
from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from opacus.utils.batch_memory_manager import BatchMemoryManager
from health_multimodal.image.utils import get_image_inference, ImageModelType

from config.serde import read_config, write_config

import warnings
warnings.filterwarnings('ignore')
epsilon = 1e-15






def get_valid_num_groups(num_channels: int, max_groups: int = 32) -> int:
    divisors = [g for g in range(1, num_channels + 1) if num_channels % g == 0]
    valid = [g for g in divisors if g <= max_groups]
    return max(valid) if len(valid) > 0 else 1


def replace_bn_with_gn(module: nn.Module, max_groups: int = 32):
    for name, sub_module in module.named_children():
        if isinstance(sub_module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            num_channels = sub_module.num_features
            num_groups = get_valid_num_groups(num_channels=num_channels, max_groups=max_groups)

            gn = nn.GroupNorm(
                num_groups=num_groups,
                num_channels=num_channels,
                eps=sub_module.eps,
                affine=sub_module.affine,
            )

            if sub_module.affine:
                with torch.no_grad():
                    gn.weight.copy_(sub_module.weight.data)
                    gn.bias.copy_(sub_module.bias.data)

            setattr(module, name, gn)
        else:
            replace_bn_with_gn(sub_module, max_groups=max_groups)


def replace_inplace_relu(module: nn.Module):
    for name, sub_module in module.named_children():
        if isinstance(sub_module, nn.ReLU):
            setattr(module, name, nn.ReLU(inplace=False))
        else:
            replace_inplace_relu(sub_module)



class BioViLTProjectionHeadDPModel(nn.Module):
    """
    Train only:
        - image_model.projector
        - text_model.cls_projection_head
    """

    def __init__(self, model_name="microsoft/BiomedVLP-BioViL-T", max_text_tokens=256):
        super().__init__()

        self.model_name = model_name
        self.max_text_tokens = int(max_text_tokens)

        self.image_engine = get_image_inference(ImageModelType.BIOVIL_T)
        self.image_transform = self.image_engine.transform
        self.image_model = self.image_engine.model

        self.text_tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=True,
        )
        self.text_model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_safetensors=False,
        )

    def make_dp_compatible(self, max_groups=32):
        replace_bn_with_gn(self.image_model, max_groups=max_groups)
        replace_inplace_relu(self.image_model)

    def freeze_backbones_and_unfreeze_projection_heads(self):
        for p in self.image_model.parameters():
            p.requires_grad = False
        for p in self.text_model.parameters():
            p.requires_grad = False

        for p in self.image_model.projector.parameters():
            p.requires_grad = True

        for p in self.text_model.cls_projection_head.parameters():
            p.requires_grad = True

        trainable_param_num = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\nTrainable projection-head parameters: {trainable_param_num:,}")
        print('----------------------------------------------------\n')

    def _forward_image_projected(self, image_paths, device):
        image_tensors = []
        for p in image_paths:
            img = Image.open(Path(p))
            x = self.image_transform(img)
            image_tensors.append(x)

        x = torch.stack(image_tensors, dim=0).to(device, non_blocking=True)
        out = self.image_model(x)
        z_img = out.projected_global_embedding
        return z_img

    def _forward_text_projected(self, report_texts, device):
        tok = self.text_tokenizer(
            report_texts,
            padding=True,
            truncation=True,
            max_length=self.max_text_tokens,
            return_tensors="pt",
        )
        tok = {k: v.to(device, non_blocking=True) for k, v in tok.items()}

        z_txt = self.text_model.get_projected_text_embeddings(
            input_ids=tok["input_ids"],
            attention_mask=tok["attention_mask"],
        )
        return z_txt

    def forward(self, image_paths, report_texts):
        device = next(self.parameters()).device

        z_img = self._forward_image_projected(image_paths=image_paths, device=device)
        z_txt = self._forward_text_projected(report_texts=report_texts, device=device)

        z_img = F.normalize(z_img.float(), p=2, dim=-1)
        z_txt = F.normalize(z_txt.float(), p=2, dim=-1)
        return z_img, z_txt



class TrainingCrossModalDP:
    def __init__(self, cfg_path, resume=False, physical_batch_size=64):
        self.params = read_config(cfg_path)
        self.cfg_path = cfg_path
        self.physical_batch_size = int(physical_batch_size)

        if resume is False:
            self.model_info = self.params['Network']
            self.epoch = 0
            self.best_val_recall1 = float("inf")
            self.setup_cuda()
            self.writer = SummaryWriter(
                log_dir=os.path.join(self.params['target_dir'], self.params['tb_logs_path'])
            )

    def setup_cuda(self, cuda_device_id=0):
        if torch.cuda.is_available():
            torch.backends.cudnn.fastest = True
            torch.cuda.set_device(cuda_device_id)
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def time_duration(self, start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_hours = int(elapsed_time / 3600)
        if elapsed_hours >= 1:
            elapsed_mins = int((elapsed_time / 60) - (elapsed_hours * 60))
            elapsed_secs = int(elapsed_time - (elapsed_hours * 3600) - (elapsed_mins * 60))
        else:
            elapsed_mins = int(elapsed_time / 60)
            elapsed_secs = elapsed_time - (elapsed_mins * 60)
        return elapsed_hours, elapsed_mins, elapsed_secs

    def setup_model(self, model, optimiser, privacy_engine=None):
        total_param_num = sum(p.numel() for p in model.parameters() if p.requires_grad)

        self.model = model.to(self.device)
        self.optimiser = optimiser

        self.model_info['total_param_num'] = total_param_num
        self.model_info['loss_function'] = "symmetric_infonce"
        self.params['Network'] = self.model_info
        write_config(self.params, self.cfg_path, sort_keys=True)

        if privacy_engine is not None:
            self.privacy_engine = privacy_engine

    def load_checkpoint_DP(self, model, optimiser, privacy_engine):
        checkpoint = torch.load(
            os.path.join(self.params['target_dir'], self.params['network_output_path'], self.params['checkpoint_name']),
            map_location="cpu"
        )

        self.device = None
        self.model_info = checkpoint['model_info']
        self.setup_cuda()
        self.model = model.to(self.device)
        self.optimiser = optimiser
        self.privacy_engine = privacy_engine

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimiser.load_state_dict(checkpoint['optimizer_state_dict'])

        self.privacy_engine.load_checkpoint(
            module=self.model,
            optimizer=self.optimiser,
            path=os.path.join(self.params['target_dir'], self.params['network_output_path'], self.params['DP_checkpoint_name'])
        )

        self.epoch = checkpoint['epoch']
        self.best_val_recall1 = checkpoint.get('best_val_recall1', float("inf"))
        self.writer = SummaryWriter(
            log_dir=os.path.join(self.params['target_dir'], self.params['tb_logs_path']),
            purge_step=self.epoch + 1
        )

    def symmetric_infonce(self, z_img, z_txt, tau=0.07):
        logits = torch.matmul(z_img, z_txt.T) / tau
        targets = torch.arange(logits.shape[0], device=logits.device)

        loss_i2t = F.cross_entropy(logits, targets)
        loss_t2i = F.cross_entropy(logits.T, targets)
        return 0.5 * (loss_i2t + loss_t2i)

    def _compute_recall_at_1(self, z_img, z_txt):
        scores = torch.matmul(z_img, z_txt.T)
        ranks = torch.argsort(scores, dim=1, descending=True)

        hits = []
        for i in range(z_img.shape[0]):
            rank_pos = (ranks[i] == i).nonzero(as_tuple=False).item() + 1
            hits.append(1.0 if rank_pos == 1 else 0.0)
        return float(np.mean(hits))

    def valid_epoch(self, valid_loader):
        self.model.eval()

        batch_losses = []
        all_img = []
        all_txt = []

        with torch.no_grad():
            for batch in tqdm(valid_loader):
                z_img, z_txt = self.model(
                    image_paths=batch["image_paths"],
                    report_texts=batch["report_texts"],
                )

                loss = self.symmetric_infonce(
                    z_img=z_img,
                    z_txt=z_txt,
                    tau=float(self.params["CrossModal"]["temperature"]),
                )
                batch_losses.append(loss.item())

                all_img.append(z_img.detach().cpu())
                all_txt.append(z_txt.detach().cpu())

        Z_img = torch.cat(all_img, dim=0).to(self.device)
        Z_txt = torch.cat(all_txt, dim=0).to(self.device)

        val_loss = float(np.mean(batch_losses)) if len(batch_losses) > 0 else 0.0
        val_recall1 = self._compute_recall_at_1(Z_img, Z_txt)
        return val_loss, val_recall1



    def train_DP_epoch(self, train_loader, valid_loader=None, num_epochs=5):
        self.params = read_config(self.cfg_path)
        total_start_time = time.time()

        for _ in range(num_epochs - self.epoch):
            self.epoch += 1

            batch_loss = 0.0
            start_time = time.time()

            with BatchMemoryManager(
                data_loader=train_loader,
                max_physical_batch_size=self.physical_batch_size,
                optimizer=self.optimiser
            ) as memory_safe_data_loader:

                for idx, batch in enumerate(tqdm(memory_safe_data_loader)):
                    self.model.train()
                    self.optimiser.zero_grad()

                    z_img, z_txt = self.model(
                        image_paths=batch["image_paths"],
                        report_texts=batch["report_texts"],
                    )

                    loss = self.symmetric_infonce(
                        z_img=z_img,
                        z_txt=z_txt,
                        tau=float(self.params["CrossModal"]["temperature"]),
                    )

                    loss.backward()
                    self.optimiser.step()

                    batch_loss += loss.item()

                    if (idx + 1) % 200 == 0:
                        epsilon_now = self.privacy_engine.get_epsilon(float(self.params['DP']['delta']))
                        print(
                            f"\tTrain Epoch: {self.epoch} | "
                            f"Loss: {batch_loss / (idx + 1):.6f} | "
                            f"(ε = {epsilon_now:.2f}, δ = {float(self.params['DP']['delta'])})"
                        )

            train_loss = batch_loss / max(1, len(train_loader))
            epsilon_now = self.privacy_engine.get_epsilon(float(self.params['DP']['delta']))

            self.writer.add_scalar('Train_loss_avg', train_loss, self.epoch)
            self.writer.add_scalar('Epsilon', epsilon_now, self.epoch)

            self.params['Network']['num_epoch'] = self.epoch
            write_config(self.params, self.cfg_path, sort_keys=True)

            self.privacy_engine.save_checkpoint(
                path=os.path.join(self.params['target_dir'], self.params['network_output_path'], self.params['DP_checkpoint_name']),
                module=self.model,
                optimizer=self.optimiser
            )

            torch.save(
                {
                    'epoch': self.epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimiser.state_dict(),
                    'model_info': self.model_info,
                    'best_val_recall1': self.best_val_recall1,
                },
                os.path.join(self.params['target_dir'], self.params['network_output_path'], self.params['checkpoint_name'])
            )

            valid_loss = None
            valid_recall1 = None

            if valid_loader is not None and (self.epoch % self.params['display_stats_freq'] == 0):
                valid_loss, valid_recall1 = self.valid_epoch(valid_loader)
                self.writer.add_scalar('Valid_loss', valid_loss, self.epoch)
                self.writer.add_scalar('Valid_RecallAt1', valid_recall1, self.epoch)

                if valid_recall1 < self.best_val_recall1:
                    self.best_val_recall1 = valid_recall1
                    torch.save(
                        self.model.state_dict(),
                        os.path.join(self.params['target_dir'], self.params['network_output_path'], self.params['trained_model_name'])
                    )

            end_time = time.time()
            total_time = end_time - total_start_time
            iteration_hours, iteration_mins, iteration_secs = self.time_duration(start_time, end_time)
            total_hours, total_mins, total_secs = self.time_duration(total_start_time, end_time)

            self.savings_prints(
                iteration_hours=iteration_hours,
                iteration_mins=iteration_mins,
                iteration_secs=iteration_secs,
                total_hours=total_hours,
                total_mins=total_mins,
                total_secs=total_secs,
                train_loss=train_loss,
                total_time=total_time,
                valid_loss=valid_loss,
                valid_recall1=valid_recall1,
            )


    def savings_prints(self, iteration_hours, iteration_mins, iteration_secs,
                       total_hours, total_mins, total_secs,
                       train_loss, total_time, valid_loss=None, valid_recall1=None):

        self.params['Network']['num_epoch'] = self.epoch
        write_config(self.params, self.cfg_path, sort_keys=True)

        epsilon_now = self.privacy_engine.get_epsilon(float(self.params['DP']['delta']))
        delta_now = float(self.params['DP']['delta'])

        print('----------------------------------------------------------------------------------------')
        print(
            f'epoch: {self.epoch} | '
            f'epoch time: {iteration_hours}h {iteration_mins}m {iteration_secs:.2f}s | '
            f'total time: {total_hours}h {total_mins}m {total_secs:.2f}s'
        )
        print(f'\n\tTrain loss: {train_loss:.6f}, ε = {epsilon_now:.2f} | δ = {delta_now}')

        if valid_loss is not None and valid_recall1 is not None:
            print(f'\tValid loss: {valid_loss:.6f} | Valid Recall@1: {valid_recall1 * 100:.4f}%\n')
            msg = (
                f'\n\n----------------------------------------------------------------------------------------\n'
                f'epoch: {self.epoch} | epoch time: {iteration_hours}h {iteration_mins}m {iteration_secs:.2f}s '
                f'| total time: {total_hours}h {total_mins}m {total_secs:.2f}s\n'
                f'\n\tTrain loss: {train_loss:.6f}, ε = {epsilon_now:.2f} | δ = {delta_now}'
                f' | Valid loss: {valid_loss:.6f} | Valid Recall@1: {valid_recall1 * 100:.4f}%\n'
            )
        else:
            msg = (
                f'\n\n----------------------------------------------------------------------------------------\n'
                f'epoch: {self.epoch} | epoch time: {iteration_hours}h {iteration_mins}m {iteration_secs:.2f}s '
                f'| total time: {total_hours}h {total_mins}m {total_secs:.2f}s\n'
                f'\n\tTrain loss: {train_loss:.6f}, ε = {epsilon_now:.2f} | δ = {delta_now}\n'
            )

        stats_path = os.path.join(self.params['target_dir'], self.params['stat_log_path'], 'Stats')
        os.makedirs(os.path.dirname(stats_path), exist_ok=True)
        with open(stats_path, 'a') as f:
            f.write(msg)


