from typing import Literal

import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from pystoi.stoi import BETA, DYN_RANGE, FS, MINFREQ, N_FRAME, NUMBAND, N
from pystoi.utils import thirdoct


class SISDRLoss(object):
    def __init__(self, eps: float = 1e-5, scale_label: bool = True, zero_mean: bool = True):
        self.eps = eps
        self.zero_mean = zero_mean
        self.scale_label = scale_label

        self.l2norm = lambda x: torch.norm(x, dim=1, keepdim=True)

    def __call__(self, estis, labels, length_list):
        utt_num, chunk_length = estis.shape

        with torch.no_grad():
            mask_for_loss = torch.zeros_like(estis)

            for i in range(utt_num):
                mask_for_loss[i, :length_list[i]] = 1.

        labels = labels[:, :chunk_length] * mask_for_loss
        estis = estis[:, :chunk_length] * mask_for_loss

        if self.zero_mean:
            labels = labels - torch.mean(labels, dim=1, keepdim=True)
            estis = estis - torch.mean(estis, dim=1, keepdim=True)

        scale = torch.sum(estis * labels, dim=1, keepdim=True)
        if self.scale_label:
            labels = scale * labels / (self.l2norm(labels) ** 2 + self.eps)
        else:
            estis = scale * estis / (self.l2norm(estis) ** 2 + self.eps)

        loss = self.l2norm(labels) / (self.l2norm(estis - labels) + self.eps)

        return -torch.mean(20 * torch.log10(loss + self.eps))


class ComMagEuclideanLoss(object):
    def __init__(self, alpha, l_type):
        self.alpha = alpha
        self.l_type = l_type

    def __call__(self, est, label, frame_list):
        b_size, _, seq_len, freq_num = est.shape
        mask_for_loss = []
        with torch.no_grad():
            for i in range(b_size):
                tmp_mask = torch.ones((frame_list[i], freq_num, 2), dtype=est.dtype)
                mask_for_loss.append(tmp_mask)
            mask_for_loss = torch.nn.utils.rnn.pad_sequence(mask_for_loss, batch_first=True).to(est.device)
            mask_for_loss = mask_for_loss.permute(0, 3, 1, 2)  # (B,2,T,F)
            mag_mask_for_loss = mask_for_loss[:, 0, ...]
        est_mag, label_mag = torch.norm(est, dim=1), torch.norm(label, dim=1)

        if self.l_type == "L1" or self.l_type == "l1":
            loss_com = (torch.abs(est - label) * mask_for_loss).sum() / mask_for_loss.sum()
            loss_mag = (torch.abs(est_mag - label_mag) * mag_mask_for_loss).sum() / mag_mask_for_loss.sum()
        elif self.l_type == "L2" or self.l_type == "l2":
            loss_com = (torch.square(est - label) * mask_for_loss).sum() / mask_for_loss.sum()
            loss_mag = (torch.square(est_mag - label_mag) * mag_mask_for_loss).sum() / mag_mask_for_loss.sum()
        else:
            raise RuntimeError("only L1 and L2 are supported!")
        return self.alpha * loss_com + (1 - self.alpha) * loss_mag