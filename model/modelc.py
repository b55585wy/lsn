import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from copy import deepcopy
import einops
from model.mea import MEA  # renamed from EMA1D
from model.idc import InceptionDWConv1d
import numpy as np


def augment_signal(x, noise_factor=0.015, scale_range=(0.85, 1.15), time_shift_ratio=0.05):
    noise = torch.randn_like(x) * noise_factor * x.std(dim=-1, keepdim=True)
    x = x + noise
    scale = torch.FloatTensor(x.size(0)).uniform_(*scale_range).to(x.device).view(-1, 1, 1)
    x = x * scale
    max_shift = int(x.size(-1) * time_shift_ratio)
    shifts = torch.randint(-max_shift, max_shift, (x.size(0),)).to(x.device)
    x_out = torch.zeros_like(x)
    for i in range(x.size(0)):
        x_out[i] = torch.roll(x[i], shifts=shifts[i].item(), dims=-1)
    x = x_out
    return x


def normalize_signal(x):
    return (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-6)


class EfficientAdditiveAttention(nn.Module):
    def __init__(self, d_model, afr_reduced_cnn_size, dropout=0.2):
        super(EfficientAdditiveAttention, self).__init__()
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)

        self.delta_attn = nn.Linear(d_model, 1)
        self.sigma_attn = nn.Linear(d_model, 1)
        self.theta_attn = nn.Linear(d_model, 1)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(d_model)

    def forward(self, query, key, value):
        Q = self.query_proj(query)
        K = self.key_proj(key)
        V = self.value_proj(value)

        e = torch.tanh(Q.unsqueeze(2) + K.unsqueeze(1))
        base_scores = e.sum(dim=-1) / self.scale
        delta_scores = self.delta_attn(e).squeeze(-1)
        sigma_scores = self.sigma_attn(e).squeeze(-1)
        theta_scores = self.theta_attn(e).squeeze(-1)
        combined_scores = base_scores + delta_scores + sigma_scores + theta_scores

        attn = torch.softmax(combined_scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V)
        return out


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerOutput(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerOutput, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, afr_reduced_cnn_size, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer_output = clones(SublayerOutput(size, dropout), 2)
        self.size = size

    def forward(self, x):
        x = self.sublayer_output[0](x, lambda x: self.self_attn(x, x, x))
        return self.sublayer_output[1](x, self.feed_forward)


class TCE(nn.Module):
    def __init__(self, layer, N):
        super(TCE, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class InceptionBlock(nn.Module):
    def __init__(self, in_channels):
        super(InceptionBlock, self).__init__()
        self.branch = InceptionDWConv1d(in_channels, 128)

    def forward(self, x):
        return self.branch(x)


class BioSleepX(nn.Module):
    def __init__(self, use_msea=False, use_gabor=False):
        super(BioSleepX, self).__init__()
        N = 2
        d_model = 80
        d_ff = 120
        dropout = 0.1
        num_classes = 5
        afr_reduced_cnn_size = 30

        self.d_model = d_model

        self.eeg_incep = InceptionBlock(1)
        self.eog_incep = InceptionBlock(1)

        self.fusion_layer = nn.Sequential(
            nn.Conv1d(256, afr_reduced_cnn_size, kernel_size=1),
            nn.BatchNorm1d(afr_reduced_cnn_size),
            nn.ReLU(),
            MEA(channels=afr_reduced_cnn_size, factor=8)
        )

        self.dim_adjust_tce = nn.Sequential(
            nn.Conv1d(afr_reduced_cnn_size, d_model, kernel_size=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU()
        )

        attn = EfficientAdditiveAttention(d_model, afr_reduced_cnn_size, dropout)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.tce = TCE(EncoderLayer(d_model, deepcopy(attn), deepcopy(ff), afr_reduced_cnn_size, dropout), N)

        self.dynamic_weight = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x_eeg, x_eog):
        if self.training:
            x_eeg = augment_signal(x_eeg)
            x_eog = augment_signal(x_eog)

        x_eeg = normalize_signal(x_eeg)
        x_eog = normalize_signal(x_eog)
        eeg_features = self.eeg_incep(x_eeg)
        eog_features = self.eog_incep(x_eog)

        fused = torch.cat([eeg_features, eog_features], dim=1)
        fused = self.fusion_layer(fused)
        fused = self.dim_adjust_tce(fused)

        if fused.size(-1) != 80:
            fused = F.adaptive_avg_pool1d(fused, 80)

        fused = fused.transpose(1, 2)
        encoded = self.tce(fused)

        weights = self.dynamic_weight(encoded.mean(dim=1))
        output = self.classifier(encoded.mean(dim=1))
        return output