import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from copy import deepcopy
import einops
from model.mea import MEA
import numpy as np


def normalize_signal(x):
    return (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-6)


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


# 从AttnSleep中导入的SE层
class SELayer(nn.Module):
    """Squeeze-and-Excitation注意力块"""
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


# 从AttnSleep中导入的多尺度残差卷积块(MRCNN)
class MultiResidualConvBlock(nn.Module):
    def __init__(self, in_channels=1, num_classes=5):
        super(MultiResidualConvBlock, self).__init__()
        # 路径1: 短时特征
        self.path1 = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=50, stride=6, bias=False, padding=24),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=8, stride=2, padding=4),
            nn.Conv1d(64, 128, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        
        # 路径2: 长时特征
        self.path2 = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=400, stride=50, bias=False, padding=200),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=2, padding=2),
            nn.Conv1d(64, 128, kernel_size=6, stride=1, bias=False, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        
        # SE注意力块
        self.se = SELayer(256, reduction=8)
    
    def forward(self, x):
        x1 = self.path1(x)
        x2 = self.path2(x)
        
        # 确保两个路径的输出大小一致
        min_len = min(x1.size(-1), x2.size(-1))
        x1 = x1[..., :min_len]
        x2 = x2[..., :min_len]
        
        # 合并输出并应用SE注意力
        x_concat = torch.cat([x1, x2], dim=1)
        x_concat = self.se(x_concat)
        
        return x_concat


# 以下是BioSleepX中的编码器相关代码
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


# 新的模型类，使用MRCNN特征提取器和TCE编码器
class BioSleepXWithMRCNN(nn.Module):
    def __init__(self, use_msea=False, use_gabor=False, use_eog=True, use_augmentation=True):
        super(BioSleepXWithMRCNN, self).__init__()
        N = 2  # TCE层数
        d_model = 80  # 特征维度
        d_ff = 120  # 前馈网络维度
        dropout = 0.1
        num_classes = 5
        afr_reduced_cnn_size = 30

        self.d_model = d_model
        self.use_eog = use_eog
        self.use_augmentation = use_augmentation

        # 使用MRCNN作为特征提取器（从AttnSleep导入）
        self.eeg_feature_extractor = MultiResidualConvBlock(in_channels=1)

        if self.use_eog:
            self.eog_feature_extractor = MultiResidualConvBlock(in_channels=1)
            # MRCNN输出通道为256，双通道则为512
            fusion_input_channels = 512  # EEG (256) + EOG (256)
        else:
            self.eog_feature_extractor = None
            fusion_input_channels = 256  # 仅 EEG (256)

        # 特征融合层（与BioSleepX类似，但调整输入通道数）
        self.fusion_layer = nn.Sequential(
            nn.Conv1d(fusion_input_channels, afr_reduced_cnn_size, kernel_size=1),
            nn.BatchNorm1d(afr_reduced_cnn_size),
            nn.ReLU(),
            MEA(channels=afr_reduced_cnn_size, factor=8)  # 使用MEA而非SE层，保持与BioSleepX一致
        )

        # 维度调整层，用于TCE
        self.dim_adjust_tce = nn.Sequential(
            nn.Conv1d(afr_reduced_cnn_size, d_model, kernel_size=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU()
        )

        # 使用TCE作为编码器（与BioSleepX一致）
        attn = EfficientAdditiveAttention(d_model, afr_reduced_cnn_size, dropout)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.tce = TCE(EncoderLayer(d_model, deepcopy(attn), deepcopy(ff), afr_reduced_cnn_size, dropout), N)

        # 分类器（与BioSleepX一致）
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x_eeg, x_eog=None):
        # 数据增强和归一化
        if self.training and self.use_augmentation:
            x_eeg = augment_signal(x_eeg)
            if self.use_eog and x_eog is not None:
                x_eog = augment_signal(x_eog)

        x_eeg = normalize_signal(x_eeg)
        eeg_features = self.eeg_feature_extractor(x_eeg)  # 使用MRCNN提取EEG特征

        # EOG处理（如果使用）
        if self.use_eog and self.eog_feature_extractor is not None and x_eog is not None:
            x_eog = normalize_signal(x_eog)
            eog_features = self.eog_feature_extractor(x_eog)  # 使用MRCNN提取EOG特征
            
            # 确保特征维度一致
            min_len = min(eeg_features.size(-1), eog_features.size(-1))
            eeg_features = eeg_features[..., :min_len]
            eog_features = eog_features[..., :min_len]
            
            # 合并EEG和EOG特征
            fused = torch.cat([eeg_features, eog_features], dim=1)
        else:
            fused = eeg_features

        # 特征融合
        fused = self.fusion_layer(fused)
        
        # 维度调整
        fused = self.dim_adjust_tce(fused)

        # 确保时间维度匹配
        if fused.size(-1) != self.d_model:
            fused = F.adaptive_avg_pool1d(fused, self.d_model)
            
        # 准备TCE输入
        fused = fused.transpose(1, 2)  # [B, C, T] -> [B, T, C]
        
        # TCE编码
        encoded = self.tce(fused)
        
        # 全局特征提取
        output_features = encoded.mean(dim=1)
        
        # 分类
        output = self.classifier(output_features)
        
        return output 