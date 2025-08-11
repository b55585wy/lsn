import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from copy import deepcopy


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


class SELayer(nn.Module):
    """squeeze-and-excitation注意力块"""
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


class TransformerBlock(nn.Module):
    def __init__(self, input_dim, num_heads, hidden_dim, dropout_rate=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(input_dim, num_heads, dropout=dropout_rate)
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.ff = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, input_dim)
        )
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        # 注意力层
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈网络
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class AttnSleep(nn.Module):
    def __init__(self, use_eog=True, use_augmentation=True):
        super(AttnSleep, self).__init__()
        self.use_eog = use_eog
        self.use_augmentation = use_augmentation
        num_classes = 5
        
        # EEG和EOG的多尺度卷积模块
        self.eeg_feature_extractor = MultiResidualConvBlock(in_channels=1)
        if self.use_eog:
            self.eog_feature_extractor = MultiResidualConvBlock(in_channels=1)
        
        # 特征融合卷积层
        if self.use_eog:
            self.fusion_layer = nn.Sequential(
                nn.Conv1d(512, 128, kernel_size=1),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                SELayer(128, reduction=8)
            )
        else:
            self.fusion_layer = nn.Sequential(
                nn.Conv1d(256, 128, kernel_size=1),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                SELayer(128, reduction=8)
            )
        
        # 时序建模 - Transformer编码器
        self.positional_encoding = nn.Parameter(torch.zeros(1, 20, 128))
        torch.nn.init.normal_(self.positional_encoding, mean=0, std=0.02)
        
        self.transformer_encoder = nn.Sequential(
            TransformerBlock(128, num_heads=8, hidden_dim=256),
            TransformerBlock(128, num_heads=8, hidden_dim=256)
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x_eeg, x_eog=None):
        # 数据增强（如果启用）
        if self.training and self.use_augmentation:
            x_eeg = augment_signal(x_eeg)
            if self.use_eog and x_eog is not None:
                x_eog = augment_signal(x_eog)
        
        # 归一化信号
        x_eeg = normalize_signal(x_eeg)
        eeg_features = self.eeg_feature_extractor(x_eeg)
        
        # 处理EOG（如果使用）
        if self.use_eog and self.eog_feature_extractor is not None and x_eog is not None:
            x_eog = normalize_signal(x_eog)
            eog_features = self.eog_feature_extractor(x_eog)
            
            # 确保特征维度一致
            min_len = min(eeg_features.size(-1), eog_features.size(-1))
            eeg_features = eeg_features[..., :min_len]
            eog_features = eog_features[..., :min_len]
            
            # 合并EEG和EOG特征
            fused = torch.cat([eeg_features, eog_features], dim=1)  # [B, 512, T]
        else:
            fused = eeg_features  # [B, 256, T]
        
        # 融合层处理
        fused = self.fusion_layer(fused)  # [B, 128, T]
        
        # 调整特征维度为固定长度，适应Transformer输入
        fused = F.adaptive_avg_pool1d(fused, 20)  # [B, 128, 20]
        
        # 转置为Transformer输入格式 [T, B, C]
        fused = fused.permute(2, 0, 1)  # [20, B, 128]
        
        # 添加位置编码
        fused = fused + self.positional_encoding.permute(1, 0, 2)
        
        # Transformer编码
        encoded = self.transformer_encoder(fused)  # [20, B, 128]
        
        # 获取全局表示（平均池化）
        output_features = encoded.mean(dim=0)  # [B, 128]
        
        # 分类
        output = self.classifier(output_features)  # [B, 5]
        
        return output 