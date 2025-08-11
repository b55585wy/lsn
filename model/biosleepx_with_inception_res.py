import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from copy import deepcopy
import einops
from model.mea import MEA
from model.idc import InceptionDWConv1d
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
        x_out[i] = torch.roll(x[i], shifts[i].item(), dims=-1)
    return x_out


class ResidualInceptionBlock(nn.Module):
    """添加残差连接的InceptionBlock，类似于MRCNN结构"""
    def __init__(self, in_channels=1, out_channels=128, kernel_sizes=[7, 5, 3], dropout=0.1):
        super(ResidualInceptionBlock, self).__init__()
        
        # 主分支使用InceptionDWConv1d
        self.branch = InceptionDWConv1d(in_channels, out_channels)
        
        # 添加跳跃连接 (1x1卷积用于维度匹配)
        self.skip_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1)
        self.skip_bn = nn.BatchNorm1d(out_channels)
        
        # SE注意力层增强特征重要性
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(out_channels, out_channels // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels // 8, out_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 残差融合后的激活
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        
        # 添加N1特有分支
        self.n1_branch = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=9, stride=1, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, kernel_size=1, stride=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True)
        )
        
        # 融合常规特征和N1特征
        self.fusion = nn.Conv1d(out_channels + 32, out_channels, kernel_size=1, stride=1)
        self.fusion_bn = nn.BatchNorm1d(out_channels)
    
    def forward(self, x):
        # 主路径
        main = self.branch(x)
        
        # 跳跃连接
        skip = self.skip_bn(self.skip_conv(x))
        
        # 注意力增强
        att = self.se(main)
        main = main * att
        
        # 残差相加
        out = main + skip
        out = self.relu(out)
        out = self.dropout(out)
        
        # 添加N1特有分支
        n1_features = self.n1_branch(x)
        
        # 融合常规特征和N1特征
        combined = torch.cat([out, n1_features], dim=1)
        out = self.fusion(combined)
        out = self.fusion_bn(out)
        out = self.relu(out)
        
        return out


class EnhancedSublayerOutput(nn.Module):
    """增强的残差子层"""
    def __init__(self, size, dropout):
        super(EnhancedSublayerOutput, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, sublayer):
        "使用残差连接: x + dropout(sublayer(norm(x)))"
        return x + self.dropout(sublayer(self.norm(x)))


class EnhancedTCE(nn.Module):
    """带强化残差连接的时间上下文编码器"""
    def __init__(self, d_model=80, n_head=4, d_ff=120, dropout=0.1, num_layers=2):
        super(EnhancedTCE, self).__init__()
        
        from model.tce import EncoderLayer, EfficientAdditiveAttention, PositionwiseFeedForward
        
        # 创建多层编码器，使用增强的残差连接
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'self_attn': EfficientAdditiveAttention(d_model, n_head, dropout),
                'feed_forward': PositionwiseFeedForward(d_model, d_ff, dropout),
                'sublayer1': EnhancedSublayerOutput(d_model, dropout),
                'sublayer2': EnhancedSublayerOutput(d_model, dropout)
            })
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x, mask=None):
        # 遍历所有编码器层，应用注意力和前馈网络
        for layer in self.layers:
            x = layer['sublayer1'](x, lambda x: layer['self_attn'](x, x, x, mask))
            x = layer['sublayer2'](x, layer['feed_forward'])
        return self.norm(x)


class BioSleepXWithInceptionRes(nn.Module):
    """改进的BioSleepX模型，使用残差Inception块和强化的稳定性"""
    def __init__(self, use_eog=True, use_augmentation=True, n1_weight=1.8):
        super(BioSleepXWithInceptionRes, self).__init__()
        
        # 特征提取器
        self.eeg_feature_extractor = ResidualInceptionBlock(in_channels=1, out_channels=128)
        if use_eog:
            self.eog_feature_extractor = ResidualInceptionBlock(in_channels=1, out_channels=128)
        
        # 融合层
        in_chann = 256 if use_eog else 128
        self.fusion_layer = nn.Sequential(
            nn.Conv1d(in_chann, 30, kernel_size=1, stride=1),
            nn.BatchNorm1d(30),
            nn.ReLU(),
            MEA(channels=30)
        )
        
        # 维度调整
        self.dim_adjust_tce = nn.Sequential(
            nn.Conv1d(30, 80, kernel_size=1, stride=1),
            nn.BatchNorm1d(80),
            nn.ReLU()
        )
        
        # 增强的时间上下文编码器
        self.tce = EnhancedTCE(d_model=80, num_layers=2)
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(80, 40),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(40, 5)
        )
        
        # 辅助分类器（针对N1类）
        self.n1_classifier = nn.Sequential(
            nn.Linear(80, 40),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(40, 5)
        )
        
        self.use_eog = use_eog
        self.use_augmentation = use_augmentation
        self.n1_weight = n1_weight
        
    def forward(self, x, aug=False):
        # 获取EEG和EOG信号
        eeg = x[:, 0:1]
        eog = x[:, 1:2] if self.use_eog else None
        
        # 应用数据增强
        if self.use_augmentation and aug and self.training:
            eeg = augment_signal(eeg)
            if eog is not None:
                eog = augment_signal(eog)
        
        # 信号归一化
        eeg = normalize_signal(eeg)
        if eog is not None:
            eog = normalize_signal(eog)
        
        # 特征提取
        eeg_features = self.eeg_feature_extractor(eeg)
        if self.use_eog:
            eog_features = self.eog_feature_extractor(eog)
            features = torch.cat((eeg_features, eog_features), dim=1)
        else:
            features = eeg_features
        
        # 特征融合
        features = self.fusion_layer(features)
        
        # 准备TCE输入
        tce_input = self.dim_adjust_tce(features)
        tce_input = tce_input.permute(0, 2, 1)  # [B, L, D]
        
        # 时间上下文编码
        encoded = self.tce(tce_input)
        
        # 全局池化
        encoded = encoded.mean(dim=1)  # [B, D]
        
        # 主分类器
        main_output = self.classifier(encoded)
        
        # N1专用分类器
        n1_output = self.n1_classifier(encoded)
        
        # 组合输出 (使主分类器的输出为主，但增强N1类的权重)
        if self.training:
            combined_output = main_output.clone()
            combined_output[:, 1] = (main_output[:, 1] + self.n1_weight * n1_output[:, 1]) / (1 + self.n1_weight)
            return combined_output
        else:
            return main_output 