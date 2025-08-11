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
        x_out[i] = torch.roll(x[i], shifts=shifts[i].item(), dims=-1)
    x = x_out
    return x


# 针对N1的数据增强
def n1_targeted_augmentation(x, labels=None, is_n1=False):
    # 如果提供了标签，只对N1类进行增强
    if labels is not None:
        if torch.is_tensor(labels):
            is_n1 = (labels == 1)
        else:
            is_n1 = (labels == 1)
    
    # 如果不是N1类或没有指定，直接返回
    if not is_n1:
        return x
    
    # N1特有的频率混合
    alpha_range = (8, 12)  # alpha波频率范围
    theta_range = (4, 7)   # theta波频率范围
    
    # 创建时间点
    batch_size, channels, seq_len = x.shape
    t = torch.linspace(0, 1, seq_len).to(x.device)
    t = t.view(1, 1, -1).repeat(batch_size, 1, 1)
    
    # 创建alpha和theta波
    alpha_freq = torch.FloatTensor(batch_size, 1, 1).uniform_(*alpha_range).to(x.device)
    theta_freq = torch.FloatTensor(batch_size, 1, 1).uniform_(*theta_range).to(x.device)
    
    alpha_wave = torch.sin(2 * math.pi * alpha_freq * t)
    theta_wave = torch.sin(2 * math.pi * theta_freq * t)
    
    # 创建混合波
    mixed_wave = (alpha_wave * 0.02 + theta_wave * 0.03) * x.std(dim=-1, keepdim=True)
    
    # 应用到原始信号
    x = x + mixed_wave
    
    return x


# 改进版Inception块，带有专门针对N1特征的分支
class EnhancedInceptionBlock(nn.Module):
    def __init__(self, in_channels=1, out_channels=128, branch_ratios=None, kernel_sizes=None):
        super(EnhancedInceptionBlock, self).__init__()
        # 主Inception分支
        self.main_branch = InceptionDWConv1d(in_channels, out_channels, branch_ratios=branch_ratios, kernel_sizes=kernel_sizes)
        
        # N1特征专用分支 - 捕捉N1阶段特有的低频和转变特征
        self.n1_branch = nn.Sequential(
            # 较长卷积核捕捉慢波
            nn.Conv1d(in_channels, 16, kernel_size=25, stride=1, padding=12),
            nn.BatchNorm1d(16),
            nn.PReLU(),
            # 捕捉Alpha波特征
            nn.Conv1d(16, 16, kernel_size=9, stride=1, padding=4, groups=4),
            nn.BatchNorm1d(16),
            nn.PReLU(),
            # 提取时间上下文
            nn.Conv1d(16, 32, kernel_size=15, stride=1, padding=7),
            nn.BatchNorm1d(32),
            nn.PReLU()
        )
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv1d(out_channels + 32, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.PReLU()
        )
        
        # 残差连接
        self.residual = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels)
        ) if in_channels != out_channels else nn.Identity()
        
        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(out_channels, out_channels // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(out_channels // 4, out_channels, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        identity = self.residual(x)
        
        # 主Inception特征
        main_feat = self.main_branch(x)
        
        # N1专属特征
        n1_feat = self.n1_branch(x)
        
        # 融合特征
        combined = torch.cat([main_feat, n1_feat], dim=1)
        fused = self.fusion(combined)
        
        # 应用通道注意力
        attn = self.channel_attention(fused)
        fused = fused * attn
        
        # 残差连接
        output = fused + identity
        output = F.relu(output)
        
        return output


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


# 加权损失函数，增加N1类的权重
class WeightedCELoss(nn.Module):
    def __init__(self, n1_weight=3.0):
        super(WeightedCELoss, self).__init__()
        self.n1_weight = n1_weight
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        
    def forward(self, preds, targets):
        loss = self.ce_loss(preds, targets)
        
        # 创建N1类样本的掩码
        n1_mask = (targets == 1).float()
        
        # 为N1类样本增加权重
        weighted_loss = loss * (1 + (self.n1_weight - 1) * n1_mask)
        
        return weighted_loss.mean()


# 改进版模型，加强N1识别和收敛速度
# 定义一个标准的CNN块，用于消融实验
class StandardCNNBlock(nn.Module):
    def __init__(self, in_channels=1, out_channels=128):
        super(StandardCNNBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(64, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
    def forward(self, x):
        return self.conv_block(x)


class BioSleepXWithInceptionImproved(nn.Module):
    def __init__(self, use_eog=True, use_inception=True, use_mea=True, use_mamba=True, use_augmentation=True, n1_weight=3.0,
                 n_layers=2, d_model=80, d_ff=120, dropout=0.1, num_classes=5, afr_reduced_cnn_size=30,
                 inception_params=None):
        super(BioSleepXWithInceptionImproved, self).__init__()
        
        self.d_model = d_model
        self.use_eog = use_eog
        self.use_augmentation = use_augmentation
        self.n1_weight = n1_weight

        # 根据 use_inception 标志选择特征提取器
        if use_inception:
            inception_kwargs = {
                'in_channels': 1,
                'out_channels': inception_params.get('out_channels', 128) if inception_params else 128,
                'branch_ratios': inception_params.get('branch_ratios') if inception_params else None,
                'kernel_sizes': inception_params.get('kernel_sizes') if inception_params else None
            }
            self.eeg_feature_extractor = EnhancedInceptionBlock(**inception_kwargs)
            if self.use_eog:
                self.eog_feature_extractor = EnhancedInceptionBlock(**inception_kwargs)
        else:
            self.eeg_feature_extractor = StandardCNNBlock(in_channels=1, out_channels=128)
            if self.use_eog:
                self.eog_feature_extractor = StandardCNNBlock(in_channels=1, out_channels=128)

        inception_out_channels = inception_params.get('out_channels', 128) if inception_params else 128
        if self.use_eog:
            fusion_input_channels = inception_out_channels * 2
        else:
            fusion_input_channels = inception_out_channels

        # 根据 use_mea 标志构建特征融合层
        if use_mea:
            self.fusion_layer = nn.Sequential(
                nn.Conv1d(fusion_input_channels, afr_reduced_cnn_size, kernel_size=1),
                nn.BatchNorm1d(afr_reduced_cnn_size),
                nn.ReLU(),
                MEA(channels=afr_reduced_cnn_size, factor=8)
            )
        else:
            self.fusion_layer = nn.Sequential(
                nn.Conv1d(fusion_input_channels, afr_reduced_cnn_size, kernel_size=1),
                nn.BatchNorm1d(afr_reduced_cnn_size),
                nn.ReLU()
            )

        # 维度调整层，用于TCE
        self.dim_adjust_tce = nn.Sequential(
            nn.Conv1d(afr_reduced_cnn_size, d_model, kernel_size=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU()
        )

        # 根据 use_mamba 标志选择时间编码器
        if use_mamba:
            # 使用TCE作为编码器 (Mamba-based)
            attn = EfficientAdditiveAttention(d_model, afr_reduced_cnn_size, dropout)
            ff = PositionwiseFeedForward(d_model, d_ff, dropout)
            self.temporal_encoder = TCE(EncoderLayer(d_model, deepcopy(attn), deepcopy(ff), afr_reduced_cnn_size, dropout), n_layers)
        else:
            # 使用Bi-LSTM作为编码器
            self.temporal_encoder = nn.LSTM(input_size=d_model, hidden_size=d_model // 2, num_layers=n_layers,
                                          batch_first=True, bidirectional=True, dropout=dropout)
        
        # N1特征增强层 - 专注于N1特征
        self.n1_enhancer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model),
            nn.Sigmoid()
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, num_classes)
        )

        # 初始化权重
        self._initialize_weights()
        
        # 集成的损失函数
        self.criterion = WeightedCELoss(n1_weight=n1_weight)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x_eeg, x_eog=None, targets=None):
        # 数据增强和归一化
        if self.training and self.use_augmentation:
            # 普通增强
            x_eeg = augment_signal(x_eeg)
            
            # 如果有目标标签，使用N1目标增强
            if targets is not None:
                x_eeg = n1_targeted_augmentation(x_eeg, targets)
            
            # EOG信号处理
            if self.use_eog and x_eog is not None:
                x_eog = augment_signal(x_eog)
                if targets is not None:
                    x_eog = n1_targeted_augmentation(x_eog, targets)

        # 信号归一化
        x_eeg = normalize_signal(x_eeg)
        eeg_features = self.eeg_feature_extractor(x_eeg)

        # EOG处理（如果使用）
        if self.use_eog and self.eog_feature_extractor is not None and x_eog is not None:
            x_eog = normalize_signal(x_eog)
            eog_features = self.eog_feature_extractor(x_eog)
            
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
        
        # 时间编码
        if isinstance(self.temporal_encoder, nn.LSTM):
            encoded, _ = self.temporal_encoder(fused)
        else:
            encoded = self.temporal_encoder(fused)
        
        # 如果正在训练且有目标，应用N1特征增强
        if self.training and targets is not None:
            # 找出N1样本
            n1_mask = (targets == 1).float().unsqueeze(1).unsqueeze(2)
            
            # 生成N1增强因子
            n1_factors = self.n1_enhancer(encoded)
            
            # 选择性地应用于N1样本
            enhanced = encoded * (1 + n1_mask * n1_factors)
            encoded = enhanced
        
        # 全局特征提取
        output_features = encoded.mean(dim=1)
        
        # 分类
        output = self.classifier(output_features)
        
        # 如果训练模式且有目标标签，计算带权重的损失
        if self.training and targets is not None:
            loss = self.criterion(output, targets)
            return output, loss
        
        return output 