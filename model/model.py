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


# 新增：细粒度门控机制模块
class FineGrainedGatingModule(nn.Module):
    def __init__(self, channels, reduction_ratio=4):
        super(FineGrainedGatingModule, self).__init__()
        self.channels = channels
        
        # 特征压缩
        self.squeeze = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(channels // reduction_ratio, channels)
        )
        
        # 空间门控
        self.spatial_gate = nn.Sequential(
            nn.Conv1d(channels, channels // reduction_ratio, kernel_size=1),
            nn.BatchNorm1d(channels // reduction_ratio),
            nn.ReLU(),
            nn.Conv1d(channels // reduction_ratio, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 通道门控
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, channels // reduction_ratio, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(channels // reduction_ratio, channels, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # 输入x形状: [B, C, T]
        batch_size, channels, seq_len = x.size()
        
        # 通道门控
        channel_att = self.channel_gate(x)
        
        # 空间门控
        spatial_att = self.spatial_gate(x)
        
        # 动态特征压缩
        x_avg = torch.mean(x, dim=2)  # [B, C]
        feature_weights = self.squeeze(x_avg).view(batch_size, channels, 1)  # [B, C, 1]
        feature_weights = torch.sigmoid(feature_weights)
        
        # 应用门控
        gated_output = x * channel_att * spatial_att * feature_weights
        
        return gated_output


# 新增：记忆增强模块
class MemoryEnhancedModule(nn.Module):
    def __init__(self, embed_dim, memory_size=64, topk=3):
        super(MemoryEnhancedModule, self).__init__()
        self.embed_dim = embed_dim
        self.memory_size = memory_size
        self.topk = topk
        
        # 记忆矩阵
        self.memory = nn.Parameter(torch.randn(memory_size, embed_dim))
        nn.init.kaiming_uniform_(self.memory)
        
        # 记忆读取层
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        
        # 特征融合层
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU()
        )
        
    def forward(self, x):
        # x形状: [B, S, D] (batch, seq_len, dim)
        batch_size, seq_len, dim = x.size()
        
        # 生成查询向量
        query = self.query_proj(x)  # [B, S, D]
        
        # 计算查询与记忆的相似度
        query = query.view(batch_size * seq_len, self.embed_dim)  # [B*S, D]
        sim = torch.matmul(query, self.memory.transpose(0, 1))  # [B*S, M]
        
        # 选择topk最相似的记忆
        topk_sim, topk_idx = torch.topk(sim, k=self.topk, dim=1)  # [B*S, k]
        topk_sim_weights = F.softmax(topk_sim, dim=1).unsqueeze(2)  # [B*S, k, 1]
        
        # 检索记忆
        retrieved_memory = self.memory[topk_idx]  # [B*S, k, D]
        
        # 加权求和
        memory_output = torch.sum(retrieved_memory * topk_sim_weights, dim=1)  # [B*S, D]
        memory_output = memory_output.view(batch_size, seq_len, self.embed_dim)  # [B, S, D]
        
        # 融合原始特征和记忆特征
        fused_output = self.fusion(torch.cat([x, memory_output], dim=2))  # [B, S, D]
        
        return fused_output


# 新增：跨Epoch注意力模块
class CrossEpochAttention(nn.Module):
    def __init__(self, d_model, num_heads=4, dropout=0.1):
        super(CrossEpochAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert self.head_dim * num_heads == d_model, "d_model必须能被num_heads整除"
        
        # 多头注意力
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # 位置偏置 - 用于对相对位置有偏好的注意力
        self.seq_length = 10  # 默认最大序列长度
        self.pos_bias = nn.Parameter(torch.zeros(2 * self.seq_length - 1))
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
    def get_rel_pos_bias(self, seq_len):
        # 根据相对位置生成偏置
        range_vec = torch.arange(seq_len)
        rel_pos_idx = range_vec[None, :] - range_vec[:, None] + seq_len - 1
        return self.pos_bias[rel_pos_idx]
    
    def forward(self, x, seq_length=5):
        # x形状: [B, S, D]
        batch_size, seq_len, dim = x.size()
        
        # 多头注意力投影
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, S, D/H]
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, S, D/H]
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, S, D/H]
        
        # 计算注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, S, S]
        
        # 添加相对位置偏置
        if seq_len <= self.seq_length:  # 确保序列长度不超过预设长度
            rel_pos_bias = self.get_rel_pos_bias(seq_len)
            attn_scores = attn_scores + rel_pos_bias.unsqueeze(0).unsqueeze(0)
        
        # 获取注意力权重
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力权重
        output = torch.matmul(attn_weights, v)  # [B, H, S, D/H]
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, dim)  # [B, S, D]
        
        # 输出投影
        output = self.out_proj(output)
        
        return output


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


class BioSleepXSeq(nn.Module):
    def __init__(self, seq_length=5, use_msea=False, use_gabor=False):
        super(BioSleepXSeq, self).__init__()
        N = 2
        seq_N = 1  # 序列TCE的层数（减少到1层以控制参数量）
        d_model = 80
        d_ff = 120
        dropout = 0.1
        num_classes = 5
        afr_reduced_cnn_size = 30
        
        self.d_model = d_model
        self.seq_length = seq_length

        # 原始BioSleepX特征提取部分
        self.eeg_incep = InceptionBlock(1)
        self.eog_incep = InceptionBlock(1)
        
        self.fusion_layer = nn.Sequential(
            nn.Conv1d(256, afr_reduced_cnn_size, kernel_size=1),
            nn.BatchNorm1d(afr_reduced_cnn_size),
            nn.ReLU(),
            MEA(channels=afr_reduced_cnn_size, factor=8)
        )
        
        # 新增：细粒度门控，用于特征增强
        self.fine_grained_gating = FineGrainedGatingModule(afr_reduced_cnn_size)
        
        self.dim_adjust_tce = nn.Sequential(
            nn.Conv1d(afr_reduced_cnn_size, d_model, kernel_size=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU()
        )
        
        # 单epoch TCE编码器
        attn = EfficientAdditiveAttention(d_model, afr_reduced_cnn_size, dropout)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.tce = TCE(EncoderLayer(d_model, deepcopy(attn), deepcopy(ff), afr_reduced_cnn_size, dropout), N)
        
        # 新增：记忆增强模块
        self.memory_module = MemoryEnhancedModule(d_model)
        
        # 新增：跨Epoch注意力模块
        self.cross_epoch_attn = CrossEpochAttention(d_model)
        
        # 序列级TCE编码器 - 使用1层TCE以减少参数量
        seq_attn = EfficientAdditiveAttention(d_model, d_model, dropout)
        seq_ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.seq_tce = TCE(EncoderLayer(d_model, deepcopy(seq_attn), deepcopy(seq_ff), d_model, dropout), seq_N)
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, num_classes)
        )
        
    def forward(self, x_eeg_seq, x_eog_seq):
        """
        参数:
            x_eeg_seq: 序列EEG数据 [batch_size, seq_len, channels, time]
            x_eog_seq: 序列EOG数据 [batch_size, seq_len, channels, time]
        """
        batch_size, seq_len = x_eeg_seq.size(0), x_eeg_seq.size(1)
        
        # 重塑输入以批量处理
        x_eeg_flat = x_eeg_seq.reshape(-1, x_eeg_seq.size(2), x_eeg_seq.size(3))
        x_eog_flat = x_eog_seq.reshape(-1, x_eog_seq.size(2), x_eog_seq.size(3))
        
        # 数据增强和标准化
        if self.training:
            # x_eeg_flat = augment_signal(x_eeg_flat)
            # x_eog_flat = augment_signal(x_eog_flat)
            pass
            
        x_eeg_flat = normalize_signal(x_eeg_flat)
        x_eog_flat = normalize_signal(x_eog_flat)
        
        # 特征提取
        eeg_features = self.eeg_incep(x_eeg_flat)
        eog_features = self.eog_incep(x_eog_flat)
        
        fused = torch.cat([eeg_features, eog_features], dim=1)
        fused = self.fusion_layer(fused)
        
        # 应用细粒度门控
        fused = self.fine_grained_gating(fused)
        
        fused = self.dim_adjust_tce(fused)
        
        if fused.size(-1) != 80:
            fused = F.adaptive_avg_pool1d(fused, 80)
            
        fused = fused.transpose(1, 2)  # [B*S, T, C]
        encoded = self.tce(fused)  # [B*S, T, d_model]
        
        # 获取每个epoch的特征表示
        epoch_features = encoded.mean(dim=1)  # [B*S, d_model]
        
        # 重塑回序列形式
        seq_features = epoch_features.view(batch_size, seq_len, self.d_model)  # [B, S, d_model]
        
        # 应用记忆增强
        memory_enhanced = self.memory_module(seq_features)
        
        # 应用跨Epoch注意力
        cross_enhanced = self.cross_epoch_attn(memory_enhanced, seq_length=seq_len)
        
        # 序列编码 - 使用TCE进行最终的序列编码
        enhanced_seq = self.seq_tce(cross_enhanced)  # [B, S, d_model]
        
        # 分类
        logits = self.classifier(enhanced_seq)  # [B, S, num_classes]
        
        return logits