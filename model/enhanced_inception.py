import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SELayer(nn.Module):
    """压缩激发(Squeeze-and-Excitation)注意力块"""
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

class EnhancedInceptionDWConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, padding_mode='zeros'):
        super(EnhancedInceptionDWConv1d, self).__init__()
        # 使用更长的卷积核来捕获长时间依赖
        self.branch_ratio = [0.5, 0.3, 0.2]  # 分支通道比例
        c1, c2, c3 = [int(out_channels * r) for r in self.branch_ratio]
        c3 = out_channels - (c1 + c2)  # 确保总通道数正确
        
        # 分支1：大卷积核 (kernel_size=49，接近AttnSleep中的50)
        self.branch1_conv = nn.Conv1d(in_channels, c1, kernel_size=1)
        self.branch1_dwconv = nn.Conv1d(c1, c1, kernel_size=49, padding=24, groups=c1, padding_mode=padding_mode)
        
        # 分支2：中等卷积核 (kernel_size=25)
        self.branch2_conv = nn.Conv1d(in_channels, c2, kernel_size=1)
        self.branch2_dwconv = nn.Conv1d(c2, c2, kernel_size=25, padding=12, groups=c2, padding_mode=padding_mode)
        
        # 分支3：小卷积核 (kernel_size=9)
        self.branch3_conv = nn.Conv1d(in_channels, c3, kernel_size=1)
        self.branch3_dwconv = nn.Conv1d(c3, c3, kernel_size=9, padding=4, groups=c3, padding_mode=padding_mode)
        
        # 批归一化层
        self.bn1 = nn.BatchNorm1d(c1)
        self.bn2 = nn.BatchNorm1d(c2)
        self.bn3 = nn.BatchNorm1d(c3)
        self.bn_proj = nn.BatchNorm1d(out_channels)
        
        # 激活函数
        self.relu = nn.ReLU(inplace=True)
        
        # SE注意力块
        self.se = SELayer(out_channels, reduction=8)
        
        # 输出投影
        self.project = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        
        # 残差连接（如果输入和输出通道数不同，则进行投影）
        self.use_residual = (in_channels == out_channels)
        if not self.use_residual:
            self.residual_proj = nn.Conv1d(in_channels, out_channels, kernel_size=1)
            self.residual_bn = nn.BatchNorm1d(out_channels)
        
    def forward(self, x):
        # 保存输入用于残差连接
        residual = x
        
        # 分支1
        branch1 = self.branch1_conv(x)
        branch1 = self.branch1_dwconv(branch1)
        branch1 = self.bn1(branch1)
        branch1 = self.relu(branch1)
        
        # 分支2
        branch2 = self.branch2_conv(x)
        branch2 = self.branch2_dwconv(branch2)
        branch2 = self.bn2(branch2)
        branch2 = self.relu(branch2)
        
        # 分支3
        branch3 = self.branch3_conv(x)
        branch3 = self.branch3_dwconv(branch3)
        branch3 = self.bn3(branch3)
        branch3 = self.relu(branch3)
        
        # 确保所有分支输出维度一致
        seq_len = min(branch1.size(-1), branch2.size(-1), branch3.size(-1))
        branch1 = adaptive_pad1d(branch1, seq_len)
        branch2 = adaptive_pad1d(branch2, seq_len)
        branch3 = adaptive_pad1d(branch3, seq_len)
        
        # 合并所有分支
        out = torch.cat([branch1, branch2, branch3], dim=1)
        
        # 投影并应用SE注意力
        out = self.project(out)
        out = self.bn_proj(out)
        out = self.se(out)
        
        # 应用残差连接
        if self.use_residual:
            # 确保残差和输出长度一致
            residual = adaptive_pad1d(residual, out.size(-1))
            out = out + residual
        else:
            # 当输入输出通道数不同，使用投影残差
            residual = self.residual_proj(residual)
            residual = self.residual_bn(residual)
            residual = adaptive_pad1d(residual, out.size(-1))
            out = out + residual
        
        # 最终激活
        out = self.relu(out)
        
        return out

class EnhancedInceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels=128):
        super(EnhancedInceptionBlock, self).__init__()
        self.branch = EnhancedInceptionDWConv1d(in_channels, out_channels)

    def forward(self, x):
        return self.branch(x)

def adaptive_pad1d(x, target_size):
    """确保一维序列长度与目标长度匹配"""
    diff = target_size - x.size(-1)
    if diff > 0:
        # 需要填充
        pad_left = diff // 2
        pad_right = diff - pad_left
        return F.pad(x, (pad_left, pad_right))
    elif diff < 0:
        # 需要裁剪
        start = abs(diff) // 2
        return x[..., start:start + target_size]
    return x

# 双路径特征提取器（类似MRCNN）
class DualPathFeatureExtractor(nn.Module):
    def __init__(self, out_channels=128):
        super(DualPathFeatureExtractor, self).__init__()
        self.path1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=50, stride=6, bias=False, padding=24),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=8, stride=2, padding=4),
            nn.Conv1d(64, out_channels, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.path2 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=400, stride=50, bias=False, padding=200),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=2, padding=2),
            nn.Conv1d(64, out_channels, kernel_size=6, stride=1, bias=False, padding=3),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # SE块增强特征
        self.se = SELayer(out_channels * 2, reduction=8)
    
    def forward(self, x):
        x1 = self.path1(x)
        x2 = self.path2(x)
        
        # 确保长度一致
        min_len = min(x1.size(-1), x2.size(-1))
        x1 = x1[..., :min_len]
        x2 = x2[..., :min_len]
        
        # 特征融合
        x_concat = torch.cat([x1, x2], dim=1)
        x_concat = self.se(x_concat)
        
        return x_concat 