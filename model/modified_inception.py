import torch
import torch.nn as nn
import torch.nn.functional as F

class ModifiedInceptionDWConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, padding_mode='zeros'):
        super(ModifiedInceptionDWConv1d, self).__init__()
        # 修改分支的通道比例
        self.branch_ratio = [0.4, 0.4, 0.2]  # 修改后的分支通道比例
        c1, c2, c3 = [int(out_channels * r) for r in self.branch_ratio]
        c3 = out_channels - (c1 + c2)  # 确保总通道数正确
        
        # 分支1：修改卷积核大小 (kernel_size=9 代替 7)
        self.branch1_conv = nn.Conv1d(in_channels, c1, kernel_size=1)
        self.branch1_dwconv = nn.Conv1d(c1, c1, kernel_size=9, padding=4, groups=c1, padding_mode=padding_mode)
        
        # 分支2：中等卷积核 (kernel_size=5)
        self.branch2_conv = nn.Conv1d(in_channels, c2, kernel_size=1)
        self.branch2_dwconv = nn.Conv1d(c2, c2, kernel_size=5, padding=2, groups=c2, padding_mode=padding_mode)
        
        # 分支3：小卷积核 (kernel_size=3)
        self.branch3_conv = nn.Conv1d(in_channels, c3, kernel_size=1)
        self.branch3_dwconv = nn.Conv1d(c3, c3, kernel_size=3, padding=1, groups=c3, padding_mode=padding_mode)
        
        # 添加注意力机制到投影层前
        self.channel_attention = ChannelAttention(out_channels)
        
        # 输出投影
        self.project = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        
        # 添加批归一化和激活函数
        self.bn1 = nn.BatchNorm1d(c1)
        self.bn2 = nn.BatchNorm1d(c2)
        self.bn3 = nn.BatchNorm1d(c3)
        self.bn_proj = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        # 保存输入维度
        batch_size, _, seq_len = x.size()
        
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
        branch1 = F.adaptive_pad1d(branch1, seq_len)
        branch2 = F.adaptive_pad1d(branch2, seq_len)
        branch3 = F.adaptive_pad1d(branch3, seq_len)
        
        # 合并所有分支
        out = torch.cat([branch1, branch2, branch3], dim=1)
        
        # 应用通道注意力
        out = self.channel_attention(out)
        
        # 投影到输出通道
        out = self.project(out)
        out = self.bn_proj(out)
        out = self.relu(out)
        
        # 确保输出维度与输入维度匹配
        out = F.adaptive_pad1d(out, seq_len)
        
        return out


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        self.fc = nn.Sequential(
            nn.Conv1d(channels, channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv1d(channels // reduction_ratio, channels, kernel_size=1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        scale = self.sigmoid(out)
        
        return x * scale


class ModifiedInceptionBlock(nn.Module):
    def __init__(self, in_channels):
        super(ModifiedInceptionBlock, self).__init__()
        self.branch = ModifiedInceptionDWConv1d(in_channels, 128)

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

# 添加到F命名空间
if not hasattr(F, 'adaptive_pad1d'):
    F.adaptive_pad1d = adaptive_pad1d 