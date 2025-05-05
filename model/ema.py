import torch
from torch import nn
import math

class EMA1D(nn.Module):
    def __init__(self, channels, factor=8):
        super(EMA1D, self).__init__()
        # 确保factor能整除channels
        self.groups = min(factor, channels)
        while channels % self.groups != 0:
            self.groups -= 1
        
        self.channels = channels
        self.channels_per_group = channels // self.groups
        
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool1d(1)
        
        # 使用调整后的channels_per_group
        self.gn = nn.GroupNorm(self.groups, channels)
        self.conv1x1 = nn.Conv1d(self.channels_per_group, self.channels_per_group, kernel_size=1)
        self.conv3x3 = nn.Conv1d(self.channels_per_group, self.channels_per_group, kernel_size=3, padding=1)

    def forward(self, x):
        b, c, t = x.size()
        
        # 重塑为(batch * groups, channels_per_group, time)
        group_x = x.view(b, self.groups, self.channels_per_group, t)
        group_x = group_x.reshape(b * self.groups, self.channels_per_group, t)
        
        # 1x1卷积处理时间特征
        t_att = self.conv1x1(group_x)
        
        # 应用GroupNorm和注意力
        x1 = self.gn(x).reshape(b * self.groups, self.channels_per_group, t) * t_att.sigmoid()
        
        # 3x3卷积分支
        x2 = self.conv3x3(group_x)
        
        # 计算通道注意力权重
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).transpose(1, 2))
        x12 = x2.reshape(b * self.groups, self.channels_per_group, -1)
        
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).transpose(1, 2))
        x22 = x1.reshape(b * self.groups, self.channels_per_group, -1)
        
        # 融合两个分支的注意力
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, t)
        
        # 应用注意力权重并恢复原始形状
        out = group_x * weights.sigmoid()
        return out.reshape(b, c, t) 