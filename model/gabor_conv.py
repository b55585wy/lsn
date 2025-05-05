import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GaborConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, num_kernels=8):
        super(GaborConv1d, self).__init__()
        
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.num_kernels = num_kernels
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Gabor 滤波器参数 - 为每个输出通道和核分别创建参数
        self.mean = nn.Parameter(torch.randn(out_channels, num_kernels))
        self.std = nn.Parameter(torch.randn(out_channels, num_kernels))
        self.freq = nn.Parameter(torch.randn(out_channels, num_kernels))
        self.phase = nn.Parameter(torch.randn(out_channels, num_kernels))
        
        # 权重参数 - 调整维度以匹配输入输出通道
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, num_kernels))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x):
        batch_size = x.size(0)
        
        # 生成时间步长向量，确保在与输入相同的设备上
        device = x.device
        t = torch.linspace(-self.kernel_size//2, self.kernel_size//2, self.kernel_size, device=device)
        t = t.view(1, 1, -1)  # [1, 1, kernel_size]
        
        # 扩展参数维度以进行广播
        mean = self.mean.view(self.out_channels, self.num_kernels, 1)  # [out_channels, num_kernels, 1]
        std = torch.exp(self.std).view(self.out_channels, self.num_kernels, 1)  # [out_channels, num_kernels, 1]
        freq = self.freq.view(self.out_channels, self.num_kernels, 1)  # [out_channels, num_kernels, 1]
        phase = self.phase.view(self.out_channels, self.num_kernels, 1)  # [out_channels, num_kernels, 1]
        
        # 计算Gabor滤波器
        gaussian = torch.exp(-0.5 * ((t - mean) / std) ** 2)  # [out_channels, num_kernels, kernel_size]
        sinusoid = torch.cos(2 * math.pi * freq * t + phase)  # [out_channels, num_kernels, kernel_size]
        gabor = gaussian * sinusoid  # [out_channels, num_kernels, kernel_size]
        
        # 应用权重并合并核
        weight = self.weight.view(self.out_channels, self.in_channels // self.groups, self.num_kernels, 1)
        kernel = (weight * gabor.unsqueeze(1)).sum(dim=2)  # [out_channels, in_channels//groups, kernel_size]
        
        # 执行卷积
        return F.conv1d(x, kernel, self.bias, self.stride, self.padding, self.dilation, self.groups) 