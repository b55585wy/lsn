import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class GaborConv1d(nn.Module):
    """
    专门设计用于睡眠EEG/EOG信号处理的Gabor卷积层
    
    特点：
    1. 针对不同睡眠阶段的特征波形优化的参数初始化
    2. 可解释性强，滤波器参数有明确的物理意义
    3. 适合提取N1、N2、N3中的特征波形
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
                 dilation=1, groups=1, bias=True, num_kernels=8):
        super(GaborConv1d, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.num_kernels = num_kernels
        
        # 针对睡眠特征波形优化的参数初始化
        # delta波(0.5-4Hz)、theta波(4-8Hz)、alpha波(8-13Hz)、spindle(12-14Hz)
        typical_freqs = torch.tensor([0.5, 4.0, 8.0, 12.0]) / 100.0  # 归一化频率
        
        # 初始化Gabor滤波器参数
        self.mean = nn.Parameter(torch.randn(num_kernels, kernel_size) * 0.1)
        self.std = nn.Parameter(torch.rand(num_kernels, kernel_size) * 2.0 + 1.0)
        # 使用典型睡眠频率初始化
        freq_init = torch.zeros(num_kernels, kernel_size)
        for i in range(min(num_kernels, len(typical_freqs))):
            freq_init[i].fill_(typical_freqs[i])
        self.freq = nn.Parameter(freq_init)
        self.phase = nn.Parameter(torch.rand(num_kernels) * 2 * math.pi)
        
        # 将Gabor核心映射到输出通道
        self.weight_mapper = nn.Conv1d(num_kernels, out_channels, 1, groups=1, bias=False)
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
            
    def get_gabor_kernel(self):
        """
        生成Gabor滤波器核心
        每个核心专门用于提取特定频率范围的睡眠特征波形
        """
        # 生成时间轴
        t = torch.linspace(-self.kernel_size//2, self.kernel_size//2, self.kernel_size)
        t = t.view(1, -1).expand(self.num_kernels, -1)
        
        # 计算高斯包络
        gaussian = torch.exp(-0.5 * ((t - self.mean) / self.std) ** 2)
        
        # 计算正弦波
        sinusoid = torch.sin(2 * math.pi * self.freq * t + self.phase.view(-1, 1))
        
        # 组合得到Gabor滤波器
        kernel = gaussian * sinusoid
        
        # 归一化
        kernel = F.normalize(kernel, p=2, dim=1)
        
        # 扩展维度以适应conv1d的要求
        kernel = kernel.view(self.num_kernels, 1, self.kernel_size)
        kernel = kernel.repeat(1, self.in_channels // self.groups, 1)
        
        # 映射到输出通道
        kernel = self.weight_mapper(kernel)
        return kernel
        
    def forward(self, x):
        """
        前向传播，应用Gabor滤波器提取睡眠特征波形
        """
        kernel = self.get_gabor_kernel()
        return F.conv1d(x, kernel, self.bias, self.stride, 
                       self.padding, self.dilation, self.groups) 