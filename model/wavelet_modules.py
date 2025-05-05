import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward

class Down_wt1D(nn.Module):
    """
    1D版本的Haar小波下采样模块，用于时序信号处理
    基于论文：'Haar Wavelet Downsampling: A Simple but Effective Downsampling Module For Semantic Segmentation'
    适配于AttnSleep的1D时序信号处理
    """
    def __init__(self, in_ch, out_ch):
        super(Down_wt1D, self).__init__()
        # 使用1D小波变换
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        # 1D卷积层用于调整通道数
        self.conv_bn_relu = nn.Sequential(
            nn.Conv1d(in_ch * 2, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        # 对于1D信号，需要先扩展为2D进行小波变换，然后再转回1D
        # 输入x形状: [batch, channels, length]
        batch, channels, length = x.shape
        
        # 将1D信号重塑为2D信号以适应pytorch_wavelets
        # 将长度维度分成两部分，形成一个2D形状
        # 确保长度是偶数
        if length % 2 != 0:
            # 如果长度是奇数，填充一个零
            x = torch.nn.functional.pad(x, (0, 1))
            length += 1
        
        # 重塑为[batch, channels, height=1, width=length]
        x_2d = x.unsqueeze(2)
        
        # 应用小波变换
        yL, yH = self.wt(x_2d)
        
        # yL是低频部分，形状为[batch, channels, height/2, width/2]
        # yH是高频部分，是一个列表，yH[0]形状为[batch, channels, 3, height/2, width/2]
        
        # 提取低频和高频分量
        y_L = yL.squeeze(2)  # 移除高度维度
        y_H = yH[0][:, :, 0, :].squeeze(2)  # 只使用第一个高频分量
        
        # 连接低频和高频分量
        x = torch.cat([y_L, y_H], dim=1)  # 在通道维度上连接
        
        # 通过卷积调整通道数
        x = self.conv_bn_relu(x)
        
        return x

# 用于替换MRCNN中的MaxPool1d的小波下采样模块
class WaveletPool1d(nn.Module):
    """
    使用小波下采样替代MaxPool1d
    """
    def __init__(self, in_channels, out_channels=None, kernel_size=2, stride=2):
        super(WaveletPool1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels is not None else in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        
        # 使用Down_wt1D进行下采样
        self.down_wt = Down_wt1D(in_channels, self.out_channels)
    
    def forward(self, x):
        return self.down_wt(x)