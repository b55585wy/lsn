import torch
import torch.nn as nn
import torch.nn.functional as F

class InceptionDWConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, padding_mode='zeros',
                 branch_ratios=None, kernel_sizes=None):
        super(InceptionDWConv1d, self).__init__()
        
        if branch_ratios is None:
            branch_ratios = [0.5, 0.35, 0.15]
        if kernel_sizes is None:
            kernel_sizes = [7, 5, 3]

        if len(branch_ratios) != len(kernel_sizes):
            raise ValueError("branch_ratios and kernel_sizes must have the same length.")
        
        self.num_branches = len(branch_ratios)
        self.branches = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # 计算每个分支的通道数
        branch_channels = [int(out_channels * r) for r in branch_ratios]
        # 确保总通道数正确
        branch_channels[-1] += out_channels - sum(branch_channels)

        for i in range(self.num_branches):
            c = branch_channels[i]
            k = kernel_sizes[i]
            padding = k // 2 # For 'same' padding

            branch_conv = nn.Conv1d(in_channels, c, kernel_size=1)
            branch_dwconv = nn.Conv1d(c, c, kernel_size=k, padding=padding, groups=c, padding_mode=padding_mode)
            
            self.branches.append(nn.Sequential(
                branch_conv,
                branch_dwconv
            ))
            self.bns.append(nn.BatchNorm1d(c))
        
        # 输出投影
        self.project = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        
        # 添加批归一化和激活函数
        self.bn_proj = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.bn_proj = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        # 保存输入维度
        batch_size, _, seq_len = x.size()
        
        branch_outputs = []
        for i in range(self.num_branches):
            branch_out = self.branches[i](x)
            branch_out = self.bns[i](branch_out)
            branch_out = self.relu(branch_out)
            branch_outputs.append(F.adaptive_pad1d(branch_out, seq_len))
        
        # 合并所有分支
        out = torch.cat(branch_outputs, dim=1)
        
        # 投影到输出通道
        out = self.project(out)
        out = self.bn_proj(out)
        out = self.relu(out)
        
        # 确保输出维度与输入维度匹配
        out = F.adaptive_pad1d(out, seq_len)
        
        return out

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
F.adaptive_pad1d = adaptive_pad1d 

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class InceptionDWConv1d(nn.Module):
#     def __init__(self, in_channels, padding_mode='zeros',
#                  square_kernel_size=3, band_kernel_size=9, branch_ratio=0.125):
#         super(InceptionDWConv1d, self).__init__()

#         # 计算每个卷积分支通道数
#         gc = int(in_channels * branch_ratio)
#         self.split_channels = (in_channels - 3 * gc, gc, gc, gc)

#         # 分支1：Identity（保留原始特征）
#         self.identity = nn.Identity()

#         # 分支2：square kernel depthwise conv (local feature)
#         self.dwconv_square = nn.Sequential(
#             nn.Conv1d(gc, gc, kernel_size=square_kernel_size, padding=square_kernel_size // 2, groups=gc, padding_mode=padding_mode),
#             nn.BatchNorm1d(gc),
#             nn.ReLU(inplace=True)
#         )

#         # 分支3：long-range left band kernel
#         self.dwconv_band_l = nn.Sequential(
#             nn.Conv1d(gc, gc, kernel_size=band_kernel_size, padding=band_kernel_size // 2, groups=gc, padding_mode=padding_mode),
#             nn.BatchNorm1d(gc),
#             nn.ReLU(inplace=True)
#         )

#         # 分支4：long-range right band kernel
#         self.dwconv_band_r = nn.Sequential(
#             nn.Conv1d(gc, gc, kernel_size=band_kernel_size, padding=band_kernel_size // 2, groups=gc, padding_mode=padding_mode),
#             nn.BatchNorm1d(gc),
#             nn.ReLU(inplace=True)
#         )

#         # 通道融合（projection）
#         self.project = nn.Conv1d(in_channels, in_channels, kernel_size=1)
#         self.bn_proj = nn.BatchNorm1d(in_channels)
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x):
#         # 通道拆分
#         x_id, x_s, x_l, x_r = torch.split(x, self.split_channels, dim=1)

#         # 多分支并行
#         out_id = self.identity(x_id)
#         out_s  = self.dwconv_square(x_s)
#         out_l  = self.dwconv_band_l(x_l)
#         out_r  = self.dwconv_band_r(x_r)

#         # 拼接
#         out = torch.cat([out_id, out_s, out_l, out_r], dim=1)

#         # 融合 & 激活
#         out = self.project(out)
#         out = self.bn_proj(out)
#         out = self.relu(out)

#         return out


class MultiBranchStandardCNN(nn.Module):
    """
    A multi-branch CNN using standard convolutions, mirroring the structure of 
    InceptionDWConv1d for a fair ablation study comparison.
    """
    def __init__(self, in_channels, out_channels, padding_mode='zeros',
                 branch_ratios=None, kernel_sizes=None):
        super(MultiBranchStandardCNN, self).__init__()
        
        if branch_ratios is None:
            branch_ratios = [0.5, 0.35, 0.15]
        if kernel_sizes is None:
            kernel_sizes = [7, 5, 3]

        if len(branch_ratios) != len(kernel_sizes):
            raise ValueError("branch_ratios and kernel_sizes must have the same length.")
        
        self.num_branches = len(branch_ratios)
        self.branches = nn.ModuleList()
        
        # Calculate channels for each branch
        branch_channels = [int(out_channels * r) for r in branch_ratios]
        branch_channels[-1] += out_channels - sum(branch_channels)

        for i in range(self.num_branches):
            c = branch_channels[i]
            k = kernel_sizes[i]
            padding = k // 2

            # Use a standard nn.Conv1d for each branch
            branch_conv = nn.Sequential(
                nn.Conv1d(in_channels, c, kernel_size=k, padding=padding, padding_mode=padding_mode),
                nn.BatchNorm1d(c),
                nn.ReLU(inplace=True)
            )
            self.branches.append(branch_conv)
        
        # Output projection
        self.project = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        self.bn_proj = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        batch_size, _, seq_len = x.size()
        
        branch_outputs = []
        for branch in self.branches:
            branch_out = branch(x)
            # Use adaptive_pad1d from this file if needed, assuming it's available
            branch_outputs.append(F.adaptive_pad1d(branch_out, seq_len))
        
        out = torch.cat(branch_outputs, dim=1)
        
        out = self.project(out)
        out = self.bn_proj(out)
        out = self.relu(out)
        
        out = F.adaptive_pad1d(out, seq_len)
        
        return out
