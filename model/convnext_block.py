import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None] * x + self.bias[:, None]
            return x

class ConvNeXtBlock1D(nn.Module):
    """ 1D ConvNeXt Block.
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, kernel_size=7, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        padding = kernel_size // 2
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=padding, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = nn.Identity() # Placeholder for DropPath, if needed

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 1) # (N, C, L) -> (N, L, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 2, 1) # (N, L, C) -> (N, C, L)

        if self.gamma is not None:
            x = self.gamma.view(-1, 1) * x

        x = input + self.drop_path(x)
        return x

class MultiBranchConvNeXtBlock1D(nn.Module):
    """ 1D ConvNeXt Block with multiple branches for multi-scale feature extraction,
        driven by a configuration list.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of final output channels after fusion.
        branch_configs (list): A list of dictionaries, where each dictionary
                               defines a branch.
                               Example: [{'kernel_size': 3, 'out_channels': 32},
                                         {'kernel_size': 5, 'out_channels': 32}]
    """
    def __init__(self, in_channels, out_channels, branch_configs):
        super().__init__()
        self.branches = nn.ModuleList()
        total_branch_out_channels = 0

        for config in branch_configs:
            kernel_size = config['kernel_size']
            branch_out_channels = config['out_channels']
            
            # Each branch consists of a projection to its specific channel size,
            # followed by a ConvNeXt block with a specific kernel size.
            branch = nn.Sequential(
                nn.Conv1d(in_channels, branch_out_channels, kernel_size=1),
                ConvNeXtBlock1D(dim=branch_out_channels, kernel_size=kernel_size)
            )
            self.branches.append(branch)
            total_branch_out_channels += branch_out_channels
            
        # Fusion layer to combine the outputs of all branches
        self.fusion = nn.Conv1d(total_branch_out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        branch_outputs = [branch(x) for branch in self.branches]
        concatenated = torch.cat(branch_outputs, dim=1)
        fused = self.fusion(concatenated)
        return fused