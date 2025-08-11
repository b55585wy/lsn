import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss实现，适用于睡眠阶段分类。
    
    Args:
        alpha (list): 每个类别的权重, 默认使用[1.0, 1.8, 1.5, 1.8, 1.2]，对应W, N1, N2, N3, REM
        gamma (float): 焦点损失的gamma参数，用于调整对难分类样本的关注程度
        reduction (str): 'mean', 'sum' 或 'none'，损失函数的归约方式
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        
        # 如果未提供alpha，使用默认的睡眠阶段权重
        if alpha is None:
            self.alpha = torch.tensor([1.0, 1.8, 1.5, 1.8, 1.2])
        else:
            self.alpha = torch.tensor(alpha)
    
    def forward(self, inputs, targets):
        """
        计算Focal Loss
        
        Args:
            inputs (torch.Tensor): 模型的预测输出，形状为 [batch_size, num_classes]
            targets (torch.Tensor): 真实标签，形状为 [batch_size]
            
        Returns:
            torch.Tensor: 计算的损失值
        """
        # 确保alpha和输入在同一设备上
        device = inputs.device
        alpha = self.alpha.to(device)
        
        # 将alpha扩展为与targets相同的形状
        alpha_t = alpha[targets]
        
        # 计算交叉熵
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # 计算概率和焦点权重
        probs = F.softmax(inputs, dim=1)
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # 焦点损失公式的实现: -alpha_t * (1 - p_t) ** gamma * log(p_t)
        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = alpha_t * focal_weight * ce_loss
        
        # 应用归约
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss

class ImprovedFocalLoss(nn.Module):
    """
    改进版Focal Loss，针对N1和REM睡眠阶段添加额外的权重增强。
    
    Args:
        alpha (list): 每个类别的权重
        gamma (float): 焦点损失的gamma参数
        n1_boost (float): N1类别的额外增强权重
        rem_boost (float): REM类别的额外增强权重
        reduction (str): 损失函数的归约方式
    """
    def __init__(self, alpha=None, gamma=2.0, n1_boost=0.5, rem_boost=0.3, reduction='mean'):
        super(ImprovedFocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.n1_boost = n1_boost
        self.rem_boost = rem_boost
        
        # 如果未提供alpha，使用默认的睡眠阶段权重
        if alpha is None:
            self.alpha = torch.tensor([1.0, 1.8, 1.5, 1.8, 1.2])
        else:
            self.alpha = torch.tensor(alpha)
    
    def forward(self, inputs, targets, epoch=None):
        """
        计算改进版Focal Loss
        
        Args:
            inputs (torch.Tensor): 模型的预测输出
            targets (torch.Tensor): 真实标签
            epoch (int, optional): 当前训练轮次，用于动态调整权重
            
        Returns:
            torch.Tensor: 计算的损失值
        """
        # 确保alpha在正确的设备上
        device = inputs.device
        alpha = self.alpha.to(device)
        
        # 可选的动态权重调整
        if epoch is not None:
            # 随着训练进行逐渐增加N1和REM的权重
            progress = min(epoch / 30.0, 1.0)  # 30轮后达到最大值
            dynamic_n1_boost = self.n1_boost * (1.0 + progress)
            dynamic_rem_boost = self.rem_boost * (1.0 + progress)
        else:
            dynamic_n1_boost = self.n1_boost
            dynamic_rem_boost = self.rem_boost
        
        # 将alpha应用到对应的样本
        alpha_t = alpha[targets]
        
        # 计算交叉熵损失
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # 计算概率和焦点权重
        probs = F.softmax(inputs, dim=1)
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - p_t) ** self.gamma
        
        # 对N1和REM样本应用额外的权重
        is_n1 = (targets == 1).float()
        is_rem = (targets == 4).float()
        
        # 计算N1和REM增强
        stage_boost = 1.0 + is_n1 * dynamic_n1_boost + is_rem * dynamic_rem_boost
        
        # 最终的损失值
        loss = alpha_t * focal_weight * ce_loss * stage_boost
        
        # 应用归约
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss 