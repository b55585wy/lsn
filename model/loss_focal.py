import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def weighted_CrossEntropyLoss(output, target, classes_weights, device):
    # 确保权重是浮点类型并且在正确的设备上
    weight_tensor = torch.tensor(classes_weights, dtype=torch.float32).to(device)
    cr = nn.CrossEntropyLoss(weight=weight_tensor)
    return cr(output, target)

# 新增：CrossEntropyLoss类，修复AttributeError: module 'model.loss' has no attribute 'CrossEntropyLoss'
class CrossEntropyLoss(nn.Module):
    def __init__(self, classes_weights=None, device=None):
        super(CrossEntropyLoss, self).__init__()
        self.device = device
        self.classes_weights = classes_weights
        
        if classes_weights is not None and device is not None:
            weight_tensor = torch.tensor(classes_weights, dtype=torch.float32).to(device)
            self.criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, output, target, classes_weights=None, device=None):
        # 忽略额外传入的参数，直接使用初始化时设置的criterion
        return self.criterion(output, target)

# 新增：时序连续性损失函数，用于增强序列模型的时间依赖性捕获
class TemporalContinuityLoss(nn.Module):
    def __init__(self, lambda_smooth=0.3, lambda_trans=0.3):
        super(TemporalContinuityLoss, self).__init__()
        self.lambda_smooth = lambda_smooth  # 平滑性权重
        self.lambda_trans = lambda_trans    # 转移权重
        
        # 预定义的睡眠阶段转移矩阵 (5x5)
        # 0-W, 1-N1, 2-N2, 3-N3, 4-REM
        self.sleep_stage_transitions = torch.tensor([
            [0.8, 0.2, 0.0, 0.0, 0.0],  # W->W(高), W->N1(低), 其他极少
            [0.1, 0.4, 0.4, 0.0, 0.1],  # N1->各阶段都有可能
            [0.0, 0.1, 0.7, 0.2, 0.0],  # N2主要向N2, N3转移
            [0.0, 0.0, 0.3, 0.7, 0.0],  # N3主要向N3, N2转移
            [0.1, 0.2, 0.0, 0.0, 0.7]   # REM主要维持自身，有时向W, N1转移
        ], dtype=torch.float32)
        
    def forward(self, logits, targets=None):
        """
        参数:
            logits: 模型输出 [batch_size, seq_len, num_classes]
            targets: 可选的目标标签 [batch_size, seq_len]
        """
        batch_size, seq_len, num_classes = logits.shape
        device = logits.device
        
        # 将转移矩阵移至相同设备
        self.sleep_stage_transitions = self.sleep_stage_transitions.to(device)
        
        # 计算时间平滑损失 - 鼓励相邻时间步的预测相似
        probs = F.softmax(logits, dim=-1)
        smooth_loss = 0.0
        
        for t in range(1, seq_len):
            # 计算相邻预测的KL散度
            prev_prob = probs[:, t-1, :]
            curr_prob = probs[:, t, :]
            
            # 使用L1距离作为平滑损失
            diff = torch.abs(prev_prob - curr_prob).sum(dim=1).mean()
            smooth_loss += diff
            
        smooth_loss = smooth_loss / (seq_len - 1) if seq_len > 1 else 0.0
        
        # 基于睡眠周期转移规律的转移损失
        trans_loss = 0.0
        if seq_len > 1:
            for t in range(1, seq_len):
                prev_prob = probs[:, t-1, :]
                curr_prob = probs[:, t, :]
                
                # 计算预期的下一个阶段分布
                expected_prob = torch.matmul(prev_prob, self.sleep_stage_transitions)
                
                # 使用KL散度计算当前预测与预期预测的差异
                kl_div = F.kl_div(
                    curr_prob.log(), 
                    expected_prob, 
                    reduction='batchmean',
                    log_target=False
                )
                
                trans_loss += kl_div
                
            trans_loss = trans_loss / (seq_len - 1)
            
        # 综合损失
        total_loss = self.lambda_smooth * smooth_loss + self.lambda_trans * trans_loss
        
        return total_loss

# 新增：组合损失函数，将分类损失和时序连续性损失结合
class CombinedSequentialLoss(nn.Module):
    def __init__(self, class_weights=None, lambda_temporal=0.4, lambda_smooth=0.3, lambda_trans=0.3, device=None):
        super(CombinedSequentialLoss, self).__init__()
        self.class_weights = class_weights
        self.lambda_temporal = lambda_temporal
        self.device = device
        
        # 如果在初始化时提供了类别权重和设备
        if class_weights is not None and device is not None:
            weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
            self.cls_criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        else:
            self.cls_criterion = nn.CrossEntropyLoss()
        
        # 时序连续性损失
        self.temporal_criterion = TemporalContinuityLoss(
            lambda_smooth=lambda_smooth, 
            lambda_trans=lambda_trans
        )
        
    def forward(self, logits, targets, classes_weights=None, device=None):
        """
        参数:
            logits: 模型输出 [batch_size, seq_len, num_classes]
            targets: 目标标签 [batch_size, seq_len]
            classes_weights: 类别权重，可选
            device: 计算设备，可选
        """
        batch_size, seq_len, num_classes = logits.shape
        
        # 使用初始化时设置的criterion进行分类损失计算
        # 重塑输出和目标以适应交叉熵损失
        flat_logits = logits.reshape(-1, num_classes)
        flat_targets = targets.reshape(-1)
        
        # 计算分类损失
        cls_loss = self.cls_criterion(flat_logits, flat_targets)
        
        # 计算时序连续性损失
        temporal_loss = self.temporal_criterion(logits, targets)
        
        # 综合损失
        total_loss = cls_loss + self.lambda_temporal * temporal_loss
        
        return total_loss


class LabelSmoothingLoss(nn.Module):
    """
    标签平滑损失函数
    当训练数据存在噪声标签时有效缓解过拟合
    """
    def __init__(self, classes, smoothing=0.0, dim=-1, weight=None):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim
        self.weight = weight
        
    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # 创建平滑后的one-hot标签
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        
        if self.weight is not None:
            # 如果提供了类别权重，则应用它们
            weight = self.weight.to(pred.device)
            weighted_loss = -torch.sum(true_dist * pred, dim=self.dim)
            weighted_loss = weighted_loss * weight.gather(0, target)
            return weighted_loss.mean()
        else:
            return -torch.sum(true_dist * pred, dim=self.dim).mean()


class FocalLoss(nn.Module):
    """
    Focal Loss: 专为严重不平衡数据集设计的损失函数
    
    参数:
        alpha: 类别权重参数，用于平衡不同类别的贡献
        gamma: 聚焦参数，调整简单/困难样本的权重(越大，对容易分类的样本惩罚越小)
        label_smoothing: 标签平滑系数 (0.0-1.0)
        n1_class_idx: N1类别的索引 (通常为1)
        n1_weight_multiplier: N1类别权重倍增器
        reduction: 损失归约方式 ('mean', 'sum', 'none')
    """
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0, 
                 n1_class_idx=1, n1_weight_multiplier=1.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.n1_class_idx = n1_class_idx
        self.n1_weight_multiplier = n1_weight_multiplier
        self.reduction = reduction
        self.num_classes = len(alpha) if alpha is not None else None
        
    def forward(self, inputs, targets):
        # 获取设备信息
        device = inputs.device
        
        # 应用标签平滑
        if self.label_smoothing > 0 and self.num_classes is not None:
            # 创建平滑后的目标分布
            smooth_targets = torch.zeros_like(inputs).scatter_(
                1, targets.unsqueeze(1), 1
            )
            smooth_targets = smooth_targets * (1 - self.label_smoothing) + \
                             self.label_smoothing / self.num_classes
            
            # 计算交叉熵
            log_probs = F.log_softmax(inputs, dim=1)
            loss = -(smooth_targets * log_probs).sum(dim=1)
            
            # 计算预测概率用于focal loss的权重
            probs = F.softmax(inputs, dim=1)
            target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze()
            
            # 应用focal loss权重
            focal_weight = (1 - target_probs) ** self.gamma
            
            # 获取并应用alpha权重
            if self.alpha is not None:
                alpha = torch.tensor(self.alpha, dtype=torch.float32).to(device)
                
                # 对N1类别应用额外权重
                if self.n1_weight_multiplier > 1.0:
                    n1_mask = (targets == self.n1_class_idx)
                    alpha_weight = alpha.gather(0, targets)
                    alpha_weight[n1_mask] = alpha_weight[n1_mask] * self.n1_weight_multiplier
                else:
                    alpha_weight = alpha.gather(0, targets)
                
                # 应用两种权重
                loss = alpha_weight * focal_weight * loss
            else:
                loss = focal_weight * loss
            
        else:
            # 将输入转换为对数概率
            log_softmax = F.log_softmax(inputs, dim=1)
            
            # 获取目标类别的对数概率
            targets = targets.view(-1, 1)
            log_pt = log_softmax.gather(1, targets)
            log_pt = log_pt.view(-1)
            
            # 计算概率并获取pt
            pt = log_pt.exp()
            
            # 如果提供了类别权重，则应用它们
            if self.alpha is not None:
                # alpha应该是一个多类别权重列表或张量
                if isinstance(self.alpha, (list, tuple, np.ndarray)):
                    alpha = torch.tensor(self.alpha, dtype=torch.float32).to(device)
                else:
                    # 如果是张量，直接使用
                    alpha = self.alpha.to(device)
                
                # 根据目标类别获取alpha权重
                batch_alpha = alpha.gather(0, targets.view(-1))
                
                # 对N1类别应用额外权重
                if self.n1_weight_multiplier > 1.0:
                    n1_mask = (targets.view(-1) == self.n1_class_idx)
                    if n1_mask.any():
                        batch_alpha[n1_mask] = batch_alpha[n1_mask] * self.n1_weight_multiplier
                
                focal_weight = batch_alpha * ((1 - pt) ** self.gamma)
            else:
                focal_weight = (1 - pt) ** self.gamma
            
            # 计算focal loss
            loss = -1 * focal_weight * log_pt
        
        # 应用归约
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def focal_loss(output, target, classes_weights, device, gamma=2.0, label_smoothing=0.0, 
               n1_weight_multiplier=1.0, n1_class_idx=1):
    """
    Focal Loss的函数接口，与weighted_CrossEntropyLoss保持一致的接口
    
    参数:
        output: 模型输出 (batch_size, num_classes)
        target: 目标标签 (batch_size,)
        classes_weights: 类别权重列表
        device: 计算设备
        gamma: 聚焦参数，默认为2.0
        label_smoothing: 标签平滑系数，默认为0.0
        n1_weight_multiplier: N1类别权重倍增器，默认为1.0
        n1_class_idx: N1类别的索引，默认为1
    """
    # 获取配置参数
    num_classes = len(classes_weights)
    
    # 转换为张量并移至正确设备
    alpha = torch.tensor(classes_weights, dtype=torch.float32).to(device)
    
    criterion = FocalLoss(
        alpha=alpha, 
        gamma=gamma,
        label_smoothing=label_smoothing,
        n1_class_idx=n1_class_idx,
        n1_weight_multiplier=n1_weight_multiplier,
        reduction='mean'
    )
    
    return criterion(output, target)


class SequentialFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0, n1_weight_multiplier=1.0, 
                 n1_class_idx=1, transition_weight=0.2, num_classes=5, reduction='mean', device=None):
        super(SequentialFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.n1_weight_multiplier = n1_weight_multiplier
        self.n1_class_idx = n1_class_idx
        self.reduction = reduction
        self.num_classes = num_classes
        self.transition_weight = transition_weight
        self.device = device
        
        # 预定义的合理转移矩阵 - 基于睡眠阶段转移规律
        self.transition_matrix = self._create_transition_matrix(num_classes)
        
    def _create_transition_matrix(self, num_classes):
        """创建睡眠阶段转移概率矩阵"""
        # 基于睡眠阶段的自然转移规律
        # 0-W, 1-N1, 2-N2, 3-N3, 4-REM
        
        # 初始化零矩阵
        matrix = torch.zeros(num_classes, num_classes)
        
        # 定义合理的转移
        # W -> W, N1
        matrix[0, 0] = 0.8  # W自身转移概率高
        matrix[0, 1] = 0.2  # W->N1可能性较低
        
        # N1 -> N1, W, N2, REM
        matrix[1, 0] = 0.1  # N1->W
        matrix[1, 1] = 0.4  # N1自身转移
        matrix[1, 2] = 0.4  # N1->N2常见
        matrix[1, 4] = 0.1  # N1->REM有时发生
        
        # N2 -> N2, N1, N3
        matrix[2, 1] = 0.1  # N2->N1
        matrix[2, 2] = 0.7  # N2自身转移
        matrix[2, 3] = 0.2  # N2->N3常见
        
        # N3 -> N3, N2
        matrix[3, 2] = 0.3  # N3->N2
        matrix[3, 3] = 0.7  # N3自身转移
        
        # REM -> REM, N1, W
        matrix[4, 0] = 0.1  # REM->W
        matrix[4, 1] = 0.2  # REM->N1常见
        matrix[4, 4] = 0.7  # REM自身转移
        
        return matrix
        
    def forward(self, inputs, targets, classes_weights=None, device=None):
        """
        参数:
            inputs: 模型输出 [batch_size, seq_len, num_classes]
            targets: 序列标签 [batch_size, seq_len]
            classes_weights: 可选的类别权重
            device: 可选的设备
        """
        # 使用传入或初始化时的设备
        device = device or self.device or inputs.device
        batch_size, seq_len, num_classes = inputs.shape
        
        # 为转移矩阵设置设备
        self.transition_matrix = self.transition_matrix.to(device)
        
        # 使用传入或初始化时的alpha权重
        alpha = classes_weights if classes_weights is not None else self.alpha
        if alpha is not None and not isinstance(alpha, torch.Tensor):
            alpha = torch.tensor(alpha, dtype=torch.float32).to(device)
        
        # 分类损失
        flat_inputs = inputs.reshape(-1, num_classes)
        flat_targets = targets.reshape(-1)
        
        # 应用标签平滑
        if self.label_smoothing > 0:
            smooth_targets = torch.zeros_like(flat_inputs).scatter_(
                1, flat_targets.unsqueeze(1), 1
            )
            smooth_targets = smooth_targets * (1 - self.label_smoothing) + \
                             self.label_smoothing / num_classes
                             
            # 计算交叉熵
            log_probs = F.log_softmax(flat_inputs, dim=1)
            class_loss = -(smooth_targets * log_probs).sum(dim=1)
        else:
            class_loss = F.cross_entropy(flat_inputs, flat_targets, reduction='none')
        
        # 计算focal loss权重
        probs = F.softmax(flat_inputs, dim=1)
        target_probs = probs.gather(1, flat_targets.unsqueeze(1)).squeeze()
        focal_weight = (1 - target_probs) ** self.gamma
        
        # 应用N1类别加权和alpha权重
        if alpha is not None:
            if self.n1_weight_multiplier > 1.0:
                n1_mask = (flat_targets == self.n1_class_idx)
                alpha_weight = alpha.gather(0, flat_targets)
                alpha_weight[n1_mask] = alpha_weight[n1_mask] * self.n1_weight_multiplier
            else:
                alpha_weight = alpha.gather(0, flat_targets)
                
            class_loss = alpha_weight * focal_weight * class_loss
        else:
            class_loss = focal_weight * class_loss
            
        # 添加转移约束损失
        transition_loss = 0
        if seq_len > 1 and self.transition_weight > 0:
            for i in range(batch_size):
                for j in range(seq_len-1):
                    current_prob = F.softmax(inputs[i, j], dim=0)
                    next_prob = F.softmax(inputs[i, j+1], dim=0)
                    
                    # 计算当前预测到下一个预测的转移概率
                    expected_next = torch.matmul(current_prob, self.transition_matrix)
                    
                    # KL散度作为转移损失
                    transition_kl = F.kl_div(
                        next_prob.log(), 
                        expected_next, 
                        reduction='sum'
                    )
                    transition_loss += transition_kl
            
            transition_loss = transition_loss / (batch_size * (seq_len-1))
            
        # 合并损失
        total_loss = class_loss.mean() + self.transition_weight * transition_loss
        
        return total_loss


def sequential_focal_loss(output, target, classes_weights, device, gamma=2.0, label_smoothing=0.0, 
                          n1_weight_multiplier=1.0, n1_class_idx=1, transition_weight=0.2):
    """
    序列Focal Loss的函数接口
    
    参数:
        output: 模型输出 (batch_size, seq_len, num_classes)
        target: 目标标签 (batch_size, seq_len)
        classes_weights: 类别权重列表
        device: 计算设备
        gamma: 聚焦参数，默认为2.0
        label_smoothing: 标签平滑系数，默认为0.0
        n1_weight_multiplier: N1类别权重倍增器，默认为1.0
        n1_class_idx: N1类别的索引，默认为1
        transition_weight: 转移约束权重，默认为0.2
    """
    # 获取配置参数
    num_classes = len(classes_weights)
    
    # 转换为张量并移至正确设备
    alpha = torch.tensor(classes_weights, dtype=torch.float32).to(device)
    
    criterion = SequentialFocalLoss(
        alpha=alpha, 
        gamma=gamma,
        label_smoothing=label_smoothing,
        n1_class_idx=n1_class_idx,
        n1_weight_multiplier=n1_weight_multiplier,
        transition_weight=transition_weight,
        num_classes=num_classes,
        reduction='mean',
        device=device
    )
    
    return criterion(output, target)

