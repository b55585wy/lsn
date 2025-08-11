import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging # 添加导入


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


def contrastive_loss(projections, temperature=0.5):
    """
    计算对比学习损失（InfoNCE/NT-Xent）
    
    参数:
        projections: 投影特征 [batch_size, seq_len, proj_dim]
        temperature: 温度参数，调整相似度分布的锐度
    
    返回:
        对比学习损失
    """
    # 重塑为 [batch_size*seq_len, proj_dim]
    batch_size, seq_len, proj_dim = projections.shape
    projections = projections.reshape(-1, proj_dim)
    
    # 归一化投影向量
    projections = F.normalize(projections, p=2, dim=1)
    
    # 计算相似度矩阵
    similarity_matrix = torch.matmul(projections, projections.T) / temperature
    
    # 掩码对角线（自相似度）
    mask = torch.eye(batch_size * seq_len, dtype=torch.bool, device=projections.device)
    similarity_matrix.masked_fill_(mask, -float('inf'))
    
    # 创建用于识别同一序列不同时间步的正样本掩码
    # 对于每个(i,j)，如果它们来自同一序列但不同时间步，则为正样本
    pos_mask = torch.zeros_like(similarity_matrix, dtype=torch.bool)
    for i in range(batch_size):
        start_idx = i * seq_len
        end_idx = (i + 1) * seq_len
        # 设置同一序列内的所有位置为正样本（除了自己与自己比较的对角线）
        pos_mask[start_idx:end_idx, start_idx:end_idx] = True
    
    # 移除对角线（不把自己作为正样本）
    pos_mask.masked_fill_(mask, False)
    
    # 应用正样本掩码，计算对比学习损失
    logits = similarity_matrix.reshape(-1)
    labels = pos_mask.reshape(-1).float()
    
    # 计算每个锚点的对比损失
    # 注意：由于正样本是多个，我们使用交叉熵损失
    # 对于每个锚点，其所有正样本都应该有较高的概率
    logits_pos = similarity_matrix[pos_mask]
    logits_neg = similarity_matrix[~pos_mask & ~mask.reshape_as(pos_mask)]
    
    # 使用InfoNCE损失
    pos_term = -logits_pos.mean()  # 正样本相似度应该高
    neg_term = torch.logsumexp(logits_neg.reshape(batch_size * seq_len, -1), dim=1).mean()  # 负样本相似度应该低
    
    loss = pos_term + neg_term
    
    return loss


def combined_contrastive_classification_loss(output, target, classes_weights, device, 
                                            lambda_contrast=0.3, temperature=0.5, 
                                            gamma=2.0, label_smoothing=0.05,
                                            n1_weight_multiplier=1.5, transition_weight=0.2):
    """
    结合分类损失和对比学习损失
    
    参数:
        output: 模型输出的元组 (logits, projections, features)
            - logits: 分类预测 [batch_size, seq_len, num_classes]
            - projections: 投影特征 [batch_size, seq_len, proj_dim]
            - features: 序列特征表示 [batch_size, seq_len, d_model]
        target: 目标标签 [batch_size, seq_len]
        classes_weights: 类别权重列表
        device: 计算设备
        lambda_contrast: 对比学习损失权重
        temperature: 对比学习温度参数
        gamma: focal损失聚焦参数
        label_smoothing: 标签平滑系数
        n1_weight_multiplier: N1类别权重倍增器
        transition_weight: 转移约束权重
    """
    logits, projections, _ = output
    
    # 计算分类损失（使用序列focal损失）
    class_loss = sequential_focal_loss(
        logits, target, classes_weights, device, 
        gamma=gamma, 
        label_smoothing=label_smoothing,
        n1_weight_multiplier=n1_weight_multiplier, 
        n1_class_idx=1,
        transition_weight=transition_weight
    )
    
    # 计算对比学习损失
    contrast_loss = contrastive_loss(projections, temperature=temperature)
    
    # 综合损失
    total_loss = class_loss + lambda_contrast * contrast_loss
    
    return total_loss


class CombinedContrastiveClassificationLoss(nn.Module):
    """
    结合分类损失和对比学习损失的类实现
    
    参数:
        lambda_contrast: 对比学习损失权重
        temperature: 对比学习温度参数
        class_weights: 类别权重列表
        gamma: focal损失聚焦参数
        label_smoothing: 标签平滑系数
        n1_weight_multiplier: N1类别权重倍增器
        transition_weight: 转移约束权重
        device: 计算设备
    """
    def __init__(self, lambda_contrast=0.3, temperature=0.5, class_weights=None,
                 gamma=2.0, label_smoothing=0.05, n1_weight_multiplier=1.5,
                 transition_weight=0.2, device=None):
        super(CombinedContrastiveClassificationLoss, self).__init__()
        self.lambda_contrast = lambda_contrast
        self.temperature = temperature
        self.class_weights = class_weights
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.n1_weight_multiplier = n1_weight_multiplier
        self.transition_weight = transition_weight
        self.device = device
        
        # 初始化序列focal损失
        if class_weights is not None and device is not None:
            self.focal_criterion = SequentialFocalLoss(
                alpha=class_weights,
                gamma=gamma,
                label_smoothing=label_smoothing,
                n1_weight_multiplier=n1_weight_multiplier,
                transition_weight=transition_weight,
                device=device
            )
        else:
            self.focal_criterion = None
    
    def forward(self, output, target, classes_weights=None, device=None):
        """
        参数:
            output: 模型输出的元组 (logits, projections, features)
            target: 目标标签 [batch_size, seq_len]
            classes_weights: 可选的类别权重
            device: 可选的设备
        """
        device = device or self.device or target.device
        logits, projections, _ = output
        
        # 计算分类损失
        if self.focal_criterion is not None:
            class_loss = self.focal_criterion(logits, target)
        else:
            # 使用函数接口
            weights = classes_weights if classes_weights is not None else self.class_weights
            class_loss = sequential_focal_loss(
                logits, target, weights, device,
                gamma=self.gamma,
                label_smoothing=self.label_smoothing,
                n1_weight_multiplier=self.n1_weight_multiplier,
                transition_weight=self.transition_weight
            )
        
        # 计算对比学习损失
        contrast_loss = contrastive_loss(projections, temperature=self.temperature)
        
        # 综合损失
        total_loss = class_loss + self.lambda_contrast * contrast_loss
        
        return total_loss


# W: 0, N1: 1, N2: 2, N3: 3, REM: 4
CLASS_IDX_W = 0
CLASS_IDX_N1 = 1
CLASS_IDX_N2 = 2
CLASS_IDX_N3 = 3
CLASS_IDX_REM = 4

class TargetedMistakePenaltyLoss(nn.Module):
    def __init__(self, base_loss_fn=None, class_weights=None, device=None,
                 n2_to_n1_penalty=2.5, rem_to_n1_penalty=2.5, # 提高现有惩罚
                 n2_to_w_penalty=4.0, n3_to_w_penalty=3.0, # 新增高惩罚
                 n3_to_n2_penalty=2.0, rem_to_w_penalty=3.0, # 新增惩罚
                 rem_to_n3_penalty=2.0, # 新增 REM 到 N3 的惩罚
                 n3_to_rem_penalty=2.0, # 新增 N3 到 REM 的惩罚
                 n1_to_rem_penalty=2.0, # 新增 N1 到 REM 的惩罚
                 n2_to_rem_penalty=2.0): # 新增 N2 到 REM 的惩罚
        super().__init__()
        self.device = device
        
        if base_loss_fn is None:
            if class_weights is not None:
                self.class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
                if self.device:
                    self.class_weights_tensor = self.class_weights_tensor.to(self.device)
            else:
                self.class_weights_tensor = None
            self.base_loss_fn = nn.CrossEntropyLoss(weight=self.class_weights_tensor, reduction='none')
        else:
            self.base_loss_fn = base_loss_fn # User needs to ensure it has reduction='none'
            self.class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(self.device if self.device else 'cpu') if class_weights is not None else None

        self.n2_to_n1_penalty = n2_to_n1_penalty
        self.rem_to_n1_penalty = rem_to_n1_penalty
        self.n2_to_w_penalty = n2_to_w_penalty # 新增
        self.n3_to_w_penalty = n3_to_w_penalty # 新增
        self.n3_to_n2_penalty = n3_to_n2_penalty # 新增
        self.rem_to_w_penalty = rem_to_w_penalty # 新增
        self.rem_to_n3_penalty = rem_to_n3_penalty # 新增
        self.n3_to_rem_penalty = n3_to_rem_penalty # 新增
        self.n1_to_rem_penalty = n1_to_rem_penalty # 新增
        self.n2_to_rem_penalty = n2_to_rem_penalty # 新增
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Initialized TargetedMistakePenaltyLoss with: "
                         f"n2_to_n1_penalty={self.n2_to_n1_penalty}, "
                         f"rem_to_n1_penalty={self.rem_to_n1_penalty}, "
                         f"n2_to_w_penalty={self.n2_to_w_penalty}, " # 新增
                         f"n3_to_w_penalty={self.n3_to_w_penalty}, " # 新增
                         f"n3_to_n2_penalty={self.n3_to_n2_penalty}, " # 新增
                         f"rem_to_w_penalty={self.rem_to_w_penalty}, " # 新增
                         f"rem_to_n3_penalty={self.rem_to_n3_penalty}, " # 新增
                         f"n3_to_rem_penalty={self.n3_to_rem_penalty}, " # 新增
                         f"n1_to_rem_penalty={self.n1_to_rem_penalty}, " # 新增
                         f"n2_to_rem_penalty={self.n2_to_rem_penalty}, " # 新增
                         f"base_loss_uses_weights: {self.class_weights_tensor is not None}")

    def forward(self, outputs, targets):
        current_device = self.device or outputs.device
        
        if isinstance(self.base_loss_fn, nn.CrossEntropyLoss):
             # Ensure the internal CE loss is on the correct device if it was created in init
            self.base_loss_fn = self.base_loss_fn.to(current_device)
            if self.base_loss_fn.weight is not None:
                 self.base_loss_fn.weight = self.base_loss_fn.weight.to(current_device)
            per_sample_loss = self.base_loss_fn(outputs.to(current_device), targets.to(current_device))
        elif callable(self.base_loss_fn) and not isinstance(self.base_loss_fn, nn.Module):
            # Handle functional losses like weighted_CrossEntropyLoss
            # These functions might take weights and device as arguments
            try:
                per_sample_loss = self.base_loss_fn(outputs.to(current_device), targets.to(current_device),
                                                   classes_weights=self.class_weights_tensor.tolist() if self.class_weights_tensor is not None else None,
                                                   device=current_device)
            except TypeError: # If it does not take these args, call it simply
                per_sample_loss = self.base_loss_fn(outputs.to(current_device), targets.to(current_device))
        else: # For other nn.Module based losses passed externally
            self.base_loss_fn = self.base_loss_fn.to(current_device) # Ensure module is on correct device
            per_sample_loss = self.base_loss_fn(outputs.to(current_device), targets.to(current_device))

        predicted_classes = torch.argmax(outputs, dim=1)
        penalties = torch.ones_like(per_sample_loss, device=current_device, dtype=torch.float32)

        n2_to_n1_mask = (targets == CLASS_IDX_N2) & (predicted_classes == CLASS_IDX_N1)
        penalties[n2_to_n1_mask] = self.n2_to_n1_penalty

        rem_to_n1_mask = (targets == CLASS_IDX_REM) & (predicted_classes == CLASS_IDX_N1)
        penalties[rem_to_n1_mask] = self.rem_to_n1_penalty
        
        # 新增的特定错误惩罚
        n2_to_w_mask = (targets == CLASS_IDX_N2) & (predicted_classes == CLASS_IDX_W)
        penalties[n2_to_w_mask] = self.n2_to_w_penalty

        n3_to_w_mask = (targets == CLASS_IDX_N3) & (predicted_classes == CLASS_IDX_W)
        penalties[n3_to_w_mask] = self.n3_to_w_penalty

        n3_to_n2_mask = (targets == CLASS_IDX_N3) & (predicted_classes == CLASS_IDX_N2)
        penalties[n3_to_n2_mask] = self.n3_to_n2_penalty

        rem_to_w_mask = (targets == CLASS_IDX_REM) & (predicted_classes == CLASS_IDX_W)
        penalties[rem_to_w_mask] = self.rem_to_w_penalty

        rem_to_n3_mask = (targets == CLASS_IDX_REM) & (predicted_classes == CLASS_IDX_N3)
        penalties[rem_to_n3_mask] = self.rem_to_n3_penalty

        n3_to_rem_mask = (targets == CLASS_IDX_N3) & (predicted_classes == CLASS_IDX_REM)
        penalties[n3_to_rem_mask] = self.n3_to_rem_penalty

        n1_to_rem_mask = (targets == CLASS_IDX_N1) & (predicted_classes == CLASS_IDX_REM)
        penalties[n1_to_rem_mask] = self.n1_to_rem_penalty

        n2_to_rem_mask = (targets == CLASS_IDX_N2) & (predicted_classes == CLASS_IDX_REM)
        penalties[n2_to_rem_mask] = self.n2_to_rem_penalty
        
        penalized_loss = per_sample_loss * penalties
        return penalized_loss.mean()

