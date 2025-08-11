import torch
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, classification_report, precision_recall_fscore_support, cohen_kappa_score


def accuracy(output, target):
    """
    计算准确率 - 支持序列模型输出
    """
    with torch.no_grad():
        # 处理序列输出 (batch_size, seq_len, num_classes)
        if len(output.shape) == 3:
            batch_size, seq_len, num_classes = output.shape
            # 重塑为 (batch_size*seq_len, num_classes)
            output = output.reshape(-1, num_classes)
            target = target.reshape(-1)
            
        # 提前克隆到CPU并分离梯度，避免后续操作出错
        output_cpu = output.detach().cpu()
        target_cpu = target.detach().cpu()
        
        pred = torch.argmax(output_cpu, dim=1)
        assert pred.shape[0] == len(target_cpu)
        
        # 确保目标和预测使用相同的设备和数据类型
        correct = torch.sum(pred == target_cpu).item()
        
    return correct / len(target_cpu)


def calc_f1_score(y_true, y_pred, num_classes=5, average='macro'):
    """
    自定义F1分数计算，避免sklearn的转换问题
    
    Args:
        y_true: 真实标签数组 (numpy array)
        y_pred: 预测标签数组 (numpy array)
        num_classes: 类别数
        average: 平均方法，'macro'或None
    
    Returns:
        平均F1分数或每类F1分数
    """
    # 确保输入是numpy数组
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # 计算每个类别的TP, FP, FN
    f1_scores = []
    for cls in range(num_classes):
        true_positives = np.sum((y_true == cls) & (y_pred == cls))
        false_positives = np.sum((y_true != cls) & (y_pred == cls))
        false_negatives = np.sum((y_true == cls) & (y_pred != cls))
        
        # 计算精确率和召回率
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        
        # 计算F1分数
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)
    
    # 根据average参数返回结果
    if average == 'macro':
        return np.mean(f1_scores)
    else:
        return np.array(f1_scores)


def f1(output, target):
    """
    计算宏平均F1分数 (Macro F1-score) - 默认为5个主要睡眠阶段。
    支持序列模型输出。使用自定义F1分数计算。
    """
    with torch.no_grad():
        # 处理序列输出 (batch_size, seq_len, num_classes)
        if len(output.shape) == 3:
            batch_size, seq_len, num_classes = output.shape
            # 重塑为 (batch_size*seq_len, num_classes)
            output = output.reshape(-1, num_classes)
            target = target.reshape(-1)
            
        # 提前克隆到CPU并分离梯度，避免后续操作出错
        output_cpu = output.detach().cpu()
        target_cpu = target.detach().cpu()
        
        # 计算预测标签
        pred = torch.argmax(output_cpu, dim=1).numpy()
        target_np = target_cpu.numpy()
    
    # 使用自定义F1分数计算，固定为5个类别 (通常对应Wake, N1, N2, N3, REM)
    return calc_f1_score(target_np, pred, num_classes=5, average='macro')


def calc_precision_recall(y_true, y_pred, num_classes=5, average='macro'):
    """
    自定义精确率和召回率计算，避免sklearn的转换问题
    
    Args:
        y_true: 真实标签数组 (numpy array)
        y_pred: 预测标签数组 (numpy array)
        num_classes: 类别数
        average: 平均方法，'macro'或None
    
    Returns:
        (precision, recall) 精确率和召回率的元组
    """
    # 确保输入是numpy数组
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # 计算每个类别的TP, FP, FN
    precisions = []
    recalls = []
    for cls in range(num_classes):
        true_positives = np.sum((y_true == cls) & (y_pred == cls))
        false_positives = np.sum((y_true != cls) & (y_pred == cls))
        false_negatives = np.sum((y_true == cls) & (y_pred != cls))
        
        # 计算精确率和召回率
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
    
    # 根据average参数返回结果
    if average == 'macro':
        return np.mean(precisions), np.mean(recalls)
    else:
        return np.array(precisions), np.array(recalls)


def precision(output, target):
    """
    计算精确率 - 支持序列模型输出
    使用自定义精确率计算，避免sklearn的转换问题
    """
    with torch.no_grad():
        # 处理序列输出 (batch_size, seq_len, num_classes)
        if len(output.shape) == 3:
            batch_size, seq_len, num_classes = output.shape
            # 重塑为 (batch_size*seq_len, num_classes)
            output = output.reshape(-1, num_classes)
            target = target.reshape(-1)
            
        # 提前克隆到CPU并分离梯度，避免后续操作出错
        output_cpu = output.detach().cpu()
        target_cpu = target.detach().cpu()
        
        # 计算预测标签
        pred = torch.argmax(output_cpu, dim=1).numpy()
        target_np = target_cpu.numpy()
    
    # 使用自定义精确率计算
    precision_val, _ = calc_precision_recall(target_np, pred, num_classes=output.size(1), average='macro')
    return precision_val


def recall(output, target):
    """
    计算召回率 - 支持序列模型输出
    使用自定义召回率计算，避免sklearn的转换问题
    """
    with torch.no_grad():
        # 处理序列输出 (batch_size, seq_len, num_classes)
        if len(output.shape) == 3:
            batch_size, seq_len, num_classes = output.shape
            # 重塑为 (batch_size*seq_len, num_classes)
            output = output.reshape(-1, num_classes)
            target = target.reshape(-1)
            
        # 提前克隆到CPU并分离梯度，避免后续操作出错
        output_cpu = output.detach().cpu()
        target_cpu = target.detach().cpu()
        
        # 计算预测标签
        pred = torch.argmax(output_cpu, dim=1).numpy()
        target_np = target_cpu.numpy()
    
    # 使用自定义召回率计算
    _, recall_val = calc_precision_recall(target_np, pred, num_classes=output.size(1), average='macro')
    return recall_val


def kappa(outputs, targets):
    """
    计算Cohen's Kappa评分 - 支持序列模型输出
    """
    with torch.no_grad():
        # 处理序列输出 (batch_size, seq_len, num_classes)
        if len(outputs.shape) == 3:
            batch_size, seq_len, num_classes = outputs.shape
            # 重塑为 (batch_size*seq_len, num_classes)
            outputs = outputs.reshape(-1, num_classes)
            targets = targets.reshape(-1)
        
        # 提前克隆到CPU并分离梯度，避免后续操作出错
        outputs_cpu = outputs.detach().cpu()
        targets_cpu = targets.detach().cpu()
        
        preds = torch.argmax(outputs_cpu, dim=1).numpy()
        targets_np = targets_cpu.numpy()
        
        # 检查是否所有的预测和目标都只有一个类别
        unique_preds = np.unique(preds)
        unique_targets = np.unique(targets_np)
    
    # 如果预测或目标只有一个类别，则可能会导致除以零
    if len(unique_preds) <= 1 or len(unique_targets) <= 1:
        # print("警告: 计算Kappa系数时检测到单一类别数据，跳过Kappa计算")
        return None  # 返回None表示没有有效的Kappa值
    
    # 确保使用所有可能的类别标签（假设是5类睡眠阶段分类）
    all_labels = list(range(outputs.size(1)))  # 根据输出张量的类别数获取所有可能的标签
    
    try:
        # 使用指定的所有标签计算kappa，即使数据中没有出现某些标签
        kappa_value = cohen_kappa_score(targets_np, preds, labels=all_labels)
        if np.isnan(kappa_value):
            # print("警告: Kappa系数计算为NaN，跳过Kappa输出")
            return None
        return kappa_value
    except Exception as e:
        # 发生错误时回退到直接计算
        # print(f"计算Kappa系数时出错: {e}，跳过Kappa输出")
        return None


def per_class_accuracy(output, target):
    """计算每个类别的准确率
    
    Args:
        output: 模型输出的预测概率
        target: 真实标签
        
    Returns:
        dict: 包含每个类别准确率的字典
    """
    with torch.no_grad():
        # 处理序列输出 (batch_size, seq_len, num_classes)
        if len(output.shape) == 3:
            batch_size, seq_len, num_classes = output.shape
            # 重塑为 (batch_size*seq_len, num_classes)
            output = output.reshape(-1, num_classes)
            target = target.reshape(-1)
            
        # 提前克隆到CPU并分离梯度，避免后续操作出错
        output_cpu = output.detach().cpu()
        target_cpu = target.detach().cpu()
        
        pred = torch.argmax(output_cpu, dim=1).numpy()
        target_np = target_cpu.numpy()
        
        # 初始化结果字典和计数器
        class_correct = {}
        class_total = {}
        
        # 计算每个类别的准确率
        for i in range(len(target_np)):
            label = target_np[i]
            if label not in class_total:
                class_total[label] = 0
                class_correct[label] = 0
            
            class_total[label] += 1
            if pred[i] == label:
                class_correct[label] += 1
        
        # 计算每个类别的准确率
        class_accuracy = {}
        for label in class_total:
            class_accuracy[f'acc_class_{int(label)}'] = class_correct[label] / class_total[label]
            
        return class_accuracy


def per_class_metrics(outputs, targets):
    """
    计算每个睡眠阶段的精确度、召回率和F1分数
    使用自定义指标计算，避免sklearn的转换问题
    
    参数:
        outputs: 模型输出
        targets: 真实标签
        
    返回:
        混淆矩阵、详细报告、每类F1分数
    """
    with torch.no_grad():
        # 处理序列输出 (batch_size, seq_len, num_classes)
        if len(outputs.shape) == 3:
            batch_size, seq_len, num_classes = outputs.shape
            # 重塑为 (batch_size*seq_len, num_classes)
            outputs = outputs.reshape(-1, num_classes)
            targets = targets.reshape(-1)
        
        # 提前克隆到CPU并分离梯度，避免后续操作出错
        outputs_cpu = outputs.detach().cpu()
        targets_cpu = targets.detach().cpu()
        
        # 获取预测
        preds = torch.argmax(outputs_cpu, dim=1).numpy()
        targets_np = targets_cpu.numpy()
    
    # 确保使用所有可能的类别标签
    num_classes = outputs.size(1)  # 根据输出张量的类别数获取所有可能的标签
    
    # 计算混淆矩阵
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(targets_np, preds):
        cm[t, p] += 1
    
    # 使用自定义函数计算每类精确率和召回率
    precisions, recalls = calc_precision_recall(targets_np, preds, num_classes=num_classes, average=None)
    
    # 使用自定义函数计算每类F1分数
    per_class_f1 = calc_f1_score(targets_np, preds, num_classes=num_classes, average=None)
    
    # 创建详细报告字典
    report = {}
    for i in range(num_classes):
        report[str(i)] = {
            'precision': precisions[i],
            'recall': recalls[i],
            'f1-score': per_class_f1[i],
            'support': np.sum(targets_np == i)
        }
    
    # 计算宏平均和加权平均
    macro_precision = np.mean(precisions)
    macro_recall = np.mean(recalls)
    macro_f1 = np.mean(per_class_f1)
    total_support = len(targets_np)
    
    # 加权平均
    weights = np.array([np.sum(targets_np == i) for i in range(num_classes)]) / total_support
    weighted_precision = np.sum(precisions * weights)
    weighted_recall = np.sum(recalls * weights)
    weighted_f1 = np.sum(per_class_f1 * weights)
    
    # 添加平均指标
    report['accuracy'] = np.mean(preds == targets_np)
    report['macro avg'] = {
        'precision': macro_precision,
        'recall': macro_recall,
        'f1-score': macro_f1,
        'support': total_support
    }
    report['weighted avg'] = {
        'precision': weighted_precision,
        'recall': weighted_recall,
        'f1-score': weighted_f1,
        'support': total_support
    }
    
    return cm, report, per_class_f1


def cohen_kappa(outputs, targets):
    """kappa的别名，为了与config.json中的度量名称匹配"""
    return kappa(outputs, targets)


def confusion(outputs, targets):
    """
    计算混淆矩阵 - 为了与config.json中的度量名称匹配
    这个函数只返回一个标量（为了训练时的监控），实际的混淆矩阵通过evaluate_model函数获取
    """
    with torch.no_grad():
        # 处理序列输出 (batch_size, seq_len, num_classes)
        if len(outputs.shape) == 3:
            batch_size, seq_len, num_classes = outputs.shape
            # 重塑为 (batch_size*seq_len, num_classes)
            outputs = outputs.reshape(-1, num_classes)
            targets = targets.reshape(-1)
        
        # 提前克隆到CPU并分离梯度，避免后续操作出错
        outputs_cpu = outputs.detach().cpu()
        targets_cpu = targets.detach().cpu()
    
        # 为了与其他度量函数保持一致，这里只返回一个标量值
        # 在实际使用时，完整的混淆矩阵应通过evaluate_model函数获取
        preds = torch.argmax(outputs_cpu, dim=1).numpy()
        targets_np = targets_cpu.numpy()
    
    # 返回预测准确率作为标量值
    return np.mean(preds == targets_np)


def transition_accuracy(outputs, targets):
    """
    计算睡眠阶段转换的准确率
    
    参数:
        outputs: 模型输出 (batch_size, seq_len, num_classes) 或 (batch_size, num_classes)
        targets: 真实标签 (batch_size, seq_len) 或 (batch_size,)
    
    返回:
        转换准确率
    """
    # 处理序列输出 (batch_size, seq_len, num_classes)
    if len(outputs.shape) == 3:
        batch_size, seq_len, num_classes = outputs.shape
        
        # 获取预测标签
        preds = torch.argmax(outputs, dim=2).cpu().numpy()  # [batch_size, seq_len]
        targets = targets.cpu().numpy()  # [batch_size, seq_len]
        
        # 计算转换准确率
        correct_transitions = 0
        total_transitions = 0
        
        for i in range(batch_size):
            for j in range(seq_len - 1):
                # 真实转换
                true_trans = (targets[i, j], targets[i, j+1])
                # 预测转换
                pred_trans = (preds[i, j], preds[i, j+1])
                
                # 判断转换是否正确
                if true_trans == pred_trans:
                    correct_transitions += 1
                    
                total_transitions += 1
                
        # 防止除以零
        if total_transitions == 0:
            return 0.0
            
        return correct_transitions / total_transitions
    
    # 处理普通输出 (batch_size, num_classes) - 无法计算转换
    else:
        # 对于单个epoch的输出，无法评估转换
        return 0.0


def evaluate_model(model, data_loader, device):
    """完整的模型评估函数"""
    model.eval()
    all_outputs = []
    all_targets = []
    
    with torch.no_grad():
        for batch in data_loader:
            # 处理序列数据
            if len(batch) == 3 and isinstance(batch[0], torch.Tensor) and len(batch[0].shape) == 4:
                eeg_seq, eog_seq, target_seq = batch
                eeg_seq, eog_seq = eeg_seq.to(device), eog_seq.to(device)
                target_seq = target_seq.to(device)
                output = model(eeg_seq, eog_seq)
                
                # 处理对比学习输出
                if isinstance(output, tuple) and len(output) == 3:
                    output = output[0]  # 获取logits
                
                all_outputs.append(output.detach().cpu())
                all_targets.append(target_seq.detach().cpu())
                
            # 处理单个epoch数据
            elif len(batch) == 3:
                data_eeg, data_eog, target = batch
                data_eeg, data_eog = data_eeg.to(device), data_eog.to(device)
                target = target.to(device)
                output = model(data_eeg, data_eog)
                
                # 处理对比学习输出
                if isinstance(output, tuple) and len(output) == 3:
                    output = output[0]  # 获取logits
                
                all_outputs.append(output.detach().cpu())
                all_targets.append(target.detach().cpu())
            else:
                raise ValueError("不支持的数据格式")
    
    # 合并所有输出和目标
    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # 计算准确率
    acc = accuracy(all_outputs, all_targets)
    
    # 计算F1分数
    f1_score_val = f1(all_outputs, all_targets)
    
    # 计算Kappa系数
    kappa_score = kappa(all_outputs, all_targets)
    
    # 如果kappa_score为None（单一类别情况），设置为0.0
    if kappa_score is None:
        kappa_score = 0.0
    
    # 计算睡眠转换准确率
    trans_acc = transition_accuracy(all_outputs, all_targets)
    
    # 计算混淆矩阵和每类指标
    cm, report, per_class_f1 = per_class_metrics(all_outputs, all_targets)
    
    return {
        'accuracy': acc,
        'f1_score': f1_score_val,
        'kappa': kappa_score,
        'transition_accuracy': trans_acc,
        'confusion_matrix': cm,
        'detailed_report': report,
        'per_class_f1': per_class_f1
    }


def f1_score(output, target):
    """
    f1函数的别名，与配置文件中的度量名称匹配
    """
    return f1(output, target)


def precision_score(output, target):
    """
    precision函数的别名，与配置文件中的度量名称匹配
    """
    return precision(output, target)


def recall_score(output, target):
    """
    recall函数的别名，与配置文件中的度量名称匹配
    """
    return recall(output, target)


def f1_score_wake(output, target):
    """计算 Wake 阶段的 F1 分数"""
    with torch.no_grad():
        if len(output.shape) == 3:
            output = output.reshape(-1, output.shape[-1])
            target = target.reshape(-1)
        output_cpu = output.detach().cpu()
        target_cpu = target.detach().cpu()
        pred = torch.argmax(output_cpu, dim=1).numpy()
        target_np = target_cpu.numpy()
    # Wake 阶段对应类别 0
    per_class_f1 = calc_f1_score(target_np, pred, num_classes=output.size(1), average=None)
    return per_class_f1[0] if len(per_class_f1) > 0 else 0.0


def f1_score_n1(output, target):
    """计算 N1 阶段的 F1 分数"""
    with torch.no_grad():
        if len(output.shape) == 3:
            output = output.reshape(-1, output.shape[-1])
            target = target.reshape(-1)
        output_cpu = output.detach().cpu()
        target_cpu = target.detach().cpu()
        pred = torch.argmax(output_cpu, dim=1).numpy()
        target_np = target_cpu.numpy()
    # N1 阶段对应类别 1
    per_class_f1 = calc_f1_score(target_np, pred, num_classes=output.size(1), average=None)
    return per_class_f1[1] if len(per_class_f1) > 1 else 0.0


def f1_score_n2(output, target):
    """计算 N2 阶段的 F1 分数"""
    with torch.no_grad():
        if len(output.shape) == 3:
            output = output.reshape(-1, output.shape[-1])
            target = target.reshape(-1)
        output_cpu = output.detach().cpu()
        target_cpu = target.detach().cpu()
        pred = torch.argmax(output_cpu, dim=1).numpy()
        target_np = target_cpu.numpy()
    # N2 阶段对应类别 2
    per_class_f1 = calc_f1_score(target_np, pred, num_classes=output.size(1), average=None)
    return per_class_f1[2] if len(per_class_f1) > 2 else 0.0


def f1_score_n3(output, target):
    """计算 N3 阶段的 F1 分数"""
    with torch.no_grad():
        if len(output.shape) == 3:
            output = output.reshape(-1, output.shape[-1])
            target = target.reshape(-1)
        output_cpu = output.detach().cpu()
        target_cpu = target.detach().cpu()
        pred = torch.argmax(output_cpu, dim=1).numpy()
        target_np = target_cpu.numpy()
    # N3 阶段对应类别 3
    per_class_f1 = calc_f1_score(target_np, pred, num_classes=output.size(1), average=None)
    return per_class_f1[3] if len(per_class_f1) > 3 else 0.0


def f1_score_rem(output, target):
    """计算 REM 阶段的 F1 分数"""
    with torch.no_grad():
        if len(output.shape) == 3:
            output = output.reshape(-1, output.shape[-1])
            target = target.reshape(-1)
        output_cpu = output.detach().cpu()
        target_cpu = target.detach().cpu()
        pred = torch.argmax(output_cpu, dim=1).numpy()
        target_np = target_cpu.numpy()
    # REM 阶段对应类别 4
    per_class_f1 = calc_f1_score(target_np, pred, num_classes=output.size(1), average=None)
    return per_class_f1[4] if len(per_class_f1) > 4 else 0.0


def avg_f1_score(output, target):
    """计算平均F1分数，作为 f1 函数的别名"""
    return f1(output, target)


def precision_wake(output, target):
    """计算 Wake 阶段的精确率"""
    with torch.no_grad():
        if len(output.shape) == 3:
            output = output.reshape(-1, output.shape[-1])
            target = target.reshape(-1)
        output_cpu = output.detach().cpu()
        target_cpu = target.detach().cpu()
        pred = torch.argmax(output_cpu, dim=1).numpy()
        target_np = target_cpu.numpy()
    # Wake 阶段对应类别 0
    precisions, _ = calc_precision_recall(target_np, pred, num_classes=output.size(1), average=None)
    return precisions[0] if len(precisions) > 0 else 0.0


def precision_n1(output, target):
    """计算 N1 阶段的精确率"""
    with torch.no_grad():
        if len(output.shape) == 3:
            output = output.reshape(-1, output.shape[-1])
            target = target.reshape(-1)
        output_cpu = output.detach().cpu()
        target_cpu = target.detach().cpu()
        pred = torch.argmax(output_cpu, dim=1).numpy()
        target_np = target_cpu.numpy()
    # N1 阶段对应类别 1
    precisions, _ = calc_precision_recall(target_np, pred, num_classes=output.size(1), average=None)
    return precisions[1] if len(precisions) > 1 else 0.0


def precision_n2(output, target):
    """计算 N2 阶段的精确率"""
    with torch.no_grad():
        if len(output.shape) == 3:
            output = output.reshape(-1, output.shape[-1])
            target = target.reshape(-1)
        output_cpu = output.detach().cpu()
        target_cpu = target.detach().cpu()
        pred = torch.argmax(output_cpu, dim=1).numpy()
        target_np = target_cpu.numpy()
    # N2 阶段对应类别 2
    precisions, _ = calc_precision_recall(target_np, pred, num_classes=output.size(1), average=None)
    return precisions[2] if len(precisions) > 2 else 0.0


def precision_n3(output, target):
    """计算 N3 阶段的精确率"""
    with torch.no_grad():
        if len(output.shape) == 3:
            output = output.reshape(-1, output.shape[-1])
            target = target.reshape(-1)
        output_cpu = output.detach().cpu()
        target_cpu = target.detach().cpu()
        pred = torch.argmax(output_cpu, dim=1).numpy()
        target_np = target_cpu.numpy()
    # N3 阶段对应类别 3
    precisions, _ = calc_precision_recall(target_np, pred, num_classes=output.size(1), average=None)
    return precisions[3] if len(precisions) > 3 else 0.0


def precision_rem(output, target):
    """计算 REM 阶段的精确率"""
    with torch.no_grad():
        if len(output.shape) == 3:
            output = output.reshape(-1, output.shape[-1])
            target = target.reshape(-1)
        output_cpu = output.detach().cpu()
        target_cpu = target.detach().cpu()
        pred = torch.argmax(output_cpu, dim=1).numpy()
        target_np = target_cpu.numpy()
    # REM 阶段对应类别 4
    precisions, _ = calc_precision_recall(target_np, pred, num_classes=output.size(1), average=None)
    return precisions[4] if len(precisions) > 4 else 0.0


def recall_wake(output, target):
    """计算 Wake 阶段的召回率"""
    with torch.no_grad():
        if len(output.shape) == 3:
            output = output.reshape(-1, output.shape[-1])
            target = target.reshape(-1)
        output_cpu = output.detach().cpu()
        target_cpu = target.detach().cpu()
        pred = torch.argmax(output_cpu, dim=1).numpy()
        target_np = target_cpu.numpy()
    # Wake 阶段对应类别 0
    _, recalls = calc_precision_recall(target_np, pred, num_classes=output.size(1), average=None)
    return recalls[0] if len(recalls) > 0 else 0.0


def recall_n1(output, target):
    """计算 N1 阶段的召回率"""
    with torch.no_grad():
        if len(output.shape) == 3:
            output = output.reshape(-1, output.shape[-1])
            target = target.reshape(-1)
        output_cpu = output.detach().cpu()
        target_cpu = target.detach().cpu()
        pred = torch.argmax(output_cpu, dim=1).numpy()
        target_np = target_cpu.numpy()
    # N1 阶段对应类别 1
    _, recalls = calc_precision_recall(target_np, pred, num_classes=output.size(1), average=None)
    return recalls[1] if len(recalls) > 1 else 0.0


def recall_n2(output, target):
    """计算 N2 阶段的召回率"""
    with torch.no_grad():
        if len(output.shape) == 3:
            output = output.reshape(-1, output.shape[-1])
            target = target.reshape(-1)
        output_cpu = output.detach().cpu()
        target_cpu = target.detach().cpu()
        pred = torch.argmax(output_cpu, dim=1).numpy()
        target_np = target_cpu.numpy()
    # N2 阶段对应类别 2
    _, recalls = calc_precision_recall(target_np, pred, num_classes=output.size(1), average=None)
    return recalls[2] if len(recalls) > 2 else 0.0


def recall_n3(output, target):
    """计算 N3 阶段的召回率"""
    with torch.no_grad():
        if len(output.shape) == 3:
            output = output.reshape(-1, output.shape[-1])
            target = target.reshape(-1)
        output_cpu = output.detach().cpu()
        target_cpu = target.detach().cpu()
        pred = torch.argmax(output_cpu, dim=1).numpy()
        target_np = target_cpu.numpy()
    # N3 阶段对应类别 3
    _, recalls = calc_precision_recall(target_np, pred, num_classes=output.size(1), average=None)
    return recalls[3] if len(recalls) > 3 else 0.0


def recall_rem(output, target):
    """计算 REM 阶段的召回率"""
    with torch.no_grad():
        if len(output.shape) == 3:
            output = output.reshape(-1, output.shape[-1])
            target = target.reshape(-1)
        output_cpu = output.detach().cpu()
        target_cpu = target.detach().cpu()
        pred = torch.argmax(output_cpu, dim=1).numpy()
        target_np = target_cpu.numpy()
    # REM 阶段对应类别 4
    _, recalls = calc_precision_recall(target_np, pred, num_classes=output.size(1), average=None)
    return recalls[4] if len(recalls) > 4 else 0.0
