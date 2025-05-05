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
            
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        # 使用dtype=torch.long确保类型匹配
        correct += torch.sum(pred == target.to(pred.device, dtype=torch.long)).item()
    return correct / len(target)


def f1(output, target):
    """
    计算F1分数 - 支持序列模型输出
    """
    with torch.no_grad():
        # 处理序列输出 (batch_size, seq_len, num_classes)
        if len(output.shape) == 3:
            batch_size, seq_len, num_classes = output.shape
            # 重塑为 (batch_size*seq_len, num_classes)
            output = output.reshape(-1, num_classes)
            target = target.reshape(-1)
            
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
    # 确保转换为numpy前数据类型一致
    return f1_score(pred.cpu().numpy(), target.cpu().detach().numpy(), average='macro')


def kappa(outputs, targets):
    """
    计算Cohen's Kappa评分 - 支持序列模型输出
    """
    # 处理序列输出 (batch_size, seq_len, num_classes)
    if len(outputs.shape) == 3:
        batch_size, seq_len, num_classes = outputs.shape
        # 重塑为 (batch_size*seq_len, num_classes)
        outputs = outputs.reshape(-1, num_classes)
        targets = targets.reshape(-1)
        
    preds = torch.argmax(outputs, dim=1).cpu().numpy()
    targets = targets.cpu().numpy()
    
    # 获取实际存在的类别标签
    unique_labels = np.unique(np.concatenate((preds, targets)))
    n_classes = len(unique_labels)
    
    # 创建混淆矩阵 - 使用实际存在的标签
    cm = confusion_matrix(targets, preds, labels=unique_labels)
    
    # 计算观察到的准确率
    n_samples = len(preds)
    observed_accuracy = np.trace(cm) / n_samples
    
    # 计算期望的准确率
    expected_accuracy = 0
    for i in range(n_classes):
        expected_accuracy += (np.sum(cm[i, :]) * np.sum(cm[:, i])) 
    
    expected_accuracy = expected_accuracy / (n_samples * n_samples)
    
    # 防止除以零
    if expected_accuracy >= 1.0 or expected_accuracy == 0.0:
        return 1.0 if observed_accuracy == 1.0 else 0.0
        
    # 计算Kappa
    kappa = (observed_accuracy - expected_accuracy) / (1 - expected_accuracy)
    
    return kappa


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
            
        pred = torch.argmax(output, dim=1)
        pred = pred.cpu().numpy()
        target = target.cpu().detach().numpy()
        
        # 初始化结果字典和计数器
        class_correct = {}
        class_total = {}
        
        # 计算每个类别的准确率
        for i in range(len(target)):
            label = target[i]
            if label not in class_total:
                class_total[label] = 0
                class_correct[label] = 0
            
            class_total[label] += 1
            if pred[i] == label:
                class_correct[label] += 1
        
        # 计算每个类别的准确率
        class_accuracy = {}
        for label in class_total:
            class_accuracy[f'acc_class_{label}'] = class_correct[label] / class_total[label]
            
        return class_accuracy


def per_class_metrics(outputs, targets):
    """
    计算每个睡眠阶段的精确度、召回率和F1分数
    
    参数:
        outputs: 模型输出
        targets: 真实标签
        
    返回:
        混淆矩阵、详细报告、每类F1分数
    """
    # 处理序列输出 (batch_size, seq_len, num_classes)
    if len(outputs.shape) == 3:
        batch_size, seq_len, num_classes = outputs.shape
        # 重塑为 (batch_size*seq_len, num_classes)
        outputs = outputs.reshape(-1, num_classes)
        targets = targets.reshape(-1)
        
    # 获取预测
    preds = torch.argmax(outputs, dim=1).cpu().numpy()
    targets = targets.cpu().numpy()
    
    # 计算混淆矩阵
    cm = confusion_matrix(targets, preds)
    
    # 每类详细指标
    report = classification_report(targets, preds, output_dict=True)
    
    # 每类F1分数
    per_class_f1 = f1_score(targets, preds, average=None)
    
    return cm, report, per_class_f1


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
                all_outputs.append(output)
                all_targets.append(target_seq)
            # 处理单个epoch数据
            elif len(batch) == 3:
                data_eeg, data_eog, target = batch
                data_eeg, data_eog = data_eeg.to(device), data_eog.to(device)
                target = target.to(device)
                output = model(data_eeg, data_eog)
                all_outputs.append(output)
                all_targets.append(target)
            else:
                data, target = batch
                data = data.to(device)
                target = target.to(device)
                output = model(data)
                all_outputs.append(output)
                all_targets.append(target)
    
    # 合并所有批次的结果
    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # 计算详细指标
    cm, report, per_class_f1 = per_class_metrics(all_outputs, all_targets)
    acc = accuracy(all_outputs, all_targets)
    f1_score_val = f1(all_outputs, all_targets)
    kappa_score = kappa(all_outputs, all_targets)
    
    # 计算转换准确率（如果适用）
    trans_acc = 0.0
    if len(all_outputs.shape) == 3:
        trans_acc = transition_accuracy(all_outputs, all_targets)
    
    return {
        'accuracy': acc,
        'f1_score': f1_score_val,
        'kappa': kappa_score,
        'transition_accuracy': trans_acc,
        'confusion_matrix': cm,
        'detailed_report': report,
        'per_class_f1': per_class_f1
    }
