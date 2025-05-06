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
    
    # 检查是否所有的预测和目标都只有一个类别
    unique_preds = np.unique(preds)
    unique_targets = np.unique(targets)
    
    # 如果预测或目标只有一个类别，则可能会导致除以零
    if len(unique_preds) <= 1 or len(unique_targets) <= 1:
        print("警告: 计算Kappa系数时检测到单一类别数据，跳过Kappa计算")
        return None  # 返回None表示没有有效的Kappa值
    
    # 确保使用所有可能的类别标签（假设是5类睡眠阶段分类）
    all_labels = list(range(outputs.size(1)))  # 根据输出张量的类别数获取所有可能的标签
    
    try:
        # 使用指定的所有标签计算kappa，即使数据中没有出现某些标签
        kappa_value = cohen_kappa_score(targets, preds, labels=all_labels)
        if np.isnan(kappa_value):
            print("警告: Kappa系数计算为NaN，跳过Kappa输出")
            return None
        return kappa_value
    except Exception as e:
        # 发生错误时回退到直接计算
        print(f"计算Kappa系数时出错: {e}，跳过Kappa输出")
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
    
    # 确保使用所有可能的类别标签
    all_labels = list(range(outputs.size(1)))  # 根据输出张量的类别数获取所有可能的标签
    
    # 处理单一类别的情况
    unique_labels = np.unique(np.concatenate([preds, targets]))
    if len(unique_labels) <= 1:
        print("警告: 只检测到一个类别，某些指标可能不准确")
        # 对于单一类别，创建一个简单的混淆矩阵
        cm = np.array([[len(preds)]])
        
        # 创建一个简化的报告
        report = {
            str(unique_labels[0]): {
                'precision': 1.0,
                'recall': 1.0,
                'f1-score': 1.0,
                'support': len(preds)
            },
            'accuracy': 1.0,
            'macro avg': {
                'precision': 1.0,
                'recall': 1.0,
                'f1-score': 1.0,
                'support': len(preds)
            },
            'weighted avg': {
                'precision': 1.0,
                'recall': 1.0,
                'f1-score': 1.0,
                'support': len(preds)
            }
        }
        
        # 创建F1分数
        per_class_f1 = np.ones(len(all_labels))
        
        return cm, report, per_class_f1
    
    # 计算混淆矩阵，确保指定所有可能的标签
    cm = confusion_matrix(targets, preds, labels=all_labels)
    
    # 每类详细指标，确保指定所有可能的标签
    report = classification_report(targets, preds, labels=all_labels, output_dict=True)
    
    # 每类F1分数，确保指定所有可能的标签
    per_class_f1 = f1_score(targets, preds, average=None, labels=all_labels)
    
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
