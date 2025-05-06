import torch
import torch.nn as nn
import numpy as np
import time
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix, cohen_kappa_score
from model.model import BioSleepX, BioSleepXSeq
from model.metric import kappa
import json
import os
from thop import profile
import argparse
from glob import glob
from utils.util import load_folds_data
from torch.utils.data import DataLoader
from data_loader.data_loaders import DualModalityDataset, SequentialEpochDataset

def count_parameters(model):
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def evaluate_model(model_path, data_dir, device='cuda', is_sequence=True, seq_length=5, stride=1, fold_id=None):
    """
    评估模型性能
    
    Args:
        model_path: 模型权重文件路径
        data_dir: 数据目录路径
        device: 运行设备
        is_sequence: 是否为序列模型
        seq_length: 序列长度
        stride: 序列步长
        fold_id: 当前评估的折数
    """
    # 加载检查点
    checkpoint = torch.load(model_path, map_location=device)
    
    # 初始化模型
    if is_sequence:
        model = BioSleepXSeq(seq_length=seq_length)
        model_name = "BioSleepXSeq"
    else:
        model = BioSleepX()
        model_name = "BioSleepX"
    
    # 加载模型权重
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        # 移除模块前缀
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k[7:]  # 移除 'module.' 前缀
            new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    # 加载预定义的数据划分
    n_folds = 10  # 假设使用10折交叉验证
    folds_data, test_files = load_folds_data(data_dir, n_folds)
    
    # 创建测试集数据加载器
    if is_sequence:
        test_dataset = SequentialEpochDataset(test_files, seq_length, stride)
    else:
        test_dataset = DualModalityDataset(test_files)
    
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 计算模型参数量
    total_params = count_parameters(model)
    
    # 计算FLOPs
    if is_sequence:
        input_eeg = torch.randn(1, seq_length, 1, 3000).to(device)
        input_eog = torch.randn(1, seq_length, 1, 3000).to(device)
    else:
        input_eeg = torch.randn(1, 1, 3000).to(device)
        input_eog = torch.randn(1, 1, 3000).to(device)
    
    flops, _ = profile(model, inputs=(input_eeg, input_eog))
    
    # 推理时间测试
    start_time = time.time()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch_eeg, batch_eog, target = batch
            batch_eeg, batch_eog = batch_eeg.to(device), batch_eog.to(device)
            output = model(batch_eeg, batch_eog)
            
            if is_sequence:
                # 对于序列模型，我们使用序列中间位置的预测
                mid_idx = output.size(1) // 2
                output = output[:, mid_idx, :]
                target = target[:, mid_idx]
            
            pred = output.argmax(dim=1).cpu().numpy()
            predictions.extend(pred)
            targets.extend(target.cpu().numpy())
    
    inference_time = (time.time() - start_time) / len(predictions)
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # 定义所有可能的类别标签
    all_labels = list(range(5))  # 假设有5类: W, N1, N2, N3, REM
    
    # 计算各项指标
    accuracy = accuracy_score(targets, predictions)
    f1_macro = f1_score(targets, predictions, average='macro', labels=all_labels)
    f1_per_class = f1_score(targets, predictions, average=None, labels=all_labels)
    f1_weighted = f1_score(targets, predictions, average='weighted', labels=all_labels)
    
    # 计算 Kappa 系数 - 增加错误处理
    try:
        # 检查是否只有一个类别
        if len(np.unique(targets)) <= 1 or len(np.unique(predictions)) <= 1:
            print("警告: 只有一个类别，Kappa 系数设为 0")
            kappa_score = 0.0
        else:
            kappa_score = cohen_kappa_score(targets, predictions, labels=all_labels)
            # 检查 NaN 值
            if np.isnan(kappa_score):
                print("警告: Kappa 系数计算为 NaN，设为 0")
                kappa_score = 0.0
    except Exception as e:
        print(f"计算 Kappa 系数时出错: {e}，设为 0")
        kappa_score = 0.0
    
    # 生成详细的分类报告
    class_report = classification_report(targets, predictions, labels=all_labels, output_dict=True)
    
    # 计算混淆矩阵
    cm = confusion_matrix(targets, predictions, labels=all_labels)
    
    # 整理结果
    results = {
        "模型信息": {
            "模型名称": model_name,
            "参数量": total_params,
            "FLOPs": flops,
            "模型权重路径": model_path,
            "测试数据路径": data_dir,
            "测试文件数量": len(test_files),
            "评估折数": fold_id,
            "模型配置": {
                "use_msea": False,
                "use_gabor": False,
                "is_sequence": is_sequence,
                "seq_length": seq_length if is_sequence else None,
                "stride": stride if is_sequence else None
            }
        },
        "性能指标": {
            "准确率": float(accuracy),
            "宏平均F1分数": float(f1_macro),
            "加权平均F1分数": float(f1_weighted),
            "Kappa系数": float(kappa_score),
            "每类F1分数": {f"类别{i}": float(score) for i, score in enumerate(f1_per_class)},
            "平均推理时间(ms)": float(inference_time * 1000)
        },
        "详细分类报告": class_report,
        "混淆矩阵": cm.tolist()
    }
    
    # 保存结果
    results_dir = os.path.dirname(model_path)
    results_file = os.path.join(results_dir, 'evaluation_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    # 打印主要结果
    print("\n=== 模型评估结果 ===")
    if fold_id is not None:
        print(f"评估折数: Fold {fold_id}")
    print(f"模型名称: {model_name}")
    print(f"模型路径: {model_path}")
    print(f"测试数据路径: {data_dir}")
    print(f"测试文件数量: {len(test_files)}")
    print(f"模型参数量: {total_params:,}")
    print(f"FLOPs: {flops:,}")
    print(f"平均推理时间: {inference_time*1000:.2f}ms")
    print(f"\n准确率: {accuracy:.4f}")
    print(f"宏平均F1分数: {f1_macro:.4f}")
    print(f"加权平均F1分数: {f1_weighted:.4f}")
    print(f"Kappa系数: {kappa_score:.4f}")
    print("\n每类F1分数:")
    for i, score in enumerate(f1_per_class):
        class_name = ['W', 'N1', 'N2', 'N3', 'REM'][i]
        print(f"{class_name}: {score:.4f}")
    
    print("\n混淆矩阵:")
    print(cm)
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='评估睡眠分期模型')
    parser.add_argument('--model_path', type=str, required=True, help='模型权重文件路径')
    parser.add_argument('--data_dir', type=str, required=True, help='数据目录路径')
    parser.add_argument('--sequence', action='store_true', help='是否为序列模型')
    parser.add_argument('--seq_length', type=int, default=5, help='序列长度')
    parser.add_argument('--stride', type=int, default=1, help='序列步长')
    parser.add_argument('--fold_id', type=int, help='评估的折数ID')
    args = parser.parse_args()
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 运行评估
    evaluate_model(
        args.model_path, 
        args.data_dir, 
        DEVICE,
        is_sequence=args.sequence,
        seq_length=args.seq_length,
        stride=args.stride,
        fold_id=args.fold_id
    ) 