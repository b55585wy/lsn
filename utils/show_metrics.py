import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
import seaborn as sns
import pandas as pd
import os
import torch
import json
from model.metric import evaluate_model


def plot_confusion_matrix(cm, class_names=None, save_path=None):
    """
    绘制混淆矩阵
    
    Args:
        cm: 混淆矩阵
        class_names: 类别名称列表
        save_path: 保存路径
    """
    if class_names is None:
        class_names = [f'Class {i}' for i in range(cm.shape[0])]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵')
    
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_per_class_metrics(report, save_path=None):
    """
    绘制每个类别的精确度、召回率和F1分数
    
    Args:
        report: classification_report的输出字典
        save_path: 保存路径
    """
    # 提取类别指标
    classes = []
    precision = []
    recall = []
    f1 = []
    
    for key, value in report.items():
        if key.isdigit() or (isinstance(key, str) and key.startswith('class_')):
            class_name = f'Class {key}' if key.isdigit() else key
            classes.append(class_name)
            precision.append(value['precision'])
            recall.append(value['recall'])
            f1.append(value['f1-score'])
    
    # 创建DataFrame
    df = pd.DataFrame({
        'Class': classes,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    })
    
    # 绘制条形图
    plt.figure(figsize=(12, 6))
    df.set_index('Class').plot(kind='bar', figsize=(12, 6))
    plt.title('每个类别的性能指标')
    plt.ylabel('分数')
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc='lower right')
    
    if save_path:
        plt.savefig(save_path)
    plt.show()
    
    return df


def save_metrics_to_file(metrics, save_path):
    """
    将指标保存到文件
    
    Args:
        metrics: 指标字典
        save_path: 保存路径
    """
    # 转换numpy数组为列表以便JSON序列化
    serializable_metrics = {}
    for key, value in metrics.items():
        if key == 'confusion_matrix':
            serializable_metrics[key] = value.tolist()
        elif key == 'detailed_report' or key == 'per_class_f1':
            serializable_metrics[key] = value
        else:
            serializable_metrics[key] = float(value)
    
    with open(save_path, 'w') as f:
        json.dump(serializable_metrics, f, indent=4)
    
    print(f"指标已保存到 {save_path}")


def print_metrics_summary(metrics):
    """
    打印指标摘要
    
    Args:
        metrics: 指标字典
    """
    print("\n===== 模型评估指标摘要 =====")
    print(f"总体准确率: {metrics['accuracy']:.4f}")
    print(f"总体F1分数: {metrics['f1_score']:.4f}")
    if 'kappa' in metrics:
        print(f"Cohen's Kappa: {metrics['kappa']:.4f}")
    
    print("\n每个类别的F1分数:")
    for class_name, f1_val in metrics['per_class_f1'].items():
        print(f"  {class_name}: {f1_val:.4f}")
    
    print("\n详细分类报告:")
    for class_name, metrics_dict in metrics['detailed_report'].items():
        if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
            print(f"  类别 {class_name}:")
            print(f"    精确度: {metrics_dict['precision']:.4f}")
            print(f"    召回率: {metrics_dict['recall']:.4f}")
            print(f"    F1分数: {metrics_dict['f1-score']:.4f}")
            print(f"    支持度: {metrics_dict['support']}")


def evaluate_and_show_metrics(model, data_loader, device, save_dir=None, class_names=None):
    """
    评估模型并显示详细指标
    
    Args:
        model: 训练好的模型
        data_loader: 数据加载器
        device: 设备
        save_dir: 保存目录
        class_names: 类别名称列表
    """
    # 评估模型
    metrics = evaluate_model(model, data_loader, device)
    
    # 打印指标摘要
    print_metrics_summary(metrics)
    
    # 如果提供了保存目录，则创建它
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存指标到JSON文件
        save_metrics_to_file(metrics, os.path.join(save_dir, 'detailed_metrics.json'))
        
        # 绘制并保存混淆矩阵
        plot_confusion_matrix(
            metrics['confusion_matrix'], 
            class_names=class_names,
            save_path=os.path.join(save_dir, 'confusion_matrix.png')
        )
        
        # 绘制并保存每个类别的指标
        df = plot_per_class_metrics(
            metrics['detailed_report'],
            save_path=os.path.join(save_dir, 'per_class_metrics.png')
        )
        
        # 保存每个类别的指标到CSV
        df.to_csv(os.path.join(save_dir, 'per_class_metrics.csv'), index=False)
    
    return metrics