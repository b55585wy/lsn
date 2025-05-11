import torch
import torch.nn as nn
import numpy as np
import time
from sklearn.metrics import f1_score, accuracy_score, classification_report
from model.model import DualStreamAttnSleep  # 更新为双流模型
import json
import os
from thop import profile
import argparse

def count_parameters(model):
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def evaluate_model(model_path, data_path, device='cuda'):
    """
    评估双流模型性能
    
    Args:
        model_path: 模型权重文件路径
        data_path: 测试数据路径
        device: 运行设备
    """
    # 加载检查点
    checkpoint = torch.load(model_path)
    
    # 初始化双流模型
    model = DualStreamAttnSleep(
        attention_type="efficient_additive",
        use_wavelet=False
    )
    
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
    
    # 加载测试数据
    test_data = np.load(data_path)
    x_test_eeg = torch.FloatTensor(test_data['x_test_eeg']).to(device)
    x_test_eog = torch.FloatTensor(test_data['x_test_eog']).to(device)
    y_test = test_data['y_test']
    
    # 计算模型参数量
    total_params = count_parameters(model)
    
    # 计算FLOPs
    input_eeg = torch.randn(1, 1, 3000).to(device)
    input_eog = torch.randn(1, 1, 3000).to(device)
    flops, _ = profile(model, inputs=(input_eeg, input_eog))  # 双流模型需要两个输入
    
    # 推理时间测试
    start_time = time.time()
    predictions = []
    batch_size = 32  # 使用批处理来加速推理
    
    with torch.no_grad():
        for i in range(0, len(x_test_eeg), batch_size):
            batch_eeg = x_test_eeg[i:i+batch_size]
            batch_eog = x_test_eog[i:i+batch_size]
            output = model(batch_eeg, batch_eog)  # 双流模型接收两个输入
            pred = output.argmax(dim=1).cpu().numpy()
            predictions.extend(pred)
    
    inference_time = (time.time() - start_time) / len(x_test_eeg)
    predictions = np.array(predictions)
    
    # 计算各项指标
    accuracy = accuracy_score(y_test, predictions)
    f1_macro = f1_score(y_test, predictions, average='macro')
    f1_per_class = f1_score(y_test, predictions, average=None)
    f1_weighted = f1_score(y_test, predictions, average='weighted')
    
    # 生成详细的分类报告
    class_report = classification_report(y_test, predictions, output_dict=True)
    
    # 整理结果
    results = {
        "模型信息": {
            "模型名称": "DualStreamAttnSleep",
            "参数量": total_params,
            "FLOPs": flops,
            "模型权重路径": model_path,
            "测试数据路径": data_path,
            "模型配置": {
                "attention_type": "efficient_additive",
                "use_wavelet": False
            }
        },
        "性能指标": {
            "准确率": float(accuracy),
            "宏平均F1分数": float(f1_macro),
            "加权平均F1分数": float(f1_weighted),
            "每类F1分数": {f"类别{i}": float(score) for i, score in enumerate(f1_per_class)},
            "平均推理时间(ms)": float(inference_time * 1000)
        },
        "详细分类报告": class_report
    }
    
    # 保存结果
    results_dir = os.path.dirname(model_path)
    results_file = os.path.join(results_dir, 'evaluation_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    # 打印主要结果
    print("\n=== 模型评估结果 ===")
    print(f"模型名称: DualStreamAttnSleep")
    print(f"模型参数量: {total_params:,}")
    print(f"FLOPs: {flops:,}")
    print(f"平均推理时间: {inference_time*1000:.2f}ms")
    print(f"\n准确率: {accuracy:.4f}")
    print(f"宏平均F1分数: {f1_macro:.4f}")
    print(f"加权平均F1分数: {f1_weighted:.4f}")
    print("\n每类F1分数:")
    for i, score in enumerate(f1_per_class):
        print(f"类别{i}: {score:.4f}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='评估双流睡眠分期模型')
    parser.add_argument('--model_path', type=str, required=True, help='模型权重文件路径')
    parser.add_argument('--data_path', type=str, required=True, help='测试数据路径')
    args = parser.parse_args()
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 运行评估
    evaluate_model(args.model_path, args.data_path, DEVICE) 