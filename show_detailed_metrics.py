import os
import torch
import argparse
import json
from pathlib import Path

# 导入必要的模块
from model.model import AttnSleep
from data_loader.data_loaders import data_generator_np
from utils.util import load_folds_data, load_folds_data_shhs
from utils.show_metrics import evaluate_and_show_metrics


def main():
    parser = argparse.ArgumentParser(description='显示详细的评估指标')
    parser.add_argument('-c', '--checkpoint', type=str, required=True,
                      help='模型检查点路径')
    parser.add_argument('-d', '--data_dir', type=str, required=True,
                      help='包含numpy文件的数据目录')
    parser.add_argument('-f', '--fold_id', type=int, default=0,
                      help='要评估的折叠ID')
    parser.add_argument('-o', '--output_dir', type=str, default='metrics_results',
                      help='保存结果的目录')
    parser.add_argument('-b', '--batch_size', type=int, default=128,
                      help='批次大小')
    parser.add_argument('--shhs', action='store_true',
                      help='是否使用SHHS数据集')
    parser.add_argument('--num_folds', type=int, default=10,
                      help='交叉验证的折叠数')
    
    args = parser.parse_args()
    
    # 检查检查点文件是否存在
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"找不到检查点文件: {args.checkpoint}")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载检查点
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    config = checkpoint['config']
    
    # 初始化模型
    model = AttnSleep()
    model.load_state_dict(checkpoint['state_dict'])
    
    # 确定设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 加载数据
    if args.shhs:
        folds_data = load_folds_data_shhs(args.data_dir, args.num_folds)
    else:
        folds_data = load_folds_data(args.data_dir, args.num_folds)
    
    # 获取验证数据加载器
    _, valid_data_loader, _ = data_generator_np(folds_data[args.fold_id][0],
                                              folds_data[args.fold_id][1],
                                              args.batch_size)
    
    # 类别名称（根据您的数据集调整）
    class_names = ['Wake', 'N1', 'N2', 'N3', 'REM']
    
    # 评估模型并显示详细指标
    print(f"\n正在评估模型: {args.checkpoint}")
    print(f"数据目录: {args.data_dir}")
    print(f"折叠ID: {args.fold_id}")
    
    metrics = evaluate_and_show_metrics(
        model=model,
        data_loader=valid_data_loader,
        device=device,
        save_dir=str(output_dir),
        class_names=class_names
    )
    
    print(f"\n详细指标已保存到: {output_dir}")
    print("您可以查看以下文件:")
    print(f"  - {output_dir}/detailed_metrics.json (详细指标的JSON文件)")
    print(f"  - {output_dir}/confusion_matrix.png (混淆矩阵图)")
    print(f"  - {output_dir}/per_class_metrics.png (每个类别的指标图)")
    print(f"  - {output_dir}/per_class_metrics.csv (每个类别的指标CSV文件)")


if __name__ == '__main__':
    main()