import argparse
import collections
import numpy as np
from pathlib import Path
import json
import os # 确保导入 os 模块
print(f"[DEBUG] train_Kfold_CV.py: Initial CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

from data_loader.data_loaders import *
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer
from utils.util import *
from utils import read_json

import torch
import torch.nn as nn
print(f"[DEBUG] train_Kfold_CV.py: torch.cuda.is_available(): {torch.cuda.is_available()}")
print(f"[DEBUG] train_Kfold_CV.py: torch.cuda.device_count() right after torch import: {torch.cuda.device_count()}")

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)  # 为所有GPU设置种子
torch.backends.cudnn.deterministic = True  # 确保结果可复现
torch.backends.cudnn.benchmark = False  # 禁用cudnn基准测试
np.random.seed(SEED)


def weights_init_normal(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif type(m) == nn.Conv1d:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif type(m) == nn.BatchNorm1d:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def main(config, fold_id):
    batch_size = config["data_loader"]["args"]["batch_size"]
    logger = config.get_logger('train')

    # 构建模型架构，初始化权重
    use_wavelet = config["arch"]["args"].get("use_wavelet", False)
    model = config.init_obj('arch', module_arch)
    model.apply(weights_init_normal)
    logger.info(model)

    # 统计模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型总参数量: {total_params}")
    logger.info(f"可训练参数量: {trainable_params}")

    # 获取损失函数和评估指标
    # criterion = getattr(module_loss, config['loss'])
    # 获取损失函数
    if config['loss'] == 'focal_loss' or config['loss'] == 'sequential_focal_loss':
        # 直接获取函数引用，不进行实例化
        criterion = getattr(module_loss, config['loss'])
    else:
        # 获取损失函数类并实例化
        criterion_class = getattr(module_loss, config['loss'])
        criterion = criterion_class()
    
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # 构建优化器
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)

    # 获取当前折的训练集和验证集
    fold_data = folds_data[fold_id]
    
    # 根据配置文件选择适当的数据加载器
    if "type" in config["data_loader"] and config["data_loader"]["type"] == "data_generator_np_sequence":
        # 使用序列数据加载器
        seq_length = config["data_loader"]["args"]["seq_length"]
        stride = config["data_loader"]["args"]["stride"]
        data_loader, valid_data_loader, data_count = data_generator_np_sequence(
            fold_data['train'], 
            fold_data['val'], 
            batch_size,
            seq_length,
            stride
        )
    else:
        # 使用原始双模态数据加载器
        data_loader, valid_data_loader, data_count = data_generator_np_dual(
            fold_data['train'], 
            fold_data['val'], 
            batch_size
        )
    
    # 创建测试集数据加载器
    if test_files and len(test_files) > 0:
        if "type" in config["data_loader"] and config["data_loader"]["type"] == "data_generator_np_sequence":
            # 序列测试集
            seq_length = config["data_loader"]["args"]["seq_length"]
            stride = config["data_loader"]["args"]["stride"]
            test_dataset = SequentialEpochDataset(test_files, seq_length, stride)
        else:
            # 原始测试集
            test_dataset = DualModalityDataset(test_files)
            
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0
        )
    else:
        logger.info("没有测试集数据，跳过测试集初始化")
        test_loader = None
    
    # 计算类别权重
    weights_for_each_class = calc_class_weight(data_count)

    # 使用自定义权重替换计算的权重
    weights_for_each_class = [1.0, 1.80, 1.0, 1.8, 1.20]
    logger.info(f"使用自定义类别权重: {weights_for_each_class}")

    # 创建训练器
    trainer = Trainer(model, criterion, metrics, optimizer,
                     config=config,
                     data_loader=data_loader,
                     fold_id=fold_id,
                     valid_data_loader=valid_data_loader,
                     test_data_loader=test_loader,
                     class_weights=weights_for_each_class)

    # 训练模型并返回结果
    return trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--config', default="config.json", type=str,
                      help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (e.g., "0,1,2"). If not set, uses CUDA_VISIBLE_DEVICES or all available.')
    parser.add_argument('-f', '--fold_id', type=str, required=True,
                      help='fold_id')
    parser.add_argument('-da', '--np_data_dir', type=str, required=True,
                      help='Directory containing numpy files')
    parser.add_argument('--use_wavelet', action='store_true',
                      help='使用Haar小波下采样替代最大池化')

    args = parser.parse_args()
    fold_id = int(args.fold_id)
    
    print(f"[DEBUG] train_Kfold_CV.py: args.device from command line: {args.device}")
    # Only set CUDA_VISIBLE_DEVICES if -d or --device is explicitly passed
    if args.device is not None:
        print(f"[DEBUG] train_Kfold_CV.py: Setting CUDA_VISIBLE_DEVICES to: {args.device} based on -d flag")
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    
    print(f"[DEBUG] train_Kfold_CV.py: CUDA_VISIBLE_DEVICES before ConfigParser: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"[DEBUG] train_Kfold_CV.py: torch.cuda.device_count() before ConfigParser: {torch.cuda.device_count()}")
    config = ConfigParser.from_args(parser, fold_id)
    print(f"[DEBUG] train_Kfold_CV.py: CUDA_VISIBLE_DEVICES after ConfigParser: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"[DEBUG] train_Kfold_CV.py: torch.cuda.device_count() after ConfigParser: {torch.cuda.device_count()}")
    
    # 加载数据
    if "shhs" in args.np_data_dir:
        folds_data = load_folds_data_shhs(args.np_data_dir, config["data_loader"]["args"]["num_folds"])
        test_files = None  # SHHS数据集不划分测试集
    else:
        folds_data, test_files = load_folds_data(args.np_data_dir, config["data_loader"]["args"]["num_folds"])

    # 训练当前折
    fold_metrics = main(config, fold_id)
    
    # 如果是最后一折，计算并保存所有折的平均分数
    if fold_id == config["data_loader"]["args"]["num_folds"] - 1:
        # 收集所有折的结果
        all_metrics = []
        metrics_dir = os.path.join(config.save_dir, 'fold_metrics')
        for i in range(config["data_loader"]["args"]["num_folds"]):
            metrics_file = os.path.join(metrics_dir, f'fold_{i}_metrics.json')
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    fold_metrics = json.load(f)
                    all_metrics.append(fold_metrics)
        
        # 计算平均分数
        avg_metrics = {}
        for metric in all_metrics[0].keys():
            if not metric.startswith('test_'):  # 排除测试集指标
                avg_metrics[metric] = np.mean([fold[metric] for fold in all_metrics])
        
        # 计算测试集的平均分数
        test_metrics = {}
        for metric in all_metrics[0].keys():
            if metric.startswith('test_'):
                test_metrics[metric] = np.mean([fold[metric] for fold in all_metrics])
        
        # 保存验证集平均分数
        with open(os.path.join(config.save_dir, 'average_metrics.json'), 'w') as f:
            json.dump(avg_metrics, f, indent=4)
        
        # 保存测试集平均分数
        with open(os.path.join(config.save_dir, 'test_metrics.json'), 'w') as f:
            json.dump(test_metrics, f, indent=4)
        
        # 打印验证集平均分数
        logger.info("========== 交叉验证平均分数 ==========")
        for metric, value in avg_metrics.items():
            logger.info(f"{metric}: {value:.4f}")
            
        # 打印测试集平均分数
        logger.info("========== 测试集平均分数 ==========")
        for metric, value in test_metrics.items():
            logger.info(f"{metric}: {value:.4f}")