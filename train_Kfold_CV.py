import argparse
import collections
import numpy as np
from pathlib import Path
import json
import os # 确保导入 os 模块
print(f"[DEBUG] train_Kfold_CV.py: Initial CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
import os # 确保导入 os 模块
print(f"[DEBUG] train_Kfold_CV.py: Initial CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

from data_loader.data_loaders import *
import model.loss as module_loss
import model.metric as module_metric
import model as module_arch
from parse_config import ConfigParser
from trainer import Trainer
from utils.util import *
from utils import read_json
from model.augmentation import Compose, RandomNoise, RandomScaling # 导入增强类

import torch
import torch.nn as nn
print(f"[DEBUG] train_Kfold_CV.py: torch.cuda.is_available(): {torch.cuda.is_available()}")
print(f"[DEBUG] train_Kfold_CV.py: torch.cuda.device_count() right after torch import: {torch.cuda.device_count()}")
print(f"[DEBUG] train_Kfold_CV.py: torch.cuda.is_available(): {torch.cuda.is_available()}")
print(f"[DEBUG] train_Kfold_CV.py: torch.cuda.device_count() right after torch import: {torch.cuda.device_count()}")

# fix random seeds for reproducibility
SEED = 42
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


def finetune_model(config, fold_id, model_path, folds_data):
    """
    加载预训练模型并微调最后几层，专注于改进N1和REM分类
    
    Args:
        config: 配置对象
        fold_id: 当前折ID
        model_path: 预训练模型路径
        folds_data: 所有折的数据
    """
    batch_size = config["data_loader"]["args"]["batch_size"]
    logger = config.get_logger('finetune')
    
    # 获取设备信息
    device = torch.device(f'cuda:{config.config["device_ids"][0]}' if config["n_gpu"] > 0 and len(config.config["device_ids"]) > 0 else 'cpu')
    logger.info(f"微调阶段 - 使用设备: {device}")
    
    # 加载模型
    logger.info(f"从 {model_path} 加载预训练模型进行微调")
    model = config.init_obj('arch', module_arch)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    
    # 冻结特征提取部分
    logger.info("冻结特征提取器层...")
    for param in model.eeg_feature_extractor.parameters():
        param.requires_grad = False
    for param in model.eog_feature_extractor.parameters():
        param.requires_grad = False
    
    # 可选：选择性冻结其他层
    # 冻结fusion_layer
    logger.info("冻结fusion_layer，保留TCE和分类器可训练...")
    for param in model.fusion_layer.parameters():
        param.requires_grad = False
        
    # 保留dim_adjust_tce层可训练（如果有的话）
    if hasattr(model, 'dim_adjust_tce'):
        for param in model.dim_adjust_tce.parameters():
            param.requires_grad = True
            
    # 保留TCE和分类器可训练
    for param in model.tce.parameters():
        param.requires_grad = True
    
    for param in model.classifier.parameters():
        param.requires_grad = True
    
    # 计算可训练参数数量
    trainable_params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params_count = sum(p.numel() for p in model.parameters())
    logger.info(f"微调可训练参数: {trainable_params_count}/{total_params_count} ({trainable_params_count/total_params_count:.2%})")
    
    # 获取当前折的训练集和测试集
    current_fold_data = folds_data[fold_id]
    train_files_for_fold = current_fold_data['train']
    test_files_for_fold = current_fold_data['test']
    
    # 创建数据加载器，这里我们仍然使用所有数据，但将通过类别权重关注N1和REM
    if "type" in config["data_loader"] and config["data_loader"]["type"] == "data_generator_np_sequence":
        seq_length = config["data_loader"]["args"]["seq_length"]
        stride = config["data_loader"]["args"]["stride"]
        data_loader, valid_data_loader, data_count = data_generator_np_sequence(
            train_files_for_fold, 
            test_files_for_fold,
            batch_size,
            seq_length,
            stride
        )
    else:
        data_loader, valid_data_loader, data_count = data_generator_np_dual(
            train_files_for_fold, 
            test_files_for_fold,
            batch_size
        )
    
    # 设置针对N1和REM的更高权重
    # 从配置文件中获取微调阶段的类别权重，如果未指定则使用默认值
    weights_for_each_class = config['loss']['args'].get('finetune_class_weights', [0.5, 3.0, 0.5, 0.5, 2.5])
    logger.info(f"微调阶段使用增强类别权重: {weights_for_each_class}")
    
    # 实例化损失函数，增加针对N1的惩罚
    loss_config = config['loss']
    if isinstance(loss_config, dict) and loss_config['type'] == "TargetedMistakePenaltyLoss":
        loss_args = loss_config.get('args', {}).copy()
        # 增强惩罚系数
        loss_args['n2_to_n1_penalty'] = 4.0  # 加强惩罚
        loss_args['rem_to_n1_penalty'] = 4.0  # 加强惩罚
        loss_args['device'] = device
        loss_args['class_weights'] = weights_for_each_class
        criterion = module_loss.TargetedMistakePenaltyLoss(**loss_args)
        logger.info(f"微调阶段使用增强惩罚系数的TargetedMistakePenaltyLoss: n2_to_n1={loss_args['n2_to_n1_penalty']}, rem_to_n1={loss_args['rem_to_n1_penalty']}")
    else:
        # 使用交叉熵损失
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights_for_each_class).to(device))
        logger.info(f"微调阶段使用CrossEntropyLoss，权重={weights_for_each_class}")
    
    # 准备优化器（使用较小学习率）
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    # 降低学习率
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * 0.1
    logger.info(f"微调阶段使用降低的学习率: {optimizer.param_groups[0]['lr']}")
    
    # 使用指标
    metrics = [getattr(module_metric, met) for met in config['metrics']]
    
    # 创建训练器并开始微调
    # 我们创建一个全新的trainer对象，为其添加finetune_prefix以区分保存的模型
    finetune_config = config.config.copy()
    finetune_config['trainer']['save_dir'] = os.path.join(config.save_dir, f"finetune_{fold_id}")
    finetune_config['trainer']['early_stop'] = 10  # 减少早停周期
    finetune_config['trainer']['epochs'] = 30      # 减少训练周期
    finetune_config = ConfigParser(config=finetune_config)
    
    trainer = Trainer(model, criterion, metrics, optimizer,
                     config=finetune_config,
                     data_loader=data_loader,
                     fold_id=fold_id,
                     valid_data_loader=valid_data_loader,
                     test_data_loader=valid_data_loader,
                     class_weights=weights_for_each_class)
    
    logger.info("开始N1和REM微调阶段...")
    return trainer.train()


def main(config, fold_id):
    batch_size = config["data_loader"]["args"]["batch_size"]
    logger = config.get_logger('train')

    # 获取设备信息，损失函数实例化时可能需要
    if config["n_gpu"] > 0:
        # 假设 ConfigParser 已经通过 os.environ["CUDA_VISIBLE_DEVICES"] 正确设置了可见的GPU
        # 'cuda:0' 将会是 PyTorch 能看到的第一个（也可能是唯一一个）GPU
        device = torch.device('cuda:0') 
    else:
        device = torch.device('cpu')
    logger.info(f"Loss function will be configured for device: {device}")

    # 获取损失函数配置
    loss_config = config['loss'] 
    
    # 获取类别权重 (这将作为 TargetedMistakePenaltyLoss 的基础CE权重，或直接用于其他损失)
    # 您当前的硬编码值: weights_for_each_class = [1.2, 1.8, 1.4, 1.5, 1.6]
    # 我们从脚本中获取这个列表，而不是从loss_config['args']中再次读取，以保持一致性
    # (在更下面几行代码 weights_for_each_class 会被定义)
    # current_class_weights = weights_for_each_class 

    if isinstance(loss_config, str):
        loss_name = loss_config
        loss_args_from_config = {}
    elif isinstance(loss_config, dict):
        loss_name = loss_config['type']
        loss_args_from_config = loss_config.get('args', {})
    else:
        raise TypeError("Loss configuration in JSON should be a string (loss name) or a dict (type and args).")

    criterion = None
    if loss_name == "TargetedMistakePenaltyLoss":
        # class_weights 将在下面 weights_for_each_class 定义后获取并传入
        # device 已经获取
        # 其他参数如 n2_to_n1_penalty 从 loss_args_from_config 获取，可以设置默认值
        penalty_args = {
            'n2_to_n1_penalty': loss_args_from_config.get('n2_to_n1_penalty', 2.0), # 默认值2.0
            'rem_to_n1_penalty': loss_args_from_config.get('rem_to_n1_penalty', 2.0), # 默认值2.0
            'device': device
            # 'class_weights' 将在下面单独处理并传入
        }
        # 实例化将在获取 class_weights 之后完成
        logger.info(f"Preparing TargetedMistakePenaltyLoss with penalty args: {penalty_args} (class_weights to be added)")
    elif loss_name in ['focal_loss', 'sequential_focal_loss', 'weighted_CrossEntropyLoss']:
        criterion = getattr(module_loss, loss_name)
        logger.info(f"Using functional loss: {loss_name}. class_weights and device will be passed by Trainer.")
    elif loss_name == 'combined_contrastive_classification_loss':
        contrastive_params = config.config.get('contrastive_params', {})
        focal_loss_params = config.config.get('focal_loss_params', {})
        criterion = module_loss.CombinedContrastiveClassificationLoss(
            lambda_contrast=contrastive_params.get('lambda_contrast', loss_args_from_config.get('lambda_contrast', 0.3)),
            temperature=contrastive_params.get('temperature', loss_args_from_config.get('temperature', 0.5)),
            gamma=focal_loss_params.get('gamma', loss_args_from_config.get('gamma', 2.0)),
            label_smoothing=focal_loss_params.get('label_smoothing', loss_args_from_config.get('label_smoothing', 0.05)),
            n1_weight_multiplier=focal_loss_params.get('n1_weight_multiplier', loss_args_from_config.get('n1_weight_multiplier', 1.5)),
            transition_weight=focal_loss_params.get('transition_weight', loss_args_from_config.get('transition_weight', 0.2)),
            device=device 
        )
        logger.info(f"Using CombinedContrastiveClassificationLoss with merged args.")
    else:
        criterion_class = getattr(module_loss, loss_name)
        try:
            criterion = criterion_class(**loss_args_from_config) # 尝试用config中的args实例化
            if hasattr(criterion, 'to'):
                criterion = criterion.to(device)
        except TypeError:
            logger.warning(f"Loss class {loss_name} could not be instantiated with args {loss_args_from_config}. Trying without args.")
            criterion = criterion_class()
            if hasattr(criterion, 'to'):
                criterion = criterion.to(device)
        logger.info(f"Using loss: {loss_name} instantiated from class.")
    
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # 初始化模型
    logger.info("初始化模型...")
    model = config.init_obj('arch', module_arch)
    model = model.to(device)

    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型总参数量: {total_params:,}")
    logger.info(f"模型可训练参数量: {trainable_params:,} ({trainable_params/total_params:.2%} of total)")
    
    # 构建优化器
    trainable_params_filter = filter(lambda p: p.requires_grad, model.parameters()) # 使用一个新的变量名以避免覆盖
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params_filter)

    # 获取当前折的训练集和测试集 (previously validation set)
    current_fold_data = folds_data[fold_id] # This is now {'train': [...], 'test': [...]}
    
    train_files_for_fold = current_fold_data['train']
    test_files_for_fold = current_fold_data['test']

    # 根据配置文件选择适当的数据加载器
    if "type" in config["data_loader"] and config["data_loader"]["type"] == "data_generator_np_sequence":
        # 使用序列数据加载器
        seq_length = config["data_loader"]["args"]["seq_length"]
        stride = config["data_loader"]["args"]["stride"]
        
        # 根据配置文件中的use_augmentation决定是否应用数据增强
        train_transform = None
        if config['arch']['args'].get('use_augmentation', False):
            train_transform = Compose([
                RandomNoise(noise_std=0.01), # 可以根据需要调整参数
                RandomScaling(scaling_range=(0.9, 1.1)) # 可以根据需要调整参数
            ])
            logger.info("训练数据将应用 RandomNoise 和 RandomScaling 增强。")
        else:
            logger.info("未启用数据增强。")

        # data_loader is for training, valid_data_loader is for epoch validation (using the fold's test set)
        data_loader, valid_data_loader, data_count = data_generator_np_sequence(
            train_files_for_fold,
            test_files_for_fold, # Use test set of the fold for validation during training
            batch_size,
            seq_length,
            stride,
            transform=train_transform # 传递数据增强转换
        )
    else:
        # 使用原始双模态数据加载器
        # data_loader is for training, valid_data_loader is for epoch validation (using the fold's test set)
        data_loader, valid_data_loader, data_count = data_generator_np_dual(
            train_files_for_fold, 
            test_files_for_fold, # Use test set of the fold for validation during training
            batch_size
        )
    
    # 计算类别权重
    # weights_for_each_class = calc_class_weight(data_count)

    # 使用自定义权重替换计算的权重
    # weights_for_each_class = [1.0, 1.80, 1.0, 1.8, 1.20]
    # 从配置文件中获取类别权重，如果未指定则使用默认值
    weights_for_each_class = config['loss']['args'].get('class_weights', [1.2, 1.8, 1.4, 1.5, 1.6])
    logger.info(f"使用自定义类别权重 (for Trainer and base CE in TargetedMistakePenaltyLoss): {weights_for_each_class}")

    # 如果之前准备的是 TargetedMistakePenaltyLoss，现在实例化它
    if loss_name == "TargetedMistakePenaltyLoss" and criterion is None:
        penalty_args['class_weights'] = weights_for_each_class
        criterion = module_loss.TargetedMistakePenaltyLoss(**penalty_args)
        logger.info(f"Finalized TargetedMistakePenaltyLoss instantiation with class_weights.")
    elif criterion is None:
        logger.error(f"Criterion was not set for loss_name: {loss_name}. Check logic.")
        raise ValueError(f"Criterion not set for {loss_name}")

    # 创建训练器
    trainer = Trainer(model, criterion, metrics, optimizer,
                     config=config,
                     data_loader=data_loader,  # Training data for the current fold
                     fold_id=fold_id,
                     valid_data_loader=valid_data_loader, # Validation data for epoch monitoring (from current fold's test set)
                     test_data_loader=valid_data_loader,   # Test data for final evaluation of this fold (also from current fold's test set)
                     class_weights=weights_for_each_class)

    # 训练模型并返回结果
    return trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--config', default="config.json", type=str,
                      help='config file path (default: config.json)')
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
    parser.add_argument('--no-finetune', action='store_true',
                      help='跳过第二阶段的N1和REM微调')

    args = parser.parse_args()
    fold_id = int(args.fold_id)
    
    print(f"[DEBUG] train_Kfold_CV.py: args.device from command line: {args.device}")
    # Only set CUDA_VISIBLE_DEVICES if -d or --device is explicitly passed
    print(f"[DEBUG] train_Kfold_CV.py: args.device from command line: {args.device}")
    # Only set CUDA_VISIBLE_DEVICES if -d or --device is explicitly passed
    if args.device is not None:
        print(f"[DEBUG] train_Kfold_CV.py: Setting CUDA_VISIBLE_DEVICES to: {args.device} based on -d flag")
        print(f"[DEBUG] train_Kfold_CV.py: Setting CUDA_VISIBLE_DEVICES to: {args.device} based on -d flag")
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    
    print(f"[DEBUG] train_Kfold_CV.py: CUDA_VISIBLE_DEVICES before ConfigParser: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"[DEBUG] train_Kfold_CV.py: torch.cuda.device_count() before ConfigParser: {torch.cuda.device_count()}")
    print(f"[DEBUG] train_Kfold_CV.py: CUDA_VISIBLE_DEVICES before ConfigParser: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"[DEBUG] train_Kfold_CV.py: torch.cuda.device_count() before ConfigParser: {torch.cuda.device_count()}")
    config = ConfigParser.from_args(parser, fold_id)
    logger = config.get_logger('KFold_Main') # 为主脚本块添加logger
    print(f"[DEBUG] train_Kfold_CV.py: CUDA_VISIBLE_DEVICES after ConfigParser: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"[DEBUG] train_Kfold_CV.py: torch.cuda.device_count() after ConfigParser: {torch.cuda.device_count()}")
    print(f"[DEBUG] train_Kfold_CV.py: CUDA_VISIBLE_DEVICES after ConfigParser: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"[DEBUG] train_Kfold_CV.py: torch.cuda.device_count() after ConfigParser: {torch.cuda.device_count()}")
    
    # 加载数据
    num_folds_from_config = config["data_loader"]["args"]["num_folds"]
    if "shhs" in args.np_data_dir:
        # load_folds_data_shhs now returns dict {'train': [...], 'test': [...]} per fold
        folds_data = load_folds_data_shhs(args.np_data_dir, num_folds_from_config)
        # No global test_files
    else:
        # load_folds_data now returns dict {'train': [...], 'test': [...]} per fold
        folds_data = load_folds_data(args.np_data_dir, num_folds_from_config)
        # No global test_files

    # 第一阶段：正常训练
    print("=" * 50)
    print(f"开始第一阶段训练 (fold {fold_id})...")
    print("=" * 50)
    fold_metrics = main(config, fold_id)
    
    # 第二阶段：微调N1和REM
    if not args.no_finetune:
        print("=" * 50)
        print(f"准备第二阶段N1和REM微调 (fold {fold_id})...")
        print("=" * 50)
        # 获取第一阶段训练的最佳模型路径
        best_model_path = os.path.join(config.save_dir, f"{config['name']}_{fold_id}", "model_best.pth")
        if os.path.exists(best_model_path):
            print(f"找到最佳模型，开始针对N1和REM进行微调: {best_model_path}")
            finetune_metrics = finetune_model(config, fold_id, best_model_path, folds_data)
            # 合并微调后的指标
            fold_metrics.update({f"finetune_{k}": v for k, v in finetune_metrics.items()})
        else:
            print(f"警告: 未找到最佳模型 {best_model_path}，跳过微调阶段")
    else:
        print("用户选择跳过微调阶段 (--no-finetune)")
    
    # 保存当前这一折的最终指标 (确保这部分代码在 fold_metrics 最终确定之后)
    # 这一步是新增的，以确保每折的最佳结果被保存下来用于后续聚合
    metrics_dir_for_saving = os.path.join(config.save_dir, 'fold_metrics')
    os.makedirs(metrics_dir_for_saving, exist_ok=True)
    current_fold_metrics_path = os.path.join(metrics_dir_for_saving, f'fold_{fold_id}_metrics.json')
    try:
        with open(current_fold_metrics_path, 'w') as f:
            json.dump(fold_metrics, f, indent=4)
        logger.info(f"第 {fold_id} 折的最佳指标已保存到: {current_fold_metrics_path}")
    except Exception as e:
        logger.error(f"保存第 {fold_id} 折的指标到 {current_fold_metrics_path} 时发生错误: {e}")

    # 如果是最后一折，计算并保存所有折的平均分数和标准差
    if fold_id == config["data_loader"]["args"]["num_folds"] - 1:
        # 收集所有折的结果
        all_metrics_data = [] # 重命名以区分指标名称列表和实际数据
        metrics_dir = os.path.join(config.save_dir, 'fold_metrics')
        logger.info(f"开始聚合所有 {config['data_loader']['args']['num_folds']} 折的指标数据，从目录: {metrics_dir}")

        for i in range(config["data_loader"]["args"]["num_folds"]):
            metrics_file = os.path.join(metrics_dir, f'fold_{i}_metrics.json')
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    try:
                        fold_metrics_content = json.load(f)
                        all_metrics_data.append(fold_metrics_content)
                        logger.debug(f"成功加载 fold {i} 的指标: {metrics_file}")
                    except json.JSONDecodeError as e:
                        logger.error(f"无法解码JSON文件 {metrics_file}: {e}")
            else:
                logger.warning(f"指标文件未找到，跳过: {metrics_file}")
        
        if not all_metrics_data:
            logger.error("未能收集到任何折的指标数据，无法进行聚合统计。")

        # 获取所有出现过的指标名称 (从第一个有效数据中获取键)
        all_metric_keys = all_metrics_data[0].keys()

        aggregated_val_metrics = {}
        aggregated_test_metrics = {}

        logger.info("计算验证集指标的平均值和标准差...")
        for key in all_metric_keys:
            if not key.startswith('test_'):
                values = [fold_data.get(key) for fold_data in all_metrics_data if fold_data.get(key) is not None]
                if values:
                    aggregated_val_metrics[f"{key}_mean"] = np.mean(values)
                    aggregated_val_metrics[f"{key}_std"] = np.std(values)
                else:
                    logger.warning(f"验证集指标 '{key}' 在所有折中均无有效数据.")

        logger.info("计算测试集指标的平均值和标准差...")
        for key in all_metric_keys:
            if key.startswith('test_'):
                values = [fold_data.get(key) for fold_data in all_metrics_data if fold_data.get(key) is not None]
                if values:
                    aggregated_test_metrics[f"{key}_mean"] = np.mean(values)
                    aggregated_test_metrics[f"{key}_std"] = np.std(values)
                else:
                    logger.warning(f"测试集指标 '{key}' 在所有折中均无有效数据.")
        
        # 保存验证集平均分数和标准差
        val_metrics_path = os.path.join(config.save_dir, 'aggregated_validation_metrics.json')
        with open(val_metrics_path, 'w') as f:
            json.dump(aggregated_val_metrics, f, indent=4)
        logger.info(f"聚合后的验证集指标已保存到: {val_metrics_path}")
        
        # 保存测试集平均分数和标准差
        test_metrics_path = os.path.join(config.save_dir, 'aggregated_test_metrics.json')
        with open(test_metrics_path, 'w') as f:
            json.dump(aggregated_test_metrics, f, indent=4)
        logger.info(f"聚合后的测试集指标已保存到: {test_metrics_path}")
        
        # 打印验证集平均分数和标准差
        logger.info("========== 交叉验证聚合验证集指标 (Mean ± Std) ==========")
        for key in all_metric_keys:
            if not key.startswith('test_'):
                mean_val = aggregated_val_metrics.get(f"{key}_mean")
                std_val = aggregated_val_metrics.get(f"{key}_std")
                if mean_val is not None and std_val is not None:
                    logger.info(f"{key}: {mean_val:.4f} ± {std_val:.4f}")
            
        # 打印测试集平均分数和标准差
        logger.info("========== 交叉验证聚合测试集指标 (Mean ± Std) ==========")
        for key in all_metric_keys:
            if key.startswith('test_'):
                mean_val = aggregated_test_metrics.get(f"{key}_mean")
                std_val = aggregated_test_metrics.get(f"{key}_std")
                if mean_val is not None and std_val is not None:
                    logger.info(f"{key}: {mean_val:.4f} ± {std_val:.4f}")