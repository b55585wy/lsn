import os
import numpy as np
import random
from glob import glob
from utils.util import load_folds_data  # 使用修改后的load_folds_data函数

# 设置随机种子，确保结果可重复
SEED = 42  # 固定的随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    print(f"已设置随机种子: {seed}")

def get_subject_id(file_path):
    """从文件路径中提取受试者ID"""
    file_name = os.path.split(file_path)[-1]
    return file_name[3:5]  # 提取主体ID (SC4XXX中的XX部分)

def check_data_split(fold_id, np_data_path, n_folds=10):
    """检查数据划分是否存在ID重叠问题"""
    # 加载数据划分
    folds_data, test_files = load_folds_data(np_data_path, n_folds)
    
    if fold_id not in folds_data:
        print(f"错误: fold_id {fold_id} 不在数据中")
        return False
    
    # 获取训练、验证和测试数据的IDs
    train_files = folds_data[fold_id]['train']
    val_files = folds_data[fold_id]['val']
    
    train_ids = set(get_subject_id(f) for f in train_files)
    val_ids = set(get_subject_id(f) for f in val_files)
    test_ids = set(get_subject_id(f) for f in test_files)
    
    # 检查重叠
    train_val_overlap = train_ids.intersection(val_ids)
    train_test_overlap = train_ids.intersection(test_ids)
    val_test_overlap = val_ids.intersection(test_ids)
    
    print(f"\n============= 折 {fold_id} 检查结果 =============")
    print(f"训练集包含 {len(train_ids)} 个不同主体: {sorted(list(train_ids))}")
    print(f"验证集包含 {len(val_ids)} 个不同主体: {sorted(list(val_ids))}")
    print(f"测试集包含 {len(test_ids)} 个不同主体: {sorted(list(test_ids))}")
    
    print("\n重叠检查:")
    if train_val_overlap:
        print(f"警告! 训练集和验证集有 {len(train_val_overlap)} 个重叠ID: {sorted(list(train_val_overlap))}")
    else:
        print("✓ 训练集和验证集没有ID重叠")
    
    if train_test_overlap:
        print(f"警告! 训练集和测试集有 {len(train_test_overlap)} 个重叠ID: {sorted(list(train_test_overlap))}")
    else:
        print("✓ 训练集和测试集没有ID重叠")
        
    if val_test_overlap:
        print(f"警告! 验证集和测试集有 {len(val_test_overlap)} 个重叠ID: {sorted(list(val_test_overlap))}")
    else:
        print("✓ 验证集和测试集没有ID重叠")
    
    is_valid = not (train_val_overlap or train_test_overlap or val_test_overlap)
    if is_valid:
        print("\n✓ 数据划分正确: 没有ID重叠")
    else:
        print("\n✗ 数据划分存在问题: 发现ID重叠")
    
    return is_valid

def find_valid_folds(np_data_path, n_folds=10):
    """找出所有没有ID重叠问题的折"""
    valid_folds = []
    for fold_id in range(n_folds):
        print(f"\n正在检查折 {fold_id}...")
        is_valid = check_data_split(fold_id, np_data_path, n_folds)
        if is_valid:
            valid_folds.append(fold_id)
    
    print("\n=============== 结果汇总 ===============")
    if valid_folds:
        print(f"✓ 找到 {len(valid_folds)} 个没有ID重叠问题的折: {valid_folds}")
    else:
        print("✗ 所有折都存在ID重叠问题")
    
    return valid_folds

def main():
    # 设置随机种子
    set_seed(SEED)
    
    # 使用正确的数据路径
    data_paths = [
        "/root/autodl-fs/data20/data20npy",
        "/root/autodl-fs/data78/processed"
    ]
    n_folds = 10
    
    for data_path in data_paths:
        print(f"\n===== 分析数据集: {data_path} =====")
        
        # 找出所有没有问题的折
        valid_folds = find_valid_folds(data_path, n_folds)

if __name__ == "__main__":
    main()