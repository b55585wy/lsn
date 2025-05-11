import os
import numpy as np
import random
from glob import glob
# 确保导入的是我们修改过的 utils.util 中的函数
from utils.util import load_folds_data, load_folds_data_shhs 

# 设置随机种子，确保结果可重复
SEED = 42  # 固定的随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    print(f"已设置随机种子: {seed}")

def get_subject_id(file_path):
    """从文件路径中提取受试者ID"""
    file_name = os.path.split(file_path)[-1]
    # 假设ID在特定位置，例如 'SC4XX...' 中的 'XX'
    # 如果ID提取逻辑更复杂或文件名格式不同，需要调整此处
    if "shhs" in file_path.lower(): # SHHS数据集文件名格式可能不同
        # 示例：'shhs1-200001.npz' -> '200001' (如果这是受试者ID)
        # 这个提取逻辑需要根据SHHS文件名具体确定
        # 为简单起见，这里暂时用完整文件名（不含扩展名）代替，实际应用中需精确提取
        return os.path.splitext(file_name)[0] 
    return file_name[3:5]

def print_and_check_fold_split(fold_id, all_folds_data, np_data_path):
    """打印指定折的训练集和测试集文件，并检查ID重叠问题"""
    
    if fold_id not in all_folds_data:
        print(f"错误: fold_id {fold_id} 不在 all_folds_data 中")
        return False
    
    current_fold_files = all_folds_data[fold_id]
    train_files = current_fold_files['train']
    test_files = current_fold_files['test'] # 每一折都有自己的测试集
    
    train_ids = set(get_subject_id(f) for f in train_files)
    test_ids = set(get_subject_id(f) for f in test_files) # 当前折的测试集ID
    
    print(f"\n============= 折 {fold_id} (从路径 {np_data_path}) =============")
    
    print(f"\n--- 训练集 ({len(train_files)} 文件, {len(train_ids)} 受试者) ---")
    # 为了避免打印过多文件，可以只打印前几个或受试者列表
    # print("训练文件列表:")
    # for f_idx, f_path in enumerate(train_files):
    #     print(f"  {f_idx+1}. {f_path}")
    #     if f_idx >= 4 and len(train_files) > 10 : # 只打印前5个和后5个（如果文件多）
    #         if f_idx == 5 and len(train_files) > 10: print("  ...")
    #         if f_idx >= len(train_files) -5: print(f"  {f_idx+1}. {f_path}")
    #         elif f_idx == len(train_files) -6 : continue # skip printing ... again
    print(f"训练集受试者IDs: {sorted(list(train_ids))}")


    print(f"\n--- 测试集 ({len(test_files)} 文件, {len(test_ids)} 受试者) ---")
    # print("测试文件列表:")
    # for f_idx, f_path in enumerate(test_files):
    #     print(f"  {f_idx+1}. {f_path}")
    #     if f_idx >= 4 and len(test_files) > 10: # 只打印前5个和后5个
    #         if f_idx == 5 and len(test_files) > 10: print("  ...")
    #         if f_idx >= len(test_files) -5: print(f"  {f_idx+1}. {f_path}")
    #         elif f_idx == len(test_files) -6 : continue
    print(f"测试集受试者IDs: {sorted(list(test_ids))}")
    
    # K-fold交叉验证的核心是训练集和测试集在当前折是互斥的
    # 检查当前折的训练集和测试集之间是否有ID重叠
    train_current_test_overlap = train_ids.intersection(test_ids)
    
    print("\n重叠检查:")
    if train_current_test_overlap:
        print(f"警告! 当前折的训练集和测试集有 {len(train_current_test_overlap)} 个重叠ID: {sorted(list(train_current_test_overlap))}")
        # 这个警告理论上不应该出现，如果出现了说明 load_folds_data 实现有问题
    else:
        print("✓ 当前折的训练集和测试集没有ID重叠 (符合预期)")
    
    is_valid_split_for_fold = not bool(train_current_test_overlap)
    if is_valid_split_for_fold:
        print("\n✓ 当前折数据划分正确。")
    else:
        print("\n✗ 当前折数据划分存在问题: 训练集与测试集有重叠！")
    
    return is_valid_split_for_fold

def show_all_fold_splits(np_data_path, n_folds):
    """加载数据并为所有折打印训练/测试集文件和ID"""
    print(f"\n===== 正在加载和分析数据集: {np_data_path} (配置 {n_folds} 折) =====")
    
    # 根据数据路径选择加载函数
    if "shhs" in np_data_path.lower():
        all_folds_data = load_folds_data_shhs(np_data_path, n_folds)
    else:
        all_folds_data = load_folds_data(np_data_path, n_folds)
        
    if not all_folds_data:
        print("错误：未能加载到数据。")
        return

    overall_validity = True
    for fold_id in range(n_folds):
        print(f"\n--- 正在处理折 {fold_id} ---")
        fold_is_valid = print_and_check_fold_split(fold_id, all_folds_data, np_data_path)
        if not fold_is_valid:
            overall_validity = False
            
    print("\n=============== 整体检查汇总 =================")
    if overall_validity:
        print(f"✓ 所有 {n_folds} 折的数据划分均符合预期 (训练集和对应测试集无重叠)。")
    else:
        print(f"✗ 注意！在 {n_folds} 折交叉验证中，部分折的训练集和其对应的测试集存在ID重叠。请检查 `load_folds_data` 或 `load_folds_data_shhs` 函数逻辑。")


def main():
    # 设置随机种子
    set_seed(SEED)
    
    # 定义要检查的数据路径和折数组合
    # 您可以修改这些路径和折数来检查不同的配置
    configurations = [
        # {"path": "/hpc2hdd/home/ywang183/biosleep_3/data20/data20npy", "folds": 10},
        # {"path": "/hpc2hdd/home/ywang183/biosleep_3/data20/data20npy", "folds": 20},
        # {"path": "/hpc2hdd/home/ywang183/biosleep_3/data78/processed", "folds": 10},
        {"path": "/hpc2hdd/home/ywang183/biosleep_3/data78/processed", "folds": 20},
        # 如果有SHHS数据，也可以添加对应的路径
        # {"path": "/path/to/your/shhs_data_npy", "folds": 5}, # 示例
    ]
    
    for config in configurations:
        show_all_fold_splits(config["path"], config["folds"])

if __name__ == "__main__":
    main()