import numpy as np
import os

# --- 配置 ---
# 1. 定义 permutation 文件存放的基础目录
base_output_dir = "utils/permutations"

# 2. 定义您的数据集信息
#    对于每个数据集，指定受试者总数和用于文件名的标识符
datasets_info = [
    {"id": "78", "num_subjects": 78},  # 例如，"78" 数据集有 78 个受试者
    {"id": "20", "num_subjects": 20},  # 例如，"20" 数据集有 20 个受试者
    # 如果您有其他数据集，请在此处添加
]

# 3. 定义您计划为哪些交叉验证折数生成 permutation 文件
folds_to_generate_for = [10, 20]  # 例如，为10折和20折都生成
# folds_to_generate_for = [20] # 或者如果您当前只想为20折生成

# --- 脚本执行 ---
if not os.path.exists(base_output_dir):
    os.makedirs(base_output_dir)
    print(f"Created base directory: {base_output_dir}")

for n_folds in folds_to_generate_for:
    fold_specific_dir = os.path.join(base_output_dir, f"{n_folds}_fold")
    if not os.path.exists(fold_specific_dir):
        os.makedirs(fold_specific_dir)
        print(f"Created fold-specific directory: {fold_specific_dir}")

    for dataset in datasets_info:
        dataset_id = dataset["id"]
        num_subjects = dataset["num_subjects"]

        # 生成随机排列的索引 (从 0 到 num_subjects-1)
        permutation_array = np.random.permutation(num_subjects)

        # 构建输出文件路径
        output_filename = f"r_permute_{dataset_id}.npy"
        output_path = os.path.join(fold_specific_dir, output_filename)

        # 保存 .npy 文件
        np.save(output_path, permutation_array)
        print(f"Generated and saved: {output_path} (contains {len(permutation_array)} indices)")

print("\nPermutation file generation complete.") 