import json
from pathlib import Path
from collections import OrderedDict
from itertools import repeat
import pandas as pd
import os
import numpy as np
from glob import glob
import math
import torch

def load_folds_data_shhs(np_data_path, n_folds):
    """
    按受试者级别划分SHHS数据集并进行K折交叉验证，避免数据泄露。
    Args:
        np_data_path: 数据目录路径
        n_folds: 交叉验证折数
    Returns:
        folds_data: 包含每折训练集和测试集的字典
    """
    files = sorted(glob(os.path.join(np_data_path, "*.npz")))
    
    # 动态构建置换文件路径
    r_p_path = os.path.join("utils", "permutations", f"{n_folds}_fold", "r_permute_shhs.npy")

    if os.path.exists(r_p_path):
        r_permute = np.load(r_p_path)
    else:
        raise FileNotFoundError(f"置换文件未找到: {r_p_path}. 请确保它存在或调整数据加载逻辑。")

    # 按主体组织文件
    files_dict = dict()
    for i in files:
        file_name = os.path.split(i)[-1]
        # 对于SHHS文件，受试者ID通常在文件名中，例如 shhs1-200001.npz
        # 提取 '200001' 部分作为受试者ID
        file_num = file_name[6:12]
        if file_num not in files_dict:
            files_dict[file_num] = [i]
        else:
            files_dict[file_num].append(i)
            
    # 转换为列表并按r_permute打乱主体顺序
    subject_level_files_grouped = []
    for key in files_dict:
        subject_level_files_grouped.append(files_dict[key])
    subject_level_files_grouped = np.array(subject_level_files_grouped, dtype=object)
    subject_level_files_grouped = subject_level_files_grouped[r_permute] # Permute subjects
    
    # 将所有受试者分成K折
    subject_folds_split = np.array_split(subject_level_files_grouped, n_folds)
    folds_data = {}
    
    for fold_id in range(n_folds):
        # 当前折的主体文件作为测试集
        test_subject_groups = subject_folds_split[fold_id]
        test_files_flat = []
        for subject_files_list in test_subject_groups:
            test_files_flat.extend(subject_files_list)
        
        # 其余折的主体文件作为训练集
        train_subject_groups = []
        for i in range(n_folds):
            if i != fold_id:
                train_subject_groups.extend(subject_folds_split[i])
        
        train_files_flat = []
        for subject_files_list in train_subject_groups:
            train_files_flat.extend(subject_files_list)
        
        folds_data[fold_id] = {
            'train': train_files_flat,
            'test': test_files_flat
        }
    
    return folds_data


def load_folds_data(np_data_path, n_folds):
    """
    按受试者级别划分数据集并进行K折交叉验证，避免数据泄露
    Each fold serves as a test set, and the remaining K-1 folds as training set.
    Args:
        np_data_path: 数据目录路径
        n_folds: 交叉验证折数
    Returns:
        folds_data: 包含每折训练集和测试集的字典
    """
    files = sorted(glob(os.path.join(np_data_path, "*.npz")))
    
    # Dynamically construct the path to the permutation file
    dataset_identifier = ""
    if "SHHS" in np_data_path.upper(): # Check for "SHHS" (case-insensitive)
        dataset_identifier = "shhs"
    elif "78" in np_data_path:
        dataset_identifier = "78"
    elif "20" in np_data_path:
        dataset_identifier = "20"
    else:
        print(f"Warning: Could not determine dataset identifier from path {np_data_path}, defaulting to '20' for permutation file.")
        dataset_identifier = "20"

    r_p_path = os.path.join("utils", "permutations", f"{n_folds}_fold", f"r_permute_{dataset_identifier}.npy")

    if os.path.exists(r_p_path):
        r_permute = np.load(r_p_path)
    else:
        raise FileNotFoundError(f"Permutation file not found at {r_p_path}. Please ensure it exists or adjust data loading logic.")

    # 按主体组织文件
    files_dict = dict()
    for i in files:
        file_name = os.path.split(i)[-1] 
        if dataset_identifier == "shhs":
            file_num = file_name[6:12] # For SHHS files like shhs1-200001.npz
        else: # Assuming "20" or "78" for PhysioNet-like datasets
            file_num = file_name[3:5] # For PhysioNet files like SC4001E0.npz, subject ID is 4001
        if file_num not in files_dict:
            files_dict[file_num] = [i]
        else:
            files_dict[file_num].append(i)
            
    # 转换为列表并按r_permute打乱主体顺序
    subject_level_files_grouped = []
    for key in files_dict:
        subject_level_files_grouped.append(files_dict[key])
    subject_level_files_grouped = np.array(subject_level_files_grouped, dtype=object)
    subject_level_files_grouped = subject_level_files_grouped[r_permute] # Permute subjects
    
    # 将所有受试者分成K折
    subject_folds_split = np.array_split(subject_level_files_grouped, n_folds)
    folds_data = {}
    
    for fold_id in range(n_folds):
        # 当前折的主体文件作为测试集
        test_subject_groups = subject_folds_split[fold_id]
        test_files_flat = []
        for subject_files_list in test_subject_groups:
            test_files_flat.extend(subject_files_list)
        
        # 其余折的主体文件作为训练集
        train_subject_groups = []
        for i in range(n_folds):
            if i != fold_id:
                train_subject_groups.extend(subject_folds_split[i])
        
        train_files_flat = []
        for subject_files_list in train_subject_groups:
            train_files_flat.extend(subject_files_list)
        
        folds_data[fold_id] = {
            'train': train_files_flat,
            'test': test_files_flat  # Changed 'val' to 'test' and ensure correct data source
        }
    
    return folds_data # Return only folds_data, no global test_files

def calc_class_weight(labels_count):
    total = np.sum(labels_count)
    class_weight = dict()
    num_classes = len(labels_count)

    factor = 1 / num_classes
    mu = [factor * 1.5, factor * 2, factor * 1.5, factor, factor * 1.5] # THESE CONFIGS ARE FOR SLEEP-EDF-20 ONLY

    for key in range(num_classes):
        score = math.log(mu[key] * total / float(labels_count[key]))
        class_weight[key] = score if score > 1.0 else 1.0
        class_weight[key] = round(class_weight[key] * mu[key], 2)

    class_weight = [class_weight[i] for i in range(num_classes)]

    return class_weight


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data.loc[:, col] = 0

    def update(self, key, value, n=1):
        # Skip update if value is None
        if value is None:
            return
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.loc[key, 'total'] += value * n
        self._data.loc[key, 'counts'] += n
        self._data.loc[key, 'average'] = self._data.loc[key, 'total'] / self._data.loc[key, 'counts']

    def avg(self, key):
        return self._data.loc[key, 'average']

    def result(self):
        return dict(self._data['average'])

def prepare_device(n_gpu_use):
    """
    设置GPU设备，如果请求的GPU数量大于可用数量，则降低请求数量
    
    Args:
        n_gpu_use (int): GPU设备数量，None代表使用全部可用设备
    
    Returns:
        device (torch.device): 主设备
        list_ids (list): 活跃GPU ID列表
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("警告: 请求的GPU不可用，使用CPU代替")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"警告: 请求的GPU数量{n_gpu_use}超过可用数量{n_gpu}，仅使用{n_gpu}个GPU")
        n_gpu_use = n_gpu
    
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    
    return device, list_ids