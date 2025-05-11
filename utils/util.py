import json
from pathlib import Path
from collections import OrderedDict
from itertools import repeat
import pandas as pd
import os
import numpy as np
from glob import glob
import math

def load_folds_data_shhs(np_data_path, n_folds):
    files = sorted(glob(os.path.join(np_data_path, "*.npz")))
    r_p_path = r"utils/r_permute_shhs.npy"
    r_permute = np.load(r_p_path)
    npzfiles = np.asarray(files , dtype='<U200')[r_permute]
    train_files = np.array_split(npzfiles, n_folds)
    folds_data = {}
    for fold_id in range(n_folds):
        subject_files = train_files[fold_id]
        training_files = list(set(npzfiles) - set(subject_files))
        folds_data[fold_id] = [training_files, subject_files]
    return folds_data

def load_folds_data_correctly(np_data_path, n_folds):
    files = sorted(glob(os.path.join(np_data_path, "*.npz")))

    # Dynamically construct the path to the permutation file
    dataset_identifier = ""
    if "78" in np_data_path:
        dataset_identifier = "78"
    elif "20" in np_data_path: # Or any other logic to determine dataset type from path
        dataset_identifier = "20"
    else:
        # Fallback or error for unknown dataset type in path
        print(f"Warning: Could not determine dataset identifier from path {np_data_path} for load_folds_data_correctly, defaulting to '20' for permutation file.")
        dataset_identifier = "20" # Defaulting, consider making this stricter if necessary

    r_p_path = os.path.join("utils", "permutations", f"{n_folds}_fold", f"r_permute_{dataset_identifier}.npy")

    if os.path.exists(r_p_path):
        r_permute = np.load(r_p_path)
    else:
        # Updated error handling
        raise FileNotFoundError(f"Permutation file not found at {r_p_path} for load_folds_data_correctly. Please ensure it exists.")

    # 按主体组织文件
    files_dict = dict()
    for i in files:
        file_name = os.path.split(i)[-1] 
        file_num = file_name[3:5]
        if file_num not in files_dict:
            files_dict[file_num] = [i]
        else:
            files_dict[file_num].append(i)
            
    # 转换为列表并按r_permute打乱主体顺序
    files_pairs = []
    for key in files_dict:
        files_pairs.append(files_dict[key])
    files_pairs = np.array(files_pairs, dtype=object)
    files_pairs = files_pairs[r_permute]

    # 将主体分成n_folds组
    train_files = np.array_split(files_pairs, n_folds)
    folds_data = {}
    
    for fold_id in range(n_folds):
        # 当前折的主体文件作为验证集
        subject_files = train_files[fold_id]
        subject_files = [item for sublist in subject_files for item in sublist]
        
        # 所有其他主体文件作为训练集
        files_pairs2 = [item for sublist in files_pairs for item in sublist]
        training_files = list(set(files_pairs2) - set(subject_files))
        
        folds_data[fold_id] = [training_files, subject_files]
        
    return folds_data

def load_folds_data(np_data_path, n_folds, train_ratio=0.8, val_ratio=0.2):
    """
    按受试者级别划分数据集并进行K折交叉验证，避免数据泄露
    Args:
        np_data_path: 数据目录路径
        n_folds: 交叉验证折数
        train_ratio: 训练+验证集比例（占总数据的比例）
        val_ratio: 验证集比例（保留兼容性，不使用）
    Returns:
        folds_data: 包含每折训练集和验证集的字典
        test_files: 测试集文件列表
    """
    files = sorted(glob(os.path.join(np_data_path, "*.npz")))
    
    # Dynamically construct the path to the permutation file
    dataset_identifier = "78" if "78" in np_data_path else "20" # Assuming "20" is the only other option
    # It might be safer to explicitly check for "20" or raise an error if neither is found
    if "78" in np_data_path:
        dataset_identifier = "78"
    elif "20" in np_data_path: # Or any other logic to determine dataset type from path
        dataset_identifier = "20"
    else:
        # Fallback or error for unknown dataset type in path
        # For now, let's assume it's "20" if not "78", but this should be robust
        print(f"Warning: Could not determine dataset identifier from path {np_data_path}, defaulting to '20' for permutation file.")
        dataset_identifier = "20" 


    r_p_path = os.path.join("utils", "permutations", f"{n_folds}_fold", f"r_permute_{dataset_identifier}.npy")

    if os.path.exists(r_p_path):
        r_permute = np.load(r_p_path)
    else:
        print(f"============== ERROR: Permutation file not found at {r_p_path} ================= ")
        # Depending on requirements, you might want to raise an error here or fall back
        # to a default permutation or no permutation.
        # For now, the code will likely fail later if r_permute is not defined.
        # Consider adding: raise FileNotFoundError(f"Permutation file not found: {r_p_path}")
        # Or, generate a default permutation on the fly (though this loses reproducibility if file is expected)
        # For this example, let's assume the original error message and potential downstream failure is acceptable.
        # Fallback to old paths if new structure is not found, for backward compatibility (optional)
        # old_r_p_path_78 = r"utils/r_permute_78.npy"
        # old_r_p_path_20 = r"utils/r_permute_20.npy"
        # if "78" in np_data_path and os.path.exists(old_r_p_path_78):
        #     print(f"Warning: Using old permutation file {old_r_p_path_78}")
        #     r_permute = np.load(old_r_p_path_78)
        # elif "20" in np_data_path and os.path.exists(old_r_p_path_20):
        #     print(f"Warning: Using old permutation file {old_r_p_path_20}")
        #     r_permute = np.load(old_r_p_path_20)
        # else:
        #     print(f"============== ERROR: Permutation file not found at {r_p_path} and no fallback available ================= ")
        #     # This will cause an UnboundLocalError for r_permute if not handled.
        #     # It's better to raise an explicit error.
        raise FileNotFoundError(f"Permutation file not found at {r_p_path}. Please ensure it exists or adjust data loading logic.")


    # 按主体组织文件
    files_dict = dict()
    for i in files:
        file_name = os.path.split(i)[-1] 
        file_num = file_name[3:5]
        if file_num not in files_dict:
            files_dict[file_num] = [i]
        else:
            files_dict[file_num].append(i)
            
    # 转换为列表并按r_permute打乱主体顺序
    files_pairs = []
    for key in files_dict:
        files_pairs.append(files_dict[key])
    files_pairs = np.array(files_pairs, dtype=object)
    files_pairs = files_pairs[r_permute]
    
    # 按受试者级别分割训练+验证集和测试集
    train_val_size = int(len(files_pairs) * train_ratio)
    train_val_subjects = files_pairs[:train_val_size]
    test_subjects = files_pairs[train_val_size:]
    
    # 将测试集受试者的文件展平为列表
    test_files = []
    for subject_files in test_subjects:
        test_files.extend(subject_files)
    
    # 将训练+验证数据分成K折（受试者级别）
    train_files = np.array_split(train_val_subjects, n_folds)
    folds_data = {}
    
    for fold_id in range(n_folds):
        # 当前折的主体文件作为验证集
        val_subjects = train_files[fold_id]
        val_files = []
        for subject_files in val_subjects:
            val_files.extend(subject_files)
        
        # 其余折的主体文件作为训练集
        train_subjects = []
        for i in range(n_folds):
            if i != fold_id:
                train_subjects.extend(train_files[i])
        
        train_files_flat = []
        for subject_files in train_subjects:
            train_files_flat.extend(subject_files)
        
        # 使用与原函数相同的字典格式返回
        folds_data[fold_id] = {
            'train': train_files_flat,
            'val': val_files
        }
    
    return folds_data, test_files

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