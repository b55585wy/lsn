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
    if "78" in np_data_path:
        r_p_path = r"utils/r_permute_78.npy"
    else:
        r_p_path = r"utils/r_permute_20.npy"

    if os.path.exists(r_p_path):
        r_permute = np.load(r_p_path)
    else:
        print("============== ERROR =================")

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
    按比例划分数据集并进行K折交叉验证
    Args:
        np_data_path: 数据目录路径
        n_folds: 交叉验证折数
        train_ratio: 训练集比例（包含验证集）
        val_ratio: 验证集比例（从训练集中划分）
    Returns:
        folds_data: 包含每折训练集和验证集的字典
        test_files: 测试集文件列表
    """
    files = sorted(glob(os.path.join(np_data_path, "*.npz")))
    if "78" in np_data_path:
        r_p_path = r"utils/r_permute_78.npy"
    else:
        r_p_path = r"utils/r_permute_20.npy"

    if os.path.exists(r_p_path):
        r_permute = np.load(r_p_path)
    else:
        print("============== ERROR =================")

    # 按subject组织文件
    files_dict = dict()
    for i in files:
        file_name = os.path.split(i)[-1] 
        file_num = file_name[3:5]
        if file_num not in files_dict:
            files_dict[file_num] = [i]
        else:
            files_dict[file_num].append(i)
    
    # 转换为列表并随机打乱
    files_pairs = []
    for key in files_dict:
        files_pairs.append(files_dict[key])
    
    # 使用dtype=object来避免警告
    files_pairs = np.array(files_pairs, dtype=object)
    files_pairs = files_pairs[r_permute]

    # 将所有文件展平为一个列表
    all_files = [item for sublist in files_pairs for item in sublist]
    
    # 按比例划分训练集和测试集
    train_val_size = int(len(all_files) * train_ratio)
    train_val_files = all_files[:train_val_size]
    test_files = all_files[train_val_size:]  # 测试集
    
    # 将训练验证数据分成K折
    folds_data = {}
    fold_size = len(train_val_files) // n_folds
    
    for fold_id in range(n_folds):
        # 当前折的验证集
        val_start = fold_id * fold_size
        val_end = val_start + fold_size
        val_files = train_val_files[val_start:val_end]
        
        # 训练集（剩余的训练数据）
        train_fold_files = train_val_files[:val_start] + train_val_files[val_end:]
        
        folds_data[fold_id] = {
            'train': train_fold_files,
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