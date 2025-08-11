import torch
from torch.utils.data import Dataset
import os
import numpy as np
from model.augmentation import Compose, RandomNoise, RandomScaling # 导入增强类

class DualModalityDataset(Dataset):
    def __init__(self, np_dataset):
        super(DualModalityDataset, self).__init__()

        # 加载第一个文件的数据
        data = np.load(np_dataset[0])
        X_train_eeg = data["x"]
        X_train_eog = data["x_eog"]
        y_train = data["y"]

        # 加载剩余文件的数据并合并
        for np_file in np_dataset[1:]:
            data = np.load(np_file)
            X_train_eeg = np.vstack((X_train_eeg, data["x"]))
            X_train_eog = np.vstack((X_train_eog, data["x_eog"]))
            y_train = np.append(y_train, data["y"])

        self.len = X_train_eeg.shape[0]
        
        # 转换为PyTorch张量
        self.x_eeg = torch.from_numpy(X_train_eeg).float()
        self.x_eog = torch.from_numpy(X_train_eog).float()
        self.y_data = torch.from_numpy(y_train).long()

        # 调整EEG数据维度
        if len(self.x_eeg.shape) == 3:
            if self.x_eeg.shape[1] != 1:
                self.x_eeg = self.x_eeg.permute(0, 2, 1)
        else:
            self.x_eeg = self.x_eeg.unsqueeze(1)
            
        # 调整EOG数据维度
        if len(self.x_eog.shape) == 3:
            if self.x_eog.shape[1] != 1:
                self.x_eog = self.x_eog.permute(0, 2, 1)
        else:
            self.x_eog = self.x_eog.unsqueeze(1)

    def __getitem__(self, index):
        return self.x_eeg[index], self.x_eog[index], self.y_data[index]

    def __len__(self):
        return self.len

# 新增序列数据集类
class SequentialEpochDataset(Dataset):
    def __init__(self, np_dataset, seq_length=5, stride=1, transform=None):
        """
        加载连续多个epoch的数据集
        
        参数:
            np_dataset: 数据文件列表
            seq_length: 要加载的连续epoch数量
            stride: 滑动窗口的步长
            transform: 要应用于数据的转换 (数据增强)
        """
        super(SequentialEpochDataset, self).__init__()
        self.transform = transform # 添加这一行
        
        # 加载所有文件数据并记录边界
        all_eeg = []
        all_eog = []
        all_labels = []
        file_boundaries = [0]  # 记录每个文件的起始索引
        
        # 加载第一个文件的数据
        data = np.load(np_dataset[0])
        X_train_eeg = data["x"]
        X_train_eog = data["x_eog"] if "x_eog" in data else data["x"]  # 兼容只有EEG的数据
        y_train = data["y"]
        
        all_eeg.append(X_train_eeg)
        all_eog.append(X_train_eog)
        all_labels.append(y_train)
        file_boundaries.append(len(y_train))
        
        # 加载剩余文件的数据
        for np_file in np_dataset[1:]:
            data = np.load(np_file)
            X_train_eeg = data["x"]
            X_train_eog = data["x_eog"] if "x_eog" in data else data["x"]
            y_train = data["y"]
            
            all_eeg.append(X_train_eeg)
            all_eog.append(X_train_eog)
            all_labels.append(y_train)
            file_boundaries.append(file_boundaries[-1] + len(y_train))
        
        # 合并所有数据
        self.x_eeg_all = np.vstack(all_eeg)
        self.x_eog_all = np.vstack(all_eog)
        self.y_all = np.hstack(all_labels)
        self.file_boundaries = file_boundaries
        
        # 创建有效序列索引
        self.valid_indices = []
        for i in range(len(file_boundaries)-1):
            start_idx = file_boundaries[i]
            end_idx = file_boundaries[i+1] - seq_length + 1
            if end_idx > start_idx:  # 确保文件长度足够
                self.valid_indices.extend(range(start_idx, end_idx, stride))
        
        self.seq_length = seq_length
        
        # 转换为PyTorch张量
        self.x_eeg_all = torch.from_numpy(self.x_eeg_all).float()
        self.x_eog_all = torch.from_numpy(self.x_eog_all).float()
        self.y_all = torch.from_numpy(self.y_all).long()
        
        # 调整维度 (N, C, L)
        if len(self.x_eeg_all.shape) == 3:
            if self.x_eeg_all.shape[1] != 1:
                self.x_eeg_all = self.x_eeg_all.permute(0, 2, 1)
        else:
            self.x_eeg_all = self.x_eeg_all.unsqueeze(1)
            
        if len(self.x_eog_all.shape) == 3:
            if self.x_eog_all.shape[1] != 1:
                self.x_eog_all = self.x_eog_all.permute(0, 2, 1)
        else:
            self.x_eog_all = self.x_eog_all.unsqueeze(1)

    def __getitem__(self, index):
        # 获取原始索引
        orig_idx = self.valid_indices[index]
        
        # 提取序列
        eeg_seq = torch.stack([self.x_eeg_all[orig_idx + i] for i in range(self.seq_length)])
        eog_seq = torch.stack([self.x_eog_all[orig_idx + i] for i in range(self.seq_length)])
        label_seq = torch.stack([self.y_all[orig_idx + i] for i in range(self.seq_length)])
        
        sample = {'eeg': eeg_seq, 'eog': eog_seq, 'label': label_seq}

        if self.transform:
            sample = self.transform(sample)
        
        return sample['eeg'], sample['eog'], sample['label']

    def __len__(self):
        return len(self.valid_indices)

# 保留原来的单模态数据集类以保持向后兼容性
class LoadDataset_from_numpy(Dataset):
    def __init__(self, np_dataset):
        super(LoadDataset_from_numpy, self).__init__()

        X_train = np.load(np_dataset[0])["x"]
        y_train = np.load(np_dataset[0])["y"]

        for np_file in np_dataset[1:]:
            X_train = np.vstack((X_train, np.load(np_file)["x"]))
            y_train = np.append(y_train, np.load(np_file)["y"])

        self.len = X_train.shape[0]
        self.x_data = torch.from_numpy(X_train).float()
        self.y_data = torch.from_numpy(y_train).long()

        if len(self.x_data.shape) == 3:
            if self.x_data.shape[1] != 1:
                self.x_data = self.x_data.permute(0, 2, 1)
        else:
            self.x_data = self.x_data.unsqueeze(1)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


def data_generator_np(train_files, val_files, batch_size):
    """
    生成训练集和验证集的数据加载器
    Args:
        train_files: 训练集文件列表
        val_files: 验证集文件列表
        batch_size: 批次大小
    """
    train_dataset = LoadDataset_from_numpy(train_files)
    val_dataset = LoadDataset_from_numpy(val_files)

    # 计算类别权重
    all_ys = np.concatenate((train_dataset.y_data, val_dataset.y_data))
    all_ys = all_ys.tolist()
    num_classes = len(np.unique(all_ys))
    counts = [all_ys.count(i) for i in range(num_classes)]

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             drop_last=False,
                                             num_workers=4)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                           batch_size=batch_size,
                                           shuffle=False,
                                           drop_last=False,
                                           num_workers=4)

    return train_loader, val_loader, counts

def data_generator_np_dual(train_files, val_files, batch_size):
    """
    生成训练集和验证集的双模态数据加载器
    Args:
        train_files: 训练集文件列表
        val_files: 验证集文件列表
        batch_size: 批次大小
    """
    train_dataset = DualModalityDataset(train_files)
    val_dataset = DualModalityDataset(val_files)

    # 计算类别权重
    all_ys = np.concatenate((train_dataset.y_data, val_dataset.y_data))
    all_ys = all_ys.tolist()
    num_classes = len(np.unique(all_ys))
    counts = [all_ys.count(i) for i in range(num_classes)]

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             drop_last=False,
                                             num_workers=4)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                           batch_size=batch_size,
                                           shuffle=False,
                                           drop_last=False,
                                           num_workers=4)

    return train_loader, val_loader, counts

# 新增序列数据加载函数
def data_generator_np_sequence(train_files, val_files, batch_size, seq_length=5, stride=1, transform=None):
    """
    生成训练集和验证集的序列数据加载器
    Args:
        train_files: 训练集文件列表
        val_files: 验证集文件列表
        batch_size: 批次大小
        seq_length: 序列长度
        stride: 滑动窗口步长
        transform: 要应用于训练数据的转换 (数据增强)
    """
    train_dataset = SequentialEpochDataset(train_files, seq_length, stride, transform=transform)
    val_dataset = SequentialEpochDataset(val_files, seq_length, stride) # 验证集通常不应用增强

    # 计算类别权重 - 使用所有标签
    all_ys = train_dataset.y_all.numpy().tolist() + val_dataset.y_all.numpy().tolist()
    num_classes = len(np.unique(all_ys))
    counts = [all_ys.count(i) for i in range(num_classes)]

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             drop_last=False,
                                             num_workers=4)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                           batch_size=batch_size,
                                           shuffle=False,
                                           drop_last=False,
                                           num_workers=4)

    return train_loader, val_loader, counts


# for epilepsy dataset
class EpilepsyDataset(Dataset):
    def __init__(self, data_dir, seq_length=1, stride=1):
        super(EpilepsyDataset, self).__init__()
        self.seq_length = seq_length
        self.stride = stride
        
        all_files = []
        # 遍历 Z, O, N, F, S 文件夹
        for folder in ['Z', 'O', 'N', 'F', 'S']:
            folder_path = os.path.join(data_dir, folder)
            if os.path.isdir(folder_path):
                files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.txt')]
                all_files.extend(files)

        # 如果在根目录，直接加载
        if not all_files:
            all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.txt')]

        data = []
        labels = []
        
        # 定义标签映射
        label_map = {'Z': 0, 'O': 0, 'N': 0, 'F': 0, 'S': 1}

        for file_path in all_files:
            # 从文件名或路径中提取类别
            filename = os.path.basename(file_path)
            # 类别是文件名的第一个字母
            class_char = filename[0].upper()
            label = label_map.get(class_char, -1) # 如果找不到，默认为-1

            if label != -1:
                # 读取每个txt文件，每行是一个整数
                with open(file_path, 'r') as f:
                    # 将所有行读入一个列表，并转换为整数
                    raw_data = [int(line.strip()) for line in f.readlines()]
                    # 将整个文件作为一个样本
                    data.append(np.array(raw_data, dtype=np.float32))
                    labels.append(label)

        self.x_data = data
        self.y_data = torch.tensor(labels, dtype=torch.long)
        
        # 创建有效序列索引
        self.valid_indices = []
        for i in range(len(self.x_data)):
            # 这里的逻辑简化为每个文件是一个样本，不进行滑动窗口
            self.valid_indices.append(i)

    def __getitem__(self, index):
        orig_idx = self.valid_indices[index]
        
        # 获取数据和标签
        sample_data = self.x_data[orig_idx]
        sample_label = self.y_data[orig_idx]
        
        # 转换为Tensor
        x_tensor = torch.from_numpy(sample_data).float().unsqueeze(0) # 增加一个通道维度 (1, 4096)
        
        return x_tensor, sample_label

    def __len__(self):
        return len(self.valid_indices)


def epilepsy_data_generator(data_dir, batch_size, seq_length=1, stride=1):
    """
    为Bonn数据集生成数据加载器
    """
    dataset = EpilepsyDataset(data_dir, seq_length, stride)
    
    # 计算类别权重
    all_ys = dataset.y_data.numpy().tolist()
    num_classes = len(np.unique(all_ys))
    counts = [all_ys.count(i) for i in range(num_classes)]

    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             drop_last=False,
                                             num_workers=4)

    return data_loader, counts
