import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from imblearn.over_sampling import SMOTE
import logging

logger = logging.getLogger("smote_loader")

class SMOTEDualModalityDataset(Dataset):
    def __init__(self, np_dataset, apply_smote=True, random_state=42, k_neighbors=5, 
                 sampling_strategy='auto', n1_target=8000, n3_target=8000):
        """
        使用SMOTE技术增强的双模态数据集
        
        参数:
            np_dataset: 数据文件列表
            apply_smote: 是否应用SMOTE增强
            random_state: 随机种子
            k_neighbors: SMOTE算法的近邻数
            sampling_strategy: SMOTE采样策略，可以是'auto'、'all'或具体的采样数量字典
            n1_target: 当采用具体数量时，N1类别的目标样本数
            n3_target: 当采用具体数量时，N3类别的目标样本数
        """
        super(SMOTEDualModalityDataset, self).__init__()

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
        
        # 原始数据统计
        original_class_counts = {}
        for cls in range(5):  # 假设有5个类别
            original_class_counts[cls] = np.sum(y_train == cls)
        
        logger.info(f"原始数据类别分布: {original_class_counts}")
        
        # 如果需要应用SMOTE
        if apply_smote:
            # 重塑EEG和EOG以适应SMOTE
            eeg_shape = X_train_eeg.shape
            eog_shape = X_train_eog.shape
            
            X_train_eeg_flat = X_train_eeg.reshape(eeg_shape[0], -1)
            X_train_eog_flat = X_train_eog.reshape(eog_shape[0], -1)
            
            # 合并EEG和EOG以便同时进行SMOTE处理
            X_combined = np.hstack((X_train_eeg_flat, X_train_eog_flat))
            
            # 确定采样策略
            if sampling_strategy == 'auto' or sampling_strategy == 'all':
                # 使用auto或all策略
                smote_strategy = sampling_strategy
                logger.info(f"使用SMOTE '{smote_strategy}'策略进行数据增强")
            else:
                # 使用具体数量策略
                smote_strategy = {}
                
                # 对N1类进行过采样
                if 1 in original_class_counts and original_class_counts[1] < n1_target:
                    smote_strategy[1] = n1_target
                
                # 对N3类进行过采样
                if 3 in original_class_counts and original_class_counts[3] < n3_target:
                    smote_strategy[3] = n3_target
                
                logger.info(f"使用SMOTE具体数量策略进行数据增强: {smote_strategy}")
            
            # 应用SMOTE
            try:
                smote = SMOTE(
                    sampling_strategy=smote_strategy, 
                    random_state=random_state, 
                    k_neighbors=min(k_neighbors, original_class_counts[1]-1) # 确保k_neighbors不超过最小类别样本数-1
                )
                
                X_combined_resampled, y_train_resampled = smote.fit_resample(X_combined, y_train)
                
                # 分离EEG和EOG
                eeg_size = eeg_shape[1] * eeg_shape[2] if len(eeg_shape) > 2 else eeg_shape[1]
                
                X_train_eeg_resampled = X_combined_resampled[:, :eeg_size]
                X_train_eog_resampled = X_combined_resampled[:, eeg_size:]
                
                # 恢复原始形状
                if len(eeg_shape) > 2:
                    X_train_eeg_resampled = X_train_eeg_resampled.reshape(-1, eeg_shape[1], eeg_shape[2])
                else:
                    X_train_eeg_resampled = X_train_eeg_resampled.reshape(-1, eeg_shape[1])
                    
                if len(eog_shape) > 2:
                    X_train_eog_resampled = X_train_eog_resampled.reshape(-1, eog_shape[1], eog_shape[2])
                else:
                    X_train_eog_resampled = X_train_eog_resampled.reshape(-1, eog_shape[1])
                
                # 更新数据
                X_train_eeg = X_train_eeg_resampled
                X_train_eog = X_train_eog_resampled
                y_train = y_train_resampled
                
                # SMOTE后数据统计
                resampled_class_counts = {}
                for cls in range(5):  # 假设有5个类别
                    resampled_class_counts[cls] = np.sum(y_train == cls)
                
                logger.info(f"SMOTE后数据类别分布: {resampled_class_counts}")
                
            except Exception as e:
                logger.error(f"SMOTE处理失败: {e}")
                logger.info("使用原始数据继续...")

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


def smote_data_generator_np_dual(train_files, val_files, batch_size, 
                                apply_smote=True, 
                                random_state=42, 
                                k_neighbors=5, 
                                sampling_strategy='auto',
                                n1_target=8000,
                                n3_target=8000):
    """
    使用SMOTE技术生成训练集和验证集的双模态数据加载器
    Args:
        train_files: 训练集文件列表
        val_files: 验证集文件列表
        batch_size: 批次大小
        apply_smote: 是否应用SMOTE (仅适用于训练集)
        random_state: 随机种子
        k_neighbors: SMOTE算法的近邻数
        sampling_strategy: SMOTE采样策略
        n1_target: N1类别目标样本数
        n3_target: N3类别目标样本数
    """
    # 对训练集使用SMOTE增强
    train_dataset = SMOTEDualModalityDataset(
        train_files, 
        apply_smote=apply_smote, 
        random_state=random_state, 
        k_neighbors=k_neighbors,
        sampling_strategy=sampling_strategy,
        n1_target=n1_target,
        n3_target=n3_target
    )
    
    # 验证集不使用SMOTE
    val_dataset = SMOTEDualModalityDataset(val_files, apply_smote=False)

    # 计算类别权重
    all_ys = train_dataset.y_data.numpy().tolist()
    num_classes = len(np.unique(all_ys))
    counts = [all_ys.count(i) for i in range(num_classes)]

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0
    )

    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0
    )

    return train_loader, val_loader, counts 