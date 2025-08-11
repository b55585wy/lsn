import torch
import numpy as np

class RandomNoise(object):
    """
    向输入数据添加随机高斯噪声。
    Args:
        noise_std (float): 高斯噪声的标准差。
    """
    def __init__(self, noise_std=0.01):
        self.noise_std = noise_std

    def __call__(self, sample):
        # sample 是一个字典，包含 'eeg', 'eog', 'label'
        eeg_data = sample['eeg']
        eog_data = sample['eog']

        # 添加高斯噪声
        eeg_noise = torch.randn_like(eeg_data) * self.noise_std
        eog_noise = torch.randn_like(eog_data) * self.noise_std
        
        sample['eeg'] = eeg_data + eeg_noise
        sample['eog'] = eog_data + eog_noise
        
        return sample

class RandomScaling(object):
    """
    随机缩放输入数据的幅度。
    Args:
        scaling_range (tuple): 缩放因子的范围 (min_scale, max_scale)。
    """
    def __init__(self, scaling_range=(0.9, 1.1)):
        self.scaling_range = scaling_range

    def __call__(self, sample):
        eeg_data = sample['eeg']
        eog_data = sample['eog']

        # 生成随机缩放因子
        scale_factor = torch.rand(1) * (self.scaling_range[1] - self.scaling_range[0]) + self.scaling_range[0]
        
        sample['eeg'] = eeg_data * scale_factor
        sample['eog'] = eog_data * scale_factor
        
        return sample

# 示例：组合多个增强
class Compose(object):
    """
    组合多个数据增强操作。
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample