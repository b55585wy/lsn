import numpy as np
import os
import json
from scipy import signal

def normalize_signal(data):
    """标准化信号"""
    return (data - np.mean(data)) / (np.std(data) + 1e-6)

def prepare_demo_data(input_file, output_dir):
    """准备用于Android演示的数据
    
    Args:
        input_file: 输入的npz文件路径
        output_dir: 输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据
    data = np.load(input_file)
    x = data['x']  # shape: (n_segments, 3000)
    y = data['y']  # shape: (n_segments,)
    fs = data['fs']  # 采样率
    
    # 为每个睡眠阶段选择多个代表性片段
    stage_examples = {}
    for stage in range(5):  # 0-4分别代表W, N1, N2, N3, REM
        stage_indices = np.where(y == stage)[0]
        if len(stage_indices) > 0:
            # 选择该阶段的多个片段
            n_examples = min(5, len(stage_indices))  # 每个阶段最多选择5个片段
            selected_indices = stage_indices[::len(stage_indices)//n_examples][:n_examples]
            
            stage_examples[str(stage)] = []
            for idx in selected_indices:
                # 对信号进行预处理
                eeg_data = x[idx]
                eeg_normalized = normalize_signal(eeg_data)
                
                # 计算频谱特征
                freqs, psd = signal.welch(eeg_normalized, fs=fs, nperseg=256)
                
                # 计算各频带能量
                delta_power = np.mean(psd[(freqs >= 0.5) & (freqs <= 4)])  # Delta: 0.5-4 Hz
                theta_power = np.mean(psd[(freqs >= 4) & (freqs <= 8)])    # Theta: 4-8 Hz
                alpha_power = np.mean(psd[(freqs >= 8) & (freqs <= 13)])   # Alpha: 8-13 Hz
                beta_power = np.mean(psd[(freqs >= 13) & (freqs <= 30)])   # Beta: 13-30 Hz
                
                segment_info = {
                    'data': eeg_normalized.tolist(),
                    'label': int(y[idx]),
                    'features': {
                        'delta_power': float(delta_power),
                        'theta_power': float(theta_power),
                        'alpha_power': float(alpha_power),
                        'beta_power': float(beta_power)
                    }
                }
                stage_examples[str(stage)].append(segment_info)
    
    # 保存元数据
    metadata = {
        'sampling_rate': int(fs),
        'segment_duration': 30,  # 30秒
        'n_samples': 3000,
        'stage_mapping': {
            '0': 'Wake',
            '1': 'N1',
            '2': 'N2',
            '3': 'N3',
            '4': 'REM'
        },
        'frequency_bands': {
            'delta': '0.5-4 Hz',
            'theta': '4-8 Hz',
            'alpha': '8-13 Hz',
            'beta': '13-30 Hz'
        }
    }
    
    # 保存为JSON文件
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    with open(os.path.join(output_dir, 'demo_segments.json'), 'w') as f:
        json.dump(stage_examples, f, indent=2)
    
    print(f"Demo data prepared in {output_dir}")
    print("Files created:")
    print("- metadata.json: 包含数据格式和采样率信息")
    print("- demo_segments.json: 包含每个睡眠阶段的示例数据")
    
    # 打印每个阶段的样本数量
    for stage, examples in stage_examples.items():
        print(f"Stage {metadata['stage_mapping'][stage]}: {len(examples)} examples")

if __name__ == "__main__":
    input_file = "/hpc2hdd/home/ywang183/biosleepx/data78/SC4281G0.npz"
    output_dir = "android_demo/app/src/main/assets"
    prepare_demo_data(input_file, output_dir) 