import torch
import torch.nn as nn
from model.model import BioSleepX
import os

def export_model(model_path, output_path):
    # 加载模型
    model = BioSleepX()
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    # 创建示例输入
    example_eeg = torch.randn(1, 1, 3000)  # 假设输入长度为3000
    example_eog = torch.randn(1, 1, 3000)
    
    # 转换为TorchScript
    traced_script_module = torch.jit.trace(model, (example_eeg, example_eog))
    
    # 保存模型
    traced_script_module.save(output_path)
    print(f"Model exported to: {output_path}")

if __name__ == "__main__":
    model_path = "/hpc2hdd/home/ywang183/biosleepx/saved/Exp1/04_05_2025_13_31_59_fold0/checkpoint-epoch90.pth"
    output_path = "biosleepx_mobile.pt"
    export_model(model_path, output_path) 