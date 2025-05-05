# Haar小波下采样模块集成到AttnSleep

本项目将Haar小波下采样模块（Haar Wavelet Downsampling, HWD）集成到AttnSleep模型中，用于睡眠阶段分类任务。HWD模块基于论文《Haar Wavelet Downsampling: A Simple but Effective Downsampling Module For Semantic Segmentation》，原本用于2D图像分割，现已适配为1D时序信号处理版本。

## 实现原理

原始的HWD模块使用Haar小波变换替代传统的最大池化或平均池化操作，能够更好地保留信号的频率信息。在本项目中，我们将2D版本的HWD模块适配为1D版本，用于处理睡眠EEG时序信号。

### 主要修改

1. **Down_wt1D类**：1D版本的Haar小波下采样模块，用于替代MaxPool1d
2. **WaveletPool1d类**：封装Down_wt1D的接口类，使其可以直接替代MaxPool1d
3. **MRCNN类修改**：添加use_wavelet参数，支持在特征提取路径中使用小波下采样
4. **AttnSleep类修改**：添加use_wavelet参数，传递给MRCNN

## 安装依赖

使用小波下采样模块需要安装pytorch_wavelets包：

```bash
# 运行安装脚本
chmod +x install_wavelets.sh
./install_wavelets.sh
```

或手动安装：

```bash
git clone https://github.com/fbcotter/pytorch_wavelets
cd pytorch_wavelets
pip install .
```

## 使用方法

### 评估不同下采样方法的性能

使用修改后的评估脚本可以比较标准下采样和小波下采样的性能差异：

```bash
# 仅评估标准下采样（MaxPool1d）
python eval_attention_compare.py

# 同时评估标准下采样和小波下采样
python eval_attention_compare.py --use_wavelet

# 使用SHHS数据集
python eval_attention_compare.py --use_wavelet --use_shhs

# 指定数据目录和输出文件
python eval_attention_compare.py --use_wavelet --data_dir /path/to/data --output_file results.json
```

### 参数说明

- `--data_dir`: 数据目录路径
- `--num_folds`: 交叉验证折数
- `--fold_id`: 要评估的折叠ID
- `--batch_size`: 批次大小
- `--use_shhs`: 是否使用SHHS数据集
- `--use_wavelet`: 是否使用小波下采样替代最大池化
- `--output_file`: 结果输出文件名

## 结果分析

评估脚本会输出以下指标的比较结果：

1. 模型参数量
2. 推理速度（毫秒/样本）
3. 验证集准确率
4. 每个类别的精确率、召回率和F1分数
5. 混淆矩阵

结果将保存在指定的JSON文件中，并在metrics_results_*目录下生成详细的评估指标和可视化图表。

## 引用

如果您使用了本项目的Haar小波下采样模块，请引用原论文：

```
@article{XU2023109819,
title = {Haar Wavelet Downsampling: A Simple but Effective Downsampling Module for Semantic Segmentation},
journal = {Pattern Recognition},
pages = {109819},
year = {2023},
issn = {0031-3203},
doi = {https://doi.org/10.1016/j.patcog.2023.109819},
url = {https://www.sciencedirect.com/science/article/pii/S0031320323005174},
author = {Guoping Xu and Wentao Liao and Xuan Zhang and Chang Li and Xinwei He and Xinglong Wu},
keywords = {Semantic segmentation, Downsampling, Haar wavelet, Information Entropy}
}
```