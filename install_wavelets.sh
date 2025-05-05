#!/bin/bash

# 安装pytorch_wavelets包
echo "正在安装pytorch_wavelets包..."
git clone https://github.com/fbcotter/pytorch_wavelets
cd pytorch_wavelets
pip install .
cd ..

echo "安装完成！"
echo "现在可以使用以下命令运行评估脚本："
echo "python eval_attention_compare.py --use_wavelet"