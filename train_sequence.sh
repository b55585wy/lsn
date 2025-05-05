#!/bin/bash

# 设置数据目录
DATA_DIR="data20"  # 修改为您的数据目录

# 设置CUDA设备
CUDA_DEVICE="0"

# 设置序列模型的配置文件
CONFIG="config_sequence.json"

# 运行10折交叉验证
for fold_id in {0..9}
do
  echo "开始训练第 $fold_id 折..."
  python train_Kfold_CV.py --config $CONFIG --fold_id $fold_id --device $CUDA_DEVICE --np_data_dir $DATA_DIR
  echo "第 $fold_id 折训练完成"
done

echo "所有折交叉验证完成，结果保存在 saved/sequential_model/ 目录" 