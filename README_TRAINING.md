# AttnSleep 模型训练指南

本文档提供了如何使用 AttnSleep 模型进行训练的详细说明，包括标准下采样和 Haar 小波下采样两种方式。

## 准备工作

在开始训练之前，请确保：

1. 已安装所有必要的依赖项
2. 如果要使用小波下采样，请安装 pytorch_wavelets 包：

```bash
# 运行安装脚本
chmod +x install_wavelets.sh
./install_wavelets.sh
```

3. 准备好训练数据集

## 训练命令

### 使用标准下采样（MaxPool1d）进行训练

```bash
python train_Kfold_CV.py -c config.json -f 0 -da /path/to/data
```

### 使用 Haar 小波下采样进行训练

```bash
python train_Kfold_CV.py -c config.json -f 0 -da /path/to/data --use_wavelet
```

## 参数说明

- `-c, --config`: 配置文件路径，默认为 "config.json"
- `-r, --resume`: 从检查点恢复训练的路径，默认为 None
- `-d, --device`: 要使用的 GPU 索引，默认为 "0"
- `-f, --fold_id`: 交叉验证的折叠 ID
- `-da, --np_data_dir`: 包含 numpy 文件的数据目录
- `-w, --use_wavelet`: 使用 Haar 小波下采样替代最大池化

## 配置文件

在 `config.json` 文件中，您可以设置各种训练参数，包括：

```json
{
    "name": "Exp1",
    "n_gpu": 1,
    "arch": {
        "type": "AttnSleep",
        "args": {
            "attention_type": "efficient_additive",
            "use_wavelet": false
        }
    },
    "data_loader": {
        "args":{
            "batch_size": 128,
            "num_folds": 20
        }
    },
    "optimizer": {
        "type": "Adam",
        "args":{
            "lr": 0.001,
            "weight_decay": 0.001,
            "amsgrad": true
        }
    },
    "loss": "weighted_CrossEntropyLoss",
    "metrics": [
        "accuracy"
    ],
    "trainer": {
        "epochs": 100,
        "save_dir": "saved/",
        "save_period": 30,
        "verbosity": 2,
        "monitor": "min val_loss"
    }
}
```

您可以通过命令行参数 `--use_wavelet` 启用小波下采样，或者在配置文件中将 `use_wavelet` 设置为 `true`。

## 批量训练

对于多折交叉验证的批量训练，可以使用 `batch_train.sh` 脚本：

```bash
# 使用标准下采样进行批量训练
./batch_train.sh /path/to/data

# 使用小波下采样进行批量训练
./batch_train.sh /path/to/data --use_wavelet
```

## 训练结果

训练结果将保存在 `saved/` 目录中，包括：

- 模型检查点
- 训练日志
- TensorBoard 可视化数据

## 评估模型

训练完成后，您可以使用 `eval_attention_compare.py` 脚本评估模型性能：

```bash
# 评估标准下采样模型
python eval_attention_compare.py --data_dir /path/to/data

# 评估小波下采样模型
python eval_attention_compare.py --data_dir /path/to/data --use_wavelet
```