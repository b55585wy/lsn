# AttnSleep 详细指标评估指南

本文档介绍了如何使用新添加的详细指标评估功能，帮助您查看每个睡眠阶段类别的F1分数、精确度、召回率等详细信息。

## 功能介绍

我们对`model/metric.py`文件进行了增强，添加了以下功能：

1. **每个类别的详细指标计算**：通过`per_class_metrics`函数，可以获取每个睡眠阶段的精确度、召回率和F1分数。
2. **完整的模型评估**：通过`evaluate_model`函数，可以一次性获取模型的所有评估指标。
3. **可视化工具**：提供了混淆矩阵和每个类别性能指标的可视化功能。

## 使用方法

### 1. 使用示例脚本查看详细指标

我们提供了一个简单的脚本`show_detailed_metrics.py`，可以直接使用它来查看训练好的模型在验证集上的详细指标：

```bash
python show_detailed_metrics.py \
    --checkpoint saved/exp1/model_best.pth \
    --data_dir /path/to/your/data \
    --fold_id 0 \
    --output_dir metrics_results
```

参数说明：
- `--checkpoint`：训练好的模型检查点路径
- `--data_dir`：数据目录
- `--fold_id`：要评估的折叠ID
- `--output_dir`：保存结果的目录
- `--batch_size`：批次大小（默认128）
- `--shhs`：如果使用SHHS数据集，添加此标志
- `--num_folds`：交叉验证的折叠数（默认10）

### 2. 在自己的代码中使用

您也可以在自己的代码中直接使用这些功能：

```python
from model.metric import evaluate_model
from utils.show_metrics import evaluate_and_show_metrics

# 方法1：使用evaluate_model获取详细指标
metrics = evaluate_model(model, data_loader, device)
print(f"总体准确率: {metrics['accuracy']}")
print(f"总体F1分数: {metrics['f1_score']}")
print(f"每个类别的F1分数: {metrics['per_class_f1']}")

# 方法2：使用evaluate_and_show_metrics获取详细指标并可视化
class_names = ['Wake', 'N1', 'N2', 'N3', 'REM']
metrics = evaluate_and_show_metrics(
    model=model,
    data_loader=data_loader,
    device=device,
    save_dir='output_dir',
    class_names=class_names
)
```

## 输出结果说明

### 1. 控制台输出

脚本会在控制台输出以下信息：

- 总体准确率和F1分数
- 每个类别的F1分数
- 每个类别的精确度、召回率、F1分数和支持度

### 2. 保存的文件

脚本会在指定的输出目录保存以下文件：

- `detailed_metrics.json`：包含所有详细指标的JSON文件
- `confusion_matrix.png`：混淆矩阵图
- `per_class_metrics.png`：每个类别的指标条形图
- `per_class_metrics.csv`：每个类别的指标CSV文件

### 3. 指标解释

- **精确度(Precision)**：正确预测为某类别的样本数 / 预测为该类别的样本总数
- **召回率(Recall)**：正确预测为某类别的样本数 / 该类别的实际样本总数
- **F1分数**：精确度和召回率的调和平均值，F1 = 2 * (precision * recall) / (precision + recall)
- **支持度(Support)**：该类别的实际样本数量

## 注意事项

1. 确保您的模型检查点文件包含正确的模型状态和配置信息。
2. 如果您使用的是自定义的数据集或模型，可能需要调整脚本中的相关参数。
3. 可视化功能需要matplotlib和seaborn库，请确保已安装这些依赖。