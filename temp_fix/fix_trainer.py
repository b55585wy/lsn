#!/usr/bin/env python
import os
import sys
import re

# 读取原始trainer.py文件
trainer_path = '/hpc2hdd/home/ywang183/biosleepx/trainer/trainer.py'
with open(trainer_path, 'r') as file:
    content = file.read()

# 备份原文件
backup_path = '/hpc2hdd/home/ywang183/biosleepx/trainer/trainer.py.bak'
with open(backup_path, 'w') as file:
    file.write(content)
print(f"原文件已备份到 {backup_path}")

# 修改调用损失函数的行
# 对于非特殊损失函数的情况进行替换
content = re.sub(
    r'loss = self\.criterion\(reshaped_output, reshaped_target, self\.class_weights, self\.device\)',
    'try:\n                        loss = self.criterion(reshaped_output, reshaped_target, self.class_weights, self.device)\n                    except TypeError:\n                        # 如果损失函数不接受额外参数，则只传递必要参数\n                        loss = self.criterion(reshaped_output, reshaped_target)',
    content
)

# 同样处理原始(非序列)模型的损失函数调用
content = re.sub(
    r'loss = self\.criterion\(output, target, self\.class_weights, self\.device\)',
    'try:\n                    loss = self.criterion(output, target, self.class_weights, self.device)\n                except TypeError:\n                    # 如果损失函数不接受额外参数，则只传递必要参数\n                    loss = self.criterion(output, target)',
    content
)

# 写入修改后的文件
with open(trainer_path, 'w') as file:
    file.write(content)

print("训练器文件已成功修复")
print("现在您可以使用以下命令运行训练:")
print("python train_Kfold_CV.py -c config_sequence.json -f 0 -d 0 -da data20/data20npy") 