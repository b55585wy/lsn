import numpy as np
import torch
from base import BaseTrainer
from utils import inf_loop, MetricTracker
import torch.nn as nn
import os
import json
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.optim.lr_scheduler import CosineAnnealingLR

selected_d = {"outs": [], "trg": []}
class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader, fold_id,
                 valid_data_loader=None, test_data_loader=None, class_weights=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config, fold_id)
        self.config = config
        self.data_loader = data_loader
        self.len_epoch = len(self.data_loader)

        self.valid_data_loader = valid_data_loader
        self.test_data_loader = test_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.do_test = self.test_data_loader is not None
        
        # 初始化学习率调度器
        try:
            if 'lr_scheduler' in config.config:  # 使用config.config访问原始配置字典
                scheduler_args = config.config['lr_scheduler']['args']
                self.lr_scheduler = CosineAnnealingLR(
                    optimizer,
                    T_max=scheduler_args['T_max'],
                    eta_min=scheduler_args['eta_min']
                )
            else:
                self.lr_scheduler = None
        except Exception as e:
            self.logger.warning(f"初始化学习率调度器时出错: {str(e)}")
            self.lr_scheduler = None
            
        self.log_step = int(data_loader.batch_size) * 1  # reduce this if you want more logs

        # 添加每个类别的指标
        metric_names = ['loss'] + [m.__name__ for m in self.metric_ftns]
        # 添加五个类别的准确率指标
        class_metrics = [f'acc_class_{i}' for i in range(5)]
        all_metrics = metric_names + class_metrics

        self.train_metrics = MetricTracker(*all_metrics)
        self.valid_metrics = MetricTracker(*all_metrics)
        self.test_metrics = MetricTracker(*all_metrics)

        self.fold_id = fold_id
        self.selected = 0
        self.class_weights = class_weights
        self.class_names = ['W', 'N1', 'N2', 'N3', 'REM']  # 睡眠阶段类别名称

    def _train_epoch(self, epoch, total_epochs):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
               total_epochs: Integer, the total number of epoch
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        overall_outs = []
        overall_trgs = []
        batch_outputs = []
        batch_targets = []
        
        for batch_idx, batch_data in enumerate(self.data_loader):
            # 处理序列数据
            if len(batch_data) == 3 and isinstance(batch_data[0], torch.Tensor) and len(batch_data[0].shape) == 4:
                # 序列模型: (batch_size, seq_len, channels, time)
                data_eeg, data_eog, target = batch_data
                data_eeg, data_eog = data_eeg.to(self.device), data_eog.to(self.device)
                target = target.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(data_eeg, data_eog)
                
                # 检查是否使用序列损失函数
                if self.config.config['loss'] == 'sequential_focal_loss' and 'loss_args' in self.config.config:
                    loss_args = self.config.config['loss_args']
                    loss = self.criterion(
                        output, target, self.class_weights, self.device,
                        gamma=loss_args.get('gamma', 2.0),
                        label_smoothing=loss_args.get('label_smoothing', 0.0),
                        n1_weight_multiplier=loss_args.get('n1_weight_multiplier', 1.0),
                        n1_class_idx=loss_args.get('n1_class_idx', 1),
                        transition_weight=loss_args.get('transition_weight', 0.2)
                    )
                elif self.config.config['loss'] == 'focal_loss' and 'loss_args' in self.config.config:
                    # 对于序列输出，但使用普通focal_loss，需要重塑
                    loss_args = self.config.config['loss_args']
                    # 重塑为 (batch_size*seq_len, num_classes)
                    reshaped_output = output.reshape(-1, output.size(-1))
                    reshaped_target = target.reshape(-1)
                    try:
                        loss = self.criterion(reshaped_output, reshaped_target, self.class_weights, self.device)
                    except TypeError:
                        loss = self.criterion(reshaped_output, reshaped_target)
                else:
                    # 重塑为 (batch_size*seq_len, num_classes)
                    reshaped_output = output.reshape(-1, output.size(-1))
                    reshaped_target = target.reshape(-1)
                    try:
                        loss = self.criterion(reshaped_output, reshaped_target, self.class_weights, self.device)
                    except TypeError:
                        loss = self.criterion(reshaped_output, reshaped_target)
            else:
                # 原始单epoch模型: (batch_size, channels, time)
                data_eeg, data_eog, target = batch_data
                data_eeg, data_eog = data_eeg.to(self.device), data_eog.to(self.device)
                target = target.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(data_eeg, data_eog)
                
                # 检查是否使用focal_loss并是否有额外参数
                if self.config.config['loss'] == 'focal_loss' and 'loss_args' in self.config.config:
                    loss_args = self.config.config['loss_args']
                    loss = self.criterion(
                        output, target, self.class_weights, self.device,
                        gamma=loss_args.get('gamma', 2.0),
                        label_smoothing=loss_args.get('label_smoothing', 0.0),
                        n1_weight_multiplier=loss_args.get('n1_weight_multiplier', 1.0),
                        n1_class_idx=loss_args.get('n1_class_idx', 1)
                    )
                else:
                    try:
                        loss = self.criterion(output, target, self.class_weights, self.device)
                    except TypeError:
                        loss = self.criterion(output, target)

            loss.backward()
            self.optimizer.step()

            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))
                
            # 收集批次数据用于计算类别准确率
            batch_outputs.append(output.detach())
            batch_targets.append(target)

            if batch_idx % self.log_step == 0:
                # 添加当前学习率到日志
                current_lr = self.optimizer.param_groups[0]['lr']
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f} LR: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item(),
                    current_lr
                ))

            if batch_idx == self.len_epoch:
                break
                
        # 在每个epoch结束后更新学习率
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
            
        # 合并所有批次的输出和目标
        all_outputs = torch.cat(batch_outputs)
        all_targets = torch.cat(batch_targets)
        
        # 计算每个类别的准确率
        from model.metric import per_class_accuracy
        class_accs = per_class_accuracy(all_outputs, all_targets)
        
        # 更新每个类别的准确率指标
        for class_label, acc_value in class_accs.items():
            self.train_metrics.update(class_label, acc_value)
            
        # 计算混淆矩阵和详细指标
        # 对于序列输出，需要重塑为标准形式
        if len(all_outputs.shape) == 3:
            # 重塑为 (batch_size*seq_len, num_classes)
            reshaped_output = all_outputs.reshape(-1, all_outputs.size(-1))
            reshaped_target = all_targets.reshape(-1)
            pred = torch.argmax(reshaped_output, dim=1).cpu().numpy()
            true = reshaped_target.cpu().numpy()
        else:
            pred = torch.argmax(all_outputs, dim=1).cpu().numpy()
            true = all_targets.cpu().numpy()
        
        # 计算每个类别的F1分数
        f1_per_class = f1_score(true, pred, average=None)
        
        # 创建详细的训练日志
        train_log = self.train_metrics.result()
        
        # 添加每个类别的F1分数
        for i, f1_val in enumerate(f1_per_class):
            train_log[f'f1_class_{i}'] = f1_val
            
        # 打印类别准确率和F1分数
        self.logger.info('=' * 60)
        self.logger.info(f'Epoch {epoch} - 训练集类别指标:')
        for i, class_name in enumerate(self.class_names):
            acc_key = f'acc_class_{i}'
            acc_val = train_log.get(acc_key, 0.0)
            f1_val = train_log.get(f'f1_class_{i}', 0.0)
            self.logger.info(f'  {class_name}: 准确率={acc_val:.4f}, F1={f1_val:.4f}')
        self.logger.info('=' * 60)

        if self.do_validation:
            val_log, outs, trgs = self._valid_epoch(epoch)
            log = {**train_log, **{'val_' + k: v for k, v in val_log.items()}}
            
            if val_log["accuracy"] > self.selected:
                self.selected = val_log["accuracy"]
                selected_d["outs"] = outs
                selected_d["trg"] = trgs
            if epoch == total_epochs:
                overall_outs.extend(selected_d["outs"])
                overall_trgs.extend(selected_d["trg"])

            # THIS part is to reduce the learning rate after 10 epochs to 1e-4
            if epoch == 10:
                for g in self.optimizer.param_groups:
                    g['lr'] = 0.0001
        else:
            log = train_log

        return log, overall_outs, overall_trgs

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        batch_outputs = []
        batch_targets = []
        
        with torch.no_grad():
            outs = np.array([])
            trgs = np.array([])
            
            for batch_idx, batch_data in enumerate(self.valid_data_loader):
                # 处理序列数据
                if len(batch_data) == 3 and isinstance(batch_data[0], torch.Tensor) and len(batch_data[0].shape) == 4:
                    # 序列模型: (batch_size, seq_len, channels, time)
                    data_eeg, data_eog, target = batch_data
                    data_eeg, data_eog = data_eeg.to(self.device), data_eog.to(self.device)
                    target = target.to(self.device)
                    output = self.model(data_eeg, data_eog)
                    
                    # 检查是否使用序列损失函数
                    if self.config.config['loss'] == 'sequential_focal_loss' and 'loss_args' in self.config.config:
                        loss_args = self.config.config['loss_args']
                        loss = self.criterion(
                            output, target, self.class_weights, self.device,
                            gamma=loss_args.get('gamma', 2.0),
                            label_smoothing=loss_args.get('label_smoothing', 0.0),
                            n1_weight_multiplier=loss_args.get('n1_weight_multiplier', 1.0),
                            n1_class_idx=loss_args.get('n1_class_idx', 1),
                            transition_weight=loss_args.get('transition_weight', 0.2)
                        )
                    elif self.config.config['loss'] == 'focal_loss' and 'loss_args' in self.config.config:
                        # 对于序列输出，但使用普通focal_loss，需要重塑
                        loss_args = self.config.config['loss_args']
                        # 重塑为 (batch_size*seq_len, num_classes)
                        reshaped_output = output.reshape(-1, output.size(-1))
                        reshaped_target = target.reshape(-1)
                        try:
                            loss = self.criterion(reshaped_output, reshaped_target, self.class_weights, self.device)
                        except TypeError:
                            loss = self.criterion(reshaped_output, reshaped_target)
                    else:
                        # 重塑为 (batch_size*seq_len, num_classes)
                        reshaped_output = output.reshape(-1, output.size(-1))
                        reshaped_target = target.reshape(-1)
                        try:
                            loss = self.criterion(reshaped_output, reshaped_target, self.class_weights, self.device)
                        except TypeError:
                            loss = self.criterion(reshaped_output, reshaped_target)
                        
                    # 中间epoch预测（取序列中间的输出作为最终预测）
                    mid_idx = output.size(1) // 2
                    mid_output = output[:, mid_idx, :]
                    mid_target = target[:, mid_idx]
                    
                    self.valid_metrics.update('loss', loss.item())
                    for met in self.metric_ftns:
                        self.valid_metrics.update(met.__name__, met(output, target))
                        
                    # 对于序列，只保存中间epoch的预测结果
                    preds_ = torch.max(mid_output, 1, keepdim=True)[1]
                    outs = np.append(outs, preds_.cpu().detach().numpy())
                    trgs = np.append(trgs, mid_target.cpu().detach().numpy())
                else:
                    # 原始单epoch模型: (batch_size, channels, time)
                    data_eeg, data_eog, target = batch_data
                    data_eeg, data_eog = data_eeg.to(self.device), data_eog.to(self.device)
                    target = target.to(self.device)
                    output = self.model(data_eeg, data_eog)
                    
                    # 检查是否使用focal_loss并是否有额外参数
                    if self.config.config['loss'] == 'focal_loss' and 'loss_args' in self.config.config:
                        loss_args = self.config.config['loss_args']
                        loss = self.criterion(
                            output, target, self.class_weights, self.device,
                            gamma=loss_args.get('gamma', 2.0),
                            label_smoothing=loss_args.get('label_smoothing', 0.0),
                            n1_weight_multiplier=loss_args.get('n1_weight_multiplier', 1.0),
                            n1_class_idx=loss_args.get('n1_class_idx', 1)
                        )
                    else:
                        try:
                            loss = self.criterion(output, target, self.class_weights, self.device)
                        except TypeError:
                            loss = self.criterion(output, target)

                    self.valid_metrics.update('loss', loss.item())
                    for met in self.metric_ftns:
                        self.valid_metrics.update(met.__name__, met(output, target))

                    # 使用torch.max替代.data.max，并确保类型转换正确
                    preds_ = torch.max(output, 1, keepdim=True)[1]
                    outs = np.append(outs, preds_.cpu().detach().numpy())
                    trgs = np.append(trgs, target.cpu().detach().numpy())
                
                # 收集批次输出和目标用于计算类别准确率
                batch_outputs.append(output)
                batch_targets.append(target)

        # 合并所有批次的输出和目标
        all_outputs = torch.cat(batch_outputs)
        all_targets = torch.cat(batch_targets)
        
        # 计算每个类别的准确率
        from model.metric import per_class_accuracy
        class_accs = per_class_accuracy(all_outputs, all_targets)
        
        # 更新验证指标
        for class_label, acc_value in class_accs.items():
            self.valid_metrics.update(class_label, acc_value)
            
        # 计算混淆矩阵和详细指标
        # 对于序列输出，需要重塑为标准形式
        if len(all_outputs.shape) == 3:
            # 重塑为 (batch_size*seq_len, num_classes)
            reshaped_output = all_outputs.reshape(-1, all_outputs.size(-1))
            reshaped_target = all_targets.reshape(-1)
            pred = torch.argmax(reshaped_output, dim=1).cpu().numpy()
            true = reshaped_target.cpu().numpy()
        else:
            pred = torch.argmax(all_outputs, dim=1).cpu().numpy()
            true = all_targets.cpu().numpy()
        
        # 计算每个类别的F1分数
        f1_per_class = f1_score(true, pred, average=None)
        
        # 创建详细的验证日志
        val_log = self.valid_metrics.result()
        
        # 添加每个类别的F1分数
        for i, f1_val in enumerate(f1_per_class):
            val_log[f'f1_class_{i}'] = f1_val
            
        # 计算并显示混淆矩阵
        cm = confusion_matrix(true, pred)
        self.logger.info(f'验证集混淆矩阵:\n{cm}') # 添加混淆矩阵日志
        # 可选：将混淆矩阵添加到日志字典中
        # val_log['confusion_matrix'] = cm.tolist()
        
        # 打印类别准确率和F1分数
        self.logger.info('=' * 60)
        self.logger.info(f'验证集类别指标:')
        for i, class_name in enumerate(self.class_names):
            acc_key = f'acc_class_{i}'
            acc_val = val_log.get(acc_key, 0.0)
            f1_val = val_log.get(f'f1_class_{i}', 0.0)
            self.logger.info(f'  {class_name}: 准确率={acc_val:.4f}, F1={f1_val:.4f}')
        self.logger.info('=' * 60)

        return val_log, outs, trgs

    def _test_epoch(self):
        """
        测试集评估
        """
        self.model.eval()
        self.test_metrics.reset()
        batch_outputs = []
        batch_targets = []
        
        with torch.no_grad():
            outs = np.array([])
            trgs = np.array([])
            
            for batch_idx, batch_data in enumerate(self.test_data_loader):
                # 处理序列数据
                if len(batch_data) == 3 and isinstance(batch_data[0], torch.Tensor) and len(batch_data[0].shape) == 4:
                    # 序列模型: (batch_size, seq_len, channels, time)
                    data_eeg, data_eog, target = batch_data
                    data_eeg, data_eog = data_eeg.to(self.device), data_eog.to(self.device)
                    target = target.to(self.device)
                    output = self.model(data_eeg, data_eog)
                    
                    # 检查是否使用序列损失函数
                    if self.config.config['loss'] == 'sequential_focal_loss' and 'loss_args' in self.config.config:
                        loss_args = self.config.config['loss_args']
                        loss = self.criterion(
                            output, target, self.class_weights, self.device,
                            gamma=loss_args.get('gamma', 2.0),
                            label_smoothing=loss_args.get('label_smoothing', 0.0),
                            n1_weight_multiplier=loss_args.get('n1_weight_multiplier', 1.0),
                            n1_class_idx=loss_args.get('n1_class_idx', 1),
                            transition_weight=loss_args.get('transition_weight', 0.2)
                        )
                    elif self.config.config['loss'] == 'focal_loss' and 'loss_args' in self.config.config:
                        # 对于序列输出，但使用普通focal_loss，需要重塑
                        loss_args = self.config.config['loss_args']
                        # 重塑为 (batch_size*seq_len, num_classes)
                        reshaped_output = output.reshape(-1, output.size(-1))
                        reshaped_target = target.reshape(-1)
                        try:
                            loss = self.criterion(reshaped_output, reshaped_target, self.class_weights, self.device)
                        except TypeError:
                            loss = self.criterion(reshaped_output, reshaped_target)
                    else:
                        # 重塑为 (batch_size*seq_len, num_classes)
                        reshaped_output = output.reshape(-1, output.size(-1))
                        reshaped_target = target.reshape(-1)
                        try:
                            loss = self.criterion(reshaped_output, reshaped_target, self.class_weights, self.device)
                        except TypeError:
                            loss = self.criterion(reshaped_output, reshaped_target)
                        
                    # 中间epoch预测（取序列中间的输出作为最终预测）
                    mid_idx = output.size(1) // 2
                    mid_output = output[:, mid_idx, :]
                    mid_target = target[:, mid_idx]
                    
                    self.test_metrics.update('loss', loss.item())
                    for met in self.metric_ftns:
                        self.test_metrics.update(met.__name__, met(output, target))
                        
                    # 对于序列，只保存中间epoch的预测结果
                    preds_ = torch.max(mid_output, 1, keepdim=True)[1]
                    outs = np.append(outs, preds_.cpu().detach().numpy())
                    trgs = np.append(trgs, mid_target.cpu().detach().numpy())
                else:
                    # 原始单epoch模型: (batch_size, channels, time)
                    data_eeg, data_eog, target = batch_data
                    data_eeg, data_eog = data_eeg.to(self.device), data_eog.to(self.device)
                    target = target.to(self.device)
                    output = self.model(data_eeg, data_eog)
                    
                    # 检查是否使用focal_loss并是否有额外参数
                    if self.config.config['loss'] == 'focal_loss' and 'loss_args' in self.config.config:
                        loss_args = self.config.config['loss_args']
                        loss = self.criterion(
                            output, target, self.class_weights, self.device,
                            gamma=loss_args.get('gamma', 2.0),
                            label_smoothing=loss_args.get('label_smoothing', 0.0),
                            n1_weight_multiplier=loss_args.get('n1_weight_multiplier', 1.0),
                            n1_class_idx=loss_args.get('n1_class_idx', 1)
                        )
                    else:
                        try:
                            loss = self.criterion(output, target, self.class_weights, self.device)
                        except TypeError:
                            loss = self.criterion(output, target)

                    self.test_metrics.update('loss', loss.item())
                    for met in self.metric_ftns:
                        self.test_metrics.update(met.__name__, met(output, target))

                    preds_ = torch.max(output, 1, keepdim=True)[1]
                    outs = np.append(outs, preds_.cpu().detach().numpy())
                    trgs = np.append(trgs, target.cpu().detach().numpy())
                
                # 收集批次输出和目标用于计算类别准确率
                batch_outputs.append(output)
                batch_targets.append(target)

        # 合并所有批次的输出和目标
        all_outputs = torch.cat(batch_outputs)
        all_targets = torch.cat(batch_targets)
        
        # 计算每个类别的准确率
        from model.metric import per_class_accuracy
        class_accs = per_class_accuracy(all_outputs, all_targets)
        
        # 更新测试指标
        for class_label, acc_value in class_accs.items():
            self.test_metrics.update(class_label, acc_value)
            
        # 计算混淆矩阵和详细指标
        # 对于序列输出，需要重塑为标准形式
        if len(all_outputs.shape) == 3:
            # 重塑为 (batch_size*seq_len, num_classes)
            reshaped_output = all_outputs.reshape(-1, all_outputs.size(-1))
            reshaped_target = all_targets.reshape(-1)
            pred = torch.argmax(reshaped_output, dim=1).cpu().numpy()
            true = reshaped_target.cpu().numpy()
        else:
            pred = torch.argmax(all_outputs, dim=1).cpu().numpy()
            true = all_targets.cpu().numpy()
        
        # 计算每个类别的F1分数
        f1_per_class = f1_score(true, pred, average=None)
        
        # 创建详细的测试日志
        test_log = self.test_metrics.result()
        
        # 添加每个类别的F1分数
        for i, f1_val in enumerate(f1_per_class):
            test_log[f'f1_class_{i}'] = f1_val
            
        # 计算并显示混淆矩阵
        cm = confusion_matrix(true, pred)
        self.logger.info(f'测试集混淆矩阵:\n{cm}') # 添加混淆矩阵日志
        # 可选：将混淆矩阵添加到日志字典中
        # test_log['confusion_matrix'] = cm.tolist()
            
        # 打印类别准确率和F1分数
        self.logger.info('=' * 60)
        self.logger.info('测试集类别指标:')
        for i, class_name in enumerate(self.class_names):
            acc_key = f'acc_class_{i}'
            acc_val = test_log.get(acc_key, 0.0)
            f1_val = test_log.get(f'f1_class_{i}', 0.0)
            self.logger.info(f'  {class_name}: 准确率={acc_val:.4f}, F1={f1_val:.4f}')
        self.logger.info('=' * 60)

        return test_log, outs, trgs

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def train(self):
        """
        完整的训练逻辑
        """
        not_improved_count = 0
        best_metrics = None
        best_test_metrics = None
        
        for epoch in range(self.start_epoch, self.epochs + 1):
            train_log, overall_outs, overall_trgs = self._train_epoch(epoch, self.epochs)

            # 保存日志信息
            log = {'epoch': epoch}
            log.update(train_log)
            
            # 打印日志信息
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))

            # 根据配置的指标评估模型性能，保存最佳检查点
            best = False
            if self.mnt_mode != 'off':
                try:
                    # 检查模型性能是否改善
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                              (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found. "
                                      "Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                    best_metrics = log
                    
                    # 在最佳验证性能时评估测试集
                    if self.do_test:
                        test_log, test_outs, test_trgs = self._test_epoch()
                        test_metrics = {'test_' + k: v for k, v in test_log.items()}
                        best_test_metrics = test_metrics
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn't improve for {} epochs. "
                                   "Training stops.".format(self.early_stop))
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)
        
        # 保存当前折的最佳指标（包括验证集和测试集）
        if best_metrics is not None:
            metrics_dir = os.path.join(self.checkpoint_dir, 'fold_metrics')
            os.makedirs(metrics_dir, exist_ok=True)
            metrics_file = os.path.join(metrics_dir, f'fold_{self.fold_id}_metrics.json')
            
            # 合并验证集和测试集指标
            all_metrics = best_metrics.copy()
            if best_test_metrics is not None:
                all_metrics.update(best_test_metrics)
            
            with open(metrics_file, 'w') as f:
                json.dump(all_metrics, f, indent=4)
        
        return all_metrics
