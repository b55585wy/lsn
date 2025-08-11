import numpy as np
import torch
from base import BaseTrainer
from utils import inf_loop, MetricTracker
import torch.nn as nn
import os
import json
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau

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
                scheduler_type = config.config['lr_scheduler']['type']
                scheduler_args = config.config['lr_scheduler']['args']
                
                if scheduler_type == 'CosineAnnealingLR':
                    self.lr_scheduler = CosineAnnealingLR(
                        optimizer,
                        T_max=scheduler_args['T_max'],
                        eta_min=scheduler_args['eta_min']
                    )
                elif scheduler_type == 'StepLR':
                    self.lr_scheduler = StepLR(
                        optimizer,
                        step_size=scheduler_args['step_size'],
                        gamma=scheduler_args['gamma']
                    )
                elif scheduler_type == 'ReduceLROnPlateau':
                    self.lr_scheduler = ReduceLROnPlateau(
                        optimizer,
                        mode=scheduler_args.get('mode', 'min'),
                        factor=scheduler_args.get('factor', 0.1),
                        patience=scheduler_args.get('patience', 10),
                        verbose=scheduler_args.get('verbose', True),
                        threshold=scheduler_args.get('threshold', 0.0001),
                        threshold_mode=scheduler_args.get('threshold_mode', 'rel'),
                        cooldown=scheduler_args.get('cooldown', 0),
                        min_lr=scheduler_args.get('min_lr', 0),
                        eps=scheduler_args.get('eps', 1e-08)
                    )
                    self.logger.info(f"使用 ReduceLROnPlateau 学习率调度器，参数: {scheduler_args}")
                else:
                    self.logger.warning(f"不支持的学习率调度器类型: {scheduler_type}")
                    self.lr_scheduler = None
            else:
                self.lr_scheduler = None
        except Exception as e:
            self.logger.warning(f"初始化学习率调度器时出错: {str(e)}")
            self.lr_scheduler = None
            
        self.log_step = int(data_loader.batch_size) * 1  # reduce this if you want more logs

        # 添加每个类别的指标
        metric_names = ['loss'] + [m.__name__ for m in self.metric_ftns]
        # 根据类别数量动态添加类别指标
        self.num_classes = self.config['arch']['args']['num_classes']
        self.class_names = self.config.config.get('class_names', [f'Class {i}' for i in range(self.num_classes)])
        
        class_metrics = [f'acc_class_{i}' for i in range(self.num_classes)]
        all_metrics = metric_names + class_metrics

        self.train_metrics = MetricTracker(*all_metrics)
        self.valid_metrics = MetricTracker(*all_metrics)
        self.test_metrics = MetricTracker(*all_metrics)

        self.fold_id = fold_id
        self.selected = 0
        self.class_weights = class_weights

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
                
                # 提取logits，如果模型返回的是元组
                if isinstance(output, tuple):
                    logits = output[0]
                    # 如果是CombinedContrastiveClassificationLoss，则将整个元组传递给损失函数
                    if self.config.config['loss']['type'] == 'CombinedContrastiveClassificationLoss':
                        loss = self.criterion(output, target, self.class_weights, self.device)
                    else:
                        # 对于其他损失函数，只使用logits
                        # 重塑为 (batch_size*seq_len, num_classes)
                        reshaped_logits = logits.reshape(-1, logits.size(-1))
                        reshaped_target = target.reshape(-1)
                        try:
                            loss = self.criterion(reshaped_logits, reshaped_target, self.class_weights, self.device)
                        except TypeError:
                            loss = self.criterion(reshaped_logits, reshaped_target)
                    output_for_metrics = logits # 用于指标计算的输出
                else:
                    # 如果不是元组，则直接使用output
                    logits = output
                    # 重塑为 (batch_size*seq_len, num_classes)
                    reshaped_logits = logits.reshape(-1, logits.size(-1))
                    reshaped_target = target.reshape(-1)
                    try:
                        loss = self.criterion(reshaped_logits, reshaped_target, self.class_weights, self.device)
                    except TypeError:
                        loss = self.criterion(reshaped_logits, reshaped_target)
                    output_for_metrics = logits # 用于指标计算的输出
            else: # 处理非序列数据
                data, target = batch_data
                data, target = data.to(self.device), target.to(self.device)

                # Add a sequence dimension for non-sequential data to match model input
                data = data.unsqueeze(1) # [B, C, T] -> [B, 1, C, T]

                self.optimizer.zero_grad()
                output = self.model(data)
                
                if isinstance(output, tuple):
                    logits = output[0]
                else:
                    logits = output

                # Reshape for BCEWithLogitsLoss
                # Input should be (N, *), Target should be (N, *)
                # Current logits: [B, 1, 2], Current target: [B]
                logits_for_loss = logits.squeeze(1)[:, 1] # -> [B]
                target_for_loss = target.float() # -> [B]
                loss = self.criterion(logits_for_loss, target_for_loss)
                output_for_metrics = logits

            loss.backward()
            self.optimizer.step()

            self.train_metrics.update('loss', loss.item())
            
            # 使用处理后的output_for_metrics计算指标
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output_for_metrics.detach().cpu(), target.detach().cpu()))
                
            # 收集批次数据用于计算类别准确率
            batch_outputs.append(output_for_metrics.detach())
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
                
        # 合并所有批次的输出和目标
        all_outputs = torch.cat(batch_outputs)
        all_targets = torch.cat(batch_targets)
        
        # 计算每个类别的准确率
        from model.metric import per_class_accuracy
        class_accs = per_class_accuracy(all_outputs.detach().cpu(), all_targets.detach().cpu())
        
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
        f1_per_class = f1_score(true, pred, average=None, labels=list(range(self.num_classes)))
        
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
            total_loss = 0
            num_batches = 0
            
            for batch_idx, batch_data in enumerate(self.valid_data_loader):
                # 处理序列数据
                if len(batch_data) == 3 and isinstance(batch_data[0], torch.Tensor) and len(batch_data[0].shape) == 4:
                    # 序列模型: (batch_size, seq_len, channels, time)
                    data_eeg, data_eog, target = batch_data
                    data_eeg, data_eog = data_eeg.to(self.device), data_eog.to(self.device)
                    target = target.to(self.device)
                    output = self.model(data_eeg, data_eog)
                    
                    # 提取logits，如果模型返回的是元组
                    if isinstance(output, tuple):
                        logits = output[0]
                        # 如果是CombinedContrastiveClassificationLoss，则将整个元组传递给损失函数
                        if isinstance(self.config.config['loss'], dict) and self.config.config['loss'].get('type') == 'CombinedContrastiveClassificationLoss':
                            loss = self.criterion(output, target, self.class_weights, self.device)
                        else:
                            # 对于其他损失函数，只使用logits
                            # 重塑为 (batch_size*seq_len, num_classes)
                            reshaped_logits = logits.reshape(-1, logits.size(-1))
                            reshaped_target = target.reshape(-1)
                            try:
                                loss = self.criterion(reshaped_logits, reshaped_target, self.class_weights, self.device)
                            except TypeError:
                                loss = self.criterion(reshaped_logits, reshaped_target)
                        output_for_metrics = logits # 用于指标计算的输出
                    else:
                        # 如果不是元组，则直接使用output
                        logits = output
                        # 重塑为 (batch_size*seq_len, num_classes)
                        reshaped_logits = logits.reshape(-1, logits.size(-1))
                        reshaped_target = target.reshape(-1)
                        try:
                            loss = self.criterion(reshaped_logits, reshaped_target, self.class_weights, self.device)
                        except TypeError:
                            loss = self.criterion(reshaped_logits, reshaped_target)
                        output_for_metrics = logits # 用于指标计算的输出
                        
                    # 中间epoch预测（取序列中间的输出作为最终预测）
                    mid_idx = output_for_metrics.size(1) // 2
                    mid_output = output_for_metrics[:, mid_idx, :]
                    mid_target = target[:, mid_idx]
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                    preds_ = torch.max(mid_output, 1, keepdim=True)[1]
                    outs = np.append(outs, preds_.cpu().detach().numpy())
                    trgs = np.append(trgs, mid_target.cpu().detach().numpy())
                else:  # 处理非序列数据
                    # 原始单epoch模型: (batch_size, channels, time)
                    data, target = batch_data
                    data, target = data.to(self.device), target.to(self.device)
                    
                    # Add a sequence dimension for non-sequential data to match model input
                    data = data.unsqueeze(1) # [B, C, T] -> [B, 1, C, T]

                    output = self.model(data)
                    
                    # 提取logits，如果模型返回的是元组
                    if isinstance(output, tuple):
                        logits = output[0]
                        # 如果是CombinedContrastiveClassificationLoss，则将整个元组传递给损失函数
                        if isinstance(self.config.config['loss'], dict) and self.config.config['loss'].get('type') == 'CombinedContrastiveClassificationLoss':
                            loss = self.criterion(output, target, self.class_weights, self.device)
                        else:
                            # Reshape for BCEWithLogitsLoss
                            logits_for_loss = logits.squeeze(1)[:, 1] # -> [B]
                            target_for_loss = target.float() # -> [B]
                            loss = self.criterion(logits_for_loss, target_for_loss)
                        output_for_metrics = logits # 用于指标计算的输出
                    else:
                        # 如果不是元组，则直接使用output
                        logits = output
                        try:
                            loss = self.criterion(logits, target, self.class_weights, self.device)
                        except TypeError:
                            loss = self.criterion(logits, target)
                        output_for_metrics = logits # 用于指标计算的输出
 
                    total_loss += loss.item()
                    num_batches += 1
                    
                    preds_ = torch.max(output_for_metrics, 1, keepdim=True)[1]
                    outs = np.append(outs, preds_.cpu().detach().numpy())
                    trgs = np.append(trgs, target.cpu().detach().numpy())
                
                # 收集批次输出和目标用于计算类别准确率
                batch_outputs.append(output_for_metrics.detach())
                batch_targets.append(target)

        # 合并所有批次的输出和目标
        all_outputs = torch.cat(batch_outputs)
        all_targets = torch.cat(batch_targets)
        
        # 确保 all_outputs 和 all_targets 是 CPU 上的 NumPy 数组
        if len(all_outputs.shape) == 3:
            # 重塑为 (batch_size*seq_len, num_classes)
            reshaped_output = all_outputs.reshape(-1, all_outputs.size(-1))
            reshaped_target = all_targets.reshape(-1)
            pred = torch.argmax(reshaped_output, dim=1).cpu().numpy()
            true = reshaped_target.cpu().numpy()
        else:
            pred = torch.argmax(all_outputs, dim=1).cpu().numpy()
            true = all_targets.cpu().numpy()

        # 计算整体准确率、F1和Kappa
        from sklearn.metrics import accuracy_score, cohen_kappa_score
        overall_accuracy = accuracy_score(true, pred)
        overall_f1_macro = f1_score(true, pred, average='macro', labels=list(range(self.num_classes)))

        # 更新验证指标
        avg_loss = total_loss / num_batches
        self.valid_metrics.update('loss', avg_loss)
        self.valid_metrics.update('accuracy', overall_accuracy)
        self.valid_metrics.update('f1', overall_f1_macro)
        
        # 仅在配置中请求时计算和更新kappa
        if 'kappa' in [m.__name__ for m in self.metric_ftns]:
            overall_kappa = cohen_kappa_score(true, pred, labels=list(range(self.num_classes)))
            self.valid_metrics.update('kappa', overall_kappa)

        # 计算每个类别的准确率
        from model.metric import per_class_accuracy
        class_accs = per_class_accuracy(all_outputs.detach().cpu(), all_targets.detach().cpu())
        
        # 更新验证指标
        for class_label, acc_value in class_accs.items():
            self.valid_metrics.update(class_label, acc_value)
            
        # 计算每个类别的F1分数
        f1_per_class = f1_score(true, pred, average=None, labels=list(range(self.num_classes)))
        
        # 创建详细的验证日志
        val_log = self.valid_metrics.result()
        
        # 添加每个类别的F1分数
        for i, f1_val in enumerate(f1_per_class):
            val_log[f'f1_class_{i}'] = f1_val
            
        # 计算并显示混淆矩阵
        cm = confusion_matrix(true, pred, labels=list(range(self.num_classes)))
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
        total_loss = 0
        num_batches = 0
        
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
                    
                    # 处理对比学习模型的输出（可能是元组）
                    if isinstance(output, tuple) and len(output) == 3:
                        # 对比学习模型输出: (logits, projections, features)
                        logits, projections, features = output
                        
                        # 使用CombinedContrastiveClassificationLoss处理完整输出
                        if isinstance(self.config.config['loss'], dict) and self.config.config['loss'].get('type') == 'combined_contrastive_classification_loss':
                            loss = self.criterion(output, target, self.class_weights, self.device)
                        else:
                            try:
                                loss = self.criterion(logits, target, self.class_weights, self.device)
                            except TypeError:
                                loss = self.criterion(logits, target)
                        
                        # 更新用于度量计算的输出变量
                        output_for_metrics = logits
                        
                    # 检查是否使用序列损失函数
                    elif self.config.config['loss'] == 'sequential_focal_loss' and 'loss_args' in self.config.config:
                        loss_args = self.config.config['loss_args']
                        loss = self.criterion(
                            output, target, self.class_weights, self.device,
                            gamma=loss_args.get('gamma', 2.0),
                            label_smoothing=loss_args.get('label_smoothing', 0.0),
                            n1_weight_multiplier=loss_args.get('n1_weight_multiplier', 1.0),
                            n1_class_idx=loss_args.get('n1_class_idx', 1),
                            transition_weight=loss_args.get('transition_weight', 0.2)
                        )
                        output_for_metrics = output
                        
                    elif self.config.config['loss'] == 'focal_loss' and 'loss_args' in self.config.config:
                        # 对于序列输出，但使用普通focal_loss，需要重塑
                        loss_args = self.config.config['loss_args']
                        # 提取logits，如果模型返回的是元组
                        if isinstance(output, tuple):
                            logits = output[0]
                        else:
                            logits = output
                        reshaped_output = logits.reshape(-1, logits.size(-1))
                        reshaped_target = target.reshape(-1)
                        try:
                            loss = self.criterion(reshaped_output, reshaped_target, self.class_weights, self.device)
                        except TypeError:
                            loss = self.criterion(reshaped_output, reshaped_target)
                        output_for_metrics = logits # 确保 output_for_metrics 是张量
                        
                    else:
                        # 提取logits，如果模型返回的是元组
                        if isinstance(output, tuple):
                            logits = output[0]
                        else:
                            logits = output
                        reshaped_output = logits.reshape(-1, logits.size(-1))
                        reshaped_target = target.reshape(-1)
                        try:
                            loss = self.criterion(reshaped_output, reshaped_target, self.class_weights, self.device)
                        except TypeError:
                                loss = self.criterion(reshaped_output, reshaped_target)
                        output_for_metrics = logits # 确保 output_for_metrics 是张量
                        
                    # 中间epoch预测（取序列中间的输出作为最终预测）
                    mid_idx = output_for_metrics.size(1) // 2
                    mid_output = output_for_metrics[:, mid_idx, :]
                    mid_target = target[:, mid_idx]
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                    # 对于序列，只保存中间epoch的预测结果
                    preds_ = torch.max(mid_output, 1, keepdim=True)[1]
                    outs = np.append(outs, preds_.cpu().detach().numpy())
                    trgs = np.append(trgs, mid_target.cpu().detach().numpy())
                else:
                    # 原始单epoch模型: (batch_size, channels, time)
                    data, target = batch_data
                    data, target = data.to(self.device), target.to(self.device)

                    # Add a sequence dimension for non-sequential data to match model input
                    data = data.unsqueeze(1) # [B, C, T] -> [B, 1, C, T]

                    output = self.model(data)
                    
                    # 处理对比学习模型的输出（可能是元组）
                    if isinstance(output, tuple) and len(output) == 3:
                        # 对比学习模型输出: (logits, projections, features)
                        logits, projections, features = output
                        
                        # 使用CombinedContrastiveClassificationLoss处理完整输出
                        if isinstance(self.config.config['loss'], dict) and self.config.config['loss'].get('type') == 'combined_contrastive_classification_loss':
                            loss = self.criterion(output, target, self.class_weights, self.device)
                        else:
                            try:
                                loss = self.criterion(logits, target, self.class_weights, self.device)
                            except TypeError:
                                loss = self.criterion(logits, target)
                        
                        # 更新用于度量计算的输出变量
                        output_for_metrics = logits
                    # 检查是否使用focal_loss并是否有额外参数
                    elif self.config.config['loss'] == 'focal_loss' and 'loss_args' in self.config.config:
                        loss_args = self.config.config['loss_args']
                        loss = self.criterion(
                            output, target, self.class_weights, self.device,
                            gamma=loss_args.get('gamma', 2.0),
                            label_smoothing=loss_args.get('label_smoothing', 0.0),
                            n1_weight_multiplier=loss_args.get('n1_weight_multiplier', 1.0),
                            n1_class_idx=loss_args.get('n1_class_idx', 1)
                        )
                        output_for_metrics = output
                    else:
                        # Reshape for BCEWithLogitsLoss
                        logits_for_loss = output.squeeze(1)[:, 1] # -> [B]
                        target_for_loss = target.float() # -> [B]
                        loss = self.criterion(logits_for_loss, target_for_loss)
                        output_for_metrics = output

                    total_loss += loss.item()
                    num_batches += 1
                    
                    preds_ = torch.max(output_for_metrics, 1, keepdim=True)[1]
                    outs = np.append(outs, preds_.cpu().detach().numpy())
                    trgs = np.append(trgs, target.cpu().detach().numpy())
                
                # 收集批次输出和目标用于计算类别准确率
                batch_outputs.append(output_for_metrics.detach())
                batch_targets.append(target)

        # 合并所有批次的输出和目标
        all_outputs = torch.cat(batch_outputs)
        all_targets = torch.cat(batch_targets)
        
        # 确保 all_outputs 和 all_targets 是 CPU 上的 NumPy 数组
        if len(all_outputs.shape) == 3:
            # 重塑为 (batch_size*seq_len, num_classes)
            reshaped_output = all_outputs.reshape(-1, all_outputs.size(-1))
            reshaped_target = all_targets.reshape(-1)
            pred = torch.argmax(reshaped_output, dim=1).cpu().numpy()
            true = reshaped_target.cpu().numpy()
        else:
            pred = torch.argmax(all_outputs, dim=1).cpu().numpy()
            true = all_targets.cpu().numpy()

        # 计算整体准确率、F1和Kappa
        from sklearn.metrics import accuracy_score, cohen_kappa_score
        overall_accuracy = accuracy_score(true, pred)
        overall_f1_macro = f1_score(true, pred, average='macro', labels=list(range(self.num_classes)))

        # 更新测试指标
        avg_loss = total_loss / num_batches
        self.test_metrics.update('loss', avg_loss)
        self.test_metrics.update('accuracy', overall_accuracy)
        self.test_metrics.update('f1', overall_f1_macro)

        # 仅在配置中请求时计算和更新kappa
        if 'kappa' in [m.__name__ for m in self.metric_ftns]:
            overall_kappa = cohen_kappa_score(true, pred, labels=list(range(self.num_classes)))
            self.test_metrics.update('kappa', overall_kappa)
        
        # 计算每个类别的准确率
        from model.metric import per_class_accuracy
        class_accs = per_class_accuracy(all_outputs.detach().cpu(), all_targets.detach().cpu())
        
        # 更新测试指标
        for class_label, acc_value in class_accs.items():
            self.test_metrics.update(class_label, acc_value)
            
        # 计算每个类别的F1分数
        f1_per_class = f1_score(true, pred, average=None, labels=list(range(self.num_classes)))
        
        # 创建详细的测试日志
        test_log = self.test_metrics.result()
        
        # 添加每个类别的F1分数
        for i, f1_val in enumerate(f1_per_class):
            test_log[f'f1_class_{i}'] = f1_val
            
        # 计算并显示混淆矩阵
        cm = confusion_matrix(true, pred, labels=list(range(self.num_classes)))
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
            best_this_epoch = False # Flag to check if this epoch was the best
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
                    best_this_epoch = True
                    best_metrics = log
                    self.logger.info(f"New best model found at epoch {epoch} with {self.mnt_metric}: {self.mnt_best:.6f}. Saving checkpoint.")
                    self._save_checkpoint(epoch, save_best=True) # Immediately save best model and its epoch checkpoint
                    
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
                
                # 在每个epoch结束后，并且在检查早停之后，更新学习率
                if self.lr_scheduler is not None:
                    if isinstance(self.lr_scheduler, ReduceLROnPlateau):
                        # ReduceLROnPlateau 需要监控验证指标
                        if self.mnt_metric in log: # 确保监控的指标存在于日志中
                            self.lr_scheduler.step(log[self.mnt_metric])
                        else:
                            self.logger.warning(f"监控指标 {self.mnt_metric} 未在日志中找到，ReduceLROnPlateau 可能无法正确工作。")
                    else:
                        # 其他调度器（如 CosineAnnealingLR, StepLR）在每个epoch后step
                        self.lr_scheduler.step()

            # Periodic save every 'self.save_period' epochs
            if epoch % self.save_period == 0:
                if not best_this_epoch: # Only save periodically if not already saved as best in this epoch
                    self.logger.info(f"Saving periodic checkpoint at epoch {epoch}.")
                    self._save_checkpoint(epoch, save_best=False)
        
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
