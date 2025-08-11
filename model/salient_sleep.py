import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleExtraction(nn.Module):
    """多尺度特征提取模块"""
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation_rates=[1, 3, 5]):
        super().__init__()
        self.branches = nn.ModuleList()
        for dilation in dilation_rates:
            branch = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=dilation*(kernel_size-1)//2, dilation=dilation),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True)
            )
            self.branches.append(branch)
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Conv1d(out_ch * len(dilation_rates), out_ch * 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch * 2, out_ch, kernel_size=1),
            nn.BatchNorm1d(out_ch)
        )
        
    def forward(self, x):
        branch_outputs = []
        for branch in self.branches:
            branch_outputs.append(branch(x))
        
        # 连接多个分支的输出
        x = torch.cat(branch_outputs, dim=1)
        # 融合特征
        x = self.fusion(x)
        return x

class UBlock(nn.Module):
    """U型编码块"""
    def __init__(self, in_ch, out_ch, mid_ch=None, depth=2):
        super().__init__()
        if mid_ch is None:
            mid_ch = out_ch
        
        # 初始化编码器和解码器模块
        self.depth = depth
        self.in_conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True)
        )
        
        # 编码器模块
        self.encoder_modules = nn.ModuleList()
        for i in range(depth-1):
            encoder = nn.Sequential(
                nn.Conv1d(out_ch if i == 0 else mid_ch, mid_ch, kernel_size=3, padding=1),
                nn.BatchNorm1d(mid_ch),
                nn.ReLU(inplace=True)
            )
            self.encoder_modules.append(encoder)
            
        # 中间模块
        self.middle = nn.Sequential(
            nn.Conv1d(mid_ch, mid_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(mid_ch),
            nn.ReLU(inplace=True)
        )
            
        # 解码器模块
        self.decoder_modules = nn.ModuleList()
        for i in range(depth-1):
            ch_out = out_ch if i == depth-2 else mid_ch
            decoder = nn.Sequential(
                nn.Conv1d(mid_ch*2, ch_out, kernel_size=3, padding=1),
                nn.BatchNorm1d(ch_out),
                nn.ReLU(inplace=True)
            )
            self.decoder_modules.append(decoder)
        
    def forward(self, x):
        # 保存原始输入
        x = self.in_conv(x)
        identity = x
        
        # 保存编码器输出及其尺寸
        encoder_outputs = []
        encoder_sizes = []
        current = x
        
        # 编码器前向传播
        for i, encoder in enumerate(self.encoder_modules):
            current = encoder(current)
            encoder_outputs.append(current)
            encoder_sizes.append(current.size(2))  # 保存时间维度大小
            if i < len(self.encoder_modules) - 1:
                current = F.max_pool1d(current, kernel_size=2)
                
        # 中间层处理
        current = self.middle(current)
        
        # 解码器前向传播
        for i, decoder in enumerate(self.decoder_modules):
            # 获取当前要跳跃连接的编码器输出
            skip_connection = encoder_outputs[-(i+1)]
            # 上采样到特定大小而不是倍数
            current = F.interpolate(current, size=encoder_sizes[-(i+1)], mode='linear', align_corners=True)
            # 与跳跃连接特征拼接
            current = torch.cat([current, skip_connection], dim=1)
            # 解码器处理
            current = decoder(current)
        
        # 残差连接 - 确保大小匹配
        if current.size(2) != identity.size(2):
            current = F.interpolate(current, size=identity.size(2), mode='linear', align_corners=True)
        
        return current + identity

class SalientSleep(nn.Module):
    """与你现有框架兼容的SalientSleep模型"""
    def __init__(self, use_eog=True):
        super().__init__()
        
        # 基础参数设置
        base_filters = 16
        n_classes = 5
        
        # 特征提取部分
        # 为了与你现有代码兼容，我们命名为eeg_feature_extractor和eog_feature_extractor
        
        # EEG特征提取器
        self.eeg_feature_extractor = nn.Sequential(
            UBlock(1, base_filters, depth=2),
            nn.Conv1d(base_filters, base_filters*2, kernel_size=1),
            nn.BatchNorm1d(base_filters*2),
            nn.ReLU(),
            nn.MaxPool1d(10),
            
            UBlock(base_filters*2, base_filters*4, depth=3),
            nn.Conv1d(base_filters*4, base_filters*8, kernel_size=1),
            nn.BatchNorm1d(base_filters*8),
            nn.ReLU(),
            nn.MaxPool1d(8),
            
            UBlock(base_filters*8, base_filters*16, depth=3),
            nn.Conv1d(base_filters*16, base_filters*32, kernel_size=1),
            nn.BatchNorm1d(base_filters*32),
            nn.ReLU(),
            nn.MaxPool1d(6),
            
            # 增加多尺度特征提取
            MultiScaleExtraction(base_filters*32, base_filters*32)
        )
        
        # EOG特征提取器（如果使用）
        if use_eog:
            self.eog_feature_extractor = nn.Sequential(
                UBlock(1, base_filters, depth=2),
                nn.Conv1d(base_filters, base_filters*2, kernel_size=1),
                nn.BatchNorm1d(base_filters*2),
                nn.ReLU(),
                nn.MaxPool1d(10),
                
                UBlock(base_filters*2, base_filters*4, depth=3),
                nn.Conv1d(base_filters*4, base_filters*8, kernel_size=1),
                nn.BatchNorm1d(base_filters*8),
                nn.ReLU(),
                nn.MaxPool1d(8),
                
                UBlock(base_filters*8, base_filters*16, depth=3),
                nn.Conv1d(base_filters*16, base_filters*32, kernel_size=1),
                nn.BatchNorm1d(base_filters*32),
                nn.ReLU(),
                nn.MaxPool1d(6),
                
                # 增加多尺度特征提取
                MultiScaleExtraction(base_filters*32, base_filters*32)
            )
        
        # 特征融合层（与你现有代码兼容）
        fusion_ch = base_filters*32 * (2 if use_eog else 1)
        self.fusion_layer = nn.Sequential(
            nn.Conv1d(fusion_ch, base_filters*16, kernel_size=1),
            nn.BatchNorm1d(base_filters*16),
            nn.ReLU(),
            nn.Conv1d(base_filters*16, base_filters*8, kernel_size=1),
            nn.BatchNorm1d(base_filters*8),
            nn.ReLU()
        )
        
        # 维度调整层（保持代码兼容性）
        self.dim_adjust_tce = nn.Sequential(
            nn.Conv1d(base_filters*8, 80, kernel_size=1),
            nn.BatchNorm1d(80),
            nn.ReLU()
        )
        
        # TCE时序编码器（与TCE模块兼容）
        self.tce = nn.Sequential(
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=80, 
                    nhead=8,
                    dim_feedforward=120,
                    dropout=0.1,
                    batch_first=True
                ), 
                num_layers=2
            )
        )
        
        # 分类器（保持与你现有代码兼容）
        self.classifier = nn.Sequential(
            nn.Linear(80, 40),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(40, n_classes)
        )
    
    def forward(self, x_eeg, x_eog=None):
        # EEG特征提取
        eeg_features = self.eeg_feature_extractor(x_eeg)
        
        # EOG特征提取（如果有）
        if hasattr(self, 'eog_feature_extractor') and x_eog is not None:
            eog_features = self.eog_feature_extractor(x_eog)
            # 确保两个特征维度相同
            min_len = min(eeg_features.size(2), eog_features.size(2))
            eeg_features = eeg_features[:, :, :min_len]
            eog_features = eog_features[:, :, :min_len]
            # 融合特征
            fused = torch.cat([eeg_features, eog_features], dim=1)
        else:
            fused = eeg_features
        
        # 特征融合
        fused = self.fusion_layer(fused)
        
        # 维度调整
        fused = self.dim_adjust_tce(fused)
        
        # 确保长度为80（与你当前的TCE模块兼容）
        if fused.size(2) != 80:
            fused = F.adaptive_avg_pool1d(fused, 80)
        
        # 转换维度以适应Transformer
        fused = fused.transpose(1, 2)  # [B, C, T] -> [B, T, C]
        
        # TCE处理
        encoded = self.tce(fused)
        
        # 全局池化得到特征向量
        features = torch.mean(encoded, dim=1)
        
        # 分类
        logits = self.classifier(features)
        
        return logits