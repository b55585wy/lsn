import torch
import torch.nn as nn
import torch.nn.functional as F

class UBlock(nn.Module):
    """U型编码块 (从 salient_sleep.py 复制)"""
    def __init__(self, in_ch, out_ch, mid_ch=None, depth=2):
        super().__init__()
        if mid_ch is None:
            mid_ch = out_ch
        
        self.depth = depth
        self.in_conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True)
        )
        
        self.encoder_modules = nn.ModuleList()
        for i in range(depth-1):
            encoder = nn.Sequential(
                nn.Conv1d(out_ch if i == 0 else mid_ch, mid_ch, kernel_size=3, padding=1),
                nn.BatchNorm1d(mid_ch),
                nn.ReLU(inplace=True)
            )
            self.encoder_modules.append(encoder)
            
        self.middle = nn.Sequential(
            nn.Conv1d(mid_ch, mid_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(mid_ch),
            nn.ReLU(inplace=True)
        )
            
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
        x = self.in_conv(x)
        identity = x
        
        encoder_outputs = []
        encoder_sizes = []
        current = x
        
        for i, encoder in enumerate(self.encoder_modules):
            current = encoder(current)
            encoder_outputs.append(current)
            encoder_sizes.append(current.size(2))
            if i < len(self.encoder_modules) - 1:
                current = F.max_pool1d(current, kernel_size=2)
                
        current = self.middle(current)
        
        for i, decoder in enumerate(self.decoder_modules):
            skip_connection = encoder_outputs[-(i+1)]
            current = F.interpolate(current, size=encoder_sizes[-(i+1)], mode='linear', align_corners=True)
            current = torch.cat([current, skip_connection], dim=1)
            current = decoder(current)
        
        if current.size(2) != identity.size(2):
            current = F.interpolate(current, size=identity.size(2), mode='linear', align_corners=True)
        
        return current + identity

class InceptionBlock1D(nn.Module):
    """
    一维 Inception 模块.
    """
    def __init__(self, in_channels, channels_config, final_out_channels):
        super().__init__()
        self.branch1_1x1 = nn.Sequential(
            nn.Conv1d(in_channels, channels_config['b1_1x1'], kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(channels_config['b1_1x1']),
            nn.ReLU(inplace=True)
        )
        self.branch2_3x3 = nn.Sequential(
            nn.Conv1d(in_channels, channels_config['b2_3x3'], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(channels_config['b2_3x3']),
            nn.ReLU(inplace=True)
        )
        self.branch3_5x5 = nn.Sequential(
            nn.Conv1d(in_channels, channels_config['b3_5x5'], kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(channels_config['b3_5x5']),
            nn.ReLU(inplace=True)
        )
        self.branch4_pool = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, channels_config['b4_pool'], kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(channels_config['b4_pool']),
            nn.ReLU(inplace=True)
        )
        concat_out_channels = (
            channels_config['b1_1x1'] +
            channels_config['b2_3x3'] +
            channels_config['b3_5x5'] +
            channels_config['b4_pool']
        )
        self.projection = nn.Sequential(
            nn.Conv1d(concat_out_channels, final_out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(final_out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out_b1 = self.branch1_1x1(x)
        out_b2 = self.branch2_3x3(x)
        out_b3 = self.branch3_5x5(x)
        out_b4 = self.branch4_pool(x)
        concatenated_features = torch.cat([out_b1, out_b2, out_b3, out_b4], dim=1)
        output = self.projection(concatenated_features)
        return output

class SalientSleepInception(nn.Module):
    """SalientSleep 模型变体，使用 Inception 模块替代 MultiScaleExtraction"""
    def __init__(self, use_eog=True, base_filters=16, n_classes=5):
        super().__init__()
        
        self.use_eog = use_eog # 保存 use_eog 状态

        inception_in_channels = base_filters * 32
        inception_final_out_channels = base_filters * 32
        target_concat_channels = 2 * inception_final_out_channels 
        ch_b_val = target_concat_channels // 4
        cfg_b1 = ch_b_val
        cfg_b2 = ch_b_val
        cfg_b3 = ch_b_val
        cfg_b4_pool = target_concat_channels - (cfg_b1 + cfg_b2 + cfg_b3)
        inception_channels_config = {
            'b1_1x1': cfg_b1, 'b2_3x3': cfg_b2,
            'b3_5x5': cfg_b3, 'b4_pool': cfg_b4_pool
        }

        self.eeg_feature_extractor = nn.Sequential(
            UBlock(1, base_filters, depth=2),
            nn.Conv1d(base_filters, base_filters*2, kernel_size=1),
            nn.BatchNorm1d(base_filters*2), nn.ReLU(), nn.MaxPool1d(10),
            UBlock(base_filters*2, base_filters*4, depth=3),
            nn.Conv1d(base_filters*4, base_filters*8, kernel_size=1),
            nn.BatchNorm1d(base_filters*8), nn.ReLU(), nn.MaxPool1d(8),
            UBlock(base_filters*8, base_filters*16, depth=3),
            nn.Conv1d(base_filters*16, base_filters*32, kernel_size=1),
            nn.BatchNorm1d(base_filters*32), nn.ReLU(), nn.MaxPool1d(6),
            InceptionBlock1D(
                in_channels=inception_in_channels,
                channels_config=inception_channels_config,
                final_out_channels=inception_final_out_channels
            )
        )
        
        if self.use_eog:
            self.eog_feature_extractor = nn.Sequential(
                UBlock(1, base_filters, depth=2),
                nn.Conv1d(base_filters, base_filters*2, kernel_size=1),
                nn.BatchNorm1d(base_filters*2), nn.ReLU(), nn.MaxPool1d(10),
                UBlock(base_filters*2, base_filters*4, depth=3),
                nn.Conv1d(base_filters*4, base_filters*8, kernel_size=1),
                nn.BatchNorm1d(base_filters*8), nn.ReLU(), nn.MaxPool1d(8),
                UBlock(base_filters*8, base_filters*16, depth=3),
                nn.Conv1d(base_filters*16, base_filters*32, kernel_size=1),
                nn.BatchNorm1d(base_filters*32), nn.ReLU(), nn.MaxPool1d(6),
                InceptionBlock1D(
                    in_channels=inception_in_channels,
                    channels_config=inception_channels_config,
                    final_out_channels=inception_final_out_channels
                )
            )
        else:
            self.eog_feature_extractor = None 
        
        fusion_ch_multiplier = 2 if self.use_eog and self.eog_feature_extractor is not None else 1
        fusion_input_ch = inception_final_out_channels * fusion_ch_multiplier

        self.fusion_layer = nn.Sequential(
            nn.Conv1d(fusion_input_ch, base_filters*16, kernel_size=1),
            nn.BatchNorm1d(base_filters*16), nn.ReLU(),
            nn.Conv1d(base_filters*16, base_filters*8, kernel_size=1),
            nn.BatchNorm1d(base_filters*8), nn.ReLU()
        )
        
        self.dim_adjust_tce = nn.Sequential(
            nn.Conv1d(base_filters*8, 80, kernel_size=1),
            nn.BatchNorm1d(80), nn.ReLU()
        )
        
        self.tce = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=80, nhead=8, dim_feedforward=120,
                dropout=0.1, batch_first=True
            ), 
            num_layers=2
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(80, 40), nn.ReLU(),
            nn.Dropout(0.1), nn.Linear(40, n_classes)
        )
    
    def forward(self, x_eeg, x_eog=None):
        eeg_features = self.eeg_feature_extractor(x_eeg)
        
        if self.use_eog and self.eog_feature_extractor is not None and x_eog is not None:
            eog_features = self.eog_feature_extractor(x_eog)
            min_len = min(eeg_features.size(2), eog_features.size(2))
            eeg_features = eeg_features[:, :, :min_len]
            eog_features = eog_features[:, :, :min_len]
            fused = torch.cat([eeg_features, eog_features], dim=1)
        else:
            fused = eeg_features
        
        fused = self.fusion_layer(fused)
        fused = self.dim_adjust_tce(fused)
        
        if fused.size(2) != 80:
            fused = F.adaptive_avg_pool1d(fused, 80)
        
        fused = fused.transpose(1, 2)
        encoded = self.tce(fused)
        features = torch.mean(encoded, dim=1)
        logits = self.classifier(features)
        
        return logits

# 示例用法 (可选，用于快速测试模型结构)
if __name__ == '__main__':
    # 测试模型
    dummy_eeg = torch.randn(2, 1, 3000) # Batch_size=2, Channels=1, Seq_len=3000
    dummy_eog = torch.randn(2, 1, 3000)

    # 不使用 EOG
    model_no_eog = SalientSleepInception(use_eog=False, base_filters=16, n_classes=5)
    output_no_eog = model_no_eog(dummy_eeg)
    print("Output shape (no EOG):", output_no_eog.shape) # 期望: torch.Size([2, 5])

    # 使用 EOG
    model_with_eog = SalientSleepInception(use_eog=True, base_filters=16, n_classes=5)
    output_with_eog = model_with_eog(dummy_eeg, dummy_eog)
    print("Output shape (with EOG):", output_with_eog.shape) # 期望: torch.Size([2, 5])

    # 检查参数数量
    total_params_no_eog = sum(p.numel() for p in model_no_eog.parameters() if p.requires_grad)
    print(f"Trainable parameters (no EOG): {total_params_no_eog}")
    total_params_with_eog = sum(p.numel() for p in model_with_eog.parameters() if p.requires_grad)
    print(f"Trainable parameters (with EOG): {total_params_with_eog}") 