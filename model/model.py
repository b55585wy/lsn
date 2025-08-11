import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from copy import deepcopy
import einops
from model.mea import MEA  # renamed from EMA1D
from model.idc import InceptionDWConv1d, MultiBranchStandardCNN
import numpy as np

# from functools import partial # 如果需要

# 从 vmamba.py 导入 mamba_init 和 selective_scan_fn
# 假设 vmamba.py 在 model_mamba.lib_mamba 路径下
try:
    from model_mamba.lib_mamba.vmamba import mamba_init
    from model_mamba.lib_mamba.csms6s import selective_scan_fn
except ImportError as e:
    # Fallback if the exact path is different or for local testing
    print(f"Original ImportError in model/model.py: {e}")
    print("Warning: Could not import mamba_init or selective_scan_fn from model_mamba.lib_mamba.vmamba/csms6s. Please check paths.")
    # For robust import, ensure model_mamba.lib_mamba is in sys.path or adjust relative imports
    # This is a critical dependency, so re-raise if not found after trying common locations.
    raise ImportError("Critical mamba components (mamba_init, selective_scan_fn) not found. Ensure model_mamba.lib_mamba is in PYTHONPATH or accessible.")


def augment_signal(x, noise_factor=0.015, scale_range=(0.85, 1.15), time_shift_ratio=0.05):
    noise = torch.randn_like(x) * noise_factor * x.std(dim=-1, keepdim=True)
    x = x + noise
    scale = torch.FloatTensor(x.size(0)).uniform_(*scale_range).to(x.device).view(-1, 1, 1)
    x = x * scale
    max_shift = int(x.size(-1) * time_shift_ratio)
    if max_shift > 0:
        shifts = torch.randint(-max_shift, max_shift, (x.size(0),)).to(x.device)
        x_out = torch.zeros_like(x)
        for i in range(x.size(0)):
            x_out[i] = torch.roll(x[i], shifts=shifts[i].item(), dims=-1)
        x = x_out
    return x


def normalize_signal(x):
    return (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-6)


class EfficientAdditiveAttention(nn.Module):
    def __init__(self, d_model, afr_reduced_cnn_size, dropout=0.2):
        super(EfficientAdditiveAttention, self).__init__()
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)

        self.delta_attn = nn.Linear(d_model, 1)
        self.sigma_attn = nn.Linear(d_model, 1)
        self.theta_attn = nn.Linear(d_model, 1)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(d_model)

    def forward(self, query, key, value):
        Q = self.query_proj(query)
        K = self.key_proj(key)
        V = self.value_proj(value)

        e = torch.tanh(Q.unsqueeze(2) + K.unsqueeze(1))
        base_scores = e.sum(dim=-1) / self.scale
        delta_scores = self.delta_attn(e).squeeze(-1)
        sigma_scores = self.sigma_attn(e).squeeze(-1)
        theta_scores = self.theta_attn(e).squeeze(-1)
        combined_scores = base_scores + delta_scores + sigma_scores + theta_scores

        attn = torch.softmax(combined_scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V)
        return out


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerOutput(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerOutput, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, afr_reduced_cnn_size, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer_output = clones(SublayerOutput(size, dropout), 2)
        self.size = size

    def forward(self, x):
        x = self.sublayer_output[0](x, lambda x: self.self_attn(x, x, x))
        return self.sublayer_output[1](x, self.feed_forward)


class TCE(nn.Module):
    def __init__(self, layer, N):
        super(TCE, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels=128, branch_ratios=None, kernel_sizes=None):
        super(InceptionBlock, self).__init__()
        # InceptionDWConv1d 默认 out_channels = 128
        self.branch = InceptionDWConv1d(in_channels, out_channels,
                                        branch_ratios=branch_ratios,
                                        kernel_sizes=kernel_sizes)

    def forward(self, x):
        return self.branch(x)


# 新增：用于消融实验的标准CNN块
class StandardCNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels=128, kernel_size=7, stride=1, padding=3):
        super(StandardCNNBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)


# 新增：细粒度门控机制模块
class FineGrainedGatingModule(nn.Module):
    def __init__(self, channels, reduction_ratio=4):
        super(FineGrainedGatingModule, self).__init__()
        self.channels = channels
        
        # 特征压缩
        self.squeeze = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(channels // reduction_ratio, channels)
        )
        
        # 空间门控
        self.spatial_gate = nn.Sequential(
            nn.Conv1d(channels, channels // reduction_ratio, kernel_size=1),
            nn.BatchNorm1d(channels // reduction_ratio),
            nn.ReLU(),
            nn.Conv1d(channels // reduction_ratio, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 通道门控
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, channels // reduction_ratio, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(channels // reduction_ratio, channels, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # 输入x形状: [B, C, T]
        batch_size, channels, seq_len = x.size()
        
        # 通道门控
        channel_att = self.channel_gate(x)
        
        # 空间门控
        spatial_att = self.spatial_gate(x)
        
        # 动态特征压缩
        x_avg = torch.mean(x, dim=2)  # [B, C]
        feature_weights = self.squeeze(x_avg).view(batch_size, channels, 1)  # [B, C, 1]
        feature_weights = torch.sigmoid(feature_weights)
        
        # 应用门控
        gated_output = x * channel_att * spatial_att * feature_weights
        
        return gated_output


# 新增：记忆增强模块
class MemoryEnhancedModule(nn.Module):
    def __init__(self, embed_dim, memory_size=64, topk=3):
        super(MemoryEnhancedModule, self).__init__()
        self.embed_dim = embed_dim
        self.memory_size = memory_size
        self.topk = topk
        
        # 记忆矩阵
        self.memory = nn.Parameter(torch.randn(memory_size, embed_dim))
        nn.init.kaiming_uniform_(self.memory)
        
        # 记忆读取层
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        
        # 特征融合层
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU()
        )
        
    def forward(self, x):
        # x形状: [B, S, D] (batch, seq_len, dim)
        batch_size, seq_len, dim = x.size()
        
        # 生成查询向量
        query = self.query_proj(x)  # [B, S, D]
        
        # 计算查询与记忆的相似度
        query = query.view(batch_size * seq_len, self.embed_dim)  # [B*S, D]
        sim = torch.matmul(query, self.memory.transpose(0, 1))  # [B*S, M]
        
        # 选择topk最相似的记忆
        topk_sim, topk_idx = torch.topk(sim, k=self.topk, dim=1)  # [B*S, k]
        topk_sim_weights = F.softmax(topk_sim, dim=1).unsqueeze(2)  # [B*S, k, 1]
        
        # 检索记忆
        retrieved_memory = self.memory[topk_idx]  # [B*S, k, D]
        
        # 加权求和
        memory_output = torch.sum(retrieved_memory * topk_sim_weights, dim=1)  # [B*S, D]
        memory_output = memory_output.view(batch_size, seq_len, self.embed_dim)  # [B, S, D]
        
        # 融合原始特征和记忆特征
        fused_output = self.fusion(torch.cat([x, memory_output], dim=2))  # [B, S, D]
        
        return fused_output


# 新增：跨Epoch注意力模块
class CrossEpochAttention(nn.Module):
    # Add dummy CrossEpochAttention if not already present or handle import
    def __init__(self, d_model, num_heads=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        # Simplified for brevity, actual implementation might be more complex
        self.dummy_layer = nn.Linear(d_model, d_model) 
        print("Warning: Using a dummy CrossEpochAttention implementation.")

    def forward(self, x, seq_length=5):
        return self.dummy_layer(x)


class Vmamba1DBlock(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        d_conv=3,
        conv_bias=True,
        bias=False, # Bias for linear layers
        act_layer_str="silu", # Activation: "silu", "gelu", etc.
        dropout=0.0,
        k_group_1d=1, # 1 for unidirectional, 2 for bidirectional
        # dt_init parameters from vmamba.py (mamba_init.dt_init)
        dt_scale=1.0,
        dt_init_method="random", # "random" or "constant"
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        factory_kwargs=None, # For device/dtype, pass as dict e.g. {"device": "cuda", "dtype": torch.float32}
    ):
        super().__init__()
        if factory_kwargs is None:
            factory_kwargs = {}
        
        self.d_model = d_model
        self.d_state = d_state
        self.ssm_ratio = ssm_ratio
        self.d_inner = int(ssm_ratio * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.d_conv = d_conv
        self.k_group_1d = k_group_1d
        assert k_group_1d in [1, 2], "k_group_1d must be 1 (unidirectional) or 2 (bidirectional)"

        if act_layer_str.lower() == "silu":
            self.act_layer = nn.SiLU()
        elif act_layer_str.lower() == "gelu":
            self.act_layer = nn.GELU()
        else:
            self.act_layer = nn.SiLU() 
            print(f"Warning: Unknown act_layer_str '{act_layer_str}', defaulting to SiLU.")

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner, 
            padding=(d_conv - 1) // 2,
            bias=conv_bias,
            **factory_kwargs,
        )
        
        # x_proj_weights: (k_group_1d, dt_rank + 2*d_state, d_inner)
        self.x_proj_weights = nn.Parameter(
            torch.empty(self.k_group_1d, self.dt_rank + 2 * self.d_state, self.d_inner, **factory_kwargs)
        )
        nn.init.xavier_uniform_(self.x_proj_weights)

        A_logs, Ds, dt_projs_weight, dt_projs_bias = mamba_init.init_dt_A_D(
            d_state=self.d_state,
            dt_rank=self.dt_rank,
            d_inner=self.d_inner,
            dt_scale=dt_scale,
            dt_init=dt_init_method,
            dt_min=dt_min,
            dt_max=dt_max,
            dt_init_floor=dt_init_floor,
            k_group=self.k_group_1d, # Use k_group_1d here
        )
        self.A_logs = A_logs 
        self.Ds = Ds         
        self.dt_projs_weight = dt_projs_weight 
        self.dt_projs_bias = dt_projs_bias     
        
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x):
        B, L, D = x.shape
        
        xz = self.in_proj(x) 
        x_conv, z = xz.chunk(2, dim=-1) 

        x_conv = x_conv.transpose(1, 2).contiguous() 
        x_conv = self.conv1d(x_conv) 
        x_conv = self.act_layer(x_conv) 
        
        u = x_conv.transpose(1, 2).contiguous() 
        y_scan_list = []

        for k_idx in range(self.k_group_1d):
            current_u = u
            if k_idx == 1: 
                current_u = torch.flip(u, dims=[1])
            
            x_proj_weights_k = self.x_proj_weights[k_idx] 
            x_dbl_k = F.linear(current_u, x_proj_weights_k) 
            
            dt_input_k, B_param_k, C_param_k = torch.split(
                x_dbl_k, [self.dt_rank, self.d_state, self.d_state], dim=-1
            )

            _dt_proj_w = self.dt_projs_weight[k_idx] 
            _dt_proj_b = self.dt_projs_bias[k_idx]   
            
            delta_k_reshaped = dt_input_k.reshape(B * L, self.dt_rank)
            delta_val_pre_softplus = F.linear(delta_k_reshaped, _dt_proj_w, _dt_proj_b) 
            delta_k_final = F.softplus(delta_val_pre_softplus.view(B, L, self.d_inner))

            start_idx = k_idx * self.d_inner
            end_idx = (k_idx + 1) * self.d_inner
            
            A_k = -torch.exp(self.A_logs[start_idx:end_idx, :].float()) 
            D_k = self.Ds[start_idx:end_idx].float() 

            # Determine backend based on device
            backend = "torch" if current_u.device.type == "cpu" else None

            y_k = selective_scan_fn(
                current_u.transpose(1, 2).contiguous(),
                delta_k_final.transpose(1, 2).contiguous(),
                A_k,
                B_param_k.unsqueeze(1).transpose(2,3).contiguous(),
                C_param_k.unsqueeze(1).transpose(2,3).contiguous(),
                D_k,
                delta_bias=None,
                delta_softplus=False,
                backend=backend
            )

            if k_idx == 1: 
                y_k = torch.flip(y_k, dims=[2]) 
            
            y_scan_list.append(y_k.transpose(1,2)) 

        if self.k_group_1d == 1:
            y_scanned = y_scan_list[0]
        else: 
            y_scanned = y_scan_list[0] + y_scan_list[1] 
        
        y_gated = y_scanned * z 
        output = self.out_proj(y_gated) 
        output = self.dropout(output)
        
        return output


class BioSleepX(nn.Module):
    def __init__(self, use_msea=False, use_gabor=False, 
                 # --- 新增的消融实验控制参数 ---
                 use_eog=True,
                 use_augmentation=True
                ):
        super(BioSleepX, self).__init__()
        N = 2
        d_model = 80
        d_ff = 120
        dropout = 0.1
        num_classes = 5
        afr_reduced_cnn_size = 30

        self.d_model = d_model
        self.use_eog = use_eog
        self.use_augmentation = use_augmentation

        self.eeg_incep = InceptionBlock(1) # EEG InceptionDWConv1d default out_channels = 128

        if self.use_eog:
            self.eog_incep = InceptionBlock(1) # EOG InceptionDWConv1d default out_channels = 128
            fusion_input_channels = 256 # EEG (128) + EOG (128)
        else:
            self.eog_incep = None
            fusion_input_channels = 128 # 仅 EEG (128)

        self.fusion_layer = nn.Sequential(
            nn.Conv1d(fusion_input_channels, afr_reduced_cnn_size, kernel_size=1),
            nn.BatchNorm1d(afr_reduced_cnn_size),
            nn.ReLU(),
            MEA(channels=afr_reduced_cnn_size, factor=8) # MEA is part of the original BioSleepX fusion
        )

        self.dim_adjust_tce = nn.Sequential(
            nn.Conv1d(afr_reduced_cnn_size, d_model, kernel_size=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU()
        )

        attn = EfficientAdditiveAttention(d_model, afr_reduced_cnn_size, dropout)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.tce = TCE(EncoderLayer(d_model, deepcopy(attn), deepcopy(ff), afr_reduced_cnn_size, dropout), N)

        self.dynamic_weight = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        ) # This dynamic_weight seems unused in the original forward pass for classification output
          # It produces weights but these weights are not used to combine/scale the features before classifier
          # For now, I will leave it as is. If it was intended for something, the forward pass would need adjustment.

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x_eeg, x_eog=None):
        if self.training and self.use_augmentation:
            x_eeg = augment_signal(x_eeg)
            if self.use_eog and x_eog is not None:
                x_eog = augment_signal(x_eog)

        x_eeg = normalize_signal(x_eeg)
        eeg_features = self.eeg_incep(x_eeg)

        if self.use_eog and self.eog_incep is not None and x_eog is not None:
            x_eog = normalize_signal(x_eog)
            eog_features = self.eog_incep(x_eog)
            # Ensure time dimension alignment if InceptionBlock output length is variable
            min_len = min(eeg_features.size(-1), eog_features.size(-1))
            eeg_features = eeg_features[..., :min_len]
            eog_features = eog_features[..., :min_len]
            fused = torch.cat([eeg_features, eog_features], dim=1)
        else:
            fused = eeg_features

        fused = self.fusion_layer(fused)
        fused = self.dim_adjust_tce(fused)

        if fused.size(-1) != self.d_model: # d_model is 80
            # The original code adaptive_avg_pool1d to 80, which is d_model
            # This was likely intended to ensure the sequence length for TCE is d_model (80)
            fused = F.adaptive_avg_pool1d(fused, self.d_model) 

        fused = fused.transpose(1, 2) # Shape: [B, T_adj, C_adj] -> [B, C_adj, T_adj]
                                      # Original TCE expects [Batch, SeqLen, FeatureDim]
                                      # Here, d_model is the FeatureDim for TCE.
                                      # So, after transpose, it should be [B, T_adj, d_model]
                                      # Where T_adj is self.d_model (80) from adaptive_avg_pool1d
        encoded = self.tce(fused) # TCE processes features of size d_model (EncoderLayer.size)

        # The original dynamic_weight layer produces weights but doesn't seem to use them to scale 'encoded' for the classifier.
        # weights = self.dynamic_weight(encoded.mean(dim=1)) # Example of how it might have been used.
        # If these weights were [B, 2] and encoded.mean(dim=1) was [B, d_model], direct multiplication isn't obvious for weighting.
        # For now, we directly pass the mean of encoded features to the classifier as in the original code.
        
        output_features = encoded.mean(dim=1) # Take mean over the sequence dimension (T_adj)
        output = self.classifier(output_features)
        return output


class BioSleepXSeq(nn.Module):
    def __init__(self, seq_length=5, use_msea=False, use_gabor=False,
                 # --- 新增的消融实验控制参数 ---
                 use_eog=True,
                 use_fine_grained_gating=True,
                 use_mea_in_fusion=True,
                 use_memory_module=True,
                 use_cross_epoch_attn=True,
                 use_seq_tce=True,
                 tce_N=2,
                 seq_tce_N=1,
                 use_augmentation=True,
                 use_simplified_mamba=False,
                 # --- Parameters for Vmamba1DBlock ---
                 use_advanced_mamba=False,
                 use_bilstm=False, # <-- 新增：控制是否使用Bi-LSTM
                 vmamba_d_state=16,
                 vmamba_ssm_ratio=2.0,
                 vmamba_dt_rank="auto",
                 vmamba_d_conv=3,
                 vmamba_conv_bias=True,
                 vmamba_bias=False, # Bias for Vmamba1DBlock's linear layers
                 vmamba_act_layer_str="silu",
                 vmamba_dropout=0.0,
                 vmamba_k_group_1d=1,
                 vmamba_dt_scale=1.0,
                 vmamba_dt_init_method="random",
                 vmamba_dt_min=0.001,
                 vmamba_dt_max=0.1,
                 vmamba_dt_init_floor=1e-4,
                 # --- Configurable dimensions ---
                 afr_reduced_cnn_size_config=30,
                 d_model_config=80,
                 pool_target_len_config=80,
                 # --- Configurable dropout rates ---
                 tce_dropout_config=0.1,
                 classifier_dropout_config=0.1,
                 # --- 新增的频率特征提取控制参数 ---
                 use_gabor_conv=False,
                 use_wavelet_pool=False,
                 # --- 新增的Inception控制参数 ---
                 use_inception_block=True,
                 use_standard_cnn=False, # <-- 新增：控制是否使用标准CNN
                 inception_out_channels=128,
                 inception_branch_ratios=None,
                 inception_kernel_sizes=None,
                 # --- 新增：数据增强控制参数 ---
                 noise_factor_config=0.015
               ,
               # --- 新增：用于任务迁移的参数 ---
               num_classes=5,
               in_channels=1
              ):
       super(BioSleepXSeq, self).__init__()
       N = tce_N
       seq_N = seq_tce_N
       d_model = d_model_config
       afr_reduced_cnn_size = afr_reduced_cnn_size_config
       self.pool_target_len = pool_target_len_config

       d_ff = 120
       # Use tce_dropout_config for TCE internal dropout
       tce_internal_dropout = tce_dropout_config
       # num_classes is now a parameter
       
       self.d_model = d_model
       self.seq_length = seq_length
       # 保存消融控制参数的状态
       self.use_eog = use_eog
       self.use_fine_grained_gating = use_fine_grained_gating
       self.use_memory_module = use_memory_module
       self.use_cross_epoch_attn = use_cross_epoch_attn
       self.use_seq_tce = use_seq_tce
       self.use_augmentation = use_augmentation
       self.use_simplified_mamba = use_simplified_mamba # Save Vmamba usage option
       self.use_advanced_mamba = use_advanced_mamba # Save Vmamba usage option
       self.use_gabor_conv = use_gabor_conv
       self.use_wavelet_pool = use_wavelet_pool
       self.use_inception_block = use_inception_block
       self.use_standard_cnn = use_standard_cnn # 保存标准CNN使用状态
       self.use_bilstm = use_bilstm # 保存Bi-LSTM使用状态
       self.noise_factor = noise_factor_config # 保存噪声因子

       # 导入Gabor和Wavelet模块
       if self.use_gabor_conv:
           from model.gabor_conv import GaborConv1d
       if self.use_wavelet_pool:
           from model.wavelet_modules import WaveletPool1d

       # 导入Gabor和Wavelet模块
       if self.use_gabor_conv:
           from model.gabor_conv import GaborConv1d
       if self.use_wavelet_pool:
           from model.wavelet_modules import WaveletPool1d

       # EEG特征提取模块
       if self.use_standard_cnn:
           # For a fair ablation, use a multi-branch standard CNN
           self.eeg_incep = MultiBranchStandardCNN(in_channels, out_channels=inception_out_channels,
                                                   branch_ratios=inception_branch_ratios,
                                                   kernel_sizes=inception_kernel_sizes)
       else:
           self.eeg_incep = InceptionBlock(in_channels,
                                           out_channels=inception_out_channels,
                                           branch_ratios=inception_branch_ratios,
                                           kernel_sizes=inception_kernel_sizes) if self.use_inception_block else None
       self.eeg_gabor = GaborConv1d(in_channels, 64, kernel_size=16, padding=8) if self.use_gabor_conv else None
       self.eeg_wavelet = WaveletPool1d(in_channels, out_channels=64) if self.use_wavelet_pool else None
       
       # EOG特征提取模块 (条件创建)
       if self.use_standard_cnn:
           self.eog_incep = MultiBranchStandardCNN(in_channels, out_channels=inception_out_channels,
                                                   branch_ratios=inception_branch_ratios,
                                                   kernel_sizes=inception_kernel_sizes) if self.use_eog else None
       else:
           self.eog_incep = InceptionBlock(in_channels,
                                           out_channels=inception_out_channels,
                                           branch_ratios=inception_branch_ratios,
                                           kernel_sizes=inception_kernel_sizes) if (self.use_eog and self.use_inception_block) else None
       self.eog_gabor = GaborConv1d(in_channels, 64, kernel_size=16, padding=8) if (self.use_eog and self.use_gabor_conv) else None
       self.eog_wavelet = WaveletPool1d(in_channels, out_channels=64) if (self.use_eog and self.use_wavelet_pool) else None
       
       # 计算融合层输入通道数
       channels_per_signal = 0
       if self.use_inception_block or self.use_standard_cnn:
           channels_per_signal += inception_out_channels # 使用可配置的输出通道数
       if self.use_gabor_conv:
           channels_per_signal += 64
       if self.use_wavelet_pool:
           channels_per_signal += 64

       # 确保至少有一种特征提取方式被启用
       if channels_per_signal == 0:
           raise ValueError("At least one feature extraction method (Inception, Gabor, Wavelet) must be enabled.")

       if self.use_eog:
           fusion_input_channels = channels_per_signal * 2
       else:
           fusion_input_channels = channels_per_signal
       
       # Fusion Layer (MEA可配置)
       fusion_modules = [
           nn.Conv1d(fusion_input_channels, afr_reduced_cnn_size, kernel_size=1),
           nn.BatchNorm1d(afr_reduced_cnn_size),
           nn.ReLU()
       ]
       if use_mea_in_fusion:
           fusion_modules.append(MEA(channels=afr_reduced_cnn_size, factor=8))
       self.fusion_layer = nn.Sequential(*fusion_modules)
       
       # 新增：细粒度门控，用于特征增强 (条件创建)
       if self.use_fine_grained_gating:
           self.fine_grained_gating = FineGrainedGatingModule(afr_reduced_cnn_size)
       else:
           self.fine_grained_gating = None
       
       self.dim_adjust_tce = nn.Sequential(
           nn.Conv1d(afr_reduced_cnn_size, d_model, kernel_size=1),
           nn.BatchNorm1d(d_model),
           nn.ReLU()
       )
       
       # 单epoch TCE编码器
       attn = EfficientAdditiveAttention(d_model, afr_reduced_cnn_size, tce_internal_dropout)
       ff = PositionwiseFeedForward(d_model, d_ff, tce_internal_dropout)
       self.tce = TCE(EncoderLayer(d_model, deepcopy(attn), deepcopy(ff), afr_reduced_cnn_size, tce_internal_dropout), N)
       
       # 新增：记忆增强模块 (条件创建)
       if self.use_memory_module:
           self.memory_module = MemoryEnhancedModule(d_model)
       else:
           self.memory_module = None
       
       # 新增：跨Epoch注意力模块 (条件创建)
       if self.use_cross_epoch_attn:
           self.cross_epoch_attn = CrossEpochAttention(d_model)
       else:
           self.cross_epoch_attn = None
       
       self.simplified_mamba_encoder = None
       self.advanced_mamba_encoder = None
       self.seq_tce = None
       self.bilstm_encoder = None # <-- 新增：初始化Bi-LSTM编码器

       if self.use_bilstm:
           self.bilstm_encoder = nn.LSTM(
               input_size=d_model,
               hidden_size=d_model,
               num_layers=2, # 可以设为可配置参数
               batch_first=True,
               bidirectional=True
           )
           # LSTM的输出是2*d_model，需要一个线性层映射回d_model
           self.lstm_out_proj = nn.Linear(d_model * 2, d_model)
           print(f"BioSleepXSeq: Using Bi-LSTM for sequence encoding.")
       elif self.use_advanced_mamba:
           self.advanced_mamba_encoder = Vmamba1DBlock(
               d_model=d_model,
               d_state=vmamba_d_state,
               ssm_ratio=vmamba_ssm_ratio,
               dt_rank=vmamba_dt_rank,
               d_conv=vmamba_d_conv,
               conv_bias=vmamba_conv_bias,
               bias=vmamba_bias,
               act_layer_str=vmamba_act_layer_str,
               dropout=vmamba_dropout,
               k_group_1d=vmamba_k_group_1d,
               dt_scale=vmamba_dt_scale,
               dt_init_method=vmamba_dt_init_method,
               dt_min=vmamba_dt_min,
               dt_max=vmamba_dt_max,
               dt_init_floor=vmamba_dt_init_floor
           )
           self.simplified_mamba_encoder = None
           self.seq_tce = None
           print(f"BioSleepXSeq: Using Vmamba1DBlock for sequence encoding (k_group_1d={vmamba_k_group_1d}).")
       elif self.use_simplified_mamba:
           self.simplified_mamba_encoder = SimplifiedMamba1DLayer(d_model=d_model)
           self.seq_tce = None
           print(f"BioSleepXSeq: Using SimplifiedMamba1DLayer for sequence encoding.")
       elif self.use_seq_tce and seq_N > 0:
           seq_attn = EfficientAdditiveAttention(d_model, d_model, tce_internal_dropout)
           seq_ff = PositionwiseFeedForward(d_model, d_ff, tce_internal_dropout)
           self.seq_tce = TCE(EncoderLayer(d_model, deepcopy(seq_attn), deepcopy(seq_ff), d_model, tce_internal_dropout), seq_N)
           print(f"BioSleepXSeq: Using TCE for sequence encoding.")
       else:
           self.seq_tce = None
           print(f"BioSleepXSeq: No final sequence encoder (neither TCE nor Mamba).")
       
       # 分类器
       self.classifier = nn.Sequential(
           nn.Linear(d_model, d_model // 2),
           nn.ReLU(),
           nn.Dropout(classifier_dropout_config), # Use classifier_dropout_config
           nn.Linear(d_model // 2, num_classes)
       )
        
    def forward(self, x_eeg_seq, x_eog_seq=None):
        """
        参数:
            x_eeg_seq: 序列EEG数据 [batch_size, seq_len, channels, time]
            x_eog_seq: 序列EOG数据 [batch_size, seq_len, channels, time] (可选)
        """
        # Handle case where a single batch item is passed (e.g., during profiling)
        if x_eeg_seq.ndim == 3:
            x_eeg_seq = x_eeg_seq.unsqueeze(0)
        if x_eog_seq is not None and x_eog_seq.ndim == 3:
            x_eog_seq = x_eog_seq.unsqueeze(0)
        # 确保输入是 (B, S, C, T)
        if x_eeg_seq.ndim == 4 and x_eeg_seq.size(3) == 1:
            x_eeg_seq = x_eeg_seq.permute(0, 1, 3, 2)
        if x_eog_seq is not None and x_eog_seq.ndim == 4 and x_eog_seq.size(3) == 1:
            x_eog_seq = x_eog_seq.permute(0, 1, 3, 2)

        batch_size, seq_len = x_eeg_seq.size(0), x_eeg_seq.size(1)
        
        # 重塑输入以批量处理
        x_eeg_flat = x_eeg_seq.contiguous().view(-1, x_eeg_seq.size(2), x_eeg_seq.size(3))
        
        # EOG处理，如果启用
        if self.use_eog and x_eog_seq is not None:
            x_eog_flat = x_eog_seq.contiguous().view(-1, x_eog_seq.size(2), x_eog_seq.size(3))
            x_eog_flat = normalize_signal(x_eog_flat)
            if self.training and self.use_augmentation:
                x_eog_flat = augment_signal(x_eog_flat, noise_factor=self.noise_factor)
        else:
            x_eog_flat = None
            
        # EEG处理和增强
        x_eeg_flat = normalize_signal(x_eeg_flat)
        if self.training and self.use_augmentation:
            # The original BioSleepXSeq had augment_signal commented out.
            # Adding it here under the control of self.use_augmentation.
            x_eeg_flat = augment_signal(x_eeg_flat, noise_factor=self.noise_factor)
            
        # 特征提取
        eeg_features_list = []
        if self.eeg_incep is not None:
            eeg_features_list.append(self.eeg_incep(x_eeg_flat))
        if self.eeg_gabor is not None:
            eeg_features_list.append(self.eeg_gabor(x_eeg_flat))
        if self.eeg_wavelet is not None:
            eeg_features_list.append(self.eeg_wavelet(x_eeg_flat))
        
        # 确保至少有一种特征被提取
        if not eeg_features_list:
            raise ValueError("No EEG feature extraction module is enabled.")

        # 确保所有EEG特征的时间维度对齐
        min_len_eeg = min([f.size(-1) for f in eeg_features_list])
        eeg_features_aligned = [f[..., :min_len_eeg] for f in eeg_features_list]
        eeg_features = torch.cat(eeg_features_aligned, dim=1)

        if self.use_eog and x_eog_flat is not None:
            eog_features_list = []
            if self.eog_incep is not None:
                eog_features_list.append(self.eog_incep(x_eog_flat))
            if self.eog_gabor is not None:
                eog_features_list.append(self.eog_gabor(x_eog_flat))
            if self.eog_wavelet is not None:
                eog_features_list.append(self.eog_wavelet(x_eog_flat))
            
            if not eog_features_list:
                raise ValueError("EOG is enabled but no EOG feature extraction module is enabled.")

            # 确保所有EOG特征的时间维度对齐
            min_len_eog = min([f.size(-1) for f in eog_features_list])
            eog_features_aligned = [f[..., :min_len_eog] for f in eog_features_list]
            eog_features = torch.cat(eog_features_aligned, dim=1)

            # 确保EEG和EOG特征的时间维度对齐
            min_len_fused = min(eeg_features.size(-1), eog_features.size(-1))
            eeg_features = eeg_features[..., :min_len_fused]
            eog_features = eog_features[..., :min_len_fused]
            fused = torch.cat([eeg_features, eog_features], dim=1)
        else:
            fused = eeg_features

        fused = self.fusion_layer(fused)
        
        # 应用细粒度门控 (如果启用)
        if self.fine_grained_gating is not None:
            fused = self.fine_grained_gating(fused)
        
        fused = self.dim_adjust_tce(fused)
        
        if fused.size(-1) != self.pool_target_len: # 目标时间/特征维度调整
            fused = F.adaptive_avg_pool1d(fused, self.pool_target_len)
            
        fused = fused.transpose(1, 2)
        encoded = self.tce(fused)

        epoch_features = encoded.mean(dim=1)
        seq_features = epoch_features.view(batch_size, seq_len, self.d_model)
        
        current_features = seq_features

        if self.memory_module is not None:
            current_features = self.memory_module(current_features)
        
        if self.cross_epoch_attn is not None:
            current_features = self.cross_epoch_attn(current_features, seq_length=seq_len)
        
        # 应用序列编码器 (Bi-LSTM, Mamba 或 TCE)
        if self.bilstm_encoder is not None:
            lstm_out, _ = self.bilstm_encoder(current_features)
            enhanced_seq = self.lstm_out_proj(lstm_out)
        elif self.advanced_mamba_encoder is not None:
            enhanced_seq = self.advanced_mamba_encoder(current_features)
        elif self.simplified_mamba_encoder is not None:
            enhanced_seq = self.simplified_mamba_encoder(current_features)
        elif self.seq_tce is not None:
            enhanced_seq = self.seq_tce(current_features)
        else:
            enhanced_seq = current_features
            
        logits = self.classifier(enhanced_seq)
        
        return logits, enhanced_seq


class BioSleepXContrastive(nn.Module):
    def __init__(self, seq_length=5, use_msea=False, use_gabor=False, proj_dim=128,
                 # --- 新增的消融实验控制参数 ---
                 use_eog=True,                     # 是否使用EOG通路
                 use_fine_grained_gating=True,     # 是否使用细粒度门控
                 use_mea_in_fusion=True,           # 是否在融合层使用MEA
                 use_memory_module=True,           # 是否使用记忆增强模块
                 use_cross_epoch_attn=True,        # 是否使用跨Epoch注意力
                 use_seq_tce=True,                 # 是否使用序列TCE
                 tce_N=2,                          # 单epoch TCE的层数
                 seq_tce_N=1                       # 序列TCE的层数
                ):
        super(BioSleepXContrastive, self).__init__()
        N = tce_N  # 使用传入的参数
        seq_N = seq_tce_N # 使用传入的参数
        d_model = 80
        d_ff = 120
        dropout = 0.1
        num_classes = 5
        afr_reduced_cnn_size = 30
        
        self.d_model = d_model
        self.seq_length = seq_length
        self.proj_dim = proj_dim
        # 保存消融控制参数的状态
        self.use_eog = use_eog
        self.use_fine_grained_gating = use_fine_grained_gating
        self.use_memory_module = use_memory_module
        self.use_cross_epoch_attn = use_cross_epoch_attn
        self.use_seq_tce = use_seq_tce

        # EEG Inception Block (总是需要)
        self.eeg_incep = InceptionBlock(1)
        
        # EOG Inception Block (条件创建)
        if self.use_eog:
            self.eog_incep = InceptionBlock(1)
            fusion_input_channels = 256 # EEG (128) + EOG (128)
        else:
            self.eog_incep = None
            fusion_input_channels = 128 # 仅 EEG (128)

        # Fusion Layer (MEA可配置)
        fusion_modules = [
            nn.Conv1d(fusion_input_channels, afr_reduced_cnn_size, kernel_size=1),
            nn.BatchNorm1d(afr_reduced_cnn_size),
            nn.ReLU()
        ]
        if use_mea_in_fusion:
            fusion_modules.append(MEA(channels=afr_reduced_cnn_size, factor=8))
        self.fusion_layer = nn.Sequential(*fusion_modules)
        
        # 细粒度门控，用于特征增强 (条件创建)
        if self.use_fine_grained_gating:
            self.fine_grained_gating = FineGrainedGatingModule(afr_reduced_cnn_size)
        else:
            self.fine_grained_gating = None # Will be handled in forward pass
        
        self.dim_adjust_tce = nn.Sequential(
            nn.Conv1d(afr_reduced_cnn_size, d_model, kernel_size=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU()
        )
        
        # 单epoch TCE编码器
        attn = EfficientAdditiveAttention(d_model, afr_reduced_cnn_size, dropout)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.tce = TCE(EncoderLayer(d_model, deepcopy(attn), deepcopy(ff), afr_reduced_cnn_size, dropout), N)
        
        # 记忆增强模块 (条件创建)
        if self.use_memory_module:
            self.memory_module = MemoryEnhancedModule(d_model)
        else:
            self.memory_module = None # Will be handled in forward pass
        
        # 跨Epoch注意力模块 (条件创建)
        if self.use_cross_epoch_attn:
            self.cross_epoch_attn = CrossEpochAttention(d_model)
        else:
            self.cross_epoch_attn = None # Will be handled in forward pass
        
        # 序列级TCE编码器 (条件创建)
        if self.use_seq_tce and seq_N > 0: # 只有当需要且层数大于0时才创建
            seq_attn = EfficientAdditiveAttention(d_model, d_model, dropout) # Note: afr_reduced_cnn_size for seq_attn? Here d_model used based on original code.
            seq_ff = PositionwiseFeedForward(d_model, d_ff, dropout)
            self.seq_tce = TCE(EncoderLayer(d_model, deepcopy(seq_attn), deepcopy(seq_ff), d_model, dropout), seq_N)
        else:
            self.seq_tce = None # Will be handled in forward pass
        
        # 对比学习投影头 - 用于生成表示向量用于对比学习
        self.projection_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, proj_dim)
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, num_classes)
        )
        
    def forward(self, x_eeg_seq, x_eog_seq):
        """
        参数:
            x_eeg_seq: 序列EEG数据 [batch_size, seq_len, channels, time]
            x_eog_seq: 序列EOG数据 [batch_size, seq_len, channels, time]
        返回:
            logits: 分类预测 [batch_size, seq_len, num_classes]
            projections: 投影特征，用于对比学习 [batch_size, seq_len, proj_dim]
            features: 序列特征表示 [batch_size, seq_len, d_model]
        """
        batch_size, seq_len = x_eeg_seq.size(0), x_eeg_seq.size(1)
        
        # 重塑输入以批量处理
        x_eog_flat = x_eog_seq.contiguous().view(-1, x_eog_seq.size(2), x_eog_seq.size(3))
        
        # EOG处理，如果启用
        if self.use_eog and x_eog_seq is not None:
            x_eog_flat = x_eog_seq.contiguous().view(-1, x_eog_seq.size(2), x_eog_seq.size(3))
            x_eog_flat = normalize_signal(x_eog_flat)
            if self.training:
                x_eog_flat = augment_signal(x_eog_flat)
        else:
            x_eog_flat = None

        # EEG处理
        x_eeg_flat = normalize_signal(x_eeg_flat)
        if self.training:
            x_eeg_flat = augment_signal(x_eeg_flat)
        
        # 特征提取
        eeg_features = self.eeg_incep(x_eeg_flat)
        
        if self.eog_incep is not None and x_eog_flat is not None:
            eog_features = self.eog_incep(x_eog_flat)
            # 确保时间维度对齐（如果InceptionBlock输出可变长度，这里需要注意）
            # 假设InceptionBlock输出固定长度或者已经处理好了对齐
            min_len = min(eeg_features.size(-1), eog_features.size(-1))
            eeg_features = eeg_features[..., :min_len]
            eog_features = eog_features[..., :min_len]
            fused = torch.cat([eeg_features, eog_features], dim=1)
        else:
            fused = eeg_features
            
        fused = self.fusion_layer(fused)
        
        # 应用细粒度门控 (如果启用)
        if self.fine_grained_gating is not None:
            fused = self.fine_grained_gating(fused)
        
        fused = self.dim_adjust_tce(fused)
        
        if fused.size(-1) != 80:
            fused = F.adaptive_avg_pool1d(fused, 80)
            
        fused = fused.transpose(1, 2)  # [B*S, 80, d_model_internal] -> [B*S, d_model_internal, 80]
        encoded = self.tce(fused)  # [B*S, 80, d_model] (TCE operates on last dim if it's feature dim)
                                     # Original TCE expects [B*S, T, C], so input should be [B*S, 80, d_model]
                                     # Here, TCE's 'size' is d_model, so it expects input of shape [..., d_model]
                                     # The transpose makes it [B*S, 80, d_model_from_conv]
                                     # Let's assume d_model_from_conv from dim_adjust_tce is d_model (80)
                                     # So 'encoded' becomes [B*S, 80, d_model]
        
        # 获取每个epoch的特征表示
        epoch_features = encoded.mean(dim=1)  # [B*S, d_model]
        
        # 重塑回序列形式
        seq_features = epoch_features.view(batch_size, seq_len, self.d_model)  # [B, S, d_model]
        
        current_features = seq_features # 用于逐步应用可选模块

        # 应用记忆增强 (如果启用)
        if self.memory_module is not None:
            current_features = self.memory_module(current_features)
        
        # 应用跨Epoch注意力
        if self.cross_epoch_attn is not None:
            current_features = self.cross_epoch_attn(current_features, seq_length=seq_len)
        
        # 序列编码 - 使用TCE进行最终的序列编码
        if self.seq_tce is not None:
            enhanced_seq = self.seq_tce(current_features)  # [B, S, d_model]
        else:
            enhanced_seq = current_features # 如果不使用 seq_tce, 直接用前面的结果
        
        # 生成对比学习的投影特征
        projections = self.projection_head(enhanced_seq)  # [B, S, proj_dim]
        
        # 分类
        logits = self.classifier(enhanced_seq)  # [B, S, num_classes]
        
        return logits, projections, enhanced_seq


class BioSleepXNoTCE(nn.Module):
    def __init__(self, use_msea=False, use_gabor=False, 
                 # --- 消融实验控制参数 ---
                 use_eog=True,
                 use_augmentation=True
                ):
        super(BioSleepXNoTCE, self).__init__()
        d_model = 80
        dropout = 0.1
        num_classes = 5
        afr_reduced_cnn_size = 30

        self.d_model = d_model
        self.use_eog = use_eog
        self.use_augmentation = use_augmentation

        # EEG特征提取
        self.eeg_incep = InceptionBlock(1)

        # EOG特征提取（条件性）
        if self.use_eog:
            self.eog_incep = InceptionBlock(1)
            fusion_input_channels = 256 # EEG (128) + EOG (128)
        else:
            self.eog_incep = None
            fusion_input_channels = 128 # 仅 EEG (128)

        # 特征融合层
        self.fusion_layer = nn.Sequential(
            nn.Conv1d(fusion_input_channels, afr_reduced_cnn_size, kernel_size=1),
            nn.BatchNorm1d(afr_reduced_cnn_size),
            nn.ReLU(),
            MEA(channels=afr_reduced_cnn_size, factor=8)
        )

        # 维度调整层（在原模型中是为TCE准备的）
        self.dim_adjust = nn.Sequential(
            nn.Conv1d(afr_reduced_cnn_size, d_model, kernel_size=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU()
        )

        # 直接使用全局池化，替代TCE编码
        self.global_pooling = nn.AdaptiveAvgPool1d(1)
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x_eeg, x_eog=None):
        # 数据增强（如果启用）
        if self.training and self.use_augmentation:
            x_eeg = augment_signal(x_eeg)
            if self.use_eog and x_eog is not None:
                x_eog = augment_signal(x_eog)

        # 信号标准化
        x_eeg = normalize_signal(x_eeg)
        eeg_features = self.eeg_incep(x_eeg)

        # EOG处理（如果使用）
        if self.use_eog and self.eog_incep is not None and x_eog is not None:
            x_eog = normalize_signal(x_eog)
            eog_features = self.eog_incep(x_eog)
            # 确保时间维度对齐
            min_len = min(eeg_features.size(-1), eog_features.size(-1))
            eeg_features = eeg_features[..., :min_len]
            eog_features = eog_features[..., :min_len]
            fused = torch.cat([eeg_features, eog_features], dim=1)
        else:
            fused = eeg_features

        # 特征融合
        fused = self.fusion_layer(fused)
        
        # 维度调整
        fused = self.dim_adjust(fused)

        # 全局池化（替代TCE）
        pooled = self.global_pooling(fused).squeeze(-1)
        
        # 分类
        output = self.classifier(pooled)
        return output


class BioSleepXModifiedInception(nn.Module):
    def __init__(self, use_msea=False, use_gabor=False, 
                 # --- 消融实验控制参数 ---
                 use_eog=True,
                 use_augmentation=True
                ):
        super(BioSleepXModifiedInception, self).__init__()
        N = 2
        d_model = 80
        d_ff = 120
        dropout = 0.1
        num_classes = 5
        afr_reduced_cnn_size = 30

        self.d_model = d_model
        self.use_eog = use_eog
        self.use_augmentation = use_augmentation

        # 导入修改后的Inception模块
        from model.modified_inception import ModifiedInceptionBlock
        
        # 使用修改后的Inception模块
        self.eeg_incep = ModifiedInceptionBlock(1)

        if self.use_eog:
            self.eog_incep = ModifiedInceptionBlock(1)
            fusion_input_channels = 256 # EEG (128) + EOG (128)
        else:
            self.eog_incep = None
            fusion_input_channels = 128 # 仅 EEG (128)

        # 特征融合层
        self.fusion_layer = nn.Sequential(
            nn.Conv1d(fusion_input_channels, afr_reduced_cnn_size, kernel_size=1),
            nn.BatchNorm1d(afr_reduced_cnn_size),
            nn.ReLU(),
            MEA(channels=afr_reduced_cnn_size, factor=8)
        )

        # 维度调整层
        self.dim_adjust_tce = nn.Sequential(
            nn.Conv1d(afr_reduced_cnn_size, d_model, kernel_size=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU()
        )

        # 使用TCE进行时序特征编码
        attn = EfficientAdditiveAttention(d_model, afr_reduced_cnn_size, dropout)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.tce = TCE(EncoderLayer(d_model, deepcopy(attn), deepcopy(ff), afr_reduced_cnn_size, dropout), N)

        self.dynamic_weight = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x_eeg, x_eog=None):
        # 数据增强（如果启用）
        if self.training and self.use_augmentation:
            x_eeg = augment_signal(x_eeg)
            if self.use_eog and x_eog is not None:
                x_eog = augment_signal(x_eog)

        # 信号标准化
        x_eeg = normalize_signal(x_eeg)
        eeg_features = self.eeg_incep(x_eeg)

        # EOG处理（如果使用）
        if self.use_eog and self.eog_incep is not None and x_eog is not None:
            x_eog = normalize_signal(x_eog)
            eog_features = self.eog_incep(x_eog)
            # 确保时间维度对齐
            min_len = min(eeg_features.size(-1), eog_features.size(-1))
            eeg_features = eeg_features[..., :min_len]
            eog_features = eog_features[..., :min_len]
            fused = torch.cat([eeg_features, eog_features], dim=1)
        else:
            fused = eeg_features

        # 特征融合
        fused = self.fusion_layer(fused)
        
        # 维度调整
        fused = self.dim_adjust_tce(fused)

        # 确保时间维度匹配TCE输入要求
        if fused.size(-1) != self.d_model:
            fused = F.adaptive_avg_pool1d(fused, self.d_model)
            
        # 转置维度以适应TCE的输入格式
        fused = fused.transpose(1, 2) # [B, C, T] -> [B, T, C]
        
        # 使用TCE进行特征编码
        encoded = self.tce(fused)
        
        # 获取全局特征表示
        output_features = encoded.mean(dim=1)
        
        # 分类
        output = self.classifier(output_features)
        return output


class BioSleepXEnhanced(nn.Module):
    def __init__(self, use_msea=False, use_gabor=False, 
                 # --- 消融实验控制参数 ---
                 use_eog=True,
                 use_augmentation=True,
                 use_dual_path=True  # 是否使用双路径特征提取
                ):
        super(BioSleepXEnhanced, self).__init__()
        N = 2
        d_model = 80
        d_ff = 120
        dropout = 0.1
        num_classes = 5
        afr_reduced_cnn_size = 30

        self.d_model = d_model
        self.use_eog = use_eog
        self.use_augmentation = use_augmentation
        self.use_dual_path = use_dual_path

        # 导入增强版Inception模块
        from model.enhanced_inception import EnhancedInceptionBlock, DualPathFeatureExtractor
        
        # 特征提取方式选择
        if self.use_dual_path:
            # 使用双路径特征提取器(类似AttnSleep的MRCNN)
            self.eeg_extractor = DualPathFeatureExtractor(out_channels=128)
            if self.use_eog:
                self.eog_extractor = DualPathFeatureExtractor(out_channels=128)
                fusion_input_channels = 512  # 修改：256 -> 512，双路径EEG+EOG时实际通道数
            else:
                self.eog_extractor = None
                fusion_input_channels = 256  # 单通道但有两个路径 128x2=256
        else:
            # 使用增强版Inception模块
            self.eeg_incep = EnhancedInceptionBlock(1)
            if self.use_eog:
                self.eog_incep = EnhancedInceptionBlock(1)
                fusion_input_channels = 256  # EEG (128) + EOG (128)
            else:
                self.eog_incep = None
                fusion_input_channels = 128  # 仅 EEG (128)

        # 特征融合层
        self.fusion_layer = nn.Sequential(
            nn.Conv1d(fusion_input_channels, afr_reduced_cnn_size, kernel_size=1),
            nn.BatchNorm1d(afr_reduced_cnn_size),
            nn.ReLU(),
            MEA(channels=afr_reduced_cnn_size, factor=8)
        )

        # 维度调整层
        self.dim_adjust_tce = nn.Sequential(
            nn.Conv1d(afr_reduced_cnn_size, d_model, kernel_size=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU()
        )

        # 使用TCE进行时序特征编码
        attn = EfficientAdditiveAttention(d_model, afr_reduced_cnn_size, dropout)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.tce = TCE(EncoderLayer(d_model, deepcopy(attn), deepcopy(ff), afr_reduced_cnn_size, dropout), N)

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x_eeg, x_eog=None):
        # 数据增强（如果启用）
        if self.training and self.use_augmentation:
            x_eeg = augment_signal(x_eeg)
            if self.use_eog and x_eog is not None:
                x_eog = augment_signal(x_eog)

        # 信号标准化
        x_eeg = normalize_signal(x_eeg)
        
        # 特征提取
        if self.use_dual_path:
            # 使用双路径特征提取
            eeg_features = self.eeg_extractor(x_eeg)
            
            if self.use_eog and self.eog_extractor is not None and x_eog is not None:
                x_eog = normalize_signal(x_eog)
                eog_features = self.eog_extractor(x_eog)
                # 确保时间维度对齐
                min_len = min(eeg_features.size(-1), eog_features.size(-1))
                eeg_features = eeg_features[..., :min_len]
                eog_features = eog_features[..., :min_len]
                fused = torch.cat([eeg_features, eog_features], dim=1)
            else:
                fused = eeg_features
        else:
            # 使用增强版Inception
            eeg_features = self.eeg_incep(x_eeg)
            
            if self.use_eog and self.eog_incep is not None and x_eog is not None:
                x_eog = normalize_signal(x_eog)
                eog_features = self.eog_incep(x_eog)
                # 确保时间维度对齐
                min_len = min(eeg_features.size(-1), eog_features.size(-1))
                eeg_features = eeg_features[..., :min_len]
                eog_features = eog_features[..., :min_len]
                fused = torch.cat([eeg_features, eog_features], dim=1)
            else:
                fused = eeg_features

        # 特征融合
        fused = self.fusion_layer(fused)
        
        # 维度调整
        fused = self.dim_adjust_tce(fused)

        # 确保时间维度匹配TCE输入要求
        if fused.size(-1) != self.d_model:
            fused = F.adaptive_avg_pool1d(fused, self.d_model)
            
        # 转置维度以适应TCE的输入格式
        fused = fused.transpose(1, 2) # [B, C, T] -> [B, T, C]
        
        # 使用TCE进行特征编码
        encoded = self.tce(fused)
        
        # 获取全局特征表示
        output_features = encoded.mean(dim=1)
        
        # 分类
        output = self.classifier(output_features)
        return output


# 新增：简化的1D Mamba式层 (占位符性质)
class SimplifiedMamba1DLayer(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand_factor=2,bias=False, conv_bias=True, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand_factor = expand_factor
        
        self.d_inner = int(self.expand_factor * self.d_model)

        # 输入投影和扩展 (B, L, D) -> (B, L, E*D)
        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        
        # 1D深度卷积 (B, L, E*D) -> (B, E*D, L) -> (B, E*D, L)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=self.d_conv,
            padding=(self.d_conv - 1) // 2,
            groups=self.d_inner, # 深度卷积
            bias=conv_bias,
            **factory_kwargs,
        )
        
        # 激活函数
        self.act = nn.SiLU() # Swish/SiLU 激活

        # 门控机制 (简化)
        # x_proj (B, L, E*D) and z_proj (B, L, E*D)
        self.x_proj = nn.Linear(self.d_inner, self.d_inner, bias=bias, **factory_kwargs) 
        self.z_proj = nn.Linear(self.d_inner, self.d_inner, bias=bias, **factory_kwargs)

        # 输出投影 (B, L, E*D) -> (B, L, D)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.norm = LayerNorm(self.d_model) # 使用已有的LayerNorm

    def forward(self, x):
        # x: (B, L, D) - B: batch_size, L: sequence_length, D: d_model
        
        # 1. 输入投影和扩展
        x_expanded = self.in_proj(x) # (B, L, E*D)
        
        # 2. 1D深度卷积
        #    Conv1d 需要 (B, C, L) 格式
        x_conv_input = x_expanded.transpose(1, 2) # (B, E*D, L)
        x_conv_out = self.conv1d(x_conv_input)    # (B, E*D, L)
        x_conv_out = x_conv_out.transpose(1, 2)   # (B, L, E*D)
        
        # 3. 激活
        x_activated = self.act(x_conv_out)
        
        # 4. 门控 (简化版本，没有真正的状态空间扫描)
        #    Mamba中的门控通常是 x * silu(ssm(x)) * silu(z)
        #    这里我们简化为 (x_activated * silu(x_proj(x_activated))) * silu(z_proj(x_activated))
        #    或者更简单的门控： x_activated * sigmoid(gate_signal)
        #    我们用一个简化的元素乘法门控
        
        x_gate_input = x_activated # (B, L, E*D)
        x_gated = self.act(self.x_proj(x_gate_input)) # 一个SiLU门
        z_gated = self.act(self.z_proj(x_gate_input)) # 另一个SiLU门 (或者可以是不同的投影)
        
        # 这里的简化 "SSM" 部分被省略了，直接将卷积后的结果与门控结合
        # 在实际Mamba中，门控的x是SSM的输出
        gated_features = x_gated * z_gated # (B, L, E*D)

        # 5. 输出投影
        x_out = self.out_proj(gated_features) # (B, L, D)
        
        # 添加残差连接 (通常Mamba块会有残差)
        # 为了简单集成，这里先不加，可以在BioSleepXSeq的forward中加
        # 或者在块级别实现
        return self.norm(x_out) # 返回归一化后的输出