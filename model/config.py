"""
睡眠分期模型配置和评估指标
"""

class ModelConfig:
    # 模型架构配置
    N_TCE_LAYERS = 2  # Transformer编码器层数
    D_MODEL = 80      # 模型维度
    D_FF = 120       # 前馈网络维度
    N_HEADS = 5      # 注意力头数
    DROPOUT = 0.1    # Dropout率
    
    # Gabor滤波器配置
    GABOR_KERNELS = 8  # Gabor核心数量
    # 典型睡眠频率范围(Hz)
    SLEEP_FREQS = {
        'delta': (0.5, 4),   # 深睡眠
        'theta': (4, 8),     # 浅睡眠
        'alpha': (8, 13),    # 清醒
        'spindle': (12, 14)  # 睡眠纺锤波
    }
    
    # 训练配置
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    
    # 评估指标
    @staticmethod
    def get_metrics():
        """
        返回评估指标配置
        
        评估重点：
        1. 整体性能：准确率、宏平均F1
        2. 各类别性能：每类F1分数
        3. 特殊关注：N1和REM的识别能力
        """
        return {
            'accuracy': 'Overall classification accuracy',
            'macro_f1': 'Macro-averaged F1 score',
            'per_class_f1': {
                'W': 'Wake stage F1 (EEG dominant)',
                'N1': 'Stage 1 F1 (EOG dominant)',
                'N2': 'Stage 2 F1 (EEG dominant)',
                'N3': 'Stage 3 F1 (EEG dominant)',
                'REM': 'REM stage F1 (EOG dominant)'
            },
            'weighted_f1': 'Weighted average F1 score'
        }
    
    # 实验配置
    EXPERIMENT_CONFIGS = {
        'baseline': {
            'use_gabor': False,
            'use_ema': False,
            'description': '基准模型，使用标准卷积'
        },
        'gabor_only': {
            'use_gabor': True,
            'use_ema': False,
            'description': '仅使用Gabor滤波器'
        },
        'gabor_ema': {
            'use_gabor': True,
            'use_ema': True,
            'description': '完整模型：Gabor + EMA'
        }
    }
    
    # 模型解释性配置
    INTERPRETABILITY = {
        'gabor_vis': True,    # 可视化Gabor滤波器
        'attention_vis': True, # 可视化注意力权重
        'feature_vis': True,   # 可视化提取的特征
    } 