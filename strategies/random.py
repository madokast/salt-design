"""
随机采样策略
"""
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from typing import List, Any

from .base import ActiveLearningStrategy


class RandomStrategy(ActiveLearningStrategy):
    """
    随机采样策略
    
    从未标注样本中随机选择指定数量的样本进行标注。
    """
    
    def __init__(self):
        """初始化随机策略"""
        super().__init__("Random")
    
    def select_samples(
        self, 
        model: torch.nn.Module, 
        unlabeled_indices: List[int], 
        dataset: Any, 
        batch_size: int,
        device: torch.device
    ) -> List[int]:
        """
        随机选择样本
        
        参数:
            model: 训练好的模型
            unlabeled_indices: 未标注样本的索引列表
            dataset: 数据集
            batch_size: 要选择的样本数量
            device: 计算设备
            
        返回:
            List[int]: 随机选择的样本索引列表
        """
        if len(unlabeled_indices) <= batch_size:
            return unlabeled_indices
        
        # 随机选择指定数量的样本
        selected_indices = np.random.choice(unlabeled_indices, batch_size, replace=False)
        return selected_indices.tolist()
