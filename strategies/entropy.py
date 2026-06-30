"""
Entropy策略
"""
import numpy as np
import torch
from typing import List, Any

from .uncertainty import UncertaintyStrategy


class EntropyStrategy(UncertaintyStrategy):
    """
    Entropy策略
    
    选择模型预测熵值最高的样本，即模型最不确定的样本。
    熵值越高，表示模型对样本的预测越不确定。
    """

    def __init__(self):
        """初始化Entropy策略"""
        super().__init__("Entropy")

    def select_samples(
        self, 
        model: torch.nn.Module, 
        unlabeled_indices: List[int], 
        dataset: Any, 
        batch_size: int,
        device: torch.device
    ) -> List[int]:
        """
        根据熵值选择样本
        
        参数:
            model: 训练好的模型
            unlabeled_indices: 未标注样本的索引列表
            dataset: 数据集
            batch_size: 要选择的样本数量
            device: 计算设备
            
        返回:
            List[int]: 基于熵值选择的样本索引列表
        """
        if len(unlabeled_indices) <= batch_size:
            return unlabeled_indices
        
        # 获取模型预测概率
        probabilities = self.get_probabilities(model, unlabeled_indices, dataset, device)
        
        # 计算每个样本的信息熵
        from scipy.stats import entropy
        uncertainty_scores = entropy(probabilities.numpy().T)
        
        # 找到熵值最高的样本（最不确定）
        max_entropy_indices = np.argsort(uncertainty_scores)[-batch_size:]

        # 获取原始数据集中的索引
        selected_indices = [unlabeled_indices[i] for i in max_entropy_indices]

        return selected_indices
