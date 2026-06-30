"""
Margin策略
"""
import numpy as np
import torch
from typing import List, Any

from .uncertainty import UncertaintyStrategy


class MarginStrategy(UncertaintyStrategy):
    """
    Margin策略
    
    选择模型预测边缘值最小的样本，即最靠近决策边界的样本。
    边缘值定义为第一大概率与第二大概率之差。
    """

    def __init__(self):
        """初始化Margin策略"""
        super().__init__("Margin")

    def select_samples(
        self, 
        model: torch.nn.Module, 
        unlabeled_indices: List[int], 
        dataset: Any, 
        batch_size: int,
        device: torch.device
    ) -> List[int]:
        """
        根据边缘值选择样本
        
        参数:
            model: 训练好的模型
            unlabeled_indices: 未标注样本的索引列表
            dataset: 数据集
            batch_size: 要选择的样本数量
            device: 计算设备
            
        返回:
            List[int]: 基于边缘值选择的样本索引列表
        """
        if len(unlabeled_indices) <= batch_size:
            return unlabeled_indices
        
        # 获取模型预测概率
        probabilities = self.get_probabilities(model, unlabeled_indices, dataset, device)
        
        # 对概率进行排序，找到第一和第二大的概率
        sorted_probs, _ = torch.sort(probabilities, dim=1, descending=True)
        
        # 计算边缘值：第一大概率 - 第二大概率
        margin_scores = sorted_probs[:, 0] - sorted_probs[:, 1]
        
        # 找到边缘值最小的样本（最靠近决策边界）
        min_margin_indices = np.argsort(margin_scores.numpy())[:batch_size]
    
        # 获取原始数据集中的索引
        selected_indices = [unlabeled_indices[i] for i in min_margin_indices]

        return selected_indices
