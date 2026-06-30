"""
K-center Greedy策略
"""
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from typing import List, Any
from sklearn.metrics.pairwise import euclidean_distances

from .base import ActiveLearningStrategy


class KCGreedyStrategy(ActiveLearningStrategy):
    """
    K-center Greedy策略
    
    使用贪心算法选择样本，使得选中的样本能够最大化覆盖未标注样本的特征空间。
    通过迭代选择距离当前核心集最远的样本来实现。
    """

    def __init__(self):
        """初始化K-center Greedy策略"""
        super().__init__("KCGreedy")

    def select_samples(
        self, 
        model: torch.nn.Module, 
        unlabeled_indices: List[int], 
        dataset: Any, 
        batch_size: int,
        device: torch.device
    ) -> List[int]:
        """
        使用K-center贪心算法选择样本
        
        参数:
            model: 训练好的模型
            unlabeled_indices: 未标注样本的索引列表
            dataset: 数据集
            batch_size: 要选择的样本数量
            device: 计算设备
            
        返回:
            List[int]: 基于K-center贪心算法选择的样本索引列表
        """
        if len(unlabeled_indices) <= batch_size:
            return unlabeled_indices
        
        model.eval()
        unlabeled_dataset = Subset(dataset, unlabeled_indices)
        unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=64, shuffle=False)

        all_features = []
        with torch.no_grad():
            for inputs, _ in unlabeled_loader:
                inputs = inputs.to(device)
                inputs = inputs.view(inputs.shape[0], -1)
                all_features.append(inputs.cpu().numpy())

        all_features = np.concatenate(all_features, axis=0)

        # 贪心算法实现
        current_set = []
        for _ in range(batch_size):
            # 计算未选中样本到当前核心集的最小距离
            dist_to_current_set = np.full(len(all_features), np.inf)
            if current_set:
                dist_to_current_set = euclidean_distances(all_features, all_features[current_set]).min(axis=1)

            # 找到距离最远的样本，作为下一个要添加的核心点
            farthest_idx = np.argmax(dist_to_current_set).item()
            current_set.append(farthest_idx)

        # 获取原始数据集中的索引
        selected_indices = [unlabeled_indices[i] for i in current_set]

        return selected_indices
