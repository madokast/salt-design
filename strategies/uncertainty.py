"""
不确定性策略基类
"""
import torch
from torch.utils.data import DataLoader, Subset
from typing import List, Any

from .base import ActiveLearningStrategy


class UncertaintyStrategy(ActiveLearningStrategy):
    """
    采用不确定性思想的策略基类
    
    提供获取模型预测概率的通用方法，子类可以基于这些概率计算不同的不确定性度量。
    """
    
    def get_probabilities(self, model, unlabeled_indices, dataset, device) -> torch.Tensor:
        """
        获取模型对未标注样本的预测概率
        
        参数:
            model: 训练好的模型
            unlabeled_indices: 未标注样本的索引列表
            dataset: 数据集
            device: 计算设备
            
        返回:
            torch.Tensor: 每个样本的预测概率，形状为 [num_samples, num_classes]
        """
        model.eval()
        unlabeled_dataset = Subset(dataset, unlabeled_indices)
        unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=64, shuffle=False)

        all_probs = []
        with torch.no_grad():
            for inputs, _ in unlabeled_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                probabilities = torch.softmax(outputs, dim=1)
                all_probs.append(probabilities.cpu())

        return torch.cat(all_probs, dim=0)
