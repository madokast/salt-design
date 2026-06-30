"""
最小置信度策略
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from typing import List, Any

from .base import ActiveLearningStrategy


class LeastConfidenceStrategy(ActiveLearningStrategy):
    """
    最小置信度策略
    
    选择模型预测置信度最低的样本，即模型最不确定的样本。
    置信度定义为模型对预测类别的最大概率。
    """
    
    def __init__(self):
        """初始化最小置信度策略"""
        super().__init__("LeastConfidence")
    
    def select_samples(
        self, 
        model: torch.nn.Module, 
        unlabeled_indices: List[int], 
        dataset: Any, 
        batch_size: int,
        device: torch.device
    ) -> List[int]:
        """
        根据最小置信度选择样本
        
        参数:
            model: 训练好的模型
            unlabeled_indices: 未标注样本的索引列表
            dataset: 数据集
            batch_size: 要选择的样本数量
            device: 计算设备
            
        返回:
            List[int]: 基于最小置信度选择的样本索引列表
        """
        if len(unlabeled_indices) <= batch_size:
            return unlabeled_indices
        
        # 创建未标注数据的数据加载器
        unlabeled_dataset = Subset(dataset, unlabeled_indices)
        unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=64, shuffle=False)
        
        # 获取模型预测的置信度
        model.eval()
        confidences = []
        
        with torch.no_grad():
            for inputs, _ in unlabeled_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                # 计算每个样本的最大概率（置信度）
                probs = F.softmax(outputs, dim=1)
                max_probs, _ = torch.max(probs, dim=1)
                confidences.extend(max_probs.cpu().numpy())
        
        # 选择置信度最低的样本（即模型最不确定的样本）
        confidences = np.array(confidences)
        # 获取最小置信度样本的索引（在unlabeled_indices中的相对位置）
        least_confident_indices = np.argsort(confidences)[:batch_size]
        
        # 获取原始数据集中的索引
        selected_indices = [unlabeled_indices[i] for i in least_confident_indices]
        
        return selected_indices
