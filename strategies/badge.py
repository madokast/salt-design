"""
BADGE策略
"""
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from typing import List, Any
from sklearn.cluster import MiniBatchKMeans

from .base import ActiveLearningStrategy


class BADGEStrategy(ActiveLearningStrategy):
    """
    BADGE (Batch Active Learning by Diverse Gradient Embeddings)策略
    
    通过计算样本的梯度嵌入，并使用K-Means++聚类选择最具多样性的样本。
    该策略同时考虑了不确定性和多样性。
    """
    
    def __init__(self):
        """初始化BADGE策略"""
        super().__init__("BADGE")
    
    def select_samples(
        self, 
        model: torch.nn.Module, 
        unlabeled_indices: List[int], 
        dataset: Any, 
        batch_size: int,
        device: torch.device
    ) -> List[int]:
        """
        使用BADGE策略选择样本
        
        参数:
            model: 训练好的模型
            unlabeled_indices: 未标注样本的索引列表
            dataset: 数据集
            batch_size: 要选择的样本数量
            device: 计算设备
            
        返回:
            List[int]: 基于BADGE策略选择的样本索引列表
        """
        if len(unlabeled_indices) <= batch_size:
            return unlabeled_indices
        
        # 创建未标注数据的数据加载器
        unlabeled_dataset = Subset(dataset, unlabeled_indices)
        unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=64, shuffle=False)
        
        model.eval()
        all_gradients = []

        # 提取最后一层分类器的参数
        classifier = self.get_classifier(model)
        W = classifier.weight      # [C, D]
        C, D = W.shape

        grad_embeddings = []

        with torch.no_grad():
            for x, _ in unlabeled_loader:
                x = x.to(device)

                logits = model(x)                  # [B, C]
                probs = torch.nn.functional.softmax(logits, dim=1)   # [B, C]
                y_hat = probs.argmax(dim=1)        # [B]

                # 提取 embedding
                feats = self.extract_features(model, [(x, None)], device)  # [B, D]

                for i in range(x.size(0)):
                    g = torch.zeros(C, D, device=device)
                    for c in range(C):
                        coeff = probs[i, c]
                        if c == y_hat[i]:
                            coeff -= 1.0
                        g[c] = coeff * feats[i]

                    grad_embeddings.append(g.view(-1))

        grad_embeddings = torch.stack(grad_embeddings).cpu().numpy()

        # 使用K-Means++对梯度嵌入进行聚类，选择最接近聚类中心的样本
        kmeans = MiniBatchKMeans(n_clusters=batch_size, init='k-means++', random_state=0, n_init='auto').fit(grad_embeddings)
        distances = kmeans.transform(grad_embeddings)

        # 为每个聚类中心选择一个“尚未被其他中心选中过”的最近样本。
        # 直接使用 `np.argmin(distances, axis=0)` 时，不同中心可能映射到同一个样本，
        # 从而导致返回的索引出现重复，后续在未标注池中执行 `remove(idx)` 时触发异常。
        # 这里改为逐个中心贪心分配最近且未使用的样本，保证返回索引唯一。
        selected_positions = []
        used_positions = set()
        for center_idx in range(distances.shape[1]):
            candidate_positions = np.argsort(distances[:, center_idx])
            for sample_position in candidate_positions.tolist():
                if sample_position not in used_positions:
                    selected_positions.append(sample_position)
                    used_positions.add(sample_position)
                    break

        # 获取原始数据集中的索引
        selected_indices = [unlabeled_indices[i] for i in selected_positions]
        
        return selected_indices

    def extract_features(self, model, dataloader, device):
        """
        提取模型的特征表示
        
        参数:
            model: 训练好的模型
            dataloader: 数据加载器
            device: 计算设备
            
        返回:
            torch.Tensor: 特征表示
        """
        model.eval()
        features = []

        def hook_fn(module, input):
            features.append(input[0].detach())

        classifier = self.get_classifier(model)
        handle = classifier.register_forward_pre_hook(hook_fn)

        with torch.no_grad():
            for x, _ in dataloader:
                x = x.to(device)
                model(x)

        handle.remove()
        return torch.cat(features, dim=0)

    def get_classifier(self, model):
        """
        获取模型的分类器层
        
        参数:
            model: 训练好的模型
            
        返回:
            torch.nn.Module: 分类器层
            
        异常:
            ValueError: 当找不到分类器层时抛出
        """
        for attr in ["classifier"]:
            if hasattr(model, attr):
                return getattr(model, attr)
        
        last_linear = None
        for m in model.modules():
            if isinstance(m, torch.nn.Linear):
                last_linear = m
        if last_linear:
            return last_linear

        raise ValueError("No classifier layer found")
