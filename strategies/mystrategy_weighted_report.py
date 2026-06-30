"""
MyStrategy 加权报告版策略。

该策略与 MyStrategy 的采样逻辑一致，但在每轮训练中会额外执行一次
“基于半径权重的加权训练”，其测试准确率仅用于本轮最终准确率统计，
而采样仍然依赖不加权训练模型。
"""
from typing import List, Any, Tuple, Optional

import numpy as np
import torch

from .mystrategy import MyStrategy


class MyStrategyWeightedReport(MyStrategy):
    """
    在 MyStrategy 基础上增加“加权训练用于报告”的策略变体。

    核心差异：
    - 使用不加权模型计算半径与权重；
    - 使用权重重新训练一个新模型，仅用于该轮准确率统计；
    - 采样仍沿用不加权模型的半径/覆盖/打分逻辑。
    """

    def __init__(
        self,
        epsilon: float = 0.01,
        h_min: float = 1e-6,
        spectral_norm_product: float = 10.0,
        coverage_threshold: float = 0.2,
        continue_training: bool = True,
        weighted_epochs: Optional[int] = None,
        weighted_learning_rate: Optional[float] = None,
        distance_cache_dir: str = "./distance_cache",
        log_eps: float = 1e-12,
        use_soft_labels: bool = False,
        normalize_radius: bool = False,
        radius_top_percent: float = 0.1
    ) -> None:
        """
        初始化加权报告策略，并复用 MyStrategy 的所有配置项。

        参数:
            epsilon: 半径方程中的 epsilon 参数。
            h_min: Hessian 谱范数下界。
            spectral_norm_product: 预先给定的谱范数乘积。
            coverage_threshold: 覆盖率阈值 T。
            continue_training: 兼容外部调用保留参数（本策略不使用）。
            weighted_epochs: 兼容外部调用保留参数（本策略不使用）。
            weighted_learning_rate: 兼容外部调用保留参数（本策略不使用）。
            distance_cache_dir: 距离矩阵缓存目录。
            log_eps: log 稳定项，避免 log(0)。
            use_soft_labels: 是否启用软标签半径计算。
            normalize_radius: 是否启用半径归一化。
            radius_top_percent: 半径归一化前的百分比阈值。
        """
        super().__init__(
            epsilon=epsilon,
            h_min=h_min,
            spectral_norm_product=spectral_norm_product,
            coverage_threshold=coverage_threshold,
            continue_training=continue_training,
            weighted_epochs=weighted_epochs,
            weighted_learning_rate=weighted_learning_rate,
            distance_cache_dir=distance_cache_dir,
            log_eps=log_eps,
            use_soft_labels=use_soft_labels,
            normalize_radius=normalize_radius,
            radius_top_percent=radius_top_percent
        )
        self.name = "MyStrategyWeightedReport"

    def compute_radii_and_weights(
        self,
        dataset: Any,
        labeled_indices: List[int],
        unlabeled_indices: List[int],
        model: torch.nn.Module
    ) -> Tuple[List[float], List[float]]:
        """
        计算当前已标注点的半径（含可选归一化）并基于半径计算权重。

        该方法用于加权报告策略的“权重训练”阶段：
        1) 使用不加权训练模型计算每个已标注点的半径；
        2) 若启用 normalize_radius，则对半径进行归一化；
        3) 基于完整无标注池（过滤前）与距离矩阵，统计每个已标注点的权重。

        参数:
            dataset: 完整训练集（用于构建已标注点数据）。
            labeled_indices: 已标注样本索引列表（顺序即权重顺序）。
            unlabeled_indices: 无标注样本索引列表（过滤前的完整池）。
            model: 不加权训练得到的模型。

        返回:
            Tuple[List[float], List[float]]:
                - radii: 与 labeled_indices 对齐的半径列表（可能已归一化）。
                - weights: 与 labeled_indices 对齐的权重列表。
        """
        labeled_points = self._build_labeled_points(dataset, labeled_indices)
        raw_radii, spectral_norm_product = self._compute_raw_radii(
            labeled_points=labeled_points,
            model=model
        )
        radii = self._normalize_radii(raw_radii, spectral_norm_product)

        distance_matrix = self._ensure_distance_matrix(split="train")
        weights = self._compute_weights_from_distance_matrix(
            distance_matrix=distance_matrix,
            labeled_indices=labeled_indices,
            unlabeled_indices=unlabeled_indices,
            radii=radii
        )
        return radii, weights

    def _compute_weights_from_distance_matrix(
        self,
        distance_matrix: np.ndarray,
        labeled_indices: List[int],
        unlabeled_indices: List[int],
        radii: List[float]
    ) -> List[float]:
        """
        使用距离矩阵与半径计算每个已标注点的权重。

        权重定义为：被分配到该已标注点半径球内的无标注点数量（基础权重为 1）。
        对每个无标注点：
        1) 找出所有满足 dist <= radius 的已标注点；
        2) 若存在多个候选，则选取 dist/radius 最小的已标注点作为归属；
        3) 将该已标注点的权重 +1。

        参数:
            distance_matrix: 预计算的距离矩阵，形状为 (N, N)。
            labeled_indices: 已标注样本索引列表（与 radii 对齐）。
            unlabeled_indices: 无标注样本索引列表（过滤前的完整池）。
            radii: 已标注点半径列表，顺序与 labeled_indices 一致。

        返回:
            List[float]: 权重列表，顺序与 labeled_indices 一致。
        """
        num_labeled = len(labeled_indices)
        if num_labeled == 0:
            return []

        weights = np.ones(num_labeled, dtype=np.float64)
        radii_array = np.asarray(radii, dtype=np.float64)

        if len(unlabeled_indices) == 0:
            return weights.tolist()

        for unlabeled_idx in unlabeled_indices:
            distances = distance_matrix[unlabeled_idx, labeled_indices]
            in_radius = distances <= radii_array
            if not np.any(in_radius):
                continue

            candidate_positions = np.where(in_radius)[0]
            candidate_distances = distances[candidate_positions]
            candidate_radii = radii_array[candidate_positions]

            ratios = candidate_distances / candidate_radii
            target_pos = candidate_positions[int(np.argmin(ratios))]
            weights[target_pos] += 1.0

        return weights.tolist()
