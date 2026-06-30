"""
Our Method Margin Dist - margin 评分 + 动态距离乘法修正策略。

该策略等价于 `our_method_margin` 加上 batch 内动态 diversity 修正：

base_score_t(x) = log(D_t(x)) + alpha * coverage_ratio * (-log(margin(x)))
score_t(x) = base_score_t(x) * (1 + d_L,t(x) / d_max)

其中 `d_L,t(x)` 是候选点到“本轮开始前已标注点 + 本轮已选点”的最近距离，
`d_max` 是本轮贪心开始前所有候选点初始最近已标注距离的最大值，并在本轮内
保持固定。`alpha=None` 时仍沿用 margin 版本的裸 `logD` 与 `-log(margin)`
自动尺度校准，不把距离因子混入 alpha 语义。
"""
from typing import Any, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from .our_method_dist import OurMethodDistStrategy
from .our_method_margin import OurMethodMarginStrategy


class OurMethodMarginDistStrategy(OurMethodMarginStrategy, OurMethodDistStrategy):
    """
    `our_method_margin` 的动态距离乘法修正版。

    本类通过多重继承复用 `OurMethodMarginStrategy` 的 margin 不确定性静态项，
    以及 `OurMethodDistStrategy` 中“初始最近已标注距离、距离因子、增量距离更新”
    这些辅助函数。实际采样流程在本类中重写，确保距离因子乘在完整 base score
    外侧，而不是只修正 `logD` 覆盖项。
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        初始化 margin + 距离乘法修正策略实例。

        该构造函数不增加任何新的外部参数，所有参数都沿用 `our_method_margin` 和
        `our_method_budget_version` 既有接口。父类初始化会建立半径、覆盖率、局部
        Lipschitz、早期半径归一化等全部状态；本函数只在最后把策略展示名称改成
        `OurMethodMarginDist`，用于日志输出和实验结果区分。

        参数:
            *args: 传递给父类策略构造函数的位置参数。
            **kwargs: 传递给父类策略构造函数的关键字参数。

        返回:
            None
        """
        super().__init__(*args, **kwargs)
        self.name = "OurMethodMarginDist"

    def _subsequent_round_selection(
        self,
        model: torch.nn.Module,
        dataset: Any,
        device: torch.device,
        distance_matrix: np.ndarray,
        labeled_indices: List[int],
        unlabeled_indices: List[int],
        spectral_norm_product: Optional[float]
    ) -> List[int]:
        """
        执行第二轮及之后的 margin + 动态距离乘法修正采样。

        该方法复用预算版本后续轮次的半径与覆盖流程：先计算已标注点半径和覆盖率，
        过滤已被覆盖的无标注候选点，再对剩余候选点预测概率、估计半径，并在预算
        约束下做动态贪心选择。与 `our_method_margin` 的区别只在贪心打分：
        1. 先用 margin 静态项与当前动态覆盖项构造
           `base_score_t(x) = static_margin_term(x) + log(D_t(x))`；
        2. 再乘以动态距离因子 `1 + d_L,t(x) / d_max`；
        3. 每选中一个点后，把该点视为本轮新增标注点，用一次向量化 `min` 增量
           更新剩余候选点的 `d_L,t(x)`，从而在 batch 内体现 diversity。

        当 `alpha=None` 时，本方法传给 `_compute_static_score_terms(...)` 的
        `coverage_terms` 仍是未乘距离因子的初始 `logD`，因此自动 alpha 只校准
        `logD` 与 `-log(margin)` 的相对尺度，不把距离乘子混入不确定性权重。

        参数:
            model: 当前轮训练完成后的模型。
            dataset: 完整训练集对象。
            device: 当前计算设备。
            distance_matrix: 与完整训练集索引对齐的距离矩阵。
            labeled_indices: 本轮开始前已有标注样本索引。
            unlabeled_indices: 本轮开始前仍未标注样本索引。
            spectral_norm_product: 固定全局放大系数；`None` 表示使用逐点局部系数。

        返回:
            List[int]: 本轮选择的样本全局索引列表。
        """
        print(f"\n[{self.name}] 第{self.round_counter + 1}轮采样")

        max_theoretical_radius = self._compute_round_max_theoretical_radius(
            list(labeled_indices) + list(unlabeled_indices),
            spectral_norm_product
        )
        labeled_points = self._build_labeled_points(dataset, labeled_indices)
        raw_labeled_radii = self._compute_labeled_radii(
            labeled_points=labeled_points,
            labeled_indices=labeled_indices,
            model=model,
            spectral_norm_product=spectral_norm_product
        )

        coverage_ratio, covered_mask = self._compute_coverage(
            distance_matrix, labeled_indices, raw_labeled_radii.tolist()
        )
        self.last_coverage = coverage_ratio
        print(f"[{self.name}] 当前轮次覆盖率: {coverage_ratio:.6f}")

        normalization_applied = False
        normalization_scale_ratio = 1.0
        normalization_pivot_radius = (
            float(np.max(raw_labeled_radii)) if raw_labeled_radii.size > 0 else 0.0
        )
        if self.enable_early_radius_normalization:
            (
                effective_labeled_radii,
                normalization_applied,
                normalization_scale_ratio,
                normalization_pivot_radius,
            ) = self._normalize_radii_for_early_rounds(
                radii=raw_labeled_radii,
                coverage_ratio=coverage_ratio,
                max_theoretical_radius=max_theoretical_radius,
            )
            if normalization_applied:
                effective_labeled_radii = self._clip_radii_to_max_theoretical(
                    radii=effective_labeled_radii,
                    indices=labeled_indices,
                    spectral_norm_product=spectral_norm_product
                )
                normalized_coverage_ratio, covered_mask = self._compute_coverage(
                    distance_matrix,
                    labeled_indices,
                    effective_labeled_radii.tolist(),
                )
                coverage_ratio = normalized_coverage_ratio
                self.last_coverage = coverage_ratio
                print(
                    f"[{self.name}] 早期半径归一化已触发: "
                    f"T={self.early_radius_normalization_threshold:.6f}, "
                    f"k={self.early_radius_normalization_percentile_k:.2f}, "
                    f"q_k={normalization_pivot_radius:.6f}, "
                    f"r={normalization_scale_ratio:.6f}, "
                    f"max_radius={max_theoretical_radius:.6f}, "
                    f"normalized_coverage={coverage_ratio:.6f}"
                )
            else:
                print(
                    f"[{self.name}] 早期半径归一化未触发: "
                    f"coverage={coverage_ratio:.6f} >= "
                    f"T={self.early_radius_normalization_threshold:.6f}"
                )

        filtered_unlabeled = [idx for idx in unlabeled_indices if not covered_mask[idx]]
        print(f"[{self.name}] 原始无标注池大小: {len(unlabeled_indices)}")
        print(f"[{self.name}] 被已标注点覆盖的点数: {len(unlabeled_indices) - len(filtered_unlabeled)}")
        print(f"[{self.name}] 过滤后无标注池大小: {len(filtered_unlabeled)}")

        if not filtered_unlabeled:
            print(f"[{self.name}] 无标注池为空，无需选择样本")
            return []

        probabilities = self._get_probabilities_for_indices(
            model, dataset, filtered_unlabeled, device
        )
        h_values = self._compute_h_from_probabilities(probabilities)
        estimated_radii = self._estimate_radii_for_unlabeled(
            h_values=h_values,
            unlabeled_indices=filtered_unlabeled,
            spectral_norm_product=spectral_norm_product
        )
        if normalization_applied:
            estimated_radii = np.minimum(
                estimated_radii * normalization_scale_ratio,
                max_theoretical_radius,
            )
            estimated_radii = self._clip_radii_to_max_theoretical(
                radii=estimated_radii,
                indices=filtered_unlabeled,
                spectral_norm_product=spectral_norm_product
            )

        print(
            f"[{self.name}] 预估半径统计: "
            f"min={estimated_radii.min():.6f}, "
            f"max={estimated_radii.max():.6f}, "
            f"mean={estimated_radii.mean():.6f}"
        )

        selected_indices = []
        candidate_indices = np.asarray(filtered_unlabeled, dtype=np.int64)
        candidate_radii = np.asarray(estimated_radii, dtype=np.float64)
        candidate_distance_matrix = distance_matrix[
            np.ix_(candidate_indices, candidate_indices)
        ]
        nearest_labeled_distances, distance_max = (
            self._compute_initial_nearest_labeled_distances(
                candidate_indices=candidate_indices,
                labeled_indices=labeled_indices,
                distance_matrix=distance_matrix
            )
        )
        initial_distance_factors = self._compute_distance_factors(
            nearest_distances=nearest_labeled_distances,
            distance_max=distance_max
        )
        initial_log_coverage_terms = self._compute_log_coverage_terms(
            candidate_distance_matrix=candidate_distance_matrix,
            candidate_radii=candidate_radii
        )
        print(
            f"[{self.name}] 距离修正: "
            f"d_max={distance_max:.6f}, "
            f"factor_min={initial_distance_factors.min():.6f}, "
            f"factor_max={initial_distance_factors.max():.6f}, "
            f"factor_mean={initial_distance_factors.mean():.6f}"
        )

        static_score_terms = self._compute_static_score_terms(
            probabilities,
            coverage_ratio,
            coverage_terms=initial_log_coverage_terms
        )

        active_mask = np.ones(candidate_indices.shape[0], dtype=bool)
        with tqdm(
            total=min(self.query_budget, len(candidate_indices)),
            desc=f"[{self.name}] 第{self.round_counter + 1}轮选择进度",
            unit="sample"
        ) as progress_bar:
            while active_mask.any() and len(selected_indices) < self.query_budget:
                active_positions = np.flatnonzero(active_mask)
                best_local_position = -1
                best_score = -np.inf
                best_covered_positions = np.empty((0,), dtype=np.int64)
                best_distance_factor = 1.0

                distance_factors = self._compute_distance_factors(
                    nearest_distances=nearest_labeled_distances,
                    distance_max=distance_max
                )

                for local_position in active_positions:
                    radius = candidate_radii[local_position]
                    covered_mask = (
                        candidate_distance_matrix[local_position, active_positions] <= radius
                    )
                    covered_positions = active_positions[covered_mask]
                    D = max(int(covered_positions.size), 1)
                    base_score = static_score_terms[local_position] + np.log(D)
                    score = base_score * distance_factors[local_position]

                    if score > best_score:
                        best_score = float(score)
                        best_local_position = int(local_position)
                        best_covered_positions = covered_positions
                        best_distance_factor = float(distance_factors[local_position])

                if best_local_position < 0:
                    break

                selected_idx = int(candidate_indices[best_local_position])
                selected_indices.append(selected_idx)
                active_mask[best_covered_positions] = False
                self._update_nearest_distances_after_selection(
                    nearest_distances=nearest_labeled_distances,
                    candidate_distance_matrix=candidate_distance_matrix,
                    active_mask=active_mask,
                    selected_local_position=best_local_position
                )
                progress_bar.update(1)
                progress_bar.set_postfix(
                    covered=int(best_covered_positions.size),
                    remaining=int(active_mask.sum())
                )
                print(
                    f"[{self.name}] 选择样本 {selected_idx}, "
                    f"分数={best_score:.6f}, "
                    f"距离因子={best_distance_factor:.6f}, "
                    f"覆盖点数={int(best_covered_positions.size)}"
                )

        print(f"[{self.name}] 第{self.round_counter + 1}轮选择的标注点数量: {len(selected_indices)}")
        return selected_indices
