"""
Our Method Dist - 带动态距离修正的预算版本主动学习策略。

该策略继承 `our_method_budget_version` 的半径计算、候选过滤和预算贪心流程，
只在第二轮及之后的评分中加入一个动态 diversity 修正项：

score_t(x) = log(D_t(x)) * (1 + d_L,t(x) / d_max)
             + alpha * coverage_ratio * log(||eta - p||_2)

其中 d_L,t(x) 是候选点到“本轮开始前已标注点 + 本轮已选点”的最近距离，
d_max 是本轮贪心开始前所有候选点初始 d_L,0(x) 的最大值，并在本轮内固定。
"""
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from .our_method_budget_version import OurMethodBudgetVersionStrategy


class OurMethodDistStrategy(OurMethodBudgetVersionStrategy):
    """
    `our_method_budget_version` 的距离修正版。

    第一轮完全沿用预算版本的最大理论半径覆盖贪心逻辑；第二轮及之后，在原有
    动态覆盖项 `logD` 上乘以 `1 + d_L,t(x) / d_max`。该距离项只在当前策略中
    生效，不会改变 `OurMethodBudgetVersionStrategy` 及其派生策略的默认行为。
    """

    _DISTANCE_EPS = 1e-12

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        初始化距离修正版策略实例。

        该构造函数不引入任何新的外部参数，所有位置参数与关键字参数都原样传给
        `OurMethodBudgetVersionStrategy`，从而复用预算版本的完整配置接口。调用
        父类初始化后，仅把策略展示名称改为 `OurMethodDist`，便于日志和实验输出
        区分该距离修正版策略。

        参数:
            *args: 传递给预算版本策略构造函数的位置参数。
            **kwargs: 传递给预算版本策略构造函数的关键字参数。

        返回:
            None
        """
        super().__init__(*args, **kwargs)
        self.name = "OurMethodDist"

    def _compute_initial_nearest_labeled_distances(
        self,
        candidate_indices: np.ndarray,
        labeled_indices: List[int],
        distance_matrix: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        计算本轮候选点到初始已标注集的最近距离和固定归一化分母。

        该函数在后续轮次进入贪心循环前调用一次。它把过滤后的候选点集合记为
        `C`，本轮开始前已有标注点集合记为 `L`，并计算：
        `d_L,0(x) = min_{z in L} distance(x, z)`。随后用所有候选点初始最近距离的
        最大值作为 `d_max`，本轮后续即使选入新点也不更新该分母，保证距离修正项
        的尺度稳定。

        当 `labeled_indices` 为空时，策略无法定义到已标注集的最近距离；此时返回
        全零距离和 `d_max=0`，使距离因子退化为 1，不改变原预算版本评分。

        参数:
            candidate_indices: 过滤后候选点的全局索引数组，形状为 `(U,)`。
            labeled_indices: 本轮开始前已有标注点的全局索引列表。
            distance_matrix: 与完整训练集索引对齐的距离矩阵。

        返回:
            Tuple[np.ndarray, float]:
                - nearest_distances: 每个候选点到初始已标注集的最近距离；
                - distance_max: `nearest_distances` 的最大值，用作本轮固定分母。
        """
        if candidate_indices.size == 0 or not labeled_indices:
            return np.zeros(candidate_indices.shape[0], dtype=np.float64), 0.0

        labeled_array = np.asarray(labeled_indices, dtype=np.int64)
        distances_to_labeled = distance_matrix[np.ix_(candidate_indices, labeled_array)]
        nearest_distances = np.min(distances_to_labeled, axis=1).astype(np.float64)
        finite_nearest_distances = nearest_distances[np.isfinite(nearest_distances)]
        if finite_nearest_distances.size == 0:
            return np.zeros(candidate_indices.shape[0], dtype=np.float64), 0.0

        distance_max = float(np.max(finite_nearest_distances))
        return nearest_distances, distance_max

    def _compute_distance_factors(
        self,
        nearest_distances: np.ndarray,
        distance_max: float
    ) -> np.ndarray:
        """
        把最近已标注距离转换为覆盖项乘法因子。

        该函数实现 `1 + d_L,t(x) / d_max`。`d_max` 在本轮开始时固定，而
        `nearest_distances` 会在每次选入新样本后通过增量 `min` 更新。为避免
        浮点误差或异常距离值导致因子越界，距离比值会被截断到 `[0, 1]`，因此
        返回因子位于 `[1, 2]`。当 `distance_max` 非正或非有限时，函数返回全 1，
        表示距离修正自动退化为原始覆盖项。

        参数:
            nearest_distances: 与候选点顺序一致的当前最近已标注距离数组。
            distance_max: 本轮开始时固定的最大初始最近已标注距离。

        返回:
            np.ndarray: 与 `nearest_distances` 形状一致的覆盖项乘法因子。
        """
        nearest_distances = np.asarray(nearest_distances, dtype=np.float64)
        if not np.isfinite(distance_max) or distance_max <= self._DISTANCE_EPS:
            return np.ones_like(nearest_distances, dtype=np.float64)

        ratios = nearest_distances / (distance_max + self._DISTANCE_EPS)
        ratios = np.clip(ratios, 0.0, 1.0)
        return 1.0 + ratios

    def _update_nearest_distances_after_selection(
        self,
        nearest_distances: np.ndarray,
        candidate_distance_matrix: np.ndarray,
        active_mask: np.ndarray,
        selected_local_position: int
    ) -> None:
        """
        在本轮选入一个新样本后，增量更新剩余候选点的最近已标注距离。

        被选中的点在当前轮内被视为新加入的标注点，因此每个仍处于 active 状态的
        候选点 `x` 只需执行一次：
        `d_L,t+1(x) = min(d_L,t(x), distance(x, selected))`。
        该更新只访问候选子矩阵中选中点对应的一行，复杂度为 `O(U_remaining)`，
        避免了对完整“已标注点 + 本轮已选点”集合重新计算最近距离。

        参数:
            nearest_distances: 与候选点顺序一致的当前最近已标注距离数组；本函数会
                原地更新其中仍处于 active 状态的位置。
            candidate_distance_matrix: 候选点内部距离矩阵，行列顺序与候选点数组一致。
            active_mask: 标记当前仍在无标注候选池中的布尔数组。
            selected_local_position: 本次选中样本在候选点数组中的局部位置。

        返回:
            None
        """
        remaining_positions = np.flatnonzero(active_mask)
        if remaining_positions.size == 0:
            return

        distances_to_selected = candidate_distance_matrix[
            selected_local_position,
            remaining_positions
        ]
        nearest_distances[remaining_positions] = np.minimum(
            nearest_distances[remaining_positions],
            distances_to_selected
        )

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
        执行第二轮及之后的带距离修正预算采样。

        本方法复用预算版本后续轮次的完整采样流程：先根据已标注点半径计算覆盖率，
        过滤已覆盖无标注点，再对剩余候选点估计半径、计算不确定性静态项，并进行
        动态覆盖贪心。唯一差异是贪心打分时把动态覆盖项从 `logD` 替换为
        `logD * (1 + d_L,t(x) / d_max)`，且在每次选入样本后把该样本纳入距离项的
        已标注集合，通过增量最近距离更新体现 batch 内 diversity。

        当 `alpha=None` 自动尺度校准时，本策略使用初始
        `logD * (1 + d_L,0(x) / d_max)` 作为覆盖项尺度，而不是预算版本的裸
        `logD`，使不确定性项与新增距离修正后的覆盖项保持同量纲校准。

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
        initial_distance_corrected_coverage_terms = (
            initial_log_coverage_terms * initial_distance_factors
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
            coverage_terms=initial_distance_corrected_coverage_terms
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
                score_by_active_position = np.empty(
                    active_positions.shape[0],
                    dtype=np.float64
                )
                covered_count_by_active_position = np.empty(
                    active_positions.shape[0],
                    dtype=np.int64
                )

                distance_factors = self._compute_distance_factors(
                    nearest_distances=nearest_labeled_distances,
                    distance_max=distance_max
                )

                for active_offset, local_position in enumerate(active_positions):
                    radius = candidate_radii[local_position]
                    covered_mask = (
                        candidate_distance_matrix[local_position, active_positions] <= radius
                    )
                    covered_positions = active_positions[covered_mask]
                    D = max(int(covered_positions.size), 1)
                    log_coverage = np.log(D)
                    distance_corrected_coverage = (
                        log_coverage * distance_factors[local_position]
                    )
                    score = (
                        static_score_terms[local_position]
                        + distance_corrected_coverage
                    )
                    score_by_active_position[active_offset] = score
                    covered_count_by_active_position[active_offset] = D

                    if score > best_score:
                        best_score = float(score)
                        best_local_position = int(local_position)
                        best_covered_positions = covered_positions
                        best_distance_factor = float(distance_factors[local_position])

                if best_local_position < 0:
                    break

                # 当当前最高分点已经只能覆盖自己时，log(D)=0，因此这些 D=1 点的
                # 分数退化为静态项，不再依赖会被 batch 内新选点改变的距离修正项。
                # 于是按当前分数排序后，把从第1名开始连续满足 D(x)=1 的前缀一起
                # 选走，与逐个选择并逐步更新 active_mask 严格等效。
                if int(best_covered_positions.size) == 1:
                    ranked_offsets = np.argsort(
                        -score_by_active_position,
                        kind="mergesort"
                    )
                    unit_cover_prefix_length = 0
                    for ranked_offset in ranked_offsets:
                        if covered_count_by_active_position[ranked_offset] != 1:
                            break
                        unit_cover_prefix_length += 1

                    batch_size = min(
                        unit_cover_prefix_length,
                        self.query_budget - len(selected_indices)
                    )
                    batch_offsets = ranked_offsets[:batch_size]
                    batch_local_positions = active_positions[batch_offsets]
                    batch_selected_indices = [
                        int(candidate_indices[local_position])
                        for local_position in batch_local_positions
                    ]
                    selected_indices.extend(batch_selected_indices)
                    active_mask[batch_local_positions] = False
                    for batch_local_position in batch_local_positions:
                        self._update_nearest_distances_after_selection(
                            nearest_distances=nearest_labeled_distances,
                            candidate_distance_matrix=candidate_distance_matrix,
                            active_mask=active_mask,
                            selected_local_position=int(batch_local_position)
                        )
                    progress_bar.update(batch_size)
                    progress_bar.set_postfix(
                        covered=batch_size,
                        remaining=int(active_mask.sum())
                    )
                    print(
                        f"[{self.name}] 触发 D=1 批量选择: "
                        f"batch_size={batch_size}, "
                        f"score_max={score_by_active_position[batch_offsets[0]]:.6f}, "
                        f"score_min={score_by_active_position[batch_offsets[-1]]:.6f}, "
                        f"remaining={int(active_mask.sum())}"
                    )
                    continue

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
