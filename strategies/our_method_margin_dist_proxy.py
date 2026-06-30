"""
Our Method Margin Dist Proxy - margin 评分 + 动态距离乘法修正 + 代理批量更新策略。

该策略以 `our_method_margin_dist` 为基础，只在后续轮次的贪心选择阶段增加一个
“代理批量更新”机制：
1. 正常计算当前 active 候选集上的 `D(x)`、距离因子与采样分数；
2. 若当前最高分点已经满足 `D(x)=1`，说明进入“只能覆盖自己”的尾段；
3. 此时不再一次只选 1 个点，而是按当前分数排序取出从第 1 名开始连续满足
   `D(x)=1` 的前缀中的前 `proxy_batch_size` 个点；
4. 把这一个小批次一起加入已选集合，并立即更新剩余候选点的最近已标注距离；
5. 然后重新计算距离因子、覆盖数和分数，再决定下一小批次。

由于 `our_method_margin_dist` 的距离乘子会在 batch 内动态变化，本策略明确不追求
与原版逐点贪心“严格等效”；它的目标是在接近尾段时减少频繁的逐点重算开销，
同时通过“每小批次后立即重算”把偏离控制在可接受范围内。
"""
from typing import Any, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from .our_method_margin_dist import OurMethodMarginDistStrategy


class OurMethodMarginDistProxyStrategy(OurMethodMarginDistStrategy):
    """
    `our_method_margin_dist` 的代理批量更新版本。

    本类完整复用 `OurMethodMarginDistStrategy` 的半径估计、覆盖过滤、margin 静态项、
    距离因子定义和 batch 内最近距离增量更新逻辑；唯一改变的是当当前最高分已经
    进入 `D(x)=1` 尾段时，不再一次只选 1 个点，而是小批量选择若干个
    `D(x)=1` 前缀样本，再重新计算动态量。

    这样做的目的不是保持与原版逐点贪心完全一致，而是在尾段用更少的重算次数换取
    更高的采样吞吐，同时仍然通过小批量重算保留距离修正的动态性。
    """

    def __init__(
        self,
        *args: Any,
        proxy_batch_size: int = 4,
        **kwargs: Any
    ) -> None:
        """
        初始化 margin_dist 的代理批量更新策略实例。

        该构造函数沿用 `our_method_margin_dist` 的全部已有参数，并额外引入一个
        `proxy_batch_size`：
        1. 当采样仍处于正常覆盖竞争阶段时，行为与父类完全一致；
        2. 当当前最高分点已经满足 `D(x)=1` 时，策略会把当前分数排序里连续的
           `D(x)=1` 前缀按 `proxy_batch_size` 切成小批次；
        3. 每完成一个小批次，就立即更新最近已标注距离，并重新计算下一批次的
           分数，从而近似保留父类的 batch 内动态 diversity 语义。

        参数:
            *args: 传递给父类构造函数的位置参数。
            proxy_batch_size: 进入 `D(x)=1` 尾段后的代理小批次大小，必须为正整数。
            **kwargs: 传递给父类构造函数的关键字参数。

        返回:
            None

        异常:
            ValueError: 当 `proxy_batch_size` 不是正整数时抛出。
        """
        if proxy_batch_size <= 0:
            raise ValueError("proxy_batch_size 必须是正整数。")

        super().__init__(*args, **kwargs)
        self.proxy_batch_size = int(proxy_batch_size)
        self.name = "OurMethodMarginDistProxy"

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
        执行第二轮及之后的 margin + 动态距离乘法修正 + 代理批量采样。

        该方法保留父类的整体采样框架：
        1. 先根据当前已标注点计算覆盖率并过滤候选池；
        2. 对剩余候选点预测概率、估计半径并构造静态 margin 项；
        3. 在贪心循环中维护“到已标注集的最近距离”并由此更新距离因子。

        本方法与父类的唯一区别发生在尾段：
        - 若当前最高分点仍满足 `D(x) > 1`，说明覆盖竞争依旧存在，完全沿用原版
          的逐点贪心；
        - 若当前最高分点已满足 `D(x) = 1`，则按当前分数排序找到从第 1 名开始
          连续满足 `D(x)=1` 的前缀，并只取其中前 `proxy_batch_size` 个点作为
          一个代理小批次；
        - 选中这一小批次后，立即把它们纳入“本轮新增标注点”集合，增量更新剩余
          active 候选点的最近已标注距离；
        - 然后重新计算距离因子、覆盖数和分数，再处理下一批次。

        这种设计保留了 `margin_dist` 中“距离因子会随 batch 内已选点动态变化”的
        核心语义，但用小批量重算代替逐点重算，以减少尾段开销。

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
        else:
            effective_labeled_radii = raw_labeled_radii

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

        selected_indices: List[int] = []
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
                    base_score = static_score_terms[local_position] + np.log(D)
                    score = base_score * distance_factors[local_position]
                    score_by_active_position[active_offset] = score
                    covered_count_by_active_position[active_offset] = D

                    if score > best_score:
                        best_score = float(score)
                        best_local_position = int(local_position)
                        best_covered_positions = covered_positions
                        best_distance_factor = float(distance_factors[local_position])

                if best_local_position < 0:
                    break

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
                        self.proxy_batch_size,
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
                        f"[{self.name}] 触发代理批量选择: "
                        f"batch_size={batch_size}, "
                        f"proxy_batch_size={self.proxy_batch_size}, "
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
