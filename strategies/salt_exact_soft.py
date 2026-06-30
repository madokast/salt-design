"""
SALT exact soft-coverage strategy.

第一轮完全复用 ``SALTExactStrategy``。后续轮次不再从候选池中删除已被
标注点覆盖的样本，而是让这些样本以 ``D=1`` 继续参与 margin 评分。只有
当前仍未覆盖的候选点能够用预估半径扩张本轮覆盖；已覆盖候选被选中时生效
半径固定为零，因此其下一轮复制权重保持为一。
"""

import heapq
from typing import Any, List, Optional, Tuple

import numpy as np
import torch

from .salt_exact import CandidateCoverCSR, SALTExactStrategy, SparseRadiusGraph


class SALTExactSoftStrategy(SALTExactStrategy):
    """
    保留已覆盖候选的 SALT exact 软过滤版本。

    本类只重写后续轮次选择逻辑。第一轮仍由父类使用统一最大理论半径完成
    硬过滤和覆盖贪心；从第二轮开始，完整无标注池都会保留在可选择集合中。
    已覆盖候选只有 margin 不确定性收益，未覆盖候选同时具有动态覆盖收益。
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        使用与 ``SALTExactStrategy`` 完全相同的参数初始化软覆盖策略。

        所有半径、局部 Lipschitz、最大半径图和复制式训练参数均直接交给父类，
        避免软覆盖版本产生另一套不兼容配置。这里只更新展示名称；CSR 最大半径
        图仍复用 ``salt_exact`` 的磁盘缓存，因为两种策略的底层距离关系相同。

        参数:
            *args: 传给 ``SALTExactStrategy`` 的位置参数。
            **kwargs: 传给 ``SALTExactStrategy`` 的关键字参数。

        返回:
            None
        """
        super().__init__(*args, **kwargs)
        self.name = "SALTExactSoft"

    def _push_soft_heap_entry(
        self,
        candidate_position: int,
        static_score_terms: np.ndarray,
        dynamic_counts: np.ndarray,
        versions: np.ndarray,
        unit_heap: List[Tuple[float, int, int]],
        nonunit_heap: List[Tuple[float, int, int]],
    ) -> None:
        """
        将候选点当前版本的评分写入对应的延迟更新最大堆。

        Python 只提供最小堆，因此存储 ``-score``。候选位置作为第二排序键，
        保证同分时与 numpy ``argmax`` 的“选择较早候选”行为一致。每次动态
        覆盖计数改变时，调用方先增加 ``versions``；旧堆项随后会因版本不匹配
        被忽略，从而避免在堆中执行昂贵的原地更新或删除。

        参数:
            candidate_position: 候选点在完整无标注候选数组中的局部位置。
            static_score_terms: 每个候选固定不变的 margin 评分项。
            dynamic_counts: 每个候选当前生效的 ``D``；已覆盖点固定为一。
            versions: 每个候选当前有效堆项的版本号。
            unit_heap: 存放 ``D=1`` 候选的堆。
            nonunit_heap: 存放 ``D>1`` 候选的堆。

        返回:
            None
        """
        position = int(candidate_position)
        count = max(int(dynamic_counts[position]), 1)
        score = float(static_score_terms[position] + np.log(count))
        entry = (-score, position, int(versions[position]))
        if count == 1:
            heapq.heappush(unit_heap, entry)
        else:
            heapq.heappush(nonunit_heap, entry)

    def _peek_valid_soft_heap_entry(
        self,
        heap: List[Tuple[float, int, int]],
        expect_unit: bool,
        active_mask: np.ndarray,
        dynamic_counts: np.ndarray,
        versions: np.ndarray,
    ) -> Optional[Tuple[float, int, int]]:
        """
        清理一个延迟更新堆并返回当前有效的堆顶项。

        堆中可能保留已经被选择、版本过期或从 ``D>1`` 转为 ``D=1`` 的旧项。
        本函数逐个丢弃这些无效项，直到堆顶同时满足活跃状态、版本一致和堆类别
        一致。它只查看有效堆顶而不弹出，便于调用方比较 unit 与 non-unit 候选。

        参数:
            heap: 待清理的 unit 或 non-unit 堆。
            expect_unit: ``True`` 表示该堆应只接受 ``D=1`` 的当前项。
            active_mask: 候选是否仍可被选择的布尔数组。
            dynamic_counts: 候选当前生效的覆盖计数。
            versions: 候选当前版本号。

        返回:
            Optional[Tuple[float, int, int]]: 有效堆顶；堆耗尽时返回 ``None``。
        """
        while heap:
            entry = heap[0]
            _, position, version = entry
            current_is_unit = int(dynamic_counts[position]) == 1
            if (
                not bool(active_mask[position])
                or int(versions[position]) != int(version)
                or current_is_unit != expect_unit
            ):
                heapq.heappop(heap)
                continue
            return entry
        return None

    def _apply_soft_coverage_update(
        self,
        selected_uncovered_position: int,
        cover_data: CandidateCoverCSR,
        uncovered_mask: np.ndarray,
        uncovered_to_candidate_positions: np.ndarray,
        dynamic_counts: np.ndarray,
        active_mask: np.ndarray,
        versions: np.ndarray,
        static_score_terms: np.ndarray,
        unit_heap: List[Tuple[float, int, int]],
        nonunit_heap: List[Tuple[float, int, int]],
    ) -> int:
        """
        用一个新选未覆盖点的真实覆盖集合增量更新所有动态 ``D``。

        ``cover_data`` 的局部坐标只包含本轮开始时未覆盖的候选。函数先找出
        当前仍未覆盖、且落入所选点半径的目标；这些目标只会在第一次被覆盖时
        触发更新。随后通过反向 CSR 汇总每个候选中心损失的覆盖目标数，一次性
        减少其 ``D`` 并写入新堆项。刚变成已覆盖的候选被强制设置为 ``D=1``，
        即使其原覆盖集合中仍有其他未覆盖点，也不再享有覆盖收益。

        参数:
            selected_uncovered_position: 所选中心在“初始未覆盖子池”中的位置。
            cover_data: 初始未覆盖子池的正向与反向覆盖 CSR。
            uncovered_mask: 子池中当前仍未覆盖的状态数组；本函数会原地修改。
            uncovered_to_candidate_positions: 子池位置到完整候选位置的映射。
            dynamic_counts: 完整候选池当前生效的 ``D``；本函数会原地修改。
            active_mask: 完整候选池是否仍可选择的状态。
            versions: 完整候选池堆版本号；本函数会原地递增。
            static_score_terms: 完整候选池的静态 margin 评分项。
            unit_heap: ``D=1`` 候选延迟更新堆。
            nonunit_heap: ``D>1`` 候选延迟更新堆。

        返回:
            int: 本次从未覆盖变成已覆盖的候选数量。
        """
        covered_targets = self._candidate_cover_row(
            cover_data=cover_data,
            local_position=int(selected_uncovered_position),
        )
        newly_covered_targets = covered_targets[uncovered_mask[covered_targets]]
        if newly_covered_targets.size == 0:
            return 0

        # 先冻结新覆盖目标，使这些候选后续无条件采用 D=1。
        uncovered_mask[newly_covered_targets] = False
        newly_covered_candidate_positions = uncovered_to_candidate_positions[
            newly_covered_targets
        ]
        dynamic_counts[newly_covered_candidate_positions] = 1

        # 汇总所有反向边；同一中心可能同时损失多个目标，只生成一个新堆版本。
        reverse_rows: List[np.ndarray] = []
        for target_position in newly_covered_targets:
            start = int(cover_data.reverse_indptr[int(target_position)])
            end = int(cover_data.reverse_indptr[int(target_position) + 1])
            if start < end:
                reverse_rows.append(
                    cover_data.reverse_indices[start:end].astype(
                        np.int64,
                        copy=False,
                    )
                )

        affected_uncovered_centers = np.empty((0,), dtype=np.int64)
        if reverse_rows:
            all_affected_centers = np.concatenate(reverse_rows)
            unique_affected_centers, decrement_counts = np.unique(
                all_affected_centers,
                return_counts=True,
            )
            still_uncovered = uncovered_mask[unique_affected_centers]
            affected_uncovered_centers = unique_affected_centers[still_uncovered]
            active_decrement_counts = decrement_counts[still_uncovered]
            affected_candidate_positions = uncovered_to_candidate_positions[
                affected_uncovered_centers
            ]
            dynamic_counts[affected_candidate_positions] -= active_decrement_counts
            if np.any(dynamic_counts[affected_candidate_positions] < 1):
                raise RuntimeError("SALTExactSoft 动态覆盖计数不应小于 1。")

        changed_candidate_positions = np.unique(np.concatenate([
            newly_covered_candidate_positions,
            uncovered_to_candidate_positions[affected_uncovered_centers],
        ]))
        for candidate_position in changed_candidate_positions:
            position = int(candidate_position)
            versions[position] += 1
            if active_mask[position]:
                self._push_soft_heap_entry(
                    candidate_position=position,
                    static_score_terms=static_score_terms,
                    dynamic_counts=dynamic_counts,
                    versions=versions,
                    unit_heap=unit_heap,
                    nonunit_heap=nonunit_heap,
                )

        return int(newly_covered_targets.size)

    def _soft_greedy_select_candidate_positions(
        self,
        candidate_indices: np.ndarray,
        candidate_radii: np.ndarray,
        static_score_terms: np.ndarray,
        cover_data: CandidateCoverCSR,
        uncovered_to_candidate_positions: np.ndarray,
        query_budget: int,
    ) -> Tuple[List[int], List[float], int]:
        """
        使用双延迟堆执行软覆盖贪心，并保留安全的 ``D=1`` 批处理。

        完整候选池包含初始已覆盖点和未覆盖点，而覆盖 CSR 只描述初始未覆盖
        子池。初始已覆盖点以及本轮中新覆盖的点都固定为 ``D=1``；选中它们时
        返回半径零。当前仍未覆盖的候选使用 CSR 边数作为动态 ``D``，选中后
        才会扩张覆盖，并通过反向 CSR 降低相关候选收益。

        当全局最优项来自 unit 堆时，本函数批量弹出所有严格领先于当前最佳
        non-unit 项的 unit 前缀；同分时用候选位置保持稳定顺序。unit 选择只会
        让 non-unit 分数下降而不会上升，因此该批处理与逐点贪心等价。

        参数:
            candidate_indices: 完整无标注候选的全局索引数组。
            candidate_radii: 完整候选的预估半径；初始已覆盖位置为零。
            static_score_terms: 完整候选的静态 margin 评分项。
            cover_data: 初始未覆盖子池的候选覆盖 CSR。
            uncovered_to_candidate_positions: 未覆盖子池到完整候选池的位置映射。
            query_budget: 本轮最多选择的样本数量。

        返回:
            Tuple[List[int], List[float], int]: 依次为所选完整候选位置、与选择时
            覆盖状态对应的生效半径，以及本轮新增覆盖的候选总数。
        """
        num_candidates = int(candidate_indices.shape[0])
        num_uncovered = int(uncovered_to_candidate_positions.shape[0])
        selected_positions: List[int] = []
        selected_radii: List[float] = []
        if num_candidates == 0 or query_budget <= 0:
            return selected_positions, selected_radii, 0

        candidate_to_uncovered_positions = np.full(
            num_candidates,
            -1,
            dtype=np.int64,
        )
        candidate_to_uncovered_positions[uncovered_to_candidate_positions] = np.arange(
            num_uncovered,
            dtype=np.int64,
        )
        uncovered_mask = np.ones(num_uncovered, dtype=bool)
        active_mask = np.ones(num_candidates, dtype=bool)
        dynamic_counts = np.ones(num_candidates, dtype=np.int64)
        dynamic_counts[uncovered_to_candidate_positions] = cover_data.covered_counts
        if np.any(dynamic_counts < 1):
            raise RuntimeError("SALTExactSoft 的未覆盖候选必须至少覆盖自身。")

        versions = np.zeros(num_candidates, dtype=np.int64)
        unit_heap: List[Tuple[float, int, int]] = []
        nonunit_heap: List[Tuple[float, int, int]] = []
        for candidate_position in range(num_candidates):
            self._push_soft_heap_entry(
                candidate_position=candidate_position,
                static_score_terms=static_score_terms,
                dynamic_counts=dynamic_counts,
                versions=versions,
                unit_heap=unit_heap,
                nonunit_heap=nonunit_heap,
            )

        total_newly_covered = 0
        effective_budget = min(int(query_budget), num_candidates)
        while len(selected_positions) < effective_budget:
            best_unit = self._peek_valid_soft_heap_entry(
                heap=unit_heap,
                expect_unit=True,
                active_mask=active_mask,
                dynamic_counts=dynamic_counts,
                versions=versions,
            )
            best_nonunit = self._peek_valid_soft_heap_entry(
                heap=nonunit_heap,
                expect_unit=False,
                active_mask=active_mask,
                dynamic_counts=dynamic_counts,
                versions=versions,
            )
            if best_unit is None and best_nonunit is None:
                break

            # 堆键已经包含稳定位置 tie-break，元组较小者就是全局贪心最优项。
            choose_unit = best_unit is not None and (
                best_nonunit is None or best_unit[:2] < best_nonunit[:2]
            )
            if not choose_unit:
                _, candidate_position, _ = heapq.heappop(nonunit_heap)
                position = int(candidate_position)
                active_mask[position] = False
                selected_positions.append(position)
                selected_radii.append(float(candidate_radii[position]))
                uncovered_position = int(candidate_to_uncovered_positions[position])
                total_newly_covered += self._apply_soft_coverage_update(
                    selected_uncovered_position=uncovered_position,
                    cover_data=cover_data,
                    uncovered_mask=uncovered_mask,
                    uncovered_to_candidate_positions=uncovered_to_candidate_positions,
                    dynamic_counts=dynamic_counts,
                    active_mask=active_mask,
                    versions=versions,
                    static_score_terms=static_score_terms,
                    unit_heap=unit_heap,
                    nonunit_heap=nonunit_heap,
                )
                continue

            # 固定当前 non-unit 门槛。该门槛在 unit 选择后只可能下降，故前缀安全。
            nonunit_threshold = best_nonunit[:2] if best_nonunit is not None else None
            remaining_budget = effective_budget - len(selected_positions)
            batch_positions: List[int] = []
            while len(batch_positions) < remaining_budget:
                current_unit = self._peek_valid_soft_heap_entry(
                    heap=unit_heap,
                    expect_unit=True,
                    active_mask=active_mask,
                    dynamic_counts=dynamic_counts,
                    versions=versions,
                )
                if current_unit is None:
                    break
                if (
                    nonunit_threshold is not None
                    and current_unit[:2] >= nonunit_threshold
                ):
                    break
                _, candidate_position, _ = heapq.heappop(unit_heap)
                batch_positions.append(int(candidate_position))

            if not batch_positions:
                # 理论上 choose_unit 已保证至少有一个安全项；此分支防止数值异常死循环。
                _, candidate_position, _ = heapq.heappop(unit_heap)
                batch_positions.append(int(candidate_position))

            expansion_uncovered_positions: List[int] = []
            for position in batch_positions:
                uncovered_position = int(candidate_to_uncovered_positions[position])
                was_uncovered = (
                    uncovered_position >= 0 and uncovered_mask[uncovered_position]
                )
                active_mask[position] = False
                selected_positions.append(position)
                if was_uncovered:
                    selected_radii.append(float(candidate_radii[position]))
                    expansion_uncovered_positions.append(uncovered_position)
                else:
                    selected_radii.append(0.0)

            for uncovered_position in expansion_uncovered_positions:
                total_newly_covered += self._apply_soft_coverage_update(
                    selected_uncovered_position=uncovered_position,
                    cover_data=cover_data,
                    uncovered_mask=uncovered_mask,
                    uncovered_to_candidate_positions=uncovered_to_candidate_positions,
                    dynamic_counts=dynamic_counts,
                    active_mask=active_mask,
                    versions=versions,
                    static_score_terms=static_score_terms,
                    unit_heap=unit_heap,
                    nonunit_heap=nonunit_heap,
                )

            print(
                "[SALTExactSoft] 触发 D=1 批量选择: "
                f"batch_size={len(batch_positions)}, "
                f"remaining={int(active_mask.sum())}"
            )

        return selected_positions, selected_radii, total_newly_covered

    def _subsequent_round_selection_sparse(
        self,
        model: torch.nn.Module,
        dataset: Any,
        device: torch.device,
        radius_graph: SparseRadiusGraph,
        labeled_indices: List[int],
        unlabeled_indices: List[int],
        spectral_norm_product: Optional[float],
    ) -> List[int]:
        """
        执行第二轮及之后的 SALT exact 软覆盖选择。

        该流程先按父类公式计算已标注点半径、覆盖率以及可选的早期半径归一化，
        但不会过滤被覆盖的无标注点。模型概率在完整无标注池上计算，用于所有点
        的 margin 项；候选半径只为初始未覆盖子池估计。随后以该子池构建精确
        覆盖 CSR，并调用双堆贪心直到预算耗尽。选择时已覆盖的点生效半径为零，
        从而在下一轮复制权重计算中保持基础权重一。

        参数:
            model: 当前轮训练完成后的模型。
            dataset: 完整训练数据集。
            device: 模型推理和距离计算设备。
            radius_graph: 父类加载的最大理论半径 CSR 图。
            labeled_indices: 当前已标注样本的全局索引。
            unlabeled_indices: 当前完整无标注样本的全局索引。
            spectral_norm_product: 固定全局放大系数；``None`` 表示局部模式。

        返回:
            List[int]: 本轮选中的全局样本索引，数量最多为查询预算。
        """
        print(f"\n[SALTExactSoft] 第{self.round_counter + 1}轮采样")

        max_theoretical_radius = float(self.max_theoretical_radius)
        labeled_points = self._build_labeled_points(dataset, labeled_indices)
        raw_labeled_radii = self._compute_labeled_radii(
            labeled_points=labeled_points,
            labeled_indices=labeled_indices,
            model=model,
            spectral_norm_product=spectral_norm_product,
        )
        coverage_ratio, covered_mask = self._compute_sparse_coverage(
            radius_graph=radius_graph,
            labeled_indices=labeled_indices,
            radii=raw_labeled_radii.tolist(),
        )
        self.last_coverage = coverage_ratio
        print(f"[SALTExactSoft] 当前轮次覆盖率: {coverage_ratio:.6f}")

        effective_labeled_radii = raw_labeled_radii
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
                    spectral_norm_product=spectral_norm_product,
                )
                coverage_ratio, covered_mask = self._compute_sparse_coverage(
                    radius_graph=radius_graph,
                    labeled_indices=labeled_indices,
                    radii=effective_labeled_radii.tolist(),
                )
                self.last_coverage = coverage_ratio
                print(
                    "[SALTExactSoft] 早期半径归一化已触发: "
                    f"T={self.early_radius_normalization_threshold:.6f}, "
                    f"k={self.early_radius_normalization_percentile_k:.2f}, "
                    f"q_k={normalization_pivot_radius:.6f}, "
                    f"r={normalization_scale_ratio:.6f}, "
                    f"max_radius={max_theoretical_radius:.6f}, "
                    f"normalized_coverage={coverage_ratio:.6f}"
                )
            else:
                print(
                    "[SALTExactSoft] 早期半径归一化未触发: "
                    f"coverage={coverage_ratio:.6f} >= "
                    f"T={self.early_radius_normalization_threshold:.6f}"
                )

        candidate_indices = np.asarray(unlabeled_indices, dtype=np.int64)
        initially_covered = covered_mask[candidate_indices]
        uncovered_candidate_positions = np.flatnonzero(~initially_covered)
        print(f"[SALTExactSoft] 无标注池大小: {candidate_indices.size}")
        print(
            "[SALTExactSoft] 保留的已覆盖候选数: "
            f"{int(np.count_nonzero(initially_covered))}"
        )
        print(
            "[SALTExactSoft] 初始未覆盖候选数: "
            f"{int(uncovered_candidate_positions.size)}"
        )

        if candidate_indices.size == 0:
            self._prepare_next_round_training_setup_sparse(
                radius_graph=radius_graph,
                current_labeled_indices=labeled_indices,
                current_labeled_radii=effective_labeled_radii.tolist(),
                selected_indices=[],
                selected_radii=[],
            )
            return []

        probabilities = self._get_probabilities_for_indices(
            model,
            dataset,
            unlabeled_indices,
            device,
        )
        candidate_radii = np.zeros(candidate_indices.shape[0], dtype=np.float64)
        if uncovered_candidate_positions.size > 0:
            uncovered_probabilities = probabilities[uncovered_candidate_positions]
            uncovered_h_values = self._compute_h_from_probabilities(
                uncovered_probabilities
            )
            uncovered_indices = candidate_indices[
                uncovered_candidate_positions
            ].astype(np.int64, copy=False).tolist()
            uncovered_radii = self._estimate_radii_for_unlabeled(
                h_values=uncovered_h_values,
                unlabeled_indices=uncovered_indices,
                spectral_norm_product=spectral_norm_product,
            )
            if normalization_applied:
                uncovered_radii = np.minimum(
                    uncovered_radii * normalization_scale_ratio,
                    max_theoretical_radius,
                )
                uncovered_radii = self._clip_radii_to_max_theoretical(
                    radii=uncovered_radii,
                    indices=uncovered_indices,
                    spectral_norm_product=spectral_norm_product,
                )
            candidate_radii[uncovered_candidate_positions] = uncovered_radii
            print(
                "[SALTExactSoft] 未覆盖候选预估半径统计: "
                f"min={uncovered_radii.min():.6f}, "
                f"max={uncovered_radii.max():.6f}, "
                f"mean={uncovered_radii.mean():.6f}"
            )

        uncovered_indices_array = candidate_indices[uncovered_candidate_positions]
        uncovered_radii_array = candidate_radii[uncovered_candidate_positions]
        cover_data = self._prepare_candidate_cover_csr_from_radius_graph(
            radius_graph=radius_graph,
            candidate_indices=uncovered_indices_array,
            candidate_radii=uncovered_radii_array,
        )
        initial_dynamic_counts = np.ones(candidate_indices.shape[0], dtype=np.int64)
        initial_dynamic_counts[uncovered_candidate_positions] = cover_data.covered_counts
        initial_log_coverage_terms = np.log(
            np.maximum(initial_dynamic_counts, 1)
        )
        static_score_terms = self._compute_static_score_terms(
            probabilities,
            coverage_ratio,
            coverage_terms=initial_log_coverage_terms,
        )

        (
            selected_candidate_positions,
            selected_radii,
            newly_covered_count,
        ) = self._soft_greedy_select_candidate_positions(
            candidate_indices=candidate_indices,
            candidate_radii=candidate_radii,
            static_score_terms=static_score_terms,
            cover_data=cover_data,
            uncovered_to_candidate_positions=uncovered_candidate_positions,
            query_budget=self.query_budget,
        )
        selected_indices = [
            int(candidate_indices[position])
            for position in selected_candidate_positions
        ]
        zero_radius_count = int(np.count_nonzero(np.asarray(selected_radii) == 0.0))
        print(
            f"[SALTExactSoft] 第{self.round_counter + 1}轮选择数量: "
            f"{len(selected_indices)}, 新增覆盖: {newly_covered_count}, "
            f"零半径选择: {zero_radius_count}"
        )

        self._prepare_next_round_training_setup_sparse(
            radius_graph=radius_graph,
            current_labeled_indices=labeled_indices,
            current_labeled_radii=effective_labeled_radii.tolist(),
            selected_indices=selected_indices,
            selected_radii=selected_radii,
        )
        return selected_indices
