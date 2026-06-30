"""
Our Method Unfil - 基于半径覆盖的主动学习策略（不过滤候选池版本）

该策略整体复用 `our_method_budget_version` 的训练后采样流程与参数接口，
唯一核心差异是：
1. 每轮都会先维护一个“当前已覆盖索引集合”；
2. 已覆盖点不会从无标注候选池中删除；
3. 贪心选择时，`D(x)` 只统计点 `x` 当前能够新增覆盖多少“尚未覆盖”的
   无标注点；
4. 每选中一个点，只更新覆盖集合，不批量删除其覆盖到的候选点。

这样做的目的，是保留那些虽然已经被其他点覆盖、但本身仍处于决策边界附近的
高不确定性样本，使其仍有机会因为不确定性项较高而被选中。
"""
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from .our_method_budget_version import OurMethodBudgetVersionStrategy


class OurMethodUnfilStrategy(OurMethodBudgetVersionStrategy):
    """
    基于半径覆盖的主动学习策略（不过滤候选池版本）。

    该实现继承 `OurMethodBudgetVersionStrategy` 的参数约定、距离矩阵缓存、
    半径计算、概率预测与早期半径归一化逻辑，只重写采样阶段的“候选池维护”
    方式。与父类相比，本类在所有轮次中都遵循以下原则：
    1. 候选集始终保持为当前完整的无标注池；
    2. 通过累计覆盖集合来表达“哪些点已经被已有标注点或本轮已选点覆盖”；
    3. 贪心分数中的 `D(x)` 改为“新增覆盖量”，而不是候选池中的总覆盖量；
    4. 已被覆盖的高不确定样本仍可被再次评估，并在必要时被选中。
    """

    def __init__(
        self,
        epsilon: float = 0.01,
        h_min: float = 1e-6,
        diff_min: float = 0.0,
        spectral_norm_product: Optional[float] = 10.0,
        distance_cache_dir: str = "./distance_cache",
        query_budget: int = 100,
        alpha: float = 0.1,
        inference_batch_size: int = 256,
        enable_early_radius_normalization: bool = False,
        early_radius_normalization_threshold: float = 0.2,
        early_radius_normalization_percentile_k: float = 20.0,
        local_lipschitz_k: int = 50,
        local_lipschitz_quantile: float = 0.90,
        local_lipschitz_distance_eps: float = 1e-12,
        local_lipschitz_min_value: float = 1e-12,
        max_theoretical_radius: Optional[float] = None
    ):
        """
        初始化不过滤候选池版本策略。

        该构造函数保持与 `OurMethodBudgetVersionStrategy` 完全一致的参数接口，
        以便实验脚本、配置解析和日志记录能够无缝复用现有 OurMethod 系列配置。
        初始化过程直接调用父类实现，只在最后把策略名称改为
        `OurMethodUnfil`，用于区分实验结果与日志输出。

        参数:
            epsilon: 半径方程中的 `epsilon` 参数。
            h_min: Hessian l2 谱范数的下界，用于稳定半径计算。
            diff_min: 半径公式中的 `diff` 下界，用于避免分母过小。
            spectral_norm_product: 固定全局放大系数；若为 `None` 则使用每点KNN局部放大系数。
            distance_cache_dir: 距离矩阵缓存目录。
            query_budget: 每轮最多可选择的样本数量。
            alpha: 后续轮次静态不确定性项的权重系数。
            inference_batch_size: 采样阶段批量预测概率时使用的批大小。
            enable_early_radius_normalization: 是否启用早期半径归一化。
            early_radius_normalization_threshold: 触发早期半径归一化的覆盖率阈值。
            early_radius_normalization_percentile_k: 早期半径归一化使用的百分位参数。
            local_lipschitz_k: `spectral_norm_product=None` 时估计每点局部放大
                系数使用的近邻数量。
            local_lipschitz_quantile: `spectral_norm_product=None` 时在每点 KNN
                finite-difference ratios 上取的分位数。
            local_lipschitz_distance_eps: `spectral_norm_product=None` 时距离分母
                使用的数值稳定项。
            local_lipschitz_min_value: `spectral_norm_product=None` 时局部放大系数
                的纯数值下界。
            max_theoretical_radius: 手动指定的最大理论半径；为 `None` 时保留公式
                计算方式。

        返回:
            None
        """
        super().__init__(
            epsilon=epsilon,
            h_min=h_min,
            diff_min=diff_min,
            spectral_norm_product=spectral_norm_product,
            distance_cache_dir=distance_cache_dir,
            query_budget=query_budget,
            alpha=alpha,
            inference_batch_size=inference_batch_size,
            enable_early_radius_normalization=enable_early_radius_normalization,
            early_radius_normalization_threshold=early_radius_normalization_threshold,
            early_radius_normalization_percentile_k=early_radius_normalization_percentile_k,
            local_lipschitz_k=local_lipschitz_k,
            local_lipschitz_quantile=local_lipschitz_quantile,
            local_lipschitz_distance_eps=local_lipschitz_distance_eps,
            local_lipschitz_min_value=local_lipschitz_min_value,
            max_theoretical_radius=max_theoretical_radius
        )
        if self.alpha is None:
            raise ValueError(
                "alpha=None 自动尺度校准仅支持 our_method_budget_version、"
                "our_method_margin 和 our_method_margin_weighted。"
            )
        self.name = "OurMethodUnfil"

    def _greedy_selection_without_filtering(
        self,
        unlabeled_indices: List[int],
        distance_matrix: np.ndarray,
        radii: np.ndarray,
        initial_covered_mask: np.ndarray,
        static_score_terms: Optional[np.ndarray],
        progress_desc: str,
        allow_zero_gain_selection: bool
    ) -> Tuple[List[int], np.ndarray]:
        """
        在“不删除已覆盖候选点”的前提下执行贪心采样。

        该函数是本策略与父类行为差异的核心实现。它把“候选点是否还能被评估”
        与“该点还能带来多少新增覆盖”这两个概念解耦：
        1. `unlabeled_indices` 中的所有点在整个贪心过程中都保留为候选；
        2. 只有已经被选中的点会从“可继续被选”的集合中移除，避免重复选择；
        3. `initial_covered_mask` 用于记录本轮开始前就已被覆盖的样本；
        4. 每次评估候选点 `x` 时，只把其半径范围内、且当前还未被覆盖的
           无标注点计入 `D(x)`；
        5. 若 `static_score_terms` 为 `None`，表示第一轮纯覆盖贪心，只依据
           新增覆盖量做选择；
        6. 若 `allow_zero_gain_selection=False`，则当所有候选点都无法新增覆盖时
           立即停止，避免第一轮在所有点都已覆盖时继续做无意义的任意选择；
        7. 每次选中一个点后，只更新累计覆盖掩码，不批量删除其覆盖到的候选点。

        参数:
            unlabeled_indices: 当前轮完整的无标注样本索引列表。
            distance_matrix: 全量样本距离矩阵。
            radii: 与 `unlabeled_indices` 一一对应的候选点生效半径数组。
            initial_covered_mask: 当前轮开始前的全量覆盖掩码。函数内部会复制后
                再更新，保证调用方的原始掩码不被原地修改。
            static_score_terms: 与候选点顺序对齐的静态评分项数组。第一轮若只做
                最大新增覆盖贪心，可传入 `None`。
            progress_desc: 进度条描述文本。
            allow_zero_gain_selection: 当候选点新增覆盖量为 0 时，是否仍允许依赖
                静态不确定性项继续选择。第一轮应为 `False`，后续轮次应为 `True`。

        返回:
            Tuple[List[int], np.ndarray]:
                - selected_indices: 本轮按顺序选出的样本全局索引列表；
                - covered_mask: 完成本轮贪心后更新得到的全量覆盖掩码。

        异常:
            ValueError: 当 `radii`、`initial_covered_mask` 与输入索引长度不匹配时
                抛出。
        """
        if len(unlabeled_indices) != len(radii):
            raise ValueError("radii 的长度必须与 unlabeled_indices 一致。")
        if initial_covered_mask.ndim != 1:
            raise ValueError("initial_covered_mask 必须是一维布尔数组。")
        if initial_covered_mask.shape[0] != distance_matrix.shape[0]:
            raise ValueError(
                "initial_covered_mask 的长度必须与距离矩阵样本总数一致。"
            )

        selected_indices: List[int] = []
        if not unlabeled_indices:
            return selected_indices, initial_covered_mask.copy()

        candidate_indices = np.asarray(unlabeled_indices, dtype=np.int64)
        candidate_radii = np.asarray(radii, dtype=np.float64)
        if static_score_terms is None:
            candidate_static_terms = np.zeros(candidate_indices.shape[0], dtype=np.float64)
        else:
            candidate_static_terms = np.asarray(static_score_terms, dtype=np.float64)
            if candidate_static_terms.shape[0] != candidate_indices.shape[0]:
                raise ValueError(
                    "static_score_terms 的长度必须与 unlabeled_indices 一致。"
                )

        covered_mask = initial_covered_mask.astype(bool, copy=True)
        local_distance_matrix = distance_matrix[np.ix_(candidate_indices, candidate_indices)]
        active_mask = np.ones(candidate_indices.shape[0], dtype=bool)

        with tqdm(
            total=min(self.query_budget, len(unlabeled_indices)),
            desc=progress_desc,
            unit="sample"
        ) as progress_bar:
            while active_mask.any() and len(selected_indices) < self.query_budget:
                active_positions = np.flatnonzero(active_mask)
                candidate_covered_mask = covered_mask[candidate_indices]

                best_local_position = -1
                best_score = -np.inf
                best_new_coverage_count = -1
                best_all_covered_positions = np.empty((0,), dtype=np.int64)

                for local_position in active_positions:
                    radius = candidate_radii[local_position]
                    all_covered_positions = np.flatnonzero(
                        local_distance_matrix[local_position] <= radius
                    )
                    uncovered_positions = all_covered_positions[
                        ~candidate_covered_mask[all_covered_positions]
                    ]
                    new_coverage_count = int(uncovered_positions.size)
                    score = float(candidate_static_terms[local_position]) + np.log(
                        max(new_coverage_count, 1)
                    )

                    if (
                        score > best_score
                        or (
                            np.isclose(score, best_score)
                            and new_coverage_count > best_new_coverage_count
                        )
                    ):
                        best_local_position = int(local_position)
                        best_score = score
                        best_new_coverage_count = new_coverage_count
                        best_all_covered_positions = all_covered_positions

                if best_local_position < 0:
                    break
                if best_new_coverage_count <= 0 and not allow_zero_gain_selection:
                    break

                selected_idx = int(candidate_indices[best_local_position])
                selected_indices.append(selected_idx)
                active_mask[best_local_position] = False
                covered_mask[candidate_indices[best_all_covered_positions]] = True

                progress_bar.update(1)
                progress_bar.set_postfix(
                    new_cover=best_new_coverage_count,
                    selected=len(selected_indices)
                )
                print(
                    f"[OurMethodUnfil] 选择样本 {selected_idx}, "
                    f"分数={best_score:.6f}, 新增覆盖点数={best_new_coverage_count}"
                )

        return selected_indices, covered_mask

    def _first_round_selection(
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
        执行第一轮采样，但不因“已覆盖”而过滤候选池。

        第一轮仍然与父类一样使用统一的最大理论半径，但对“已覆盖”的处理方式
        改为：
        1. 先根据初始已标注点与最大理论半径构造本轮初始覆盖集合；
        2. 保留完整的 `unlabeled_indices` 作为候选集，不移除这些已覆盖点；
        3. 对每个候选点只统计其还能新增覆盖多少当前未覆盖的无标注点；
        4. 使用该新增覆盖量做纯覆盖贪心；
        5. 一旦所有候选点都无法带来新增覆盖，就提前停止，避免做任意选择。

        参数:
            model: 当前轮已训练完成的模型。该参数在第一轮中不直接参与计算，
                但保留与父类完全一致的方法签名，便于无缝替换。
            dataset: 完整训练数据集。该参数在第一轮中不直接参与计算，
                同样保留签名兼容性。
            device: 当前计算设备。第一轮中不直接使用，但保留兼容签名。
            distance_matrix: 全量样本距离矩阵。
            labeled_indices: 当前已标注样本索引列表。
            unlabeled_indices: 当前无标注样本索引列表。
            spectral_norm_product: 固定全局放大系数；None 表示使用逐点局部放大系数。

        返回:
            List[int]: 第一轮选出的样本索引列表。
        """
        del model, dataset, device

        print("\n[OurMethodUnfil] 第一轮采样")

        round_indices = list(labeled_indices) + list(unlabeled_indices)
        round_max_radii = self._compute_max_theoretical_radii_for_indices(
            round_indices,
            spectral_norm_product
        )
        round_radius_by_index = {
            index: float(radius)
            for index, radius in zip(round_indices, round_max_radii)
        }
        if round_max_radii.size > 0:
            print(
                "[OurMethodUnfil] 最大理论半径统计: "
                f"min={round_max_radii.min():.6f}, "
                f"max={round_max_radii.max():.6f}, "
                f"mean={round_max_radii.mean():.6f}"
            )

        initial_covered_mask = np.zeros(distance_matrix.shape[0], dtype=bool)
        if labeled_indices:
            for idx in labeled_indices:
                initial_covered_mask |= distance_matrix[idx] <= round_radius_by_index[idx]

        initially_covered_unlabeled = int(initial_covered_mask[unlabeled_indices].sum())
        print(f"[OurMethodUnfil] 原始无标注池大小: {len(unlabeled_indices)}")
        print(
            f"[OurMethodUnfil] 初始已覆盖的无标注点数: {initially_covered_unlabeled}"
        )

        if not unlabeled_indices:
            print("[OurMethodUnfil] 无标注池为空，无需选择样本")
            return []

        radii = np.array(
            [round_radius_by_index[idx] for idx in unlabeled_indices],
            dtype=np.float64
        )
        selected_indices, final_covered_mask = self._greedy_selection_without_filtering(
            unlabeled_indices=unlabeled_indices,
            distance_matrix=distance_matrix,
            radii=radii,
            initial_covered_mask=initial_covered_mask,
            static_score_terms=None,
            progress_desc="[OurMethodUnfil] 第一轮选择进度",
            allow_zero_gain_selection=False
        )

        print(f"[OurMethodUnfil] 第一轮选择的标注点数量: {len(selected_indices)}")
        print(
            "[OurMethodUnfil] 第一轮结束时无标注池中的累计已覆盖点数: "
            f"{int(final_covered_mask[unlabeled_indices].sum())}"
        )
        return selected_indices

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
        执行后续轮次采样，但不因“已覆盖”而过滤候选池。

        该方法完整保留父类的半径估计与打分公式，只把候选池维护逻辑改成
        “累计覆盖集合驱动”的形式。具体流程如下：
        1. 计算已标注点半径，并得到当前轮开始前的覆盖率与覆盖掩码；
        2. 若触发早期半径归一化，则复用父类逻辑放大已标注点半径，并同步更新
           覆盖率与初始覆盖掩码；
        3. 不再构造 `filtered_unlabeled`，而是直接对完整 `unlabeled_indices`
           预测概率、估计半径并计算静态不确定性项；
        4. 对任一候选点 `x`，其 `D(x)` 只统计在当前累计覆盖集合之外，
           `x` 还能新增覆盖多少无标注点；
        5. 每选择一个点，只把它覆盖到的点并入累计覆盖集合，不批量删除候选；
        6. 即使某个候选点已在累计覆盖集合中，只要其不确定性足够高，仍可能被选。

        参数:
            model: 当前轮训练完成后的模型。
            dataset: 完整训练数据集。
            device: 当前计算设备。
            distance_matrix: 全量样本距离矩阵。
            labeled_indices: 当前已标注样本索引列表。
            unlabeled_indices: 当前无标注样本索引列表。
            spectral_norm_product: 固定全局放大系数；None 表示使用逐点局部放大系数。

        返回:
            List[int]: 本轮选出的样本索引列表。
        """
        print(f"\n[OurMethodUnfil] 第{self.round_counter + 1}轮采样")

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
        print(f"[OurMethodUnfil] 当前轮次覆盖率: {coverage_ratio:.6f}")

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
                coverage_ratio, covered_mask = self._compute_coverage(
                    distance_matrix,
                    labeled_indices,
                    effective_labeled_radii.tolist(),
                )
                self.last_coverage = coverage_ratio
                print(
                    "[OurMethodUnfil] 早期半径归一化已触发: "
                    f"T={self.early_radius_normalization_threshold:.6f}, "
                    f"k={self.early_radius_normalization_percentile_k:.2f}, "
                    f"q_k={normalization_pivot_radius:.6f}, "
                    f"r={normalization_scale_ratio:.6f}, "
                    f"max_radius={max_theoretical_radius:.6f}, "
                    f"normalized_coverage={coverage_ratio:.6f}"
                )
            else:
                print(
                    "[OurMethodUnfil] 早期半径归一化未触发: "
                    f"coverage={coverage_ratio:.6f} >= "
                    f"T={self.early_radius_normalization_threshold:.6f}"
                )

        print(f"[OurMethodUnfil] 原始无标注池大小: {len(unlabeled_indices)}")
        print(
            "[OurMethodUnfil] 已被当前标注集覆盖的无标注点数: "
            f"{int(covered_mask[unlabeled_indices].sum())}"
        )

        if not unlabeled_indices:
            print("[OurMethodUnfil] 无标注池为空，无需选择样本")
            return []

        probabilities = self._get_probabilities_for_indices(
            model, dataset, unlabeled_indices, device
        )
        h_values = self._compute_h_from_probabilities(probabilities)
        estimated_radii = self._estimate_radii_for_unlabeled(
            h_values=h_values,
            unlabeled_indices=unlabeled_indices,
            spectral_norm_product=spectral_norm_product
        )
        if normalization_applied:
            estimated_radii = np.minimum(
                estimated_radii * normalization_scale_ratio,
                max_theoretical_radius,
            )
            estimated_radii = self._clip_radii_to_max_theoretical(
                radii=estimated_radii,
                indices=unlabeled_indices,
                spectral_norm_product=spectral_norm_product
            )

        print(
            f"[OurMethodUnfil] 预估半径统计: min={estimated_radii.min():.6f}, "
            f"max={estimated_radii.max():.6f}, mean={estimated_radii.mean():.6f}"
        )

        static_score_terms = self._compute_static_score_terms(
            probabilities,
            coverage_ratio
        )

        selected_indices, final_covered_mask = self._greedy_selection_without_filtering(
            unlabeled_indices=unlabeled_indices,
            distance_matrix=distance_matrix,
            radii=estimated_radii,
            initial_covered_mask=covered_mask,
            static_score_terms=static_score_terms,
            progress_desc=f"[OurMethodUnfil] 第{self.round_counter + 1}轮选择进度",
            allow_zero_gain_selection=True
        )

        print(
            f"[OurMethodUnfil] 第{self.round_counter + 1}轮选择的标注点数量: "
            f"{len(selected_indices)}"
        )
        print(
            "[OurMethodUnfil] 本轮结束时无标注池中的累计已覆盖点数: "
            f"{int(final_covered_mask[unlabeled_indices].sum())}"
        )
        return selected_indices
