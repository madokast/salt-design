"""
Our Method Margin Weighted - margin 采样 + 下一轮复制式加权训练策略

该策略保持 `our_method_margin` 的采样流程不变，只修改“下一轮如何训练”：
1. 当前轮仍使用普通训练模型进行覆盖率估计、半径计算与 margin 打分采样；
2. 当本轮采样完成后，基于“当前已标注点的生效半径 + 新选点的生效预测半径”
   在全集上重新做一次覆盖归属；
3. 每个已标注点获得一个整数复制权重，语义为：
   - 初始值为 1（表示包含自己）；
   - 其余全集样本若落入多个已标注点半径内，则按 `distance / radius`
     最小的规则归属给某一个已标注点；
4. 下一轮训练开始时，不做 loss 加权，而是把已标注样本按该整数权重重复复制，
   打乱后按普通 DataLoader 训练。
"""
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from tqdm import tqdm

from get_radius import get_weight_from_indices_and_radii

from .our_method_margin import OurMethodMarginStrategy


class OurMethodMarginWeightedStrategy(OurMethodMarginStrategy):
    """
    `our_method_margin` 的复制式加权训练版本。

    该类的核心职责不是改变“当前轮如何选点”，而是在每轮选点结束后，为下一轮
    训练缓存一份“复制式训练配置”：
    - 采样阶段完全复用 `OurMethodMarginStrategy` 的行为；
    - 训练阶段则把权重解释为样本重复次数，而不是 loss 系数。

    因此它天然适合与你当前的实验目标对齐：
    采样逻辑保持可比，训练机制则显式放大覆盖更多区域的已标注点。
    """

    def __init__(
        self,
        epsilon: float = 0.01,
        h_min: float = 1e-6,
        diff_min: float = 0.0,
        spectral_norm_product: Optional[float] = 10.0,
        distance_cache_dir: str = "./distance_cache",
        query_budget: int = 100,
        alpha: Optional[float] = 0.1,
        inference_batch_size: int = 256,
        enable_early_radius_normalization: bool = False,
        early_radius_normalization_threshold: float = 0.2,
        early_radius_normalization_percentile_k: float = 20.0,
        local_lipschitz_k: int = 50,
        local_lipschitz_quantile: float = 0.90,
        local_lipschitz_distance_eps: float = 1e-12,
        local_lipschitz_min_value: float = 1e-12,
        max_theoretical_radius: Optional[float] = None,
        weighted_start_round: int = 2
    ) -> None:
        """
        初始化 margin + 复制式加权训练策略。

        参数保持与 `OurMethodMarginStrategy` 一致，目的是让实验入口、配置文件
        和现有脚本都可以直接复用，不需要额外为 weighted 版本拆一套配置字段。

        除父类状态外，本类额外维护一份“下一轮训练配置缓存”，其中记录：
        1. 该配置对应的下一轮已标注索引顺序；
        2. 与该顺序严格对齐的复制次数列表；
        3. 训练模式标记为 `replicated`，供主动学习主循环识别。
        当 `spectral_norm_product=None` 时，新增的 local_lipschitz_* 参数会透传给
        父类，用于按每个点的 KNN 邻域估计局部 logits 放大系数。
        当 `max_theoretical_radius` 为数值时，采样半径会被该手动最大半径截断。
        当 `alpha=None` 时，父类会在每轮候选集上用 `-log(margin)` 与 `logD`
        的 IQR 自动做尺度校准。
        `weighted_start_round` 控制“从第几轮训练开始启用复制式加权训练”：
        - `2` 表示保持当前默认行为，第 1 轮采样后就为第 2 轮训练准备复制权重；
        - `3` 表示第 1、2 轮训练都仍然使用普通训练，仅在第 2 轮采样结束后
          第一次计算复制权重，供第 3 轮训练使用；
        - 更一般地，前 `weighted_start_round - 2` 轮采样结束后都不会计算任何
          下一轮复制权重，从而避免无意义的额外计算。
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
        if weighted_start_round < 2:
            raise ValueError("weighted_start_round 必须大于等于 2。")
        self.name = "OurMethodMarginWeighted"
        self.weighted_start_round = int(weighted_start_round)
        self._next_round_training_setup: Optional[Dict[str, Any]] = None

    def get_training_setup(
        self,
        labeled_indices: Sequence[int]
    ) -> Optional[Dict[str, Any]]:
        """
        返回与当前已标注集合严格对齐的下一轮训练配置。

        主动学习主循环会在每轮训练开始前调用该方法。这里需要特别保证：
        1. 只有当主循环当前持有的 `labeled_indices` 与策略上轮缓存的
           `expected_labeled_indices` 完全一致时，才返回复制式训练配置；
        2. 若顺序或内容不一致，则说明缓存已经失效或尚未准备好，必须返回 None，
           避免把上一轮的权重错误套到新的标注集上；
        3. 返回值中的 `repeat_counts` 与 `labeled_indices` 一一对齐，可直接用于
           主循环展开重复索引。

        参数:
            labeled_indices: 当前轮训练前的已标注样本索引序列。

        返回:
            Optional[Dict[str, Any]]:
                - 若缓存命中，返回复制式训练配置字典；
                - 若缓存为空或与当前标注集不匹配，返回 None。
        """
        if self._next_round_training_setup is None:
            return None

        expected_indices = self._next_round_training_setup.get("expected_labeled_indices")
        if expected_indices != list(labeled_indices):
            return None

        return self._next_round_training_setup

    def _should_prepare_next_round_training_setup(self) -> bool:
        """
        判断当前采样轮结束后，是否需要为下一轮训练计算并缓存复制权重。

        这个判断函数只负责回答“现在该不该算下一轮的 `repeat_counts`”，不参与
        权重本身的数学定义。其设计目的是把“轮次门控逻辑”集中在一个地方，避免
        第一轮与后续轮次各自散落重复判断。

        本策略中的时间关系如下：
        1. 当前 `select_samples(...)` 执行时，`self.round_counter + 1` 表示当前
           正在进行的采样轮次；
        2. 当前采样结束后，若要影响“下一轮训练”，则对应的训练轮次编号是
           `current_selection_round + 1`；
        3. 只有当这个“下一轮训练编号”大于等于 `weighted_start_round` 时，才应
           该开始计算和缓存复制权重。

        例如：
        - `weighted_start_round=2` 时，第 1 轮采样结束后就应开始准备复制权重；
        - `weighted_start_round=3` 时，第 1 轮采样结束后不应准备，第 2 轮采样
          结束后才应首次准备。

        返回:
            bool:
                - `True` 表示当前采样结束后应为下一轮训练准备复制权重；
                - `False` 表示下一轮训练仍应保持普通训练，同时应清空任何旧缓存，
                  防止误用上一轮遗留的复制配置。
        """
        current_selection_round = self.round_counter + 1
        next_training_round = current_selection_round + 1
        return next_training_round >= self.weighted_start_round

    def _cache_next_round_training_setup(
        self,
        next_labeled_indices: Sequence[int],
        repeat_counts: Sequence[int]
    ) -> None:
        """
        缓存下一轮训练所需的复制式训练配置。

        该函数只负责“写缓存”，不负责计算权重本身。它把下一轮训练所需的最小信息
        固化下来，供主循环在训练阶段读取：
        - `mode='replicated'`：表示训练应按复制索引展开，而非做 loss 加权；
        - `repeat_counts`：与下一轮 `labeled_indices` 严格对齐的复制次数；
        - `expected_labeled_indices`：用于校验主循环读取时的标注集是否一致。

        参数:
            next_labeled_indices: 下一轮训练时将使用的已标注索引序列。
            repeat_counts: 与 `next_labeled_indices` 对齐的整数复制次数列表。

        返回:
            None

        异常:
            ValueError: 当索引数与复制次数数目不一致时抛出。
        """
        next_labeled_indices = list(next_labeled_indices)
        repeat_counts = [int(count) for count in repeat_counts]
        if len(next_labeled_indices) != len(repeat_counts):
            raise ValueError(
                "next_labeled_indices 与 repeat_counts 长度不一致: "
                f"{len(next_labeled_indices)} vs {len(repeat_counts)}"
            )

        self._next_round_training_setup = {
            "mode": "replicated",
            "repeat_counts": repeat_counts,
            "expected_labeled_indices": next_labeled_indices,
            "state_dict": None,
        }

    def _build_unlabeled_placeholder_points(
        self,
        unlabeled_indices: Sequence[int]
    ) -> List[Dict[str, Any]]:
        """
        为 `get_weight(..., distance_matrix=...)` 构造占位无标注点列表。

        `get_weight` 在走距离矩阵优化路径时，只要求：
        1. `unlabeled_points` 的长度与 `unlabeled_indices` 一致；
        2. `labeled_points` / `unlabeled_points` 与索引映射长度一致。

        在这种情况下，函数内部不会再访问无标注点的真实 `data` 内容，因此这里
        只需要构造最小占位结构，避免为了权重归属重复从数据集读取整份样本。

        参数:
            unlabeled_indices: 将参与“覆盖归属”的全集其余样本索引。

        返回:
            List[Dict[str, Any]]: 与 `unlabeled_indices` 一一对应的占位字典列表。
        """
        return [{"data": None} for _ in unlabeled_indices]

    def _compute_repeat_counts_for_next_round(
        self,
        dataset: Any,
        distance_matrix: np.ndarray,
        current_labeled_indices: Sequence[int],
        current_labeled_radii: Sequence[float],
        selected_indices: Sequence[int],
        selected_radii: Sequence[float]
    ) -> List[int]:
        """
        计算下一轮训练所需的复制次数列表。

        计算规则完全对应你确认过的权重定义：
        1. 下一轮的已标注集合由“当前已标注点 + 本轮新选点”构成；
        2. 当前已标注点使用本轮真实生效半径；
        3. 新选点使用本轮采样阶段得到的生效预测半径；
        4. 在“完整训练集”上重新做覆盖归属；
        5. 若某个样本同时落在多个已标注点半径内，则按 `distance / radius`
           最小的规则归属；
        6. 每个已标注点基础权重为 1，表示“至少包含自己”。

        注意：
        这里的“完整训练集覆盖归属”实现采用 `get_weight(...)` 的既有语义：
        已标注点自身通过初始权重 1 计入，其余全集样本通过归属规则分配给某一个
        已标注点，因此最终返回的结果天然适合作为整数复制次数。

        参数:
            dataset: 完整训练集，用于构造下一轮已标注点列表。
            distance_matrix: 全训练集距离矩阵。
            current_labeled_indices: 当前轮采样前的已标注索引列表。
            current_labeled_radii: 当前已标注点的生效半径列表。
            selected_indices: 本轮新选中的样本索引列表。
            selected_radii: 本轮新选中样本的生效预测半径列表。

        返回:
            List[int]: 与“下一轮已标注索引顺序”严格对齐的整数复制次数。

        异常:
            ValueError: 当索引与半径列表长度不一致时抛出。
        """
        if len(current_labeled_indices) != len(current_labeled_radii):
            raise ValueError(
                "current_labeled_indices 与 current_labeled_radii 长度不一致: "
                f"{len(current_labeled_indices)} vs {len(current_labeled_radii)}"
            )
        if len(selected_indices) != len(selected_radii):
            raise ValueError(
                "selected_indices 与 selected_radii 长度不一致: "
                f"{len(selected_indices)} vs {len(selected_radii)}"
            )
        _ = dataset

        next_labeled_indices = list(current_labeled_indices) + list(selected_indices)
        next_labeled_radii = list(current_labeled_radii) + list(selected_radii)

        if not next_labeled_indices:
            return []

        distance_backend = (
            self._cuda_distance_matrix
            if self._cuda_distance_matrix is not None else distance_matrix
        )
        repeat_counts = get_weight_from_indices_and_radii(
            distance_matrix=distance_backend,
            labeled_indices=list(next_labeled_indices),
            radii=list(next_labeled_radii)
        )
        return [int(count) for count in repeat_counts]

    def _prepare_next_round_training_setup(
        self,
        dataset: Any,
        distance_matrix: np.ndarray,
        current_labeled_indices: Sequence[int],
        current_labeled_radii: Sequence[float],
        selected_indices: Sequence[int],
        selected_radii: Sequence[float]
    ) -> None:
        """
        根据当前轮采样结果，为下一轮训练生成并缓存复制式训练配置。

        这是本策略区别于 `our_method_margin` 的核心步骤。它把“本轮采样结果”
        转换成“下一轮该如何训练”的显式配置，具体包括：
        1. 组合下一轮的完整已标注集合；
        2. 依据已确认的覆盖归属规则计算每个已标注点的整数复制次数；
        3. 打印当前最大复制权重，便于调试与实验日志分析；
        4. 把结果缓存到策略内部，供下一轮训练开始时读取。

        参数:
            dataset: 完整训练集。
            distance_matrix: 全训练集距离矩阵。
            current_labeled_indices: 当前轮采样前的已标注索引列表。
            current_labeled_radii: 当前轮已标注点的生效半径列表。
            selected_indices: 本轮新选样本索引列表。
            selected_radii: 本轮新选样本的生效预测半径列表。

        返回:
            None
        """
        if not self._should_prepare_next_round_training_setup():
            self._next_round_training_setup = None
            print(
                "[OurMethodMarginWeighted] 跳过下一轮复制权重计算: "
                f"current_selection_round={self.round_counter + 1}, "
                f"next_training_round={self.round_counter + 2}, "
                f"weighted_start_round={self.weighted_start_round}"
            )
            return

        next_labeled_indices = list(current_labeled_indices) + list(selected_indices)
        repeat_counts = self._compute_repeat_counts_for_next_round(
            dataset=dataset,
            distance_matrix=distance_matrix,
            current_labeled_indices=current_labeled_indices,
            current_labeled_radii=current_labeled_radii,
            selected_indices=selected_indices,
            selected_radii=selected_radii
        )

        if repeat_counts:
            max_weight_position = int(np.argmax(repeat_counts))
            max_weight_index = int(next_labeled_indices[max_weight_position])
            max_weight_value = int(repeat_counts[max_weight_position])
            print(
                "[OurMethodMarginWeighted] 下一轮最大复制权重: "
                f"{max_weight_value} (idx: {max_weight_index})"
            )
        else:
            print("[OurMethodMarginWeighted] 下一轮复制权重为空。")

        self._cache_next_round_training_setup(
            next_labeled_indices=next_labeled_indices,
            repeat_counts=repeat_counts
        )

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
        执行第一轮采样，并额外为下一轮缓存复制式训练权重。

        第一轮采样逻辑与父类完全一致：
        - 候选点一律使用最大理论半径；
        - 按覆盖数做贪心选择直到达到预算。

        唯一新增的步骤是：
        在得到本轮新选点后，额外计算下一轮训练所需的复制次数。其中：
        - 旧已标注点使用本轮第一轮采样中生效的最大理论半径；
        - 新选点使用本轮第一轮采样中生效的最大理论半径。

        参数与返回值含义与父类同名方法保持一致。
        """
        if self._should_use_cuda_sampling(device):
            print(f"\n[OurMethodMarginWeighted] 第一轮采样")
            (
                selected_indices,
                current_labeled_radii,
                selected_radii,
            ) = self._first_round_selection_cuda_common(
                dataset=dataset,
                device=device,
                distance_matrix=distance_matrix,
                labeled_indices=labeled_indices,
                unlabeled_indices=unlabeled_indices,
                spectral_norm_product=spectral_norm_product,
                log_prefix="OurMethodMarginWeighted"
            )
            self._prepare_next_round_training_setup(
                dataset=dataset,
                distance_matrix=distance_matrix,
                current_labeled_indices=labeled_indices,
                current_labeled_radii=current_labeled_radii,
                selected_indices=selected_indices,
                selected_radii=selected_radii
            )
            return selected_indices

        _ = device
        print(f"\n[OurMethodMarginWeighted] 第一轮采样")

        round_indices = list(labeled_indices) + list(unlabeled_indices)
        fixed_round_radius: Optional[float] = None
        radius_by_global: Optional[np.ndarray] = None
        round_indices_array = np.asarray(round_indices, dtype=np.int64)
        if round_indices_array.size > 0:
            if self.max_theoretical_radius is not None:
                fixed_round_radius = float(self.max_theoretical_radius)
            elif spectral_norm_product is not None:
                fixed_round_radius = float(
                    self._compute_max_theoretical_radius(spectral_norm_product)
                )
            else:
                round_max_radii = self._compute_max_theoretical_radii_for_indices(
                    round_indices,
                    spectral_norm_product
                )
                radius_by_global = np.empty(distance_matrix.shape[0], dtype=np.float64)
                radius_by_global[round_indices_array] = round_max_radii

        if round_indices_array.size > 0:
            if fixed_round_radius is not None:
                round_radius_min = fixed_round_radius
                round_radius_max = fixed_round_radius
                round_radius_mean = fixed_round_radius
            else:
                round_radius_min = float(round_max_radii.min())
                round_radius_max = float(round_max_radii.max())
                round_radius_mean = float(round_max_radii.mean())
            print(
                "[OurMethodMarginWeighted] 最大理论半径统计: "
                f"min={round_radius_min:.6f}, "
                f"max={round_radius_max:.6f}, "
                f"mean={round_radius_mean:.6f}"
            )

        if fixed_round_radius is not None:
            current_labeled_radii_array = np.full(
                len(labeled_indices),
                fixed_round_radius,
                dtype=np.float64
            )
        elif labeled_indices:
            labeled_indices_array = np.asarray(labeled_indices, dtype=np.int64)
            current_labeled_radii_array = radius_by_global[labeled_indices_array]
        else:
            current_labeled_radii_array = np.empty((0,), dtype=np.float64)

        if labeled_indices:
            _, covered_mask = self._compute_coverage(
                distance_matrix=distance_matrix,
                labeled_indices=labeled_indices,
                radii=current_labeled_radii_array.tolist()
            )
            filtered_unlabeled = self._filter_unlabeled_indices(
                unlabeled_indices=unlabeled_indices,
                covered_mask=covered_mask
            )
            print(f"[OurMethodMarginWeighted] 原始无标注池大小: {len(unlabeled_indices)}")
            print(
                "[OurMethodMarginWeighted] 被已标注点覆盖的点数: "
                f"{len(unlabeled_indices) - len(filtered_unlabeled)}"
            )
            print(f"[OurMethodMarginWeighted] 过滤后无标注池大小: {len(filtered_unlabeled)}")
        else:
            filtered_unlabeled = list(unlabeled_indices)
            print(f"[OurMethodMarginWeighted] 无标注池大小: {len(filtered_unlabeled)}")

        selected_indices: List[int] = []
        if filtered_unlabeled:
            if fixed_round_radius is not None:
                radii = np.full(
                    len(filtered_unlabeled),
                    fixed_round_radius,
                    dtype=np.float64
                )
            else:
                filtered_unlabeled_array = np.asarray(
                    filtered_unlabeled,
                    dtype=np.int64
                )
                radii = radius_by_global[filtered_unlabeled_array]
            selected_indices, covered_indices = self._greedy_selection(
                unlabeled_indices=filtered_unlabeled,
                distance_matrix=distance_matrix,
                radii=radii,
                query_budget=self.query_budget
            )
            print(f"[OurMethodMarginWeighted] 第一轮选择的标注点数量: {len(selected_indices)}")
            print(f"[OurMethodMarginWeighted] 第一轮覆盖的点数量: {len(covered_indices)}")
        else:
            print("[OurMethodMarginWeighted] 无标注池为空，无需选择样本")

        current_labeled_radii = current_labeled_radii_array.astype(
            np.float64,
            copy=False
        ).tolist()
        if fixed_round_radius is not None:
            selected_radii = np.full(
                len(selected_indices),
                fixed_round_radius,
                dtype=np.float64
            ).tolist()
        elif selected_indices:
            selected_indices_array = np.asarray(selected_indices, dtype=np.int64)
            selected_radii = radius_by_global[selected_indices_array].astype(
                np.float64,
                copy=False
            ).tolist()
        else:
            selected_radii = []
        self._prepare_next_round_training_setup(
            dataset=dataset,
            distance_matrix=distance_matrix,
            current_labeled_indices=labeled_indices,
            current_labeled_radii=current_labeled_radii,
            selected_indices=selected_indices,
            selected_radii=selected_radii
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
        执行后续轮次采样，并缓存下一轮复制式训练权重。

        该实现与父类后续轮次流程保持一致：
        1. 计算当前已标注点真实半径；
        2. 按需要做早期半径归一化；
        3. 过滤已被覆盖的无标注样本；
        4. 为候选点预测概率并估计半径；
        5. 使用 margin 版本静态项 + `ln(D)` 做贪心采样。

        新增的部分仅有一项：
        每次最终选中一个候选点时，把它当时生效的预测半径同步记录下来；
        采样结束后，基于“旧点真实生效半径 + 新点生效预测半径”重新计算下一轮
        训练用的复制次数。
        """
        if self._should_use_cuda_sampling(device):
            print(f"\n[OurMethodMarginWeighted] 第{self.round_counter + 1}轮采样")
            (
                selected_indices,
                effective_labeled_radii,
                selected_radii,
            ) = self._subsequent_round_selection_cuda_common(
                model=model,
                dataset=dataset,
                device=device,
                distance_matrix=distance_matrix,
                labeled_indices=labeled_indices,
                unlabeled_indices=unlabeled_indices,
                spectral_norm_product=spectral_norm_product,
                log_prefix="OurMethodMarginWeighted"
            )
            self._prepare_next_round_training_setup(
                dataset=dataset,
                distance_matrix=distance_matrix,
                current_labeled_indices=labeled_indices,
                current_labeled_radii=effective_labeled_radii,
                selected_indices=selected_indices,
                selected_radii=selected_radii
            )
            return selected_indices

        _ = device
        print(f"\n[OurMethodMarginWeighted] 第{self.round_counter + 1}轮采样")

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
        print(f"[OurMethodMarginWeighted] 当前轮次覆盖率: {coverage_ratio:.6f}")

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
                    "[OurMethodMarginWeighted] 早期半径归一化已触发: "
                    f"T={self.early_radius_normalization_threshold:.6f}, "
                    f"k={self.early_radius_normalization_percentile_k:.2f}, "
                    f"q_k={normalization_pivot_radius:.6f}, "
                    f"r={normalization_scale_ratio:.6f}, "
                    f"max_radius={max_theoretical_radius:.6f}, "
                    f"normalized_coverage={coverage_ratio:.6f}"
                )
            else:
                print(
                    "[OurMethodMarginWeighted] 早期半径归一化未触发: "
                    f"coverage={coverage_ratio:.6f} >= "
                    f"T={self.early_radius_normalization_threshold:.6f}"
                )

        filtered_unlabeled = self._filter_unlabeled_indices(
            unlabeled_indices=unlabeled_indices,
            covered_mask=covered_mask
        )
        print(f"[OurMethodMarginWeighted] 原始无标注池大小: {len(unlabeled_indices)}")
        print(
            "[OurMethodMarginWeighted] 被已标注点覆盖的点数: "
            f"{len(unlabeled_indices) - len(filtered_unlabeled)}"
        )
        print(f"[OurMethodMarginWeighted] 过滤后无标注池大小: {len(filtered_unlabeled)}")

        if not filtered_unlabeled:
            print("[OurMethodMarginWeighted] 无标注池为空，无需选择样本")
            self._prepare_next_round_training_setup(
                dataset=dataset,
                distance_matrix=distance_matrix,
                current_labeled_indices=labeled_indices,
                current_labeled_radii=effective_labeled_radii.tolist(),
                selected_indices=[],
                selected_radii=[]
            )
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
            "[OurMethodMarginWeighted] 预估半径统计: "
            f"min={estimated_radii.min():.6f}, "
            f"max={estimated_radii.max():.6f}, "
            f"mean={estimated_radii.mean():.6f}"
        )

        selected_indices: List[int] = []
        selected_radii: List[float] = []
        candidate_indices = np.asarray(filtered_unlabeled, dtype=np.int64)
        candidate_radii = np.asarray(estimated_radii, dtype=np.float64)
        (
            covered_positions_by_candidate,
            reverse_cover_rows,
            initial_covered_counts,
        ) = self._prepare_candidate_cover_data_from_full_distance(
            distance_matrix=distance_matrix,
            candidate_indices=candidate_indices,
            candidate_radii=candidate_radii
        )
        initial_log_coverage_terms = np.log(np.maximum(initial_covered_counts, 1))
        static_score_terms = self._compute_static_score_terms(
            probabilities,
            coverage_ratio,
            coverage_terms=initial_log_coverage_terms
        )

        selected_local_positions = self._greedy_select_candidate_positions_from_cover_data(
            candidate_indices=candidate_indices,
            static_score_terms=static_score_terms,
            covered_positions_by_candidate=covered_positions_by_candidate,
            reverse_cover_rows=reverse_cover_rows,
            initial_covered_counts=initial_covered_counts,
            query_budget=self.query_budget,
            progress_desc=f"[OurMethodMarginWeighted] 第{self.round_counter + 1}轮选择进度",
            log_prefix="OurMethodMarginWeighted"
        )
        selected_indices = [
            int(candidate_indices[local_position])
            for local_position in selected_local_positions
        ]
        selected_radii = [
            float(candidate_radii[local_position])
            for local_position in selected_local_positions
        ]

        print(
            "[OurMethodMarginWeighted] "
            f"第{self.round_counter + 1}轮选择的标注点数量: {len(selected_indices)}"
        )
        self._prepare_next_round_training_setup(
            dataset=dataset,
            distance_matrix=distance_matrix,
            current_labeled_indices=labeled_indices,
            current_labeled_radii=effective_labeled_radii.tolist(),
            selected_indices=selected_indices,
            selected_radii=selected_radii
        )
        return selected_indices
