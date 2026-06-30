"""
Our Method - 基于半径覆盖的主动学习策略

该策略实现了一个基于半径覆盖的贪心采样方法：
1. 第一轮：使用最大理论半径进行贪心采样，直到无标注池为空
2. 后续轮次：使用预估半径进行贪心采样，直到无标注池为空或达到查询预算B
"""
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from typing import List, Any, Dict, Optional, Tuple
from tqdm import tqdm

from .base import ActiveLearningStrategy
from distance_calculator import DistanceCalculator
from get_radius import get_radius

from .local_expansion import (
    compute_estimated_radii_from_expansion,
    compute_knn_local_expansion_factors,
    compute_max_theoretical_radii_from_expansion,
    predict_logits_for_indices,
    resolve_expansion_factors_for_indices,
    softmax_numpy,
)


class OurMethodStrategy(ActiveLearningStrategy):
    """
    基于半径覆盖的主动学习策略
    
    该策略在第一轮使用最大理论半径进行贪心采样，
    在后续轮次使用预估半径进行贪心采样。
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
        初始化策略
        
        参数:
            epsilon: 半径方程的epsilon参数
            h_min: Hessian l2谱范数的下界，用于计算最大理论半径
            diff_min: diff的下界，用于防止diff过小导致半径计算不稳定
            spectral_norm_product: 预先给定的全局放大系数（默认10.0）；
                                若传入None，则使用每点KNN局部放大系数
            distance_cache_dir: 距离矩阵缓存目录
            query_budget: 查询预算B，每轮最多选择的样本数量
            alpha: 后续轮次采样分数中 coverage_ratio * ln|eta-p|_2
                这一项的权重系数（默认0.1）
            enable_early_radius_normalization: 是否启用早期半径归一化机制。
                关闭时保持当前策略行为不变；开启后仅在第二轮及之后生效。
            early_radius_normalization_threshold: 早期半径归一化的覆盖率阈值T。
                当当前轮次原始覆盖率小于该阈值时，触发半径放大。
            early_radius_normalization_percentile_k: 从大到小用于选取基准半径的
                百分位k，取值范围为 (0, 100]。例如 k=20 表示取前20%位置的半径
                作为放大基准。
            local_lipschitz_k: 当 `spectral_norm_product=None` 时，用于估计每个
                点局部 logits 放大系数的近邻数量。
            local_lipschitz_quantile: 当 `spectral_norm_product=None` 时，在每个点
                的 KNN finite-difference ratios 上取的分位数。
            local_lipschitz_distance_eps: 当 `spectral_norm_product=None` 时，用于
                距离分母稳定和过滤过近邻居的正数。
            local_lipschitz_min_value: 当 `spectral_norm_product=None` 时，对局部
                放大系数施加的纯数值下界，用于避免除零。
            max_theoretical_radius: 手动指定的最大理论半径。为 `None` 时保留当前
                公式计算方式；为正数时使用该数值作为所有点的最大半径上限。
        """
        super().__init__("OurMethod")
        if not 0.0 <= early_radius_normalization_threshold <= 1.0:
            raise ValueError(
                "early_radius_normalization_threshold 必须位于 [0, 1] 区间内"
            )
        if not 0.0 < early_radius_normalization_percentile_k <= 100.0:
            raise ValueError(
                "early_radius_normalization_percentile_k 必须位于 (0, 100] 区间内"
            )
        if local_lipschitz_k <= 0:
            raise ValueError("local_lipschitz_k 必须是正整数")
        if not 0.0 <= local_lipschitz_quantile <= 1.0:
            raise ValueError("local_lipschitz_quantile 必须位于 [0, 1] 区间内")
        if local_lipschitz_distance_eps <= 0.0:
            raise ValueError("local_lipschitz_distance_eps 必须是正数")
        if local_lipschitz_min_value <= 0.0:
            raise ValueError("local_lipschitz_min_value 必须是正数")
        if max_theoretical_radius is not None:
            max_theoretical_radius = float(max_theoretical_radius)
            if (
                not np.isfinite(max_theoretical_radius)
                or max_theoretical_radius <= 0.0
            ):
                raise ValueError("max_theoretical_radius 必须是有限正数或 None")

        self.epsilon = epsilon
        self.h_min = h_min
        self.diff_min = diff_min
        self.spectral_norm_product: Optional[float] = spectral_norm_product
        self.distance_cache_dir = distance_cache_dir
        self.query_budget = query_budget
        self.alpha = alpha
        self.enable_early_radius_normalization = enable_early_radius_normalization
        self.early_radius_normalization_threshold = early_radius_normalization_threshold
        self.early_radius_normalization_percentile_k = (
            early_radius_normalization_percentile_k
        )
        self.local_lipschitz_k = int(local_lipschitz_k)
        self.local_lipschitz_quantile = float(local_lipschitz_quantile)
        self.local_lipschitz_distance_eps = float(local_lipschitz_distance_eps)
        self.local_lipschitz_min_value = float(local_lipschitz_min_value)
        self.max_theoretical_radius = max_theoretical_radius
        
        # 数据集元数据（用于定位距离矩阵缓存）
        self.dataset_name: Optional[str] = None
        self.distance_matrix_suffix: str = ""
        
        # 缓存的距离矩阵
        self.distance_matrix: Optional[np.ndarray] = None
        
        # 轮次计数器
        self.round_counter: int = 0
        
        # 保存覆盖率供后续使用
        self.last_coverage: Optional[float] = None

        # 软标签状态：默认关闭，只有外部显式注入全量软标签矩阵后才启用。
        self.use_soft_labels: bool = False
        self.soft_labels: Optional[np.ndarray] = None

        # 局部放大系数缓存：仅当 spectral_norm_product=None 时，在每轮采样前刷新。
        self._current_logits_by_index: Optional[np.ndarray] = None
        self._current_local_expansion_factors: Optional[np.ndarray] = None

    def set_dataset_metadata(
        self,
        dataset_name: str,
    ) -> None:
        """
        设置数据集元数据
        
        参数:
            dataset_name: 数据集名称
        """
        self.dataset_name = dataset_name
        self.distance_matrix_suffix = ""

    def set_soft_labels(self, soft_labels: np.ndarray) -> None:
        """
        设置与完整数据集索引对齐的软标签矩阵，并启用软标签半径计算。

        该方法用于把外部预计算好的软标签注入当前策略。注入后：
        1. `_build_labeled_points(...)` 会在每个已标注样本字典中附加
           `soft_label` 字段；
        2. 后续轮次在调用 `get_radius(...)` 计算已标注点半径时，会显式使用
           `use_soft_label=True`；
        3. 软标签矩阵必须与完整训练池索引严格对齐，保证 `soft_labels[i]`
           对应第 `i` 个样本。

        参数:
            soft_labels (np.ndarray): 全量样本的软标签矩阵，形状通常为
                `(N, num_classes)`。

        返回:
            None

        异常:
            ValueError: 当 `soft_labels` 为空时抛出。
        """
        if soft_labels is None:
            raise ValueError("soft_labels 不能为空")
        if isinstance(soft_labels, torch.Tensor):
            soft_labels = soft_labels.detach().cpu().numpy()
        self.soft_labels = np.asarray(soft_labels)
        self.use_soft_labels = True

    def _get_soft_label_for_index(self, index: int) -> np.ndarray:
        """
        根据全局样本索引返回对应的软标签向量。

        该辅助函数把“索引对齐”的约束集中在一个位置处理，避免多个半径相关
        代码路径直接操作 `self.soft_labels`：
        1. 外部只需要保证传入 `set_soft_labels(...)` 的矩阵与完整样本池对齐；
        2. 策略内部统一通过该函数读取；
        3. 一旦软标签未设置却误开启软标签模式，会立即抛出异常，避免静默退化为
           错误的硬标签半径。

        参数:
            index (int): 完整数据集中的样本索引。

        返回:
            np.ndarray: 与该索引对应的软标签概率向量。

        异常:
            RuntimeError: 当软标签尚未设置时抛出。
        """
        if not self.use_soft_labels or self.soft_labels is None:
            raise RuntimeError("软标签未设置，但尝试获取软标签。")
        return self.soft_labels[index]

    def _resolve_spectral_norm_product(
        self,
        model: torch.nn.Module
    ) -> Optional[float]:
        """
        解析当前轮次是否使用固定全局放大系数。

        当 `self.spectral_norm_product` 是数值时，本函数沿用旧逻辑，返回该固定
        全局系数并做有限正数校验；当其为 `None` 时，不再调用全局谱范数乘积
        上界，而是返回 `None`，表示后续半径计算应使用每个点自己的 KNN 局部
        放大系数。局部系数会在 `select_samples(...)` 中按当前模型单独准备。
        
        参数:
            model: 当前轮次用于训练/采样的模型
            
        返回:
            Optional[float]: 固定全局放大系数；若为 `None` 则表示使用逐点局部
            放大系数。
        """
        if self.spectral_norm_product is None:
            return None

        _ = model
        spectral_norm_product = float(self.spectral_norm_product)
        
        if not np.isfinite(spectral_norm_product) or spectral_norm_product <= 0.0:
            raise ValueError(
                f"spectral_norm_product 必须是有限正数，当前值: {spectral_norm_product}"
            )
        return spectral_norm_product

    def _prepare_local_expansion_context(
        self,
        model: torch.nn.Module,
        dataset: Any,
        distance_matrix: np.ndarray,
        device: torch.device
    ) -> None:
        """
        为当前采样轮次准备每个样本的 KNN 局部放大系数。

        该函数只在 `spectral_norm_product=None` 时调用。它先对完整训练池做一次
        batched forward，得到与距离矩阵行号对齐的 logits；随后对每个样本基于
        `local_lipschitz_k` 个近邻计算 finite-difference ratios，并取
        `local_lipschitz_quantile` 分位数作为该样本自己的 `L_i`。缓存结果只在
        当前轮使用，每次 `select_samples(...)` 都会重新计算，以反映训练后模型
        参数变化。

        参数:
            model: 当前轮训练完成后的模型。
            dataset: 完整训练集对象。
            distance_matrix: 与完整训练集索引对齐的距离矩阵。
            device: 当前计算设备。

        返回:
            None

        异常:
            RuntimeError: 当数据集大小与距离矩阵行数不一致时抛出。
        """
        if len(dataset) != distance_matrix.shape[0]:
            raise RuntimeError(
                "数据集大小与距离矩阵行数不一致，无法估计局部放大系数: "
                f"{len(dataset)} vs {distance_matrix.shape[0]}"
            )

        all_indices = list(range(distance_matrix.shape[0]))
        self._current_logits_by_index = predict_logits_for_indices(
            model=model,
            dataset=dataset,
            indices=all_indices,
            device=device,
            batch_size=256
        )
        self._current_local_expansion_factors = compute_knn_local_expansion_factors(
            query_indices=all_indices,
            logits_by_index=self._current_logits_by_index,
            distance_matrix=distance_matrix,
            knn_k=self.local_lipschitz_k,
            quantile=self.local_lipschitz_quantile,
            distance_eps=self.local_lipschitz_distance_eps,
            min_factor=self.local_lipschitz_min_value
        )
        print(
            f"[{self.name}] 使用KNN局部放大系数: "
            f"k={self.local_lipschitz_k}, "
            f"q={self.local_lipschitz_quantile:.2f}, "
            f"min={self._current_local_expansion_factors.min():.6f}, "
            f"max={self._current_local_expansion_factors.max():.6f}, "
            f"mean={self._current_local_expansion_factors.mean():.6f}"
        )

    def _clear_local_expansion_context(self) -> None:
        """
        清空当前轮缓存的局部 logits 与放大系数。

        固定全局放大系数模式不需要保存全量 logits，也不需要逐点 `L_i` 数组。
        每轮进入该模式前主动清空缓存，可以避免上一轮局部模式的状态被误用，
        同时释放不必要的内存引用。

        返回:
            None
        """
        self._current_logits_by_index = None
        self._current_local_expansion_factors = None

    def _get_expansion_factors_for_indices(
        self,
        indices: List[int],
        spectral_norm_product: Optional[float]
    ) -> np.ndarray:
        """
        获取一组样本当前应使用的放大系数。

        该方法把“固定全局系数”和“KNN 局部逐点系数”统一成数组输出：若
        `spectral_norm_product` 是数值，则返回同值数组；若为 `None`，则从当前轮
        已缓存的 `self._current_local_expansion_factors` 中按全局索引取出每个点的
        `L_i`。后续半径计算只依赖该数组，不需要关心当前处于哪种模式。

        参数:
            indices: 需要解析放大系数的全局样本索引列表。
            spectral_norm_product: 固定全局放大系数；`None` 表示使用局部系数。

        返回:
            np.ndarray: 与 `indices` 顺序一致的一维放大系数数组。
        """
        return resolve_expansion_factors_for_indices(
            indices=indices,
            fixed_expansion_factor=spectral_norm_product,
            local_expansion_factors=self._current_local_expansion_factors
        )

    def _compute_max_theoretical_radii_for_indices(
        self,
        indices: List[int],
        spectral_norm_product: Optional[float]
    ) -> np.ndarray:
        """
        为一组样本计算逐点最大理论半径。

        固定全局模式下，返回数组中的每个元素都等于旧公式的最大理论半径；局部
        模式下，数组中第 `i` 个元素使用对应样本的 KNN 局部放大系数 `L_i`。该函数
        不额外施加局部半径上界，只做半径公式里的放大系数替换。

        参数:
            indices: 需要计算半径的全局样本索引列表。
            spectral_norm_product: 固定全局放大系数；`None` 表示使用局部系数。

        返回:
            np.ndarray: 与 `indices` 顺序一致的最大理论半径数组。
        """
        if self.max_theoretical_radius is not None:
            return np.full(
                (len(indices),),
                self.max_theoretical_radius,
                dtype=np.float64
            )

        expansion_factors = self._get_expansion_factors_for_indices(
            indices=indices,
            spectral_norm_product=spectral_norm_product
        )
        return compute_max_theoretical_radii_from_expansion(
            expansion_factors=expansion_factors,
            epsilon=self.epsilon,
            h_min=self.h_min,
            diff_min=self.diff_min
        )

    def _compute_round_max_theoretical_radius(
        self,
        indices: List[int],
        spectral_norm_product: Optional[float]
    ) -> float:
        """
        为当前轮早期半径归一化解析一个兼容旧逻辑的最大理论半径。

        早期半径归一化原本只接受一个标量 `max_theoretical_radius` 作为统一截断值。
        在局部放大系数模式下，每个点都有自己的理论半径，因此这里取当前相关样本
        的逐点最大理论半径中的最大值，用于保持原归一化逻辑可运行。该值只服务于
        已存在的早期归一化机制；若该机制关闭，则不会改变未归一化半径。

        参数:
            indices: 当前轮相关的全局样本索引列表。
            spectral_norm_product: 固定全局放大系数；`None` 表示使用局部系数。

        返回:
            float: 当前轮用于早期归一化的标量理论半径。
        """
        if not indices:
            if self.max_theoretical_radius is not None:
                return float(self.max_theoretical_radius)
            return float(self._compute_max_theoretical_radius(1.0))
        radii = self._compute_max_theoretical_radii_for_indices(
            indices=indices,
            spectral_norm_product=spectral_norm_product
        )
        return float(np.max(radii))

    def _clip_radii_to_max_theoretical(
        self,
        radii: np.ndarray,
        indices: List[int],
        spectral_norm_product: Optional[float]
    ) -> np.ndarray:
        """
        用当前最大理论半径上限截断一组样本半径。

        最大半径可以来自手动配置 `max_theoretical_radius`，也可以在其为 `None` 时
        由现有公式按点计算。该函数保证半径计算出口统一执行
        `min(radius_i, max_radius_i)`，避免后续早期归一化或候选半径估计产生超过
        最大理论半径的值。

        参数:
            radii: 与 `indices` 顺序一致的半径数组。
            indices: 半径对应的全局样本索引列表。
            spectral_norm_product: 固定全局放大系数；`None` 表示使用局部系数。

        返回:
            np.ndarray: 与输入顺序一致、已按最大理论半径截断的半径数组。

        异常:
            ValueError: 当 `radii` 与 `indices` 长度不一致时抛出。
        """
        radii = np.asarray(radii, dtype=np.float64)
        if radii.shape[0] != len(indices):
            raise ValueError("radii 的长度必须与 indices 一致。")
        max_radii = self._compute_max_theoretical_radii_for_indices(
            indices=indices,
            spectral_norm_product=spectral_norm_product
        )
        return np.minimum(radii, max_radii)

    def _compute_labeled_radii(
        self,
        labeled_points: List[Dict[str, Any]],
        labeled_indices: List[int],
        model: torch.nn.Module,
        spectral_norm_product: Optional[float]
    ) -> np.ndarray:
        """
        计算已标注点半径，并为每个点使用对应的放大系数。

        该函数保留 `get_radius(...)` 中关于标签、软标签、预测概率、Hessian 谱范数
        与 diff 的原始实现，只在传入 `spectral_norm_product` 时做逐点解析：固定
        模式下每个点传入同一个全局值；局部模式下第 `i` 个已标注点传入自己的
        `L_i`。这样可以最小化对原半径公式的改动。

        参数:
            labeled_points: 与 `labeled_indices` 顺序一致的已标注点字典列表。
            labeled_indices: 已标注点的全局样本索引列表。
            model: 当前轮训练完成后的模型。
            spectral_norm_product: 固定全局放大系数；`None` 表示使用局部系数。

        返回:
            np.ndarray: 与 `labeled_indices` 顺序一致的已标注点半径数组。

        异常:
            ValueError: 当点列表与索引列表长度不一致时抛出。
        """
        if len(labeled_points) != len(labeled_indices):
            raise ValueError("labeled_points 与 labeled_indices 长度必须一致。")

        expansion_factors = self._get_expansion_factors_for_indices(
            indices=labeled_indices,
            spectral_norm_product=spectral_norm_product
        )
        raw_radii = np.array([
            get_radius(
                labeled_point=point,
                model=model,
                epsilon=self.epsilon,
                h_min=self.h_min,
                use_soft_label=self.use_soft_labels,
                spectral_norm_product=float(expansion_factor),
                diff_min=self.diff_min
            )
            for point, expansion_factor in zip(labeled_points, expansion_factors)
        ], dtype=np.float64)
        return self._clip_radii_to_max_theoretical(
            radii=raw_radii,
            indices=labeled_indices,
            spectral_norm_product=spectral_norm_product
        )

    def _estimate_radii_for_unlabeled(
        self,
        h_values: np.ndarray,
        unlabeled_indices: List[int],
        spectral_norm_product: Optional[float]
    ) -> np.ndarray:
        """
        批量预估无标注候选点半径，并使用每个点对应的放大系数。

        固定全局模式下，该函数与旧逻辑等价，只是把逐点循环改成向量化公式；局部
        模式下，候选点 `x_i` 使用自己的 KNN 局部放大系数 `L_i`，从而让半径反映
        该点附近 logits 对输入距离的实际经验放大程度。

        参数:
            h_values: 与 `unlabeled_indices` 顺序一致的 H 最大特征值数组。
            unlabeled_indices: 无标注候选点的全局样本索引列表。
            spectral_norm_product: 固定全局放大系数；`None` 表示使用局部系数。

        返回:
            np.ndarray: 与 `unlabeled_indices` 顺序一致的预估半径数组。
        """
        expansion_factors = self._get_expansion_factors_for_indices(
            indices=unlabeled_indices,
            spectral_norm_product=spectral_norm_product
        )
        estimated_radii = compute_estimated_radii_from_expansion(
            h_values=h_values,
            expansion_factors=expansion_factors,
            epsilon=self.epsilon,
            diff_min=self.diff_min
        )
        return self._clip_radii_to_max_theoretical(
            radii=estimated_radii,
            indices=unlabeled_indices,
            spectral_norm_product=spectral_norm_product
        )

    def _get_probabilities_for_indices(
        self,
        model: torch.nn.Module,
        dataset: Any,
        indices: List[int],
        device: torch.device
    ) -> np.ndarray:
        """
        获取指定样本的预测概率，并在局部模式下复用已缓存 logits。

        当本轮启用 KNN 局部放大系数时，`_prepare_local_expansion_context(...)`
        已经对完整训练池做过一次 forward，因此这里可以直接从缓存 logits 中取出
        对应行并做 softmax，避免重复模型前向。固定全局模式没有该缓存，则回退到
        原有的 `_predict_probabilities(...)`。

        参数:
            model: 当前轮训练完成后的模型。
            dataset: 完整训练集对象。
            indices: 需要预测概率的全局样本索引列表。
            device: 当前计算设备。

        返回:
            np.ndarray: 与 `indices` 顺序一致的概率矩阵。
        """
        if self._current_logits_by_index is not None:
            return softmax_numpy(self._current_logits_by_index[indices])
        return self._predict_probabilities(model, dataset, indices, device)

    def _compute_max_theoretical_radius(
        self,
        spectral_norm_product: float
    ) -> float:
        """
        计算最大理论半径
        
        公式: 2*epsilon/(spectral_norm_product*(diff_min + sqrt(diff_min^2 + 2*h_min*epsilon)))
        
        参数:
            spectral_norm_product: 固定全局放大系数。
            
        返回:
            float: 最大理论半径
        """
        sqrt_term = np.sqrt(self.diff_min**2 + 2.0 * self.h_min * self.epsilon)
        max_radius = 2.0 * self.epsilon / (
            spectral_norm_product * (self.diff_min + sqrt_term)
        )
        return max_radius

    def _normalize_radii_for_early_rounds(
        self,
        radii: np.ndarray,
        coverage_ratio: float,
        max_theoretical_radius: float
    ) -> Tuple[np.ndarray, bool, float, float]:
        """
        根据早期半径归一化配置，决定当前轮次是否需要放大半径。

        该函数只服务于第二轮及之后的“覆盖率不足”场景，输入的 `radii`
        应当是当前轮次已标注点的原始半径。处理规则如下：
        1. 若机制关闭，或当前原始覆盖率 `coverage_ratio` 已经达到阈值 `T`，
           则直接返回原始半径，不做任何修改；
        2. 若 `coverage_ratio < T`，则把半径按从大到小排序，定位到第百分之
           `k` 大的基准半径 `q_k`；
        3. 使用 `r = max_theoretical_radius / q_k` 作为统一放大比例；
        4. 所有半径都按 `min(radius * r, max_theoretical_radius)` 放大，使得：
           - 不小于 `q_k` 的半径会被截断到最大理论半径；
           - 小于 `q_k` 的半径按同一比例同步放大；
        5. 返回放大后的半径数组，以及是否触发、放大比例和基准半径，供后续
           的无标注池过滤和预测半径缩放复用。

        参数:
            radii (np.ndarray): 当前轮次已标注点的原始半径数组。
            coverage_ratio (float): 使用原始已标注点半径计算得到的当前覆盖率。
            max_theoretical_radius (float): 当前模型与参数下可使用的最大理论半径。

        返回:
            Tuple[np.ndarray, bool, float, float]:
                - normalized_radii: 归一化后的半径数组；若未触发则等于原始半径；
                - normalization_applied: 是否实际触发了早期半径归一化；
                - scale_ratio: 使用的统一放大比例 `r`；未触发时为 1.0；
                - pivot_radius: 第百分之 `k` 大的基准半径 `q_k`；未触发时返回
                  原始半径中的最大值（若半径数组为空则返回 0.0）。

        异常:
            ValueError:
                - 当 `radii` 不是一维数组时抛出；
                - 当 `max_theoretical_radius` 不是有限正数时抛出。
        """
        radii = np.asarray(radii, dtype=np.float64)
        if radii.ndim != 1:
            raise ValueError("radii 必须是一维数组。")
        if not np.isfinite(max_theoretical_radius) or max_theoretical_radius <= 0.0:
            raise ValueError("max_theoretical_radius 必须是有限正数。")
        if radii.size == 0:
            return radii.copy(), False, 1.0, 0.0
        if (
            not self.enable_early_radius_normalization
            or coverage_ratio >= self.early_radius_normalization_threshold
        ):
            return radii.copy(), False, 1.0, float(np.max(radii))

        sorted_desc = np.sort(radii)[::-1]
        percentile_index = int(
            np.ceil(
                sorted_desc.size
                * (self.early_radius_normalization_percentile_k / 100.0)
            )
        ) - 1
        percentile_index = min(max(percentile_index, 0), sorted_desc.size - 1)
        pivot_radius = float(max(sorted_desc[percentile_index], 1e-12))
        scale_ratio = float(max(1.0, max_theoretical_radius / pivot_radius))
        normalized_radii = np.minimum(
            radii * scale_ratio,
            max_theoretical_radius,
        )
        return normalized_radii, True, scale_ratio, pivot_radius

    def _ensure_distance_matrix(self, split: str = "train") -> np.ndarray:
        """
        加载并缓存距离矩阵
        
        参数:
            split: 数据集划分名称（默认为"train"）
            
        返回:
            np.ndarray: 加载的距离矩阵
        """
        if self.distance_matrix is not None:
            return self.distance_matrix
        
        if self.dataset_name is None:
            raise RuntimeError("数据集元数据必须先设置才能加载距离矩阵")
        
        calculator = DistanceCalculator(
            cache_dir=self.distance_cache_dir,
            dataset_name=self.dataset_name
        )
        distance_matrix = calculator.load_distance_matrix(
            split=split,
            suffix=self.distance_matrix_suffix
        )
        if distance_matrix is None:
            raise RuntimeError(
                "距离矩阵未找到。请先计算并缓存到distance_cache目录中"
            )
        
        self.distance_matrix = distance_matrix
        return distance_matrix

    def _build_labeled_points(
        self,
        dataset: Any,
        labeled_indices: List[int]
    ) -> List[Dict[str, Any]]:
        """
        构建已标注点列表
        
        参数:
            dataset: 完整训练数据集
            labeled_indices: 已标注样本的索引列表
            
        返回:
            List[Dict]: 已标注点列表，每个点至少包含 `data` 与 `label`；
            当启用软标签模式时，还会包含 `soft_label`。
        """
        labeled_points = []
        for idx in labeled_indices:
            data, label = dataset[idx]
            if isinstance(label, torch.Tensor):
                label_value = int(label.item())
            else:
                label_value = int(label)
            labeled_point: Dict[str, Any] = {"data": data, "label": label_value}
            if self.use_soft_labels:
                labeled_point["soft_label"] = self._get_soft_label_for_index(idx)
            labeled_points.append(labeled_point)
        return labeled_points

    def _compute_coverage(
        self,
        distance_matrix: np.ndarray,
        labeled_indices: List[int],
        radii: List[float]
    ) -> Tuple[float, np.ndarray]:
        """
        计算覆盖率
        
        覆盖率定义为：所有被已标注点覆盖的点（包括自己）数量/所有数据数量
        
        参数:
            distance_matrix: 距离矩阵
            labeled_indices: 已标注点的索引列表
            radii: 已标注点的半径列表
            
        返回:
            Tuple[float, np.ndarray]: (覆盖率, 覆盖掩码)
        """
        n_total = distance_matrix.shape[0]
        covered_mask = np.zeros(n_total, dtype=bool)
        
        for idx, radius in zip(labeled_indices, radii):
            distances = distance_matrix[idx]
            covered_mask |= distances <= radius
        
        coverage_ratio = float(covered_mask.sum()) / float(n_total) if n_total > 0 else 0.0
        return coverage_ratio, covered_mask

    def _predict_probabilities(
        self,
        model: torch.nn.Module,
        dataset: Any,
        indices: List[int],
        device: torch.device
    ) -> np.ndarray:
        """
        预测给定索引样本的概率向量
        
        参数:
            model: 训练好的模型
            dataset: 数据集
            indices: 样本索引列表
            device: 计算设备
            
        返回:
            np.ndarray: 概率向量矩阵，形状为(len(indices), num_classes)
        """
        model.eval()
        probabilities = []
        
        with torch.no_grad():
            for idx in indices:
                data, _ = dataset[idx]
                if not isinstance(data, torch.Tensor):
                    data = torch.tensor(data)
                if data.dim() == 3:  # 图像数据
                    data = data.unsqueeze(0)
                elif data.dim() == 1:  # 向量数据
                    data = data.unsqueeze(0)
                
                data = data.to(device)
                logits = model(data)
                probs = torch.softmax(logits, dim=1).squeeze(0)
                probabilities.append(probs.cpu().numpy())
        
        return np.array(probabilities)

    def _estimate_radius_for_unlabeled(
        self,
        h: float,
        spectral_norm_product: Optional[float]
    ) -> float:
        """
        预估无标注点的半径
        
        公式: 2*epsilon/(spectral_norm_product*(diff_min + sqrt(diff_min^2 + 2*h*epsilon)))
        
        参数:
            h: H的最大特征值
            spectral_norm_product: 固定全局放大系数；None 表示使用逐点局部放大系数。
            
        返回:
            float: 预估的半径
        """
        sqrt_term = np.sqrt(self.diff_min**2 + 2.0 * h * self.epsilon)
        estimated_radius = 2.0 * self.epsilon / (
            spectral_norm_product * (self.diff_min + sqrt_term)
        )
        return estimated_radius

    def _compute_eta_p_distance(
        self,
        probability: np.ndarray
    ) -> float:
        """
        计算eta和p的L2距离 |eta-p|_2
        
        参数:
            probability: 模型输出的概率向量p
            
        返回:
            float: |eta-p|_2的L2距离
        """
        # eta是p的最大分量对应的one-hot向量
        max_index = np.argmax(probability)
        eta = np.zeros_like(probability)
        eta[max_index] = 1.0
        
        # 计算L2距离
        distance = np.linalg.norm(eta - probability)
        return distance

    def _compute_D(
        self,
        idx: int,
        unlabeled_indices: List[int],
        distance_matrix: np.ndarray,
        radius: float
    ) -> int:
        """
        计算D(x): 无标注池中被点x以预估半径覆盖的点的数量（包括自己）
        
        参数:
            idx: 点x的索引
            unlabeled_indices: 无标注点的索引列表
            distance_matrix: 距离矩阵
            radius: 点x的预估半径
            
        返回:
            int: 被覆盖的点数量
        """
        # 获取点x到所有无标注点的距离
        distances = distance_matrix[idx, unlabeled_indices]
        
        # 计算被覆盖的点数量（距离 <= 半径）
        covered_count = np.sum(distances <= radius)
        
        return covered_count

    def _compute_score(
        self,
        coverage_ratio: float,
        eta_p_distance: float,
        D: int
    ) -> float:
        """
        计算后续轮次采样分数
        score(x) = alpha * coverage_ratio * ln|eta-p|_2 + ln D(x)
        
        参数:
            coverage_ratio: 当前轮次覆盖率
            eta_p_distance: |eta-p|_2的L2距离
            D: D(x)值，被覆盖的点数量
            
        返回:
            float: 采样分数
        """
        # 防止log(0)或log(负数)
        eta_p_distance = max(eta_p_distance, 1e-10)
        D = max(D, 1)
        
        score = self.alpha * coverage_ratio * np.log(eta_p_distance) + np.log(D)
        return score

    def _compute_h_for_unlabeled(
        self,
        model: torch.nn.Module,
        dataset: Any,
        unlabeled_indices: List[int],
        device: torch.device
    ) -> np.ndarray:
        """
        计算每个无标注点的H最大特征值h
        
        计算方式：
        1. 利用当前轮次训练后的模型计算所有无标注点的预测概率向量p
        2. 计算H = diag(p) - p^T p
        3. 计算h为H的最大特征值
        
        参数:
            model: 当前模型
            dataset: 数据集
            unlabeled_indices: 无标注点的索引列表
            device: 计算设备
            
        返回:
            np.ndarray: 每个无标注点的h值
        """
        # 预测所有无标注点的概率向量；局部模式下复用本轮已缓存的 logits。
        probabilities = self._get_probabilities_for_indices(
            model, dataset, unlabeled_indices, device
        )
        
        h_values = []
        for p in probabilities:
            # 计算H = diag(p) - p^T p
            diag_p = np.diag(p)
            ppT = np.outer(p, p)
            H = diag_p - ppT
            
            # 计算H的最大特征值
            eigenvalues = np.linalg.eigvals(H)
            h = np.max(np.abs(eigenvalues))
            h = max(h, self.h_min)  # 设置下界
            h_values.append(h)
        
        return np.array(h_values)

    def _greedy_selection(
        self,
        unlabeled_indices: List[int],
        distance_matrix: np.ndarray,
        radii: np.ndarray
    ) -> Tuple[List[int], List[int]]:
        """
        贪心采样：选择能覆盖最多点的点
        
        参数:
            unlabeled_indices: 无标注点的索引列表
            distance_matrix: 距离矩阵
            radii: 每个无标注点的预估半径（与unlabeled_indices一一对应）
            
        返回:
            Tuple[List[int], List[int]]: (选择的索引列表, 被覆盖的索引列表)
        """
        selected_indices = []
        covered_indices = set()
        remaining_indices = list(unlabeled_indices)
        remaining_radii = list(radii)
        
        # 构建索引到位置的映射
        idx_to_pos = {idx: i for i, idx in enumerate(unlabeled_indices)}

        with tqdm(
            total=len(remaining_indices),
            desc="[OurMethod] 第一轮覆盖进度",
            unit="point"
        ) as progress_bar:
            while remaining_indices:
                # 计算每个剩余点能覆盖多少点
                best_idx = None
                best_covered = set()
                max_covered_count = -1

                for i, idx in enumerate(remaining_indices):
                    radius = remaining_radii[i]
                    # 找出该点覆盖的所有点
                    distances = distance_matrix[idx, remaining_indices]
                    covered = set(remaining_indices[j] for j in np.where(distances <= radius)[0])

                    if len(covered) > max_covered_count:
                        max_covered_count = len(covered)
                        best_idx = i
                        best_covered = covered

                if best_idx is None:
                    break

                # 选择该点
                selected_idx = remaining_indices[best_idx]
                selected_indices.append(selected_idx)
                covered_indices.update(best_covered)
                progress_bar.update(len(best_covered))
                progress_bar.set_postfix(
                    selected=len(selected_indices),
                    remaining=len(remaining_indices) - len(best_covered)
                )

                # 从剩余列表中移除被覆盖的点
                remaining_indices = [idx for idx in remaining_indices if idx not in best_covered]
                remaining_radii = [remaining_radii[j] for j, idx in enumerate(unlabeled_indices)
                                  if idx in remaining_indices]
                unlabeled_indices = remaining_indices  # 更新用于下一次迭代的索引列表
        
        return selected_indices, list(covered_indices)

    def select_samples(
        self,
        model: torch.nn.Module,
        unlabeled_indices: List[int],
        dataset: Any,
        batch_size: int,
        device: torch.device,
        labeled_indices: Optional[List[int]] = None,
        **kwargs: Any
    ) -> List[int]:
        """
        选择要标注的样本
        
        参数:
            model: 训练好的模型
            unlabeled_indices: 未标注样本的索引列表
            dataset: 数据集
            batch_size: 要选择的样本数量（本策略可能忽略此参数）
            device: 计算设备
            labeled_indices: 已标注样本的索引列表
            **kwargs: 其他参数
            
        返回:
            List[int]: 选择的样本索引列表
        """
        if labeled_indices is None:
            all_indices = set(range(len(dataset)))
            labeled_indices = sorted(all_indices - set(unlabeled_indices))
        
        # 加载距离矩阵
        distance_matrix = self._ensure_distance_matrix(split="train")
        
        # 解析放大系数来源：固定全局值或逐点 KNN 局部估计。
        spectral_norm_product = self._resolve_spectral_norm_product(model)
        if spectral_norm_product is None:
            self._prepare_local_expansion_context(
                model=model,
                dataset=dataset,
                distance_matrix=distance_matrix,
                device=device
            )
        else:
            self._clear_local_expansion_context()
        
        if self.round_counter == 0:
            # 第一轮：使用最大理论半径进行贪心采样
            selected_indices = self._first_round_selection(
                model=model,
                dataset=dataset,
                device=device,
                distance_matrix=distance_matrix,
                labeled_indices=labeled_indices,
                unlabeled_indices=unlabeled_indices,
                spectral_norm_product=spectral_norm_product
            )
        else:
            # 后续轮次：使用预估半径进行贪心采样
            selected_indices = self._subsequent_round_selection(
                model=model,
                dataset=dataset,
                device=device,
                distance_matrix=distance_matrix,
                labeled_indices=labeled_indices,
                unlabeled_indices=unlabeled_indices,
                spectral_norm_product=spectral_norm_product
            )
        
        # 增加轮次计数器
        self.round_counter += 1
        
        return selected_indices

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
        第一轮采样
        
        步骤：
        1. 用初始标记的点训练模型（已在select_samples之前完成）
        2. 计算最大理论半径
        3. 更新无标注池：去除被已标注点以最大理论半径覆盖的点
        4. 贪心采样：选择能覆盖最多点的点，直到无标注池为空
        
        参数:
            model: 训练好的模型
            dataset: 数据集
            device: 计算设备
            distance_matrix: 距离矩阵
            labeled_indices: 已标注点的索引列表
            unlabeled_indices: 无标注点的索引列表
            spectral_norm_product: 固定全局放大系数；None 表示使用逐点局部放大系数。
            
        返回:
            List[int]: 选择的样本索引列表
        """
        print(f"\n[OurMethod] 第一轮采样")
        
        # 步骤2：计算逐点最大理论半径。固定全局模式下所有点半径相同。
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
                "[OurMethod] 最大理论半径统计: "
                f"min={round_max_radii.min():.6f}, "
                f"max={round_max_radii.max():.6f}, "
                f"mean={round_max_radii.mean():.6f}"
            )
        
        # 步骤3：更新无标注池，去除被已标注点以最大理论半径覆盖的点
        if labeled_indices:
            # 计算被已标注点覆盖的点
            covered_mask = np.zeros(distance_matrix.shape[0], dtype=bool)
            for idx in labeled_indices:
                distances = distance_matrix[idx]
                covered_mask |= distances <= round_radius_by_index[idx]
            
            # 更新无标注池
            filtered_unlabeled = [idx for idx in unlabeled_indices if not covered_mask[idx]]
            print(f"[OurMethod] 原始无标注池大小: {len(unlabeled_indices)}")
            print(f"[OurMethod] 被已标注点覆盖的点数: {len(unlabeled_indices) - len(filtered_unlabeled)}")
            print(f"[OurMethod] 过滤后无标注池大小: {len(filtered_unlabeled)}")
        else:
            filtered_unlabeled = list(unlabeled_indices)
            print(f"[OurMethod] 无标注池大小: {len(filtered_unlabeled)}")
        
        # 步骤4：贪心采样，每个候选点使用自己的最大理论半径
        if filtered_unlabeled:
            radii = np.array(
                [round_radius_by_index[idx] for idx in filtered_unlabeled],
                dtype=np.float64
            )
            
            # 执行贪心采样
            selected_indices, covered_indices = self._greedy_selection(
                unlabeled_indices=filtered_unlabeled,
                distance_matrix=distance_matrix,
                radii=radii
            )
            
            print(f"[OurMethod] 第一轮选择的标注点数量: {len(selected_indices)}")
            print(f"[OurMethod] 第一轮覆盖的点数量: {len(covered_indices)}")
            
            return selected_indices
        else:
            print(f"[OurMethod] 无标注池为空，无需选择样本")
            return []

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
        后续轮次采样（新逻辑）
        
        步骤：
        1. 用已标注点训练模型（已在select_samples之前完成）
        2. 用get_radius计算每个已标注点的半径
        3. 汇报原始覆盖率，并在需要时执行早期半径归一化
        4. 更新无标注池：去除被已标注点以“当前生效半径”覆盖的点
        5. 预估每个无标注点半径；若本轮触发归一化，则同步按相同比例放大
        6. 计算每个无标注点的采样分数
           score(x) = alpha * coverage_ratio * ln|eta-p|_2 + ln D(x)
           - D(x): 无标注池中被点x以当前生效预测半径覆盖的点的数量（包括自己）
           - eta: p的最大分量对应的one-hot向量
        7. 贪心采样：选择score(x)最大的点，将被它以当前生效预测半径覆盖的点
           从无标注池中移除
           重复直到无标注池为空或达到查询预算B
        
        参数:
            model: 训练好的模型
            dataset: 数据集
            device: 计算设备
            distance_matrix: 距离矩阵
            labeled_indices: 已标注点的索引列表
            unlabeled_indices: 无标注点的索引列表
            spectral_norm_product: 固定全局放大系数；None 表示使用逐点局部放大系数。
            
        返回:
            List[int]: 选择的样本索引列表
        """
        print(f"\n[OurMethod] 第{self.round_counter + 1}轮采样")

        # 步骤2：用get_radius计算每个已标注点的半径
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

        # 步骤3：汇报覆盖率
        coverage_ratio, covered_mask = self._compute_coverage(
            distance_matrix, labeled_indices, raw_labeled_radii.tolist()
        )
        self.last_coverage = coverage_ratio
        print(f"[OurMethod] 当前轮次覆盖率: {coverage_ratio:.6f}")

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
                    "[OurMethod] 早期半径归一化已触发: "
                    f"T={self.early_radius_normalization_threshold:.6f}, "
                    f"k={self.early_radius_normalization_percentile_k:.2f}, "
                    f"q_k={normalization_pivot_radius:.6f}, "
                    f"r={normalization_scale_ratio:.6f}, "
                    f"max_radius={max_theoretical_radius:.6f}, "
                    f"normalized_coverage={coverage_ratio:.6f}"
                )
            else:
                print(
                    "[OurMethod] 早期半径归一化未触发: "
                    f"coverage={coverage_ratio:.6f} >= "
                    f"T={self.early_radius_normalization_threshold:.6f}"
                )

        # 步骤4：更新无标注池，去除被已标注点以其自身计算的半径覆盖的点
        filtered_unlabeled = [idx for idx in unlabeled_indices if not covered_mask[idx]]
        print(f"[OurMethod] 原始无标注池大小: {len(unlabeled_indices)}")
        print(f"[OurMethod] 被已标注点覆盖的点数: {len(unlabeled_indices) - len(filtered_unlabeled)}")
        print(f"[OurMethod] 过滤后无标注池大小: {len(filtered_unlabeled)}")
        
        if not filtered_unlabeled:
            print(f"[OurMethod] 无标注池为空，无需选择样本")
            return []
        
        # 步骤5：预估每个无标注点半径
        h_values = self._compute_h_for_unlabeled(
            model, dataset, filtered_unlabeled, device
        )
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

        print(f"[OurMethod] 预估半径统计: min={estimated_radii.min():.6f}, "
              f"max={estimated_radii.max():.6f}, mean={estimated_radii.mean():.6f}")
        
        # 步骤6：预测所有无标注点的概率向量
        probabilities = self._get_probabilities_for_indices(
            model, dataset, filtered_unlabeled, device
        )
        
        # 步骤7：基于分数的贪心采样
        selected_indices = []
        remaining_indices = list(filtered_unlabeled)
        remaining_radii = list(estimated_radii)
        remaining_probs = list(probabilities)

        with tqdm(
            total=min(self.query_budget, len(remaining_indices)),
            desc=f"[OurMethod] 第{self.round_counter + 1}轮选择进度",
            unit="sample"
        ) as progress_bar:
            while remaining_indices and len(selected_indices) < self.query_budget:
                best_idx = None
                best_score = -np.inf
                best_covered = set()

                # 计算每个剩余点的分数
                for i, idx in enumerate(remaining_indices):
                    radius = remaining_radii[i]
                    prob = remaining_probs[i]

                    # 计算D(x): 被该点覆盖的无标注点数量
                    D = self._compute_D(idx, remaining_indices, distance_matrix, radius)

                    # 计算|eta-p|_2
                    eta_p_distance = self._compute_eta_p_distance(prob)

                    # 计算分数
                    score = self._compute_score(coverage_ratio, eta_p_distance, D)

                    if score > best_score:
                        best_score = score
                        best_idx = i
                        best_covered = set(remaining_indices[j] for j in np.where(
                            distance_matrix[idx, remaining_indices] <= radius
                        )[0])

                if best_idx is None:
                    break

                # 选择该点
                selected_idx = remaining_indices[best_idx]
                selected_indices.append(selected_idx)
                progress_bar.update(1)
                progress_bar.set_postfix(
                    covered=len(best_covered),
                    remaining=len(remaining_indices) - len(best_covered)
                )
                print(f"[OurMethod] 选择样本 {selected_idx}, 分数={best_score:.6f}, "
                      f"覆盖点数={len(best_covered)}")

                # 从剩余列表中移除被覆盖的点
                remaining_indices = [idx for idx in remaining_indices if idx not in best_covered]
                remaining_radii = [remaining_radii[j] for j, idx in enumerate(filtered_unlabeled)
                                  if idx in remaining_indices]
                remaining_probs = [remaining_probs[j] for j, idx in enumerate(filtered_unlabeled)
                                 if idx in remaining_indices]
                filtered_unlabeled = remaining_indices  # 更新用于下一次迭代的索引列表

        print(f"[OurMethod] 第{self.round_counter + 1}轮选择的标注点数量: {len(selected_indices)}")
        
        return selected_indices
