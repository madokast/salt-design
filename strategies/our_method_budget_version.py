"""
Our Method Budget Version - 基于半径覆盖的主动学习策略（预算版本）

该策略实现了一个基于半径覆盖的贪心采样方法：
1. 第一轮：使用最大理论半径进行贪心采样，直到达到查询预算B
2. 后续轮次：使用预估半径进行贪心采样，直到无标注池为空或达到查询预算B
"""
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from typing import List, Any, Dict, Optional, Tuple, Union
from tqdm import tqdm

from .base import ActiveLearningStrategy
from distance_calculator import DistanceCalculator
from get_radius import get_radius

from .local_expansion import (
    compute_estimated_radii_from_expansion,
    compute_estimated_radii_from_expansion_torch,
    compute_knn_local_expansion_factors,
    compute_knn_local_expansion_factors_torch,
    compute_max_theoretical_radii_from_expansion,
    compute_max_theoretical_radii_from_expansion_torch,
    predict_logits_for_indices,
    predict_logits_for_indices_torch,
    resolve_expansion_factors_for_indices,
    resolve_expansion_factors_for_indices_torch,
    softmax_numpy,
    softmax_torch,
)


class OurMethodBudgetVersionStrategy(ActiveLearningStrategy):
    """
    基于半径覆盖的主动学习策略（预算版本）
    
    该策略在第一轮使用最大理论半径进行贪心采样，直到达到查询预算B，
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
        alpha: Optional[float] = 0.1,
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
                这一项的权重系数（默认0.1）；为 None 时按当前候选集上
                uncertainty 项与 logD 项的 IQR 自动做尺度校准。
            inference_batch_size: 采样阶段做概率预测时使用的批量大小。
                该参数只影响实现效率，不改变采样逻辑或采样结果。
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
        super().__init__("OurMethodBudgetVersion")
        if not 0.0 <= early_radius_normalization_threshold <= 1.0:
            raise ValueError(
                "early_radius_normalization_threshold 必须位于 [0, 1] 区间内"
            )
        if not 0.0 < early_radius_normalization_percentile_k <= 100.0:
            raise ValueError(
                "early_radius_normalization_percentile_k 必须位于 (0, 100] 区间内"
            )
        if inference_batch_size <= 0:
            raise ValueError("inference_batch_size 必须是正整数")
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
        self.inference_batch_size = inference_batch_size
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
        self._current_logits_by_index: Optional[Union[np.ndarray, torch.Tensor]] = None
        self._current_local_expansion_factors: Optional[Union[np.ndarray, torch.Tensor]] = None

        # CUDA 采样缓存：在 CUDA 环境下把完整距离矩阵常驻设备，避免后续轮次重复搬运。
        self._cuda_distance_matrix: Optional[torch.Tensor] = None
        self._cuda_distance_matrix_device: Optional[str] = None
        self._cuda_distance_matrix_source_id: Optional[int] = None

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
        self._clear_cuda_sampling_context()

    def _clear_cuda_sampling_context(self) -> None:
        """
        清空当前轮缓存的 CUDA 距离矩阵。
        """
        self._cuda_distance_matrix = None
        self._cuda_distance_matrix_device = None
        self._cuda_distance_matrix_source_id = None

    def _should_use_cuda_sampling(self, device: torch.device) -> bool:
        """
        判断当前轮次是否启用 CUDA 原生采样路径。
        """
        return device.type == "cuda"

    def _get_cuda_distance_matrix(
        self,
        distance_matrix: np.ndarray,
        device: torch.device
    ) -> torch.Tensor:
        """
        获取当前轮次使用的 CUDA 距离矩阵缓存。
        """
        if device.type != "cuda":
            raise ValueError("只有 CUDA 设备才允许获取 CUDA 距离矩阵缓存。")

        source_id = id(distance_matrix)
        device_key = str(device)
        if (
            self._cuda_distance_matrix is None
            or self._cuda_distance_matrix_source_id != source_id
            or self._cuda_distance_matrix_device != device_key
        ):
            self._cuda_distance_matrix = torch.as_tensor(
                distance_matrix,
                device=device
            )
            self._cuda_distance_matrix_source_id = source_id
            self._cuda_distance_matrix_device = device_key
        return self._cuda_distance_matrix

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
        if self._should_use_cuda_sampling(device):
            cuda_distance_matrix = self._get_cuda_distance_matrix(
                distance_matrix=distance_matrix,
                device=device
            )
            self._current_logits_by_index = predict_logits_for_indices_torch(
                model=model,
                dataset=dataset,
                indices=all_indices,
                device=device,
                batch_size=self.inference_batch_size
            )
            self._current_local_expansion_factors = (
                compute_knn_local_expansion_factors_torch(
                    query_indices=all_indices,
                    logits_by_index=self._current_logits_by_index,
                    distance_matrix=cuda_distance_matrix,
                    knn_k=self.local_lipschitz_k,
                    quantile=self.local_lipschitz_quantile,
                    distance_eps=self.local_lipschitz_distance_eps,
                    min_factor=self.local_lipschitz_min_value
                )
            )
        else:
            self._current_logits_by_index = predict_logits_for_indices(
                model=model,
                dataset=dataset,
                indices=all_indices,
                device=device,
                batch_size=self.inference_batch_size
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

    def _get_expansion_factors_for_indices_torch(
        self,
        indices: List[int],
        spectral_norm_product: Optional[float],
        device: torch.device
    ) -> torch.Tensor:
        """
        获取一组样本当前应使用的放大系数，并保留在目标 CUDA 设备上。
        """
        return resolve_expansion_factors_for_indices_torch(
            indices=indices,
            fixed_expansion_factor=spectral_norm_product,
            local_expansion_factors=self._current_local_expansion_factors,
            device=device
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

    def _compute_max_theoretical_radii_for_indices_torch(
        self,
        indices: List[int],
        spectral_norm_product: Optional[float],
        device: torch.device
    ) -> torch.Tensor:
        """
        为一组样本计算逐点最大理论半径，并保留在目标设备上。
        """
        if self.max_theoretical_radius is not None:
            return torch.full(
                (len(indices),),
                float(self.max_theoretical_radius),
                dtype=torch.float64,
                device=device
            )

        expansion_factors = self._get_expansion_factors_for_indices_torch(
            indices=indices,
            spectral_norm_product=spectral_norm_product,
            device=device
        )
        return compute_max_theoretical_radii_from_expansion_torch(
            expansion_factors=expansion_factors,
            epsilon=self.epsilon,
            h_min=self.h_min,
            diff_min=self.diff_min
        )

    def _compute_round_max_theoretical_radius_torch(
        self,
        indices: List[int],
        spectral_norm_product: Optional[float],
        device: torch.device
    ) -> float:
        """
        为当前轮早期半径归一化解析一个兼容旧逻辑的标量最大理论半径。
        """
        if not indices:
            if self.max_theoretical_radius is not None:
                return float(self.max_theoretical_radius)
            return float(self._compute_max_theoretical_radius(1.0))
        radii = self._compute_max_theoretical_radii_for_indices_torch(
            indices=indices,
            spectral_norm_product=spectral_norm_product,
            device=device
        )
        return float(torch.max(radii).item())

    def _clip_radii_to_max_theoretical_torch(
        self,
        radii: torch.Tensor,
        indices: List[int],
        spectral_norm_product: Optional[float],
        device: torch.device
    ) -> torch.Tensor:
        """
        用当前最大理论半径上限截断一组样本半径，并保留在目标设备上。
        """
        if radii.shape[0] != len(indices):
            raise ValueError("radii 的长度必须与 indices 一致。")
        max_radii = self._compute_max_theoretical_radii_for_indices_torch(
            indices=indices,
            spectral_norm_product=spectral_norm_product,
            device=device
        )
        return torch.minimum(radii.to(dtype=torch.float64), max_radii)

    def _normalize_radii_for_early_rounds_torch(
        self,
        radii: torch.Tensor,
        coverage_ratio: float,
        max_theoretical_radius: float
    ) -> Tuple[torch.Tensor, bool, float, float]:
        """
        根据早期半径归一化配置，在 GPU 上决定当前轮次是否需要放大半径。
        """
        radii = radii.to(dtype=torch.float64)
        if radii.ndim != 1:
            raise ValueError("radii 必须是一维张量。")
        if not np.isfinite(max_theoretical_radius) or max_theoretical_radius <= 0.0:
            raise ValueError("max_theoretical_radius 必须是有限正数。")
        if radii.numel() == 0:
            return radii.clone(), False, 1.0, 0.0
        if (
            not self.enable_early_radius_normalization
            or coverage_ratio >= self.early_radius_normalization_threshold
        ):
            return radii.clone(), False, 1.0, float(torch.max(radii).item())

        sorted_desc = torch.sort(radii, descending=True).values
        percentile_index = int(
            np.ceil(
                sorted_desc.numel()
                * (self.early_radius_normalization_percentile_k / 100.0)
            )
        ) - 1
        percentile_index = min(max(percentile_index, 0), sorted_desc.numel() - 1)
        pivot_radius = float(max(sorted_desc[percentile_index].item(), 1e-12))
        scale_ratio = float(max(1.0, max_theoretical_radius / pivot_radius))
        normalized_radii = torch.minimum(
            radii * scale_ratio,
            torch.full_like(radii, max_theoretical_radius)
        )
        return normalized_radii, True, scale_ratio, pivot_radius

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

    def _compute_labeled_radii_torch(
        self,
        labeled_points: List[Dict[str, Any]],
        labeled_indices: List[int],
        model: torch.nn.Module,
        spectral_norm_product: Optional[float],
        device: torch.device
    ) -> torch.Tensor:
        """
        批量计算已标注点半径，并保留在目标 CUDA 设备上。
        """
        if len(labeled_points) != len(labeled_indices):
            raise ValueError("labeled_points 与 labeled_indices 长度必须一致。")
        if not labeled_points:
            return torch.empty((0,), dtype=torch.float64, device=device)

        expansion_factors = self._get_expansion_factors_for_indices_torch(
            indices=labeled_indices,
            spectral_norm_product=spectral_norm_product,
            device=device
        )

        batch_tensors: List[torch.Tensor] = []
        labels: List[int] = []
        soft_labels: List[torch.Tensor] = []
        for point in labeled_points:
            data = point["data"]
            if not isinstance(data, torch.Tensor):
                data = torch.as_tensor(data)
            if data.dim() == 3:
                data = data.unsqueeze(0)
            elif data.dim() == 1:
                data = data.unsqueeze(0)
            batch_tensors.append(data.to(device))
            labels.append(int(point["label"]))
            if self.use_soft_labels:
                if "soft_label" not in point:
                    raise ValueError(
                        "use_soft_label=True 但 labeled_point 中未提供 soft_label"
                    )
                soft_label = point["soft_label"]
                if not isinstance(soft_label, torch.Tensor):
                    soft_label = torch.as_tensor(soft_label)
                soft_labels.append(soft_label.to(device))

        model.eval()
        with torch.no_grad():
            inputs = torch.cat(batch_tensors, dim=0)
            logits = model(inputs)
            probabilities = self._sanitize_probabilities_torch(
                torch.softmax(logits, dim=1),
                context="已标注点概率预测"
            )

        num_classes = probabilities.shape[1]
        if self.use_soft_labels:
            label_vectors = torch.stack(soft_labels, dim=0).to(dtype=torch.float64)
            if label_vectors.shape[1] != num_classes:
                raise ValueError(
                    "soft_label 维度与类别数不一致: "
                    f"{label_vectors.shape[1]} vs {num_classes}"
                )
        else:
            label_tensor = torch.as_tensor(
                labels,
                dtype=torch.long,
                device=device
            )
            label_vectors = torch.zeros_like(probabilities)
            label_vectors.scatter_(1, label_tensor.unsqueeze(1), 1.0)

        diff_values = torch.linalg.vector_norm(
            label_vectors - probabilities,
            dim=1
        ).clamp_min(self.diff_min)
        h_values = self._compute_hessian_spectral_norms_from_probabilities_torch(
            probabilities=probabilities,
            context="已标注点半径计算"
        )

        sqrt_term = torch.sqrt(diff_values * diff_values + 2.0 * h_values * self.epsilon)
        raw_radii = (-diff_values + sqrt_term) / (h_values * expansion_factors)
        return self._clip_radii_to_max_theoretical_torch(
            radii=raw_radii,
            indices=labeled_indices,
            spectral_norm_product=spectral_norm_product,
            device=device
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

    def _estimate_radii_for_unlabeled_torch(
        self,
        h_values: torch.Tensor,
        unlabeled_indices: List[int],
        spectral_norm_product: Optional[float],
        device: torch.device
    ) -> torch.Tensor:
        """
        批量预估无标注候选点半径，并保留在目标 CUDA 设备上。
        """
        expansion_factors = self._get_expansion_factors_for_indices_torch(
            indices=unlabeled_indices,
            spectral_norm_product=spectral_norm_product,
            device=device
        )
        estimated_radii = compute_estimated_radii_from_expansion_torch(
            h_values=h_values,
            expansion_factors=expansion_factors,
            epsilon=self.epsilon,
            diff_min=self.diff_min
        )
        return self._clip_radii_to_max_theoretical_torch(
            radii=estimated_radii,
            indices=unlabeled_indices,
            spectral_norm_product=spectral_norm_product,
            device=device
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
            if isinstance(self._current_logits_by_index, torch.Tensor):
                return softmax_torch(
                    self._current_logits_by_index[indices]
                ).detach().cpu().numpy()
            return softmax_numpy(self._current_logits_by_index[indices])
        return self._predict_probabilities(model, dataset, indices, device)

    def _get_probabilities_for_indices_torch(
        self,
        model: torch.nn.Module,
        dataset: Any,
        indices: List[int],
        device: torch.device
    ) -> torch.Tensor:
        """
        获取指定样本的预测概率，并保留在目标 CUDA 设备上。
        """
        if self._current_logits_by_index is not None:
            if isinstance(self._current_logits_by_index, torch.Tensor):
                return self._sanitize_probabilities_torch(
                    softmax_torch(self._current_logits_by_index[indices]),
                    context="候选点概率预测"
                )
            logits = torch.as_tensor(
                self._current_logits_by_index[indices],
                dtype=torch.float64,
                device=device
            )
            return self._sanitize_probabilities_torch(
                softmax_torch(logits),
                context="候选点概率预测"
            )
        return self._sanitize_probabilities_torch(
            self._predict_probabilities_torch(model, dataset, indices, device),
            context="候选点概率预测"
        )

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
        if not labeled_indices:
            covered_mask = np.zeros(n_total, dtype=bool)
            return 0.0, covered_mask

        labeled_indices_array = np.asarray(labeled_indices, dtype=np.int64)
        radii_array = np.asarray(radii, dtype=np.float64)
        if labeled_indices_array.shape[0] != radii_array.shape[0]:
            raise ValueError("labeled_indices 与 radii 长度必须一致。")

        covered_mask = np.any(
            distance_matrix[labeled_indices_array] <= radii_array[:, None],
            axis=0
        )
        coverage_ratio = float(covered_mask.sum()) / float(n_total) if n_total > 0 else 0.0
        return coverage_ratio, covered_mask

    def _filter_unlabeled_indices(
        self,
        unlabeled_indices: List[int],
        covered_mask: np.ndarray
    ) -> List[int]:
        """
        根据覆盖掩码批量过滤无标注池，保持原始顺序不变。
        """
        if not unlabeled_indices:
            return []
        unlabeled_indices_array = np.asarray(unlabeled_indices, dtype=np.int64)
        return unlabeled_indices_array[~covered_mask[unlabeled_indices_array]].tolist()

    def _compute_coverage_torch(
        self,
        distance_matrix: torch.Tensor,
        labeled_indices: List[int],
        radii: Union[List[float], torch.Tensor]
    ) -> Tuple[float, torch.Tensor]:
        """
        在 CUDA 上计算覆盖率。
        """
        n_total = distance_matrix.shape[0]
        if not labeled_indices:
            covered_mask = torch.zeros(
                n_total,
                dtype=torch.bool,
                device=distance_matrix.device
            )
            return 0.0, covered_mask

        labeled_indices_tensor = torch.as_tensor(
            labeled_indices,
            dtype=torch.long,
            device=distance_matrix.device
        )
        radii_tensor = torch.as_tensor(
            radii,
            dtype=torch.float64,
            device=distance_matrix.device
        )
        if labeled_indices_tensor.shape[0] != radii_tensor.shape[0]:
            raise ValueError("labeled_indices 与 radii 长度必须一致。")

        covered_mask = torch.any(
            distance_matrix.index_select(0, labeled_indices_tensor).to(dtype=torch.float64)
            <= radii_tensor.unsqueeze(1),
            dim=0
        )
        coverage_ratio = (
            float(covered_mask.sum().item()) / float(n_total)
            if n_total > 0 else 0.0
        )
        return coverage_ratio, covered_mask

    def _filter_unlabeled_indices_torch(
        self,
        unlabeled_indices: List[int],
        covered_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        根据覆盖掩码批量过滤无标注池，并保留原始顺序。
        """
        if not unlabeled_indices:
            return torch.empty((0,), dtype=torch.long, device=covered_mask.device)
        unlabeled_indices_tensor = torch.as_tensor(
            unlabeled_indices,
            dtype=torch.long,
            device=covered_mask.device
        )
        return unlabeled_indices_tensor[
            ~covered_mask.index_select(0, unlabeled_indices_tensor)
        ]

    def _prepare_candidate_cover_matrix_from_full_distance_torch(
        self,
        distance_matrix: torch.Tensor,
        candidate_indices: torch.Tensor,
        candidate_radii: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        在 CUDA 上预计算候选集的精确覆盖关系矩阵与初始覆盖数。
        """
        candidate_indices = candidate_indices.to(
            device=distance_matrix.device,
            dtype=torch.long
        )
        candidate_radii = candidate_radii.to(
            device=distance_matrix.device,
            dtype=torch.float64
        )

        if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
            raise ValueError("distance_matrix 必须是方阵。")
        if candidate_indices.shape[0] != candidate_radii.shape[0]:
            raise ValueError("candidate_indices 与 candidate_radii 长度必须一致。")
        if candidate_indices.numel() == 0:
            return (
                torch.empty(
                    (0, 0),
                    dtype=torch.bool,
                    device=distance_matrix.device
                ),
                torch.empty(
                    (0,),
                    dtype=torch.int64,
                    device=distance_matrix.device
                )
            )

        num_candidates = int(candidate_indices.numel())
        cover_matrix = torch.empty(
            (num_candidates, num_candidates),
            dtype=torch.bool,
            device=distance_matrix.device
        )
        covered_counts = torch.empty(
            (num_candidates,),
            dtype=torch.int64,
            device=distance_matrix.device
        )
        block_size = max(1, min(int(self.inference_batch_size), num_candidates))

        for block_start in range(0, num_candidates, block_size):
            block_end = min(block_start + block_size, num_candidates)
            row_positions = torch.arange(
                block_start,
                block_end,
                dtype=torch.long,
                device=distance_matrix.device
            )
            row_global_indices = candidate_indices.index_select(0, row_positions)
            distance_block = distance_matrix.index_select(
                0,
                row_global_indices
            ).index_select(1, candidate_indices).to(dtype=torch.float64)
            covered_mask_block = (
                distance_block <= candidate_radii.index_select(0, row_positions).unsqueeze(1)
            )
            cover_matrix[block_start:block_end] = covered_mask_block
            covered_counts[block_start:block_end] = covered_mask_block.sum(
                dim=1,
                dtype=torch.int64
            )

        return cover_matrix, covered_counts

    def _greedy_select_candidate_positions_from_cover_matrix_torch(
        self,
        candidate_indices: torch.Tensor,
        static_score_terms: torch.Tensor,
        cover_matrix: torch.Tensor,
        initial_covered_counts: torch.Tensor,
        query_budget: int,
        progress_desc: str,
        log_prefix: str
    ) -> List[int]:
        """
        基于 GPU 上的覆盖关系矩阵做精确增量贪心选择。
        """
        candidate_indices = candidate_indices.to(
            device=cover_matrix.device,
            dtype=torch.long
        )
        static_score_terms = static_score_terms.to(
            device=cover_matrix.device,
            dtype=torch.float64
        )
        initial_covered_counts = initial_covered_counts.to(
            device=cover_matrix.device,
            dtype=torch.int64
        )

        selected_local_positions: List[int] = []
        if candidate_indices.numel() == 0 or query_budget <= 0:
            return selected_local_positions

        active_mask = torch.ones(
            candidate_indices.shape[0],
            dtype=torch.bool,
            device=cover_matrix.device
        )
        covered_counts = initial_covered_counts.clone()

        with tqdm(
            total=min(query_budget, int(candidate_indices.numel())),
            desc=progress_desc,
            unit="sample",
            mininterval=0.5
        ) as progress_bar:
            while bool(torch.any(active_mask).item()) and len(selected_local_positions) < query_budget:
                active_positions = torch.nonzero(
                    active_mask,
                    as_tuple=False
                ).flatten()
                active_scores = (
                    static_score_terms.index_select(0, active_positions)
                    + torch.log(
                        torch.clamp(
                            covered_counts.index_select(0, active_positions),
                            min=1
                        ).to(dtype=torch.float64)
                    )
                )
                active_covered_counts = covered_counts.index_select(0, active_positions)

                best_active_offset = int(torch.argmax(active_scores).item())
                best_local_position = int(active_positions[best_active_offset].item())
                best_covered_positions = active_positions[
                    cover_matrix[best_local_position].index_select(0, active_positions)
                ]

                if best_covered_positions.numel() == 0:
                    break

                if int(best_covered_positions.numel()) == 1:
                    unit_cover_mask = active_covered_counts == 1
                    unit_cover_offsets = torch.nonzero(
                        unit_cover_mask,
                        as_tuple=False
                    ).flatten()
                    unit_cover_scores = active_scores.index_select(0, unit_cover_offsets)

                    non_unit_cover_offsets = torch.nonzero(
                        ~unit_cover_mask,
                        as_tuple=False
                    ).flatten()
                    prefix_unit_mask = torch.ones_like(
                        unit_cover_offsets,
                        dtype=torch.bool
                    )
                    if non_unit_cover_offsets.numel() > 0:
                        top_non_unit_offset = int(torch.argmax(
                            active_scores.index_select(0, non_unit_cover_offsets)
                        ).item())
                        first_non_unit_offset = int(
                            non_unit_cover_offsets[top_non_unit_offset].item()
                        )
                        first_non_unit_score = float(
                            active_scores[first_non_unit_offset].item()
                        )
                        prefix_unit_mask = (
                            (unit_cover_scores > first_non_unit_score)
                            | (
                                (unit_cover_scores == first_non_unit_score)
                                & (unit_cover_offsets < first_non_unit_offset)
                            )
                        )

                    prefix_unit_offsets = unit_cover_offsets[prefix_unit_mask]
                    if prefix_unit_offsets.numel() == 0:
                        prefix_unit_offsets = torch.tensor(
                            [best_active_offset],
                            dtype=torch.long,
                            device=cover_matrix.device
                        )

                    prefix_scores = active_scores.index_select(0, prefix_unit_offsets)
                    prefix_order = torch.argsort(
                        prefix_scores,
                        descending=True,
                        stable=True
                    )
                    ranked_unit_offsets = prefix_unit_offsets.index_select(
                        0,
                        prefix_order
                    )

                    batch_size = min(
                        int(ranked_unit_offsets.numel()),
                        query_budget - len(selected_local_positions)
                    )
                    batch_offsets = ranked_unit_offsets[:batch_size]
                    batch_local_positions = active_positions.index_select(0, batch_offsets)
                    selected_local_positions.extend(
                        batch_local_positions.detach().cpu().tolist()
                    )
                    active_mask[batch_local_positions] = False
                    covered_counts = covered_counts - cover_matrix.index_select(
                        1,
                        batch_local_positions
                    ).sum(dim=1, dtype=torch.int64)
                    progress_bar.update(batch_size)
                    print(
                        f"[{log_prefix}] 触发 D=1 批量选择: "
                        f"batch_size={batch_size}, "
                        f"score_max={float(active_scores[batch_offsets[0]].item()):.6f}, "
                        f"score_min={float(active_scores[batch_offsets[-1]].item()):.6f}, "
                        f"remaining={int(active_mask.sum().item())}"
                    )
                    continue

                selected_local_positions.append(best_local_position)
                active_mask[best_covered_positions] = False
                covered_counts = covered_counts - cover_matrix.index_select(
                    1,
                    best_covered_positions
                ).sum(dim=1, dtype=torch.int64)
                progress_bar.update(1)

        return selected_local_positions

    def _greedy_selection_torch(
        self,
        unlabeled_indices: torch.Tensor,
        distance_matrix: torch.Tensor,
        radii: torch.Tensor,
        query_budget: int,
        progress_desc: str,
        log_prefix: str
    ) -> Tuple[List[int], List[int], torch.Tensor]:
        """
        在 CUDA 上执行第一轮最大覆盖贪心采样。
        """
        if unlabeled_indices.numel() == 0:
            empty_cover = torch.empty(
                (0, 0),
                dtype=torch.bool,
                device=distance_matrix.device
            )
            return [], [], empty_cover

        cover_matrix, initial_covered_counts = (
            self._prepare_candidate_cover_matrix_from_full_distance_torch(
                distance_matrix=distance_matrix,
                candidate_indices=unlabeled_indices,
                candidate_radii=radii
            )
        )
        selected_local_positions = (
            self._greedy_select_candidate_positions_from_cover_matrix_torch(
                candidate_indices=unlabeled_indices,
                static_score_terms=torch.zeros(
                    unlabeled_indices.shape[0],
                    dtype=torch.float64,
                    device=distance_matrix.device
                ),
                cover_matrix=cover_matrix,
                initial_covered_counts=initial_covered_counts,
                query_budget=query_budget,
                progress_desc=progress_desc,
                log_prefix=log_prefix
            )
        )
        if not selected_local_positions:
            return [], [], cover_matrix

        selected_positions_tensor = torch.as_tensor(
            selected_local_positions,
            dtype=torch.long,
            device=distance_matrix.device
        )
        selected_indices = unlabeled_indices.index_select(
            0,
            selected_positions_tensor
        ).detach().cpu().tolist()
        covered_union = torch.any(
            cover_matrix.index_select(0, selected_positions_tensor),
            dim=0
        )
        covered_indices = unlabeled_indices[covered_union].detach().cpu().tolist()
        return selected_indices, covered_indices, cover_matrix

    def _prepare_candidate_cover_data_from_full_distance(
        self,
        distance_matrix: np.ndarray,
        candidate_indices: np.ndarray,
        candidate_radii: np.ndarray
    ) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
        """
        预计算候选集的精确覆盖关系，用于后续轮次的增量贪心选择。
        """
        candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
        candidate_radii = np.asarray(candidate_radii, dtype=np.float64)

        if distance_matrix.shape[0] != distance_matrix.shape[1]:
            raise ValueError("distance_matrix 必须是方阵。")
        if candidate_indices.shape[0] != candidate_radii.shape[0]:
            raise ValueError("candidate_indices 与 candidate_radii 长度必须一致。")
        if candidate_indices.size == 0:
            return [], [], np.empty((0,), dtype=np.int64)

        num_candidates = candidate_indices.shape[0]
        covered_positions_by_candidate: List[np.ndarray] = [
            np.empty((0,), dtype=np.int64) for _ in range(num_candidates)
        ]
        covered_counts = np.empty(num_candidates, dtype=np.int64)
        block_size = max(1, min(int(self.inference_batch_size), num_candidates))

        for block_start in range(0, num_candidates, block_size):
            block_end = min(block_start + block_size, num_candidates)
            row_positions = np.arange(block_start, block_end, dtype=np.int64)
            row_global_indices = candidate_indices[row_positions]
            distance_block = distance_matrix[np.ix_(row_global_indices, candidate_indices)]
            covered_mask_block = (
                distance_block <= candidate_radii[row_positions][:, None]
            )
            covered_counts[block_start:block_end] = np.count_nonzero(
                covered_mask_block,
                axis=1
            ).astype(np.int64, copy=False)

            for row_offset, local_position in enumerate(row_positions):
                covered_positions_by_candidate[int(local_position)] = np.flatnonzero(
                    covered_mask_block[row_offset]
                ).astype(np.int64, copy=False)

        reverse_cover_rows: List[np.ndarray] = [
            np.empty((0,), dtype=np.int64) for _ in range(num_candidates)
        ]
        nonempty_row_positions = np.flatnonzero(covered_counts > 0)
        if nonempty_row_positions.size > 0:
            concatenated_covered_positions = np.concatenate([
                covered_positions_by_candidate[int(row_position)]
                for row_position in nonempty_row_positions
            ])
            concatenated_cover_rows = np.repeat(
                nonempty_row_positions,
                covered_counts[nonempty_row_positions]
            )
            order = np.argsort(concatenated_covered_positions, kind="mergesort")
            sorted_targets = concatenated_covered_positions[order]
            sorted_rows = concatenated_cover_rows[order]
            unique_targets, start_indices, counts = np.unique(
                sorted_targets,
                return_index=True,
                return_counts=True
            )
            for target, start, count in zip(unique_targets, start_indices, counts):
                reverse_cover_rows[int(target)] = sorted_rows[
                    int(start): int(start + count)
                ]

        return covered_positions_by_candidate, reverse_cover_rows, covered_counts

    def _update_covered_counts_after_removal(
        self,
        covered_counts: np.ndarray,
        active_mask: np.ndarray,
        reverse_cover_rows: List[np.ndarray],
        removed_positions: np.ndarray
    ) -> None:
        """
        在移除一批候选点后，精确更新其余活跃候选点的 D 值。
        """
        removed_positions = np.asarray(removed_positions, dtype=np.int64)
        reverse_chunks = [
            reverse_cover_rows[int(position)]
            for position in removed_positions
            if reverse_cover_rows[int(position)].size > 0
        ]
        if not reverse_chunks:
            return

        covering_rows = np.concatenate(reverse_chunks)
        if covering_rows.size == 0:
            return

        active_covering_rows = covering_rows[active_mask[covering_rows]]
        if active_covering_rows.size == 0:
            return

        unique_covering_rows, decrement_counts = np.unique(
            active_covering_rows,
            return_counts=True
        )
        covered_counts[unique_covering_rows] -= decrement_counts.astype(
            np.int64,
            copy=False
        )

    def _greedy_select_candidate_positions_from_cover_data(
        self,
        candidate_indices: np.ndarray,
        static_score_terms: np.ndarray,
        covered_positions_by_candidate: List[np.ndarray],
        reverse_cover_rows: List[np.ndarray],
        initial_covered_counts: np.ndarray,
        query_budget: int,
        progress_desc: str,
        log_prefix: str
    ) -> List[int]:
        """
        基于预计算覆盖关系做精确增量贪心选择，不改变原始打分与删点语义。
        """
        candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
        static_score_terms = np.asarray(static_score_terms, dtype=np.float64)
        initial_covered_counts = np.asarray(initial_covered_counts, dtype=np.int64)

        selected_local_positions: List[int] = []
        if candidate_indices.size == 0 or query_budget <= 0:
            return selected_local_positions

        active_mask = np.ones(candidate_indices.shape[0], dtype=bool)
        covered_counts = initial_covered_counts.copy()

        with tqdm(
            total=min(query_budget, len(candidate_indices)),
            desc=progress_desc,
            unit="sample",
            mininterval=0.5
        ) as progress_bar:
            while active_mask.any() and len(selected_local_positions) < query_budget:
                active_positions = np.flatnonzero(active_mask)
                active_scores = (
                    static_score_terms[active_positions]
                    + np.log(np.maximum(covered_counts[active_positions], 1))
                )
                active_covered_counts = covered_counts[active_positions]

                best_active_offset = int(np.argmax(active_scores))
                best_local_position = int(active_positions[best_active_offset])
                best_covered_positions = covered_positions_by_candidate[best_local_position]
                best_covered_positions = best_covered_positions[
                    active_mask[best_covered_positions]
                ]

                if best_covered_positions.size == 0:
                    break

                best_score = float(active_scores[best_active_offset])

                if int(best_covered_positions.size) == 1:
                    unit_cover_mask = active_covered_counts == 1
                    unit_cover_offsets = np.flatnonzero(unit_cover_mask)
                    unit_cover_scores = active_scores[unit_cover_offsets]
                    unit_cover_ranks = unit_cover_offsets

                    non_unit_cover_offsets = np.flatnonzero(~unit_cover_mask)
                    prefix_unit_mask = np.ones(
                        unit_cover_offsets.shape[0],
                        dtype=bool
                    )
                    if non_unit_cover_offsets.size > 0:
                        top_non_unit_offset = int(np.argmax(
                            active_scores[non_unit_cover_offsets]
                        ))
                        first_non_unit_offset = int(
                            non_unit_cover_offsets[top_non_unit_offset]
                        )
                        first_non_unit_score = float(active_scores[first_non_unit_offset])
                        prefix_unit_mask = (
                            (unit_cover_scores > first_non_unit_score)
                            | (
                                (unit_cover_scores == first_non_unit_score)
                                & (unit_cover_ranks < first_non_unit_offset)
                            )
                        )

                    prefix_unit_offsets = unit_cover_offsets[prefix_unit_mask]
                    if prefix_unit_offsets.size == 0:
                        prefix_unit_offsets = np.array(
                            [best_active_offset],
                            dtype=np.int64
                        )

                    prefix_order = np.lexsort((
                        prefix_unit_offsets,
                        -active_scores[prefix_unit_offsets]
                    ))
                    ranked_unit_offsets = prefix_unit_offsets[prefix_order]

                    batch_size = min(
                        ranked_unit_offsets.size,
                        query_budget - len(selected_local_positions)
                    )
                    batch_offsets = ranked_unit_offsets[:batch_size]
                    batch_local_positions = active_positions[batch_offsets]
                    selected_local_positions.extend(
                        batch_local_positions.astype(np.int64, copy=False).tolist()
                    )
                    active_mask[batch_local_positions] = False
                    self._update_covered_counts_after_removal(
                        covered_counts=covered_counts,
                        active_mask=active_mask,
                        reverse_cover_rows=reverse_cover_rows,
                        removed_positions=batch_local_positions
                    )
                    progress_bar.update(batch_size)
                    print(
                        f"[{log_prefix}] 触发 D=1 批量选择: "
                        f"batch_size={batch_size}, "
                        f"score_max={active_scores[batch_offsets[0]]:.6f}, "
                        f"score_min={active_scores[batch_offsets[-1]]:.6f}, "
                        f"remaining={int(active_mask.sum())}"
                    )
                    continue

                selected_local_positions.append(best_local_position)
                active_mask[best_covered_positions] = False
                self._update_covered_counts_after_removal(
                    covered_counts=covered_counts,
                    active_mask=active_mask,
                    reverse_cover_rows=reverse_cover_rows,
                    removed_positions=best_covered_positions
                )
                progress_bar.update(1)

        return selected_local_positions

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
        if not indices:
            return np.empty((0, 0), dtype=np.float32)

        model.eval()
        probabilities: List[torch.Tensor] = []
        subset = Subset(dataset, indices)
        prediction_loader = DataLoader(
            subset,
            batch_size=min(self.inference_batch_size, len(indices)),
            shuffle=False
        )

        with torch.no_grad():
            for batch_data in prediction_loader:
                if isinstance(batch_data, (list, tuple)):
                    inputs = batch_data[0]
                else:
                    inputs = batch_data

                if not isinstance(inputs, torch.Tensor):
                    inputs = torch.as_tensor(inputs)

                inputs = inputs.to(device)
                logits = model(inputs)
                probs = torch.softmax(logits, dim=1)
                probabilities.append(probs.detach().cpu())

        return torch.cat(probabilities, dim=0).numpy()

    def _predict_probabilities_torch(
        self,
        model: torch.nn.Module,
        dataset: Any,
        indices: List[int],
        device: torch.device
    ) -> torch.Tensor:
        """
        预测给定索引样本的概率向量，并保留在目标 CUDA 设备上。
        """
        if not indices:
            return torch.empty((0, 0), dtype=torch.float64, device=device)

        model.eval()
        probabilities: List[torch.Tensor] = []
        subset = Subset(dataset, indices)
        prediction_loader = DataLoader(
            subset,
            batch_size=min(self.inference_batch_size, len(indices)),
            shuffle=False
        )

        with torch.no_grad():
            for batch_data in prediction_loader:
                if isinstance(batch_data, (list, tuple)):
                    inputs = batch_data[0]
                else:
                    inputs = batch_data

                if not isinstance(inputs, torch.Tensor):
                    inputs = torch.as_tensor(inputs)

                inputs = inputs.to(device)
                logits = model(inputs)
                probabilities.append(
                    torch.softmax(logits, dim=1).detach().to(dtype=torch.float64)
                )

        return torch.cat(probabilities, dim=0)

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
        if self.alpha is None:
            raise RuntimeError(
                "alpha=None 需要候选集尺度信息，请使用批量静态项计算路径。"
            )
        
        score = self.alpha * coverage_ratio * np.log(eta_p_distance) + np.log(D)
        return score

    def _compute_iqr(
        self,
        values: np.ndarray
    ) -> float:
        """
        计算一维数组的四分位距 IQR。

        IQR 定义为第 75 百分位数与第 25 百分位数之差，用来刻画一组数中间
        50% 样本的典型变化范围。相比标准差，IQR 对极端 margin 或极端覆盖数更
        稳健，因此适合作为自动 alpha 的尺度估计量。

        参数:
            values: 一维数值数组。

        返回:
            float: `Q75(values) - Q25(values)`；若数组为空或没有有限值则返回 0。
        """
        values = np.asarray(values, dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return 0.0
        q75, q25 = np.percentile(values, [75.0, 25.0])
        return float(max(q75 - q25, 0.0))

    def _compute_iqr_torch(
        self,
        values: torch.Tensor
    ) -> float:
        """
        计算一维张量的四分位距 IQR。
        """
        values = values.to(dtype=torch.float64)
        values = values[torch.isfinite(values)]
        if values.numel() == 0:
            return 0.0
        q75 = torch.quantile(values, 0.75)
        q25 = torch.quantile(values, 0.25)
        return float(torch.clamp(q75 - q25, min=0.0).item())

    def _resolve_effective_alpha(
        self,
        uncertainty_terms: np.ndarray,
        coverage_terms: Optional[np.ndarray]
    ) -> float:
        """
        解析当前轮静态不确定性项实际使用的 alpha。

        当 `self.alpha` 是数值时，函数直接返回该固定权重，保持旧行为；当
        `self.alpha is None` 时，函数用当前候选集上的 IQR 比值做自动尺度校准：
        `alpha_t = IQR(logD) / IQR(uncertainty)`。这里不乘 coverage_ratio，因为
        coverage_ratio 本身被保留为原公式中的阶段性门控：覆盖率低时覆盖项主导，
        覆盖率高时不确定性项逐步增强。

        参数:
            uncertainty_terms: 当前候选集上的静态不确定性项，未乘 alpha 或
                coverage_ratio。
            coverage_terms: 当前候选集上的 `log(max(D_i, 1))` 覆盖项；为 None 时
                无法自动校准，将回退到 1.0。

        返回:
            float: 当前轮实际使用的 alpha。
        """
        if self.alpha is not None:
            return float(self.alpha)

        if coverage_terms is None:
            print(f"[{self.name}] alpha=None 但未提供覆盖项尺度，回退 alpha=0.5")
            return 0.5

        uncertainty_iqr = self._compute_iqr(uncertainty_terms)
        coverage_iqr = self._compute_iqr(coverage_terms)
        if uncertainty_iqr <= 1e-12 or coverage_iqr <= 1e-12:
            print(
                f"[{self.name}] 自动alpha回退: "
                f"IQR_uncertainty={uncertainty_iqr:.6f}, "
                f"IQR_logD={coverage_iqr:.6f}, alpha=0.5"
            )
            return 0.5

        effective_alpha = float(coverage_iqr / uncertainty_iqr)
        print(
            f"[{self.name}] 自动alpha: "
            f"IQR_logD={coverage_iqr:.6f}, "
            f"IQR_uncertainty={uncertainty_iqr:.6f}, "
            f"alpha={effective_alpha:.6f}"
        )
        return effective_alpha

    def _resolve_effective_alpha_torch(
        self,
        uncertainty_terms: torch.Tensor,
        coverage_terms: Optional[torch.Tensor]
    ) -> float:
        """
        解析当前轮静态不确定性项实际使用的 alpha。
        """
        if self.alpha is not None:
            return float(self.alpha)

        if coverage_terms is None:
            print(f"[{self.name}] alpha=None 但未提供覆盖项尺度，回退 alpha=0.5")
            return 0.5

        uncertainty_iqr = self._compute_iqr_torch(uncertainty_terms)
        coverage_iqr = self._compute_iqr_torch(coverage_terms)
        if uncertainty_iqr <= 1e-12 or coverage_iqr <= 1e-12:
            print(
                f"[{self.name}] 自动alpha回退: "
                f"IQR_uncertainty={uncertainty_iqr:.6f}, "
                f"IQR_logD={coverage_iqr:.6f}, alpha=0.5"
            )
            return 0.5

        effective_alpha = float(coverage_iqr / uncertainty_iqr)
        print(
            f"[{self.name}] 自动alpha: "
            f"IQR_logD={coverage_iqr:.6f}, "
            f"IQR_uncertainty={uncertainty_iqr:.6f}, "
            f"alpha={effective_alpha:.6f}"
        )
        return effective_alpha

    def _compute_log_coverage_terms(
        self,
        candidate_distance_matrix: np.ndarray,
        candidate_radii: np.ndarray
    ) -> np.ndarray:
        """
        计算当前候选集初始状态下每个候选点的 `logD` 覆盖项。

        自动 alpha 需要在进入贪心循环前估计覆盖项与不确定性项的相对尺度。这里
        使用过滤后候选集的初始覆盖数：
        `D_i = |{x_j: d(x_i, x_j) <= r_i}|`，并返回 `log(max(D_i, 1))`。贪心过程
        中仍会按原逻辑动态更新剩余候选集上的 `D(x)`，本函数只负责尺度校准。

        参数:
            candidate_distance_matrix: 候选集内部距离矩阵，形状为
                `(num_candidates, num_candidates)`。
            candidate_radii: 与候选点顺序一致的半径数组。

        返回:
            np.ndarray: 与候选点顺序一致的 `logD` 覆盖项数组。

        异常:
            ValueError: 当距离矩阵和半径数组形状不匹配时抛出。
        """
        candidate_radii = np.asarray(candidate_radii, dtype=np.float64)
        if candidate_distance_matrix.shape[0] != candidate_radii.shape[0]:
            raise ValueError("候选距离矩阵与 candidate_radii 长度不一致。")
        if candidate_distance_matrix.shape[0] == 0:
            return np.empty((0,), dtype=np.float64)

        covered_counts = np.sum(
            candidate_distance_matrix <= candidate_radii[:, None],
            axis=1
        )
        return np.log(np.maximum(covered_counts, 1))

    def _compute_log_coverage_terms_from_full_distance(
        self,
        distance_matrix: np.ndarray,
        candidate_indices: np.ndarray,
        candidate_radii: np.ndarray
    ) -> np.ndarray:
        """
        基于完整距离矩阵分块计算候选集初始 `logD` 覆盖项。

        该函数与 `_compute_log_coverage_terms(...)` 的数学定义完全一致，区别只在
        于实现方式：它不再先显式构造
        `candidate_distance_matrix = distance_matrix[np.ix_(candidate_indices, candidate_indices)]`
        这一整块候选子矩阵，而是直接从完整距离矩阵中按块读取当前需要的候选行，
        再对同一批候选列做覆盖判断。这样可以避免：
        1. 在大候选集上额外分配一个 `O(m^2)` 的候选距离子矩阵副本；
        2. 在只需要逐行统计覆盖数时，先做一整次完整子矩阵拷贝。

        覆盖项定义保持不变：
        `logD_i = log(max(|{x_j in candidate_set : d(x_i, x_j) <= r_i}|, 1))`

        参数:
            distance_matrix (np.ndarray): 完整训练集距离矩阵，形状为 `(N, N)`。
            candidate_indices (np.ndarray): 候选样本在完整训练集中的全局索引数组，
                长度为 `m`，顺序定义了当前候选集的“局部位置”。
            candidate_radii (np.ndarray): 与 `candidate_indices` 顺序一致的候选半径
                数组，长度为 `m`。

        返回:
            np.ndarray: 与 `candidate_indices` 顺序一致的一维 `logD` 覆盖项数组，
                长度为 `m`。

        异常:
            ValueError:
                - 当 `distance_matrix` 不是方阵时抛出；
                - 当 `candidate_indices` 与 `candidate_radii` 长度不一致时抛出。
        """
        candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
        candidate_radii = np.asarray(candidate_radii, dtype=np.float64)

        if distance_matrix.shape[0] != distance_matrix.shape[1]:
            raise ValueError("distance_matrix 必须是方阵。")
        if candidate_indices.shape[0] != candidate_radii.shape[0]:
            raise ValueError("candidate_indices 与 candidate_radii 长度必须一致。")
        if candidate_indices.size == 0:
            return np.empty((0,), dtype=np.float64)

        covered_counts = np.empty(candidate_indices.shape[0], dtype=np.int64)
        block_size = max(1, min(int(self.inference_batch_size), candidate_indices.size))

        # 分块读取“候选行 x 全候选列”距离，避免先物化整块候选子矩阵，同时保持
        # 与旧实现完全相同的逐行覆盖计数语义。
        for block_start in range(0, candidate_indices.size, block_size):
            block_end = min(block_start + block_size, candidate_indices.size)
            block_indices = candidate_indices[block_start:block_end]
            distance_block = distance_matrix[np.ix_(block_indices, candidate_indices)]
            covered_counts[block_start:block_end] = np.count_nonzero(
                distance_block <= candidate_radii[block_start:block_end][:, None],
                axis=1
            )

        return np.log(np.maximum(covered_counts, 1))

    def _compute_h_from_probabilities(
        self,
        probabilities: np.ndarray
    ) -> np.ndarray:
        """
        根据概率向量批量计算每个候选点对应的 H 最大特征值 h。

        该函数把后续轮次中“由预测概率推导半径估计所需 h 值”的计算集中起来，
        便于与概率预测过程解耦，避免对同一批候选点重复做模型前向。其处理流程为：
        1. 对每个候选点的概率向量 p 构造矩阵 H = diag(p) - p p^T；
        2. 利用 H 为实对称矩阵这一性质，使用 `np.linalg.eigvalsh(...)`
           计算特征值，减少不必要的复数开销；
        3. 取绝对值最大的特征值作为 h，并用 `self.h_min` 做下界截断，
           保持与原始实现一致的数值语义。

        参数:
            probabilities: 候选样本的预测概率矩阵，形状为
                `(num_candidates, num_classes)`。矩阵中的每一行对应一个候选点
                在当前模型下的类别概率分布。

        返回:
            np.ndarray: 与 `probabilities` 按行一一对应的 h 值数组。
        """
        if probabilities.size == 0:
            return np.empty((0,), dtype=np.float64)

        h_values = []
        for p in probabilities:
            # 计算H = diag(p) - p^T p
            diag_p = np.diag(p)
            ppT = np.outer(p, p)
            H = diag_p - ppT

            # H 是实对称矩阵，使用 eigvalsh 能减少不必要的数值开销。
            eigenvalues = np.linalg.eigvalsh(H)
            h = np.max(np.abs(eigenvalues))
            h = max(h, self.h_min)  # 设置下界
            h_values.append(h)
        
        return np.array(h_values)

    def _sanitize_probabilities_torch(
        self,
        probabilities: torch.Tensor,
        context: str
    ) -> torch.Tensor:
        """
        将概率张量修复为有限、非负且逐行归一化的形式，避免线代算子因坏值失败。
        """
        probabilities = probabilities.to(dtype=torch.float64)
        if probabilities.numel() == 0:
            return probabilities

        finite_mask = torch.isfinite(probabilities)
        has_negative = bool(torch.any(probabilities < 0.0).item())
        row_sums = torch.sum(probabilities, dim=1, keepdim=True)
        invalid_rows = (~torch.isfinite(row_sums)) | (row_sums <= 0.0)
        needs_renorm = bool(
            torch.any(torch.abs(row_sums - 1.0) > 1e-8).item()
        )
        if bool(torch.all(finite_mask).item()) and (not has_negative) and (not bool(torch.any(invalid_rows).item())) and (not needs_renorm):
            return probabilities

        if not bool(torch.all(finite_mask).item()):
            invalid_count = int((~finite_mask).sum().item())
            print(
                f"[{self.name}] {context}: 概率张量中发现 {invalid_count} 个非有限值，"
                "已执行 nan/inf 修复。"
            )
            probabilities = torch.nan_to_num(
                probabilities,
                nan=0.0,
                posinf=0.0,
                neginf=0.0
            )

        if has_negative:
            negative_count = int((probabilities < 0.0).sum().item())
            print(
                f"[{self.name}] {context}: 概率张量中发现 {negative_count} 个负值，"
                "已裁剪到 0。"
            )
            probabilities = torch.clamp(probabilities, min=0.0)

        row_sums = torch.sum(probabilities, dim=1, keepdim=True)
        invalid_rows = (~torch.isfinite(row_sums)) | (row_sums <= 0.0)
        if bool(torch.any(invalid_rows).item()):
            invalid_row_count = int(invalid_rows.sum().item())
            print(
                f"[{self.name}] {context}: 概率张量中发现 {invalid_row_count} 行无法归一化，"
                "已回退为均匀分布。"
            )
            probabilities = probabilities.clone()
            probabilities[invalid_rows.expand_as(probabilities)] = 0.0
            probabilities[invalid_rows.squeeze(1)] = (
                1.0 / probabilities.shape[1]
            )
            row_sums = torch.sum(probabilities, dim=1, keepdim=True)

        if needs_renorm and not bool(torch.any(invalid_rows).item()):
            print(
                f"[{self.name}] {context}: 概率张量行和偏离 1，已重新归一化。"
            )
        return probabilities / torch.clamp(row_sums, min=1e-12)

    def _compute_hessian_spectral_norms_from_probabilities_torch(
        self,
        probabilities: torch.Tensor,
        context: str
    ) -> torch.Tensor:
        """
        根据概率张量计算 Hessian 谱范数，并在 CUDA 线代失败时回退到 CPU。
        """
        probabilities = self._sanitize_probabilities_torch(probabilities, context)
        if probabilities.numel() == 0:
            return torch.empty((0,), dtype=torch.float64, device=probabilities.device)

        hessian = (
            torch.diag_embed(probabilities)
            - probabilities.unsqueeze(2) * probabilities.unsqueeze(1)
        ).contiguous()
        try:
            eigenvalues = torch.linalg.eigvalsh(hessian)
        except RuntimeError as exc:
            error_message = str(exc).splitlines()[0]
            print(
                f"[{self.name}] {context}: CUDA eigvalsh 失败，已回退到 CPU。"
                f" error={error_message}"
            )
            eigenvalues = torch.linalg.eigvalsh(hessian.cpu()).to(hessian.device)

        h_values = eigenvalues.abs().amax(dim=1)
        if not bool(torch.all(torch.isfinite(h_values)).item()):
            invalid_count = int((~torch.isfinite(h_values)).sum().item())
            print(
                f"[{self.name}] {context}: Hessian 谱范数中发现 {invalid_count} 个非有限值，"
                "已回退到 h_min。"
            )
            h_values = torch.nan_to_num(
                h_values,
                nan=self.h_min,
                posinf=self.h_min,
                neginf=self.h_min
            )
        return h_values.clamp_min(self.h_min)

    def _compute_h_from_probabilities_torch(
        self,
        probabilities: torch.Tensor
    ) -> torch.Tensor:
        """
        根据概率向量批量计算每个候选点对应的 H 最大特征值 h，并保留在当前设备上。
        """
        if probabilities.numel() == 0:
            return torch.empty((0,), dtype=torch.float64, device=probabilities.device)

        return self._compute_hessian_spectral_norms_from_probabilities_torch(
            probabilities=probabilities,
            context="候选点 h 计算"
        )

    def _evaluate_active_candidate_scores(
        self,
        candidate_distance_matrix: np.ndarray,
        candidate_radii: np.ndarray,
        static_score_terms: np.ndarray,
        active_positions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray]:
        """
        分块批量评估当前活跃候选集内每个候选点的覆盖数与采样分数。

        该函数专门用于替代后续轮次贪心循环中“逐候选点 Python for 循环”的
        计算热点，但不改变原始采样语义。它严格保留如下定义：
        1. 当前可参与竞争的候选点集合由 `active_positions` 指定；
        2. 对任意候选点 `x_i`，覆盖数 `D_i` 定义为：
           `|{x_j in active_positions : d(x_i, x_j) <= r_i}|`；
        3. 分数定义保持为：
           `score_i = static_score_terms[i] + log(max(D_i, 1))`；
        4. 返回的 `score_by_active_position[k]` 与
           `covered_count_by_active_position[k]` 始终对应
           `active_positions[k]`，顺序不会被重排；
        5. 为减少大候选集上的瞬时内存占用，函数按块读取
           `candidate_distance_matrix[np.ix_(rows, active_positions)]`，每块内部做
           向量化比较与求和，再把结果写回最终数组。

        这样做的收益仅在于减少 Python 层逐点循环开销。对于同一组
        `candidate_distance_matrix`、`candidate_radii`、`static_score_terms` 与
        `active_positions`，数学结果与逐点实现保持一致；不同仅可能来自你已明确
        放宽的浮点细节与并列分数时的 tie-break。

        参数:
            candidate_distance_matrix (np.ndarray): 候选集内部距离矩阵，行列顺序与
                候选局部位置严格对齐，形状为 `(num_candidates, num_candidates)`。
            candidate_radii (np.ndarray): 候选点生效半径数组，长度为
                `num_candidates`，与距离矩阵行号一一对应。
            static_score_terms (np.ndarray): 候选点静态评分项数组，长度为
                `num_candidates`，与距离矩阵行号一一对应。
            active_positions (np.ndarray): 当前仍处于活跃状态的候选“局部位置”
                数组。它不是全局样本索引，而是 `candidate_distance_matrix`
                的行列号子集。

        返回:
            Tuple[np.ndarray, np.ndarray, int, np.ndarray]:
                - `score_by_active_position`: 当前活跃候选的分数数组，长度等于
                  `len(active_positions)`，顺序与 `active_positions` 完全一致；
                - `covered_count_by_active_position`: 当前活跃候选的覆盖数数组，
                  顺序同上；
                - `best_local_position`: 当前分数最高候选的局部位置；
                - `best_covered_positions`: 该最优候选在当前活跃集合中覆盖到的
                  所有局部位置数组。

        异常:
            ValueError:
                - 当 `active_positions` 不是一维数组时抛出；
                - 当距离矩阵、半径数组、静态项数组的候选长度不一致时抛出。
        """
        active_positions = np.asarray(active_positions, dtype=np.int64)
        candidate_radii = np.asarray(candidate_radii, dtype=np.float64)
        static_score_terms = np.asarray(static_score_terms, dtype=np.float64)

        if active_positions.ndim != 1:
            raise ValueError("active_positions 必须是一维数组。")
        candidate_count = candidate_distance_matrix.shape[0]
        if candidate_distance_matrix.shape[1] != candidate_count:
            raise ValueError("candidate_distance_matrix 必须是方阵。")
        if candidate_radii.shape[0] != candidate_count:
            raise ValueError("candidate_radii 长度必须与候选距离矩阵边长一致。")
        if static_score_terms.shape[0] != candidate_count:
            raise ValueError("static_score_terms 长度必须与候选距离矩阵边长一致。")
        if active_positions.size == 0:
            return (
                np.empty((0,), dtype=np.float64),
                np.empty((0,), dtype=np.int64),
                -1,
                np.empty((0,), dtype=np.int64),
            )

        score_by_active_position = np.empty(
            active_positions.shape[0],
            dtype=np.float64
        )
        covered_count_by_active_position = np.empty(
            active_positions.shape[0],
            dtype=np.int64
        )
        block_size = max(1, min(int(self.inference_batch_size), active_positions.size))

        # 分块读取“候选行 x 当前活跃列”子矩阵，避免在大候选集上一次性构造
        # 巨大的布尔覆盖矩阵，同时仍保持块内向量化。
        for block_start in range(0, active_positions.size, block_size):
            block_end = min(block_start + block_size, active_positions.size)
            row_positions = active_positions[block_start:block_end]
            distance_block = candidate_distance_matrix[np.ix_(row_positions, active_positions)]
            covered_mask_block = (
                distance_block <= candidate_radii[row_positions][:, None]
            )
            covered_counts_block = np.maximum(
                np.count_nonzero(covered_mask_block, axis=1),
                1
            ).astype(np.int64, copy=False)

            covered_count_by_active_position[block_start:block_end] = (
                covered_counts_block
            )
            score_by_active_position[block_start:block_end] = (
                static_score_terms[row_positions] + np.log(covered_counts_block)
            )

        best_active_offset = int(np.argmax(score_by_active_position))
        best_local_position = int(active_positions[best_active_offset])
        best_covered_mask = (
            candidate_distance_matrix[best_local_position, active_positions]
            <= candidate_radii[best_local_position]
        )
        best_covered_positions = active_positions[best_covered_mask]
        return (
            score_by_active_position,
            covered_count_by_active_position,
            best_local_position,
            best_covered_positions,
        )

    def _evaluate_active_candidate_scores_from_full_distance(
        self,
        distance_matrix: np.ndarray,
        candidate_indices: np.ndarray,
        candidate_radii: np.ndarray,
        static_score_terms: np.ndarray,
        active_positions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, int, int, np.ndarray]:
        """
        基于完整距离矩阵分块评估当前活跃候选的覆盖数与采样分数。

        该函数服务于 `our_method_margin` 与 `our_method_margin_weighted` 的后续轮次
        贪心采样，实现目标是消除显式 `candidate_distance_matrix` 子矩阵拷贝，同时
        保持原打分规则不变。与旧实现相比，唯一变化是数据访问路径：
        1. 旧实现先构造候选子矩阵，再在该子矩阵上读取
           `candidate_distance_matrix[local_position, active_positions]`；
        2. 新实现直接使用 `candidate_indices` 把当前活跃候选映射回完整距离矩阵，
           分块读取 `distance_matrix[np.ix_(rows_global, active_global)]`；
        3. 在每个块内仍按完全相同的公式计算覆盖数与分数。

        数学定义保持为：
        - `D_i = |{x_j in active_set : d(x_i, x_j) <= r_i}|`
        - `score_i = static_score_terms[i] + log(max(D_i, 1))`

        参数:
            distance_matrix (np.ndarray): 完整训练集距离矩阵，形状为 `(N, N)`。
            candidate_indices (np.ndarray): 候选样本在完整训练集中的全局索引数组，
                长度为 `m`，顺序定义候选“局部位置”。
            candidate_radii (np.ndarray): 与候选局部位置对齐的半径数组，长度为 `m`。
            static_score_terms (np.ndarray): 与候选局部位置对齐的静态评分项数组，
                长度为 `m`。
            active_positions (np.ndarray): 当前仍处于活跃状态的候选局部位置数组。

        返回:
            Tuple[np.ndarray, np.ndarray, int, int, np.ndarray]:
                - `score_by_active_position`: 与 `active_positions` 顺序一致的分数数组；
                - `covered_count_by_active_position`: 与 `active_positions` 顺序一致的
                  覆盖数数组；
                - `best_active_offset`: 当前最优点在 `active_positions` 中的偏移位置；
                - `best_local_position`: 当前最优点在候选集内的局部位置；
                - `best_covered_positions`: 该最优点在当前活跃集合中覆盖到的局部位置。

        异常:
            ValueError:
                - 当 `distance_matrix` 不是方阵时抛出；
                - 当候选数组长度不匹配时抛出；
                - 当 `active_positions` 不是一维数组时抛出。
        """
        candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
        candidate_radii = np.asarray(candidate_radii, dtype=np.float64)
        static_score_terms = np.asarray(static_score_terms, dtype=np.float64)
        active_positions = np.asarray(active_positions, dtype=np.int64)

        if distance_matrix.shape[0] != distance_matrix.shape[1]:
            raise ValueError("distance_matrix 必须是方阵。")
        if candidate_indices.shape[0] != candidate_radii.shape[0]:
            raise ValueError("candidate_indices 与 candidate_radii 长度必须一致。")
        if candidate_indices.shape[0] != static_score_terms.shape[0]:
            raise ValueError("candidate_indices 与 static_score_terms 长度必须一致。")
        if active_positions.ndim != 1:
            raise ValueError("active_positions 必须是一维数组。")
        if active_positions.size == 0:
            return (
                np.empty((0,), dtype=np.float64),
                np.empty((0,), dtype=np.int64),
                -1,
                -1,
                np.empty((0,), dtype=np.int64),
            )

        score_by_active_position = np.empty(active_positions.shape[0], dtype=np.float64)
        covered_count_by_active_position = np.empty(active_positions.shape[0], dtype=np.int64)
        active_global_indices = candidate_indices[active_positions]
        block_size = max(1, min(int(self.inference_batch_size), active_positions.size))

        # 分块读取完整距离矩阵中的“活跃候选行 x 活跃候选列”，避免先拷贝出
        # 整块候选子矩阵，同时保留与旧实现完全一致的覆盖判断公式。
        for block_start in range(0, active_positions.size, block_size):
            block_end = min(block_start + block_size, active_positions.size)
            row_positions = active_positions[block_start:block_end]
            row_global_indices = candidate_indices[row_positions]
            distance_block = distance_matrix[np.ix_(row_global_indices, active_global_indices)]
            covered_mask_block = (
                distance_block <= candidate_radii[row_positions][:, None]
            )
            covered_counts_block = np.maximum(
                np.count_nonzero(covered_mask_block, axis=1),
                1
            ).astype(np.int64, copy=False)
            covered_count_by_active_position[block_start:block_end] = covered_counts_block
            score_by_active_position[block_start:block_end] = (
                static_score_terms[row_positions] + np.log(covered_counts_block)
            )

        best_active_offset = int(np.argmax(score_by_active_position))
        best_local_position = int(active_positions[best_active_offset])
        best_covered_mask = (
            distance_matrix[
                candidate_indices[best_local_position],
                active_global_indices
            ] <= candidate_radii[best_local_position]
        )
        best_covered_positions = active_positions[best_covered_mask]
        return (
            score_by_active_position,
            covered_count_by_active_position,
            best_active_offset,
            best_local_position,
            best_covered_positions,
        )

    def _compute_eta_p_distances(
        self,
        probabilities: np.ndarray
    ) -> np.ndarray:
        """
        批量计算每个候选点的 |eta - p|_2 距离。

        在后续轮次的打分阶段，`|eta-p|_2` 只依赖当前模型对候选点的预测概率，
        与贪心采样过程中“哪些点已经被移除”无关，因此适合在进入贪心循环前一次性
        计算并缓存。该函数执行的数学操作与 `_compute_eta_p_distance(...)`
        完全一致，只是把逐样本计算改成向量化实现：
        1. 对每一行概率向量 p，找到最大分量对应的位置；
        2. 用该位置对应的 one-hot 向量表示 eta；
        3. 计算每一行的欧氏距离 `||eta - p||_2`；
        4. 返回与输入行顺序严格对齐的一维距离数组，供后续打分直接复用。

        参数:
            probabilities: 候选样本的预测概率矩阵，形状为
                `(num_candidates, num_classes)`。

        返回:
            np.ndarray: 每个候选点对应的 `|eta-p|_2` 距离，形状为
                `(num_candidates,)`。
        """
        if probabilities.size == 0:
            return np.empty((0,), dtype=np.float64)

        max_indices = np.argmax(probabilities, axis=1)
        row_indices = np.arange(probabilities.shape[0])
        max_probabilities = probabilities[row_indices, max_indices]
        probability_norm_squares = np.sum(probabilities * probabilities, axis=1)
        distance_squares = 1.0 + probability_norm_squares - 2.0 * max_probabilities
        return np.sqrt(np.maximum(distance_squares, 0.0))

    def _compute_eta_p_distances_torch(
        self,
        probabilities: torch.Tensor
    ) -> torch.Tensor:
        """
        批量计算每个候选点的 |eta - p|_2 距离，并保留在当前设备上。
        """
        if probabilities.numel() == 0:
            return torch.empty((0,), dtype=torch.float64, device=probabilities.device)

        probabilities = probabilities.to(dtype=torch.float64)
        max_probabilities = torch.max(probabilities, dim=1).values
        probability_norm_squares = torch.sum(probabilities * probabilities, dim=1)
        distance_squares = 1.0 + probability_norm_squares - 2.0 * max_probabilities
        return torch.sqrt(torch.clamp(distance_squares, min=0.0))

    def _compute_static_score_terms(
        self,
        probabilities: np.ndarray,
        coverage_ratio: float,
        coverage_terms: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        根据候选点预测概率批量计算后续轮次中“与贪心删除无关”的静态评分项。

        该函数的作用是把后续轮次评分公式里只依赖候选点自身预测结果、不会随着
        贪心迭代而变化的部分集中到一个可复用的扩展点中。当前预算版本策略使用的
        是：
        `alpha * coverage_ratio * ln(|eta-p|_2)`。

        之所以单独抽成函数，是为了满足两类需求：
        1. 让当前策略在进入贪心循环前一次性完成静态项计算，避免在循环中重复做
           相同的对数运算；
        2. 允许派生策略在不改动其余覆盖、半径估计和贪心逻辑的前提下，只通过重写
           本函数来替换“不确定性”项的定义。

        参数:
            probabilities: 候选样本的预测概率矩阵，形状为
                `(num_candidates, num_classes)`。
            coverage_ratio: 当前轮次覆盖率。该值由已标注点覆盖结果决定，在本轮
                贪心采样过程中保持不变，因此可以直接参与静态项预计算。
            coverage_terms: 当前候选集初始 `logD` 覆盖项数组，仅在 `alpha=None`
                自动尺度校准时使用。

        返回:
            np.ndarray: 与 `probabilities` 行顺序对齐的一维静态评分项数组，形状为
                `(num_candidates,)`。
        """
        eta_p_distances = np.maximum(
            self._compute_eta_p_distances(probabilities),
            1e-10
        )
        uncertainty_terms = np.log(eta_p_distances)
        effective_alpha = self._resolve_effective_alpha(
            uncertainty_terms=uncertainty_terms,
            coverage_terms=coverage_terms
        )
        return effective_alpha * coverage_ratio * uncertainty_terms

    def _compute_static_score_terms_torch(
        self,
        probabilities: torch.Tensor,
        coverage_ratio: float,
        coverage_terms: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        根据候选点预测概率批量计算后续轮次中的静态评分项，并保留在当前设备上。
        """
        eta_p_distances = torch.clamp(
            self._compute_eta_p_distances_torch(probabilities),
            min=1e-10
        )
        uncertainty_terms = torch.log(eta_p_distances)
        effective_alpha = self._resolve_effective_alpha_torch(
            uncertainty_terms=uncertainty_terms,
            coverage_terms=coverage_terms
        )
        return uncertainty_terms * (effective_alpha * coverage_ratio)

    def _greedy_selection(
        self,
        unlabeled_indices: List[int],
        distance_matrix: np.ndarray,
        radii: np.ndarray,
        query_budget: int
    ) -> Tuple[List[int], List[int]]:
        """
        贪心采样：选择能覆盖最多点的点，直到达到查询预算
        
        参数:
            unlabeled_indices: 无标注点的索引列表
            distance_matrix: 距离矩阵
            radii: 每个无标注点的预估半径（与unlabeled_indices一一对应）
            query_budget: 查询预算，最多选择的样本数量
            
        返回:
            Tuple[List[int], List[int]]: (选择的索引列表, 被覆盖的索引列表)
        """
        selected_indices = []
        covered_indices: List[int] = []
        if not unlabeled_indices:
            return selected_indices, covered_indices

        local_indices = np.asarray(unlabeled_indices, dtype=np.int64)
        local_radii = np.asarray(radii, dtype=np.float64)
        (
            covered_positions_by_candidate,
            reverse_cover_rows,
            initial_covered_counts,
        ) = self._prepare_candidate_cover_data_from_full_distance(
            distance_matrix=distance_matrix,
            candidate_indices=local_indices,
            candidate_radii=local_radii
        )
        selected_local_positions = self._greedy_select_candidate_positions_from_cover_data(
            candidate_indices=local_indices,
            static_score_terms=np.zeros(local_indices.shape[0], dtype=np.float64),
            covered_positions_by_candidate=covered_positions_by_candidate,
            reverse_cover_rows=reverse_cover_rows,
            initial_covered_counts=initial_covered_counts,
            query_budget=query_budget,
            progress_desc="[OurMethodBudgetVersion] 第一轮选择进度",
            log_prefix="OurMethodBudgetVersion"
        )
        if not selected_local_positions:
            return selected_indices, covered_indices

        selected_local_positions_array = np.asarray(
            selected_local_positions,
            dtype=np.int64
        )
        selected_indices = local_indices[selected_local_positions_array].astype(
            np.int64,
            copy=False
        ).tolist()
        covered_positions = np.concatenate([
            covered_positions_by_candidate[int(local_position)]
            for local_position in selected_local_positions_array
        ])
        covered_indices = local_indices[np.unique(covered_positions)].astype(
            np.int64,
            copy=False
        ).tolist()
        return selected_indices, covered_indices

    def _first_round_selection_cuda_common(
        self,
        dataset: Any,
        device: torch.device,
        distance_matrix: np.ndarray,
        labeled_indices: List[int],
        unlabeled_indices: List[int],
        spectral_norm_product: Optional[float],
        log_prefix: str
    ) -> Tuple[List[int], List[float], List[float]]:
        """
        在 CUDA 环境下执行第一轮采样公共流程。
        """
        distance_tensor = self._get_cuda_distance_matrix(
            distance_matrix=distance_matrix,
            device=device
        )
        round_indices = list(labeled_indices) + list(unlabeled_indices)
        fixed_round_radius: Optional[float] = None
        radius_by_global: Optional[torch.Tensor] = None
        round_indices_tensor = torch.as_tensor(
            round_indices,
            dtype=torch.long,
            device=device
        )
        if round_indices_tensor.numel() > 0:
            if self.max_theoretical_radius is not None:
                fixed_round_radius = float(self.max_theoretical_radius)
            elif spectral_norm_product is not None:
                fixed_round_radius = float(
                    self._compute_max_theoretical_radius(spectral_norm_product)
                )
            else:
                round_max_radii = self._compute_max_theoretical_radii_for_indices_torch(
                    round_indices,
                    spectral_norm_product,
                    device
                )
                radius_by_global = torch.empty(
                    (distance_tensor.shape[0],),
                    dtype=torch.float64,
                    device=device
                )
                radius_by_global[round_indices_tensor] = round_max_radii

            if fixed_round_radius is not None:
                round_radius_min = fixed_round_radius
                round_radius_max = fixed_round_radius
                round_radius_mean = fixed_round_radius
            else:
                round_radius_min = float(torch.min(round_max_radii).item())
                round_radius_max = float(torch.max(round_max_radii).item())
                round_radius_mean = float(torch.mean(round_max_radii).item())
            print(
                f"[{log_prefix}] 最大理论半径统计: "
                f"min={round_radius_min:.6f}, "
                f"max={round_radius_max:.6f}, "
                f"mean={round_radius_mean:.6f}"
            )

        if fixed_round_radius is not None:
            current_labeled_radii_tensor = torch.full(
                (len(labeled_indices),),
                fixed_round_radius,
                dtype=torch.float64,
                device=device
            )
        elif labeled_indices:
            labeled_indices_tensor = torch.as_tensor(
                labeled_indices,
                dtype=torch.long,
                device=device
            )
            current_labeled_radii_tensor = radius_by_global.index_select(
                0,
                labeled_indices_tensor
            )
        else:
            current_labeled_radii_tensor = torch.empty(
                (0,),
                dtype=torch.float64,
                device=device
            )

        if labeled_indices:
            _, covered_mask = self._compute_coverage_torch(
                distance_matrix=distance_tensor,
                labeled_indices=labeled_indices,
                radii=current_labeled_radii_tensor
            )
            filtered_unlabeled_tensor = self._filter_unlabeled_indices_torch(
                unlabeled_indices=unlabeled_indices,
                covered_mask=covered_mask
            )
            print(f"[{log_prefix}] 原始无标注池大小: {len(unlabeled_indices)}")
            print(
                f"[{log_prefix}] 被已标注点覆盖的点数: "
                f"{len(unlabeled_indices) - int(filtered_unlabeled_tensor.numel())}"
            )
            print(
                f"[{log_prefix}] 过滤后无标注池大小: "
                f"{int(filtered_unlabeled_tensor.numel())}"
            )
        else:
            filtered_unlabeled_tensor = torch.as_tensor(
                unlabeled_indices,
                dtype=torch.long,
                device=device
            )
            print(
                f"[{log_prefix}] 无标注池大小: "
                f"{int(filtered_unlabeled_tensor.numel())}"
            )

        selected_indices: List[int] = []
        if filtered_unlabeled_tensor.numel() > 0:
            if fixed_round_radius is not None:
                radii_tensor = torch.full(
                    (filtered_unlabeled_tensor.numel(),),
                    fixed_round_radius,
                    dtype=torch.float64,
                    device=device
                )
            else:
                radii_tensor = radius_by_global.index_select(
                    0,
                    filtered_unlabeled_tensor
                )
            selected_indices, covered_indices, _ = self._greedy_selection_torch(
                unlabeled_indices=filtered_unlabeled_tensor,
                distance_matrix=distance_tensor,
                radii=radii_tensor,
                query_budget=self.query_budget,
                progress_desc=f"[{log_prefix}] 第一轮选择进度",
                log_prefix=log_prefix
            )
            print(
                f"[{log_prefix}] 第一轮选择的标注点数量: {len(selected_indices)}"
            )
            print(
                f"[{log_prefix}] 第一轮覆盖的点数量: {len(covered_indices)}"
            )
        else:
            print(f"[{log_prefix}] 无标注池为空，无需选择样本")

        current_labeled_radii = current_labeled_radii_tensor.detach().cpu().tolist()
        if fixed_round_radius is not None:
            selected_radii = torch.full(
                (len(selected_indices),),
                fixed_round_radius,
                dtype=torch.float64,
                device=device
            ).detach().cpu().tolist()
        elif selected_indices:
            selected_indices_tensor = torch.as_tensor(
                selected_indices,
                dtype=torch.long,
                device=device
            )
            selected_radii = radius_by_global.index_select(
                0,
                selected_indices_tensor
            ).detach().cpu().tolist()
        else:
            selected_radii = []

        _ = dataset
        return selected_indices, current_labeled_radii, selected_radii

    def _subsequent_round_selection_cuda_common(
        self,
        model: torch.nn.Module,
        dataset: Any,
        device: torch.device,
        distance_matrix: np.ndarray,
        labeled_indices: List[int],
        unlabeled_indices: List[int],
        spectral_norm_product: Optional[float],
        log_prefix: str
    ) -> Tuple[List[int], List[float], List[float]]:
        """
        在 CUDA 环境下执行后续轮次采样公共流程。
        """
        distance_tensor = self._get_cuda_distance_matrix(
            distance_matrix=distance_matrix,
            device=device
        )
        max_theoretical_radius = self._compute_round_max_theoretical_radius_torch(
            list(labeled_indices) + list(unlabeled_indices),
            spectral_norm_product,
            device
        )
        labeled_points = self._build_labeled_points(dataset, labeled_indices)
        raw_labeled_radii = self._compute_labeled_radii_torch(
            labeled_points=labeled_points,
            labeled_indices=labeled_indices,
            model=model,
            spectral_norm_product=spectral_norm_product,
            device=device
        )

        coverage_ratio, covered_mask = self._compute_coverage_torch(
            distance_tensor,
            labeled_indices,
            raw_labeled_radii
        )
        self.last_coverage = coverage_ratio
        print(f"[{log_prefix}] 当前轮次覆盖率: {coverage_ratio:.6f}")

        effective_labeled_radii = raw_labeled_radii
        normalization_applied = False
        normalization_scale_ratio = 1.0
        normalization_pivot_radius = (
            float(torch.max(raw_labeled_radii).item())
            if raw_labeled_radii.numel() > 0 else 0.0
        )
        if self.enable_early_radius_normalization:
            (
                effective_labeled_radii,
                normalization_applied,
                normalization_scale_ratio,
                normalization_pivot_radius,
            ) = self._normalize_radii_for_early_rounds_torch(
                radii=raw_labeled_radii,
                coverage_ratio=coverage_ratio,
                max_theoretical_radius=max_theoretical_radius,
            )
            if normalization_applied:
                effective_labeled_radii = self._clip_radii_to_max_theoretical_torch(
                    radii=effective_labeled_radii,
                    indices=labeled_indices,
                    spectral_norm_product=spectral_norm_product,
                    device=device
                )
                normalized_coverage_ratio, covered_mask = self._compute_coverage_torch(
                    distance_tensor,
                    labeled_indices,
                    effective_labeled_radii,
                )
                coverage_ratio = normalized_coverage_ratio
                self.last_coverage = coverage_ratio
                print(
                    f"[{log_prefix}] 早期半径归一化已触发: "
                    f"T={self.early_radius_normalization_threshold:.6f}, "
                    f"k={self.early_radius_normalization_percentile_k:.2f}, "
                    f"q_k={normalization_pivot_radius:.6f}, "
                    f"r={normalization_scale_ratio:.6f}, "
                    f"max_radius={max_theoretical_radius:.6f}, "
                    f"normalized_coverage={coverage_ratio:.6f}"
                )
            else:
                print(
                    f"[{log_prefix}] 早期半径归一化未触发: "
                    f"coverage={coverage_ratio:.6f} >= "
                    f"T={self.early_radius_normalization_threshold:.6f}"
                )

        filtered_unlabeled_tensor = self._filter_unlabeled_indices_torch(
            unlabeled_indices=unlabeled_indices,
            covered_mask=covered_mask
        )
        print(f"[{log_prefix}] 原始无标注池大小: {len(unlabeled_indices)}")
        print(
            f"[{log_prefix}] 被已标注点覆盖的点数: "
            f"{len(unlabeled_indices) - int(filtered_unlabeled_tensor.numel())}"
        )
        print(
            f"[{log_prefix}] 过滤后无标注池大小: "
            f"{int(filtered_unlabeled_tensor.numel())}"
        )

        if filtered_unlabeled_tensor.numel() == 0:
            print(f"[{log_prefix}] 无标注池为空，无需选择样本")
            return [], effective_labeled_radii.detach().cpu().tolist(), []

        filtered_unlabeled = filtered_unlabeled_tensor.detach().cpu().tolist()
        probabilities = self._get_probabilities_for_indices_torch(
            model,
            dataset,
            filtered_unlabeled,
            device
        )
        h_values = self._compute_h_from_probabilities_torch(probabilities)
        estimated_radii = self._estimate_radii_for_unlabeled_torch(
            h_values=h_values,
            unlabeled_indices=filtered_unlabeled,
            spectral_norm_product=spectral_norm_product,
            device=device
        )
        if normalization_applied:
            estimated_radii = torch.minimum(
                estimated_radii * normalization_scale_ratio,
                torch.full_like(estimated_radii, max_theoretical_radius)
            )
            estimated_radii = self._clip_radii_to_max_theoretical_torch(
                radii=estimated_radii,
                indices=filtered_unlabeled,
                spectral_norm_product=spectral_norm_product,
                device=device
            )

        print(
            f"[{log_prefix}] 预估半径统计: "
            f"min={float(torch.min(estimated_radii).item()):.6f}, "
            f"max={float(torch.max(estimated_radii).item()):.6f}, "
            f"mean={float(torch.mean(estimated_radii).item()):.6f}"
        )

        candidate_indices = filtered_unlabeled_tensor
        candidate_radii = estimated_radii
        cover_matrix, initial_covered_counts = (
            self._prepare_candidate_cover_matrix_from_full_distance_torch(
                distance_matrix=distance_tensor,
                candidate_indices=candidate_indices,
                candidate_radii=candidate_radii
            )
        )
        initial_log_coverage_terms = torch.log(
            torch.clamp(initial_covered_counts, min=1).to(dtype=torch.float64)
        )
        static_score_terms = self._compute_static_score_terms_torch(
            probabilities,
            coverage_ratio,
            coverage_terms=initial_log_coverage_terms
        )
        selected_local_positions = (
            self._greedy_select_candidate_positions_from_cover_matrix_torch(
                candidate_indices=candidate_indices,
                static_score_terms=static_score_terms,
                cover_matrix=cover_matrix,
                initial_covered_counts=initial_covered_counts,
                query_budget=self.query_budget,
                progress_desc=(
                    f"[{log_prefix}] 第{self.round_counter + 1}轮选择进度"
                ),
                log_prefix=log_prefix
            )
        )
        selected_positions_tensor = torch.as_tensor(
            selected_local_positions,
            dtype=torch.long,
            device=device
        )
        selected_indices = candidate_indices.index_select(
            0,
            selected_positions_tensor
        ).detach().cpu().tolist()
        selected_radii = candidate_radii.index_select(
            0,
            selected_positions_tensor
        ).detach().cpu().tolist()

        print(
            f"[{log_prefix}] 第{self.round_counter + 1}轮选择的标注点数量: "
            f"{len(selected_indices)}"
        )
        return (
            selected_indices,
            effective_labeled_radii.detach().cpu().tolist(),
            selected_radii
        )

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
            # 第一轮：使用最大理论半径进行贪心采样，直到达到查询预算
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
        第一轮采样（预算版本）
        
        步骤：
        1. 用初始标记的点训练模型（已在select_samples之前完成）
        2. 计算最大理论半径
        3. 更新无标注池：去除被已标注点以最大理论半径覆盖的点
        4. 贪心采样：选择能覆盖最多点的点，直到达到查询预算B
        
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
        print(f"\n[OurMethodBudgetVersion] 第一轮采样")
        
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
                "[OurMethodBudgetVersion] 最大理论半径统计: "
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
            print(f"[OurMethodBudgetVersion] 原始无标注池大小: {len(unlabeled_indices)}")
            print(f"[OurMethodBudgetVersion] 被已标注点覆盖的点数: {len(unlabeled_indices) - len(filtered_unlabeled)}")
            print(f"[OurMethodBudgetVersion] 过滤后无标注池大小: {len(filtered_unlabeled)}")
        else:
            filtered_unlabeled = list(unlabeled_indices)
            print(f"[OurMethodBudgetVersion] 无标注池大小: {len(filtered_unlabeled)}")
        
        # 步骤4：贪心采样，每个候选点使用自己的最大理论半径，直到达到查询预算
        if filtered_unlabeled:
            radii = np.array(
                [round_radius_by_index[idx] for idx in filtered_unlabeled],
                dtype=np.float64
            )
            
            # 执行贪心采样，传入查询预算
            selected_indices, covered_indices = self._greedy_selection(
                unlabeled_indices=filtered_unlabeled,
                distance_matrix=distance_matrix,
                radii=radii,
                query_budget=self.query_budget
            )
            
            print(f"[OurMethodBudgetVersion] 第一轮选择的标注点数量: {len(selected_indices)}")
            print(f"[OurMethodBudgetVersion] 第一轮覆盖的点数量: {len(covered_indices)}")
            
            return selected_indices
        else:
            print(f"[OurMethodBudgetVersion] 无标注池为空，无需选择样本")
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
        print(f"\n[OurMethodBudgetVersion] 第{self.round_counter + 1}轮采样")

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
        print(f"[OurMethodBudgetVersion] 当前轮次覆盖率: {coverage_ratio:.6f}")

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
                    "[OurMethodBudgetVersion] 早期半径归一化已触发: "
                    f"T={self.early_radius_normalization_threshold:.6f}, "
                    f"k={self.early_radius_normalization_percentile_k:.2f}, "
                    f"q_k={normalization_pivot_radius:.6f}, "
                    f"r={normalization_scale_ratio:.6f}, "
                    f"max_radius={max_theoretical_radius:.6f}, "
                    f"normalized_coverage={coverage_ratio:.6f}"
                )
            else:
                print(
                    "[OurMethodBudgetVersion] 早期半径归一化未触发: "
                    f"coverage={coverage_ratio:.6f} >= "
                    f"T={self.early_radius_normalization_threshold:.6f}"
                )

        # 步骤4：更新无标注池，去除被已标注点以其自身计算的半径覆盖的点
        filtered_unlabeled = self._filter_unlabeled_indices(
            unlabeled_indices=unlabeled_indices,
            covered_mask=covered_mask
        )
        print(f"[OurMethodBudgetVersion] 原始无标注池大小: {len(unlabeled_indices)}")
        print(f"[OurMethodBudgetVersion] 被已标注点覆盖的点数: {len(unlabeled_indices) - len(filtered_unlabeled)}")
        print(f"[OurMethodBudgetVersion] 过滤后无标注池大小: {len(filtered_unlabeled)}")
        
        if not filtered_unlabeled:
            print(f"[OurMethodBudgetVersion] 无标注池为空，无需选择样本")
            return []
        
        # 步骤5：预测候选点概率，并基于同一批概率同时计算 h 与后续打分项
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

        print(f"[OurMethodBudgetVersion] 预估半径统计: min={estimated_radii.min():.6f}, "
              f"max={estimated_radii.max():.6f}, mean={estimated_radii.mean():.6f}")

        selected_indices = []
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
        # 步骤6：缓存后续打分中不随贪心迭代变化的项
        static_score_terms = self._compute_static_score_terms(
            probabilities,
            coverage_ratio,
            coverage_terms=initial_log_coverage_terms
        )

        # 步骤7：基于分数的贪心采样
        selected_local_positions = self._greedy_select_candidate_positions_from_cover_data(
            candidate_indices=candidate_indices,
            static_score_terms=static_score_terms,
            covered_positions_by_candidate=covered_positions_by_candidate,
            reverse_cover_rows=reverse_cover_rows,
            initial_covered_counts=initial_covered_counts,
            query_budget=self.query_budget,
            progress_desc=f"[OurMethodBudgetVersion] 第{self.round_counter + 1}轮选择进度",
            log_prefix="OurMethodBudgetVersion"
        )
        selected_indices = [
            int(candidate_indices[local_position])
            for local_position in selected_local_positions
        ]

        print(f"[OurMethodBudgetVersion] 第{self.round_counter + 1}轮选择的标注点数量: {len(selected_indices)}")
        
        return selected_indices
