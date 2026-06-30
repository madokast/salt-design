"""
KNN 局部放大系数估计工具。

该模块服务于 OurMethod 系列策略：当全局谱范数乘积过于保守时，使用数据
邻域内的 logits 有限差分估计每个样本自己的局部放大系数。
"""
from typing import Any, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


def predict_logits_for_indices(
    model: torch.nn.Module,
    dataset: Any,
    indices: Sequence[int],
    device: torch.device,
    batch_size: int
) -> np.ndarray:
    """
    批量预测指定样本索引对应的 logits。

    该函数只负责模型前向，不做 softmax，原因是局部放大系数要估计的是
    “输入空间距离经过模型主体后在 logits 空间的放大程度”。如果使用 softmax
    后的概率，饱和区间会人为压缩差异，导致估计量不再直接对应半径公式里的
    logits 扰动项。函数保证输出顺序与 `indices` 完全一致，调用方可以直接按
    全局索引映射回距离矩阵的行/列。

    参数:
        model: 当前轮训练完成后的 PyTorch 模型。
        dataset: 完整训练集对象，支持按整数索引读取样本。
        indices: 需要预测 logits 的全局样本索引序列。
        device: 当前计算设备。
        batch_size: 前向预测使用的批大小，必须为正整数。

    返回:
        np.ndarray: logits 矩阵，形状为 `(len(indices), num_classes)`。

    异常:
        ValueError: 当 `batch_size` 不是正整数时抛出。
    """
    if batch_size <= 0:
        raise ValueError("batch_size 必须是正整数。")
    if len(indices) == 0:
        return np.empty((0, 0), dtype=np.float32)

    model.eval()
    logits_batches: List[torch.Tensor] = []
    subset = Subset(dataset, list(indices))
    loader = DataLoader(
        subset,
        batch_size=min(batch_size, len(indices)),
        shuffle=False
    )

    with torch.no_grad():
        for batch_data in loader:
            if isinstance(batch_data, (list, tuple)):
                inputs = batch_data[0]
            else:
                inputs = batch_data

            if not isinstance(inputs, torch.Tensor):
                inputs = torch.as_tensor(inputs)

            inputs = inputs.to(device)
            logits = model(inputs)
            logits_batches.append(logits.detach().cpu())

    return torch.cat(logits_batches, dim=0).numpy()


def predict_logits_for_indices_torch(
    model: torch.nn.Module,
    dataset: Any,
    indices: Sequence[int],
    device: torch.device,
    batch_size: int
) -> torch.Tensor:
    """
    批量预测指定样本索引对应的 logits，并保留在目标设备上。
    """
    if batch_size <= 0:
        raise ValueError("batch_size 必须是正整数。")
    if len(indices) == 0:
        return torch.empty((0, 0), dtype=torch.float32, device=device)

    model.eval()
    logits_batches: List[torch.Tensor] = []
    subset = Subset(dataset, list(indices))
    loader = DataLoader(
        subset,
        batch_size=min(batch_size, len(indices)),
        shuffle=False
    )

    with torch.no_grad():
        for batch_data in loader:
            if isinstance(batch_data, (list, tuple)):
                inputs = batch_data[0]
            else:
                inputs = batch_data

            if not isinstance(inputs, torch.Tensor):
                inputs = torch.as_tensor(inputs)

            inputs = inputs.to(device)
            logits = model(inputs)
            logits_batches.append(logits.detach())

    return torch.cat(logits_batches, dim=0)


def softmax_numpy(logits: np.ndarray) -> np.ndarray:
    """
    对 numpy logits 矩阵逐行计算 softmax 概率。

    该函数用于在已经缓存全量 logits 的局部放大系数模式下复用前向结果，避免
    后续候选点概率预测再次调用模型。实现时先减去每行最大值，以避免指数运算
    在 logits 较大时溢出；输出行顺序与输入 logits 完全一致。

    参数:
        logits: logits 矩阵，形状为 `(num_samples, num_classes)`。

    返回:
        np.ndarray: softmax 概率矩阵，形状与 `logits` 相同。
    """
    if logits.size == 0:
        return np.empty_like(logits, dtype=np.float64)

    logits = np.asarray(logits, dtype=np.float64)
    shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(shifted_logits)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def softmax_torch(logits: torch.Tensor) -> torch.Tensor:
    """
    对 torch logits 矩阵逐行计算 softmax 概率。
    """
    if logits.numel() == 0:
        return torch.empty_like(logits, dtype=torch.float64)

    logits = logits.to(dtype=torch.float64)
    shifted_logits = logits - torch.max(logits, dim=1, keepdim=True).values
    exp_logits = torch.exp(shifted_logits)
    return exp_logits / torch.sum(exp_logits, dim=1, keepdim=True)


def compute_knn_local_expansion_factors(
    query_indices: Sequence[int],
    logits_by_index: np.ndarray,
    distance_matrix: np.ndarray,
    knn_k: int,
    quantile: float,
    distance_eps: float,
    min_factor: float
) -> np.ndarray:
    """
    为每个查询点估计 KNN 局部 logits 放大系数。

    对每个查询点 `i`，函数在距离矩阵中寻找距离最近的 `knn_k` 个非自身邻居，
    然后计算这些邻居方向上的有限差分比值：
    `||logits_i - logits_j||_2 / (distance(i, j) + distance_eps)`。
    最终使用这些比值的 `quantile` 分位数作为该点的局部放大系数 `L_i`。
    这种估计量只在数据邻域内刻画模型放大程度，不再使用全网络谱范数乘积的
    全局最坏情况上界。

    参数:
        query_indices: 需要估计 `L_i` 的全局样本索引序列。
        logits_by_index: 与完整距离矩阵行号对齐的 logits 矩阵，形状为
            `(num_samples, num_classes)`。
        distance_matrix: 全量样本距离矩阵，行列索引必须与 `logits_by_index`
            的第一维一致。
        knn_k: 每个点使用的近邻数量，必须为正整数。
        quantile: 在 KNN finite-difference ratios 上取的分位数，范围 `[0, 1]`。
        distance_eps: 分母稳定项，同时用于过滤距离过小的邻居。
        min_factor: 纯数值下界，用于避免所有有效 ratio 为空或为零时出现除零。

    返回:
        np.ndarray: 与 `query_indices` 顺序一致的一维局部放大系数数组。

    异常:
        ValueError:
            - 当 `knn_k` 非正时抛出；
            - 当 `quantile` 不在 `[0, 1]` 时抛出；
            - 当 `distance_eps` 或 `min_factor` 非正时抛出；
            - 当 logits 与距离矩阵样本数不一致时抛出。
    """
    if knn_k <= 0:
        raise ValueError("knn_k 必须是正整数。")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile 必须位于 [0, 1] 区间内。")
    if distance_eps <= 0.0:
        raise ValueError("distance_eps 必须是正数。")
    if min_factor <= 0.0:
        raise ValueError("min_factor 必须是正数。")
    if logits_by_index.shape[0] != distance_matrix.shape[0]:
        raise ValueError(
            "logits_by_index 的样本数必须与 distance_matrix 的行数一致。"
        )
    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix 必须是方阵。")

    query_indices = list(query_indices)
    local_factors = np.empty((len(query_indices),), dtype=np.float64)

    for output_position, query_index in enumerate(query_indices):
        distances = np.asarray(distance_matrix[query_index], dtype=np.float64)
        valid_mask = np.isfinite(distances) & (distances > distance_eps)
        valid_mask[query_index] = False
        valid_neighbors = np.flatnonzero(valid_mask)

        if valid_neighbors.size == 0:
            local_factors[output_position] = min_factor
            continue

        neighbor_count = min(knn_k, int(valid_neighbors.size))
        valid_distances = distances[valid_neighbors]
        nearest_positions = np.argpartition(
            valid_distances,
            neighbor_count - 1
        )[:neighbor_count]
        nearest_neighbors = valid_neighbors[nearest_positions]

        logit_diffs = logits_by_index[nearest_neighbors] - logits_by_index[query_index]
        numerator = np.linalg.norm(logit_diffs, axis=1)
        denominator = np.maximum(distances[nearest_neighbors], distance_eps)
        ratios = numerator / denominator
        ratios = ratios[np.isfinite(ratios)]

        if ratios.size == 0:
            local_factors[output_position] = min_factor
        else:
            local_factors[output_position] = max(
                float(np.quantile(ratios, quantile)),
                min_factor
            )

    return local_factors


def compute_local_expansion_factors_from_knn(
    logits_by_index: np.ndarray,
    knn_indices: np.ndarray,
    knn_distances: np.ndarray,
    quantile: float,
    distance_eps: float,
    min_factor: float
) -> np.ndarray:
    """
    从预计算的 ``N x K`` KNN 缓存估计每个样本的局部 logits 放大系数。

    与完整距离矩阵版本使用相同的有限差分定义，但不再搜索邻居：对每行缓存的
    邻居计算 ``||logits_i-logits_j|| / distance(i,j)``，过滤非有限值和距离不大于
    ``distance_eps`` 的项，再取指定分位数。若某行没有有效比值，则使用
    ``min_factor``，从而保证后续半径公式不会除零。

    参数:
        logits_by_index: 全量样本 logits，形状 ``(N, C)``。
        knn_indices: 近邻全局索引，形状 ``(N, K)``。
        knn_distances: 与索引对齐的欧氏距离。
        quantile: 每行有限差分比值使用的分位数。
        distance_eps: 有效距离下界和分母稳定项。
        min_factor: 局部放大系数数值下界。

    返回:
        np.ndarray: 与全量样本索引对齐的局部放大系数，形状 ``(N,)``。

    异常:
        ValueError: 输入形状或数值参数无效时抛出。
    """
    logits_by_index = np.asarray(logits_by_index)
    if knn_indices.ndim != 2 or knn_indices.shape != knn_distances.shape:
        raise ValueError("KNN 索引与距离必须是形状一致的二维数组。")
    if logits_by_index.shape[0] != knn_indices.shape[0]:
        raise ValueError("logits 与 KNN 缓存样本数不一致。")
    if not 0.0 <= quantile <= 1.0 or distance_eps <= 0.0 or min_factor <= 0.0:
        raise ValueError("quantile 或局部放大系数稳定参数无效。")
    factors = np.full(knn_indices.shape[0], min_factor, dtype=np.float64)
    chunk_size = 1024
    for start in range(0, knn_indices.shape[0], chunk_size):
        end = min(start + chunk_size, knn_indices.shape[0])
        neighbors = np.asarray(knn_indices[start:end], dtype=np.int64)
        distances = np.asarray(knn_distances[start:end], dtype=np.float64)
        differences = logits_by_index[neighbors] - logits_by_index[start:end, None, :]
        ratios = np.linalg.norm(differences, axis=2) / np.maximum(distances, distance_eps)
        valid = np.isfinite(ratios) & np.isfinite(distances) & (distances > distance_eps)
        ratios[~valid] = np.nan
        valid_rows = np.any(valid, axis=1)
        if np.any(valid_rows):
            quantiles = np.nanquantile(ratios[valid_rows], quantile, axis=1)
            factors[start:end][valid_rows] = np.maximum(quantiles, min_factor)
    return factors


def compute_knn_local_expansion_factors_torch(
    query_indices: Sequence[int],
    logits_by_index: torch.Tensor,
    distance_matrix: torch.Tensor,
    knn_k: int,
    quantile: float,
    distance_eps: float,
    min_factor: float
) -> torch.Tensor:
    """
    为每个查询点估计 KNN 局部 logits 放大系数，并保留在当前 CUDA 设备上。
    """
    if knn_k <= 0:
        raise ValueError("knn_k 必须是正整数。")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile 必须位于 [0, 1] 区间内。")
    if distance_eps <= 0.0:
        raise ValueError("distance_eps 必须是正数。")
    if min_factor <= 0.0:
        raise ValueError("min_factor 必须是正数。")
    if logits_by_index.shape[0] != distance_matrix.shape[0]:
        raise ValueError(
            "logits_by_index 的样本数必须与 distance_matrix 的行数一致。"
        )
    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix 必须是方阵。")

    query_indices_tensor = torch.as_tensor(
        list(query_indices),
        dtype=torch.long,
        device=distance_matrix.device
    )
    local_factors = torch.empty(
        (query_indices_tensor.numel(),),
        dtype=torch.float64,
        device=distance_matrix.device
    )
    distance_eps_value = float(distance_eps)
    min_factor_value = float(min_factor)

    logits_by_index = logits_by_index.to(dtype=torch.float64)
    for output_position, query_index in enumerate(query_indices_tensor):
        distances = distance_matrix[query_index].to(dtype=torch.float64)
        valid_mask = torch.isfinite(distances) & (distances > distance_eps_value)
        valid_mask[query_index] = False
        valid_neighbors = torch.nonzero(valid_mask, as_tuple=False).flatten()

        if valid_neighbors.numel() == 0:
            local_factors[output_position] = min_factor_value
            continue

        neighbor_count = min(knn_k, int(valid_neighbors.numel()))
        valid_distances = distances[valid_neighbors]
        nearest_positions = torch.topk(
            valid_distances,
            k=neighbor_count,
            largest=False
        ).indices
        nearest_neighbors = valid_neighbors[nearest_positions]

        logit_diffs = logits_by_index[nearest_neighbors] - logits_by_index[query_index]
        numerator = torch.linalg.vector_norm(logit_diffs, dim=1)
        denominator = torch.clamp(distances[nearest_neighbors], min=distance_eps_value)
        ratios = numerator / denominator
        ratios = ratios[torch.isfinite(ratios)]

        if ratios.numel() == 0:
            local_factors[output_position] = min_factor_value
        else:
            local_factors[output_position] = torch.clamp(
                torch.quantile(ratios, quantile),
                min=min_factor_value
            )

    return local_factors


def compute_max_theoretical_radii_from_expansion(
    expansion_factors: np.ndarray,
    epsilon: float,
    h_min: float,
    diff_min: float
) -> np.ndarray:
    """
    根据每点放大系数计算最大理论半径数组。

    该函数把原来只接受单个全局 `spectral_norm_product` 的最大理论半径公式
    向量化，使每个样本都可以使用自己的局部放大系数 `L_i`。它不引入额外的
    局部半径上界，只把公式中的全局放大系数替换为逐点放大系数：
    `2 * epsilon / (L_i * (diff_min + sqrt(diff_min^2 + 2*h_min*epsilon)))`。

    参数:
        expansion_factors: 每个点的局部或全局放大系数数组。
        epsilon: 半径方程中的 epsilon 参数。
        h_min: Hessian 谱范数下界。
        diff_min: diff 下界。

    返回:
        np.ndarray: 与 `expansion_factors` 顺序一致的最大理论半径数组。
    """
    expansion_factors = np.asarray(expansion_factors, dtype=np.float64)
    sqrt_term = np.sqrt(diff_min**2 + 2.0 * h_min * epsilon)
    return 2.0 * epsilon / (expansion_factors * (diff_min + sqrt_term))


def compute_max_theoretical_radii_from_expansion_torch(
    expansion_factors: torch.Tensor,
    epsilon: float,
    h_min: float,
    diff_min: float
) -> torch.Tensor:
    """
    根据每点放大系数计算最大理论半径数组，并保留在当前设备上。
    """
    expansion_factors = expansion_factors.to(dtype=torch.float64)
    sqrt_term = torch.sqrt(
        torch.tensor(
            diff_min**2 + 2.0 * h_min * epsilon,
            dtype=torch.float64,
            device=expansion_factors.device
        )
    )
    return 2.0 * epsilon / (expansion_factors * (diff_min + sqrt_term))


def compute_estimated_radii_from_expansion(
    h_values: np.ndarray,
    expansion_factors: np.ndarray,
    epsilon: float,
    diff_min: float
) -> np.ndarray:
    """
    根据每点 H 值和每点放大系数计算候选点预估半径。

    后续轮次候选点半径使用样本自身的 Hessian 谱范数估计 `h_i`，并把原公式
    中的全局 `spectral_norm_product` 替换为该样本的局部放大系数 `L_i`：
    `2 * epsilon / (L_i * (diff_min + sqrt(diff_min^2 + 2*h_i*epsilon)))`。
    函数只做公式向量化，不额外截断半径上界。

    参数:
        h_values: 与候选点顺序一致的 H 最大特征值数组。
        expansion_factors: 与候选点顺序一致的局部或全局放大系数数组。
        epsilon: 半径方程中的 epsilon 参数。
        diff_min: diff 下界。

    返回:
        np.ndarray: 与输入候选点顺序一致的预估半径数组。

    异常:
        ValueError: 当 `h_values` 与 `expansion_factors` 形状不一致时抛出。
    """
    h_values = np.asarray(h_values, dtype=np.float64)
    expansion_factors = np.asarray(expansion_factors, dtype=np.float64)
    if h_values.shape != expansion_factors.shape:
        raise ValueError("h_values 与 expansion_factors 形状必须一致。")

    sqrt_term = np.sqrt(diff_min**2 + 2.0 * h_values * epsilon)
    return 2.0 * epsilon / (expansion_factors * (diff_min + sqrt_term))


def compute_estimated_radii_from_expansion_torch(
    h_values: torch.Tensor,
    expansion_factors: torch.Tensor,
    epsilon: float,
    diff_min: float
) -> torch.Tensor:
    """
    根据每点 H 值和每点放大系数计算候选点预估半径，并保留在当前设备上。
    """
    h_values = h_values.to(dtype=torch.float64)
    expansion_factors = expansion_factors.to(dtype=torch.float64)
    if h_values.shape != expansion_factors.shape:
        raise ValueError("h_values 与 expansion_factors 形状必须一致。")

    sqrt_term = torch.sqrt(diff_min**2 + 2.0 * h_values * epsilon)
    return 2.0 * epsilon / (expansion_factors * (diff_min + sqrt_term))


def resolve_expansion_factors_for_indices(
    indices: Sequence[int],
    fixed_expansion_factor: Optional[float],
    local_expansion_factors: Optional[np.ndarray]
) -> np.ndarray:
    """
    按索引解析当前应使用的放大系数数组。

    当 `fixed_expansion_factor` 是有限正数时，函数为所有索引返回同一个固定值，
    保持旧的全局谱范数乘积/手动常数语义；当其为 `None` 时，函数从已经按
    全局样本索引对齐的 `local_expansion_factors` 中取出每个点自己的 `L_i`。
    这样上层策略只需要调用同一个解析入口，就能同时支持全局与局部两种模式。

    参数:
        indices: 需要解析放大系数的全局样本索引序列。
        fixed_expansion_factor: 固定全局放大系数；若为 `None` 则使用局部数组。
        local_expansion_factors: 与完整距离矩阵行号对齐的一维局部放大系数数组。

    返回:
        np.ndarray: 与 `indices` 顺序一致的一维放大系数数组。

    异常:
        RuntimeError: 当固定值为 `None` 且局部数组尚未准备好时抛出。
        ValueError: 当解析出的放大系数不是有限正数时抛出。
    """
    indices = list(indices)
    if fixed_expansion_factor is None:
        if local_expansion_factors is None:
            raise RuntimeError("局部放大系数尚未准备好。")
        if isinstance(local_expansion_factors, torch.Tensor):
            expansion_factors = local_expansion_factors[
                indices
            ].detach().cpu().numpy().astype(np.float64, copy=False)
        else:
            expansion_factors = np.asarray(
                local_expansion_factors[indices],
                dtype=np.float64
            )
    else:
        expansion_factors = np.full(
            (len(indices),),
            float(fixed_expansion_factor),
            dtype=np.float64
        )

    if not np.all(np.isfinite(expansion_factors)) or np.any(expansion_factors <= 0.0):
        raise ValueError("放大系数必须全部为有限正数。")
    return expansion_factors


def resolve_expansion_factors_for_indices_torch(
    indices: Sequence[int],
    fixed_expansion_factor: Optional[float],
    local_expansion_factors: Optional[torch.Tensor],
    device: torch.device
) -> torch.Tensor:
    """
    按索引解析当前应使用的放大系数数组，并保留在目标设备上。
    """
    indices = list(indices)
    if fixed_expansion_factor is None:
        if local_expansion_factors is None:
            raise RuntimeError("局部放大系数尚未准备好。")
        if isinstance(local_expansion_factors, torch.Tensor):
            expansion_factors = local_expansion_factors[
                indices
            ].to(device=device, dtype=torch.float64)
        else:
            expansion_factors = torch.as_tensor(
                local_expansion_factors[indices],
                dtype=torch.float64,
                device=device
            )
    else:
        expansion_factors = torch.full(
            (len(indices),),
            float(fixed_expansion_factor),
            dtype=torch.float64,
            device=device
        )

    if not torch.all(torch.isfinite(expansion_factors)) or torch.any(expansion_factors <= 0.0):
        raise ValueError("放大系数必须全部为有限正数。")
    return expansion_factors
