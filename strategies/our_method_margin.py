"""
Our Method Margin - 基于半径覆盖的主动学习策略（margin 评分版本）

该策略与 `our_method_budget_version` 保持相同的采样流程：
1. 第一轮使用最大理论半径进行贪心采样，直到达到查询预算 B；
2. 后续轮次先过滤已被标注点覆盖的无标注样本，再基于候选点预估半径做贪心选择；
3. 唯一差异是后续轮次的静态不确定性评分项从 `ln(|eta-p|_2)` 改为
   `-ln(margin)`，其中 `margin = p_1 - p_2`。
"""
from typing import Optional

import numpy as np
import torch

from .our_method_budget_version import OurMethodBudgetVersionStrategy


class OurMethodMarginStrategy(OurMethodBudgetVersionStrategy):
    """
    基于半径覆盖的主动学习策略（margin 评分版本）。

    该实现复用 `OurMethodBudgetVersionStrategy` 的全部流程，包括：
    距离矩阵加载、理论半径计算、已标注点覆盖过滤、候选点半径估计、
    预算约束下的贪心覆盖删除等。与父类相比，唯一改变的是后续轮次中
    不确定性项的定义：

    `score(x) = -alpha * coverage_ratio * ln(margin(x)) + ln(D(x))`

    其中 `margin(x)` 为当前模型对样本 `x` 的最大类别概率与次大类别概率之差。
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
        初始化 margin 评分版本策略。

        该构造函数保持与 `OurMethodBudgetVersionStrategy` 完全一致的参数接口，
        这样外部配置、命令行参数解析和实验脚本无需为 margin 版本单独分叉。
        初始化过程直接复用父类实现，随后仅把策略名称更新为
        `OurMethodMargin`，便于日志输出、实验记录与结果对比。

        参数:
            epsilon: 半径方程中的 `epsilon` 参数。
            h_min: Hessian l2 谱范数的下界，用于约束半径计算稳定性。
            diff_min: 半径公式中的 `diff` 下界，用于避免分母过小。
            spectral_norm_product: 固定全局放大系数；为 `None` 时使用每点KNN局部放大系数。
            distance_cache_dir: 距离矩阵缓存目录。
            query_budget: 每轮最多可选择的样本数量。
            alpha: 后续轮次中不确定性项的权重系数；为 `None` 时按当前候选集
                上 `-log(margin)` 与 `logD` 的 IQR 自动做尺度校准。
            inference_batch_size: 采样阶段概率预测使用的批量大小。
            enable_early_radius_normalization: 是否启用早期半径归一化。
            early_radius_normalization_threshold: 触发早期半径归一化的覆盖率阈值。
            early_radius_normalization_percentile_k: 早期半径归一化的百分位参数。
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
        self.name = "OurMethodMargin"

    def _compute_margins(
        self,
        probabilities: np.ndarray
    ) -> np.ndarray:
        """
        根据候选点概率矩阵批量计算 margin 值。

        该函数把 margin 的定义固定为“最大类别概率减去次大类别概率”，并保证
        输出顺序与输入候选点顺序严格对齐，便于在进入贪心循环前一次性缓存。
        具体处理步骤如下：
        1. 对每个候选点的概率向量按降序排序；
        2. 取排序后前两项作为 `p_1` 与 `p_2`；
        3. 返回 `p_1 - p_2`，得到越小表示越不确定的 margin 值；
        4. 若类别数不足 2，则退化为返回空数组或零数组，避免索引越界。

        参数:
            probabilities: 候选样本的预测概率矩阵，形状为
                `(num_candidates, num_classes)`。

        返回:
            np.ndarray: 与输入样本逐行对应的 margin 数组，形状为
                `(num_candidates,)`。
        """
        if probabilities.size == 0:
            return np.empty((0,), dtype=np.float64)

        if probabilities.shape[1] < 2:
            return np.zeros((probabilities.shape[0],), dtype=np.float64)

        top_two_probabilities = np.partition(
            probabilities,
            kth=probabilities.shape[1] - 2,
            axis=1
        )[:, -2:]
        return (
            np.max(top_two_probabilities, axis=1)
            - np.min(top_two_probabilities, axis=1)
        )

    def _compute_margins_torch(
        self,
        probabilities: torch.Tensor
    ) -> torch.Tensor:
        """
        根据候选点概率张量批量计算 margin 值，并保留在当前设备上。
        """
        if probabilities.numel() == 0:
            return torch.empty((0,), dtype=torch.float64, device=probabilities.device)

        probabilities = probabilities.to(dtype=torch.float64)
        if probabilities.shape[1] < 2:
            return torch.zeros(
                (probabilities.shape[0],),
                dtype=torch.float64,
                device=probabilities.device
            )

        top_two_probabilities = torch.topk(
            probabilities,
            k=2,
            dim=1
        ).values
        return top_two_probabilities[:, 0] - top_two_probabilities[:, 1]

    def _compute_score(
        self,
        coverage_ratio: float,
        margin: float,
        D: int
    ) -> float:
        """
        计算 margin 评分版本在后续轮次中的单点采样分数。

        虽然父类当前主要通过批量预计算静态评分项来驱动贪心选择，但这里仍保留
        与父类一致的单点评分接口，原因有两个：
        1. 使本类的数学定义在单样本层面清晰可见，便于后续维护和调试；
        2. 若未来父类或外部调用方重新使用 `_compute_score(...)`，该实现仍能
           保证与本策略的 margin 语义一致。

        参数:
            coverage_ratio: 当前轮次覆盖率。
            margin: 当前样本的 margin 值，即最大概率与次大概率之差。
            D: 当前样本在候选池中按其生效半径可覆盖的样本数量。

        返回:
            float: 单个样本的采样分数。
        """
        margin = max(margin, 1e-10)
        D = max(D, 1)
        if self.alpha is None:
            raise RuntimeError(
                "alpha=None 需要候选集尺度信息，请使用批量静态项计算路径。"
            )
        return -self.alpha * coverage_ratio * np.log(margin) + np.log(D)

    def _compute_static_score_terms(
        self,
        probabilities: np.ndarray,
        coverage_ratio: float,
        coverage_terms: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        批量计算 margin 评分版本的静态不确定性项。

        该函数重写父类的静态项计算逻辑，把原本的 `ln(|eta-p|_2)` 替换为
        `-ln(margin)`，从而在不改动其余采样流程的前提下，将后续轮次评分公式
        切换为：

        `-alpha * coverage_ratio * ln(margin(x)) + ln(D(x))`

        其中 `D(x)` 仍在贪心循环中根据当前剩余候选集动态计算，因此这里只负责
        预计算前半部分的静态项。

        参数:
            probabilities: 候选样本的预测概率矩阵，形状为
                `(num_candidates, num_classes)`。
            coverage_ratio: 当前轮次覆盖率。该值在本轮贪心过程中不变，因此适合
                直接并入静态项中一次性计算。
            coverage_terms: 当前候选集初始 `logD` 覆盖项数组，仅在 `alpha=None`
                自动尺度校准时使用。

        返回:
            np.ndarray: 与候选点顺序对齐的一维静态评分项数组。
        """
        margins = np.maximum(self._compute_margins(probabilities), 1e-10)
        uncertainty_terms = -np.log(margins)
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
        批量计算 margin 评分版本的静态不确定性项，并保留在当前设备上。
        """
        margins = torch.clamp(self._compute_margins_torch(probabilities), min=1e-10)
        uncertainty_terms = -torch.log(margins)
        effective_alpha = self._resolve_effective_alpha_torch(
            uncertainty_terms=uncertainty_terms,
            coverage_terms=coverage_terms
        )
        return uncertainty_terms * (effective_alpha * coverage_ratio)
