"""
Our Method Margin Unfil - 基于半径覆盖的主动学习策略（margin + 不过滤候选池版本）

该策略与 `our_method_unfil` 保持完全一致的覆盖集合维护与贪心采样流程：
1. 每轮都先根据已标注点维护当前累计覆盖集合；
2. 已覆盖点不会从无标注候选池中删除；
3. 贪心时 `D(x)` 只统计点 `x` 当前还能新增覆盖多少未覆盖点；
4. 每选中一个点后，只更新累计覆盖集合，不批量删除候选点。

与 `our_method_unfil` 相比，唯一差异是后续轮次的静态不确定性评分项：
把 `alpha * coverage_ratio * ln(|eta-p|_2)` 改为
`-alpha * coverage_ratio * ln(margin)`，其中
`margin = p_1 - p_2` 为最大类别概率与次大类别概率之差。
"""
from typing import Optional

import numpy as np

from .our_method_unfil import OurMethodUnfilStrategy


class OurMethodMarginUnfilStrategy(OurMethodUnfilStrategy):
    """
    基于半径覆盖的主动学习策略（margin + 不过滤候选池版本）。

    该实现直接继承 `OurMethodUnfilStrategy`，从而完整复用“不过滤候选池、
    仅维护累计覆盖集合”的采样行为。与父类相比，本类唯一改变的是后续轮次中
    不确定性项的定义：

    `score(x) = -alpha * coverage_ratio * ln(margin(x)) + ln(D(x))`

    其中：
    1. `margin(x)` 为当前模型对样本 `x` 的最大类别概率与次大类别概率之差；
    2. `D(x)` 仍然表示点 `x` 在当前累计覆盖集合之外还能新增覆盖的无标注点数；
    3. 第一轮仍保持与 `OurMethodUnfilStrategy` 完全一致，只基于新增覆盖量做
       最大理论半径下的贪心采样。
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
        初始化 margin + 不过滤候选池版本策略。

        该构造函数保持与 `OurMethodUnfilStrategy` 完全一致的参数接口，目的是让
        外部实验脚本、配置解析器和工厂函数无需为该策略单独分叉参数结构。
        初始化过程直接复用父类实现，只在最后把策略名称改为
        `OurMethodMarginUnfil`，用于日志输出和实验结果区分。

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
        self.name = "OurMethodMarginUnfil"

    def _compute_margins(
        self,
        probabilities: np.ndarray
    ) -> np.ndarray:
        """
        根据候选点概率矩阵批量计算 margin 值。

        该函数把 margin 的定义集中在一个地方处理，确保后续轮次中所有
        margin 相关打分逻辑都与 `our_method_margin` 保持一致。具体步骤如下：
        1. 对每个候选点的类别概率按降序排序；
        2. 取前两项分别作为最大概率 `p_1` 与次大概率 `p_2`；
        3. 返回 `p_1 - p_2` 作为 margin，值越小表示该点越不确定；
        4. 若类别数不足 2，则返回零数组，避免索引越界并显式退化。

        参数:
            probabilities: 候选样本的预测概率矩阵，形状为
                `(num_candidates, num_classes)`。

        返回:
            np.ndarray: 与输入候选点顺序严格对齐的一维 margin 数组，形状为
                `(num_candidates,)`。
        """
        if probabilities.size == 0:
            return np.empty((0,), dtype=np.float64)

        if probabilities.shape[1] < 2:
            return np.zeros((probabilities.shape[0],), dtype=np.float64)

        sorted_probabilities = np.sort(probabilities, axis=1)[:, ::-1]
        return sorted_probabilities[:, 0] - sorted_probabilities[:, 1]

    def _compute_score(
        self,
        coverage_ratio: float,
        margin: float,
        D: int
    ) -> float:
        """
        计算 margin + 不过滤候选池版本在后续轮次中的单点采样分数。

        虽然当前采样主流程通过批量预计算静态评分项驱动贪心，但保留该单点评分
        接口有两个作用：
        1. 让本策略的数学定义在单点层面保持清晰，便于维护和调试；
        2. 若未来父类或外部逻辑重新调用 `_compute_score(...)`，该策略仍能
           保证其语义与 margin 版本完全一致。

        参数:
            coverage_ratio: 当前轮次的覆盖率。
            margin: 当前样本的 margin 值，即最大概率与次大概率之差。
            D: 当前样本在累计覆盖集合之外还能新增覆盖的无标注点数量。

        返回:
            float: 当前样本的采样分数。
        """
        margin = max(margin, 1e-10)
        D = max(D, 1)
        return -self.alpha * coverage_ratio * np.log(margin) + np.log(D)

    def _compute_static_score_terms(
        self,
        probabilities: np.ndarray,
        coverage_ratio: float
    ) -> np.ndarray:
        """
        批量计算 margin + 不过滤候选池版本的静态不确定性项。

        该函数只替换后续轮次评分公式中“与候选覆盖集合变化无关”的那一部分。
        对于本策略，后续轮次评分公式为：

        `-alpha * coverage_ratio * ln(margin(x)) + ln(D(x))`

        其中：
        1. `margin(x)` 只依赖当前模型对样本 `x` 的概率预测，因此适合在进入
           贪心循环前一次性批量计算；
        2. `D(x)` 依赖当前累计覆盖集合，会在贪心过程中动态变化，因此不在此处
           计算；
        3. 返回值会与候选点顺序严格对齐，供父类无过滤贪心逻辑直接复用。

        参数:
            probabilities: 候选样本的预测概率矩阵，形状为
                `(num_candidates, num_classes)`。
            coverage_ratio: 当前轮次覆盖率。该值在本轮贪心过程中不变，因此可以
                直接并入静态项中一次性计算。

        返回:
            np.ndarray: 与候选点顺序对齐的一维静态评分项数组。
        """
        margins = np.maximum(self._compute_margins(probabilities), 1e-10)
        return -self.alpha * coverage_ratio * np.log(margins)
