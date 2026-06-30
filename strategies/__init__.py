"""
主动学习策略模块

本模块包含多种主动学习策略的实现，包括：
- RandomStrategy: 随机采样策略
- LeastConfidenceStrategy: 最小置信度策略
- BADGEStrategy: BADGE策略
- EntropyStrategy: Entropy策略
- MarginStrategy: Margin策略
- KCGreedyStrategy: K-center Greedy策略
- UHerdingStrategy: 不确定性加权覆盖贪心策略
- TypiClustStrategy: 基于簇覆盖与局部典型性的采样策略
- AlphaMixStrategy: ALFA-Mix feature mixing 采样策略
- MyStrategy: 半径/覆盖率策略
- MyStrategyWeightedReport: 半径策略 + 加权报告准确率
- OurMethodStrategy: 基于半径覆盖的贪心采样策略
- OurMethodBudgetVersionStrategy: 基于半径覆盖的贪心采样策略（预算版本）
- OurMethodMarginStrategy: 基于半径覆盖的贪心采样策略（margin 评分版本）
- OurMethodMarginWeightedStrategy: margin 采样 + 下一轮复制式加权训练策略
- OurMethodMarginWeightedSqrtStrategy: margin 采样 + 下一轮复制式加权训练策略（权重开根号）
- SALTExactStrategy: margin + sqrt 复制权重的稀疏 exact 覆盖实现
- SALTExactSoftStrategy: 保留已覆盖候选的稀疏 exact 软覆盖实现
- OurMethodDistStrategy: 基于半径覆盖的贪心采样策略（动态距离修正版本）
- OurMethodMarginDistStrategy: margin 评分 + 动态距离乘法修正策略
- OurMethodMarginDistProxyStrategy: margin_dist 的代理批量更新策略
- OurMethodUnfilStrategy: 基于半径覆盖的贪心采样策略（不过滤候选池版本）
- OurMethodMarginUnfilStrategy: 基于半径覆盖的贪心采样策略（margin + 不过滤候选池版本）
"""

from .base import ActiveLearningStrategy, get_strategy
from .random import RandomStrategy
from .least_confidence import LeastConfidenceStrategy
from .badge import BADGEStrategy
from .entropy import EntropyStrategy
from .margin import MarginStrategy
from .kcgreedy import KCGreedyStrategy
from .uherding import UHerdingStrategy
from .typiclust import TypiClustStrategy
from .alpha_mix import AlphaMixStrategy
from .mystrategy import MyStrategy
from .mystrategy_weighted_report import MyStrategyWeightedReport
from .our_method import OurMethodStrategy
from .our_method_budget_version import OurMethodBudgetVersionStrategy
from .our_method_margin import OurMethodMarginStrategy
from .our_method_margin_weighted import OurMethodMarginWeightedStrategy
from .our_method_margin_weighted_sqrt import OurMethodMarginWeightedSqrtStrategy
from .salt_exact import SALTExactStrategy
from .salt_exact_soft import SALTExactSoftStrategy
from .our_method_dist import OurMethodDistStrategy
from .our_method_margin_dist import OurMethodMarginDistStrategy
from .our_method_margin_dist_proxy import OurMethodMarginDistProxyStrategy
from .our_method_unfil import OurMethodUnfilStrategy
from .our_method_margin_unfil import OurMethodMarginUnfilStrategy

__all__ = [
    'ActiveLearningStrategy',
    'get_strategy',
    'RandomStrategy',
    'LeastConfidenceStrategy',
    'BADGEStrategy',
    'EntropyStrategy',
    'MarginStrategy',
    'KCGreedyStrategy',
    'UHerdingStrategy',
    'TypiClustStrategy',
    'AlphaMixStrategy',
    'MyStrategy',
    'MyStrategyWeightedReport',
    'OurMethodStrategy',
    'OurMethodBudgetVersionStrategy',
    'OurMethodMarginStrategy',
    'OurMethodMarginWeightedStrategy',
    'OurMethodMarginWeightedSqrtStrategy',
    'SALTExactStrategy',
    'SALTExactSoftStrategy',
    'OurMethodDistStrategy',
    'OurMethodMarginDistStrategy',
    'OurMethodMarginDistProxyStrategy',
    'OurMethodUnfilStrategy',
    'OurMethodMarginUnfilStrategy',
]
