"""
主动学习策略基类和工厂函数
"""
import numpy as np
import torch
from typing import List, Any, Dict


class ActiveLearningStrategy:
    """
    主动学习策略基类
    
    所有主动学习策略都应该继承此类并实现 select_samples 方法。
    """
    
    def __init__(self, name: str):
        """
        初始化策略
        
        参数:
            name: 策略名称
        """
        self.name = name
    
    def select_samples(
        self, 
        model: torch.nn.Module, 
        unlabeled_indices: List[int], 
        dataset: Any, 
        batch_size: int,
        device: torch.device
    ) -> List[int]:
        """
        选择要标注的样本
        
        参数:
            model: 训练好的模型
            unlabeled_indices: 未标注样本的索引列表
            dataset: 数据集
            batch_size: 要选择的样本数量
            device: 计算设备
            
        返回:
            List[int]: 选择的样本索引列表
        """
        raise NotImplementedError("子类必须实现此方法")

    def export_continuation_state(self) -> Dict[str, Any]:
        """
        导出不包含模型参数的最小策略续跑状态。

        基类只统一处理多个 OurMethod 策略共用的 `round_counter`。
        普通策略没有该属性时返回空字典。需要额外状态的
        子类应覆写本方法，并将返回值保持为可由 NumPy pickle
        序列化的 Python 容器与标量类型。

        返回:
            Dict[str, Any]: 可在新策略实例上恢复的内部状态。
        """
        state: Dict[str, Any] = {}
        if hasattr(self, "round_counter"):
            state["round_counter"] = int(getattr(self, "round_counter"))
        return state

    def restore_continuation_state(self, state: Dict[str, Any]) -> None:
        """
        恢复由 `export_continuation_state()` 导出的策略状态。

        恢复会在策略已按当前实验配置完成构造之后执行。
        基类不会为原本不支持轮次的策略动态新建属性；
        只有当实例本身已存在 `round_counter` 时才恢复，从而
        避免对 random、margin 等无状态策略产生额外语义。

        参数:
            state: 从续跑文件读取的策略状态字典。

        返回:
            None
        """
        if "round_counter" in state and hasattr(self, "round_counter"):
            restored_round = int(state["round_counter"])
            if restored_round < 0:
                raise ValueError("round_counter 不能为负数")
            setattr(self, "round_counter", restored_round)


def get_strategy(strategy_name: str, **kwargs: Any) -> ActiveLearningStrategy:
    """
    根据策略名称获取对应的策略实例
    
    参数:
        strategy_name (str): 策略名称，支持 'random', 'least_confidence', 'badge',
                        'entropy', 'margin', 'kcgreedy', 'uherding', 'typiclust', 'mystrategy',
                        'alpha_mix',
                        'mystrategy_weighted_report', 'our_method', 'our_method_budget_version',
                        'our_method_margin', 'our_method_margin_weighted',
                        'our_method_margin_weighted_sqrt', 'salt_exact', 'salt_exact_soft',
                        'our_method_dist', 'our_method_margin_dist',
                        'our_method_margin_dist_proxy',
                        'our_method_unfil', 'our_method_margin_unfil'
        **kwargs: 传递给策略构造函数的额外参数（仅对需要的策略生效）
        
    返回:
        ActiveLearningStrategy: 对应的策略实例
        
    异常:
        ValueError: 当策略名称不支持时抛出
    """
    # 延迟导入以避免循环依赖
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
    from .our_method_margin_weighted_sqrt import (
        OurMethodMarginWeightedSqrtStrategy,
    )
    from .salt_exact import SALTExactStrategy
    from .salt_exact_soft import SALTExactSoftStrategy
    from .our_method_dist import OurMethodDistStrategy
    from .our_method_margin_dist import OurMethodMarginDistStrategy
    from .our_method_margin_dist_proxy import OurMethodMarginDistProxyStrategy
    from .our_method_unfil import OurMethodUnfilStrategy
    from .our_method_margin_unfil import OurMethodMarginUnfilStrategy
    
    strategies = {
        'random': RandomStrategy,
        'least_confidence': LeastConfidenceStrategy,
        'badge': BADGEStrategy,
        'entropy': EntropyStrategy,
        'margin': MarginStrategy,
        'kcgreedy': KCGreedyStrategy,
        'uherding': UHerdingStrategy,
        'typiclust': TypiClustStrategy,
        'alpha_mix': AlphaMixStrategy,
        'mystrategy': MyStrategy,
        'mystrategy_weighted_report': MyStrategyWeightedReport,
        'our_method': OurMethodStrategy,
        'our_method_budget_version': OurMethodBudgetVersionStrategy,
        'our_method_margin': OurMethodMarginStrategy,
        'our_method_margin_weighted': OurMethodMarginWeightedStrategy,
        'our_method_margin_weighted_sqrt': OurMethodMarginWeightedSqrtStrategy,
        'salt_exact': SALTExactStrategy,
        'salt_exact_soft': SALTExactSoftStrategy,
        'our_method_dist': OurMethodDistStrategy,
        'our_method_margin_dist': OurMethodMarginDistStrategy,
        'our_method_margin_dist_proxy': OurMethodMarginDistProxyStrategy,
        'our_method_unfil': OurMethodUnfilStrategy,
        'our_method_margin_unfil': OurMethodMarginUnfilStrategy
    }
    
    if strategy_name not in strategies:
        raise ValueError(
            f"不支持的策略: {strategy_name}. "
            f"支持的策略有: {list(strategies.keys())}"
        )
    
    strategy_class = strategies[strategy_name]
    try:
        return strategy_class(**kwargs)
    except TypeError:
        # 对不接收额外参数的策略，回退到无参构造
        return strategy_class()
