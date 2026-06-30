"""
ALFA-Mix 主动学习策略的当前项目适配实现。

本文件按官方 `AlphaMixSampling` 的 `--alpha_opt` 路径实现核心采样逻辑：
1. 在模型最后一层线性分类器前提取 embedding；
2. 对未标注 embedding 与按类别聚合的已标注 embedding anchor 做 feature mixing；
3. 通过优化 alpha 寻找会造成预测类别变化的候选点；
4. 对候选点 embedding 做 KMeans，并选择离各簇中心最近的样本。

官方代码参考：
`alpha_mix_active_learning-main/query_strategies/alpha_mix_sampling.py`
"""
from __future__ import annotations

import copy
import math
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, Subset

from .base import ActiveLearningStrategy


class AlphaMixStrategy(ActiveLearningStrategy):
    """
    官方 ALFA-Mix `alpha_opt` 分支的项目内适配策略。

    当前项目模型没有官方实现中的 `model.clf(x, embedding=True)` 接口，但所有
    已注册模型的最后分类器都是单个 `nn.Linear`。因此本策略通过 forward
    pre-hook 抓取最后一个 Linear 的输入作为 embedding，并用同一个 Linear
    对 mixed embedding 直接分类，等价替代官方 wrapper 的 embedding 模式。
    """

    def __init__(
        self,
        alpha_cap: float = 0.03125,
        alpha_learning_rate: float = 0.1,
        alpha_clf_coef: float = 1.0,
        alpha_l2_coef: float = 0.01,
        alpha_learning_iters: int = 5,
        alpha_learn_batch_size: int = 1_000_000,
        inference_batch_size: int = 256,
        random_state: Optional[int] = None,
        max_alpha: float = 1.0,
    ) -> None:
        """
        初始化 ALFA-Mix 策略。

        这些参数沿用官方 `main.py` 中 `--alpha_opt` 路径的默认值；本策略不实现
        closed-form 近似分支，避免与 README 推荐的 `--alpha_opt` 实验设置混淆。
        `random_state` 默认保持为 `None`，使 KMeans 和随机补齐行为与官方代码的
        非固定随机种子语义一致；若实验需要完全可复现，可显式传入整数种子。

        参数:
            alpha_cap: 每次外层搜索增加的 alpha 上界步长，对应官方 `--alpha_cap`。
            alpha_learning_rate: 优化 alpha 时使用的 Adam 初始学习率。
            alpha_clf_coef: 分类损失项系数；官方优化目标中该项为负交叉熵。
            alpha_l2_coef: alpha L2 范数惩罚系数。
            alpha_learning_iters: 每个类别 anchor 上优化 alpha 的迭代次数。
            alpha_learn_batch_size: 优化 alpha 时每个内部批次最多包含的样本数。
            inference_batch_size: 预测概率和 embedding 时 DataLoader 的批大小。
            random_state: KMeans 与随机补齐使用的随机种子；`None` 表示不固定。
            max_alpha: 外层 alpha_cap 搜索的最大上界，官方逻辑为搜索到 1.0。

        返回:
            None
        """
        super().__init__("AlphaMix")
        if alpha_cap <= 0.0:
            raise ValueError("alpha_cap 必须是正数")
        if alpha_learning_rate <= 0.0:
            raise ValueError("alpha_learning_rate 必须是正数")
        if alpha_learning_iters <= 0:
            raise ValueError("alpha_learning_iters 必须是正整数")
        if alpha_learn_batch_size <= 0:
            raise ValueError("alpha_learn_batch_size 必须是正整数")
        if inference_batch_size <= 0:
            raise ValueError("inference_batch_size 必须是正整数")
        if max_alpha <= 0.0:
            raise ValueError("max_alpha 必须是正数")

        self.alpha_cap = float(alpha_cap)
        self.alpha_learning_rate = float(alpha_learning_rate)
        self.alpha_clf_coef = float(alpha_clf_coef)
        self.alpha_l2_coef = float(alpha_l2_coef)
        self.alpha_learning_iters = int(alpha_learning_iters)
        self.alpha_learn_batch_size = int(alpha_learn_batch_size)
        self.inference_batch_size = int(inference_batch_size)
        self.random_state = random_state
        self.max_alpha = float(max_alpha)
        self.query_count = 0

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
        按 ALFA-Mix `alpha_opt` 规则选择待标注样本。

        本方法复刻官方 `AlphaMixSampling.query(...)` 的主体流程：先获取未标注点
        概率与 embedding，再获取已标注点 embedding；随后逐步增大 `alpha_cap`，
        在每个类别 anchor 上优化 alpha 并收集预测变化候选点。候选点足够时使用
        KMeans 选择多样化 batch，候选不足时按官方逻辑从剩余未标注池随机补齐。

        参数:
            model: 当前轮训练后的模型。
            unlabeled_indices: 当前未标注样本的全局索引列表。
            dataset: 完整训练集对象。
            batch_size: 本轮需要选择的样本数量。
            device: 当前计算设备。
            labeled_indices: 当前已标注样本的全局索引列表。
            **kwargs: 兼容统一策略接口的额外参数，当前未使用。

        返回:
            List[int]: 选中的样本全局索引列表，长度不超过 `batch_size`。
        """
        del kwargs
        self.query_count += 1

        if batch_size <= 0 or not unlabeled_indices:
            return []
        if len(unlabeled_indices) <= batch_size:
            return list(unlabeled_indices)

        labeled_indices = list(labeled_indices or [])
        if not labeled_indices:
            return self._sample_random_unlabeled(unlabeled_indices, batch_size)

        classifier = self._get_classifier(model)
        model.eval()
        ulb_probs, org_ulb_embedding = self._predict_prob_embed(
            model=model,
            dataset=dataset,
            indices=unlabeled_indices,
            device=device,
            classifier=classifier,
        )
        _, probs_sort_idxs = ulb_probs.sort(descending=True)
        pred_1 = probs_sort_idxs[:, 0]

        _, lb_embedding = self._predict_prob_embed(
            model=model,
            dataset=dataset,
            indices=labeled_indices,
            device=device,
            classifier=classifier,
        )
        labeled_targets = self._collect_labels(
            dataset=dataset,
            indices=labeled_indices,
        )
        n_label = int(ulb_probs.shape[1])

        unlabeled_size = int(org_ulb_embedding.size(0))
        embedding_size = int(org_ulb_embedding.size(1))
        min_alphas = torch.ones((unlabeled_size, embedding_size), dtype=torch.float)
        candidate = torch.zeros(unlabeled_size, dtype=torch.bool)

        alpha_cap = 0.0
        while alpha_cap < self.max_alpha:
            alpha_cap = min(self.max_alpha, alpha_cap + self.alpha_cap)
            tmp_pred_change, tmp_min_alphas = self._find_candidate_set(
                classifier=classifier,
                lb_embedding=lb_embedding,
                ulb_embedding=org_ulb_embedding,
                pred_1=pred_1,
                alpha_cap=alpha_cap,
                labels=labeled_targets,
                n_label=n_label,
                device=device,
            )

            is_changed = min_alphas.norm(dim=1) >= tmp_min_alphas.norm(dim=1)
            min_alphas[is_changed] = tmp_min_alphas[is_changed]
            candidate |= tmp_pred_change

            print(
                "[AlphaMix] "
                f"alpha_cap={alpha_cap:.6f}, "
                f"inconsistencies={int(tmp_pred_change.sum().item())}, "
                f"candidate_total={int(candidate.sum().item())}"
            )

            if int(candidate.sum().item()) > batch_size:
                break

        if int(candidate.sum().item()) > 0:
            candidate_count = int(candidate.sum().item())
            print(f"[AlphaMix] Number of inconsistencies: {candidate_count}")
            candidate_feats = F.normalize(
                org_ulb_embedding[candidate].view(candidate_count, -1),
                p=2,
                dim=1,
            ).detach()
            selected_candidate_positions = self._sample_by_kmeans(
                n=min(batch_size, candidate_count),
                feats=candidate_feats,
            )
            candidate_unlabeled_positions = candidate.nonzero(as_tuple=True)[0]
            selected_unlabeled_positions = candidate_unlabeled_positions[
                selected_candidate_positions
            ].cpu().numpy()
            selected_indices = [
                int(unlabeled_indices[position])
                for position in selected_unlabeled_positions.tolist()
            ]
        else:
            selected_indices = []

        if len(selected_indices) < batch_size:
            remained = batch_size - len(selected_indices)
            selected_set = set(selected_indices)
            remaining_pool = [
                idx for idx in unlabeled_indices
                if idx not in selected_set
            ]
            selected_indices.extend(self._sample_random_unlabeled(remaining_pool, remained))
            print(f"[AlphaMix] picked {remained} samples from random sampling.")

        return selected_indices[:batch_size]

    def _get_classifier(self, model: torch.nn.Module) -> nn.Linear:
        """
        获取模型最后一个线性分类器。

        ALFA-Mix 需要在 embedding 空间做 mix 后只运行分类器。当前项目中所有模型
        都以最后一个 `nn.Linear` 直接输出 logits，因此这里按模块遍历顺序取最后
        一个线性层作为官方 `clf(..., embedding=True)` 的分类头替代。

        参数:
            model: 当前训练好的模型。

        返回:
            nn.Linear: 模型最后一个线性层。

        异常:
            ValueError: 当模型中没有线性层，或最后线性层不是 `nn.Linear` 时抛出。
        """
        last_linear: Optional[nn.Linear] = None
        for module in model.modules():
            if isinstance(module, nn.Linear):
                last_linear = module
        if last_linear is None:
            raise ValueError("AlphaMix 需要模型包含至少一个 nn.Linear 分类层")
        return last_linear

    def _predict_prob_embed(
        self,
        model: torch.nn.Module,
        dataset: Any,
        indices: Sequence[int],
        device: torch.device,
        classifier: nn.Linear,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        预测样本概率并提取最后线性分类器输入 embedding。

        该函数等价替代官方 wrapper 的 `predict_prob_embed(...)`：在 `model.eval()`
        下对指定样本做前向传播，使用 forward pre-hook 捕获最后一层 Linear 的输入
        作为 embedding，并返回 softmax 概率与 embedding，二者都放在 CPU 上以便
        后续按官方逻辑统一处理。

        参数:
            model: 当前训练好的模型。
            dataset: 完整训练集对象。
            indices: 需要预测的样本全局索引序列。
            device: 当前计算设备。
            classifier: 最后一层线性分类器。

        返回:
            Tuple[torch.Tensor, torch.Tensor]:
                - probabilities: 概率矩阵，形状为 `(len(indices), num_classes)`；
                - embeddings: 最后线性层输入，形状为 `(len(indices), embedding_dim)`。
        """
        if not indices:
            return torch.empty((0, 0)), torch.empty((0, 0))

        model.eval()
        subset = Subset(dataset, list(indices))
        loader = DataLoader(
            subset,
            batch_size=min(self.inference_batch_size, len(indices)),
            shuffle=False,
        )
        probability_chunks: List[torch.Tensor] = []
        embedding_chunks: List[torch.Tensor] = []
        captured_embeddings: List[torch.Tensor] = []

        def hook_fn(module: nn.Module, inputs: Tuple[torch.Tensor, ...]) -> None:
            """
            捕获最后线性分类器的输入作为 ALFA-Mix embedding。

            参数:
                module: 触发 hook 的分类器模块，当前仅用于符合 PyTorch hook 签名。
                inputs: 传入分类器的参数元组，第一项是分类器输入特征。

            返回:
                None
            """
            del module
            captured_embeddings.append(inputs[0].detach().cpu())

        handle = classifier.register_forward_pre_hook(hook_fn)
        try:
            with torch.no_grad():
                for batch in loader:
                    inputs = self._extract_inputs_from_batch(batch).to(device)
                    captured_embeddings.clear()
                    logits = model(inputs)
                    if not captured_embeddings:
                        raise RuntimeError("AlphaMix 未能捕获最后线性层输入 embedding")
                    probability_chunks.append(F.softmax(logits, dim=1).detach().cpu())
                    embedding_chunks.append(captured_embeddings[-1])
        finally:
            handle.remove()

        return torch.cat(probability_chunks, dim=0), torch.cat(embedding_chunks, dim=0)

    def _collect_labels(
        self,
        dataset: Any,
        indices: Sequence[int],
    ) -> torch.Tensor:
        """
        收集指定索引样本的硬标签。

        官方 ALFA-Mix 在构造类别 anchor 时使用已标注样本的真实类别 `Y`。当前项目
        数据集可能返回 tensor 或 Python 标量标签，本函数统一转成一维 LongTensor。

        参数:
            dataset: 完整训练集对象。
            indices: 需要读取标签的全局索引序列。

        返回:
            torch.Tensor: 与 `indices` 顺序一致的标签张量。
        """
        labels: List[int] = []
        for index in indices:
            sample = dataset[int(index)]
            if isinstance(sample, (tuple, list)):
                label = sample[1]
            else:
                raise TypeError("AlphaMix 期望数据集样本包含输入和标签")
            if isinstance(label, torch.Tensor):
                labels.append(int(label.item()))
            else:
                labels.append(int(label))
        return torch.tensor(labels, dtype=torch.long)

    def _find_candidate_set(
        self,
        classifier: nn.Linear,
        lb_embedding: torch.Tensor,
        ulb_embedding: torch.Tensor,
        pred_1: torch.Tensor,
        alpha_cap: float,
        labels: torch.Tensor,
        n_label: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        复刻官方 `find_candidate_set(...)` 的 `alpha_opt` 分支。

        对每个类别，先用该类别已标注 embedding 的均值作为 anchor；若某个类别在
        当前已标注集中不存在，则回退为所有已标注 embedding 的均值。随后生成随机
        alpha，并调用 `_learn_alpha(...)` 优化 alpha，使 mixed embedding 尽量诱导
        分类器预测变化，同时保留 alpha 范数最小的变化结果。

        参数:
            classifier: 最后一层线性分类器。
            lb_embedding: 已标注样本 embedding，位于 CPU。
            ulb_embedding: 未标注样本 embedding，位于 CPU。
            pred_1: 未标注样本原始预测类别，位于 CPU。
            alpha_cap: 当前外层搜索使用的 alpha 上界。
            labels: 已标注样本真实标签，位于 CPU。
            n_label: 类别数量。
            device: 当前计算设备。

        返回:
            Tuple[torch.Tensor, torch.Tensor]:
                - pred_change: 每个未标注样本是否在任一类别 anchor 下预测变化；
                - min_alphas: 每个样本当前找到的最小范数 alpha。
        """
        unlabeled_size = int(ulb_embedding.size(0))
        embedding_size = int(ulb_embedding.size(1))
        min_alphas = torch.ones((unlabeled_size, embedding_size), dtype=torch.float)
        pred_change = torch.zeros(unlabeled_size, dtype=torch.bool)

        for class_index in range(n_label):
            class_embedding = lb_embedding[labels == class_index]
            if class_embedding.size(0) == 0:
                class_embedding = lb_embedding
            anchor_i = class_embedding.mean(dim=0).view(1, -1).repeat(unlabeled_size, 1)

            alpha = self._generate_alpha(
                size=unlabeled_size,
                embedding_size=embedding_size,
                alpha_cap=alpha_cap,
            )
            alpha, pc = self._learn_alpha(
                classifier=classifier,
                org_embed=ulb_embedding,
                labels=pred_1,
                anchor_embed=anchor_i,
                alpha=alpha,
                alpha_cap=alpha_cap,
                device=device,
            )

            alpha[~pc] = 1.0
            pred_change[pc] = True
            is_min = min_alphas.norm(dim=1) > alpha.norm(dim=1)
            min_alphas[is_min] = alpha[is_min]
            print(
                "[AlphaMix] "
                f"class={class_index}, inconsistencies={int(pc.sum().item())}"
            )

        return pred_change, min_alphas

    def _generate_alpha(
        self,
        size: int,
        embedding_size: int,
        alpha_cap: float,
    ) -> torch.Tensor:
        """
        按官方逻辑生成随机 alpha 初值。

        官方 `generate_alpha(...)` 从均值和标准差都为 `alpha_cap / 2` 的正态分布
        采样，再将结果截断到 `[1e-8, alpha_cap]`。本函数保持该初始化语义。

        参数:
            size: 当前未标注候选数量。
            embedding_size: embedding 维度。
            alpha_cap: 当前 alpha 上界。

        返回:
            torch.Tensor: 形状为 `(size, embedding_size)` 的 alpha 初值。
        """
        alpha = torch.normal(
            mean=alpha_cap / 2.0,
            std=alpha_cap / 2.0,
            size=(size, embedding_size),
        )
        alpha[torch.isnan(alpha)] = 1.0
        return self._clamp_alpha(alpha, alpha_cap)

    def _clamp_alpha(
        self,
        alpha: torch.Tensor,
        alpha_cap: float,
    ) -> torch.Tensor:
        """
        将 alpha 截断到官方使用的合法区间。

        参数:
            alpha: 需要截断的 alpha 张量。
            alpha_cap: 当前 alpha 上界。

        返回:
            torch.Tensor: 截断到 `[1e-8, alpha_cap]` 的 alpha 张量。
        """
        return torch.clamp(alpha, min=1e-8, max=alpha_cap)

    def _learn_alpha(
        self,
        classifier: nn.Linear,
        org_embed: torch.Tensor,
        labels: torch.Tensor,
        anchor_embed: torch.Tensor,
        alpha: torch.Tensor,
        alpha_cap: float,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        复刻官方 `learn_alpha(...)` 的 alpha 优化过程。

        对每个类别 anchor，函数把 alpha 作为待优化变量，在 mixed embedding 上
        运行分类器。优化目标与官方一致：
        `alpha_clf_coef * (-CE(out, original_pred)) + alpha_l2_coef * ||alpha||_2`。
        一旦某个样本预测类别改变，就记录该样本当前范数更小的 alpha，并在所有
        迭代结束后返回每个样本的最小变化 alpha 与预测变化掩码。

        参数:
            classifier: 最后一层线性分类器。
            org_embed: 未标注样本原始 embedding，位于 CPU。
            labels: 未标注样本原始预测类别，位于 CPU。
            anchor_embed: 当前类别 anchor embedding，位于 CPU。
            alpha: 当前 alpha 初值，位于 CPU。
            alpha_cap: 当前 alpha 上界。
            device: 当前计算设备。

        返回:
            Tuple[torch.Tensor, torch.Tensor]:
                - min_alpha: 对每个样本记录的最小范数变化 alpha，位于 CPU；
                - pred_changed: 每个样本是否发生过预测变化，位于 CPU。
        """
        labels = labels.to(device)
        min_alpha = torch.ones(alpha.size(), dtype=torch.float)
        pred_changed = torch.zeros(labels.size(0), dtype=torch.bool)
        loss_func = torch.nn.CrossEntropyLoss(reduction="none")
        classifier.eval()

        for iteration in range(self.alpha_learning_iters):
            for batch_index in range(
                math.ceil(float(alpha.size(0)) / self.alpha_learn_batch_size)
            ):
                classifier.zero_grad()
                start_idx = batch_index * self.alpha_learn_batch_size
                end_idx = min(
                    (batch_index + 1) * self.alpha_learn_batch_size,
                    alpha.size(0),
                )
                learning_rate = self.alpha_learning_rate / (
                    1.0 if iteration < self.alpha_learning_iters * 2 / 3 else 10.0
                )

                l = alpha[start_idx:end_idx].to(device).detach().requires_grad_(True)
                optimizer = torch.optim.Adam([l], lr=learning_rate)
                e = org_embed[start_idx:end_idx].to(device)
                c_e = anchor_embed[start_idx:end_idx].to(device)
                embedding_mix = (1 - l) * e + l * c_e
                out = classifier(embedding_mix)

                label_change = out.argmax(dim=1) != labels[start_idx:end_idx]
                tmp_pc = torch.zeros(labels.size(0), dtype=torch.bool, device=device)
                tmp_pc[start_idx:end_idx] = label_change
                pred_changed[start_idx:end_idx] |= label_change.detach().cpu()

                tmp_pc[start_idx:end_idx] = tmp_pc[start_idx:end_idx] * (
                    l.norm(dim=1)
                    < min_alpha[start_idx:end_idx].norm(dim=1).to(device)
                )
                min_alpha[tmp_pc.cpu()] = l[tmp_pc[start_idx:end_idx]].detach().cpu()

                clf_loss = loss_func(out, labels[start_idx:end_idx])
                l2_nrm = torch.norm(l, dim=1)
                clf_loss *= -1
                loss = self.alpha_clf_coef * clf_loss + self.alpha_l2_coef * l2_nrm
                loss.sum().backward(retain_graph=True)
                optimizer.step()

                l = self._clamp_alpha(l, alpha_cap)
                alpha[start_idx:end_idx] = l.detach().cpu()

                del l, e, c_e, embedding_mix, out, optimizer
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        return min_alpha.cpu(), pred_changed.cpu()

    def _sample_by_kmeans(
        self,
        n: int,
        feats: torch.Tensor,
    ) -> torch.Tensor:
        """
        对候选 embedding 做 KMeans，并返回每个簇中离中心最近的候选位置。

        该函数保持官方 `sample(...)` 的语义：先对候选特征聚类，再在每个非空簇中
        选离该簇中心最近的样本。返回的位置是候选集合内部的局部位置，而不是完整
        未标注池索引。

        参数:
            n: 需要从候选集中选择的样本数量。
            feats: 候选样本的归一化 embedding，形状为 `(candidate_count, dim)`。

        返回:
            torch.Tensor: 候选集合内部的被选位置。
        """
        if n <= 0:
            return torch.empty((0,), dtype=torch.long)
        if feats.size(0) <= n:
            return torch.arange(feats.size(0), dtype=torch.long)

        feats_np = feats.numpy()
        cluster_learner = KMeans(n_clusters=n, random_state=self.random_state)
        cluster_learner.fit(feats_np)
        cluster_idxs = cluster_learner.predict(feats_np)
        centers = cluster_learner.cluster_centers_[cluster_idxs]
        distances = ((feats_np - centers) ** 2).sum(axis=1)
        selected_positions = [
            np.arange(feats_np.shape[0])[cluster_idxs == cluster_id][
                distances[cluster_idxs == cluster_id].argmin()
            ]
            for cluster_id in range(n)
            if (cluster_idxs == cluster_id).sum() > 0
        ]
        return torch.tensor(selected_positions, dtype=torch.long)

    def _sample_random_unlabeled(
        self,
        unlabeled_indices: Sequence[int],
        sample_count: int,
    ) -> List[int]:
        """
        从未标注池中随机选择指定数量样本，用于候选不足时的官方随机补齐逻辑。

        参数:
            unlabeled_indices: 可供随机选择的未标注样本索引序列。
            sample_count: 需要选择的样本数量。

        返回:
            List[int]: 随机选出的样本索引列表。
        """
        if sample_count <= 0 or not unlabeled_indices:
            return []
        sample_count = min(sample_count, len(unlabeled_indices))
        if self.random_state is None:
            positions = np.random.choice(
                len(unlabeled_indices),
                size=sample_count,
                replace=False,
            )
        else:
            rng = np.random.default_rng(self.random_state)
            positions = rng.choice(
                len(unlabeled_indices),
                size=sample_count,
                replace=False,
            )
        return [int(unlabeled_indices[int(position)]) for position in positions.tolist()]

    def _extract_inputs_from_batch(
        self,
        batch: Any,
    ) -> torch.Tensor:
        """
        从 DataLoader 批数据中提取输入张量。

        当前项目数据集通常返回 `(inputs, labels)`，但部分包装器可能返回更长元组。
        ALFA-Mix 只需要输入张量做模型前向，因此统一取第一个元素。

        参数:
            batch: DataLoader 返回的单个批次对象。

        返回:
            torch.Tensor: 当前批次输入张量。

        异常:
            TypeError: 当批数据无法解析出输入张量时抛出。
        """
        if isinstance(batch, torch.Tensor):
            return batch
        if isinstance(batch, (tuple, list)) and batch:
            return batch[0]
        raise TypeError("AlphaMix 期望 DataLoader 返回张量或以输入为首项的元组/列表")
