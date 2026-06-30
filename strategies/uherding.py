"""
UHerding 策略。

该实现是对 `uherding-main` 中 `uherding_margin` 的项目内适配版本，
专门面向当前仓库的主动学习接口做了如下简化：
1. 仅实现 margin 不确定性；
2. 相似度直接基于当前数据集中样本输入向量计算；
3. 不引入固定 valSet 与 checkpoint 逻辑；
4. 可选地在当前已标注集内部切出 30% 样本，用于临时模型的 temperature scaling。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from model import get_model

from .base import ActiveLearningStrategy


class UHerdingStrategy(ActiveLearningStrategy):
    """
    UHerding 主动学习策略。

    该策略复现 UHerding 的核心思想：在候选样本集合上，使用“margin 不确定性”
    作为权重，对每个候选点带来的“新增覆盖增益”进行加权评分；随后使用贪心方式，
    逐个选择得分最高的样本。

    与原始仓库相比，本实现有两个重要适配点：
    1. 覆盖空间直接使用当前数据集中的输入向量，而不是额外从模型中抽取特征；
    2. 若启用 temperature scaling，则在当前 `labeled_indices` 内部切出一部分样本，
       临时训练一个模型，仅用于估计温度参数；不会保存 checkpoint，也不会维护独立 valSet。
    """

    def __init__(
        self,
        temp_scale: bool = False,
        dataset_name: Optional[str] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        temp_train_epochs: int = 10,
        temp_learning_rate: float = 1e-3,
        temp_batch_size: int = 64,
        inference_batch_size: int = 256,
        candidate_max_size: int = 35000,
        temp_values: Optional[Sequence[float]] = None,
        ece_num_bins: int = 15,
    ):
        """
        初始化 UHerding 策略。

        参数:
            temp_scale: 是否启用 temperature scaling。
                        为 True 时，策略会在当前已标注集中切出一部分样本，
                        训练临时模型并搜索最优温度。
            dataset_name: 当前实验使用的数据集名称。临时模型重建时需要该信息。
            model_kwargs: 传给 `get_model(dataset_name, **model_kwargs)` 的参数。
                         对文本数据集，该字典通常包含 `input_dim` 与 `num_classes`。
            temp_train_epochs: 临时模型的训练轮数。
            temp_learning_rate: 临时模型训练学习率。
            temp_batch_size: 临时模型训练及温度估计时使用的 batch size。
            inference_batch_size: 预测 logits、读取向量和构造核矩阵时使用的 batch size。
            candidate_max_size: 未标注候选子集的最大大小。
                                该上界用于限制核矩阵规模，避免在大数据集上 OOM。
            temp_values: 可选的温度搜索网格；若不提供，则默认使用 [1.0, 20.0) 步长 0.1。
            ece_num_bins: 估计 Expected Calibration Error 时的分箱数。
        """
        super().__init__("UHerding")
        self.temp_scale = bool(temp_scale)
        self.dataset_name = dataset_name
        self.model_kwargs = dict(model_kwargs or {})
        self.temp_train_epochs = int(temp_train_epochs)
        self.temp_learning_rate = float(temp_learning_rate)
        self.temp_batch_size = int(temp_batch_size)
        self.inference_batch_size = int(inference_batch_size)
        self.candidate_max_size = int(candidate_max_size)
        self.temp_values = (
            np.array(list(temp_values), dtype=np.float32)
            if temp_values is not None
            else np.arange(1.0, 20.0, 0.1, dtype=np.float32)
        )
        self.ece_num_bins = int(ece_num_bins)

    def select_samples(
        self,
        model: torch.nn.Module,
        unlabeled_indices: List[int],
        dataset: Any,
        batch_size: int,
        device: torch.device,
        labeled_indices: Optional[List[int]] = None,
        initial_model_state: Optional[Dict[str, torch.Tensor]] = None,
    ) -> List[int]:
        """
        从未标注池中选择一批样本。

        该函数执行的核心步骤如下：
        1. 根据当前已标注规模和查询预算，截取一个未标注候选子集；
        2. 从数据集中读取 `labeled_indices + candidate_unlabeled_indices` 对应的原始输入向量；
        3. 根据当前已标注样本的最小非对角距离，动态计算本轮 RBF 核带宽 `delta`；
        4. 在这些向量上构造 RBF 核矩阵；
        5. 使用当前模型对同一批样本做预测，得到 margin-based uncertainty；
        6. 以“不确定性加权的新增覆盖增益”为分数，贪心选取 batch_size 个样本；
        7. 返回映射回原始训练集的样本索引。

        参数:
            model: 当前主动学习轮次已经训练完成的模型。
            unlabeled_indices: 当前未标注池中的样本索引列表。
            dataset: 完整训练集对象。对于文本实验，这通常是 embedding 后的 TensorDataset。
            batch_size: 本轮需要查询的样本数量。
            device: 计算设备。
            labeled_indices: 当前已标注样本索引列表。用于初始化覆盖状态和温度估计。
            initial_model_state: 当前轮主模型的初始参数快照。
                                 若启用 temperature scaling，该状态会用于初始化临时模型，
                                 以贴近原始 UHerding 的实现方式。

        返回:
            List[int]: 选中的原始训练集索引。
        """
        if len(unlabeled_indices) <= batch_size:
            return list(unlabeled_indices)

        labeled_indices = list(labeled_indices or [])
        candidate_indices = self._sample_candidate_subset(
            unlabeled_indices=unlabeled_indices,
            labeled_size=len(labeled_indices),
            budget_size=batch_size,
        )
        relevant_indices = labeled_indices + candidate_indices

        relevant_vectors = self._collect_dataset_vectors(
            dataset=dataset,
            indices=relevant_indices,
        ).to(torch.float32)
        delta = self._compute_delta_from_labeled_vectors(
            relevant_vectors=relevant_vectors,
            labeled_size=len(labeled_indices),
            device=device,
        )
        kernel_all = self._compute_rbf_kernel_matrix(
            vectors=relevant_vectors,
            delta=delta,
            batch_size=self.inference_batch_size,
            device=device,
        )

        if labeled_indices:
            temperature = self._estimate_temperature_if_needed(
                dataset=dataset,
                labeled_indices=labeled_indices,
                current_model=model,
                device=device,
                initial_model_state=initial_model_state,
            )
        else:
            temperature = 1.0

        uncertainties = self._compute_margin_uncertainties(
            model=model,
            dataset=dataset,
            indices=relevant_indices,
            labeled_size=len(labeled_indices),
            temperature=temperature,
            batch_size=self.inference_batch_size,
            device=device,
        )

        selected_relative_indices = self._greedy_select_with_uncertainty_coverage(
            kernel_all=kernel_all,
            uncertainties=uncertainties,
            labeled_size=len(labeled_indices),
            budget_size=batch_size,
            device=device,
        )
        return [relevant_indices[idx] for idx in selected_relative_indices]

    def _sample_candidate_subset(
        self,
        unlabeled_indices: List[int],
        labeled_size: int,
        budget_size: int,
    ) -> List[int]:
        """
        从未标注集合中抽取候选子集。

        原始 UHerding 在大数据集上不会直接在整个 `uSet` 上构造核矩阵，
        而是先依据当前已标注规模与预算大小估算一个候选子集大小；随后对该子集
        做精确的贪心选择。本函数沿用相同思想，但额外加入 `candidate_max_size`
        上界，以适配当前项目中较大的文本 embedding 数据集。

        参数:
            unlabeled_indices: 当前未标注池索引。
            labeled_size: 当前已标注样本数。
            budget_size: 本轮查询预算。

        返回:
            List[int]: 候选未标注索引子集。
        """
        subset_size = self._compute_candidate_size(
            labeled_size=labeled_size,
            budget_size=budget_size,
            max_size=self.candidate_max_size,
        )
        subset_size = min(subset_size, len(unlabeled_indices))
        permuted = np.random.permutation(np.asarray(unlabeled_indices, dtype=np.int64))
        return permuted[:subset_size].tolist()

    def _collect_dataset_vectors(
        self,
        dataset: Any,
        indices: List[int],
    ) -> torch.Tensor:
        """
        从数据集中读取指定索引对应的输入向量。

        这里刻意不从模型中抽取中间特征，而是直接使用数据集中的输入：
        - 对文本嵌入数据集而言，输入本身就是预先离线生成的 embedding；
        - 对图像数据集而言，输入是图像张量，本函数会将其展平为一维向量。

        参数:
            dataset: 完整训练集对象。
            indices: 需要读取的样本索引列表。

        返回:
            torch.Tensor: 样本输入向量矩阵，形状为 `(len(indices), feature_dim)`。
        """
        subset = Subset(dataset, indices)
        loader = DataLoader(subset, batch_size=self.inference_batch_size, shuffle=False)

        vectors: List[torch.Tensor] = []
        for batch_inputs, _ in loader:
            batch_inputs = batch_inputs.view(batch_inputs.size(0), -1).cpu()
            vectors.append(batch_inputs)

        if not vectors:
            return torch.empty((0, 0), dtype=torch.float32)
        return torch.cat(vectors, dim=0)

    def _compute_delta_from_labeled_vectors(
        self,
        relevant_vectors: torch.Tensor,
        labeled_size: int,
        device: torch.device,
    ) -> float:
        """
        根据当前已标注样本动态计算 UHerding 所需的 RBF 核带宽 `delta`。

        该实现遵循原文中的设定：`delta` 取当前已标注集合内部的“最小非自身距离”。
        具体来说，先取 `relevant_vectors` 的前 `labeled_size` 行作为当前轮的已标注向量，
        再计算这些向量的两两欧氏距离矩阵；随后将对角线位置屏蔽为正无穷，
        以排除样本与自身的距离；最后返回整个距离矩阵中的最小值。

        参数:
            relevant_vectors: 当前轮参与 UHerding 计算的向量矩阵，
                              其中前 `labeled_size` 行必须对应已标注样本。
            labeled_size: 当前已标注样本数。
            device: 计算距离矩阵时使用的设备。

        返回:
            float: 当前轮动态计算得到的 RBF 核带宽 `delta`。
        """
        labeled_vectors = relevant_vectors[:labeled_size].to(device)
        labeled_distances = torch.cdist(labeled_vectors, labeled_vectors, p=2.0)
        labeled_distances.fill_diagonal_(torch.inf)
        return float(labeled_distances.min().item())

    def _compute_rbf_kernel_matrix(
        self,
        vectors: torch.Tensor,
        delta: float,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        基于输入向量计算完整的 RBF 核矩阵。

        核定义为：
            `k(x_i, x_j) = exp(-(||x_i - x_j||_2 / delta)^2)`

        为了避免一次性把大矩阵都放到设备上，本函数采用“按行分块计算距离”的方式：
        每次取一小批 query 向量，与全部向量计算 `torch.cdist`，得到一段距离矩阵，
        再转换为对应的核相似度并拼接回 CPU。

        参数:
            vectors: 输入向量矩阵，形状 `(N, D)`。
            delta: 当前轮动态计算得到的 RBF 核带宽。
            batch_size: 行分块大小。
            device: 计算设备。

        返回:
            torch.Tensor: CPU 上的核矩阵，形状 `(N, N)`。
        """
        if vectors.numel() == 0:
            return torch.empty((0, 0), dtype=torch.float32)

        all_vectors = vectors.to(device)
        kernel_chunks: List[torch.Tensor] = []
        for start in range(0, all_vectors.size(0), batch_size):
            end = min(start + batch_size, all_vectors.size(0))
            query = all_vectors[start:end]
            distances = torch.cdist(query, all_vectors, p=2.0)
            kernel = torch.exp(-torch.square(distances / delta))
            kernel_chunks.append(kernel.cpu())
        return torch.cat(kernel_chunks, dim=0)

    def _compute_margin_uncertainties(
        self,
        model: torch.nn.Module,
        dataset: Any,
        indices: List[int],
        labeled_size: int,
        temperature: float,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        计算 UHerding 所需的 margin 不确定性。

        具体定义为：
            `uncertainty = 1 - (p_top1 - p_top2)`

        其中 `p_top1` 和 `p_top2` 分别是模型输出概率中最大的两个值。
        与原始实现一致，当前已标注样本的 uncertainty 会被强制置为 0，
        避免它们继续作为“需要被覆盖的不确定目标”参与评分。

        参数:
            model: 当前轮训练好的模型。
            dataset: 完整训练集对象。
            indices: 需要计算不确定性的样本索引。
            labeled_size: `indices` 中前多少个样本属于已标注集合。
            temperature: logits 的温度缩放参数。
            batch_size: 推理 batch size。
            device: 计算设备。

        返回:
            torch.Tensor: 形状为 `(1, N)` 的 uncertainty 权重张量。
        """
        if labeled_size == 0:
            return torch.ones((1, len(indices)), dtype=torch.float32, device=device)

        model.eval()
        subset = Subset(dataset, indices)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=False)

        uncertainty_chunks: List[torch.Tensor] = []
        with torch.no_grad():
            for batch_inputs, _ in loader:
                batch_inputs = batch_inputs.to(device)
                logits = model(batch_inputs) / temperature
                probabilities = torch.softmax(logits, dim=1)
                top2_probs, _ = torch.topk(probabilities, k=2, dim=1)
                batch_uncertainty = 1.0 - (top2_probs[:, 0] - top2_probs[:, 1])
                uncertainty_chunks.append(batch_uncertainty)

        uncertainties = torch.cat(uncertainty_chunks, dim=0).to(device)
        uncertainties[:labeled_size] = 0.0
        return uncertainties.view(1, -1)

    def _greedy_select_with_uncertainty_coverage(
        self,
        kernel_all: torch.Tensor,
        uncertainties: torch.Tensor,
        labeled_size: int,
        budget_size: int,
        device: torch.device,
    ) -> List[int]:
        """
        使用“不确定性加权覆盖增益”进行贪心选样。

        设当前已被选择的集合为 `S`，则对候选点 `i` 的评分定义为：
            `score(i) = mean_j [ u_j * max(0, k(i, j) - m_j) ]`
        其中：
            - `u_j` 是第 `j` 个点的不确定性；
            - `m_j` 是当前集合 `S` 对第 `j` 个点的最大覆盖值；
            - `k(i, j)` 是候选点 `i` 与第 `j` 个点的核相似度。

        贪心过程每轮执行：
        1. 计算每个候选点带来的新增覆盖；
        2. 用不确定性进行加权平均；
        3. 选择得分最高的未标注点；
        4. 更新当前最大覆盖状态。

        参数:
            kernel_all: 所有 relevant 样本之间的核矩阵，形状 `(N, N)`。
            uncertainties: uncertainty 权重，形状 `(1, N)`。
            labeled_size: 前多少个样本属于初始已标注集。
            budget_size: 需要选择的样本数。
            device: 计算设备。

        返回:
            List[int]: 在 `relevant_indices` 局部坐标中的选中位置。
        """
        kernel_all = kernel_all.to(device)
        uncertainties = uncertainties.to(device)

        total_size = kernel_all.size(0)
        labeled_positions = torch.arange(labeled_size, device=device)
        unlabeled_positions = torch.arange(labeled_size, total_size, device=device)
        unlabeled_mask = torch.ones(len(unlabeled_positions), dtype=torch.bool, device=device)

        if labeled_size > 0:
            max_embedding = kernel_all[:labeled_size].max(dim=0, keepdim=True).values
        else:
            max_embedding = torch.zeros((1, total_size), dtype=kernel_all.dtype, device=device)

        selected_positions: List[int] = []
        for _ in range(budget_size):
            updated_max_embedding = kernel_all - max_embedding
            updated_max_embedding = torch.clamp(updated_max_embedding, min=0.0)

            scores = (uncertainties * updated_max_embedding).mean(dim=-1)
            scores[:labeled_size] = -torch.inf

            if len(unlabeled_positions) > 0:
                currently_unavailable = unlabeled_positions[~unlabeled_mask]
                scores[currently_unavailable] = -torch.inf

            selected_position = int(torch.argmax(scores).item())
            selected_positions.append(selected_position)

            if len(unlabeled_positions) > 0:
                local_unlabeled_idx = selected_position - labeled_size
                unlabeled_mask[local_unlabeled_idx] = False

            max_embedding = max_embedding + updated_max_embedding[selected_position].unsqueeze(0)

        return selected_positions

    def _estimate_temperature_if_needed(
        self,
        dataset: Any,
        labeled_indices: List[int],
        current_model: torch.nn.Module,
        device: torch.device,
        initial_model_state: Optional[Dict[str, torch.Tensor]],
    ) -> float:
        """
        在需要时估计 temperature scaling 参数。

        该函数遵循如下策略：
        1. 若未启用 `temp_scale`，直接返回 1.0；
        2. 若当前标注样本数太少，无法切出稳定的训练/验证子集，也直接返回 1.0；
        3. 否则按原始 UHerding 的思路，从 `labeled_indices` 中切出最后 30% 作为临时验证集；
        4. 使用与主模型相同结构的临时模型，在前 70% 样本上训练若干轮；
        5. 在临时验证集上搜索使 ECE 最小的温度值。

        参数:
            dataset: 完整训练集对象。
            labeled_indices: 当前已标注样本索引。
            current_model: 当前轮训练好的主模型。
                           当前实现中仅用于接口对齐与语义说明，不直接参与温度估计。
            device: 计算设备。
            initial_model_state: 当前轮主模型初始化时的参数快照；
                                 若提供，临时模型会从该状态启动。

        返回:
            float: 搜索得到的温度值；若不满足温度估计条件，则返回 1.0。
        """
        del current_model

        if not self.temp_scale:
            return 1.0
        if self.dataset_name is None:
            raise ValueError("UHerding temp_scale 启用时需要提供 dataset_name。")
        if len(labeled_indices) < 4:
            return 1.0

        num_val = max(1, int(len(labeled_indices) * 0.3))
        if num_val >= len(labeled_indices):
            return 1.0

        temp_train_indices = labeled_indices[:-num_val]
        temp_val_indices = labeled_indices[-num_val:]
        if len(temp_train_indices) == 0 or len(temp_val_indices) == 0:
            return 1.0

        temp_model = get_model(self.dataset_name, **self.model_kwargs).to(device)
        if initial_model_state is not None:
            temp_model.load_state_dict(initial_model_state)

        self._train_temporary_model(
            model=temp_model,
            dataset=dataset,
            train_indices=temp_train_indices,
            device=device,
        )
        return self._search_best_temperature(
            model=temp_model,
            dataset=dataset,
            validation_indices=temp_val_indices,
            device=device,
        )

    def _train_temporary_model(
        self,
        model: torch.nn.Module,
        dataset: Any,
        train_indices: List[int],
        device: torch.device,
    ) -> None:
        """
        训练用于 temperature scaling 的临时模型。

        这是一个极简训练过程，只承担“为温度估计提供 logits”的职责，不涉及：
        - best checkpoint 保存；
        - 独立 valSet；
        - 训练日志持久化。

        参数:
            model: 临时模型实例。
            dataset: 完整训练集对象。
            train_indices: 用于临时训练的已标注子集索引。
            device: 计算设备。
        """
        train_loader = DataLoader(
            Subset(dataset, train_indices),
            batch_size=self.temp_batch_size,
            shuffle=True,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=self.temp_learning_rate)
        criterion = torch.nn.CrossEntropyLoss()

        model.train()
        for _ in range(self.temp_train_epochs):
            for batch_inputs, batch_labels in train_loader:
                batch_inputs = batch_inputs.to(device)
                batch_labels = batch_labels.to(device)

                optimizer.zero_grad()
                logits = model(batch_inputs)
                loss = criterion(logits, batch_labels)
                loss.backward()
                optimizer.step()

    def _search_best_temperature(
        self,
        model: torch.nn.Module,
        dataset: Any,
        validation_indices: List[int],
        device: torch.device,
    ) -> float:
        """
        在临时验证集上搜索最优温度参数。

        搜索目标与原始实现保持一致：选择能够最小化 ECE 的温度值。
        过程如下：
        1. 用临时模型在验证集上一次性收集 logits 与 labels；
        2. 对每个候选温度，将 logits 除以 temperature；
        3. 计算对应的 Expected Calibration Error；
        4. 返回 ECE 最小的温度。

        参数:
            model: 已训练完成的临时模型。
            dataset: 完整训练集对象。
            validation_indices: 临时验证集索引。
            device: 计算设备。

        返回:
            float: 最优温度值。
        """
        val_loader = DataLoader(
            Subset(dataset, validation_indices),
            batch_size=self.temp_batch_size,
            shuffle=False,
        )

        logits_list: List[torch.Tensor] = []
        labels_list: List[torch.Tensor] = []
        model.eval()
        with torch.no_grad():
            for batch_inputs, batch_labels in val_loader:
                batch_inputs = batch_inputs.to(device)
                logits = model(batch_inputs)
                logits_list.append(logits.cpu())
                labels_list.append(batch_labels.cpu())

        if not logits_list:
            return 1.0

        logits = torch.cat(logits_list, dim=0)
        labels = torch.cat(labels_list, dim=0)

        best_temp = 1.0
        best_ece = float("inf")
        for temp in self.temp_values:
            scaled_logits = logits / float(temp)
            ece = self._compute_ece(
                logits=scaled_logits,
                labels=labels,
                num_bins=self.ece_num_bins,
            )
            if ece < best_ece:
                best_ece = ece
                best_temp = float(temp)
        return best_temp

    def _compute_ece(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        num_bins: int,
    ) -> float:
        """
        计算 Expected Calibration Error（ECE）。

        实现采用标准的 top-label ECE：
        1. 对 logits 做 softmax，取得每个样本的最大预测置信度与预测类别；
        2. 将置信度划分到若干区间；
        3. 在每个区间内计算平均置信度与实际准确率之差；
        4. 使用区间样本占比对这些差值加权求和。

        参数:
            logits: 未归一化分类输出，形状 `(N, C)`。
            labels: 真实标签，形状 `(N,)`。
            num_bins: 置信度分箱数。

        返回:
            float: ECE 数值，范围通常在 `[0, 1]`。
        """
        probabilities = torch.softmax(logits, dim=1)
        confidences, predictions = torch.max(probabilities, dim=1)
        accuracies = predictions.eq(labels)

        bin_boundaries = torch.linspace(0.0, 1.0, num_bins + 1)
        ece = torch.zeros(1, dtype=torch.float32)
        for bin_idx in range(num_bins):
            lower = bin_boundaries[bin_idx]
            upper = bin_boundaries[bin_idx + 1]

            if bin_idx == 0:
                in_bin = (confidences >= lower) & (confidences <= upper)
            else:
                in_bin = (confidences > lower) & (confidences <= upper)

            if not torch.any(in_bin):
                continue

            prop = in_bin.float().mean()
            accuracy_in_bin = accuracies[in_bin].float().mean()
            confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(confidence_in_bin - accuracy_in_bin) * prop

        return float(ece.item())

    def _compute_candidate_size(
        self,
        labeled_size: int,
        budget_size: int,
        max_size: int,
    ) -> int:
        """
        复现原始 UHerding/TypiClust 代码中的候选子集大小估计公式。

        该公式的目标是在低预算时保留较大的候选池，而随着已标注集增大，
        逐步压缩需要精确计算核矩阵的候选规模。当前项目中还会再套一层
        `max_size` 上界，以控制内存消耗。

        参数:
            labeled_size: 当前已标注样本数。
            budget_size: 本轮查询预算。
            max_size: 候选子集最大上界。

        返回:
            int: 估计得到的候选子集大小。
        """
        upper_bound = 45000 * (35000 + 10000)
        candidate_size = int(
            np.sqrt(upper_bound + (labeled_size + budget_size) ** 2 / 4.0)
            - 1.5 * (labeled_size + budget_size)
        )
        return max(min(max_size, candidate_size), budget_size)
