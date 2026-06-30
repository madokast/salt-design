"""
TypiClust 策略。

该实现是对论文 TypiClust 的项目内适配版本，遵循当前仓库的统一
`ActiveLearningStrategy` 接口，不依赖 `TypiClust-main` 中的任何代码。

与原始仓库相比，本实现做了两点明确约束：
1. 不再从磁盘加载外部预训练特征，而是直接使用当前 `dataset` 中的输入张量
   作为聚类与典型性计算所基于的表征；
2. 对于 MNIST，直接在原始像素空间中进行展平和归一化；对于其余已嵌入数据，
   直接复用样本自身的嵌入向量。
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

import numpy as np
import torch
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, Subset

from .base import ActiveLearningStrategy


class TypiClustStrategy(ActiveLearningStrategy):
    """
    TypiClust 主动学习策略。

    该策略同时追求两件事：
    1. 多样性：优先从当前已标注样本较少覆盖到的簇中选点；
    2. 代表性：在每个簇内部选择局部密度最高、也就是最“典型”的样本。

    具体做法为：
    1. 将 `labeled_indices + unlabeled_indices` 对应样本映射到统一表征空间；
    2. 以 `min(|L| + b, max_num_clusters)` 作为簇数进行聚类；
    3. 统计每个簇中已有多少已标注样本，并按“已标注数升序、簇大小降序”排序；
    4. 按排序后的簇顺序轮转，从每个簇中选出典型性最大的未标注样本；
    5. 直到选满 `batch_size` 个样本。
    """

    def __init__(
        self,
        min_cluster_size: int = 5,
        max_num_clusters: int = 500,
        num_neighbors: int = 20,
        loader_batch_size: int = 256,
        normalize_vectors: bool = True,
        random_state: int = 0,
    ):
        """
        初始化 TypiClust 策略。

        参数:
            min_cluster_size: 参与簇优先级排序时允许的最小簇大小。
                原论文实现会忽略过小簇，本参数用于复现该行为。
            max_num_clusters: 聚类数的全局上界，用于避免在已标注规模逐步变大时，
                聚类数无限增长带来的计算成本。
            num_neighbors: 计算典型性时使用的近邻数上界。
                簇内部真实近邻数还会根据当前簇可用样本数进一步裁剪。
            loader_batch_size: 从数据集中提取输入向量时的数据加载批大小。
            normalize_vectors: 是否在聚类和近邻搜索前，对每个样本向量做 L2 归一化。
                原始 TypiClust 依赖的预训练特征通常已标准化；该开关用于保持这一性质。
            random_state: 聚类算法的随机种子，保证实验可复现。
        """
        super().__init__("TypiClust")
        self.min_cluster_size = int(min_cluster_size)
        self.max_num_clusters = int(max_num_clusters)
        self.num_neighbors = int(num_neighbors)
        self.loader_batch_size = int(loader_batch_size)
        self.normalize_vectors = bool(normalize_vectors)
        self.random_state = int(random_state)

    def select_samples(
        self,
        model: torch.nn.Module,
        unlabeled_indices: List[int],
        dataset: Any,
        batch_size: int,
        device: torch.device,
        labeled_indices: Optional[List[int]] = None,
    ) -> List[int]:
        """
        根据 TypiClust 规则，从未标注池中选择一批最值得标注的样本。

        参数:
            model: 当前轮已经训练完成的模型。
                TypiClust 的当前实现不直接依赖模型前向结果，但保留该参数，
                以满足仓库中统一的主动学习策略接口。
            unlabeled_indices: 当前未标注样本索引列表。
            dataset: 完整训练集对象。样本的输入部分将被直接视为表征向量来源。
            batch_size: 本轮需要选择的样本数量。
            device: 当前计算设备。
                本策略的主体计算基于 CPU 上的 sklearn 实现，该参数保留是为了
                与统一接口兼容，并在未来如需扩展 GPU 版本时无需改动外层调用。
            labeled_indices: 当前已标注样本索引列表。
                TypiClust 需要该信息来统计“每个簇中已有多少已标注样本”，
                这是其“优先补足覆盖不足簇”的关键步骤。

        返回:
            List[int]: 选中的原始数据集索引列表，长度不超过 `batch_size`。
        """
        del model, device

        if len(unlabeled_indices) <= batch_size:
            return list(unlabeled_indices)

        labeled_indices = list(labeled_indices or [])
        if batch_size <= 0:
            return []

        relevant_indices = labeled_indices + list(unlabeled_indices)
        relevant_vectors = self._collect_feature_vectors(
            dataset=dataset,
            indices=relevant_indices,
        )
        cluster_labels = self._cluster_relevant_vectors(
            vectors=relevant_vectors,
            labeled_size=len(labeled_indices),
            budget_size=batch_size,
        )
        ordered_cluster_ids = self._rank_clusters_for_selection(
            cluster_labels=cluster_labels,
            labeled_size=len(labeled_indices),
        )
        selected_relative_indices = self._select_representative_samples(
            vectors=relevant_vectors,
            cluster_labels=cluster_labels,
            ordered_cluster_ids=ordered_cluster_ids,
            labeled_size=len(labeled_indices),
            budget_size=batch_size,
        )
        return [int(relevant_indices[idx]) for idx in selected_relative_indices]

    def _collect_feature_vectors(
        self,
        dataset: Any,
        indices: Sequence[int],
    ) -> np.ndarray:
        """
        从数据集中读取指定样本，并整理成二维浮点特征矩阵。

        该函数承担 TypiClust 在当前项目中的“表征准备”职责，但不做额外的
        模型编码步骤，而是直接使用数据集样本输入本身：
        1. 若输入本来就是嵌入向量，则仅做展平；
        2. 若输入是图像张量，则展平成一维像素向量；
        3. 最终可选地执行逐样本 L2 归一化，以贴近原方法对表征空间的使用方式。

        参数:
            dataset: 完整训练集对象。
            indices: 需要提取的样本原始索引序列。

        返回:
            np.ndarray: 形状为 `(len(indices), feature_dim)` 的二维特征矩阵。
        """
        subset = Subset(dataset, list(indices))
        loader = DataLoader(subset, batch_size=self.loader_batch_size, shuffle=False)

        feature_chunks: List[torch.Tensor] = []
        for batch in loader:
            batch_inputs = self._extract_inputs_from_batch(batch)
            batch_inputs = batch_inputs.view(batch_inputs.size(0), -1).detach().cpu()
            feature_chunks.append(batch_inputs)

        if not feature_chunks:
            return np.empty((0, 0), dtype=np.float32)

        vectors = torch.cat(feature_chunks, dim=0).numpy().astype(np.float32, copy=False)
        if self.normalize_vectors:
            vectors = self._l2_normalize_rows(vectors)
        return vectors

    def _extract_inputs_from_batch(self, batch: Any) -> torch.Tensor:
        """
        从 DataLoader 返回的批数据中提取“输入张量”部分。

        之所以单独写这个函数，是因为当前仓库中的数据集封装形式并不完全统一：
        - 常见情况是 `(inputs, labels)`；
        - 某些包装数据集可能返回长度大于 2 的元组；
        - 少数场景也可能直接返回输入张量。

        TypiClust 只关心输入表征，因此这里统一取批对象的第一个元素作为输入。

        参数:
            batch: DataLoader 产出的单个批次对象。

        返回:
            torch.Tensor: 当前批次的输入张量。

        异常:
            TypeError: 当批对象既不是张量，也不是元组/列表时抛出。
        """
        if isinstance(batch, torch.Tensor):
            return batch
        if isinstance(batch, (list, tuple)) and len(batch) > 0:
            return batch[0]
        raise TypeError(
            "TypiClust 期望 DataLoader 返回张量或以输入张量为第一个元素的元组/列表。"
        )

    def _l2_normalize_rows(self, vectors: np.ndarray) -> np.ndarray:
        """
        对二维特征矩阵逐行做 L2 归一化。

        原始 TypiClust 通常工作在已经标准化的预训练表征空间中。当前项目中，
        不同数据集的输入尺度可能差异较大；若直接在未归一化空间上做 KMeans 和
        局部近邻搜索，某些高范数样本可能会过度主导距离度量。

        因此该函数会将每个样本向量归一化到单位球面上，并对零向量做保护，
        避免除零产生 NaN。

        参数:
            vectors: 原始二维特征矩阵，形状为 `(N, D)`。

        返回:
            np.ndarray: 归一化后的二维特征矩阵，形状与输入一致。
        """
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        safe_norms = np.where(norms > 0.0, norms, 1.0).astype(np.float32, copy=False)
        return vectors / safe_norms

    def _cluster_relevant_vectors(
        self,
        vectors: np.ndarray,
        labeled_size: int,
        budget_size: int,
    ) -> np.ndarray:
        """
        对当前轮“已标注 + 未标注”联合样本执行聚类。

        TypiClust 原始实现使用的簇数为 `min(|L| + b, MAX_NUM_CLUSTERS)`。
        该规则的直觉是：随着已标注集扩大，可以使用更多簇来更细地刻画数据分布；
        但同时又需要上界来控制聚类计算成本。

        参数:
            vectors: 待聚类的二维特征矩阵。
            labeled_size: 当前已标注样本数。
            budget_size: 本轮查询预算。

        返回:
            np.ndarray: 每个样本所属的簇编号，形状为 `(N,)`。
        """
        num_samples = int(vectors.shape[0])
        num_clusters = max(
            1,
            min(labeled_size + budget_size, self.max_num_clusters, num_samples),
        )

        if num_clusters <= 50:
            clusterer = KMeans(
                n_clusters=num_clusters,
                random_state=self.random_state,
                n_init=10,
            )
        else:
            clusterer = MiniBatchKMeans(
                n_clusters=num_clusters,
                batch_size=min(5000, max(num_clusters, num_samples)),
                random_state=self.random_state,
                n_init=10,
            )

        return clusterer.fit_predict(vectors)

    def _rank_clusters_for_selection(
        self,
        cluster_labels: np.ndarray,
        labeled_size: int,
    ) -> List[int]:
        """
        根据 TypiClust 的优先级规则，对簇进行排序。

        排序规则与原始实现保持一致：
        1. 优先选择当前已标注样本数更少的簇；
        2. 若已标注数相同，则优先选择更大的簇。

        同时，函数会先过滤掉过小簇；如果过滤后为空，则退回到“不过滤”版本，
        这样在小规模实验或后期簇被多次抽空的情况下，策略仍然能够稳定工作。

        参数:
            cluster_labels: 联合样本的簇编号数组。
            labeled_size: `cluster_labels` 前缀中已标注样本的数量。

        返回:
            List[int]: 按优先级排好序的簇编号列表。
        """
        unique_cluster_ids, cluster_sizes = np.unique(cluster_labels, return_counts=True)

        labeled_cluster_counts = {}
        for cluster_id in unique_cluster_ids.tolist():
            labeled_cluster_counts[int(cluster_id)] = 0

        for cluster_id in cluster_labels[:labeled_size].tolist():
            labeled_cluster_counts[int(cluster_id)] += 1

        cluster_records = []
        for cluster_id, cluster_size in zip(unique_cluster_ids.tolist(), cluster_sizes.tolist()):
            cluster_records.append(
                {
                    "cluster_id": int(cluster_id),
                    "cluster_size": int(cluster_size),
                    "existing_count": int(labeled_cluster_counts[int(cluster_id)]),
                }
            )

        filtered_records = [
            record
            for record in cluster_records
            if record["cluster_size"] > self.min_cluster_size
        ]
        ranked_records = filtered_records if filtered_records else cluster_records
        ranked_records.sort(
            key=lambda record: (record["existing_count"], -record["cluster_size"])
        )
        return [record["cluster_id"] for record in ranked_records]

    def _select_representative_samples(
        self,
        vectors: np.ndarray,
        cluster_labels: np.ndarray,
        ordered_cluster_ids: Sequence[int],
        labeled_size: int,
        budget_size: int,
    ) -> List[int]:
        """
        按簇轮转地挑选每个簇中最典型的未标注样本。

        函数会先将已标注样本整体排除，仅在未标注区域内做选择；随后按排序后的簇
        列表轮转，每次从当前簇中找到“典型性”最大的样本，并将其标记为已选择，
        避免重复选中。

        参数:
            vectors: 联合样本的二维特征矩阵。
            cluster_labels: 与 `vectors` 对齐的簇编号数组。
            ordered_cluster_ids: 已按优先级排序好的簇编号序列。
            labeled_size: 联合样本前缀中已标注样本的数量。
            budget_size: 本轮查询预算。

        返回:
            List[int]: 相对于 `vectors` / `cluster_labels` 的局部索引列表。
        """
        remaining_unlabeled_mask = np.ones(vectors.shape[0], dtype=bool)
        remaining_unlabeled_mask[:labeled_size] = False

        selected_relative_indices: List[int] = []
        while len(selected_relative_indices) < budget_size:
            progress_made = False

            for cluster_id in ordered_cluster_ids:
                if len(selected_relative_indices) >= budget_size:
                    break

                candidate_indices = np.flatnonzero(
                    remaining_unlabeled_mask & (cluster_labels == cluster_id)
                )
                if candidate_indices.size == 0:
                    continue

                cluster_vectors = vectors[candidate_indices]
                typicality_scores = self._compute_typicality_scores(cluster_vectors)
                best_local_offset = int(np.argmax(typicality_scores))
                best_relative_index = int(candidate_indices[best_local_offset])

                selected_relative_indices.append(best_relative_index)
                remaining_unlabeled_mask[best_relative_index] = False
                progress_made = True

            if not progress_made:
                break

        return selected_relative_indices

    def _compute_typicality_scores(self, cluster_vectors: np.ndarray) -> np.ndarray:
        """
        计算单个簇内部每个样本的典型性分数。

        TypiClust 将“典型性”定义为局部密度的倒数表达：
        - 先求样本到其若干近邻的平均距离；
        - 再使用 `1 / (mean_distance + eps)` 作为典型性；
        - 平均距离越小，说明样本处在越密集的局部区域，其典型性越高。

        参数:
            cluster_vectors: 单个簇内候选样本的二维特征矩阵。

        返回:
            np.ndarray: 当前簇内每个样本的典型性分数，形状为 `(cluster_size,)`。
        """
        cluster_size = int(cluster_vectors.shape[0])
        if cluster_size <= 1:
            return np.ones(cluster_size, dtype=np.float32)

        effective_neighbors = max(1, min(self.num_neighbors, cluster_size // 2))
        neighbor_model = NearestNeighbors(
            n_neighbors=min(cluster_size, effective_neighbors + 1),
            metric="euclidean",
        )
        neighbor_model.fit(cluster_vectors)
        distances, _ = neighbor_model.kneighbors(cluster_vectors)

        if distances.shape[1] > 1:
            mean_distances = distances[:, 1:].mean(axis=1)
        else:
            mean_distances = np.zeros(cluster_size, dtype=np.float32)

        return 1.0 / (mean_distances.astype(np.float32, copy=False) + 1e-5)
