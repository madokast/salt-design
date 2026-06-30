"""
Radius/coverage active learning strategy.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from typing import List, Any, Dict, Optional, Tuple

from tqdm import tqdm

from .base import ActiveLearningStrategy
from distance_calculator import DistanceCalculator
from get_radius import get_radius, get_spectralnorm


class MyStrategy(ActiveLearningStrategy):
    """
    Custom active learning strategy with radius, weight, and coverage logic.
    """

    def __init__(
        self,
        epsilon: float = 0.01,
        h_min: float = 1e-6,
        spectral_norm_product: Optional[float] = 10.0,
        coverage_threshold: float = 0.2,
        continue_training: bool = True,
        weighted_epochs: Optional[int] = None,
        weighted_learning_rate: Optional[float] = None,
        distance_cache_dir: str = "./distance_cache",
        log_eps: float = 1e-12,
        use_soft_labels: bool = False,
        normalize_radius: bool = False,
        radius_top_percent: float = 0.1,
        diff_min: float = 0.0
    ):
        """
        Initialize the strategy and cache containers.

        Args:
            epsilon: Radius equation epsilon parameter.
            h_min: Lower bound for Hessian spectral norm.
            spectral_norm_product:
                预先给定的谱范数乘积（默认 10.0）。
                若传入 None，则在每轮选择样本时基于当前模型自动调用
                get_spectralnorm(model) 计算。
            coverage_threshold: Threshold T used in the score definition.
            continue_training: Unused (kept for compatibility).
            weighted_epochs: Unused (kept for compatibility).
            weighted_learning_rate: Unused (kept for compatibility).
            distance_cache_dir: Directory containing the cached distance matrices.
            log_eps: Small constant to avoid log(0).
            use_soft_labels: Whether to use precomputed soft labels for radius computation.
            normalize_radius: Whether to normalize radii so the maximum radius matches r_max.
            radius_top_percent: Top percentage (0-1) used as the cutoff for normalization.
            diff_min: Lower bound for diff (label vector - predicted probability vector).
                      默认值为 0.0，用于防止 diff 过小导致半径计算不稳定。
        """
        super().__init__("MyStrategy")
        self.epsilon = epsilon
        self.h_min = h_min
        self.spectral_norm_product: Optional[float]
        if spectral_norm_product is None:
            self.spectral_norm_product = None
        else:
            self.spectral_norm_product = float(spectral_norm_product)
        self.coverage_threshold = coverage_threshold
        # 保留参数以兼容外部调用，但当前策略不使用这些设置。
        _ = continue_training
        _ = weighted_epochs
        _ = weighted_learning_rate
        self.distance_cache_dir = distance_cache_dir
        self.log_eps = log_eps
        self.use_soft_labels = use_soft_labels
        self.normalize_radius = normalize_radius
        self.radius_top_percent = radius_top_percent
        self.diff_min = diff_min
        self.soft_labels: Optional[np.ndarray] = None

        # Dataset metadata (used to select the correct distance-matrix cache).
        self.dataset_name: Optional[str] = None
        self.distance_matrix_suffix: str = ""

        # Cached distance matrix for reuse across iterations.
        self.distance_matrix: Optional[np.ndarray] = None

        self.last_coverage: Optional[float] = None
        
        # 保存归一化参数供后续使用
        self.last_r_cut: Optional[float] = None
        self.last_r_max: Optional[float] = None

    def set_dataset_metadata(
        self,
        dataset_name: str,
    ) -> None:
        """
        Store dataset identity for later cache lookup.

        The current project uses a single canonical embedding cache per text
        dataset, so distance-matrix cache lookup no longer depends on any
        text-encoder-specific suffix.

        Args:
            dataset_name: Name of the dataset used by the active learning loop.
        """
        self.dataset_name = dataset_name
        self.distance_matrix_suffix = ""

    def set_soft_labels(self, soft_labels: np.ndarray) -> None:
        """
        设置全量训练集的软标签矩阵。

        该方法用于在策略中启用软标签半径计算。传入的 soft_labels
        应与完整训练集索引对齐（shape: N x num_classes）。

        Args:
            soft_labels: 全量训练集的软标签矩阵。
        """
        if soft_labels is None:
            raise ValueError("soft_labels 不能为空")
        if isinstance(soft_labels, torch.Tensor):
            soft_labels = soft_labels.detach().cpu().numpy()
        self.soft_labels = np.asarray(soft_labels)
        self.use_soft_labels = True

    def _get_soft_label_for_index(self, index: int) -> np.ndarray:
        """
        根据全局样本索引获取对应的软标签向量。

        Args:
            index: 全量训练集中的样本索引。

        Returns:
            np.ndarray: 与该样本对应的软标签概率向量。
        """
        if not self.use_soft_labels or self.soft_labels is None:
            raise RuntimeError("软标签未设置，但尝试获取软标签。")
        return self.soft_labels[index]

    def _resolve_spectral_norm_product(self, model: torch.nn.Module) -> float:
        """
        解析当前轮次实际使用的谱范数乘积，并执行有效性校验。

        该函数统一处理 `spectral_norm_product` 的两种来源，避免半径相关逻辑
        在多个位置重复判断：
        1) 若初始化时显式传入了数值，则直接复用该配置值；
        2) 若初始化时传入 None，则基于当前模型调用 `get_spectralnorm(model)`
           动态计算谱范数乘积。

        这样可以保证同一轮采样中“半径计算、半径归一化、候选点剔除”三处逻辑
        使用的是同一个谱范数乘积，避免不同阶段取值不一致。

        参数:
            model: 当前轮次用于训练/采样的模型；当配置值为 None 时用于动态计算。

        返回:
            float: 当前轮次可直接用于半径公式的谱范数乘积。

        异常:
            ValueError: 当解析出的谱范数乘积不是有限正数（<=0 或 NaN/Inf）时抛出。
        """
        if self.spectral_norm_product is None:
            spectral_norm_product = float(get_spectralnorm(model))
        else:
            spectral_norm_product = float(self.spectral_norm_product)

        if not np.isfinite(spectral_norm_product) or spectral_norm_product <= 0.0:
            raise ValueError(
                f"spectral_norm_product 必须是有限正数，当前值: {spectral_norm_product}"
            )
        return spectral_norm_product

    def _compute_raw_radii(
        self,
        labeled_points: List[Dict[str, Any]],
        model: torch.nn.Module
    ) -> Tuple[List[float], float]:
        """
        计算一组已标注点的原始半径（未归一化）。

        该函数会先解析本轮应使用的谱范数乘积：
        - 若配置值为数值，则直接使用该值；
        - 若配置值为 None，则对当前模型动态计算谱范数乘积。

        解析后的谱范数乘积会直接用于每个点的半径计算，
        并返回给后续归一化/采样逻辑复用。

        参数:
            labeled_points: 已标注点列表（元素包含 data/label，软标签可选）。
            model: 当前模型。

        返回:
            Tuple[List[float], float]:
                - raw_radii: 按 labeled_points 顺序计算得到的原始半径列表。
                - spectral_norm_product: 当前轮实际使用的谱范数乘积。
        """
        spectral_norm_product = self._resolve_spectral_norm_product(model)
        raw_radii = [
            get_radius(
                point,
                model,
                epsilon=self.epsilon,
                h_min=self.h_min,
                use_soft_label=self.use_soft_labels,
                spectral_norm_product=spectral_norm_product,
                diff_min=self.diff_min
            )
            for point in labeled_points
        ]
        return raw_radii, spectral_norm_product

    def _normalize_radii(
        self,
        raw_radii: List[float],
        spectral_norm_product: float
    ) -> List[float]:
        """
        按当前阶段的最大半径对半径列表进行归一化缩放。

        归一化规则：
            1) 取从大到小排序后前 radius_top_percent 的阈值 r_cut
               （即第 ceil(radius_top_percent * N) 大的半径）。
            2) r_max = sqrt(2 * epsilon / h_min) / spectral_norm_product
            3) 若 r >= r_cut，则 r_norm = r_max（钳制到上限）
               若 r < r_cut，则 r_norm = (r / r_cut) * r_max

        这样可以把"前 radius_top_percent 大半径"压到 r_max，其余半径按比例放缩。
        当 normalize_radius 关闭、raw_radii 为空或 r_cut 非正时，直接返回原始半径列表。

        参数:
            raw_radii: 原始半径列表。
            spectral_norm_product: 当前轮实际使用的谱范数乘积。

        返回:
            List[float]: 归一化后的半径列表（或原始半径列表）。
        """
        if not np.isfinite(spectral_norm_product) or spectral_norm_product <= 0.0:
            raise ValueError(
                f"spectral_norm_product 必须是有限正数，当前值: {spectral_norm_product}"
            )
        # 初始化 r_cut 和 r_max 为 None
        self.last_r_cut = None
        self.last_r_max = None
        
        if not self.normalize_radius:
            return raw_radii
        if not raw_radii:
            return raw_radii
        radii_array = np.asarray(raw_radii, dtype=np.float64)
        if radii_array.size == 0:
            return raw_radii
        percent = float(self.radius_top_percent)
        if percent <= 0.0:
            return raw_radii
        if percent > 1.0:
            percent = percent / 100.0
            if percent <= 0.0 or percent > 1.0:
                return raw_radii
        k = int(np.ceil(percent * radii_array.size))
        k = max(1, k)
        sorted_desc = np.sort(radii_array)[::-1]
        r_cut = float(sorted_desc[k - 1])
        if r_cut <= 0.0:
            return raw_radii
        # 使用新公式计算最大理论半径
        r_max = float(
            (-self.diff_min + np.sqrt(self.diff_min**2 + 2.0 * self.h_min * self.epsilon))
            / (self.h_min * spectral_norm_product)
        )
        
        # 保存 r_cut 和 r_max 供后续使用
        self.last_r_cut = r_cut
        self.last_r_max = r_max
        
        normalized = []
        for radius in radii_array:
            if radius >= r_cut:
                normalized.append(r_max)
            else:
                normalized.append((radius / r_cut) * r_max)
        return normalized

    def select_samples(
        self,
        model: torch.nn.Module,
        unlabeled_indices: List[int],
        dataset: Any,
        batch_size: int,
        device: torch.device,
        labeled_indices: Optional[List[int]] = None,
        test_loader: Optional[DataLoader] = None,
        weighted_epochs: Optional[int] = None,
        weighted_learning_rate: Optional[float] = None,
        **kwargs: Any
    ) -> List[int]:
        """
        Select samples based on radius, coverage, and uncertainty scoring.

        Args:
            model: Trained model for the current iteration.
            unlabeled_indices: Indices of currently unlabeled samples.
            dataset: Full training dataset.
            batch_size: Number of samples to select.
            device: Torch device to run model inference/training.
            labeled_indices: Optional labeled index list from the engine.
            test_loader: Unused (kept for compatibility).
            weighted_epochs: Unused (kept for compatibility).
            weighted_learning_rate: Unused (kept for compatibility).
            **kwargs: Unused extra arguments for compatibility.

        Returns:
            List[int]: Selected indices for labeling.
        """
        _ = test_loader
        _ = weighted_epochs
        _ = weighted_learning_rate

        if labeled_indices is None:
            all_indices = set(range(len(dataset)))
            labeled_indices = sorted(all_indices - set(unlabeled_indices))

        # Cache the distance matrix once per strategy instance.
        distance_matrix = self._ensure_distance_matrix(split="train")

        # Step 2: compute radii for labeled points with the current model.
        labeled_points = self._build_labeled_points(dataset, labeled_indices)
        raw_radii, spectral_norm_product = self._compute_raw_radii(
            labeled_points,
            model
        )
        radii = self._normalize_radii(raw_radii, spectral_norm_product)

        # Step 2: compute coverage ratio C and report stats.
        coverage_ratio, covered_mask = self._compute_coverage_ratio(
            distance_matrix,
            labeled_indices,
            radii
        )
        self.last_coverage = coverage_ratio
        self._report_iteration_stats(labeled_indices, radii, coverage_ratio)

        # Step 4: filter unlabeled pool based on labeled coverage.
        filtered_unlabeled = [idx for idx in unlabeled_indices if not covered_mask[idx]]

        # Step 5: select samples using the scoring rule on the filtered pool.
        selected_indices = self._select_with_scoring(
            model=model,
            dataset=dataset,
            device=device,
            filtered_indices=filtered_unlabeled,
            distance_matrix=distance_matrix,
            coverage_ratio=coverage_ratio,
            batch_size=batch_size,
            spectral_norm_product=spectral_norm_product
        )

        return selected_indices

    def _ensure_distance_matrix(self, split: str = "train") -> np.ndarray:
        """
        Load and cache the distance matrix for the dataset.

        Args:
            split: Dataset split name (default: "train").

        Returns:
            np.ndarray: Loaded distance matrix.

        Raises:
            RuntimeError: If no distance matrix file is found.
        """
        if self.distance_matrix is not None:
            return self.distance_matrix

        if self.dataset_name is None:
            raise RuntimeError("Dataset metadata must be set before loading distances.")

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
                "Distance matrix not found. Please compute and cache it in distance_cache."
            )

        self.distance_matrix = distance_matrix
        return distance_matrix

    def _compute_coverage_ratio(
        self,
        distance_matrix: np.ndarray,
        labeled_indices: List[int],
        radii: List[float]
    ) -> Tuple[float, np.ndarray]:
        """
        Compute the union coverage ratio over the full training set.

        Coverage is defined as the fraction of all training points that lie
        within at least one labeled point's radius (including the center point).

        Args:
            distance_matrix: Precomputed distance matrix of shape (N, N).
            labeled_indices: Indices of labeled samples.
            radii: Radii aligned with labeled_indices.

        Returns:
            Tuple of (coverage_ratio, covered_mask).
        """
        n_total = distance_matrix.shape[0]
        covered_mask = np.zeros(n_total, dtype=bool)

        for idx, radius in zip(labeled_indices, radii):
            # Include points at the boundary with <= and always include self (distance 0).
            distances = distance_matrix[idx]
            covered_mask |= distances <= radius

        coverage_ratio = float(covered_mask.sum()) / float(n_total) if n_total > 0 else 0.0
        return coverage_ratio, covered_mask

    def _build_labeled_points(self, dataset: Any, labeled_indices: List[int]) -> List[Dict[str, Any]]:
        """
        Build labeled point dictionaries required by get_radius.

        Args:
            dataset: Full training dataset.
            labeled_indices: Indices of labeled samples.

        Returns:
            List of dicts with keys: 'data' and 'label'，若启用软标签则包含 'soft_label'.
        """
        labeled_points = []
        for idx in labeled_indices:
            data, label = dataset[idx]
            if isinstance(label, torch.Tensor):
                label_value = int(label.item())
            else:
                label_value = int(label)
            if self.use_soft_labels:
                soft_label = self._get_soft_label_for_index(idx)
                labeled_points.append(
                    {"data": data, "label": label_value, "soft_label": soft_label}
                )
            else:
                labeled_points.append({"data": data, "label": label_value})
        return labeled_points

    def _report_iteration_stats(
        self,
        labeled_indices: List[int],
        radii: List[float],
        coverage_ratio: float,
        stage_label: str = "Current"
    ) -> None:
        """
        Print the max radius and coverage stats for the iteration.

        Args:
            labeled_indices: Indices aligned with radii.
            radii: Radii computed for current labeled set.
            coverage_ratio: Coverage ratio C over the full training set.
            stage_label: Label describing which stage is being reported.
        """
        if not radii:
            print(f"[MyStrategy] ({stage_label}) No radii available for reporting.")
            return

        max_radius_pos = int(np.argmax(radii))

        print(
            "[MyStrategy] ({}) Coverage C={:.6f} | max radius {:.6f} (idx {})".format(
                stage_label,
                coverage_ratio,
                float(radii[max_radius_pos]),
                labeled_indices[max_radius_pos],
            )
        )

    def _predict_probabilities(
        self,
        model: torch.nn.Module,
        dataset: Any,
        indices: List[int],
        device: torch.device,
        batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict class probabilities for a subset of indices.

        Args:
            model: Model used for prediction.
            dataset: Full training dataset.
            indices: Indices to predict.
            device: Torch device for computation.
            batch_size: Batch size for inference.

        Returns:
            Tuple of (probabilities, predicted_labels).
        """
        if not indices:
            return np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=np.int64)

        model.eval()
        subset = Subset(dataset, indices)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=False)

        probs_list = []
        with torch.no_grad():
            for batch in loader:
                # batch can be (data, label) or (data, label, weight); handle both.
                if len(batch) == 3:
                    inputs, _, _ = batch
                else:
                    inputs, _ = batch
                inputs = inputs.to(device)
                logits = model(inputs)
                probs = torch.softmax(logits, dim=1)
                probs_list.append(probs.cpu().numpy())

        probs_array = np.concatenate(probs_list, axis=0)
        pred_labels = np.argmax(probs_array, axis=1)
        return probs_array, pred_labels

    def _select_with_scoring(
        self,
        model: torch.nn.Module,
        dataset: Any,
        device: torch.device,
        filtered_indices: List[int],
        distance_matrix: np.ndarray,
        coverage_ratio: float,
        batch_size: int,
        spectral_norm_product: float
    ) -> List[int]:
        """
        Select a batch of points using the score definition and greedy pruning.

        Args:
            model: Weighted model used for prediction.
            dataset: Full training dataset.
            device: Torch device for computation.
            filtered_indices: Unlabeled indices after coverage filtering.
            distance_matrix: Precomputed distances for coverage calculations.
            coverage_ratio: Coverage ratio C used in the score.
            batch_size: Selection budget for this iteration.
            spectral_norm_product: 当前轮实际使用的谱范数乘积。

        Returns:
            List[int]: Selected indices in order of selection.
        """
        if not filtered_indices:
            return []

        if len(filtered_indices) <= batch_size:
            return list(filtered_indices)

        probs, pred_labels = self._predict_probabilities(
            model=model,
            dataset=dataset,
            indices=filtered_indices,
            device=device,
            batch_size=batch_size
        )

        if probs.size == 0:
            return []

        num_classes = probs.shape[1]
        eta_matrix = np.eye(num_classes, dtype=np.float32)[pred_labels]

        # 使用新公式计算最大理论半径
        max_theoretical_radius = float(
            (-self.diff_min + np.sqrt(self.diff_min**2 + 2.0 * self.h_min * self.epsilon))
            / (self.h_min * spectral_norm_product)
        )

        log_diff_by_index: Dict[int, float] = {}
        for idx_pos, data_index in enumerate(filtered_indices):
            diff_norm = np.linalg.norm(eta_matrix[idx_pos] - probs[idx_pos], ord=2)
            diff_norm = max(diff_norm, self.log_eps)
            log_diff_by_index[data_index] = float(np.log(diff_norm))

        beta = 1.0 if coverage_ratio < self.coverage_threshold else 0.0
        coef = coverage_ratio - self.coverage_threshold

        selected: List[int] = []
        remaining = list(filtered_indices)
        target_size = min(batch_size, len(filtered_indices))

        # 判断是否为高覆盖率阶段
        is_high_coverage = coverage_ratio > self.coverage_threshold

        with tqdm(total=target_size, desc="MyStrategy sampling", leave=False) as pbar:
            while remaining and len(selected) < batch_size:
                remaining_array = np.array(remaining, dtype=np.int64)
                best_idx = None
                best_score = -np.inf

                for candidate_idx in remaining:
                    distances = distance_matrix[candidate_idx, remaining_array]
                    
                    # 根据覆盖率阶段选择计算 D 的半径
                    if is_high_coverage:
                        # 高覆盖率时，使用基于 Hessian 的预估半径计算 D
                        # 找到 candidate_idx 在 filtered_indices 中的位置
                        candidate_pos = None
                        for pos, idx in enumerate(filtered_indices):
                            if idx == candidate_idx:
                                candidate_pos = pos
                                break
                        
                        if candidate_pos is None:
                            # 如果找不到，使用最大理论半径
                            D = int(np.sum(distances <= max_theoretical_radius))
                        else:
                            # 计算基于 Hessian 的预估半径
                            p = probs[candidate_pos]
                            H = np.diag(p) - np.outer(p, p)
                            h_raw = np.linalg.norm(H, ord=2)
                            h = max(h_raw, self.h_min)
                            # 使用新公式计算预估半径
                            r_estimated = float(
                                2.0 * self.epsilon / (self.diff_min + np.sqrt(self.diff_min**2 + 2.0 * h * self.epsilon))
                            )
                            D = int(np.sum(distances <= r_estimated))
                    else:
                        # 低覆盖率时，使用最大理论半径计算 D
                        D = int(np.sum(distances <= max_theoretical_radius))
                    
                    D = max(D, 1)

                    score = coef * log_diff_by_index[candidate_idx] + beta * float(np.log(D))
                    if score > best_score:
                        best_score = score
                        best_idx = candidate_idx

                if best_idx is None:
                    break

                selected.append(best_idx)
                pbar.update(1)

                # 根据覆盖率与阈值关系决定使用哪种半径进行覆盖剔除
                if coverage_ratio <= self.coverage_threshold:
                    # 当 C <= T 时，使用最大理论半径进行剔除
                    distances = distance_matrix[best_idx, remaining_array]
                    keep_mask = distances > max_theoretical_radius
                    remaining = [idx for idx, keep in zip(remaining, keep_mask) if keep]
                else:
                    # 当 C > T 时，使用基于 Hessian 的预估半径进行剔除
                    # 步骤 1: 获取被选中点的概率向量
                    # 找到 best_idx 在 filtered_indices 中的位置
                    best_idx_pos = None
                    for pos, idx in enumerate(filtered_indices):
                        if idx == best_idx:
                            best_idx_pos = pos
                            break
                    
                    if best_idx_pos is None:
                        # 如果找不到，仅移除 best_idx
                        remaining = [idx for idx in remaining if idx != best_idx]
                    else:
                        # 获取概率向量 p（列向量）
                        p = probs[best_idx_pos]  # 形状: (num_classes,)
                        
                        # 步骤 2: 计算 Hessian = diag(p) - pp^T
                        H = np.diag(p) - np.outer(p, p)
                        
                        # 步骤 3: 计算 Hessian 的谱范数并在 h_min 处截断
                        h_raw = np.linalg.norm(H, ord=2)
                        h = max(h_raw, self.h_min)
                        
                        # 步骤 4: 使用新公式计算基础预估半径
                        r = float(
                            2.0 * self.epsilon / (self.diff_min + np.sqrt(self.diff_min**2 + 2.0 * h * self.epsilon))
                        )
                        
                        # 步骤 5: 根据 normalize_radius 设置调整半径
                        if not self.normalize_radius:
                            r_final = r
                        else:
                            # 需要使用归一化参数
                            if self.last_r_cut is None or self.last_r_max is None:
                                # 如果归一化参数不可用，使用基础半径
                                r_final = r
                            else:
                                scale = self.last_r_max / self.last_r_cut
                                r_scaled = r * scale
                                # 钳制到 r_max
                                r_final = min(r_scaled, self.last_r_max)
                        
                        # 步骤 6: 剔除半径内的候选点
                        distances = distance_matrix[best_idx, remaining_array]
                        keep_mask = distances > r_final
                        remaining = [idx for idx, keep in zip(remaining, keep_mask) if keep]

        return selected
