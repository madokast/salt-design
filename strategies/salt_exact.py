"""
SALT exact sparse strategy.

This strategy keeps the sampling and next-round replicated-training semantics of
`OurMethodMarginWeightedSqrtStrategy`, but replaces the dense precomputed
distance matrix with an exact radius graph stored in CSR form.
"""
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

try:
    from scipy import sparse as scipy_sparse
except ImportError:
    scipy_sparse = None

from .our_method_margin_weighted_sqrt import OurMethodMarginWeightedSqrtStrategy
from .local_expansion import (
    compute_local_expansion_factors_from_knn,
    predict_logits_for_indices,
)
from knn_cache import (
    compute_exact_knn_for_queries,
    load_knn_cache,
    save_knn_cache,
)


@dataclass
class SparseRadiusGraph:
    """
    CSR representation of the exact maximum-radius coverage graph.

    The graph stores every directed edge `(x, u)` whose embedding-space distance
    is at most the configured global `r_max`.  Rows are aligned with full dataset
    indices, so row `x` can be read with `indptr[x]:indptr[x + 1]`.

    Attributes:
        n_samples: Number of samples in the full training pool.
        r_max: Global maximum radius used to build the graph.
        indptr: CSR row pointer array with shape `(n_samples + 1,)`.
        indices: CSR column index array with shape `(num_edges,)`.
        distances: CSR edge distance array with shape `(num_edges,)`.
    """

    n_samples: int
    r_max: float
    indptr: np.ndarray
    indices: np.ndarray
    distances: np.ndarray

    def row(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return one CSR row as neighbor indices and distances.

        This helper centralizes CSR slicing so all later coverage code uses the
        same row-boundary logic.  It returns views into the backing arrays rather
        than copies, which is important for large radius graphs.

        Args:
            index: Full dataset index whose maximum-radius neighborhood is read.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Neighbor indices and aligned distances
            for the requested row.
        """
        start = int(self.indptr[index])
        end = int(self.indptr[index + 1])
        return self.indices[start:end], self.distances[start:end]


@dataclass
class CandidateCoverCSR:
    """
    CSR representation of the current round's true candidate coverage graph.

    This structure is built by filtering `SparseRadiusGraph` rows with the current
    round's effective radii and restricting targets to the active candidate pool.
    `cover_*` stores `C(x)` for each candidate row, while `reverse_*` stores the
    inverted index `I(u)` for each candidate target position.

    Attributes:
        cover_indptr: CSR row pointer for candidate-to-covered positions.
        cover_indices: Covered local candidate positions.
        reverse_indptr: CSR row pointer for target-position-to-covering rows.
        reverse_indices: Covering local candidate rows.
        covered_counts: Initial marginal coverage count for each candidate.
    """

    cover_indptr: np.ndarray
    cover_indices: np.ndarray
    reverse_indptr: np.ndarray
    reverse_indices: np.ndarray
    covered_counts: np.ndarray


class SALTExactStrategy(OurMethodMarginWeightedSqrtStrategy):
    """
    Sparse exact implementation of `our_method_margin_weighted_sqrt`.

    The mathematical scoring path is inherited from
    `OurMethodMarginWeightedSqrtStrategy`: margin uncertainty, dynamic `log(D)`
    coverage gain, optional early radius normalization, and next-round
    `ceil(sqrt(weight))` replicated training all keep the same definitions.  The
    implementation difference is that coverage sets are produced from a cached
    exact `r_max` CSR radius graph instead of from a dense `N x N` distance
    matrix.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Initialize SALT exact with the same public parameters as the sqrt strategy.

        The constructor deliberately accepts the parent strategy's full parameter
        set so existing experiment configuration can switch only the strategy
        name.  `max_theoretical_radius` is required because this value is the
        global `r_max` used to build `C_max`; without it the sparse graph would
        not have a single exact maximum radius bound.

        Args:
            *args: Positional arguments forwarded to
                `OurMethodMarginWeightedSqrtStrategy`.
            **kwargs: Keyword arguments forwarded to
                `OurMethodMarginWeightedSqrtStrategy`.

        Raises:
            ValueError: If `max_theoretical_radius` is missing.
        """
        super().__init__(*args, **kwargs)
        if self.max_theoretical_radius is None:
            raise ValueError(
                "salt_exact 需要显式设置 our_method_max_theoretical_radius，"
                "该值会作为全局 r_max 构建 CSR 覆盖图。"
            )
        self.name = "SALTExact"
        self._radius_graph: Optional[SparseRadiusGraph] = None

    def _radius_graph_cache_dir(self) -> str:
        """
        Resolve the on-disk cache directory for this strategy's CSR radius graph.

        The path includes dataset name and `r_max`, which avoids accidental reuse
        across datasets or different maximum-radius settings.  It intentionally
        lives under the existing `distance_cache_dir` so current experiment
        directory conventions continue to work.

        Returns:
            str: Directory path containing CSR graph files.

        Raises:
            RuntimeError: If dataset metadata has not been set before sampling.
        """
        if self.dataset_name is None:
            raise RuntimeError("数据集元数据必须先设置才能构建 salt_exact 半径图。")

        radius_token = f"{float(self.max_theoretical_radius):.12g}"
        radius_token = re.sub(r"[^0-9A-Za-z_.-]+", "_", radius_token)
        dirname = f"{self.dataset_name}_salt_exact_rmax_{radius_token}"
        return os.path.join(self.distance_cache_dir, dirname)

    def _extract_feature_matrix(self, dataset: Any) -> np.ndarray:
        """
        Extract the full training feature matrix from the active-learning dataset.

        SALT exact builds an input-space radius graph, so it needs the same vector
        representation that the dense distance matrix used previously.  The
        current embedded-data pipeline uses `torch.utils.data.TensorDataset`,
        where the first tensor is the feature matrix.  This function also accepts
        a raw numpy array or torch tensor to keep small demos and tests simple.

        Args:
            dataset: Full training dataset passed by the active-learning loop.

        Returns:
            np.ndarray: Two-dimensional float32 feature matrix with shape
            `(n_samples, n_features)`.

        Raises:
            RuntimeError: If the dataset does not expose vector features in a
            supported format.
            ValueError: If the extracted feature matrix is not two-dimensional.
        """
        if isinstance(dataset, np.ndarray):
            features = dataset
        elif isinstance(dataset, torch.Tensor):
            features = dataset.detach().cpu().numpy()
        elif hasattr(dataset, "tensors") and len(dataset.tensors) > 0:
            feature_tensor = dataset.tensors[0]
            if isinstance(feature_tensor, torch.Tensor):
                features = feature_tensor.detach().cpu().numpy()
            else:
                features = np.asarray(feature_tensor)
        else:
            raise RuntimeError(
                "salt_exact 当前需要向量形式数据集，例如 TensorDataset(features, labels)。"
            )

        features = np.asarray(features, dtype=np.float32)
        if features.ndim != 2:
            raise ValueError(
                "salt_exact 需要二维特征矩阵，当前特征形状为 "
                f"{features.shape}。"
            )
        return features

    def _load_radius_graph_from_cache(self, cache_dir: str, n_samples: int) -> Optional[SparseRadiusGraph]:
        """
        Load an existing CSR radius graph cache if it matches the current run.

        The cache stores large edge arrays as raw binary files so they can be
        memory-mapped instead of fully loaded.  Metadata is checked before use to
        ensure the graph was built for the same dataset size and `r_max`.

        Args:
            cache_dir: Directory that should contain `metadata.json`, `indptr.npy`,
                `indices.int32.bin`, and `distances.float32.bin`.
            n_samples: Expected number of rows in the graph.

        Returns:
            Optional[SparseRadiusGraph]: Loaded graph when the cache is valid,
            otherwise `None`.
        """
        metadata_path = os.path.join(cache_dir, "metadata.json")
        indptr_path = os.path.join(cache_dir, "indptr.npy")
        indices_path = os.path.join(cache_dir, "indices.int32.bin")
        distances_path = os.path.join(cache_dir, "distances.float32.bin")

        required_paths = [metadata_path, indptr_path, indices_path, distances_path]
        if not all(os.path.exists(path) for path in required_paths):
            return None

        with open(metadata_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)

        cached_n = int(metadata.get("n_samples", -1))
        cached_r_max = float(metadata.get("r_max", np.nan))
        num_edges = int(metadata.get("num_edges", -1))
        if (
            cached_n != int(n_samples)
            or not np.isclose(cached_r_max, float(self.max_theoretical_radius))
            or num_edges < 0
        ):
            return None

        indptr = np.load(indptr_path, mmap_mode="r")
        indices = np.memmap(indices_path, dtype=np.int32, mode="r", shape=(num_edges,))
        distances = np.memmap(
            distances_path,
            dtype=np.float32,
            mode="r",
            shape=(num_edges,)
        )
        print(
            "[SALTExact] 已加载 CSR 最大覆盖邻域: "
            f"n={cached_n}, edges={num_edges}, r_max={cached_r_max:.6f}"
        )
        return SparseRadiusGraph(
            n_samples=cached_n,
            r_max=cached_r_max,
            indptr=indptr,
            indices=indices,
            distances=distances
        )

    def _compute_squared_distance_block(
        self,
        query_vectors: np.ndarray,
        target_vectors: np.ndarray,
        device: torch.device
    ) -> np.ndarray:
        """
        Compute an exact squared Euclidean distance block.

        This function is used only while building the maximum-radius graph.  It
        computes the same metric as the original dense distance matrix, but only
        for one query-target tile at a time.  CUDA/MPS devices are used when
        available through the active-learning device argument; otherwise numpy is
        used on CPU.

        Args:
            query_vectors: Query feature tile with shape `(q, d)`.
            target_vectors: Target feature tile with shape `(t, d)`.
            device: Current torch device.

        Returns:
            np.ndarray: Squared distance matrix with shape `(q, t)`.
        """
        if device.type in {"cuda", "mps"}:
            with torch.no_grad():
                query_tensor = torch.as_tensor(
                    query_vectors,
                    dtype=torch.float32,
                    device=device
                )
                target_tensor = torch.as_tensor(
                    target_vectors,
                    dtype=torch.float32,
                    device=device
                )
                query_norms = torch.sum(query_tensor * query_tensor, dim=1, keepdim=True)
                target_norms = torch.sum(target_tensor * target_tensor, dim=1, keepdim=True)
                distance_sq = query_norms + target_norms.t() - 2.0 * (
                    query_tensor @ target_tensor.t()
                )
                distance_sq = torch.clamp(distance_sq, min=0.0)
                return distance_sq.detach().cpu().numpy()

        query_norms = np.sum(query_vectors * query_vectors, axis=1, keepdims=True)
        target_norms = np.sum(target_vectors * target_vectors, axis=1, keepdims=True)
        distance_sq = query_norms + target_norms.T - 2.0 * (query_vectors @ target_vectors.T)
        return np.maximum(distance_sq, 0.0, out=distance_sq)

    def _resolve_cuda_radius_graph_block_sizes(
        self,
        n_samples: int,
        device: torch.device
    ) -> Tuple[int, int]:
        """
        Resolve CUDA tile sizes for exact radius-graph construction.

        The CPU fallback uses small tiles because it copies every dense distance
        block back to host memory before thresholding.  On CUDA the threshold is
        applied on the device, so much larger matrix-multiplication tiles are
        preferable: they reduce Python loop overhead and keep the A800-style GPU
        saturated.  The defaults are intentionally conservative for 80GB cards,
        and can be overridden without changing experiment code through:

        - ``SALT_EXACT_CUDA_QUERY_BLOCK``
        - ``SALT_EXACT_CUDA_TARGET_BLOCK``
        - ``SALT_EXACT_CUDA_MEMORY_FRACTION``

        The memory fraction limits the approximate ``query_block * target_block``
        pair budget after feature tensors have already been moved to CUDA.  This
        does not change any mathematical result; it only controls how much of the
        exact all-pairs scan is processed per GPU kernel batch.

        Args:
            n_samples: Number of feature rows in the full dataset.
            device: CUDA device used by the current active-learning run.

        Returns:
            Tuple[int, int]: Query and target block sizes for the CUDA builder.
        """
        default_query_block = 8192
        default_target_block = 131072

        query_block_text = os.environ.get("SALT_EXACT_CUDA_QUERY_BLOCK")
        target_block_text = os.environ.get("SALT_EXACT_CUDA_TARGET_BLOCK")
        memory_fraction_text = os.environ.get("SALT_EXACT_CUDA_MEMORY_FRACTION")

        query_block_size = default_query_block
        target_block_size = default_target_block
        if query_block_text is not None:
            query_block_size = int(query_block_text)
        if target_block_text is not None:
            target_block_size = int(target_block_text)
        if query_block_size <= 0 or target_block_size <= 0:
            raise ValueError("SALT exact CUDA block size 必须为正整数。")

        query_block_size = min(int(query_block_size), int(n_samples))
        target_block_size = min(int(target_block_size), int(n_samples))

        memory_fraction = 0.25
        if memory_fraction_text is not None:
            memory_fraction = float(memory_fraction_text)
        if not np.isfinite(memory_fraction) or memory_fraction <= 0.0:
            raise ValueError("SALT_EXACT_CUDA_MEMORY_FRACTION 必须为正数。")
        memory_fraction = min(memory_fraction, 0.90)

        try:
            free_bytes, _ = torch.cuda.mem_get_info(device)
        except (AttributeError, RuntimeError, TypeError):
            free_bytes = torch.cuda.get_device_properties(device).total_memory

        # The dominant live tensors per tile are the float32 distance block, a
        # boolean mask, and PyTorch expression temporaries.  Keep the estimate
        # conservative so the automatic tile size is stable on shared servers.
        bytes_per_pair_budget = 16
        pair_budget = max(
            1,
            int((float(free_bytes) * memory_fraction) // bytes_per_pair_budget)
        )
        current_pairs = int(query_block_size) * int(target_block_size)
        if current_pairs > pair_budget:
            target_block_size = max(1, int(pair_budget // int(query_block_size)))
            target_block_size = min(target_block_size, int(n_samples))
            if int(query_block_size) * int(target_block_size) > pair_budget:
                query_block_size = max(1, int(np.sqrt(pair_budget)))
                target_block_size = max(1, int(pair_budget // query_block_size))
                query_block_size = min(query_block_size, int(n_samples))
                target_block_size = min(target_block_size, int(n_samples))

        return int(query_block_size), int(target_block_size)

    def _finalize_radius_graph_cache(
        self,
        cache_dir: str,
        counts: np.ndarray,
        edge_count: int,
        tmp_indices_path: str,
        tmp_distances_path: str,
        n_samples: int,
        r_max: float,
        builder_name: str
    ) -> SparseRadiusGraph:
        """
        Finish writing a radius graph cache and reload it as memory-mapped CSR.

        Both the original tiled builder and the CUDA-side filtering builder emit
        the same two raw edge streams: int32 neighbor indices and float32
        Euclidean distances.  This helper centralizes the shared finalization
        steps so the on-disk cache format stays identical regardless of how the
        exact edges were discovered.

        Args:
            cache_dir: Directory that receives the final CSR cache files.
            counts: Number of stored edges in each graph row.
            edge_count: Total number of directed radius edges written.
            tmp_indices_path: Temporary raw index stream path.
            tmp_distances_path: Temporary raw distance stream path.
            n_samples: Number of rows in the graph.
            r_max: Maximum radius used for edge inclusion.
            builder_name: Human-readable builder identifier saved in metadata.

        Returns:
            SparseRadiusGraph: Reloaded CSR graph backed by memory-mapped arrays.

        Raises:
            RuntimeError: If the just-written cache cannot be loaded.
        """
        final_indices_path = os.path.join(cache_dir, "indices.int32.bin")
        final_distances_path = os.path.join(cache_dir, "distances.float32.bin")
        final_indptr_path = os.path.join(cache_dir, "indptr.npy")
        final_metadata_path = os.path.join(cache_dir, "metadata.json")

        indptr = np.empty(n_samples + 1, dtype=np.int64)
        indptr[0] = 0
        np.cumsum(counts, out=indptr[1:])
        np.save(final_indptr_path, indptr)
        os.replace(tmp_indices_path, final_indices_path)
        os.replace(tmp_distances_path, final_distances_path)

        metadata: Dict[str, Any] = {
            "format": "salt_exact_csr_radius_graph_v1",
            "dataset_name": self.dataset_name,
            "n_samples": n_samples,
            "r_max": r_max,
            "num_edges": int(edge_count),
            "index_dtype": "int32",
            "distance_dtype": "float32",
            "metric": "euclidean",
            "builder": builder_name,
        }
        with open(final_metadata_path, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)

        print(
            "[SALTExact] CSR 最大覆盖邻域构建完成: "
            f"edges={edge_count}, avg_degree={edge_count / max(n_samples, 1):.2f}"
        )
        loaded_graph = self._load_radius_graph_from_cache(cache_dir, n_samples)
        if loaded_graph is None:
            raise RuntimeError("CSR 半径图已写入但无法重新加载。")
        return loaded_graph

    def _build_radius_graph_cache_cuda(
        self,
        features: np.ndarray,
        cache_dir: str,
        device: torch.device
    ) -> SparseRadiusGraph:
        """
        Build the exact `r_max` CSR radius graph with CUDA-side edge filtering.

        This path keeps the same distance formula and edge definition as the
        original tiled builder:

        ``||x - y||_2^2 = ||x||_2^2 + ||y||_2^2 - 2 x^T y``

        and stores exactly those directed edges whose Euclidean distance is at
        most ``r_max``.  The speed improvement comes from where the threshold is
        applied.  Instead of copying every dense distance tile back to CPU and
        then running ``np.flatnonzero`` row by row, this method keeps the dense
        tile on GPU, computes ``distance_sq <= r_max_sq`` on GPU, and transfers
        only the matching edge coordinates and distances to host memory.  For a
        sparse radius graph this changes PCIe traffic from all pairwise distances
        to the final CSR edge stream.

        Args:
            features: Full feature matrix with shape `(n_samples, n_features)`.
            cache_dir: Directory where graph files will be written.
            device: CUDA device used for tiled distance computation.

        Returns:
            SparseRadiusGraph: Memory-mapped graph loaded from the written cache.
        """
        os.makedirs(cache_dir, exist_ok=True)
        n_samples = int(features.shape[0])
        r_max = float(self.max_theoretical_radius)
        r_max_sq = float(np.float32(r_max * r_max))

        tmp_indices_path = os.path.join(cache_dir, "indices.int32.tmp")
        tmp_distances_path = os.path.join(cache_dir, "distances.float32.tmp")

        features = np.asarray(features, dtype=np.float32, order="C")
        with torch.no_grad():
            feature_tensor = torch.as_tensor(
                features,
                dtype=torch.float32,
                device=device
            ).contiguous()
            feature_norms = torch.sum(feature_tensor * feature_tensor, dim=1)

            query_block_size, target_block_size = (
                self._resolve_cuda_radius_graph_block_sizes(
                    n_samples=n_samples,
                    device=device
                )
            )
            print(
                "[SALTExact] 使用 CUDA 构建 CSR 最大覆盖邻域: "
                f"n={n_samples}, r_max={r_max:.6f}, "
                f"query_block={query_block_size}, target_block={target_block_size}"
            )

            counts = np.zeros(n_samples, dtype=np.int64)
            edge_count = 0
            with open(tmp_indices_path, "wb") as index_file, open(
                tmp_distances_path,
                "wb"
            ) as distance_file:
                query_iter = range(0, n_samples, query_block_size)
                for query_start in tqdm(
                    query_iter,
                    total=(n_samples + query_block_size - 1) // query_block_size,
                    desc="[SALTExact] CUDA 构建 r_max 邻域",
                    unit="block"
                ):
                    query_end = min(query_start + query_block_size, n_samples)
                    query_vectors = feature_tensor[query_start:query_end]
                    query_norms = feature_norms[query_start:query_end].unsqueeze(1)
                    row_count = query_end - query_start
                    row_neighbors: List[List[np.ndarray]] = [
                        [] for _ in range(row_count)
                    ]
                    row_distances: List[List[np.ndarray]] = [
                        [] for _ in range(row_count)
                    ]

                    for target_start in range(0, n_samples, target_block_size):
                        target_end = min(target_start + target_block_size, n_samples)
                        target_vectors = feature_tensor[target_start:target_end]
                        target_norms = feature_norms[target_start:target_end].unsqueeze(0)
                        distance_sq = query_norms + target_norms - 2.0 * (
                            query_vectors @ target_vectors.t()
                        )
                        distance_sq.clamp_(min=0.0)
                        in_radius = distance_sq <= r_max_sq
                        hit_rows, hit_cols = torch.nonzero(
                            in_radius,
                            as_tuple=True
                        )
                        if int(hit_rows.numel()) == 0:
                            del distance_sq, in_radius
                            continue

                        hit_distances = torch.sqrt(distance_sq[hit_rows, hit_cols])
                        hit_rows_cpu = hit_rows.cpu().numpy().astype(
                            np.int64,
                            copy=False
                        )
                        hit_cols_cpu = hit_cols.cpu().numpy().astype(
                            np.int64,
                            copy=False
                        )
                        hit_targets_cpu = (
                            hit_cols_cpu + int(target_start)
                        ).astype(np.int32, copy=False)
                        hit_distances_cpu = hit_distances.cpu().numpy().astype(
                            np.float32,
                            copy=False
                        )

                        unique_rows, start_offsets, row_edge_counts = np.unique(
                            hit_rows_cpu,
                            return_index=True,
                            return_counts=True
                        )
                        for row_offset, start_offset, row_edge_count in zip(
                            unique_rows,
                            start_offsets,
                            row_edge_counts
                        ):
                            start_offset = int(start_offset)
                            end_offset = start_offset + int(row_edge_count)
                            row_index = int(row_offset)
                            row_neighbors[row_index].append(
                                hit_targets_cpu[start_offset:end_offset]
                            )
                            row_distances[row_index].append(
                                hit_distances_cpu[start_offset:end_offset]
                            )

                        del (
                            distance_sq,
                            in_radius,
                            hit_rows,
                            hit_cols,
                            hit_distances,
                        )

                    for row_offset, global_row in enumerate(range(query_start, query_end)):
                        if row_neighbors[row_offset]:
                            neighbors = np.concatenate(row_neighbors[row_offset]).astype(
                                np.int32,
                                copy=False
                            )
                            distances = np.concatenate(row_distances[row_offset]).astype(
                                np.float32,
                                copy=False
                            )
                        else:
                            neighbors = np.empty((0,), dtype=np.int32)
                            distances = np.empty((0,), dtype=np.float32)

                        counts[global_row] = int(neighbors.size)
                        edge_count += int(neighbors.size)
                        neighbors.tofile(index_file)
                        distances.tofile(distance_file)

        return self._finalize_radius_graph_cache(
            cache_dir=cache_dir,
            counts=counts,
            edge_count=edge_count,
            tmp_indices_path=tmp_indices_path,
            tmp_distances_path=tmp_distances_path,
            n_samples=n_samples,
            r_max=r_max,
            builder_name="cuda_device_threshold"
        )

    def _build_radius_graph_cache_tiled(
        self,
        features: np.ndarray,
        cache_dir: str,
        device: torch.device
    ) -> SparseRadiusGraph:
        """
        Build and persist the exact `r_max` CSR radius graph without a dense matrix.

        The builder scans the feature matrix in tiles, writes edge indices and
        distances sequentially to raw binary files, and stores only one row-count
        vector in memory.  This keeps storage proportional to `E_max` instead of
        `N^2`.  The computation is still an exact all-pairs radius search, so the
        worst-case compute time remains quadratic; the important difference is
        that no dense pairwise distance matrix is materialized or cached.

        Args:
            features: Full feature matrix with shape `(n_samples, n_features)`.
            cache_dir: Directory where graph files will be written.
            device: Current torch device used for tiled distance computation.

        Returns:
            SparseRadiusGraph: Memory-mapped graph loaded from the written cache.
        """
        os.makedirs(cache_dir, exist_ok=True)
        n_samples = int(features.shape[0])
        r_max = float(self.max_theoretical_radius)
        r_max_sq = np.float32(r_max * r_max)

        query_block_size = max(1, min(int(self.inference_batch_size), 1024))
        target_block_size = min(n_samples, max(query_block_size, 16384))

        tmp_indices_path = os.path.join(cache_dir, "indices.int32.tmp")
        tmp_distances_path = os.path.join(cache_dir, "distances.float32.tmp")

        counts = np.zeros(n_samples, dtype=np.int64)
        edge_count = 0

        print(
            "[SALTExact] 构建 CSR 最大覆盖邻域: "
            f"n={n_samples}, r_max={r_max:.6f}, "
            f"query_block={query_block_size}, target_block={target_block_size}"
        )

        with open(tmp_indices_path, "wb") as index_file, open(
            tmp_distances_path,
            "wb"
        ) as distance_file:
            query_iter = range(0, n_samples, query_block_size)
            for query_start in tqdm(
                query_iter,
                total=(n_samples + query_block_size - 1) // query_block_size,
                desc="[SALTExact] 构建 r_max 邻域",
                unit="block"
            ):
                query_end = min(query_start + query_block_size, n_samples)
                query_vectors = features[query_start:query_end]
                row_neighbors: List[List[np.ndarray]] = [
                    [] for _ in range(query_end - query_start)
                ]
                row_distances: List[List[np.ndarray]] = [
                    [] for _ in range(query_end - query_start)
                ]

                for target_start in range(0, n_samples, target_block_size):
                    target_end = min(target_start + target_block_size, n_samples)
                    target_vectors = features[target_start:target_end]
                    distance_sq = self._compute_squared_distance_block(
                        query_vectors=query_vectors,
                        target_vectors=target_vectors,
                        device=device
                    )
                    in_radius = distance_sq <= r_max_sq
                    if not np.any(in_radius):
                        continue

                    for row_offset in range(in_radius.shape[0]):
                        local_targets = np.flatnonzero(in_radius[row_offset])
                        if local_targets.size == 0:
                            continue
                        global_targets = (
                            local_targets.astype(np.int64, copy=False) + target_start
                        )
                        row_neighbors[row_offset].append(
                            global_targets.astype(np.int32, copy=False)
                        )
                        row_distances[row_offset].append(
                            np.sqrt(
                                distance_sq[row_offset, local_targets],
                                dtype=np.float32
                            ).astype(np.float32, copy=False)
                        )

                for row_offset, global_row in enumerate(range(query_start, query_end)):
                    if row_neighbors[row_offset]:
                        neighbors = np.concatenate(row_neighbors[row_offset]).astype(
                            np.int32,
                            copy=False
                        )
                        distances = np.concatenate(row_distances[row_offset]).astype(
                            np.float32,
                            copy=False
                        )
                    else:
                        neighbors = np.empty((0,), dtype=np.int32)
                        distances = np.empty((0,), dtype=np.float32)

                    counts[global_row] = int(neighbors.size)
                    edge_count += int(neighbors.size)
                    neighbors.tofile(index_file)
                    distances.tofile(distance_file)

        return self._finalize_radius_graph_cache(
            cache_dir=cache_dir,
            counts=counts,
            edge_count=edge_count,
            tmp_indices_path=tmp_indices_path,
            tmp_distances_path=tmp_distances_path,
            n_samples=n_samples,
            r_max=r_max,
            builder_name="host_threshold_tiled"
        )

    def _build_radius_graph_cache(
        self,
        features: np.ndarray,
        cache_dir: str,
        device: torch.device
    ) -> SparseRadiusGraph:
        """
        Build and persist the exact `r_max` CSR radius graph.

        CUDA runs use a device-side thresholding path that avoids copying dense
        distance tiles back to CPU.  All non-CUDA runs, and CUDA runs that hit an
        out-of-memory condition, use the original host-thresholded tiled builder.
        Both paths compute the same Euclidean distances and write the same CSR
        cache format, so downstream selection and weighting semantics are
        unchanged.

        Args:
            features: Full feature matrix with shape `(n_samples, n_features)`.
            cache_dir: Directory where graph files will be written.
            device: Current torch device used for tiled distance computation.

        Returns:
            SparseRadiusGraph: Memory-mapped graph loaded from the written cache.
        """
        if device.type == "cuda" and torch.cuda.is_available():
            try:
                return self._build_radius_graph_cache_cuda(
                    features=features,
                    cache_dir=cache_dir,
                    device=device
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(
                    "[SALTExact] CUDA 构建 CSR 半径图显存不足，回退到原始分块构建。"
                    " 可调小 SALT_EXACT_CUDA_QUERY_BLOCK 或 "
                    "SALT_EXACT_CUDA_TARGET_BLOCK 后重试以保留 CUDA fast path。"
                )

        return self._build_radius_graph_cache_tiled(
            features=features,
            cache_dir=cache_dir,
            device=device
        )

    def _ensure_radius_graph(self, dataset: Any, device: torch.device) -> SparseRadiusGraph:
        """
        Load or build the exact maximum-radius CSR graph for the current dataset.

        This method is the sparse replacement for `_ensure_distance_matrix`.
        Once loaded, the graph is cached on the strategy instance for all later
        sampling rounds.

        Args:
            dataset: Full training dataset.
            device: Current torch device.

        Returns:
            SparseRadiusGraph: CSR maximum-radius coverage graph.
        """
        if self._radius_graph is not None:
            return self._radius_graph

        features = self._extract_feature_matrix(dataset)
        cache_dir = self._radius_graph_cache_dir()
        graph = self._load_radius_graph_from_cache(
            cache_dir=cache_dir,
            n_samples=features.shape[0]
        )
        if graph is None:
            graph = self._build_radius_graph_cache(
                features=features,
                cache_dir=cache_dir,
                device=device
            )

        self._radius_graph = graph
        return graph

    def _compute_sparse_coverage(
        self,
        radius_graph: SparseRadiusGraph,
        labeled_indices: Sequence[int],
        radii: Sequence[float]
    ) -> Tuple[float, np.ndarray]:
        """
        Compute current labeled coverage using only CSR maximum-radius rows.

        This is mathematically equivalent to the dense implementation because
        every input radius is clipped to `r_max`; therefore each true coverage set
        is exactly the subset of the precomputed CSR row whose stored distance is
        at most that point's current radius.

        Args:
            radius_graph: Maximum-radius CSR graph.
            labeled_indices: Full indices of currently labeled points.
            radii: Effective radii aligned with `labeled_indices`.

        Returns:
            Tuple[float, np.ndarray]: Coverage ratio and boolean covered mask over
            the full training pool.
        """
        if len(labeled_indices) != len(radii):
            raise ValueError("labeled_indices 与 radii 长度必须一致。")

        covered_mask = np.zeros(radius_graph.n_samples, dtype=bool)
        for index, radius in zip(labeled_indices, radii):
            neighbors, distances = radius_graph.row(int(index))
            true_neighbors = neighbors[distances <= float(radius)]
            covered_mask[true_neighbors.astype(np.int64, copy=False)] = True

        coverage_ratio = (
            float(np.count_nonzero(covered_mask)) / float(radius_graph.n_samples)
            if radius_graph.n_samples > 0 else 0.0
        )
        return coverage_ratio, covered_mask

    def _prepare_candidate_cover_data_from_radius_graph(
        self,
        radius_graph: SparseRadiusGraph,
        candidate_indices: np.ndarray,
        candidate_radii: np.ndarray
    ) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
        """
        Build exact candidate coverage lists and reverse index from the CSR graph.

        The returned data has the same shape and semantics as the parent class's
        dense `_prepare_candidate_cover_data_from_full_distance` output, allowing
        the existing exact incremental greedy update to be reused unchanged.

        Args:
            radius_graph: Maximum-radius CSR graph.
            candidate_indices: Candidate full indices in local greedy order.
            candidate_radii: Effective candidate radii aligned with
                `candidate_indices`.

        Returns:
            Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
            `covered_positions_by_candidate`, `reverse_cover_rows`, and initial
            marginal coverage counts.
        """
        candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
        candidate_radii = np.asarray(candidate_radii, dtype=np.float64)
        if candidate_indices.shape[0] != candidate_radii.shape[0]:
            raise ValueError("candidate_indices 与 candidate_radii 长度必须一致。")
        if candidate_indices.size == 0:
            return [], [], np.empty((0,), dtype=np.int64)

        num_candidates = int(candidate_indices.shape[0])
        global_to_local = np.full(radius_graph.n_samples, -1, dtype=np.int64)
        global_to_local[candidate_indices] = np.arange(num_candidates, dtype=np.int64)

        covered_positions_by_candidate: List[np.ndarray] = [
            np.empty((0,), dtype=np.int64) for _ in range(num_candidates)
        ]
        covered_counts = np.empty(num_candidates, dtype=np.int64)

        for local_position, (global_index, radius) in enumerate(
            zip(candidate_indices, candidate_radii)
        ):
            neighbors, distances = radius_graph.row(int(global_index))
            in_radius_neighbors = neighbors[distances <= float(radius)]
            local_targets = global_to_local[
                in_radius_neighbors.astype(np.int64, copy=False)
            ]
            local_targets = local_targets[local_targets >= 0]
            local_targets = local_targets.astype(np.int64, copy=False)
            covered_positions_by_candidate[int(local_position)] = local_targets
            covered_counts[int(local_position)] = int(local_targets.size)

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

    def _candidate_cover_scipy_memory_limit_bytes(self) -> Optional[int]:
        """
        Resolve the optional memory guard for SciPy candidate CSR construction.

        The SciPy fast path creates temporary CPU-side sparse matrices.  This is
        intentional because the cached radius graph is already a CPU/mmap CSR and
        SciPy can slice and transpose it in compiled code.  Large 1M-scale runs
        can still produce very large temporary edge sets, so this helper reads
        ``SALT_EXACT_SCIPY_MAX_TEMP_GB`` and converts it to a byte limit before
        the fast path allocates those arrays.  A value less than or equal to zero
        disables the guard.  When the variable is absent, the default is 256 GB,
        which is conservative for the 1.5 TiB RAM server while still protecting
        smaller machines from accidental huge allocations.

        Returns:
            Optional[int]: Maximum estimated temporary bytes, or ``None`` when
            no explicit guard should be applied.
        """
        limit_gb_text = os.environ.get("SALT_EXACT_SCIPY_MAX_TEMP_GB", "256")
        limit_gb = float(limit_gb_text)
        if not np.isfinite(limit_gb):
            raise ValueError("SALT_EXACT_SCIPY_MAX_TEMP_GB 必须为有限数值。")
        if limit_gb <= 0.0:
            return None
        return int(limit_gb * (1024 ** 3))

    def _check_candidate_cover_scipy_memory_budget(
        self,
        estimated_edges: int,
        context: str
    ) -> bool:
        """
        Decide whether the SciPy fast path may allocate a temporary edge set.

        The estimate is deliberately simple: for each temporary edge, budget for
        forward indices/data, sliced candidate graph storage, and reverse graph
        storage.  The exact SciPy overhead depends on index dtype and internal
        copies, so this guard is only a safety valve; it never changes which edges
        are valid.  Returning ``False`` merely sends the caller to the original
        Python two-pass builder.

        Args:
            estimated_edges: Number of candidate-row edges expected in the
                largest temporary sparse matrix before target-column slicing.
            context: Short label included in the fallback log message.

        Returns:
            bool: ``True`` when the fast path should continue, ``False`` when it
            should fall back before allocating large temporaries.
        """
        max_temp_bytes = self._candidate_cover_scipy_memory_limit_bytes()
        if max_temp_bytes is None:
            return True

        estimated_bytes = int(estimated_edges) * 24
        if estimated_bytes <= max_temp_bytes:
            return True

        print(
            "[SALTExact] 跳过 CandidateCoverCSR SciPy fast path: "
            f"context={context}, estimated_edges={int(estimated_edges)}, "
            f"estimated_temp_gb={estimated_bytes / (1024 ** 3):.2f}, "
            f"limit_gb={max_temp_bytes / (1024 ** 3):.2f}"
        )
        return False

    def _candidate_cover_csr_from_scipy_graph(
        self,
        cover_graph: Any
    ) -> CandidateCoverCSR:
        """
        Convert a SciPy candidate coverage graph into ``CandidateCoverCSR``.

        SciPy is used only as a fast constructor for sparse structure.  The SALT
        greedy code expects plain numpy CSR arrays and never reads sparse matrix
        data values, so this helper extracts the forward graph pointers/indices,
        builds the reverse graph with a compiled transpose, and converts dtypes
        back to the project convention.  Explicit zero data values are preserved
        by SciPy slicing, which matters for self-edges whose stored distance is
        zero.

        Args:
            cover_graph: SciPy CSR matrix whose rows and columns are both aligned
                with the current candidate order.

        Returns:
            CandidateCoverCSR: Forward and reverse CSR arrays plus row counts.
        """
        cover_graph = cover_graph.tocsr()
        reverse_graph = cover_graph.transpose().tocsr()

        cover_indptr = np.asarray(cover_graph.indptr, dtype=np.int64)
        cover_indices = np.asarray(cover_graph.indices, dtype=np.int32)
        reverse_indptr = np.asarray(reverse_graph.indptr, dtype=np.int64)
        reverse_indices = np.asarray(reverse_graph.indices, dtype=np.int32)
        covered_counts = np.diff(cover_indptr).astype(np.int64, copy=False)

        return CandidateCoverCSR(
            cover_indptr=cover_indptr,
            cover_indices=cover_indices,
            reverse_indptr=reverse_indptr,
            reverse_indices=reverse_indices,
            covered_counts=covered_counts
        )

    def _prepare_candidate_cover_csr_from_radius_graph_scipy(
        self,
        radius_graph: SparseRadiusGraph,
        candidate_indices: np.ndarray,
        candidate_radii: np.ndarray
    ) -> Optional[CandidateCoverCSR]:
        """
        Build ``CandidateCoverCSR`` with SciPy sparse slicing and transpose.

        This is the conservative CPU/RAM fast path for large experiments.  It
        keeps the exact same edge rule as the Python fallback:

        ``global target u is kept for candidate x iff
        distance(x, u) <= radius_x and u is also in the candidate pool``.

        When every candidate radius covers the full cached ``r_max`` row, the
        current graph is simply the induced subgraph
        ``C_max[candidates][:, candidates]``.  For per-candidate radii, the
        helper first constructs a temporary ``candidate -> global`` CSR after
        applying the row-specific radius threshold, then lets SciPy perform the
        expensive target-column restriction and reverse-CSR transpose in compiled
        code.  Returning ``None`` means the caller should use the original Python
        fallback; it does not change any mathematical result.

        Args:
            radius_graph: Maximum-radius CSR graph.
            candidate_indices: Candidate full indices in local greedy order.
            candidate_radii: Effective candidate radii aligned with candidates.

        Returns:
            Optional[CandidateCoverCSR]: Fast-path result, or ``None`` if SciPy is
            unavailable or a memory guard asks the caller to fall back.
        """
        if scipy_sparse is None:
            return None

        num_candidates = int(candidate_indices.shape[0])
        if num_candidates == 0:
            return self._empty_candidate_cover_csr()

        if np.all(candidate_radii >= float(radius_graph.r_max)):
            row_degrees = np.diff(radius_graph.indptr).astype(np.int64, copy=False)
            estimated_edges = int(np.sum(
                row_degrees[candidate_indices],
                dtype=np.int64
            ))
            if not self._check_candidate_cover_scipy_memory_budget(
                estimated_edges=estimated_edges,
                context="full-radius-subgraph"
            ):
                return None

            max_radius_graph = scipy_sparse.csr_matrix(
                (
                    radius_graph.distances,
                    radius_graph.indices,
                    radius_graph.indptr,
                ),
                shape=(radius_graph.n_samples, radius_graph.n_samples),
                copy=False,
            )
            cover_graph = max_radius_graph[candidate_indices, :][:, candidate_indices]
            return self._candidate_cover_csr_from_scipy_graph(cover_graph)

        filtered_counts = np.empty(num_candidates, dtype=np.int64)
        total_filtered_edges = 0
        for local_position, (global_index, radius) in enumerate(
            zip(candidate_indices, candidate_radii)
        ):
            start = int(radius_graph.indptr[int(global_index)])
            end = int(radius_graph.indptr[int(global_index) + 1])
            count = int(np.count_nonzero(
                radius_graph.distances[start:end] <= float(radius)
            ))
            filtered_counts[int(local_position)] = count
            total_filtered_edges += count

        if not self._check_candidate_cover_scipy_memory_budget(
            estimated_edges=total_filtered_edges,
            context="radius-filtered-subgraph"
        ):
            return None

        filtered_indptr = np.empty(num_candidates + 1, dtype=np.int64)
        filtered_indptr[0] = 0
        np.cumsum(filtered_counts, out=filtered_indptr[1:])
        filtered_indices = np.empty(total_filtered_edges, dtype=np.int32)
        filtered_data = np.ones(total_filtered_edges, dtype=np.uint8)

        for local_position, (global_index, radius) in enumerate(
            zip(candidate_indices, candidate_radii)
        ):
            start = int(radius_graph.indptr[int(global_index)])
            end = int(radius_graph.indptr[int(global_index) + 1])
            keep_mask = radius_graph.distances[start:end] <= float(radius)
            row_start = int(filtered_indptr[int(local_position)])
            row_end = int(filtered_indptr[int(local_position) + 1])
            filtered_indices[row_start:row_end] = radius_graph.indices[start:end][
                keep_mask
            ]

        filtered_graph = scipy_sparse.csr_matrix(
            (filtered_data, filtered_indices, filtered_indptr),
            shape=(num_candidates, radius_graph.n_samples),
            copy=False,
        )
        cover_graph = filtered_graph[:, candidate_indices]
        return self._candidate_cover_csr_from_scipy_graph(cover_graph)

    def _empty_candidate_cover_csr(self) -> CandidateCoverCSR:
        """
        Return an empty ``CandidateCoverCSR`` with the standard project dtypes.

        Several construction paths need to handle an empty candidate set before
        allocating any temporary graph.  Keeping this in one helper prevents dtype
        drift between the SciPy fast path and the original Python fallback.

        Returns:
            CandidateCoverCSR: Empty forward/reverse CSR arrays.
        """
        empty_indptr = np.zeros(1, dtype=np.int64)
        return CandidateCoverCSR(
            cover_indptr=empty_indptr,
            cover_indices=np.empty((0,), dtype=np.int32),
            reverse_indptr=empty_indptr.copy(),
            reverse_indices=np.empty((0,), dtype=np.int32),
            covered_counts=np.empty((0,), dtype=np.int64)
        )

    def _prepare_candidate_cover_csr_from_radius_graph_python(
        self,
        radius_graph: SparseRadiusGraph,
        candidate_indices: np.ndarray,
        candidate_radii: np.ndarray
    ) -> CandidateCoverCSR:
        """
        Build current-round true coverage and inverted index as compact CSR arrays.

        For every active candidate `x`, this function scans the precomputed
        maximum-radius row `C_max(x)`, filters edges with `distance <= r_x`, and
        then keeps only targets that are also in the current candidate pool.  The
        result is exact because all radii are clipped to the same `r_max` used to
        build `C_max`.

        The construction is deliberately two-pass:
        1. Count forward and reverse degrees without storing per-row Python lists;
        2. Allocate flat CSR arrays and fill them in a second scan.
        This keeps the per-round representation proportional to the true edge
        count `E` and avoids a million small numpy arrays on 1M-scale pools.

        Args:
            radius_graph: Maximum-radius CSR graph.
            candidate_indices: Candidate full indices in local greedy order.
            candidate_radii: Effective candidate radii aligned with
                `candidate_indices`.

        Returns:
            CandidateCoverCSR: Forward and reverse CSR coverage structures plus
            initial marginal coverage counts.
        """
        candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
        candidate_radii = np.asarray(candidate_radii, dtype=np.float64)
        if candidate_indices.shape[0] != candidate_radii.shape[0]:
            raise ValueError("candidate_indices 与 candidate_radii 长度必须一致。")

        num_candidates = int(candidate_indices.shape[0])
        if num_candidates == 0:
            empty_indptr = np.zeros(1, dtype=np.int64)
            return CandidateCoverCSR(
                cover_indptr=empty_indptr,
                cover_indices=np.empty((0,), dtype=np.int32),
                reverse_indptr=empty_indptr.copy(),
                reverse_indices=np.empty((0,), dtype=np.int32),
                covered_counts=np.empty((0,), dtype=np.int64)
            )

        global_to_local = np.full(radius_graph.n_samples, -1, dtype=np.int64)
        global_to_local[candidate_indices] = np.arange(num_candidates, dtype=np.int64)

        forward_counts = np.zeros(num_candidates, dtype=np.int64)
        reverse_counts = np.zeros(num_candidates, dtype=np.int64)

        for local_position, (global_index, radius) in enumerate(
            zip(candidate_indices, candidate_radii)
        ):
            neighbors, distances = radius_graph.row(int(global_index))
            in_radius_neighbors = neighbors[distances <= float(radius)]
            local_targets = global_to_local[
                in_radius_neighbors.astype(np.int64, copy=False)
            ]
            local_targets = local_targets[local_targets >= 0]
            forward_counts[int(local_position)] = int(local_targets.size)
            if local_targets.size > 0:
                np.add.at(reverse_counts, local_targets, 1)

        cover_indptr = np.empty(num_candidates + 1, dtype=np.int64)
        cover_indptr[0] = 0
        np.cumsum(forward_counts, out=cover_indptr[1:])
        reverse_indptr = np.empty(num_candidates + 1, dtype=np.int64)
        reverse_indptr[0] = 0
        np.cumsum(reverse_counts, out=reverse_indptr[1:])

        num_edges = int(cover_indptr[-1])
        cover_indices = np.empty(num_edges, dtype=np.int32)
        reverse_indices = np.empty(num_edges, dtype=np.int32)
        reverse_cursor = reverse_indptr[:-1].copy()

        for local_position, (global_index, radius) in enumerate(
            zip(candidate_indices, candidate_radii)
        ):
            neighbors, distances = radius_graph.row(int(global_index))
            in_radius_neighbors = neighbors[distances <= float(radius)]
            local_targets = global_to_local[
                in_radius_neighbors.astype(np.int64, copy=False)
            ]
            local_targets = local_targets[local_targets >= 0].astype(
                np.int64,
                copy=False
            )

            row_start = int(cover_indptr[int(local_position)])
            row_end = int(cover_indptr[int(local_position) + 1])
            cover_indices[row_start:row_end] = local_targets.astype(
                np.int32,
                copy=False
            )
            if local_targets.size > 0:
                reverse_positions = reverse_cursor[local_targets]
                reverse_indices[reverse_positions] = int(local_position)
                reverse_cursor[local_targets] += 1

        return CandidateCoverCSR(
            cover_indptr=cover_indptr,
            cover_indices=cover_indices,
            reverse_indptr=reverse_indptr,
            reverse_indices=reverse_indices,
            covered_counts=forward_counts
        )

    def _prepare_candidate_cover_csr_from_radius_graph(
        self,
        radius_graph: SparseRadiusGraph,
        candidate_indices: np.ndarray,
        candidate_radii: np.ndarray
    ) -> CandidateCoverCSR:
        """
        Build current-round true coverage CSR, preferring the SciPy fast path.

        This public-in-class helper preserves the exact semantics used by both
        ``salt_exact`` and ``salt_exact_soft``.  It first asks SciPy to perform
        the candidate subgraph slicing and reverse transpose in compiled sparse
        code.  If SciPy is unavailable, the memory guard declines the temporary
        allocation, or SciPy raises an allocation/format error, the method falls
        back to the original Python two-pass implementation.  The fallback uses
        the same edge predicate, so the selected samples and training weights
        remain mathematically unchanged.

        Args:
            radius_graph: Maximum-radius CSR graph.
            candidate_indices: Candidate full indices in local greedy order.
            candidate_radii: Effective candidate radii aligned with candidates.

        Returns:
            CandidateCoverCSR: Forward and reverse CSR coverage structures plus
            initial marginal coverage counts.
        """
        candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
        candidate_radii = np.asarray(candidate_radii, dtype=np.float64)
        if candidate_indices.shape[0] != candidate_radii.shape[0]:
            raise ValueError("candidate_indices 与 candidate_radii 长度必须一致。")
        if candidate_indices.size == 0:
            return self._empty_candidate_cover_csr()

        try:
            cover_data = self._prepare_candidate_cover_csr_from_radius_graph_scipy(
                radius_graph=radius_graph,
                candidate_indices=candidate_indices,
                candidate_radii=candidate_radii
            )
            if cover_data is not None:
                return cover_data
        except (MemoryError, OverflowError, RuntimeError, ValueError) as error:
            print(
                "[SALTExact] CandidateCoverCSR SciPy fast path 失败，"
                f"回退到 Python 构建: {type(error).__name__}: {error}"
            )

        return self._prepare_candidate_cover_csr_from_radius_graph_python(
            radius_graph=radius_graph,
            candidate_indices=candidate_indices,
            candidate_radii=candidate_radii
        )

    def _candidate_cover_row(
        self,
        cover_data: CandidateCoverCSR,
        local_position: int
    ) -> np.ndarray:
        """
        Return the true covered target positions for one candidate row.

        The helper hides CSR pointer arithmetic from the greedy loop and returns a
        view over `cover_indices`, so selecting a row does not allocate unless the
        caller later applies a boolean mask.

        Args:
            cover_data: Current-round candidate coverage CSR.
            local_position: Candidate row in local greedy order.

        Returns:
            np.ndarray: Covered local candidate positions.
        """
        start = int(cover_data.cover_indptr[local_position])
        end = int(cover_data.cover_indptr[local_position + 1])
        return cover_data.cover_indices[start:end].astype(np.int64, copy=False)

    def _decrement_counts_with_reverse_csr(
        self,
        covered_counts: np.ndarray,
        active_mask: np.ndarray,
        cover_data: CandidateCoverCSR,
        removed_positions: np.ndarray
    ) -> None:
        """
        Exact marginal-gain update using the current true inverted CSR index.

        For each newly removed/covered target `u`, the reverse CSR row lists every
        candidate `x` whose true `C(x)` contains `u`.  Each active such candidate
        loses exactly one marginal coverage unit.  Repeating this for all removed
        targets implements `Delta'(x) = Delta(x) - |C(x) cap A|` without scanning
        all candidates.

        Args:
            covered_counts: Mutable marginal coverage counts for candidates.
            active_mask: Boolean mask of candidates still available for selection.
            cover_data: Current true cover CSR and reverse CSR.
            removed_positions: Local candidate positions newly covered/removed.
        """
        removed_positions = np.asarray(removed_positions, dtype=np.int64)
        for target_position in removed_positions:
            start = int(cover_data.reverse_indptr[int(target_position)])
            end = int(cover_data.reverse_indptr[int(target_position) + 1])
            if start == end:
                continue
            covering_rows = cover_data.reverse_indices[start:end].astype(
                np.int64,
                copy=False
            )
            active_rows = covering_rows[active_mask[covering_rows]]
            if active_rows.size > 0:
                np.add.at(covered_counts, active_rows, -1)

    def _greedy_select_candidate_positions_from_cover_csr(
        self,
        candidate_indices: np.ndarray,
        static_score_terms: np.ndarray,
        cover_data: CandidateCoverCSR,
        query_budget: int,
        progress_desc: str,
        log_prefix: str
    ) -> List[int]:
        """
        Select candidate positions with exact greedy updates over CSR cover data.

        This is the CSR equivalent of the parent class's list-based greedy loop.
        Scores remain `static_score + log(current_marginal_coverage)`, selected
        points are removed from the active pool, and marginal gains are updated
        through the true reverse CSR index.  The D=1 batch shortcut is preserved
        because it does not change selection order relative to repeated greedy
        choices among unit-coverage candidates.

        Args:
            candidate_indices: Candidate full indices in local greedy order.
            static_score_terms: Static uncertainty terms aligned with candidates.
            cover_data: Current true coverage CSR.
            query_budget: Maximum number of samples to select.
            progress_desc: Progress-bar description.
            log_prefix: Prefix for log messages.

        Returns:
            List[int]: Selected local candidate positions.
        """
        candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
        static_score_terms = np.asarray(static_score_terms, dtype=np.float64)

        selected_local_positions: List[int] = []
        if candidate_indices.size == 0 or query_budget <= 0:
            return selected_local_positions

        active_mask = np.ones(candidate_indices.shape[0], dtype=bool)
        covered_counts = cover_data.covered_counts.astype(np.int64, copy=True)

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
                best_covered_positions = self._candidate_cover_row(
                    cover_data=cover_data,
                    local_position=best_local_position
                )
                best_covered_positions = best_covered_positions[
                    active_mask[best_covered_positions]
                ]

                if best_covered_positions.size == 0:
                    break

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
                    self._decrement_counts_with_reverse_csr(
                        covered_counts=covered_counts,
                        active_mask=active_mask,
                        cover_data=cover_data,
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
                self._decrement_counts_with_reverse_csr(
                    covered_counts=covered_counts,
                    active_mask=active_mask,
                    cover_data=cover_data,
                    removed_positions=best_covered_positions
                )
                progress_bar.update(1)

        return selected_local_positions

    def _greedy_selection_from_radius_graph(
        self,
        unlabeled_indices: List[int],
        radius_graph: SparseRadiusGraph,
        radii: np.ndarray,
        query_budget: int,
        progress_desc: str,
        log_prefix: str
    ) -> Tuple[List[int], List[int]]:
        """
        Run the parent exact greedy update using sparse CSR-derived cover data.

        This method mirrors the first-round parent `_greedy_selection`, but the
        coverage data comes from `C_max` filtering instead of dense matrix rows.

        Args:
            unlabeled_indices: Active unlabeled candidate full indices.
            radius_graph: Maximum-radius CSR graph.
            radii: Effective radii aligned with `unlabeled_indices`.
            query_budget: Maximum number of points to select.
            progress_desc: Progress-bar description.
            log_prefix: Prefix used in greedy log messages.

        Returns:
            Tuple[List[int], List[int]]: Selected full indices and all full
            indices covered by the selected candidates within this candidate pool.
        """
        selected_indices: List[int] = []
        covered_indices: List[int] = []
        if not unlabeled_indices:
            return selected_indices, covered_indices

        local_indices = np.asarray(unlabeled_indices, dtype=np.int64)
        local_radii = np.asarray(radii, dtype=np.float64)
        cover_data = self._prepare_candidate_cover_csr_from_radius_graph(
            radius_graph=radius_graph,
            candidate_indices=local_indices,
            candidate_radii=local_radii
        )
        selected_local_positions = self._greedy_select_candidate_positions_from_cover_csr(
            candidate_indices=local_indices,
            static_score_terms=np.zeros(local_indices.shape[0], dtype=np.float64),
            cover_data=cover_data,
            query_budget=query_budget,
            progress_desc=progress_desc,
            log_prefix=log_prefix
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
            self._candidate_cover_row(
                cover_data=cover_data,
                local_position=int(local_position)
            )
            for local_position in selected_local_positions_array
        ])
        covered_indices = local_indices[np.unique(covered_positions)].astype(
            np.int64,
            copy=False
        ).tolist()
        return selected_indices, covered_indices

    def _compute_repeat_counts_for_next_round_sparse(
        self,
        radius_graph: SparseRadiusGraph,
        current_labeled_indices: Sequence[int],
        current_labeled_radii: Sequence[float],
        selected_indices: Sequence[int],
        selected_radii: Sequence[float]
    ) -> List[int]:
        """
        Compute next-round replicated-training counts from the sparse CSR graph.

        This keeps the same assignment rule as `get_weight_from_indices_and_radii`:
        each unlabeled point covered by one or more next-round labeled points is
        assigned to the labeled point with minimal `distance / radius`, and every
        labeled point starts with base weight one.  The final returned counts are
        compressed with `ceil(sqrt(weight))`, matching
        `OurMethodMarginWeightedSqrtStrategy`.

        Args:
            radius_graph: Maximum-radius CSR graph.
            current_labeled_indices: Labeled indices before the current selection.
            current_labeled_radii: Effective radii for `current_labeled_indices`.
            selected_indices: Newly selected indices.
            selected_radii: Effective radii for `selected_indices`.

        Returns:
            List[int]: Sqrt-compressed repeat counts aligned with
            `current_labeled_indices + selected_indices`.
        """
        if len(current_labeled_indices) != len(current_labeled_radii):
            raise ValueError(
                "current_labeled_indices 与 current_labeled_radii 长度不一致: "
                f"{len(current_labeled_indices)} vs {len(current_labeled_radii)}"
            )
        if len(selected_indices) != len(selected_radii):
            raise ValueError(
                "selected_indices 与 selected_radii 长度不一致: "
                f"{len(selected_indices)} vs {len(selected_radii)}"
            )

        next_labeled_indices = list(current_labeled_indices) + list(selected_indices)
        next_labeled_radii = list(current_labeled_radii) + list(selected_radii)
        if not next_labeled_indices:
            return []

        labeled_mask = np.zeros(radius_graph.n_samples, dtype=bool)
        labeled_mask[np.asarray(next_labeled_indices, dtype=np.int64)] = True
        best_ratio = np.full(radius_graph.n_samples, np.inf, dtype=np.float64)
        best_owner = np.full(radius_graph.n_samples, -1, dtype=np.int64)

        for owner_position, (global_index, radius) in enumerate(
            zip(next_labeled_indices, next_labeled_radii)
        ):
            radius = float(radius)
            if radius <= 0.0:
                continue
            neighbors, distances = radius_graph.row(int(global_index))
            in_radius_mask = distances <= radius
            if not np.any(in_radius_mask):
                continue

            covered_neighbors = neighbors[in_radius_mask].astype(np.int64, copy=False)
            covered_distances = distances[in_radius_mask].astype(np.float64, copy=False)
            unlabeled_neighbors = covered_neighbors[~labeled_mask[covered_neighbors]]
            if unlabeled_neighbors.size == 0:
                continue

            unlabeled_distances = covered_distances[~labeled_mask[covered_neighbors]]
            ratios = unlabeled_distances / radius
            improve_mask = ratios < best_ratio[unlabeled_neighbors]
            improved_neighbors = unlabeled_neighbors[improve_mask]
            best_ratio[improved_neighbors] = ratios[improve_mask]
            best_owner[improved_neighbors] = int(owner_position)

        raw_repeat_counts = np.ones(len(next_labeled_indices), dtype=np.int64)
        assigned_owner_positions = best_owner[best_owner >= 0]
        if assigned_owner_positions.size > 0:
            raw_repeat_counts += np.bincount(
                assigned_owner_positions,
                minlength=len(next_labeled_indices)
            ).astype(np.int64, copy=False)

        sqrt_repeat_counts = np.ceil(
            np.sqrt(raw_repeat_counts.astype(np.float64, copy=False))
        )
        sqrt_repeat_counts = np.maximum(sqrt_repeat_counts, 1.0)
        return [int(count) for count in sqrt_repeat_counts.tolist()]

    def _prepare_next_round_training_setup_sparse(
        self,
        radius_graph: SparseRadiusGraph,
        current_labeled_indices: Sequence[int],
        current_labeled_radii: Sequence[float],
        selected_indices: Sequence[int],
        selected_radii: Sequence[float]
    ) -> None:
        """
        Prepare the next-round replicated-training cache using sparse weights.

        The parent method performs the same high-level task through a dense
        distance matrix.  This sparse replacement keeps the round gating, logging,
        and cache format, while computing raw coverage-assignment weights from
        the CSR radius graph.

        Args:
            radius_graph: Maximum-radius CSR graph.
            current_labeled_indices: Labeled indices before current selection.
            current_labeled_radii: Effective radii for current labeled indices.
            selected_indices: Newly selected indices.
            selected_radii: Effective radii for newly selected indices.
        """
        if not self._should_prepare_next_round_training_setup():
            self._next_round_training_setup = None
            print(
                "[SALTExact] 跳过下一轮复制权重计算: "
                f"current_selection_round={self.round_counter + 1}, "
                f"next_training_round={self.round_counter + 2}, "
                f"weighted_start_round={self.weighted_start_round}"
            )
            return

        next_labeled_indices = list(current_labeled_indices) + list(selected_indices)
        repeat_counts = self._compute_repeat_counts_for_next_round_sparse(
            radius_graph=radius_graph,
            current_labeled_indices=current_labeled_indices,
            current_labeled_radii=current_labeled_radii,
            selected_indices=selected_indices,
            selected_radii=selected_radii
        )

        if repeat_counts:
            max_weight_position = int(np.argmax(repeat_counts))
            max_weight_index = int(next_labeled_indices[max_weight_position])
            max_weight_value = int(repeat_counts[max_weight_position])
            print(
                "[SALTExact] 下一轮最大复制权重: "
                f"{max_weight_value} (idx: {max_weight_index})"
            )
        else:
            print("[SALTExact] 下一轮复制权重为空。")

        self._cache_next_round_training_setup(
            next_labeled_indices=next_labeled_indices,
            repeat_counts=repeat_counts
        )

    def _first_round_selection_sparse(
        self,
        radius_graph: SparseRadiusGraph,
        labeled_indices: List[int],
        unlabeled_indices: List[int],
        spectral_norm_product: Optional[float]
    ) -> List[int]:
        """
        Run first-round SALT selection with sparse exact coverage.

        First-round semantics match the sqrt strategy: all candidates use the
        maximum theoretical radius, existing labeled coverage filters the pool,
        then greedy coverage selection runs up to the query budget.  Coverage sets
        are obtained by filtering the CSR `C_max` rows.

        Args:
            radius_graph: Maximum-radius CSR graph.
            labeled_indices: Current labeled sample indices.
            unlabeled_indices: Current unlabeled sample indices.
            spectral_norm_product: Fixed global expansion factor.

        Returns:
            List[int]: Selected full dataset indices.
        """
        print("\n[SALTExact] 第一轮采样")

        round_indices = list(labeled_indices) + list(unlabeled_indices)
        fixed_round_radius = float(self.max_theoretical_radius)
        if round_indices:
            print(
                "[SALTExact] 最大理论半径统计: "
                f"min={fixed_round_radius:.6f}, "
                f"max={fixed_round_radius:.6f}, "
                f"mean={fixed_round_radius:.6f}"
            )

        current_labeled_radii_array = np.full(
            len(labeled_indices),
            fixed_round_radius,
            dtype=np.float64
        )
        if labeled_indices:
            _, covered_mask = self._compute_sparse_coverage(
                radius_graph=radius_graph,
                labeled_indices=labeled_indices,
                radii=current_labeled_radii_array.tolist()
            )
            filtered_unlabeled = self._filter_unlabeled_indices(
                unlabeled_indices=unlabeled_indices,
                covered_mask=covered_mask
            )
            print(f"[SALTExact] 原始无标注池大小: {len(unlabeled_indices)}")
            print(
                "[SALTExact] 被已标注点覆盖的点数: "
                f"{len(unlabeled_indices) - len(filtered_unlabeled)}"
            )
            print(f"[SALTExact] 过滤后无标注池大小: {len(filtered_unlabeled)}")
        else:
            filtered_unlabeled = list(unlabeled_indices)
            print(f"[SALTExact] 无标注池大小: {len(filtered_unlabeled)}")

        selected_indices: List[int] = []
        if filtered_unlabeled:
            radii = np.full(
                len(filtered_unlabeled),
                fixed_round_radius,
                dtype=np.float64
            )
            selected_indices, covered_indices = self._greedy_selection_from_radius_graph(
                unlabeled_indices=filtered_unlabeled,
                radius_graph=radius_graph,
                radii=radii,
                query_budget=self.query_budget,
                progress_desc="[SALTExact] 第一轮选择进度",
                log_prefix="SALTExact"
            )
            print(f"[SALTExact] 第一轮选择的标注点数量: {len(selected_indices)}")
            print(f"[SALTExact] 第一轮覆盖的点数量: {len(covered_indices)}")
        else:
            print("[SALTExact] 无标注池为空，无需选择样本")

        selected_radii = np.full(
            len(selected_indices),
            fixed_round_radius,
            dtype=np.float64
        ).tolist()
        self._prepare_next_round_training_setup_sparse(
            radius_graph=radius_graph,
            current_labeled_indices=labeled_indices,
            current_labeled_radii=current_labeled_radii_array.tolist(),
            selected_indices=selected_indices,
            selected_radii=selected_radii
        )
        _ = spectral_norm_product
        return selected_indices

    def _subsequent_round_selection_sparse(
        self,
        model: torch.nn.Module,
        dataset: Any,
        device: torch.device,
        radius_graph: SparseRadiusGraph,
        labeled_indices: List[int],
        unlabeled_indices: List[int],
        spectral_norm_product: Optional[float]
    ) -> List[int]:
        """
        Run subsequent-round SALT selection with sparse exact coverage.

        This method mirrors `OurMethodMarginWeightedSqrtStrategy` round logic:
        labeled radii are computed from the current model, coverage ratio is
        measured, uncovered candidates receive estimated radii and margin scores,
        and the existing exact incremental greedy selector chooses the batch.
        Every coverage relation is produced from CSR row filtering.

        Args:
            model: Current trained model.
            dataset: Full training dataset.
            device: Current torch device.
            radius_graph: Maximum-radius CSR graph.
            labeled_indices: Current labeled sample indices.
            unlabeled_indices: Current unlabeled sample indices.
            spectral_norm_product: Fixed global expansion factor.

        Returns:
            List[int]: Selected full dataset indices.
        """
        print(f"\n[SALTExact] 第{self.round_counter + 1}轮采样")

        max_theoretical_radius = float(self.max_theoretical_radius)
        labeled_points = self._build_labeled_points(dataset, labeled_indices)
        raw_labeled_radii = self._compute_labeled_radii(
            labeled_points=labeled_points,
            labeled_indices=labeled_indices,
            model=model,
            spectral_norm_product=spectral_norm_product
        )

        coverage_ratio, covered_mask = self._compute_sparse_coverage(
            radius_graph=radius_graph,
            labeled_indices=labeled_indices,
            radii=raw_labeled_radii.tolist()
        )
        self.last_coverage = coverage_ratio
        print(f"[SALTExact] 当前轮次覆盖率: {coverage_ratio:.6f}")

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
                coverage_ratio, covered_mask = self._compute_sparse_coverage(
                    radius_graph=radius_graph,
                    labeled_indices=labeled_indices,
                    radii=effective_labeled_radii.tolist(),
                )
                self.last_coverage = coverage_ratio
                print(
                    "[SALTExact] 早期半径归一化已触发: "
                    f"T={self.early_radius_normalization_threshold:.6f}, "
                    f"k={self.early_radius_normalization_percentile_k:.2f}, "
                    f"q_k={normalization_pivot_radius:.6f}, "
                    f"r={normalization_scale_ratio:.6f}, "
                    f"max_radius={max_theoretical_radius:.6f}, "
                    f"normalized_coverage={coverage_ratio:.6f}"
                )
            else:
                print(
                    "[SALTExact] 早期半径归一化未触发: "
                    f"coverage={coverage_ratio:.6f} >= "
                    f"T={self.early_radius_normalization_threshold:.6f}"
                )

        filtered_unlabeled = self._filter_unlabeled_indices(
            unlabeled_indices=unlabeled_indices,
            covered_mask=covered_mask
        )
        print(f"[SALTExact] 原始无标注池大小: {len(unlabeled_indices)}")
        print(
            "[SALTExact] 被已标注点覆盖的点数: "
            f"{len(unlabeled_indices) - len(filtered_unlabeled)}"
        )
        print(f"[SALTExact] 过滤后无标注池大小: {len(filtered_unlabeled)}")

        if not filtered_unlabeled:
            print("[SALTExact] 无标注池为空，无需选择样本")
            self._prepare_next_round_training_setup_sparse(
                radius_graph=radius_graph,
                current_labeled_indices=labeled_indices,
                current_labeled_radii=effective_labeled_radii.tolist(),
                selected_indices=[],
                selected_radii=[]
            )
            return []

        probabilities = self._get_probabilities_for_indices(
            model,
            dataset,
            filtered_unlabeled,
            device
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

        print(
            "[SALTExact] 预估半径统计: "
            f"min={estimated_radii.min():.6f}, "
            f"max={estimated_radii.max():.6f}, "
            f"mean={estimated_radii.mean():.6f}"
        )

        candidate_indices = np.asarray(filtered_unlabeled, dtype=np.int64)
        candidate_radii = np.asarray(estimated_radii, dtype=np.float64)
        cover_data = self._prepare_candidate_cover_csr_from_radius_graph(
            radius_graph=radius_graph,
            candidate_indices=candidate_indices,
            candidate_radii=candidate_radii
        )
        initial_log_coverage_terms = np.log(
            np.maximum(cover_data.covered_counts, 1)
        )
        static_score_terms = self._compute_static_score_terms(
            probabilities,
            coverage_ratio,
            coverage_terms=initial_log_coverage_terms
        )

        selected_local_positions = self._greedy_select_candidate_positions_from_cover_csr(
            candidate_indices=candidate_indices,
            static_score_terms=static_score_terms,
            cover_data=cover_data,
            query_budget=self.query_budget,
            progress_desc=f"[SALTExact] 第{self.round_counter + 1}轮选择进度",
            log_prefix="SALTExact"
        )
        selected_indices = [
            int(candidate_indices[local_position])
            for local_position in selected_local_positions
        ]
        selected_radii = [
            float(candidate_radii[local_position])
            for local_position in selected_local_positions
        ]

        print(
            "[SALTExact] "
            f"第{self.round_counter + 1}轮选择的标注点数量: {len(selected_indices)}"
        )
        self._prepare_next_round_training_setup_sparse(
            radius_graph=radius_graph,
            current_labeled_indices=labeled_indices,
            current_labeled_radii=effective_labeled_radii.tolist(),
            selected_indices=selected_indices,
            selected_radii=selected_radii
        )
        return selected_indices

    def _load_or_build_local_knn_from_radius_graph(
        self,
        dataset: Any,
        radius_graph: SparseRadiusGraph,
        device: torch.device
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        从最大半径 CSR 图构造局部 Lipschitz 所需的精确全局 KNN 缓存。

        对半径内已有至少 K 个有效邻居的行，直接从 CSR 行选取最近 K 个；因为
        半径外距离严格大于 ``r_max``，这些邻居也是全局精确 KNN。仅对有效邻居
        不足 K 的行执行面向完整特征集的分块搜索。最终统一保存 ``N x K`` 缓存，
        后续主动学习轮次无需再次计算 embedding 距离。

        参数:
            dataset: 完整向量训练集。
            radius_graph: 已构造的最大半径 CSR 图。
            device: 缺失行分块精确搜索使用的设备。

        返回:
            Tuple[np.ndarray, np.ndarray]: 按距离升序排列的 KNN 索引和距离。

        异常:
            RuntimeError: 数据集有效样本数不足以为每行提供 K 个邻居时抛出。
        """
        k = int(self.local_lipschitz_k)
        cached = load_knn_cache(
            self.distance_cache_dir, self.dataset_name, radius_graph.n_samples, k
        )
        if cached is not None:
            cached_indices, cached_distances = cached
            row_ids = np.arange(radius_graph.n_samples, dtype=np.int64)[:, None]
            valid_counts = np.sum(
                (np.asarray(cached_indices) != row_ids)
                & np.isfinite(cached_distances)
                & (np.asarray(cached_distances) > self.local_lipschitz_distance_eps),
                axis=1
            )
            if np.all(valid_counts >= k):
                return cached
            print("[SALTExact] 现有 KNN 缓存含无效零距离邻居，将按局部模式规则重建。")
        features = self._extract_feature_matrix(dataset)
        indices = np.empty((radius_graph.n_samples, k), dtype=np.int32)
        distances = np.empty((radius_graph.n_samples, k), dtype=np.float32)
        deficient: List[int] = []
        for row_index in range(radius_graph.n_samples):
            row_neighbors, row_distances = radius_graph.row(row_index)
            valid = (
                (row_neighbors.astype(np.int64, copy=False) != row_index)
                & np.isfinite(row_distances)
                & (row_distances > self.local_lipschitz_distance_eps)
            )
            valid_neighbors = row_neighbors[valid]
            valid_distances = row_distances[valid]
            if valid_neighbors.size < k:
                deficient.append(row_index)
                continue
            nearest = np.argpartition(valid_distances, k - 1)[:k]
            nearest = nearest[np.argsort(valid_distances[nearest], kind="stable")]
            indices[row_index] = valid_neighbors[nearest]
            distances[row_index] = valid_distances[nearest]
        if deficient:
            print(
                "[SALTExact] CSR 内有效邻居不足 K，分块补齐精确 KNN: "
                f"rows={len(deficient)}/{radius_graph.n_samples}, k={k}"
            )
            missing_indices, missing_distances = compute_exact_knn_for_queries(
                features=features,
                query_indices=deficient,
                k=k,
                device=device,
                min_distance=self.local_lipschitz_distance_eps
            )
            if not np.all(np.isfinite(missing_distances)):
                raise RuntimeError("部分样本不足 K 个有效非重复邻居，无法估计局部 Lipschitz。")
            indices[np.asarray(deficient, dtype=np.int64)] = missing_indices
            distances[np.asarray(deficient, dtype=np.int64)] = missing_distances
        return save_knn_cache(
            self.distance_cache_dir, self.dataset_name, indices, distances
        )

    def _prepare_sparse_local_expansion_context(
        self,
        model: torch.nn.Module,
        dataset: Any,
        radius_graph: SparseRadiusGraph,
        device: torch.device
    ) -> None:
        """
        使用稀疏/补齐后的精确 KNN 为当前模型准备逐点局部放大系数。

        KNN 索引与 embedding 距离跨轮复用；模型 logits 和有限差分比值每轮重新
        计算，以反映当前主动学习轮训练后的模型。结果写入父类约定的两个当前轮
        缓存，使现有半径和候选概率代码无需区分数据来源。

        参数:
            model: 当前轮训练完成后的模型。
            dataset: 完整训练集。
            radius_graph: 最大半径 CSR 图。
            device: 模型前向及缺失 KNN 搜索设备。

        返回:
            None
        """
        knn_indices, knn_distances = self._load_or_build_local_knn_from_radius_graph(
            dataset=dataset, radius_graph=radius_graph, device=device
        )
        all_indices = list(range(radius_graph.n_samples))
        self._current_logits_by_index = predict_logits_for_indices(
            model=model,
            dataset=dataset,
            indices=all_indices,
            device=device,
            batch_size=self.inference_batch_size
        )
        self._current_local_expansion_factors = compute_local_expansion_factors_from_knn(
            logits_by_index=self._current_logits_by_index,
            knn_indices=knn_indices,
            knn_distances=knn_distances,
            quantile=self.local_lipschitz_quantile,
            distance_eps=self.local_lipschitz_distance_eps,
            min_factor=self.local_lipschitz_min_value
        )
        factors = self._current_local_expansion_factors
        print(
            "[SALTExact] 使用KNN局部放大系数: "
            f"k={self.local_lipschitz_k}, q={self.local_lipschitz_quantile:.2f}, "
            f"min={factors.min():.6f}, max={factors.max():.6f}, mean={factors.mean():.6f}"
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
        Select samples with sparse exact SALT coverage.

        This method is the dense-distance-free counterpart of the parent
        `select_samples`.  It loads or builds the CSR radius graph, resolves the
        fixed expansion factor, and dispatches to first/subsequent round sparse
        implementations.  The `batch_size` argument is retained for interface
        compatibility; the active-learning loop controls the effective budget by
        setting `self.query_budget`.

        Args:
            model: Current trained model.
            unlabeled_indices: Full indices currently available for selection.
            dataset: Full training dataset.
            batch_size: Requested batch size, kept for interface compatibility.
            device: Current torch device.
            labeled_indices: Full indices already labeled.
            **kwargs: Reserved for interface compatibility.

        Returns:
            List[int]: Selected full dataset indices.

        Local-Lipschitz mode uses an exact ``N x K`` cache derived from the CSR
        graph and targeted completion, without a dense distance matrix.
        """
        _ = batch_size, kwargs
        if labeled_indices is None:
            all_indices = set(range(len(dataset)))
            labeled_indices = sorted(all_indices - set(unlabeled_indices))

        radius_graph = self._ensure_radius_graph(dataset=dataset, device=device)

        spectral_norm_product = self._resolve_spectral_norm_product(model)
        if spectral_norm_product is None:
            if self.round_counter == 0:
                self._clear_local_expansion_context()
            else:
                self._prepare_sparse_local_expansion_context(
                    model=model,
                    dataset=dataset,
                    radius_graph=radius_graph,
                    device=device
                )
        else:
            self._clear_local_expansion_context()

        if self.round_counter == 0:
            selected_indices = self._first_round_selection_sparse(
                radius_graph=radius_graph,
                labeled_indices=labeled_indices,
                unlabeled_indices=unlabeled_indices,
                spectral_norm_product=spectral_norm_product
            )
        else:
            selected_indices = self._subsequent_round_selection_sparse(
                model=model,
                dataset=dataset,
                device=device,
                radius_graph=radius_graph,
                labeled_indices=labeled_indices,
                unlabeled_indices=unlabeled_indices,
                spectral_norm_product=spectral_norm_product
            )

        self.round_counter += 1
        return selected_indices
