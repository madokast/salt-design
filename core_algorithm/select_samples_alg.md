## strategy 成员变量 / 关键内部变量

query_budget 预算

r_max 理论最大半径，用户给出，构造 radius_graph

epsilon 半径公式用到

h_min 谱范数下界，用于计算最大理论半径

use_soft_labels

diff_min diff 下界，用于防止diff过小导致半径计算不稳定

round_counter 轮数。0-首轮，1...-后续轮次。

radius_graph 全数据集上的最大半径 CSR 图（压缩稀疏行 Compressed Sparse Row），用 r_max 提前过滤掉远距离点。

enable_early_radius_normalization = False

early_radius_normalization_threshold=0.2 覆盖率阈值，低于则进行早期半径归一化

early_radius_normalization_percentile_k = 20 早期半径归一化的基准位置，将此位置的半径拉到最大理论半径附近，其他半径按同样比例放大，但都不能超过上限

## Lipschitz || spectral_norm_product

- 用户传入的固定值 或者 None。固定值即放大系数
- None 时，采用局部 Lipschitz 系数：每个样本基于当前模型 logits 和 KNN 距离估计每个样本自己的局部放大系数。（KNN 即 K邻近）
- 在输入/embedding 距离空间里找最近的 K 个非自身邻居。然后看模型 logits 在这些邻居方向上变化得有多快。
- L_i ≈ quantile_j ( ||logits_i - logits_j||_2 / distance(x_i, x_j) )

K 个非自身邻居
```
对每个样本 i，先看 radius_graph.row(i)。
这行里存的是 r_max 半径内的邻居和距离。
排除自己，排除无效距离和过小距离。
如果里面至少有 K 个邻居，就取最近的 K 个。
如果不足 K，再对完整特征集做精确 KNN 搜索补齐。

neighbors_i = KNN(i)
ratios_i = []
for j in neighbors_i:
    ratio_ij = ||logits(i) - logits(j)||_2 / distance(i, j)
    ratios_i.append(ratio_ij)
L_i = quantile(ratios_i, local_lipschitz_quantile)
L_i = max(L_i, local_lipschitz_min_value)

默认参数来自构造函数：

local_lipschitz_k = 50
local_lipschitz_quantile = 0.90
local_lipschitz_distance_eps = 1e-12
local_lipschitz_min_value = 1e-12
所以默认是：每个点找 50 个近邻，算 50 个“logits变化 / 输入距离”的比值，取 90% 分位数作为这个点的局部放大系数。
```

最终 L_i = spectral_norm_product or 局部 L_i，作为这个样本半径公式里的分母系数。值越大，则半径越小。
- 无标注候选预估半径：radius_i = 2ε / (L_i * (diff_min + sqrt(diff_min² + 2 h_i ε)))
- 已标注点真实半径：radius_i = (-diff_i + sqrt(diff_i² + 2 h_i ε)) / (h_i * L_i)


## 分数

static(x) = alpha_t * coverage_ratio * (-log margin(x))

score(x) = static(x) + log(D(x))


### CSR 最大半径图

max_theoretical_radius = 0.5

对每个样本 x，预存所有满足 distance(x, u) <= r_max 的邻居。后续某轮真实半径 r_x <= r_max 时，只需在这一行里再过滤 distance <= r_x。


## 首轮

round_counter == 0

硬过滤：

1. 所有点半径都用 max_theoretical_radius。
2. 先用已有已标注点覆盖掉一批无标注点，把已经被初始标注点覆盖的无标注样本删掉
3. 对过滤后的无标注池做覆盖贪心。
4. 结束后，把本轮选中的点也视为下一轮已标注点，并准备下一轮训练的复制权重。

### 计算复制权重

1. 采样结束后，用“已有标注点 + 本轮新选点”的半径（首轮都是 max_theoretical_radius。），在全训练集上重新做覆盖归属。
2. 每个标注点基础权重 weight 为 1；其他被覆盖样本归属给 distance / radius 最小的标注点。
3. 最后对归属计数做 ceil(sqrt(weight))，作为下一轮 repeat_counts。

### 覆盖贪心算法

计算所有点的 score(x) = log(D(x))
- D(x) ：候选点 x 在当前仍活跃候选池里，半径 r_max 能覆盖多少个候选点。
- 同分时取当前顺序里最早的
- 选出一个点后，它覆盖的点就从候选池中删除，然后更新所有点的 score(x)

## 后续轮次

round_counter > 0

### 已标注点计算半径

拿到所有已标注点 [{"data": x_i, "label": y_i, "soft_label": soft_y_i}]。这里 soft 是软标签，nullable

每个点的 expansion_factor = spectral_norm_product OR L_i

计算每个点的半径
```
logits = model(x)
p = softmax(logits) # 模型预测概率
eta = one_hot(y) # 真实标签 one-hot
diff = ||p_i - eta_i||_2 # 预测概率和真实标签的L2距离
diff = max(diff, diff_min) # 有下界
h # 模型在这个点附近的曲率 H = diag(p) - p p^T 和 h = max(spectral_norm(H), h_min)，其中 spectral_norm 为 H 的谱范数(矩阵对向量模的最大放大能力)
L = Lipschitz_coefficient(logits) # 局部 logits 放大系数
r = (-diff + sqrt(diff^2 + 2 * h * epsilon)) / (h * L) # 半径公式

# 计算理论最大半径
sqrt_term = np.sqrt(diff_min**2 + 2.0 * h_min * epsilon)
r_max = 2.0 * epsilon / (L * (diff_min + sqrt_term))
r = min(r, r_max) # 限制到理论最大半径
```

### 已标注点计算覆盖率

就是 embedding 距离空间里，哪些点在已标注点半径范围内。

对每个已标注点 index，从 radius_graph.row(index) 取出 r_max 内邻居，再过滤 distances <= radius。

所有被过滤出的邻居合并成 covered_mask。

coverage_ratio = covered_mask 中 True 的数量 / 全训练集数量。

### 早期半径归一化

这个功能需要开启才生效 enable_early_radius_normalization

如果当前覆盖率低于阈值 T，就把已标注点半径整体放大，但最大不超过 r_max

把已标注点半径从大到小排序，取“前 k% 位置”的半径作为 q_k

scale_ratio = max_theoretical_radius / q_k
normalized_radius_i = min(raw_radius_i * scale_ratio, max_theoretical_radius)

把第 k% 大的半径拉到最大理论半径附近，其他半径按同样比例放大，但都不能超过上限

### 候选池 soft 保留

把全部 unlabeled_indices 保留下来，同时区分“初始已覆盖”和“初始未覆盖”。

只有未覆盖候选会估计半径

```
初始已覆盖候选:
  保留在 candidate_indices 中
  会计算概率和 margin
  candidate_radii = 0
初始未覆盖候选:
  保留在 candidate_indices 中
  会计算概率和 margin
  会估计 candidate_radii
```

#### 计算预测概率

概率矩阵后面分别用于 margin 静态项和未覆盖候选半径估计

概率是模型 logits 经过 softmax 得到的 p 分布，且顺序和 unlabeled_indices 对齐。就是各个标签的预测概率。

例如 p(x) = [0.70, 0.20, 0.10]

#### margin 静态分数

margin = 最大类别概率 - 第二大类别概率

例如 p(x) = [0.70, 0.20, 0.10]，则 margin = 0.70 - 0.20 = 0.50

margin 越小，模型越不确定

#### 初始未覆盖候选 估计 candidate_radii

已覆盖候选的半径就是 0

```
模型概率 p(x)
计算 h(x) 也就是谱范数 - H = diag(p) - p p^T 和 max(spectral_norm(H), h_min)
取放大系数 L(x) - 和之前一样，每个样本基于当前模型 logits 和 KNN 距离估计每个样本自己的 Lipschitz 局部放大系数。
用公式估计半径 r(x) = 2ε / ( L(x) * (diff_min + sqrt(diff_min^2 + 2 h(x) ε)) )
截断到最大理论半径以内
如果早期半径归一化触发，再乘 scale_ratio 后再次截断

区别在于没有 diff 只有 diff_min
```

### 动态覆盖 CSR 与 soft 贪心

soft 贪心如何用双堆选择、更新 D(x)

先算分数 score(x) = static(x) + log(D(x))

- D(x) 是候选点 x 在当前仍未覆盖/仍活跃的候选池里，用自己的 candidate_radii 能覆盖多少个候选点。已覆盖候选 D(x) = 1。

- static(x) 是候选点 x 的静态分数。


如果选出一个仍未覆盖候选，它覆盖到的初始未覆盖目标会被冻结为已覆盖；相关候选的 D(x) 会通过反向 CSR 减少并重新入堆。

#### static 静态分数

static(x) = alpha_t * coverage_ratio * uncertainty

uncertainty(x) = -log(margin(x))

margin(x) = top1_probability - top2_probability

coverage_ratio 当前已标注集覆盖全训练集的比例。它在本轮内固定

alpha 默认 0.1，如果 none 则估计当前轮的 alpha_t

alpha_t = IQR(logD) / IQR(uncertainty)
- IQR 是四分位距，即 75% 分位数 - 25% 分位数
- logD = log(max(D, 1))，使用进入贪心前的初始 D，不是贪心过程中动态更新后的 D
- uncertainty 当前候选池初始状态下，每个候选点的静态不确定性项

### 采样结束后的复制权重

入参

selected_indices     本轮选中的全局样本索引
selected_radii       每个新选点的生效半径

如果选中的是仍未覆盖候选：selected_radius = 它的预测半径
如果选中的是已覆盖候选：selected_radius = 0

next_labeled_indices = 当前已标注点 + 本轮新选点

next_labeled_radii = 当前已标注点有效半径 + 本轮新选点生效半径

每个 next_labeled 点基础 raw_weight = 1。

对每个半径 > 0 的 next_labeled 点，找它覆盖的“非 next_labeled”样本。

如果某个非标注样本被多个点覆盖，则归属给 distance / radius 最小的点。

被归属一个样本，该点 raw_weight += 1。

最后 repeat_count = ceil(sqrt(raw_weight))。


