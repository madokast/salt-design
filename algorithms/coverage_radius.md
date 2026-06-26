# 覆盖半径与选样计算

论文第 4+5 章。**纯计算，不操作数据库。** 上层采集数据传入，算法返回结果，上层负责入库和生成任务。

---

## 调用方职责

```
上层（编排器）:
  1. 从数据库采集: 已标注对象 + embedding、当前模型、配置参数
  2. 调用算法: coverage_result = compute(model, labeled, unlabeled, config)
  3. 入库: coverage_result 中的指标写入 project_rounds
  4. 如果 is_complete=false: 用 selected_ids 生成下一轮 HUMAN task
  5. 如果启用机器作答: 用 predictions 写入 MACHINE task
```

算法本身是无状态的纯函数。

---

## 直觉

```
嵌入空间中的标注点：

  平坦区域，模型很确定             陡峭区域，模型摇摆
  ┌─────────────────┐           ┌─────────┐
  │    ╭──────╮     │           │  ╭──╮   │
  │   ╱ 半径大  ╲   │           │ ╱半径小╲ │
  │  │    ◉    │   │           │ │  ◉  │ │
  │   ╲        ╱   │           │ ╲    ╱  │
  │    ╰──────╯     │           │  ╰──╯   │
  └─────────────────┘           └─────────┘
```

- 平坦 + 模型准确 → 半径大，一个点能 cover 一大片
- 陡峭 + 模型不准 → 半径小，需要更多点才能 cover 同样区域

---

## 输入

- \( \theta \): 当前模型参数
- \( x_i \): 已标注样本（有其 embedding）
- \( \eta(x_i) \): 该点的条件标签分布。实践中用人工标注的 one-hot 近似
- \( p_\theta(x_i) \): 模型的 softmax 输出
- \( \zeta \): 精度容忍度
- \( C \): 类别数

---

## 计算步骤

### 1. 计算局部损失预算 \( \epsilon \)

\[
\epsilon = \frac{1}{2} \zeta \ln C
\]

含义：把全局精度容忍度 \( \zeta \) 平分到每个局部区域。

### 2. 计算模型谱范数乘积 \( \Gamma_\theta \)

\[
\Gamma_\theta = \prod_{l=1}^{L} \|W_l\|_2
\]

每层权重矩阵的谱范数（最大奇异值）连乘。衡量模型的"陡峭程度"——\( \Gamma_\theta \) 越大，输出对输入变化越敏感。

实现：对每层权重矩阵做 SVD，取最大奇异值，全部相乘。

### 3. 计算局部 Lipschitz 常数 \( \lambda \)

\[
\lambda = \max_{i \neq j} \frac{|\eta(x_i) - \eta(x_j)|}{\|x_i - x_j\|_2}
\]

从已标注样本中估计。\( \eta(x_i) \) 用 one-hot 标签近似。衡量标签分布的局部变化速度。

### 4. 计算预测偏差和曲率

- 预测偏差：\( \|\eta(x_i) - p_\theta(x_i)\|_2 \)——one-hot 与 softmax 输出的欧氏距离。模型越准，值越小
- 局部曲率 \( h \)：Hessian 矩阵的谱范数。\( h \in [0, 0.5] \)，交叉熵 softmax 的性质。实现时可取上界 0.5 近似
- logit 范数：\( \|z_\theta(x_i)\|_2 \)，模型输出 logits 的 L2 范数。可统一约束上界 \( B \)

### 5. 计算系数 a 和 b

\[
a = \frac{1}{2} h \Gamma_\theta^2 + \lambda \Gamma_\theta
\]
\[
b = \|\eta - p\|_2 \cdot \Gamma_\theta + \lambda \|z\|_2
\]

- a 体现二阶效应（曲率 × 模型陡峭度）
- b 体现一阶效应（预测不准 × 模型陡峭度 + 标签变化 × logit 幅度）

### 6. 计算覆盖半径 \( \delta_x \)

\[
\delta_x(\epsilon, \theta) = \frac{2\epsilon}{b + \sqrt{b^2 + 4a\epsilon}}
\]

反解 \( a\delta^2 + b\delta \leq \epsilon \) 得到。

---

## 输出

计算完成后返回一个结果对象，包含三部分：

```
CoverageResult:

  ┌─ 1. 是否继续
  │    is_complete: bool
  │
  ├─ 2. 入库展示（写入 project_rounds）
  │    coverage_ratio: float
  │    risk_gap: float
  │    model_accuracy: float
  │
  ├─ 3. 可视化数据（前端渲染覆盖图）
  │    vis_points: [{anno_id, x, y, covered: bool, is_center: bool, delta: float|null}, ...]
  │    └─ 前端拿到后：加载 2D 坐标（数据库已有），叠加 covered 着色 + 覆盖圆
  │
  └─ 4. 下一轮选样（仅 is_complete=false 时有值）
       selected_ids: [str, ...]
       predictions: [{anno_id, label, confidence}, ...]  # 若启用机器作答
```

中间结果（谱范数 \( \Gamma_\theta \)、Lipschitz \( \lambda \)、各点 a/b 值等）计算完即丢弃，不落盘。仅 `CoverageResult` 中的字段对外可见。

---

## 实践简化

论文给出的公式中有几项在实践中可简化：

| 项 | 理论要求 | 实践近似 |
|----|---------|---------|
| \( \eta(x) \) | 真实条件概率分布 | 已标注样本的 one-hot 标签 |
| \( h \) | Hessian 谱范数 | 取上界 0.5 |
| \( \lambda \) | 标签分布 Lipschitz 常数 | 从已标注样本估算 \( \max \Delta label / \Delta dist \) |
| \( \|z\|_2 \) | logit 范数上界 | 所有已标注样本 logit 范数的 95 分位数或 max |

这些简化使算法在实际数据上可计算，论文实验也验证了有效性。
