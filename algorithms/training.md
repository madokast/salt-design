# 模型训练

---

## 定位

用已标注对象训练一个分类器，为后续覆盖分析和机器作答提供模型。论文使用四层 MLP。

---

## 输入

- `X`: 已标注对象的 embedding 矩阵（从 `anno_content_{pid}.embedding` 读取）
- `y`: 对应 `final_label`（已裁定/回写的最终标签）
- `labels`: 标签集（如 `["正面", "负面", "中性"]`，决定 softmax 输出维度）
- `prev_model_id` | `null`: 上轮模型 ID（热启动时传入，冷启动为 null）

## 输出

- 模型权重文件（保存到磁盘）
- 数据库模型记录（id、文件路径、accuracy、loss、轮次）

---

## 模型结构

```
Embedding (768) → FC(256) → ReLU → FC(128) → ReLU → FC(64) → ReLU → FC(C) → Softmax
```

论文中的四层 MLP。C 为标签数。输入 embedding 是预计算的，不参与梯度更新。

---

## 训练参数

- 损失函数：交叉熵
- 优化器：Adam
- 学习率：可配置默认值，如 0.001
- epochs：可配置，如 50
- batch size：可配置，如 64

---

## 冷启动 vs 热启动

| 模式 | 初始化 | 适用 |
|------|--------|------|
| 冷启动 | 随机初始化 | 论文默认，每轮独立训练 |
| 热启动 | 加载上轮权重，继续训练 | 可选，加速收敛；克隆项目复用模型时 |

论文使用冷启动。平台默认冷启动，管理员可选热启动。复用模型时自动热启动。

---

## 模型存储

磁盘结构：

```
models/
  {project_id}/
    round_1.pt
    round_2.pt
    round_3.pt
```

数据库记录：

```
models:
  id: UUID
  project_id: UUID
  round: int
  file_path: str              # models/{pid}/round_2.pt
  metrics: json               # {"accuracy": 0.85, "loss": 0.42}
  status: enum                # TRAINING | READY | ARCHIVED
  created_at: timestamp
```

每轮训练产出新记录，最新 READY 即当前模型。

---

## 推理

```
输入: model_id + 待预测对象的 embedding 列表
处理: 加载 model_id 对应权重文件，forward pass
输出: [{anno_id, label, confidence}, ...]
```

- label: argmax(softmax)
- confidence: max(softmax)，用于置信度驱动人力分配和标注界面推荐
