# 算法、插件接口与模型管理

---

## 1. 用户视角 vs 内部实现

用户看到的是一个**算法名称**，选中即可。内部是多个组件的装配。

```
用户看到:                    内部实现:

  "SALT 标准版"  ──────→  ┌─ 编码器: Qwen v4
                          ├─ 选样: SALT 内置
                          ├─ 训练: MLP Trainer
                          └─ 弱监督: 无

  "SALT + 弱监督"  ────→  ┌─ 编码器: Qwen v4
                          ├─ 选样: SALT 内置
                          ├─ 训练: MLP Trainer
                          └─ 弱监督: LRR 规则引擎
```

---

## 2. 内部组件

| 组件 | 接口 | 职责 |
|------|------|------|
| 编码器 | `Encoder` | 文本/图像 → embedding 向量 |
| 选样策略 | `Selector` | 从无标注对象中选出下一轮给专家标 |
| 模型训练 | `Trainer` | 用已标注数据训练分类器 |
| 弱监督（可选）| `WeakSupervisor` | 用 LRR 规则生成/修正弱标签 |

---

## 3. 组件接口

### 3.1 Selector（选样策略）

```python
class Selector:
    def select_initial(self, config: ProjectConfig,
                       objects: List[ObjectEmbedding]) -> List[str]:
        """项目启动时选初始批，返回 anno_id 列表"""
        pass

    def select_round(self, round_number: int,
                     labeled: List[LabeledObject],
                     unlabeled: List[ObjectEmbedding],
                     model: ModelHandle) -> SelectResult:
        """每轮结束后选下一批"""
        pass

@dataclass
class SelectResult:
    selected_ids: List[str]
    coverage_ratio: float
    risk_gap: float
    is_complete: bool
```

### 3.2 Trainer（模型训练）

```python
class Trainer:
    def train(self, labeled: List[LabeledObject],
              labels: List[str],
              prev_model: ModelHandle | None) -> ModelHandle:
        pass

    def predict(self, model: ModelHandle,
                objects: List[ObjectEmbedding]) -> List[Prediction]:
        pass
```

### 3.3 Encoder（编码器）

```python
class Encoder:
    def encode(self, texts: List[str]) -> List[List[float]]:
        pass

    @property
    def dimension(self) -> int:
        pass
```

### 3.4 WeakSupervisor（弱监督，可选）

```python
class WeakSupervisor:
    def generate(self, objects: List[ObjectEmbedding],
                 lfs: List[Callable]) -> List[WeakLabel]:
        pass

    def correct(self, weak_labels: List[WeakLabel],
                human_labels: List[LabeledObject],
                rules: List[LRR]) -> List[WeakLabel]:
        pass
```

---

## 4. 数据结构

```python
@dataclass
class ProjectConfig:
    labels: List[str]
    zeta: float
    batch_size: int
    alpha: float
    initial_labeled: int

@dataclass
class ObjectEmbedding:
    anno_id: str
    embedding: List[float]

@dataclass
class LabeledObject:
    anno_id: str
    embedding: List[float]
    label: str                     # final_label

@dataclass
class Prediction:
    anno_id: str
    label: str
    confidence: float              # 0~1
```

---

## 5. 编码器管理

```
encoders:
  name: string              # "qwen-embedding-v4"
  display_name: string      # "Qwen Text Embedding v4"
  type: enum                # TEXT | IMAGE
  dimension: int            # 768
  builtin: boolean
```

编码器选定后，所有标记对象的 embedding 只计算一次入库。后续轮次直接读，不重算。

---

## 6. 模型管理

```
models:
  id: UUID
  project_id: UUID
  round: int                    # 产生于第几轮
  file_path: str                # 模型文件路径
  metrics: json                 # {"accuracy": 0.85, ...}
  status: enum                  # TRAINING | READY | ARCHIVED
  created_at: timestamp
```

每轮训练产出一个新模型版本。最新 READY 的用于机器作答。历史保留支持回溯。

---

## 7. 算法注册

### 7.1 算法定义

```json
// algorithms/salt_standard.json
{
  "name": "salt_standard",
  "display_name": "SALT 标准版",
  "description": "基于曲率自适应覆盖的选样 + MLP 分类器",
  "builtin": true,
  "components": {
    "encoder":    "qwen-embedding-v4",
    "selector":   "salt_builtin",
    "trainer":    "mlp_trainer",
    "weak_supervisor": null
  }
}
```

一个算法 = 一组组件装配。内置若干，后续管理员可注册自定义算法。

### 7.2 项目关联

```
Project:
  algorithm_name: str       # "salt_standard"
```

创建项目时从下拉列表选即可，内部装配对用户透明。

---

## 8. 调用时序

```
项目启动
  → encoder.encode(所有对象) → embedding 入库
  → selector.select_initial(config, all_embeddings)
  → selected_ids → 生成 HUMAN task

每轮结束：
  → trainer.train(labeled, labels, prev_model) → 新模型入库
  → 若启用机器作答：trainer.predict(model, unlabeled) → MACHINE task
  → selector.select_round(round, labeled, unlabeled, model)
  → SelectResult
      ├─ is_complete → 项目完成
      └─ selected_ids → 生成下一轮 task
```
