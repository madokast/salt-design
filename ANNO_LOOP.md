# 标注协作流程

三表分离：原始数据（只读）、任务表（抢夺，含机器专家）、结果回写。机器推荐 = 任务表中的一行 MACHINE 任务，与人类专家地位对等。

---

## 1. 表结构

### anno_content_{项目id} — 标记对象（只读）

```
anno_content_{project_id}:
  anno_id         UUID PK
  ... 由 anno_type.fields 动态生成 ...
  final_label     string | null        # 最终标签（任务全部完成后写入）
  label_status    int  DEFAULT 0       # 0=待标注  1=标注中  2=已完成
  labeled_round   int | null           # 在第几轮被标注完成
  labeled_at      timestamp | null     # 标注完成时间
```

导入后只读，只有 `final_label`、`label_status`、`labeled_round`、`labeled_at` 在标注过程中会变化。

### anno_task_{项目id} — 任务表（争夺）

```
anno_task_{project_id}:
  id            SERIAL PK
  anno_id       UUID NOT NULL        # FK → anno_content.anno_id
  round         int NOT NULL          # 产生于第几轮
  label         string | null        # 标注值
  confidence    float | null         # 机器置信度（仅 MACHINE 行有值，0~1）
  source        enum DEFAULT 'HUMAN' # HUMAN | MACHINE
  status        int  DEFAULT 0       # 0=待抢  1=抢占中  2=已提交
  expert_id     UUID | null          # HUMAN 时为专家 user_id，MACHINE 时为 NULL
  locked_at     timestamp | null     # 抢占时间（MACHINE 为 NULL）
  submitted_at  timestamp | null     # 提交时间
  version       int  DEFAULT 0       # 乐观锁
```

- HUMAN 行：专家抢夺、提交、乐观锁
- MACHINE 行：模型预测结果直接写入，`status = 2`，不经抢夺，`confidence` 记录预测概率

- 每个 `anno_content` 行在任务表中可以有 N 行 HUMAN（N 由配置或机器置信度动态决定）
- 所有抢夺、提交、乐观锁冲突都在任务表上发生
- 标记对象表不受争夺影响

---

## 2. 映射关系

```
anno_content_{pid}              anno_task_{pid}
┌──────────────────┐           ┌──────────────────────────────┐
│ anno_id: a1      │←─────────│ anno_id: a1, source: MACHINE  │
│ raw_text: "..."  │  1:N     │ label: "正面", status: 2      │
│ final_label: null│           ├──────────────────────────────┤
└──────────────────┘           │ anno_id: a1, source: HUMAN    │
                               │ label: "负面", expert: 张三   │
                               │ status: 2                    │
                               └──────────────────────────────┘
```

一个标记对象 → N 个任务，人类和机器各占若干行。全部完成后汇总回写。

---

## 3. 项目启动时初始化

```
导入数据 → anno_content_{pid} 写入所有行

项目启动
  → 读取 annotators_per_sample (默认 1)
  → 读取 model_suggestion (默认 false)
  → 对每个 anno_content 行：
      - model_suggestion = false：创建 annotators_per_sample 行 HUMAN
      - model_suggestion = true ：创建 1 行 MACHINE，HUMAN 行数在机器作答后按置信度决定
```

---

## 4. 抢单（在任务表上争夺）

### 4.1 问题场景

```
anno_task_{pid}:
  id=0  anno_id=X  status=0  ← 待抢
  id=1  anno_id=X  status=0  ← 待抢，但和 id=0 指向同一个标记对象
```

如果专家 A 先抢到 id=0，又抢到 id=1，相当于同一个标记对象被同一人标了两次——无效重复。

### 4.2 SQL 逻辑

```
专家点击"开始标注"
  → BEGIN

  → SELECT * FROM anno_task_{pid}
    WHERE status = 0
      AND anno_id NOT IN (                           -- 排除已接触过的标记对象
        SELECT anno_id FROM anno_task_{pid}
        WHERE expert_id = {user_id}
          AND status IN (1, 2)                       -- 抢占中 或 已提交
      )
    ORDER BY random()
    LIMIT 1
    FOR UPDATE SKIP LOCKED

  → 没拿到 → ROLLBACK，显示 "无任务可抢"

  → UPDATE anno_task_{pid}
    SET status = 1, expert_id = {user_id},
        locked_at = NOW(), version = version + 1
    WHERE id = {id} AND version = {旧version}

  → affected_rows = 0 → ROLLBACK，重试
  → affected_rows = 1 → COMMIT
  → 根据 anno_id 查 anno_content → 渲染 HTML 返回
```

### 4.3 场景推导

```
初始：id=0(anno_id=X, status=0), id=1(anno_id=X, status=0)

专家 A 抢单：
  → SELECT 命中 id=0（status=0 且 anno_id=X 尚未被 A 占）
  → UPDATE id=0 → status=1, expert_id=A

专家 A 再次抢单：
  → SELECT 扫描
  → id=1：status=0 ✓
  → 但子查询返回 anno_id=X（因为 A 持有 id=0，status=1）
  → WHERE anno_id NOT IN (X) → 排除 id=1
  → 没拿到 → "无任务可抢"

专家 B 抢单：
  → SELECT 扫描
  → id=1：status=0 ✓
  → 子查询查 B 的记录 → 空，不排除 X
  → 命中 id=1 → 抢到
```

结论：id=0 和 id=1 必然由不同专家标注，同一人不会复答同一对象。

---

## 5. 提交（在任务表上写结果）

```
专家选标签，点"提交"
  → BEGIN

  → UPDATE anno_task_{pid}
    SET status = 2, label = {选中的标签},
        submitted_at = NOW(), version = version + 1
    WHERE id = {id}
      AND status = 1
      AND expert_id = {user_id}
      AND version = {旧version}

  → affected_rows = 0 → ROLLBACK，提示 "提交失败，请刷新"
  → affected_rows = 1 → COMMIT
```

---

## 6. 机器专家

将机器视为一个特殊专家，与人类专家地位对等。机器先行作答，人类后行标注。

### 6.1 每轮执行顺序

```
Round N 开始
  │
  ├─ 1. 机器作答
  │     SALT 对本轮样本进行预测
  │     → 写入 MACHINE 行（status=2, confidence=模型概率）
  │
  ├─ 2. 按置信度分配人力
  │     dynamic_annotators ? 根据 confidence 创建 1 或 2 行 HUMAN
  │     : 创建 annotators_per_sample 行 HUMAN
  │
  ├─ 3. 人类标注
  │     专家抢 HUMAN 类型 task 行
  │     若 show_suggestion = true → 标注界面显示机器推荐
  │     若 show_suggestion = false → 不显示
  │
  └─ 4. 全部 HUMAN 提交后 → 裁决
        所有标注一致 → 自动回写
        有不一致 → 进入管理员裁定
```

第一轮（模型尚未训练）无机器作答，仅人类标注。

### 6.2 启用与配置

管理员可选择**无机器作答**（`model_suggestion = false`），此时不创建 MACHINE 行，纯人类标注。

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `model_suggestion` | false | 是否启用机器作答 |
| `show_suggestion` | true | 专家标注时是否看到机器推荐 |
| `dynamic_annotators` | false | 根据机器置信度动态决定人类专家人数 |
| `confidence_threshold` | 0.3 | 低置信分界线 |
| `quality_sample_rate` | 0 | 随机双人标注比例 0~1，0=不启用 |
| `conflict_policy` | ADJUDICATE | 不一致时的处理 |

### 6.3 置信度驱动的动态人力分配

机器对每个样本给出预测和最大类别概率 `max_prob`。**不确定性** = `1 - max_prob`（值越大越不确定）。

```
机器作答完成 → 对每条 MACHINE task：

  uncertainty = 1 - max_prob
  或二分类：uncertainty = 1 - 2*|prob - 0.5|

  ┌─ uncertainty < confidence_threshold（预测很确定）
  │    → 创建 1 行 HUMAN task
  │
  └─ uncertainty ≥ confidence_threshold（预测不确定）
       → 创建 2 行 HUMAN task
```

| 场景 | max_prob | uncertainty | HUMAN 行数 |
|------|----------|-------------|-----------|
| 机器非常确定 | 0.95 | 0.05 | 1 |
| 机器比较确定 | 0.82 | 0.18 | 1 |
| 机器摇摆 | 0.58 | 0.42 | 2 |
| 机器完全随机 | 0.51 | 0.49 | 2 |

直觉：机器很确信时，一个人复核就够了；机器拿不准时，需要两个人交叉验证。

### 6.4 质量抽检（随机双人标注）

不依赖机器置信度，纯粹为了**监控专家标注一致性**。随机抽取 p% 的样本生成 2 个 HUMAN task，其余 1 个。

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `quality_sample_rate` | 0 | 随机双人标注比例，0~1。0 表示不启用 |

```
每轮生成 HUMAN task 时：

  对每个 anno_id：
    ├─ random() < quality_sample_rate
    │     → 创建 2 行 HUMAN task
    └─ 否则
          → 创建 1 行 HUMAN task
```

#### 与动态分配的叠加

两个开关独立，取 max：

```
effective_count = max(
  dynamic_annotators 决定的数目,    # 1 or 2
  quality_sample_rate 随机出的数目   # 1 or 2
)
```

| dynamic | quality 抽中 | 最终 HUMAN 行数 |
|---------|------------|----------------|
| 1 | 否 | 1 |
| 1 | 是 | 2 |
| 2 | 否 | 2 |
| 2 | 是 | 2 |

#### 用途

管理员在项目统计中查看抽检样本的专家间一致率，作为专家质量评估参考。不阻塞主流程。

### 6.5 标注界面

```
┌─────────────────────────────────────────────────────────┐
│  项目: 评论情感分类  │  池: 待抢12 / 我持3 / 已完成28     │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────────────────────────────────────────┐    │
│  │  (anno_type.views.annotate(row) 渲染的 HTML)     │    │
│  └─────────────────────────────────────────────────┘    │
│                                                         │
│  💡 机器推荐: 正面                         ← 可选，受    │
│                                           show_suggestion│
│  你的标注: ○ 正面  ○ 负面  ○ 中性  ○ 垃圾      控制      │
│                                                         │
│  [跳过]  [提交并抢下一条]                                │
└─────────────────────────────────────────────────────────┘
```

### 6.6 裁定范围

管理员裁定时看到该标记对象的**全部作答**，包括机器和所有人类专家：

```
┌──────────────────────────────────────────────────────┐
│  (渲染的 HTML)                                       │
│                                                      │
│  机器: 正面    专家A (张三): 负面    专家B (李四): 中立 │
│                                                      │
│  管理员裁定: ○ 正面  ○ 负面  ○ 中立                   │
│                                                      │
│  [确认裁定]                                           │
└──────────────────────────────────────────────────────┘
```

不一致来源：

| 类型 | 示例 | 处理 |
|------|------|------|
| 人-人 | 张三标正面，李四标负面 | 进入裁定 |
| 人-机 | 专家标负面，机器推正面 | 若 TRUST_HUMAN → 信专家，跳过裁定；若 ADJUDICATE → 进入裁定 |
| 全一致 | 全部标正面 | 自动回写，不进入裁定 |

---

## 7. 任务完成后回写

每个 `anno_id` 的全部 HUMAN 任务提交后，检查汇总：

```
SELECT COUNT(*) FILTER (WHERE status = 2) AS done,
       COUNT(DISTINCT label) FILTER (WHERE status = 2) AS distinct_labels
FROM anno_task_{pid}
WHERE anno_id = {anno_id}
```

### 7.1 全部一致 → 自动回写

```
UPDATE anno_content_{pid}
SET final_label = {一致标签}, label_status = 2,
    labeled_round = {当前轮次}, labeled_at = NOW()
WHERE anno_id = {anno_id}
```

### 7.2 不一致 → 按策略

```
├─ 仅涉及 MACHINE vs HUMAN 且 conflict_policy = TRUST_HUMAN
│     → 以专家 label 回写
└─ 其余（人-人不一致，或人-机 + ADJUDICATE）
      → 进入管理员裁定列表
```

---

## 8. 管理员裁定

```
管理员看到待裁定列表 → 点进某条

┌──────────────────────────────────────────────────────┐
│  (anno_type.views.annotate(row) 渲染的 HTML)          │
│                                                      │
│  专家A (张三): ○ 正面    专家B (李四): ○ 负面          │
│                                                      │
│  管理员裁定: ○ 正面  ○ 负面  ○ 中性                   │
│                                                      │
│  [确认裁定]                                           │
└──────────────────────────────────────────────────────┘

确认 →
  UPDATE anno_content_{pid}
  SET final_label = {裁定标签}, label_status = 2,
      labeled_round = {当前轮次}, labeled_at = NOW()
  WHERE anno_id = {anno_id}
```

裁定完成后，任务表不动（保留历史记录），标记对象表写入最终结果。

---

## 9. 放弃

```
专家点"放弃"
  → UPDATE anno_task_{pid}
    SET status = 0, expert_id = NULL, locked_at = NULL,
        version = version + 1
    WHERE id = {id}
      AND expert_id = {user_id}
      AND version = {旧version}

  → 任务回到池中
```

---

## 10. 池状态查询

```sql
SELECT
  COUNT(*) FILTER (WHERE t.status = 0
    AND t.anno_id NOT IN (
      SELECT anno_id FROM anno_task_{pid}
      WHERE expert_id = {me} AND status IN (1, 2)
    ))                                                  AS pending,
  COUNT(*) FILTER (WHERE t.status = 1 AND t.expert_id = {me}) AS held_by_me,
  COUNT(*) FILTER (WHERE c.label_status = 2)            AS completed,
  COUNT(*) FILTER (WHERE c.label_status = 1
    AND NOT EXISTS (
      SELECT 1 FROM anno_task_{pid} t2
      WHERE t2.anno_id = c.anno_id AND t2.status = 0
    ))                                                  AS needs_adjudication
FROM anno_content_{pid} c
LEFT JOIN anno_task_{pid} t ON t.anno_id = c.anno_id;
```

---

## 11. SALT 轮次与回写时机

```
Round 1 (无模型):
  人类标注 → 回写 → 训练

Round 2+ (有模型):
  机器作答（SALT 预测 → MACHINE task 写入）
    → 人类标注（看到机器推荐 或 不看）
    → 不一致裁定
    → 回写
    → 训练，计算覆盖率
    → 达标 COMPLETED / 未达标下一轮
```

---

## 12. 轮次记录与指标

每轮结束后落盘一条记录，供领导查看进度和效率。

### 12.1 轮次记录表

```
project_rounds:
  id              SERIAL PK
  project_id      UUID
  round_number    int
  status          enum                 # RUNNING | COMPLETED

  -- 覆盖与精度
  coverage_ratio  float                # 本轮覆盖率
  risk_gap        float                # 风险差距 |R_D(θ_S) - R_D(θ_D)|
  model_accuracy  float                # 当前模型估计精度

  -- 样本
  objects_selected  int                # 本轮选出的标记对象数
  objects_labeled   int                # 本轮完成标注的对象数（回写后）
  objects_auto      int                # 自动标注的对象数

  -- 耗时
  machine_train_ms  int                # 模型训练耗时（毫秒）
  machine_infer_ms  int                # 模型推理耗时
  human_total_ms    int                # 专家标注总耗时（sum of sub-locked）
  human_avg_ms      int                # 专家平均单次标注耗时

  -- 参与
  experts_assigned  int                # 本项目被分配的专家总数
  experts_active    int                # 本轮至少提交 1 次的专家数
  tasks_per_expert  json               # {"张三": 12, "李四": 8}  各专家本轮任务数

  -- 一致性（如有双人标注）
  agreement_human_human  float         # 专家间一致率
  agreement_human_machine float        # 人机一致率

  started_at        timestamp
  completed_at      timestamp

project_round_vis:                    # 可视化点数据（每轮一份）
  project_id      UUID
  round           int
  anno_id         UUID
  x               float               # 2D 坐标（归一化到 0~1）
  y               float
  covered         boolean             # 是否被覆盖
  delta           float | null        # 覆盖圆半径（仅标注中心点有值）
```

`project_round_vis` 每轮全量写入（约 40KB/万条），前端轮次滑块切换时加载对应轮次。不存历史可删除旧轮。

### 12.2 指标汇算

每轮结束时，从现有表聚合：

```
一、覆盖与精度
  → SALT 算法直接产出 coverage_ratio, risk_gap, model_accuracy

二、样本统计
  → objects_selected  = 本轮生成 HUMAN task 的 distinct anno_id 数
  → objects_labeled   = 本轮 label_status 从 0 变为 2 的 anno_id 数
  → objects_auto      = 本轮由模型直接标注的 anno_id 数

三、耗时
  → machine_train_ms  = SALT 训练计时
  → machine_infer_ms  = 模型对 MACHINE task 批量推理计时
  → human_total_ms    = SUM(submitted_at - locked_at) 本轮所有 HUMAN task
  → human_avg_ms      = human_total_ms / 本轮完成 HUMAN task 数

四、参与
  → experts_assigned  = project_members 中 ANNOTATOR 角色总数
  → experts_active    = 本轮至少提交 1 次的不同 expert_id 数
  → tasks_per_expert  = GROUP BY expert_id 聚合

五、一致性
  → agreement_human_human   = 双人标注一致数 / 双人标注总数
  → agreement_human_machine = 人机一致数 / 人机标注总数
```

### 12.3 领导仪表盘

```
┌──────────────────────────────────────────────────────────────┐
│  项目: 评论情感分类                                           │
│                                                              │
│  Round │ 覆盖率 │ 精度  │ 人耗时 │ 机耗时 │ 活跃/分配 │ 人机一致 │
│  ──────┼────────┼───────┼────────┼────────┼───────────┼────────│
│    1   │  12%   │ 0.45  │  8min  │  2min  │   3/5     │   —    │
│    2   │  34%   │ 0.61  │ 12min  │  3min  │   4/5     │  82%   │
│    3   │  58%   │ 0.74  │ 10min  │  3min  │   5/5     │  89%   │
│    4   │  81%   │ 0.85  │  6min  │  2min  │   4/5     │  93%   │
│    5   │  92%   │ 0.91  │  4min  │  2min  │   3/5     │  96%   │
│                                                              │
│  累计: 人 40min / 机 12min = 人机耗时比 3.3:1                 │
│  预计还需 1~2 轮完成                                          │
└──────────────────────────────────────────────────────────────┘
```

### 12.4 弱监督效果（启用时）

弱监督是卖点——领导要看到它省了多少人力。

```
┌──────────────────────────────────────────────────────────────┐
│  弱监督效果概览                                                │
│                                                              │
│  弱标签覆盖: 6,230 条 (92% 的未标注对象被打上了弱标签)          │
│  弱标签准确率: 87.3% (以人工标注为基准校准)                     │
│                                                              │
│  人力节省:                                                     │
│    实际人工标注:    840 条                                    │
│    纯人工达到同等精度需标: 3,200 条                            │
│    节省:           2,360 条 (73.8%)                           │
│                                                              │
│  收敛加速:                                                     │
│    达到 90% 覆盖率 — 有弱监督: 3 轮  │  无弱监督: 5 轮         │
│    总耗时           — 有弱监督: 18min │  无弱监督: 40min       │
│                                                              │
│  ┌ 弱标签来源分布 ───────────────────────────────────────┐    │
│  │  LF_keyword_positive  ████████████████  2,140 条 34%  │    │
│  │  LF_regex_business    ██████████        1,380 条 22%  │    │
│  │  LF_keyword_sports    ████████          1,020 条 16%  │    │
│  │  LF_external_kb       ██████              860 条 14%  │    │
│  │  LRR 规则修正          ██                  310 条  5%  │    │
│  │  (被 LRR 丢弃)        █                    520 条  8%  │    │
│  └───────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

计算逻辑：

- 弱标签覆盖 = 至少被 1 条 LF 打上标签的对象数 ÷ 总对象数
- 弱标签准确率 = 弱标签与人工标注一致的条数 ÷ 同时有弱标签和人工标注的条数
- 人力节省 = 同等覆盖率下纯人工所需标注数 − 实际人工标注数
- LRR 修正和被丢弃的分别展示——说明规则引擎在有效过滤噪声

---

## 13. 覆盖可视化

论文 Figure 4 的核心——把高维 embedding 降维到 2D，实时展示覆盖进度。管理员一眼看懂项目走到哪了。

### 13.1 降维

```
数据导入后：
  → 取所有标记对象的 embedding（768 维）
  → t-SNE 或 UMAP 降到 2D
  → 坐标 (x, y) 入库，只算一次
```

embedding 固定 → 2D 投影固定。每轮可视化只需重新着色（覆盖/未覆盖）和更新覆盖圆半径，不需重算投影。

### 13.2 可视化元素

```
┌──────────────────────────────────────────────────────────┐
│  覆盖可视化  │  Round 3  │  覆盖率 58%  │  [◀ 上轮] [下轮 ▶]  │
├──────────────────────────────────────────────────────────┤
│                                                          │
│       ●  ●●  ●                                          │
│     ●  ░░      ●                   ░░░ 已覆盖区域         │
│    ●●●░░░░░░░  ●                   ●   未覆盖点           │
│   ●░░░░░◉░░░░░                   ◉   标注中心（有覆盖圆）  │
│  ● ░░░░░░░░░░      ●             ░░░ 覆盖圆半径 = δ_x    │
│     ●░░░░░░░░                                                   │
│        ░░░                      ○ = 绿色 / ● = 红色 / ◆ = 蓝色  │
│          ●    ●                                              │
│                                                          │
│  ┌ 图例 ──────────────────────────────────────────┐      │
│  │ ○ 类别A  ● 类别B  ◆ 类别C    ◉ 本轮新增标注    │      │
│  │ 实心 = 未覆盖   空心 = 已覆盖                  │      │
│  └────────────────────────────────────────────────┘      │
│                                                          │
│  悬停任意点显示: anno_id, 内容摘要, 覆盖状态, 标签         │
└──────────────────────────────────────────────────────────┘
```

### 13.3 交互

| 操作 | 效果 |
|------|------|
| 缩放/拖拽 | 放大局部查看边界细节 |
| 轮次滑块 | 拖动查看每轮覆盖演变（动画过渡） |
| 悬停 | 显示标记对象摘要、标签、归属专家 |
| 点击 | 展开该对象的完整渲染内容 |
| 按类别筛选 | 只看某类别的覆盖情况 |
| 按状态筛选 | 只看未覆盖 / 已覆盖 / 本轮新增 |

### 13.4 技术要点

- 降维在前端渲染（服务端算好 2D 坐标，前端用 Canvas/SVG 绘制）
- 每轮结束时触发一次 t-SNE/UMAP（全量重算，非增量），2~3 秒可完成 1 万点
- 覆盖圆半径对应论文中 \( \delta_x(\epsilon, \theta) \)，覆盖内区域半透明着色
- 数据量过大时（>5 万点），采样展示 + 聚合热力图

---

## 14. 专家质量与效率统计

基于任务表历史数据，无需额外表。

### 14.1 人机一致性

专家与机器对同一样本的标注是否一致。

```
专家 A 的最近 N 次人机一致性：

WITH pairs AS (
  SELECT h.expert_id, h.label AS human_label, m.label AS machine_label
  FROM anno_task_{pid} h
  JOIN anno_task_{pid} m
    ON h.anno_id = m.anno_id AND m.source = 'MACHINE'
  WHERE h.source = 'HUMAN'
    AND h.expert_id = {A}
    AND h.status = 2
    AND m.status = 2
  ORDER BY h.submitted_at DESC
  LIMIT {N}
)
SELECT
  COUNT(*) AS total,
  COUNT(*) FILTER (WHERE human_label = machine_label) AS consistent,
  ROUND(100.0 * COUNT(*) FILTER (WHERE human_label = machine_label) / COUNT(*), 1) AS pct
FROM pairs;
```

### 14.2 专家间一致性

同一样本被两个专家标注时，标签是否一致。

```
专家 A 与 专家 B 的最近 N 次一致性：

WITH pairs AS (
  SELECT a.expert_id AS expert_a, a.label AS label_a,
         b.expert_id AS expert_b, b.label AS label_b
  FROM anno_task_{pid} a
  JOIN anno_task_{pid} b
    ON a.anno_id = b.anno_id
   AND a.expert_id != b.expert_id
   AND a.source = 'HUMAN' AND b.source = 'HUMAN'
  WHERE a.expert_id = {A} AND b.expert_id = {B}
    AND a.status = 2 AND b.status = 2
  ORDER BY a.submitted_at DESC
  LIMIT {N}
)
SELECT
  COUNT(*) AS total,
  COUNT(*) FILTER (WHERE label_a = label_b) AS agreed,
  ROUND(100.0 * COUNT(*) FILTER (WHERE label_a = label_b) / COUNT(*), 1) AS pct
FROM pairs;
```

也可不指定 B，统计 A 与**任意其他专家**的整体一致率：将 `b.expert_id = {B}` 去掉即可。

### 14.3 标注效率

```
专家 A 的最近 N 次平均耗时：

SELECT
  COUNT(*) AS total,
  AVG(EXTRACT(EPOCH FROM (submitted_at - locked_at))) AS avg_seconds
FROM anno_task_{pid}
WHERE source = 'HUMAN'
  AND expert_id = {A}
  AND status = 2
  AND locked_at IS NOT NULL
  AND submitted_at IS NOT NULL
ORDER BY submitted_at DESC
LIMIT {N};
```

### 12.3 展示

### 14.4 展示

```
┌──────────────────────────────────────────────────┐
│  专家: 张三                                       │
│                                                  │
│  最近 100 次标注:                                 │
│    人机一致性:  91.0%  (91/100 与机器推荐一致)      │
│    专家间一致性: 88.5%  (46/52 与他人标注一致)      │
│    平均耗时:    8.3 秒/条                          │
└──────────────────────────────────────────────────┘
```

---

## 15. 不一致复盘视图

管理员拉出不一致的标注记录，用于质量复盘。

### 15.1 全局最近 N 条不一致

```
SELECT
  t.anno_id,
  t.expert_id,
  t.label AS expert_label,
  m.label AS machine_label,
  t.submitted_at
FROM anno_task_{pid} t
JOIN anno_task_{pid} m
  ON t.anno_id = m.anno_id AND m.source = 'MACHINE'
WHERE t.source = 'HUMAN'
  AND t.status = 2 AND m.status = 2
  AND t.label != m.label
ORDER BY t.submitted_at DESC
LIMIT {N};
```

### 15.2 指定专家最近 N 条不一致

加 `t.expert_id = {A}` 即可。

```
SELECT ... (同上)
WHERE ...
  AND t.label != m.label
  AND t.expert_id = {A}
ORDER BY t.submitted_at DESC
LIMIT {N};
```

### 15.3 复盘界面

```
┌──────────────────────────────────────────────────────────────┐
│  不一致记录  │  共 23 条                                       │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  #142  张三标 "负面"  |  机器推荐 "正面"  |  最终裁定: 负面     │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  (anno_type.views.annotate(row) 渲染)                 │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                              │
│  #138  李四标 "中性"  |  机器推荐 "负面"  |  最终裁定: 负面     │
│  ...                                                         │
└──────────────────────────────────────────────────────────────┘
```

---

## 16. `annotators_per_sample = 1`（默认）

每样本仅 1 个 task 行 → 提交即回写 → 无裁定 → 行为等同于最早的单人模式。
