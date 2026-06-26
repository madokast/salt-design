# 算法任务

---

## 哪些需要异步

| 算法 | 耗时 | 同步/异步 | 原因 |
|------|------|----------|------|
| embedding 批量计算 | 分钟级 | 异步 | 调用外部 API，量大时慢 |
| 模型训练 | 秒~分钟 | 同步 | MLP 快，阻塞几秒可接受 |
| 模型预测 | 秒级 | 同步 | 批量 forward pass 很快 |
| 覆盖半径 + 选样 | 秒~数十秒 | 同步 | 纯计算，无 IO，数据量不大时快 |
| 降维计算 (t-SNE) | 数十秒~分钟 | 异步 | t-SNE 在万级以上数据上慢 |
| 弱监督 P 模型 | 秒~分钟 | 异步 | LF 数量多 + 数据量大时慢 |

---

## 任务表

```
algorithm_tasks:
  id            UUID PK
  project_id    UUID
  round         int | null            # 产生于第几轮（非轮次内的任务为 null）
  algorithm     string                # "embedding" | "tsne" | "weak_supervision"
  status        enum                  # PENDING | RUNNING | COMPLETED | FAILED
  progress      float DEFAULT 0       # 0~1
  progress_msg  string | null         # "正在计算向量 (4,500/10,000)"
  input_snapshot json                 # 入参快照（用于重试）
  result        json | null           # 出参
  error_message string | null
  error_code    string | null
  retry_count   int DEFAULT 0
  created_at    timestamp
  started_at    timestamp | null
  completed_at  timestamp | null
```

上层调用算法前先 INSERT 一条 PENDING 记录，将 `id` 传入算法。算法内部按需更新 `progress` 和 `progress_msg`，结束写 `status` 和 `result`。

---

## 前端感知

页面定时轮询任务状态：

| 算法 | 进度消息示例 |
|------|------------|
| embedding | "正在计算向量 (45%, 4,500/10,000)" |
| tsne | "正在降维 (迭代 300/1000)" |
| weak_supervision | "正在生成弱标签 (步骤 2/3: 融合模型 P)" |

任务完成 → 自动进入下一步骤；任务失败 → 展示错误码 + 重试按钮。

## 内部结构

每类算法分两层：

```
┌─────────────────────────┐
│  DB-aware Wrapper        │  ← 异步 Worker 调用这一层
│  - 读库：取 embedding、  │
│    LF 定义、配置等       │
│  - 调核心算法             │
│  - 写库：更新 progress、  │
│    落盘结果              │
└──────────┬──────────────┘
           │ 调用
┌──────────▼──────────────┐
│  Core Algorithm (纯函数)  │  ← 不接触数据库
│  - 输入 → 输出           │     可单元测试
│  - 无副作用              │     同步算法直接调这层
└─────────────────────────┘
```

同步算法（训练、覆盖计算）上层直接调 Core。异步算法 Worker 调 Wrapper，Wrapper 内部读库 → 调 Core → 写库。
