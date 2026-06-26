# SALT 多用户标注系统设计方案

---

## 1. 系统概述

基于 SALT 论文构建一个多角色的在线数据标注平台。核心流程：管理员创建标注项目 → 导入无标注数据 → 分配标注专家 → 系统自动执行 HITL 轮次（SALT 选样 → 专家标注 → 模型训练 → 覆盖率检查）→ 达标后自动停止 → 导出已标注数据。

---

## 2. 角色与权限

| 角色 | 权限 |
|------|------|
| **管理员 (Admin)** | 创建/管理项目、分配专家、全局设置、查看所有项目进度 |
| **数据导入员 (Importer)** | 往已授权的项目导入数据、查看导入历史 |
| **标注专家 (Annotator)** | 查看被分配的项目、领取标记任务、标注并提交、查看个人统计 |
| **数据导出员 (Exporter)** | 从已完成的项目导出标注结果 |
| **审阅员 (Reviewer)** （可选） | 抽检已标注数据、打回不合格标注 |

> 注：一个用户可身兼多角色。例如管理员通常也兼具导入/导出权限。

---

## 3. 核心实体与数据模型

### 3.1 项目 (Project)

```
Project:
  id: UUID
  name: string                    # 项目名称
  description: string             # 描述
  labels: string[]                # 标签集, 如 ["A","B","C","D"]，创建时确定，不可修改
  zeta: float                     # 精度容忍度 ζ (如 0.1)
  status: enum                    # DRAFT | EMBEDDING | READY | RUNNING | COMPLETED | PAUSED
  total_samples: int              # 导入的样本总数
  labeled_samples: int            # 已完成标注的样本数（专家标注 + 模型覆盖）
  sample_type: enum              # TEXT | IMAGE | VIDEO | RICH_TEXT。当前版本仅支持 TEXT
  import_config: {
    source: "file"                # 当前：文件上传。后续扩展：csv | postgres | api
    format: "txt"                 # 当前："txt" 表示按行切割。后续扩展："csv"、"jsonl"
  }
  salt_config: {
    batch_size: int               # 每轮标注量 (bgt)
    alpha: float                  # 覆盖率自适应评分权重
    decay_rate: float             # 弱监督权重衰减率 ρ
    initial_labeled: int          # 初始随机标注数量
    encoder: string               # 编码器名称，如 "qwen-embedding-v4" / "dinov2"
    skip_review: boolean          # 是否跳过审阅阶段（默认 false，即需要审阅）
  }
  created_by: user_id
  created_at: timestamp
  updated_at: timestamp
```

### 3.2 数据集 (Dataset) — 一次上传的数据批次

```
Dataset:
  id: UUID
  project_id: UUID
  source: enum                    # FILE | CSV | DATASOURCE。当前版本仅 FILE
  filename: string                # 原始文件名
  sample_count: int               # 本批次样本数
  status: enum                    # UPLOADING | EMBEDDING | READY
  uploaded_at: timestamp
```

> 当前版本：每个项目仅允许一个 Dataset。但模型设计为 1:N，后续升级只需放开逻辑层校验即可支持多份文件上传。

### 3.3 标记对象 (Sample)

```
Sample:
  id: UUID
  dataset_id: UUID                # 所属数据集
  sample_type: enum               # TEXT | IMAGE | VIDEO | RICH_TEXT。当前版本仅 TEXT
  content: string                 # 样本内容。TEXT 时为文本原文；IMAGE/VIDEO 时为资源路径或 URL
  embedding: float[] | null       # 编码后的向量，导入后异步计算
  true_label: string | null       # 最终标签（由 model 预测 或 专家标注确定）
  label_source: enum | null       # HUMAN | MODEL（标记来源）
  status: enum                    # UNLABELED | ASSIGNED | LABELED | REVIEWED
  coverage_radius: float | null   # 若为标注中心点，则有其覆盖半径 δ_x
  round_annotated: int | null     # 在哪一轮被标注
  imported_at: timestamp
  labeled_at: timestamp | null
```

### 3.4 标记任务 (AnnotationTask)

```
AnnotationTask:
  id: UUID
  project_id: UUID
  round: int                      # 第几轮
  sample_id: UUID
  assigned_to: user_id | null     # 当前持有者（谁抢到的）
  original_assignee: user_id | null # 最初持有者（被抢走后保留，用于统计）
  status: enum                    # PENDING | CLAIMED | SUBMITTED | REVIEWED | REJECTED
  annotated_label: string | null  # 专家原始标注的标签
  reviewed_label: string | null   # 审阅后修改的标签（若审阅员修改过）
  reviewed_by: user_id | null     # 审阅人
  version: int                    # 乐观锁版本号，每次抢夺/提交/审阅时 +1
  claimed_at: timestamp | null
  submitted_at: timestamp | null
  snatch_count: int               # 被抢夺次数（默认 0）
```

### 3.5 轮次 (Round)

```
Round:
  id: UUID
  project_id: UUID
  round_number: int
  status: enum                    # SAMPLING | ANNOTATING | REVIEWING | TRAINING | COVERAGE_CHECK | COMPLETED
  total_tasks: int                # 本轮生成的任务数
  completed_tasks: int            # 已完成标注的任务数
  coverage_ratio: float | null    # 本轮训完后计算的覆盖率
  risk_gap: float | null          # 本轮风险差距 |R_D(θ_S) - R_D(θ_D)|
  model_accuracy: float | null    # 当前模型估计精度
  started_at: timestamp
  completed_at: timestamp | null
```

### 3.6 项目成员 (ProjectMember)

```
ProjectMember:
  project_id: UUID
  user_id: UUID
  role: enum                      # ANNOTATOR | REVIEWER | IMPORTER | EXPORTER
  assigned_at: timestamp
```

### 3.7 通知 (Notification)

```
Notification:
  id: UUID
  user_id: UUID
  type: enum                      # NEW_ROUND | TASK_SNATCHED | REVIEW_READY | PROJECT_COMPLETED | LABEL_REJECTED
  title: string
  content: string
  project_id: UUID | null
  is_read: boolean
  created_at: timestamp
```

### 3.8 用户 (User)

```
User:
  id: UUID
  username: string
  password_hash: string
  display_name: string
  roles: enum[]                   # [ADMIN, ANNOTATOR, IMPORTER, EXPORTER, REVIEWER]
  is_active: boolean
  created_at: timestamp
```

---

## 4. 项目生命周期

项目是一次性的：数据一次性导入，标签集创建时确定，中途不可追加新的待标数据。

### 4.1 状态机

```
                    ┌──────────────────┐
                    │      DRAFT       │  项目创建，标签集已定，待导入数据
                    └────────┬─────────┘
                             │ 导入数据文件
                             ▼
                    ┌──────────────────┐
                    │    EMBEDDING     │  异步计算 embedding（可查看进度）
                    └────────┬─────────┘
                             │ embedding 全部完成
                             ▼
                    ┌──────────────────┐
              ┌─────│      READY       │  等待管理员分配专家并启动
              │     └────────┬─────────┘
              │              │ 管理员启动
              │              ▼
              │     ┌──────────────────┐
              │     │     RUNNING      │◄──────────┐
              │     └────────┬─────────┘           │
              │              │                     │
              │              │    ┌────────────────┤
              │              │    │   SALT HITL    │
              │              │    │   多轮循环      │
              │              │    │                │
              │              │    │  Round N:      │
              │     ┌────────┴────┴────┐           │
              │     │   SAMPLING       │  SALT 选取本轮样本
              │     └────────┬────────┘           │
              │              │                     │
              │     ┌────────▼────────┐           │
              │     │   ANNOTATING    │  专家在线抢单标注
              │     └────────┬────────┘           │
              │              │ 全部提交            │
              │     ┌────────▼────────┐           │
              │     │   REVIEWING     │  管理员/审阅员审核（可跳过）
              │     └────────┬────────┘           │
              │              │ 确认通过            │
              │     ┌────────▼────────┐           │
              │     │    TRAINING     │  训练模型  │
              │     └────────┬────────┘           │
              │              │                     │
              │     ┌────────▼────────┐           │
              │     │ COVERAGE_CHECK  │  计算覆盖率 │
              │     └────────┬────────┘           │
              │              │                     │
              │         ┌────┴────┐               │
              │     未达标      达标              │
              │       │          │                │
              │       └──────────┘                │
              │        下一轮 (回到 SAMPLING)       │
              │                                  │
              │                                  │
              ▼                                  ▼
     ┌──────────────────┐              ┌──────────────────┐
     │     PAUSED        │              │    COMPLETED     │
     │ (管理员可随时暂停) │              │ 达标，可导出结果  │
     └──────────────────┘              └──────────────────┘
```

### 4.2 状态说明

| 状态 | 含义 | 可执行操作 |
|------|------|-----------|
| DRAFT | 已创建，标签集已定，无数据 | 导入数据、编辑配置、删除项目 |
| EMBEDDING | 数据已导入，正在计算向量 | 查看进度，等待完成 |
| READY | embedding 就绪，等待启动 | 分配专家、编辑配置、启动标注 |
| RUNNING | 正在执行 SALT HITL 循环 | 专家标注、管理员审阅、查看轮次状态 |
| PAUSED | 管理员暂停 | 恢复运行（回到 RUNNING） |
| COMPLETED | 覆盖率达标，标注结束 | 导出结果、查看统计 |

### 4.3 约束

- 数据只能导入一次，导入后 DRAFT → EMBEDDING，不可回退
- READY 状态前不能分配专家或启动
- COMPLETED 后不可重新开始

### 4.4 项目克隆

从已有项目快速创建新项目，可选择克隆内容：

```
┌──────────────────────────────────────────┐
│  克隆项目: 评论情感分类                    │
│                                          │
│  新项目名称: [评论情感分类_v2       ]      │
│                                          │
│  克隆内容:                               │
│  ☑ 标签集 (A/B/C/D)          — 必选      │
│  ☑ 标记类型 (anno_type)       — 必选      │
│  ☐ 项目成员及角色                        │
│  ☐ 项目配置 (zeta, batch, 审阅开关等)    │
│  ☑ 算法名称                              │
│  ☐ 复用已训练模型                        │
│     └─ 仅算法名相同时可选，新项目从热启动   │
│        开始，首轮无需冷启动               │
│                                          │
│  [确认克隆]                              │
└──────────────────────────────────────────┘
```

| 克隆项 | 说明 |
|--------|------|
| 标签集 | 必选，新项目标签与源项目一致 |
| 标记类型 | 必选，anno_type 不变 |
| 项目成员 | 可选，复制所有成员的 project_role |
| 项目配置 | 可选，复制 zeta/batch_size/alpha/skip_review 等 |
| 算法名称 | 必选，沿用同一算法 |
| 复用已训练模型 | 可选，仅算法名相同时出现。克隆最后一轮的模型文件作为新项目初始模型，首轮直接热启动训练 |

克隆后新项目为 `DRAFT` 状态，需重新导入数据。

---

## 5. 专家抢单标注机制（多专家在线协作）

采用**多专家同时在线抢单**的协作模式。核心原则：

- 每一轮有一个**共享任务池**，所有被分配到此项目的专家都可看到
- 专家**每次抢一条**，标注完提交后再抢下一条
- 支持**抢夺**：他人已领但超时未提交的任务，可被抢走

### 5.1 共享任务池

```
┌──────────────────────────────────────────┐
│            共享任务池（本轮 bgt 条）         │
│                                          │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐        │
│  │ P1  │ │ P2  │ │ P3  │ │ ... │  ← 待抢 │
│  └─────┘ └─────┘ └─────┘ └─────┘        │
│                                          │
│  ┌─────────┐ ┌─────────┐                │
│  │ 专家A持有 │ │ 专家B持有 │  ← 已被领     │
│  │  (2条)  │ │  (1条)   │              │
│  └─────────┘ └─────────┘                │
└──────────────────────────────────────────┘
```

- 专家进入标注页面，看到当前**可抢数量**（池中 status=PENDING 的任务数）
- 点击"开始标注"→ 系统从池中随机取一条，标记为 CLAIMED，归属该专家
- 专家在标注界面看到文本内容，选择标签，点击"提交"
- 提交成功后，该任务标记为 SUBMITTED，专家自动抢取下一条
- 池中数量定时刷新（如每 3 秒），专家可看到实时变化

### 5.2 持有上限

每位专家同时持有（已抢未提交）的任务数有上限（如 `max_held = 5`），防止一人囤积过多。达到上限后不能再抢，必须提交或放弃手中的任务才能继续。

### 5.3 抢夺机制

当池中 `PENDING` 数量为 0，但仍有任务被其他专家 `CLAIMED`（持有中）时，专家可以**抢夺**：

```
抢夺条件：
  池中 PENDING 数 = 0
  AND 存在 CLAIMED 任务，其 claimed_at 距今 > 超时阈值（如 10 分钟）

抢夺操作：
  1. 专家页面显示"池已空，N 条可抢夺"
  2. 点击"抢夺标注"
  3. 系统从超时的 CLAIMED 任务中随机取一条
  4. 将旧持有者清空，新持有者设为当前专家
  5. 重置 claimed_at 计时
```

### 5.4 被抢后的冲突处理

当原持有者回来提交已被抢夺的任务时：

```
原持有者点击"提交"
  → 后端校验：task.assigned_to != 当前用户
  → 返回错误："该样本已被他人作答，页面即将刷新"
  → 前端刷新，丢弃当前标注内容，重新抢一条（或显示池已空）
```

这就要求后端在提交时做**乐观锁校验**（比对任务版本号或持有者 ID）。

### 5.5 专家标注界面交互流

```
┌─────────────────────────────────────────────┐
│  项目: 新闻分类  │  轮次: 3  │  进度: 28/40     │
│  池中剩余: 12 条  │ 我持有: 3 条               │
├─────────────────────────────────────────────┤
│                                             │
│  ┌─────────────────────────────────────┐    │
│  │  当前文本:                           │    │
│  │  "央行宣布下调存款准备金率0.5个百分点"   │    │
│  │                                     │    │
│  │  请选择标签:                         │    │
│  │  ○ 财经  ○ 体育  ○ 科技  ○ 娱乐       │    │
│  │                                     │    │
│  │  [ 跳过 / 暂时放弃 ]  [ 提交并抢下一条 ] │    │
│  └─────────────────────────────────────┘    │
│                                             │
│  ┌ 我持有的任务 ─────────────────────────┐   │
│  │ #1 "A股三大指数集体收涨..."  [继续标注] │   │
│  │ #2 "NBA总决赛G7即将打响..." [继续标注] │   │
│  │ #3 "苹果发布新款MacBook..." [放弃]     │   │
│  └──────────────────────────────────────┘   │
│                                             │
│  池已空时可操作: [抢夺他人超时任务 (3条)]      │
└─────────────────────────────────────────────┘
```

关键交互：
- **提交并抢下一条**：原子操作，两条数据库写入在一个事务中完成（提交当前 + 领取下一条）
- **跳过/放弃**：当前持有的某条可主动放回池中，状态变回 PENDING
- **被抢后的提示**：页面顶部弹 toast "任务 #42 已被他人抢走作答"，该任务自动从"我持有"列表中消失

### 5.6 轮次推进

- 当本轮所有任务 `SUBMITTED` 数量 == `total_tasks` 时：
  - 若 `skip_review = false`：轮次进入 `REVIEWING` 状态，通知管理员/审阅员
  - 若 `skip_review = true`：直接进入模型训练阶段
- 若有少量任务因抢夺冲突、遗失等长期卡在 CLAIMED 状态，管理员可**手动强制结束轮次**，将残留的 CLAIMED 任务直接释放（变回 PENDING 进入下一轮）
- 训练完成、新一轮池子生成时，推送通知给所有专家："第 N+1 轮已开始"

---

## 6. 审阅阶段

每一轮专家标注全部提交后，默认进入审阅阶段。管理员可以逐条浏览本轮所有标注结果，修改不认可的标签，修改后点击"确认进入下一轮"。管理员可在项目配置中设置跳过审阅阶段。

### 6.1 审阅工作流

```
  专家全部提交完一轮
        │
        ▼
  ┌─────────────┐     skip_review=true     ┌──────────┐
  │  ROUND 进入   │ ───────────────────────→ │  训练模型  │
  │  REVIEWING   │                          └──────────┘
  └──────┬──────┘
         │ skip_review=false (默认)
         ▼
  ┌───────────────────────────────┐
  │ 管理员/审阅员收到通知           │
  │ "第 N 轮标注完成，请进入审阅"   │
  └──────────────┬────────────────┘
                 │
                 ▼
  ┌───────────────────────────────────────────────┐
  │              审阅界面                          │
  │                                               │
  │  本轮共 40 条标注，按专家分组或逐条浏览          │
  │                                               │
  │  ┌─────────────────────────────────────────┐  │
  │  │ #12  "央行宣布下调存款准备金率..."        │  │
  │  │ 专家A 标注: 财经                         │  │
  │  │                                         │  │
  │  │ 修改为:  [财经 ▾]    [✓ 确认] [✗ 打回]  │  │
  │  └─────────────────────────────────────────┘  │
  │                                               │
  │  [一键通过全部]    [确认并进入下一轮]            │
  └───────────────────────────────────────────────┘
```

### 6.2 审阅操作

| 操作 | 效果 |
|------|------|
| **直接通过** | 不改标签，状态从 SUBMITTED → REVIEWED |
| **修改标签** | `reviewed_label` 设为新标签，`reviewed_by` 记录审阅人，状态 → REVIEWED |
| **打回** | 状态从 SUBMITTED → PENDING，退回任务池（极少使用，仅当专家明显标错时） |
| **一键通过全部** | 本轮所有 SUBMITTED 任务批量标为 REVIEWED（不改标签） |
| **确认进入下一轮** | 所有任务处理完毕后，管理员点击此按钮，触发模型训练 |

### 6.3 审阅界面的过滤与排序

- **按专家筛选**：只看某位专家的标注
- **按标签筛选**：只看被标为某个标签的样本
- **按置信度排序**：若模型已有中间结果，按模型预测置信度从低到高排列（优先审阅模型最不确定的样本）
- **仅看修改过的**：快速定位审阅员已干预的条目

### 6.4 跳过审阅

管理员在创建项目或项目设置中开启 `skip_review = true`：
- 每轮专家标注全部提交后，**自动跳过审阅**，直接训练模型并进入下一轮
- 适用于专家质量高度可信的场景，或追求最快迭代速度的项目

---

## 7. 通知系统

| 事件 | 通知对象 | 内容 |
|------|---------|------|
| 新轮次开始 | 该项目的所有标注专家 | "项目 X 第 N 轮已开始，池中 M 条待抢" |
| 任务被抢走 | 原持有者 | "您持有的任务 #42 已被他人抢走作答" |
| 轮次标注完成，进入审阅 | 管理员、审阅员 | "项目 X 第 N 轮标注完成，请进入审阅" |
| 项目标注完成 | 管理员、导出员 | "项目 X 已达到目标精度，可以导出" |
| 标注被审阅打回 | 该专家 | "您标注的样本被审阅员退回，请修改" |
| 导入完成 | 管理员 | "数据导入完成，共 N 条" |

实现方式：轮询 + WebSocket 推送（前端收到后更新通知角标）。

---

## 8. 数据导入与导出

### 8.1 导入

**当前版本（v1）：**
- 格式：`.txt`，一行一条样本，纯文本
- 样本类型固定为 `TEXT`
- 一次上传全部待标数据，项目仅允许一个 Dataset
- 导入后创建 Dataset 记录，异步计算 embedding

**模型已为后续扩展预留：**

| 扩展 | 预留字段 | 说明 |
|------|---------|------|
| 样本类型多样化 | `Sample.sample_type` | 当前 `TEXT`，后续支持 `IMAGE`、`VIDEO`、`RICH_TEXT`。非文本类型时 `content` 存资源路径或 URL |
| 多格式导入 | `Dataset.source` / `Project.import_config` | 当前 `FILE` + `txt`，后续支持 `CSV`、`JSONL`、数据库直连等 |
| 多文件上传 | 模型已为 1:N（Project → Dataset） | 后续放开逻辑校验即可，历史数据无需迁移

### 8.2 导出

- 仅 `COMPLETED` 状态的项目可导出
- 导出格式：CSV / JSON / TXT，每行包含 `content, label, label_source, confidence_score`
- 可选择性导出：只导出模型预测部分 vs 全部
- 导出员操作需记录日志

---

## 9. 模块划分（API 层面）

```
/api/auth              # 登录/注册/Token 刷新
/api/users             # 用户管理（管理员）
/api/projects          # 项目 CRUD（管理员）
    /{pid}/members     # 项目成员管理
    /{pid}/import      # 数据导入
    /{pid}/export      # 数据导出
    /{pid}/start       # 启动标注流程
    /{pid}/pause       # 暂停
    /{pid}/rounds      # 轮次信息
    /{pid}/rounds/current  # 当前轮次实时状态
    /{pid}/rounds/{rid}/review  # 审阅本轮标注
        /tasks              # GET 本轮所有已提交任务（支持按专家/标签筛选）
        /tasks/{tid}        # PUT 修改标签 / POST 打回
        /approve-all        # POST 一键通过全部
        /confirm            # POST 确认审阅完成，进入下一轮（触发训练）
/api/tasks             # 标记任务
    /{pid}/pool/status    # GET 池子实时状态（pending数、held数、可抢夺数）
    /{pid}/grab           # POST 从池中抢一条（原子操作，返回任务内容）
    /{tid}/submit         # POST 提交标注（含乐观锁校验）
    /{tid}/release        # POST 放弃当前持有的任务，放回池中
    /{pid}/snatch         # POST 抢夺他人超时任务
    /my-held              # GET 我当前持有的任务列表
/api/notifications     # 通知列表、标记已读
/api/stats             # 统计面板
```

---

## 10. 前端页面结构

```
/login                          # 登录页

/expert/
    /projects                   # 我的项目列表
    /projects/:pid/annotate     # 标注工作台（核心页面：单条标注 + 抢单 + 我持有的列表）
    /notifications              # 通知中心

/admin/
    /projects                   # 全部项目管理
    /projects/:pid              # 项目详情（覆盖率图表、轮次历史、实时池子监控）
    /projects/:pid/members      # 成员分配
    /projects/:pid/import       # 数据导入
    /projects/:pid/export       # 数据导出
    /projects/:pid/review       # 审阅当前轮标注（逐条浏览、修改标签、一键通过）
    /users                      # 用户管理
    /settings                   # 系统配置

/reviewer/
    /projects                   # 需审阅的项目列表
    /projects/:pid/review       # 审阅工作台（同管理员审阅界面）

/importer/
    /projects                   # 可导入的项目列表
    /projects/:pid/import       # 导入界面 + 历史记录

/exporter/
    /projects                   # 已完成的项目列表
    /projects/:pid/export       # 导出界面
```

---

## 11. 关键设计决策与取舍

| 问题 | 决策 | 理由 |
|------|------|------|
| 数据导入 | 一次性上传全部，导入后不可追加 | 简化生命周期语义；中途追加会破坏覆盖率保证 |
| 项目生命周期 | 严格状态机：DRAFT→EMBEDDING→READY→RUNNING→COMPLETED | 每个阶段职责清晰，不可逆操作有明确边界 |
| 任务分配方式 | 共享任务池 + 在线抢单 + 抢夺超时任务 | 多专家实时协作，避免分配不均和僵尸任务 |
| 并发冲突处理 | 乐观锁（version 字段），提交时校验持有者 | 保证抢夺场景下数据正确性 |
| 持有上限 | 每人最多同时持有 max_held 条（默认 5） | 防止一人囤积阻塞他人 |
| 抢夺超时阈值 | 10 分钟无操作可被抢 | 平衡公平与效率 |
| 轮次触发 | 本轮全部完成 + 审阅通过后触发训练 | 先质控再迭代，保证标注质量 |
| 审阅阶段 | 默认需要审阅，管理员可配置跳过 | 灵活性优先；跳过适用于高质量专家场景 |
| 概念漂移 | 暂不纳入核心流程 | 论文理论不支持；作为 v2 迭代项 |
| 历史标注复用 | 热启动训练 + 时间衰减权重 | 论文原为冷启动，改为可选配置 |
| 弱监督集成 | 可选，管理员创建项目时决定是否启用 | 弱监督需 LFs，不是所有任务都有 |
| embedding 计算 | 异步，导入后批量计算入库 | 避免导入时阻塞，支持失败重试 |

---

## 12. 待扩展项（v2）

- 概念漂移检测与自适应遗忘（滑动窗口 + 覆盖率失效检测）
- 专家质量评分与加权投票（多人标注同一样本取多数）
- 多格式数据导入（CSV、JSONL、图片目录）
- SALT 超参数自动调优
- 项目克隆（复制配置 + 重新导入数据）
