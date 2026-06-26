# 标记对象与渲染系统

标注对象是高度可变的——纯文本、爬虫评论、问卷答案、图片描述——每种结构不同。本系统通过**标记类型插件（anno_type）**实现灵活的内容存储与展示。

---

## 1. 核心概念

```
┌─────────────┐     定义表结构      ┌────────────────────────┐
│  anno_type  │ ─────────────────→ │  anno_content_{项目id}   │
│  (插件)     │                     │  anno_id (固定)          │
│             │                     │  ... 自定义字段          │
│  - 字段定义  │                     └────────────────────────┘
│  - 渲染脚本  │                              │
│  - 解析脚本  │                    一行数据 → 渲染脚本 → HTML
└─────────────┘
```

- **anno_type**：一个可复用的插件模板，定义了一组字段 + 配套的 Python 渲染脚本
- **anno_content_{project_id}**：每个项目独立的标记对象表，结构由创建项目时选定的 anno_type 决定
- **渲染**：专家标注时看到的不是原始字段，而是渲染后的 HTML 页面

---

## 2. anno_type 定义

### 2.1 实体

```
AnnoType:
  id: UUID
  name: string                    # 如 "plain_text"、"crawled_comment"
  display_name: string            # 展示名，如 "纯文本"、"爬虫评论"
  description: string
  source: enum                    # BUILTIN | USER_SCRIPT | GUI
  fields: [                       # 字段定义列表
    {
      name: "raw_text",           # 字段名
      type: "text" | "integer" | "float" | "datetime" | "json" | "file",
      label: "文本内容",           # 人类可读的标签
      required: true
    }
  ]
  import_adapters: string[]        # 支持的导入适配器，如 ["txt_line", "csv_table"]
  encoder_field: string            # 送入编码器计算 embedding 的字段名（模型视图的输入）
  views: {                         # 四种呈现
    annotate: string               # 专家标注视图 — Python 脚本 → 完整 HTML
    list: string                   # 列表浏览视图 — Python 脚本 → 一行摘要（纯文本，无标签）
    export: {                      # 导出视图
      csv: string                  # Python 脚本 → CSV 行文本
      json: string                 # Python 脚本 → JSON 对象
    }
    model: string                  # 模型输入视图 — 拼装送入编码器的文本（默认取 encoder_field）
  }
  gui_template: json | null        # GUI 拖拽式模板定义（source=GUI 时，后续版本）
  created_by: user_id | null       # 创建者，BUILTIN 为 null
  created_at: timestamp
```

### 2.2 四种呈现视图

同一个标注对象，在不同场景下需要不同形态：

| 视图 | 使用者 | 形式 | 示例 |
|------|--------|------|------|
| **annotate** | 标注专家 | 完整 HTML | 带样式的评论卡片，含头像、时间、内容、点赞数 |
| **list** | 浏览/审阅列表 | 一行纯文本 | `"张三: 这个产品太好了... (👍128)"` |
| **model** | 机器自动标注 | 拼接文本 → embedding | `"用户名:张三 内容:这个产品太好了"` |
| **export** | 数据导出 | CSV 行 / JSON | `"张三","这个产品太好了","128"` |

### 2.3 三种来源

| source | 含义 | 当前版本 |
|--------|------|---------|
| BUILTIN | 系统内置，随版本发布，不可删除 | ✓ |
| USER_SCRIPT | 用户编写 Python 脚本上传 | ✓ |
| GUI | 通过拖拽式 GUI 设计字段和 HTML 布局，无需写代码 | 后续版本 |

### 2.4 内置 anno_type 示例

#### plain_text（纯文本）

```
fields: [
  { name: "raw_text", type: "text", label: "文本内容", required: true }
]
encoder_field: "raw_text"
```

四种视图：

```python
# annotate — 专家看到的 HTML
def annotate(row):
    return f"<div class='anno-text'>{escape(row['raw_text'])}</div>"

# list — 列表中的一行摘要（截断 50 字）
def list_view(row):
    text = row['raw_text']
    return text[:50] + "..." if len(text) > 50 else text

# model — 送入编码器的文本
def model_view(row):
    return row['raw_text']

# export — 导出
def export_csv(row):
    return escape(row['raw_text'])
def export_json(row):
    return { "text": row['raw_text'] }
```

#### crawled_comment（爬虫评论）

```
fields: [
  { name: "user_name",   type: "text",     label: "用户名" },
  { name: "user_avatar", type: "text",     label: "头像URL" },
  { name: "post_time",   type: "datetime", label: "发布时间" },
  { name: "content",     type: "text",     label: "评论内容", required: true },
  { name: "likes",       type: "integer",  label: "点赞数" },
  { name: "reply_to",    type: "text",     label: "回复对象" },
]
encoder_field: "content"
```

四种视图：

```python
# annotate
def annotate(row):
    return f"""
    <div class='comment-card'>
      <img src='{escape(row["user_avatar"])}' />
      <strong>{escape(row["user_name"])}</strong>
      <span>{escape(row["post_time"])}</span>
      <p>{escape(row["content"])}</p>
      <span>👍 {row["likes"]}</span>
    </div>"""

# list — 一行摘要
def list_view(row):
    return f"{row['user_name']}: {row['content'][:40]}..."

# model — 拼装送入编码器
def model_view(row):
    return f"用户:{row['user_name']} 内容:{row['content']}"

# export
def export_csv(row):
    return f"{row['user_name']},{row['content']},{row['likes']}"
def export_json(row):
    return {
        "user": row['user_name'],
        "content": row['content'],
        "likes": row['likes']
    }
```

#### 可扩展性

管理员可以新建 anno_type，定义字段、编写渲染脚本。例如未来支持：

| anno_type | 用途 |
|-----------|------|
| `image_classify` | 图片分类：字段含 `image_url`，渲染为 `<img>` 标签 |
| `qa_pair` | 问答对：`question` + `answer` + `context` |
| `video_segment` | 视频片段：`video_url` + `start_time` + `end_time` |
| `form_record` | 表单记录：任意多字段 |

---

## 3. 项目与 anno_type 的关系

创建项目时，必须选定一个 anno_type：

```
Project:
  anno_type_id: UUID              # 关联的标记类型
```

选定后，系统自动创建该项目专属的内容表：

```sql
CREATE TABLE anno_content_{project_id} (
  anno_id   UUID PRIMARY KEY,
  -- 以下由 anno_type.fields 动态生成 --
  raw_text  TEXT,                -- 示例：plain_text
  -- 或 --
  user_name   TEXT,              -- 示例：crawled_comment
  user_avatar TEXT,
  post_time   TIMESTAMP,
  content     TEXT NOT NULL,
  likes       INTEGER,
  reply_to    TEXT
);
```

anno_type 选定后**不可更改**（表结构已建，改类型需重建项目）。

---

## 4. 数据导入：适配器与字段映射

数据源千差万别，anno_type 通过**导入适配器**将原始数据转为 `anno_content_{project_id}` 表中的记录。

### 4.1 导入适配器总览

| 适配器 | 输入 | 映射方式 | 当前版本 |
|--------|------|---------|---------|
| **txt_line** | .txt 文件，每行一条 | 整行存入 `encoder_field` 指定的字段 | ✓ |
| **csv_table** | .csv 文件，表头为字段名 | 用户可映射、重命名、选择字段 | ✓ |
| **datasource** | 数据库表/视图连接 | 自动全部字段一一映射 | 后续 |
| **media_zip** | .zip 多媒体包 | 解压到项目文件夹，路径存为字段值 | 后续 |

### 4.2 txt_line（按行切分）

最简单：一行文本 → 一条记录。

```
上传 .txt
  → 按行 split
  → 每行作为 encoder_field 的值
  → INSERT INTO anno_content_{pid} (anno_id, {encoder_field})
```

以 plain_text 为例，encoder_field = `raw_text`，每行存入 `raw_text`。

无需字段映射界面——只有一个字段。

### 4.3 csv_table（表格映射）

上传 CSV 后，进入**字段映射界面**：

```
┌─────────────────────────────────────────────────────┐
│  CSV 字段映射                                        │
│                                                     │
│  CSV 列名          →    内容表字段                     │
│  ─────────────────────────────────────────────────  │
│  user_name         →    user_name         [✓ 导入]  │
│  avatar            →    user_avatar       [✓ 导入]  │
│  time              →    post_time         [✓ 导入]  │
│  content           →    content           [✓ 导入]  │
│  likes             →    likes             [✓ 导入]  │
│  source_url        →    [忽略 ▾]          [✗ 不导入] │
│                                                     │
│  [自动映射（同名默认匹配）]  [开始导入]                │
└─────────────────────────────────────────────────────┘
```

规则：
- **自动映射**：CSV 列名与内容表字段名**完全一致**的，自动匹配
- **改名映射**：CSV 列名不一致时（如 `avatar` → `user_avatar`），用户手动选择目标字段
- **忽略字段**：不需要的 CSV 列可选择不导入，内容表对应字段留 NULL
- **必填校验**：`required: true` 的字段必须有映射（如 `content`），否则无法开始导入

### 4.4 datasource（连接数据源）

连接外部数据库的一张表或视图，自动全部字段映射：

```
管理员填写连接信息
  → 选择表/视图
  → 自动读取表结构
  → 与 anno_type.fields 做同名匹配
  → 支持少量手动调整（改名、忽略）
  → 开始导入
```

与 csv_table 的区别：字段默认全部导入，不需要逐列勾选。

### 4.5 media_zip（多媒体包）

适用于图片/视频/音频标注场景。anno_type 中某个字段的类型为 `file`，存储资源路径。

```
上传 .zip 包
  → 自动解压到 {项目多媒体目录}/{dataset_id}/
  → 解析包内的清单文件（如 manifest.csv）或按文件名匹配
  → 将相对路径写入对应字段，如：
      anno_id: "a1"
      image_url: "dataset_01/photo_a.jpg"
```

渲染时，用户的 Python annotate 脚本将路径转为 HTML：

```python
def annotate(row):
    return f"<img src='/media/{row['image_url']}' />"
```

专家看到的即是渲染后的图片/视频，而非原始路径字符串。

### 4.6 当前版本限制

- 仅支持 `txt_line` 适配器 + `plain_text` anno_type
- 导入格式仅 `.txt`，一行一条，无字段映射界面

---

## 5. 各视图调用场景

### 5.1 专家标注 — annotate 视图

```
专家点击"抢单"
  → 后端取一条 Sample，通过 anno_id 关联 anno_content_{project_id}
  → 查出行数据
  → 调用 anno_type.views.annotate(row) 生成 HTML
  → 返回给前端
  → 前端在标注工作台中渲染该 HTML
```

### 5.2 标注界面示意

标注工作台中，样本展示区是一个 iframe 或 sandbox div，内嵌渲染后的 HTML：

```
┌─────────────────────────────────────────────┐
│  项目: 评论情感分类  │  轮次: 2  │  进度: 15/30  │
│  池中剩余: 8 条      │  我持有: 3 条           │
├─────────────────────────────────────────────┤
│                                             │
│  ┌─────────────────────────────────────┐    │
│  │  (渲染后的 HTML)                     │    │
│  │                                     │    │
│  │  ┌──────────────────────────────┐   │    │
│  │  │ 🧑 张三               2小时前  │   │    │
│  │  │                             │   │    │
│  │  │ 这个产品真的太好用了，          │   │    │
│  │  │ 强烈推荐！                   │   │    │
│  │  │                    👍 128   │   │    │
│  │  └──────────────────────────────┘   │    │
│  │                                     │    │
│  └─────────────────────────────────────┘    │
│                                             │
│  请选择标签:                                 │
│  ○ 正面  ○ 负面  ○ 中性  ○ 垃圾广告           │
│                                             │
│  [ 跳过 / 放弃 ]  [ 提交并抢下一条 ]           │
└─────────────────────────────────────────────┘
```

### 5.3 列表浏览 — list 视图

审阅列表、任务池浏览等场景，以表格形式展示所有样本，每行一列，调用 `views.list(row)`：

```
┌──────────────────────────────────────────────────┐
│  ID   │  摘要                     │ 标签   │ 状态 │
├──────────────────────────────────────────────────┤
│  #12  │ 张三: 这个产品太好...      │ 正面   │ ✓    │
│  #13  │ 李四: 不推荐，质量很...    │ 负面   │ ✓    │
│  #14  │ 王五: 还行吧，一般般...    │ 待标   │ —    │
└──────────────────────────────────────────────────┘
```

### 5.4 模型自动标注 — model 视图

SALT 训练或模型预测时，需将结构化数据拼装成编码器可消费的文本：

```
取一条 Sample
  → 调用 anno_type.views.model(row) → 得到纯文本
  → 送入编码器 → 得到 embedding
  → 送入 MLP 分类器 → 得到预测标签
```

`model` 视图决定了哪些字段参与 embedding 计算、以什么顺序拼接。

### 5.5 导出 — export 视图

项目完成时，导出标注结果：

```
取所有 labeled Sample
  → 对每条调用 anno_type.views.export.csv(row)（或 json）
  → 拼接为 CSV 文件 / JSON 数组
  → 提供下载
```

### 5.6 安全考虑

所有视图脚本在服务端执行，生成的 HTML 经过清洗（去除 `<script>`、`<iframe>` 等），前端用 sandbox iframe 展示。

---

## 6. 数据库示意

```
anno_types                           -- 标记类型注册表
  ├─ id: "plain_text"
  ├─ id: "crawled_comment"
  └─ ...

projects
  ├─ id: "proj-001"
  │   anno_type_id: "crawled_comment"
  │   ...
  └─ ...

anno_content_proj-001                -- 项目专属内容表（动态创建）
  ├─ anno_id: "a1"
  │   user_name: "张三"
  │   post_time: "2026-06-26 10:00"
  │   content: "这个产品太好了"
  │   likes: 128
  │   ...
  └─ ...

samples                             -- 标注样本表
  ├─ id: "s1", anno_id: "a1", ...
  └─ ...
```

---

## 7. 后续扩展方向

| 扩展 | 说明 |
|------|------|
| 用户自定义脚本 | 管理员上传 Python render/parse 脚本，新建 anno_type（source=USER_SCRIPT） |
| GUI 设计器 | 拖拽字段 + 布局组件直接生成 HTML 模板，无需写代码（source=GUI） |
| 富文本渲染 | `rich_text` anno_type，支持 Markdown / HTML 原文渲染 |
| 图片/视频标注 | `image_classify`、`video_segment` anno_type，字段含资源 URL，渲染为媒体标签 |
| 多表关联 | 一条标注对象关联多张子表（如评论 + 用户画像），用多个 content 表 JOIN |
| 渲染脚本沙箱 | 限制 USER_SCRIPT 可访问的模块和系统调用 |
| 表单式标注 | 不只选标签，还可在 HTML 中嵌入输入框，专家直接修改内容（如纠错） |
