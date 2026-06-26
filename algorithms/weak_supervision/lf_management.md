# LF 管理

---

## LF 的输入

LF 接收的文本 = 标记对象的 model 视图输出（由 anno_type 的 `views.model(row)` 生成），与送入编码器的文本是同一份。

```
标记对象（复杂结构）
  → anno_type.views.model(row) → 纯文本
  → 同时喂给: 编码器（用于 embedding）+ 所有 LF（用于弱标签）
```

---

## LF 类型

| 类型 | 输入 | 逻辑 | 示例 |
|------|------|------|------|
| 关键字 | 文本 | 包含指定词 → 输出标签 | 包含"涨停"→ 财经 |
| 正则 | 文本 | 匹配正则 → 输出标签 | 匹配 `\d{6}\.(SZ\|SH)` → 财经 |
| 外部知识库 | 文本 | 查词典/实体库 → 输出标签 | 公司名在科技公司表中 → 科技 |
| Python 脚本 | 文本 | 任意 Python 逻辑 | 自定义复杂规则 |

---

## LF 数据表

```
labeling_functions:
  id: UUID
  project_id: UUID              # 所属项目
  name: string                  # "LF_keyword_sports"
  description: string           # 人类可读说明
  lf_type: enum                 # KEYWORD | REGEX | KNOWLEDGE_BASE | PYTHON
  config: json                  # 类型相关配置
  is_active: boolean            # 启用/停用
  created_at: timestamp
  updated_at: timestamp
```

### config 字段（按类型）

```json
// keyword:
{ "keywords": ["涨停", "央行", "GDP"], "output_label": "财经" }

// regex:
{ "pattern": "\\d{6}\\.(SZ|SH)", "output_label": "财经" }

// knowledge_base:
{ "table": "tech_companies", "column": "name", "output_label": "科技" }

// python:
{ "script": "def LF(text):\n    if len(text) > 200 and '公告' in text:\n        return '财经'\n    return None" }
```

---

## 操作

| 操作 | 说明 |
|------|------|
| 新增 | 管理员在项目中添加一条 LF |
| 编辑 | 修改名称、描述、类型、配置 |
| 启用/停用 | 切换 is_active，停用的 LF 不参与弱标签生成 |
| 删除 | 删除不再需要的 LF |
| 测试 | 输入一条样本文本，查看该 LF 的输出 |
| 批量测试 | 输入多条样本文本，查看命中率和输出分布 |

---

## 项目克隆时

```
☑ 克隆 LF 集合（共 12 条，3 条停用）
```

勾选后新项目获得一份完整的 LF 副本（新 ID，与原项目独立），后续各自增删互不影响。
