# LRR 标签修正规则

---

## 是什么

LRR 是一组 IF-THEN 逻辑规则，修正标注函数（LF）输出的弱标签。形式：

```
IF 条件成立 THEN 标签 = w
IF 条件成立 THEN 丢弃该标签
```

例如论文中的三条规则：

```
规则 1: IF LF_keyword 输出 sports 且 LF_regex 输出 business THEN 标签 = sports
规则 2: IF 样本 x 与已标注样本 y 同主题 且 y=体育 THEN 标签 = 体育
规则 3: IF LF_keyword 输出 scitech 且 融合模型置信度 < 0.1 THEN 丢弃
```

---

## LRR 内部会冲突吗

会。多条 LRR 可能同时命中同一条对象，给出不同后果。

### 冲突示例

```
对象: "苹果发布财报，股价大涨，NBA季后赛..."

规则 A: IF LF_entity_org=科技 ∧ LF_keyword_money=财经 THEN 标签 = 财经
规则 B: IF LF_keyword_sports=体育 ∧ LF_entity_org=科技 THEN 标签 = 体育
规则 C: IF Pr(财经) > 0.4 THEN 标签 = 财经

三条规则同时命中，后果不同：财经 / 体育 / 财经
```

### 冲突解决

对每条对象，所有命中的 LRR 按加权打分排序，取最高分的一条：

```
score(规则) = confidence × 0.5 + specificity × 0.3 + agreement × 0.2

confidence:  该规则在历史标注数据上的准确率
specificity: 规则前提条件的复杂度（条件越多越特异，越高）
agreement:   规则后果与融合模型 P 软标签的一致性
```

若最高分与次高分差距 < 阈值（如 0.1），说明拿不准，保守弃权，不执行任何规则。

---

## 来源：内置 + 自动发现

### 内置（管理员手写）

管理员在项目配置中逐条编写：

```
新增 LRR:
  名称: "体育优先于科技"
  条件: T(LF_keyword_sports, x, 体育) ∧ T(LF_entity_org, x, 科技)
  后果: x.label = 体育
```

适合管理员有明确领域知识时（如"体育关键词比公司名更可靠"）。

### 自动挖掘（离线）

从历史已完成项目中，用人标数据作为 ground truth，自动发现有效 LRR。

#### 挖掘流程

```
输入: 已完成项目的数据（LF 输出 + 人工标注的 final_label）

1. 枚举所有可能的谓词组合（工具谓词、软标签谓词、关系谓词）
2. 对每个候选规则，在数据上评估:
   - 命中次数（前提条件成立多少次）
   - 正确次数（后果与人工标注一致多少次）
   - confidence = 正确次数 / 命中次数
3. 筛选: confidence > 阈值（如 0.8）且 命中次数 > 最小支持度
4. 去重: 若规则 A 完全包含规则 B 且 confidence 更高，删除 B
5. 输出 LRR 规则集
```

#### 自适应剪枝

候选规则数量爆炸 → 不全部枚举。宽松支持度阈值保留更多候选，后续用 confidence 和覆盖率双重过滤，提高支持度时收紧剪枝。

### 规则管理

```
lrr_rules:
  id: UUID
  name: string
  source: enum                 # MANUAL | MINED
  condition: json              # 谓词定义
  consequence: json            # {label: "体育"} 或 {action: "discard"}
  confidence: float            # 准确率评估
  specificity: float           # 条件复杂度评分
  is_active: boolean
```

---

## 与融合模型 P 的关系

两者串行，不冲突：

```
LF 原始输出 → P（统计推断：从矛盾中找共识）→ 软标签 → LRR（逻辑修正：修已知错误）→ 硬标签
```

| | P | LRR |
|------|--|-----|
| 解决什么问题 | 不确定时怎么折中 | 确定错了怎么修 |
| 方式 | 概率推断，权重动态估计 | 逻辑匹配，规则固定 |
| 结果 | 软标签 | 硬修正 |
| 需要批量？ | 是（需要完整 Λ 矩阵） | 否（逐条匹配） |

不是替代关系——P 处理 LF 间的噪声，LRR 修正 P 也无法发现的特定错误模式。
