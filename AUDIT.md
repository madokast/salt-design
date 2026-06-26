# 审计日志

---

## 1. 设计原则

- **不可篡改**：日志只追加不修改不删除，应用层无 DELETE 权限
- **谁在什么时候对什么做了什么**：每条的必备字段
- **低开销**：异步写入，不阻塞业务流程
- **可检索**：支持按用户、项目、操作类型、时间范围筛选

---

## 2. 日志表

```
audit_logs:
  id            BIGSERIAL PK
  user_id       UUID              # 操作者
  username      string            # 冗余，便于直接展示
  project_id    UUID | null       # 关联项目（系统级操作为 null）
  action        string            # 操作类型，如 "task.grab"
  target_type   string | null     # 操作对象类型，如 "task" / "project" / "user"
  target_id     UUID | null       # 操作对象 ID
  detail        json | null       # 操作详情，如 {"old_label": "A", "new_label": "B"}
  ip            string | null     # 客户端 IP
  created_at    timestamp DEFAULT NOW()
```

`detail` 用 JSON 存储差异化信息，不同 action 写入内容不同。

---

## 3. 操作类型与记录内容

| action | 触发时机 | detail 示例 |
|--------|---------|------------|
| `auth.login` | 登录成功 | `{"ip": "1.2.3.4"}` |
| `auth.login_failed` | 登录失败 | `{"ip": "1.2.3.4", "reason": "wrong_password"}` |
| `project.create` | 创建项目 | `{"name": "评论分类", "labels": ["A","B","C"]}` |
| `project.import` | 导入数据 | `{"dataset_id": "xxx", "sample_count": 10000}` |
| `project.start` | 启动标注 | — |
| `project.complete` | 项目完成 | `{"total_rounds": 5, "coverage": 0.92}` |
| `project.config_update` | 修改配置 | `{"changes": {"zeta": {"old": 0.1, "new": 0.05}}}` |
| `member.add` | 添加成员 | `{"user_id": "xxx", "role": "ANNOTATOR"}` |
| `member.remove` | 移除成员 | `{"user_id": "xxx", "role": "ANNOTATOR"}` |
| `task.grab` | 抢单 | `{"anno_id": "xxx"}` |
| `task.submit` | 提交标注 | `{"anno_id": "xxx", "label": "正面"}` |
| `task.release` | 放弃任务 | `{"anno_id": "xxx"}` |
| `review.approve` | 审阅通过 | `{"anno_id": "xxx"}` |
| `review.update` | 审阅修改标签 | `{"anno_id": "xxx", "old": "负面", "new": "正面"}` |
| `review.reject` | 审阅打回 | `{"anno_id": "xxx"}` |
| `adjudicate` | 管理员裁定 | `{"anno_id": "xxx", "labels": ["正面","负面"], "final": "正面"}` |
| `export` | 导出数据 | `{"format": "csv", "sample_count": 9500}` |
| `model.train` | 模型训练 | `{"round": 3, "accuracy": 0.85}` |
| `model.predict` | 模型推理 | `{"round": 3, "predicted_count": 200}` |
| `role.create` | 创建自定义角色 | `{"role_name": "auditor"}` |
| `role.update` | 修改角色权限 | `{"role_name": "auditor", "added": ["export"]}` |
| `user.create` | 创建用户 | `{"target_user": "zhangsan", "roles": ["annotator"]}` |
| `user.disable` | 禁用用户 | `{"target_user": "zhangsan"}` |

---

## 4. 写入方式

```
业务操作完成
  → 构造 audit 记录
  → INSERT INTO audit_logs（异步，不阻塞业务响应）
  → 写入失败不影响业务（降级：打应用日志告警）
```

---

## 5. 检索与界面

### 5.1 筛选条件

```
┌──────────────────────────────────────────────────────┐
│  审计日志                                             │
│                                                      │
│  用户: [全部 ▾]  项目: [全部 ▾]  操作: [全部 ▾]       │
│  时间: [2026-06-01] ~ [2026-06-26]                   │
│  [搜索]                                              │
├──────────────────────────────────────────────────────┤
│                                                      │
│  时间          用户    项目      操作         详情     │
│  ─────────────────────────────────────────────────── │
│  06-26 14:32  张三   评论分类   task.submit  正面     │
│  06-26 14:30  李四   评论分类   task.grab    —        │
│  06-26 14:28  管理员  —        project.start 评论分类 │
│  06-26 14:25  管理员  —        member.add   张三→标注  │
│  06-26 10:00  张三   —         auth.login   1.2.3.4  │
│                              ...                     │
│                                        第 1/23 页    │
└──────────────────────────────────────────────────────┘
```

### 5.2 详情展开

点击某条记录展开 `detail` JSON：

```
┌ 详情 ──────────────────────────────────┐
│ 操作: review.update                     │
│ 对象: anno_id = a1                      │
│ 变更: 标签 "负面" → "正面"              │
│ 用户: 管理员                             │
│ 时间: 2026-06-26 14:32:15               │
│ IP: 10.0.1.5                            │
└────────────────────────────────────────┘
```

### 5.3 权限

只有管理员可查看审计日志。普通用户看不到此入口。

---

## 6. 保留策略

- 默认保留 1 年
- 管理员可配置保留时长
- 超过保留期的日志可归档到外部存储（如对象存储），或直接清理
- 清理为物理删除（审计日志本身不受不可删原则约束，但清理操作本身也记一条 `audit.purge` 日志）
