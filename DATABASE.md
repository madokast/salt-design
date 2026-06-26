# 数据库设计

两个独立数据库，职责分离。

---

## 主库 (meta)

结构数据，体积小（MB 级），需高频备份。

| 表 | 说明 |
|----|------|
| users | 用户 |
| system_roles | 系统角色 + API 白名单 |
| user_system_roles | 用户-角色关联 |
| projects | 项目配置、状态、算法选择 |
| project_members | 项目成员与项目角色 |
| datasets | 批次信息 |
| models | 模型版本记录 |
| project_rounds | 轮次指标 |
| project_round_vis | 可视化点数据 |
| project_events | 用户事件日志 |
| audit_logs | 审计日志 |
| notifications | 通知 |
| labeling_functions | LF 定义 |
| lrr_rules | LRR 规则定义 |
| anno_types | 标记类型插件 |
| encoders | 编码器注册 |
| algorithm_tasks | 算法任务状态 |
| license_info | 授权信息 |

---

## 标注库 (data)

项目相关的大数据，随标注规模线性增长（GB 级），独立备份。

每个项目动态创建：

| 表 | 说明 |
|----|------|
| `anno_content_{pid}` | 标记对象数据（由 anno_type 定义字段） |
| `anno_task_{pid}` | 标记任务（抢夺、提交、乐观锁） |

---

## 分离的好处

| 主库 | 标注库 |
|------|--------|
| 高频备份（每小时） | 低频备份（每天） |
| 必须高可用 | 可接受短暂不可用 |
| 不需要分片 | 可独立扩容/分片 |
| 删项目只需 DROP 标注库的表 | 历史标注数据可归档 |

---

## 连接

后端连两个库。标注库操作通过 project_id 动态路由到对应表。事务不跨库。
