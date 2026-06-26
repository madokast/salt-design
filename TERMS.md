# 术语定义

---

## 层级关系

```
项目 (Project)
  └─ 批次 (Dataset)          1:N，一次上传的数据
       └─ 标记对象 (Object)   1:N，一条待标注的样本
            └─ 标记任务 (Task) 1:N，一个人或机器对同一对象的标注工作
               source: HUMAN | MACHINE
```

---

## 术语表

| 术语 | 英文 | 表名 | 说明 |
|------|------|------|------|
| **项目** | Project | `projects` | 一个标注项目，包含标签集、配置、成员 |
| **批次** | Dataset | `datasets` | 项目中一次上传的数据集合，v1 每项目仅 1 个批次 |
| **标记对象** | Object | `anno_content_{pid}` | 一条待标注的样本，由 anno_type 定义字段结构，导入后只读 |
| **标记任务** | Task | `anno_task_{pid}` | 一个标注工作单元，由人或机器完成。一个标记对象可以有 1~N 个任务 |
| **标记对象表** | Content Table | `anno_content_{pid}` | 存储一个项目内所有标记对象的原始数据 |
| **标记任务表** | Task Table | `anno_task_{pid}` | 存储一个项目内所有标记任务的状态和结果 |
| **标记类型** | Annotation Type | `anno_types` | 定义标记对象的字段结构和渲染方式（插件） |
| **系统角色** | System Role | `system_roles` | 控制用户能访问哪些系统模块（API 白名单） |
| **项目角色** | Project Role | `project_members` | 控制用户在具体项目内的权限（OWNER/MANAGER/ANNOTATOR/REVIEWER/OBSERVER） |

---

## 实体内关系

```
Project 1──N Dataset 1──N Object 1──N Task
                                      │
                                   source: HUMAN | MACHINE
```

- 一个标记对象可以对应多个 Task（多个专家标注，或机器 + 专家）
- Task 是专家抢单的最小单位
- Object 是所有 Task 完成后的回写目标

---

## 任务状态

| 状态值 | 含义 |
|--------|------|
| 0 | 待抢（pending） |
| 1 | 抢占中（claimed） |
| 2 | 已提交（submitted） |
| 3 | 已裁定（adjudicated，仅用于标记对象级别的最终裁定） |

---

## 标记对象生命周期

```
导入 → 标记对象表 (final_label=NULL, label_status=0)
  → 生成标记任务
  → 所有任务完成，回写 final_label → label_status=2
```
