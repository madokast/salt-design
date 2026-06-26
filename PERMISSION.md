# 权限模型设计 (RBAC + Project-scoped)

---

## 1. 模型概述

两层权限控制：

| 层级 | 作用域 | 决定什么 |
|------|--------|---------|
| 系统角色 (SystemRole) | 全局 | 你能打开哪些系统模块（API 白名单） |
| 项目角色 (ProjectRole) | 单个项目内 | 你在这个具体项目里能做什么 |

一个用户 = 若干系统角色的集合 + 若干项目成员关系的集合。

---

## 2. 实体定义

### 2.1 系统角色 (SystemRole)

```
SystemRole:
  id: UUID
  name: string              # 角色名，如 "annotator"
  display_name: string      # 展示名，如 "标注专家"
  description: string
  is_builtin: boolean       # 是否内置角色（内置角色不可删除）
  api_permissions: string[] # API 白名单，如 ["tasks:grab", "tasks:submit"]
  created_at: timestamp
```

### 2.2 用户 (User)

```
User:
  id: UUID
  username: string
  password_hash: string
  display_name: string
  system_roles: UUID[]      # 拥有的系统角色 ID 列表
  is_active: boolean
  created_at: timestamp
```

### 2.3 项目 (Project)

```
Project:
  id: UUID
  visibility: enum          # PUBLIC | PRIVATE
  ...
```

### 2.4 项目成员 (ProjectMember)

```
ProjectMember:
  id: UUID
  project_id: UUID
  user_id: UUID
  project_role: enum         # OWNER | MANAGER | ANNOTATOR | REVIEWER | OBSERVER
  joined_at: timestamp
```

### 2.5 项目角色（内置枚举）

项目内角色是**固定枚举**，不可自定义：

| project_role | 权限 |
|-------------|------|
| OWNER | 项目创建者，拥有全部权限（删项目、改配置、分配成员） |
| MANAGER | 管理权限（分配成员、导入数据、启动/暂停、导出） |
| ANNOTATOR | 标注（抢单、提交、放弃） |
| REVIEWER | 审阅（浏览本轮标注、修改标签、通过/打回） |
| OBSERVER | 只读（浏览项目、查看进度，不能操作） |

### 2.6 系统角色与项目角色的关系

系统角色 = 你**有没有资格**做某类事；项目角色 = 你在**这个项目里**能不能做。

例如：
- 用户甲没有任何系统角色 → 登录后什么都看不到
- 用户甲有 `annotator` 系统角色 + 在项目 P1 是 ANNOTATOR → 可以在 P1 标注
- 用户甲只在项目 P1 是 OBSERVER → 即使有 `annotator` 系统角色，在 P1 也只能看

```
能标注 = 拥有 annotator 系统角色  AND  在目标项目中 project_role ∈ {ANNOTATOR, MANAGER, OWNER}
```

---

## 3. 内置系统角色

| 系统角色 | API 白名单（模块级） | 说明 |
|---------|---------------------|------|
| **observer** | `projects:list`, `projects:view`, `stats:*` | 只读浏览（新用户默认） |
| **annotator** | observer + `tasks:grab`, `tasks:submit`, `tasks:release`, `tasks:snatch` | 标注能力 |
| **reviewer** | observer + `review:view`, `review:update`, `review:approve` | 审阅能力 |
| **project_manager** | observer + `projects:create`, `projects:import`, `projects:export`, `members:manage`, `projects:start` | 项目管理 |
| **admin** | `*`（全部） | 超级管理员：创建角色、管理用户、系统配置 |

### 权限模块命名约定

```
{module}:{action}

projects:list       # 项目列表
projects:view       # 查看单个项目
projects:create     # 创建项目
projects:delete     # 删除项目
projects:import     # 导入数据
projects:export     # 导出结果
projects:start      # 启动标注
projects:pause      # 暂停

tasks:grab          # 抢单
tasks:submit        # 提交标注
tasks:release       # 放弃任务
tasks:snatch        # 抢夺超时任务

review:view         # 查看本轮标注（审阅界面）
review:update       # 修改标签
review:approve      # 一键通过
review:reject       # 打回

members:manage      # 管理项目成员
members:view        # 查看项目成员

users:manage        # 管理用户
users:view          # 查看用户列表

roles:manage        # 管理自定义角色（管理员专属）
roles:view          # 查看角色列表

stats:view          # 查看统计
```

---

## 4. API 鉴权流程

```
请求进入
  │
  ▼
1. Token 校验 → 提取 user_id
  │
  ▼
2. 系统角色检查
   → 查 user.system_roles
   → 取所有角色的 api_permissions 并集
   → 当前请求的 module:action 是否在并集中？
   → 不在 → 403 Forbidden
   → 在   → 继续
  │
  ▼
3. 项目可见性检查（如果请求涉及具体 project_id）
   → project.visibility == PUBLIC？
     → 是 → 至少拥有隐式 OBSERVER 权限，继续
     → 否 → 查 ProjectMember
       → 无记录 → 404 Not Found（不暴露项目存在）
       → 有记录 → 继续
  │
  ▼
4. 项目内角色检查（如果请求涉及修改操作）
   → 查 ProjectMember.project_role
   → 当前操作所需的最小 project_role 是否满足？
     → 不满足 → 403 Forbidden
     → 满足 → 继续
  │
  ▼
5. 执行业务逻辑
```

### 操作所需 project_role 对照表

| 操作 | 所需最小 project_role |
|------|----------------------|
| 查看项目详情、统计 | 无（PUBLIC）或 OBSERVER（PRIVATE） |
| 抢单、标注、提交、放弃 | ANNOTATOR |
| 审阅（浏览/修改/通过/打回） | REVIEWER |
| 导入数据、启动/暂停、导出 | MANAGER |
| 分配成员、修改项目配置 | MANAGER |
| 删除项目 | OWNER |

---

## 5. 自定义系统角色

管理员可以创建/编辑/删除自定义系统角色（内置角色不可删除）。

### 创建自定义角色流程

```
管理员进入角色管理页面
  → 点击"新建角色"
  → 输入名称、描述
  → 勾选 API 权限（按模块分组展示，支持全选/反选）
  → 保存
```

### 设计建议

- 基于内置角色**复制后微调**，而非从空白开始
- 例如：管理员想创建一个"只能导出不能导入"的角色 → 复制 project_manager → 去掉 `projects:import` → 保存为 "exporter_only"

---

## 6. 用户注册与默认权限

```
新用户注册
  → system_roles = [observer]  （默认）
  → 无任何 project_membership
  → 只能看到 PUBLIC 项目列表，不能做任何操作
  → 管理员可为该用户：
      1. 分配更多系统角色（如 annotator）
      2. 将用户拉入某个 PRIVATE 项目的成员列表，指定 project_role
```

---

## 7. 典型场景

### 场景 1：标注专家甲在不同项目中的不同角色

```
甲的系统角色: [observer, annotator, reviewer]

项目 P1 (PUBLIC):   甲的 project_role = ANNOTATOR  → 能标注
项目 P2 (PRIVATE):  甲的 project_role = REVIEWER   → 能审阅，不能标注
项目 P3 (PRIVATE):  甲未被添加                     → 看不到
项目 P4 (PUBLIC):   甲未被添加为成员                → 能看到项目，但不能操作（隐式 OBSERVER）
```

### 场景 2：管理员创建"培训观察员"角色

```
复制 observer 角色
  → 额外勾选 tasks:grab, tasks:submit
  → 去掉 tasks:snatch（不能抢夺）
  → 保存为 "trainee_annotator"
  → 分配给新入职的标注专家
```

### 场景 3：外部审计

```
创建 "auditor" 角色
  → 勾选 projects:list, projects:view, stats:view, members:view
  → 不勾选任何写操作
  → 分配给外部审计账号
  → 该账号只能查看 PUBLIC 项目的统计信息
```

---

## 8. 数据库表关系概览

```
┌──────────┐     ┌──────────────────┐     ┌───────────────────┐
│   User   │────→│ user_system_roles│←────│   SystemRole      │
│          │     │  user_id         │     │   id               │
│   id     │     │  role_id         │     │   name             │
│   name   │     └──────────────────┘     │   api_permissions  │
└────┬─────┘                              │   is_builtin       │
     │                                    └───────────────────┘
     │
     │    ┌──────────────────┐
     └───→│  ProjectMember   │
          │  project_id      │
          │  user_id         │
          │  project_role    │──── 枚举: OWNER|MANAGER|ANNOTATOR|REVIEWER|OBSERVER
          └────────┬─────────┘
                   │
                   │    ┌──────────┐
                   └───→│ Project  │
                        │ id       │
                        │ visibility│── PUBLIC | PRIVATE
                        └──────────┘
```

---

## 9. 开源框架参考

| 框架 | 语言 | 特点 |
|------|------|------|
| [Casbin](https://casbin.org) | Go / 多语言 | 最通用，支持 RBAC/ABAC/ACL，策略与代码分离 |
| [Oso](https://www.osohq.com) | Python/Rust/Go/Node.js | 策略用 Polar 语言编写，表达能力极强 |
| [SpiceDB](https://authzed.com) | Go (gRPC) | Google Zanzibar 实现，适合大规模细粒度权限 |
| [Permify](https://permify.co) | Go | 类似 SpiceDB，DSL 更友好 |
| [Django Guardian](https://django-guardian.readthedocs.io) | Python | Django 生态，对象级权限 |
| [Ladon](https://github.com/ory/ladon) | Go | ORY 生态，适合做策略决策点 |
| [accesscontrol](https://github.com/onury/accesscontrol) | Node.js | Node.js 生态的 RBAC/ABAC 库 |

### 选型参考

- **语言无关，轻量自建**：本方案实体关系简单，几张表 + 中间件即可，不需要引入外部依赖
- **想用现成的**：Casbin 最成熟（30k+ stars），支持 Python/Go/Node.js/Java 等所有主流语言，策略文件可热更新
- **大规模细粒度场景**：SpiceDB 或 Permify，支持全局一致的关系型权限
