# Salt

基于论文 [SALT: Selective Annotation for Labeling and Training](./salt.md) 构建的企业级数据标注平台。

**核心价值：用最少的专家标注量，达到有理论保证的标注精度。**

---

## 工作方式

```
1. 创建项目，定义标签集，上传待标注数据，选择算法
2. 分配标注专家
3. 系统自动迭代：
     模型训练 → 覆盖分析 → 选出最有价值的样本 → 专家在线抢单标注 → 循环
4. 覆盖率达标，自动停止，导出标注结果
```

- 专家只需在线抢单标注，不需关心选哪些
- 支持机器辅助推荐、多人交叉验证、管理员裁定
- 弱监督规则引擎可大幅减少所需人工量

---

## 可视化设计页

→ [index.html](./index.html) &nbsp;（[GitHub Pages](https://madokast.github.io/salt-design/)）

---

## 详细设计文档

所有设计文档在 [docs/](./docs/) 目录下。

### 核心设计

| 文档 | 内容 |
|------|------|
| [DESIGN.md](./docs/DESIGN.md) | 系统架构、角色定义、项目生命周期、数据模型、数据导入导出、项目克隆 |
| [PERMISSION.md](./docs/PERMISSION.md) | 两层 RBAC 权限模型（系统角色 + 项目角色）、PUBLIC/PRIVATE 项目可见性、自定义角色 |
| [TERMS.md](./docs/TERMS.md) | 统一术语定义（项目→批次→标记对象→标记任务） |
| [DATABASE.md](./docs/DATABASE.md) | 双数据库设计（主库 + 标注库） |

### 标注流程

| 文档 | 内容 |
|------|------|
| [ANNO_LOOP.md](./docs/ANNO_LOOP.md) | 人机协作循环、共享任务池抢单、机器专家、置信度驱动动态人力分配、多人裁定、轮次指标仪表盘、覆盖可视化、弱监督效果看板、专家质量与效率统计 |
| [ANNO_CONTENT.md](./docs/ANNO_CONTENT.md) | 标记对象渲染系统、anno_type 插件（内置/自定义/GUI）、四种呈现视图（专家HTML/列表/模型/导出）、导入适配器与字段映射 |

### 算法与模型

| 文档 | 内容 |
|------|------|
| [SALT_PLUGIN.md](./docs/SALT_PLUGIN.md) | 算法组件接口（Selector/Trainer/Encoder/WeakSupervisor）、编码器管理、模型管理、算法注册与装配 |
| [algorithms/](./docs/algorithms/) | 算法总纲与各算法详细说明 |
| [salt.md](./salt.md) | SALT 论文原文（曲率自适应覆盖、选样策略、弱监督 LRR 规则框架） |

### 运营与运维

| 文档 | 内容 |
|------|------|
| [LICENSE.md](./docs/LICENSE.md) | Free/Pro 版本模型、license 签发 |
| [AUDIT.md](./docs/AUDIT.md) | 审计日志（谁在什么时候做了什么） |
| [LOGGING.md](./docs/LOGGING.md) | 开发日志（文件）+ 用户事件日志（数据库） |
| [ERRORS.md](./docs/ERRORS.md) | 三级错误处理（重试→降级→终止）、错误码设计 |
