# 模型管理器

---

## 定位

隐藏模型的加载和卸载细节。调用方只需 `predict(model_id, embeddings)`，管理器负责内存控制。

---

## 对外接口

```python
class ModelManager:
    def predict(self, model_id: str, embeddings: List[List[float]]) -> List[Prediction]:
        """用 model_id 对应的模型做批量推理"""
        pass
```

调用方不感知模型是否已在内存——管理器自动加载、缓存、淘汰。

---

## 配置项

| 配置 | 默认值 | 说明 |
|------|--------|------|
| max_online_models | 3 | 同时驻留内存的最大模型数 |
| max_memory_mb | 2048 | 模型占用内存上限（MB） |
| max_idle_seconds | 600 | 模型空闲超过此时长自动卸载（10 分钟） |

管理员可在系统设置中调整。

---

## 缓存策略

```
predict(model_id, embeddings)
  │
  ├─ 模型已在内存 → 直接推理，更新最后访问时间
  │
  └─ 模型不在内存
       │
       ├─ 当前在线数 < max_online_models → 加载模型到内存
       │
       └─ 当前在线数 ≥ max_online_models
            → 淘汰最久未使用的模型（LRU）
            → 加载新模型
```

淘汰时若模型权重有未持久化的变更，不适用（我们的模型只读，无此问题）。

---

## 空闲回收

后台定时任务，每 `max_idle_seconds / 2` 扫描一次：

```
for model in online_models:
    if now - model.last_access > max_idle_seconds:
        卸载模型，释放内存
```

---

## 内存保护

加载前预估模型文件大小。若加载后总内存超过 `max_memory_mb`：

- 先尝试卸载空闲模型
- 仍不足 → 拒绝加载，返回错误码 MD-008（GPU 资源不足）

---

## 预测流程

```
1. 加载模型权重 → PyTorch model
2. embeddings → Tensor
3. model.eval()
4. with torch.no_grad(): outputs = model(tensor)
5. softmax(outputs) → 取 argmax(label) 和 max(confidence)
6. 返回 [{anno_id, label, confidence}, ...]
```

GPU 可用时自动使用 GPU，否则降级 CPU。
