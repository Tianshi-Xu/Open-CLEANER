# Plan B：基于 IS 的回滚决策机制

> **所在分支**：`dev2`  
> **核心目标**：解决多轮对话 GRPO 训练中，工具调用失败触发回滚时产生的 off-policy 问题。

---

## 1. 背景与问题

标准 rollout 流程中，模型在工具调用出错后会触发回滚（rollback）：  
重新生成一段新的 reasoning + tool call 来替换掉原本出错的那段。

这带来了一个 **off-policy 问题**：
- 新 response（R2）是在"含错误反馈"的上下文（`error_ctx`）下采样得到的
- 但训练时它出现在"无错误"的轨迹中，上下文对不上
- 即：π_b(R2 | with_error_ctx) ≠ π_old(R2 | no_error_ctx)，IS 比值偏离 1

**dev 分支（Plan C）** 的做法：始终替换，用 token-level IS 权重在训练阶段做事后校正，并用 1-容忍机制减少触发频率。

**dev2 分支（Plan B）** 的做法：在回滚时直接计算 IS ratio，由它来决定是替换还是追加，从根源上避免不一致。

---

## 2. Plan B 核心逻辑

### 2.1 IS ratio 定义

$$\text{IS ratio} = \frac{\pi_{\text{old}}(R_2 \mid \text{no\_error\_ctx})}{\pi_b(R_2 \mid \text{with\_error\_ctx})} = \exp\!\left(\sum_t \log\pi_{\text{old}}(t \mid \text{no\_error\_ctx}) - \sum_t \log\pi_b(t \mid \text{with\_error\_ctx})\right)$$

- **no\_error\_ctx**：old\_reasoning 之前的所有 token（checkpoint prompt 去掉 old response）
- **with\_error\_ctx**：包含 old\_response + 错误反馈的完整上下文（R2 实际采样时的上下文）
- 两个 policy 使用**同一套模型权重**，只是上下文不同
- π_b 直接使用 sglang 生成时返回的 logprobs（`calculate_log_probs=True`），无额外开销
- π_old 需调用一次额外的 `score()` 推理

### 2.2 决策规则

| IS ratio | 决策 | 轨迹形状 |
|----------|------|----------|
| **< threshold** | **APPEND**（追加） | `[original \| old_resp(mask=1) \| error(mask=0) \| new_resp(mask=1)]` — 完全在线策略 |
| **≥ threshold** | **REPLACE**（替换） | `[original \| new_resp(mask=1)]` — 去掉 error 上下文，接近无错误分布 |

- IS 低 → R2 强依赖错误上下文 → 替换会造成大 off-policy gap → 追加，保留完整轨迹
- IS 高（≈1）→ R2 与上下文几乎无关 → 替换安全，轨迹更干净

### 2.3 与 Plan C 的对比

| | Plan C（dev 分支） | Plan B（dev2 分支） |
|--|---|---|
| 触发时机 | 1-容忍（第二次错才回滚） | 首次错误立即回滚 |
| 轨迹类型 | 始终替换，训练时 IS 校正 | IS 低→追加（on-policy），IS 高→替换 |
| 低 IS 样本 | gradient 被 IS 权重降权 | 保留完整轨迹，gradient 正常 |
| 高 IS 样本 | 同替换 | 替换，干净轨迹 |
| 额外推理开销 | 无 | 每次回滚+1 次 score() 调用 |

---

## 3. 修改文件清单

### 3.1 `verl/verl/workers/rollout/sglang_rollout/async_sglang_server.py`

**新增 `SGLangHttpServer.score()` 方法**（在 `generate()` 之后）：

```python
async def score(self, prompt_ids, response_ids, request_id) -> list[float]:
```

- 将 `prompt_ids + response_ids` 拼接为完整输入
- 通过 `GenerateReqInput(logprob_start_len=len(prompt_ids), return_logprob=True)` 调用 SGLang 内部 API
- 从 `meta_info["input_token_logprobs"]` 提取每个 response token 的 logprob
- `max_new_tokens=1` 仅为触发 prefill，生成结果不使用

> ⚠️ **已知问题**：`input_token_logprobs` 中首个 token 可能为 `None`，需要在 `entry[0]` 处做 `None` 判断（当前会 fallback 到 REPLACE）。

### 3.2 `verl/verl/experimental/agent_loop/agent_loop.py`

**新增 `AsyncLLMServerManager.score()` 方法**（在 `generate()` 之后）：

```python
async def score(self, request_id, *, prompt_ids, response_ids) -> list[float]:
```

- 通过 sticky session（与 `generate` 相同的 server）代理 `score()` 调用
- 复用 KV cache，减少 prefill 开销

### 3.3 `verl/verl/experimental/agent_loop/tool_agent_loop.py`

**（a）`RollbackManager.__init__`**：新增 `is_threshold: Optional[float] = None` 参数。

**（b）`ToolAgentLoopWorker.__init__`**：从 config 读取 `rollback_is_threshold` 并传入 `RollbackManager`。

**（c）`_handle_processing_tools_state`**：移除 1-容忍机制，每次工具错误直接触发回滚。

**（d）`_handle_rollback`**：新增 Plan B 分支（在 Step 4 之后）：

```
if rollback_manager.is_threshold is not None:
    use_append = await _decide_rollback_append(...)
    if use_append:
        return _handle_processing_tools_state(...)  # 直接继续，不恢复 checkpoint
    else:
        rollback_replace_count += 1
        # 走原有替换路径
```

**（e）`_decide_rollback_append()`**：新增 async 方法，实现 IS 计算核心逻辑：
- 构建 `no_error_ctx`、`with_error_ctx`
- 调用 `server_manager.score()` 获取 π_old
- 若 `calculate_log_probs=True`，直接使用生成 logprobs 作为 π_b；否则再次调用 `score()`
- 计算 IS ratio，返回 `True`（append）或 `False`（replace）
- 包含详细 debug 打印（token 级别 logprob 明细）

**（f）`_overwrite_last_assistant_turn`**：简化为始终全量替换（移除 similarity 检查的 tool-call-only 路径）。

**（g）`AgentData.__init__`**：新增统计字段：
```python
self.rollback_append_count = 0
self.rollback_replace_count = 0
```
（移除了旧的 `rollback_full_turn_count` / `rollback_tool_call_only_count`）

### 3.4 `verl/verl/workers/config/rollout.py`

`MultiTurnConfig` 新增字段：

```python
rollback_is_threshold: Optional[float] = None
```

### 3.5 `verl/verl/trainer/ppo/ray_trainer.py`

每个训练 step 新增 WandB 指标：

| metric key | 含义 |
|---|---|
| `rollback/planb_append_count` | 本 step 走追加路径的次数 |
| `rollback/planb_replace_count` | 本 step 走替换路径的次数 |
| `rollback/planb_append_rate` | 追加率 = append / (append + replace) |

### 3.6 `recipe/cleaner/qwen3_4b_cleaner.sh`

```bash
actor_rollout_ref.rollout.calculate_log_probs=True          # 为 π_b 提供生成 logprobs
+actor_rollout_ref.rollout.multi_turn.rollback_is_threshold=0.5
```

---

## 4. 数据流示意

```
工具调用失败
    │
    ▼
_handle_rollback()
    │
    ├─ Step 1-2: 追加错误反馈到 prompt_ids（mask=0）
    ├─ Step 3:   重新生成 new_response（sglang）
    │
    ├─ [Plan B] _decide_rollback_append()
    │       ├─ no_error_ctx = checkpoint.prompt_ids[:-len(old_response)]
    │       ├─ with_error_ctx = agent_data.prompt_ids[:-len(new_response)]
    │       ├─ score(no_error_ctx, new_response) → π_old log probs
    │       ├─ generation logprobs → π_b log probs
    │       ├─ IS = exp(Σπ_old - Σπ_b)
    │       └─ IS < threshold → True（append），else False（replace）
    │
    ├─ True  → APPEND：直接继续 _handle_processing_tools_state
    │          轨迹：[original | old_resp | error | new_resp]（完全 on-policy）
    │
    └─ False → REPLACE：_overwrite_last_assistant_turn → restore_checkpoint
               轨迹：[original | new_resp]（无 error 上下文）
```

---

## 5. 已知问题 / TODO

1. **`score()` 中 `input_token_logprobs` 首 token 可能为 `None`**  
   需在 `async_sglang_server.py` 的 `score()` 中改为：
   ```python
   return [float(entry[0]) if entry[0] is not None else 0.0 for entry in input_token_logprobs]
   ```

2. **`score()` 调用增加延迟**  
   每次回滚额外消耗 1 次 prefill（有 KV cache 加速）。若 `calculate_log_probs=False` 则需 2 次。强烈建议启用 `calculate_log_probs=True`。

3. **阈值 `rollback_is_threshold=0.5` 待调优**  
   建议先跑 debug 模式收集真实 IS ratio 分布，再决定合适阈值。

4. **debug 打印未来需关闭**  
   `_decide_rollback_append()` 中有大量 `print` 语句，正式训练前应改为 `logger.debug` 并通过环境变量控制。
