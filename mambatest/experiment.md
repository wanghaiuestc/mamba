# mat_eval 与 check_mat_eval_equiv 说明

这份文档说明 `mambatest/mat_eval.py` 和 `mambatest/check_mat_eval_equiv.py` 的作用、输入输出、比较逻辑，以及如何理解脚本里的公式标注。

## 1. 文件职责

### `mambatest/mat_eval.py`

这是一个 Mamba 推理的纯 PyTorch 参考实现，目标是把官方 `transformers` 里的 Mamba 前向过程拆开成更容易阅读的形式，并且保留逐步 trace：

- `mamba_mixer_mat_eval`：对单个 mixer 做参考推理
- `mamba_block_mat_eval`：对单个 block 做参考推理
- `mamba_model_mat_eval`：对 backbone 做参考推理
- `mamba_lm_mat_eval`：对带语言模型头的完整模型做参考推理

它的重点不是提升速度，而是对照公式、方便调试、方便验证和追踪中间变量。

### `mambatest/check_mat_eval_equiv.py`

这是一个一致性检查脚本，用来验证：

- `mat_eval.py` 的参考实现
- 官方 `transformers` 的 `model.forward`
- 官方 `mixer.slow_forward`

三者在数值上是否一致，误差是否低于阈值。

## 2. 核心比较内容

脚本会比较这些结果：

1. 整个模型的最后隐藏状态
2. 整个模型的 logits
3. 第 0 层 mixer 的输出
4. 第 0 层的 SSM state
5. 第 0 层的 convolution state
6. decode 阶段的 logits

如果开启 `return_step_traces=True`，还会打印第一步 trace 的形状，用来确认每个中间量的维度是否符合预期。

## 3. 运行流程

脚本的主流程大致如下：

1. 读取参数：`--threshold`、`--seed`、`--model-id`
2. 根据 `model_id` 构建随机模型或加载预训练模型
3. 关闭可选的 Mamba kernel，强制走 Python/slow path，保证对照基准一致（fast path 数学表达一致）
4. 用同一组输入分别执行官方 forward 和 `mat_eval` 前向
5. 计算误差并打印结果
6. 再做一次 decode step 的对照

### 3.1 运行实例

在仓库根目录执行：

1. 随机小模型（不传 model_id，适合本地快速检查）

	python mambatest/check_mat_eval_equiv.py --threshold 1e-4 --seed 1234

2. 指定预训练模型（例如 state-spaces/mamba-130m-hf）
    export HF_ENDPOINT=https://hf-mirror.com
	python mambatest/check_mat_eval_equiv.py --model-id state-spaces/mamba-130m-hf --threshold 1e-4 --seed 1234


运行后重点看输出里的这些条目是否通过阈值：

- full_model_hidden
- full_model_logits
- layer0_mixer_output
- layer0_ssm_state
- layer0_conv_state
- decode_logits

## 4. `mat_eval.py` 里的公式对应关系

现在 `mat_eval.py` 已把“参数准备”和“单步推理”拆成两个函数，便于对照公式。

### 4.1 两个函数

```python
def _prepare_discrete_ssm_params(
	mixer: MambaMixer,
	hidden_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
	"""
	Build per-token SSM parameters.

	Returns (A, dt_t, B_pre, C_t, A_t, B_t) where:
	  - A: continuous-time diagonal coefficients, shape [intermediate, d_state]
	  - dt_t: discretized step size, shape [batch, intermediate, seq]
	  - B_pre: pre-discretization B from projection, shape [batch, seq, d_state]
	  - C_t: per-token C from projection, shape [batch, seq, d_state]
	  - A_t: discretized transition coefficients, shape [batch, intermediate, seq, d_state]
	  - B_t: discretized input coefficients, shape [batch, intermediate, seq, d_state]
	"""

	x_t = hidden_states.transpose(1, 2)
	W_x = mixer.x_proj.weight
	b_x = mixer.x_proj.bias
	# [tau_t, B, C_t] = W_x * x_t (+ b_x)
	ssm_parameters = torch.matmul(x_t, W_x.transpose(0, 1))
	if b_x is not None:
		ssm_parameters = ssm_parameters + b_x

	tau_t, B, C_t = torch.split(
		ssm_parameters,
		[mixer.time_step_rank, mixer.ssm_state_size, mixer.ssm_state_size],
		dim=-1,
	)

	W_dt = mixer.dt_proj.weight
	b_dt = mixer.dt_proj.bias
	# dt_t pre-activation = W_dt * tau_t (+ b_dt)
	dt_t_pre_activation = torch.matmul(tau_t, W_dt.transpose(0, 1))
	if b_dt is not None:
		dt_t_pre_activation = dt_t_pre_activation + b_dt

	discrete_time_step = torch.nn.functional.softplus(dt_t_pre_activation).transpose(1, 2)  # dt_t = softplus(W_dt * tau_t)

	# A = -exp(A_log)
	A = -torch.exp(mixer.A_log.float())
	# A_t = exp(A * dt_t)
	discrete_A = torch.exp(A[None, :, None, :] * discrete_time_step[:, :, :, None])
	# B_t = dt_t * B
	discrete_B = discrete_time_step[:, :, :, None] * B[:, None, :, :].float()
	return A, discrete_time_step, B, C_t, discrete_A, discrete_B


def _mamba_ssm_step(
	A_t: torch.Tensor,
	B_t: torch.Tensor,
	C_t: torch.Tensor,
	h_t: torch.Tensor,
	x_t: torch.Tensor,
	D: torch.Tensor,
	gate_t: torch.Tensor,
	activation,
	output_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
	"""
	Symbol naming in this function header:
		A_t: per-step transition coefficient (passed from caller's At)
		B_t: per-step input coefficient (passed from caller's Bt)

	Execute one SSM step:
		h_next = A_t * h_t + B_t * x_t
		y_t = h_next * C_t + D * x_t
		y_t = y_t * act(gate_t)
	"""

	h_next = A_t * h_t + B_t * x_t[:, :, None].float()
	y_t = torch.matmul(h_next.to(output_dtype), C_t.unsqueeze(-1))[:, :, 0]
	y_t = y_t + (x_t * D[None, :])
	y_t = y_t * activation(gate_t)
	return h_next, y_t
```

### 4.2 关键对应关系（更新后）

1. 输入投影（显式矩阵乘）

$$
[\tau_t, B_{raw,t}, C_t] = W_x x_t + b_x
$$

2. 时间步投影（显式矩阵乘）

$$
dt_t = \mathrm{softplus}(W_{dt}\tau_t + b_{dt})
$$

3. 离散化

$$
A = -\exp(A_{log}), \quad A_t = \exp(A \cdot dt_t), \quad B_t = dt_t \cdot B_{raw,t}
$$

4. 单步更新

$$
h_t = A_t * h_{t-1} + B_t * x_t
$$

$$
y_t = h_t * C_t + D * x_t
$$

5. 命名层级

- `discrete_A`：所有时刻堆叠后的 $A_t$
- `At`：`discrete_A[:, :, step_idx, :]`
- `_mamba_ssm_step` 形参 `A_t`：调用时传入 `At`
- `discrete_B`、`Bt`、`B_t` 同理

## 5. 输入和输出

### 输入

脚本支持两种模式：

- 不传 `--model-id`：构建一个小型随机 Mamba 模型，适合本地快速验证
- 传 `--model-id`：加载指定的 Hugging Face 模型

### 输出

脚本会打印类似下面的信息：

- `full_model_hidden`
- `full_model_logits`
- `layer0_mixer_output`
- `layer0_ssm_state`
- `layer0_conv_state`
- `decode_logits`

每一项都会显示：

- `max_abs_error`
- `mean_abs_error`
- 是否通过阈值

## 6. 常见注意点

1. `mixer.xxx` 里所有东西都是权重
2. `time_step`、`B`、`C`、`discrete_A`、`discrete_B`、`ssm_state` 都是运行时中间量，不是固定参数。
3. 代码里的 `*` 一般表示逐元素乘，不等于矩阵乘法。
4. `torch.matmul` 才对应这里的内积/矩阵乘部分。
5. `disable_optional_mamba_kernels()` 的作用是关掉可选 kernel，让比较尽量落在同一条 slow path 上。

