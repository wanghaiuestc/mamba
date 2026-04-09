from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from typing import Any

import torch
import huggingface_hub

if not hasattr(huggingface_hub, "is_offline_mode"):
    huggingface_hub.is_offline_mode = lambda: huggingface_hub.constants.HF_HUB_OFFLINE
dependency_versions_check = sys.modules.setdefault(
    "transformers.dependency_versions_check", types.ModuleType("dependency_versions_check")
)
if not hasattr(dependency_versions_check, "dep_version_check"):
    dependency_versions_check.dep_version_check = lambda *args, **kwargs: None

from transformers.models.mamba.modeling_mamba import (
    MambaBlock,
    MambaCache,
    MambaMixer,
    MambaModel,
)
from transformers.utils import ModelOutput


def _clone_trace_tensor(value: torch.Tensor) -> torch.Tensor:
    return value.detach().clone()


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


@dataclass
class MambaStepTrace:
    step: int
    conv_input_t: torch.Tensor
    conv_output_t: torch.Tensor
    gate_t: torch.Tensor
    dt_t: torch.Tensor
    A: torch.Tensor
    At: torch.Tensor
    B_t: torch.Tensor
    Bt: torch.Tensor
    C_t: torch.Tensor
    D: torch.Tensor
    prev_state: torch.Tensor
    next_state: torch.Tensor
    y_t: torch.Tensor


@dataclass
class MambaBlockMatTrace:
    residual: torch.Tensor
    normalized_hidden_states: torch.Tensor
    mixer_output: torch.Tensor
    block_output: torch.Tensor
    steps: list[MambaStepTrace]


@dataclass
class MambaMatEvalOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor | None = None
    cache_params: MambaCache | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    layer_traces: tuple[MambaBlockMatTrace, ...] | None = None


@dataclass
class MambaCausalLMMatEvalOutput(ModelOutput):
    logits: torch.FloatTensor | None = None
    cache_params: MambaCache | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    layer_traces: tuple[MambaBlockMatTrace, ...] | None = None


def mamba_mixer_mat_eval(
    mixer: MambaMixer,
    input_states: torch.Tensor,
    cache_params: MambaCache | None = None,
    cache_position: torch.LongTensor | None = None,
    attention_mask: torch.LongTensor | None = None,
    return_step_traces: bool = False,
) -> tuple[torch.Tensor, list[MambaStepTrace]]:
    """
    Pure PyTorch reference for Mamba eval.

    Symbol hierarchy used in this function:
        discrete_A: all time steps stacked, shape [batch, intermediate, seq, d_state]
        At: t-th slice from discrete_A, i.e. discrete_A[:, :, step_idx, :]
        A_t: parameter name in _mamba_ssm_step, called with A_t=At

        discrete_B: all time steps stacked, shape [batch, intermediate, seq, d_state]
        Bt: t-th slice from discrete_B, i.e. discrete_B[:, :, step_idx, :]
        B_t: parameter name in _mamba_ssm_step, called with B_t=Bt

    The structured state update implemented by the official slow path is:

            h_t = A_t * h_{t-1} + B_t * x_t
            y_t = h_t * C_t + D * x_t

    where for each intermediate channel the state vector is length `d_state`.
    This is equivalent to a diagonal state transition matrix `diag(A_t)` per channel,
    without materializing a dense matrix.

        Formula-to-code mapping:
            h_{t-1} -> prev_state (or ssm_state before update)
            h_t -> ssm_state (after update)
            dt_t -> discrete_time_step[:, :, step_idx]
            A -> A
            A_t -> At = discrete_A[:, :, step_idx, :]
            B (pre-discretization) -> B
            B_t (used in update equation) -> Bt = discrete_B[:, :, step_idx, :]
            x_t -> x_t = hidden_states[:, :, step_idx]
            C_t -> C_t = C[:, step_idx, :]
            D -> mixer.D
            y_t -> scan_output
    """

    batch_size, seq_len, _ = input_states.shape
    dtype = input_states.dtype
    device = input_states.device

    projected_states = mixer.in_proj(input_states).transpose(1, 2)  # [u_t, g_t] = W_in * x_t
    hidden_states, gate = projected_states.chunk(2, dim=1)  # u_t: content branch, g_t: gate branch

    if attention_mask is not None:
        hidden_states = hidden_states * attention_mask.unsqueeze(1)  # u_t <- m_t * u_t

    if cache_params is not None:
        ssm_state = cache_params.ssm_states[mixer.layer_idx].clone().to(device)  # h_0 / cached h_{t-1}
        if cache_position is None:
            raise ValueError("`cache_position` must be provided when `cache_params` is passed.")

        if cache_position.shape[0] == mixer.conv_kernel_size:
            conv_state = torch.nn.functional.pad(hidden_states, (mixer.conv_kernel_size - hidden_states.shape[-1], 0))  # conv buffer padding
            cache_params.update_conv_state(mixer.layer_idx, conv_state, cache_position)
            hidden_states = mixer.act(mixer.conv1d(hidden_states)[..., :seq_len])  # x_t = phi(conv1d(u_t))
        else:
            conv_state = cache_params.update_conv_state(mixer.layer_idx, hidden_states, cache_position)
            conv_state = conv_state.to(mixer.conv1d.weight.device)
            hidden_states = torch.sum(conv_state * mixer.conv1d.weight[:, 0, :], dim=-1)  # conv1d(u_t)
            if mixer.use_conv_bias:
                hidden_states = hidden_states + mixer.conv1d.bias  # conv1d(u_t) + b
            hidden_states = mixer.act(hidden_states).to(dtype).unsqueeze(-1)  # x_t = phi(conv1d(u_t))
    else:
        ssm_state = torch.zeros(
            (batch_size, mixer.intermediate_size, mixer.ssm_state_size),
            device=device,
            dtype=dtype,
        )
        hidden_states = mixer.act(mixer.conv1d(hidden_states)[..., :seq_len])  # x_t = phi(conv1d(u_t))

    if attention_mask is not None:
        hidden_states = hidden_states * attention_mask.unsqueeze(1)  # m_t * x_t

    A, discrete_time_step, B, C, discrete_A, discrete_B = _prepare_discrete_ssm_params(mixer, hidden_states)

    scan_outputs = []
    step_traces: list[MambaStepTrace] = []
    for step_idx in range(seq_len):
        prev_state = ssm_state  # h_{t-1}
        At = discrete_A[:, :, step_idx, :]  # A_t: take the t-th discrete transition coefficient
        Bt = discrete_B[:, :, step_idx, :]  # B_t : take the t-th discrete transition coefficient
        B_t = B[:, step_idx, :]  # B (before discretization)
        C_t = C[:, step_idx, :]  # C_t
        x_t = hidden_states[:, :, step_idx]  # x_t

        ssm_state, scan_output = _mamba_ssm_step(
            A_t=At,
            B_t=Bt,
            C_t=C_t,
            h_t=ssm_state,
            x_t=x_t,
            D=mixer.D,
            gate_t=gate[:, :, step_idx],
            activation=mixer.act,
            output_dtype=dtype,
        )
        scan_outputs.append(scan_output)

        if return_step_traces:
            step_traces.append(
                MambaStepTrace(
                    step=step_idx,
                    conv_input_t=_clone_trace_tensor(projected_states[:, : mixer.intermediate_size, step_idx]),
                    conv_output_t=_clone_trace_tensor(x_t),
                    gate_t=_clone_trace_tensor(gate[:, :, step_idx]),
                    dt_t=_clone_trace_tensor(discrete_time_step[:, :, step_idx]),
                    A=_clone_trace_tensor(A),
                    At=_clone_trace_tensor(At),
                    B_t=_clone_trace_tensor(B_t),
                    Bt=_clone_trace_tensor(Bt),
                    C_t=_clone_trace_tensor(C_t),
                    D=_clone_trace_tensor(mixer.D),
                    prev_state=_clone_trace_tensor(prev_state),
                    next_state=_clone_trace_tensor(ssm_state),
                    y_t=_clone_trace_tensor(scan_output),
                )
            )

    scan_output = torch.stack(scan_outputs, dim=-1)
    contextualized_states = mixer.out_proj(scan_output.transpose(1, 2))

    if cache_params is not None:
        cache_params.ssm_states[mixer.layer_idx].copy_(ssm_state)

    return contextualized_states, step_traces


def mamba_block_mat_eval(
    block: MambaBlock,
    hidden_states: torch.Tensor,
    cache_params: MambaCache | None = None,
    cache_position: torch.LongTensor | None = None,
    attention_mask: torch.LongTensor | None = None,
    return_step_traces: bool = False,
) -> tuple[torch.Tensor, MambaBlockMatTrace | None]:
    residual = hidden_states
    normalized_hidden_states = block.norm(hidden_states.to(dtype=block.norm.weight.dtype))
    if block.residual_in_fp32:
        residual = residual.to(torch.float32)

    mixer_output, step_traces = mamba_mixer_mat_eval(
        block.mixer,
        normalized_hidden_states,
        cache_params=cache_params,
        cache_position=cache_position,
        attention_mask=attention_mask,
        return_step_traces=return_step_traces,
    )
    block_output = residual + mixer_output

    trace = None
    if return_step_traces:
        trace = MambaBlockMatTrace(
            residual=_clone_trace_tensor(residual),
            normalized_hidden_states=_clone_trace_tensor(normalized_hidden_states),
            mixer_output=_clone_trace_tensor(mixer_output),
            block_output=_clone_trace_tensor(block_output),
            steps=step_traces,
        )
    return block_output, trace


def mamba_model_mat_eval(
    model: MambaModel,
    input_ids: torch.LongTensor | None = None,
    inputs_embeds: torch.Tensor | None = None,
    cache_params: MambaCache | None = None,
    use_cache: bool | None = None,
    output_hidden_states: bool | None = None,
    return_dict: bool = True,
    cache_position: torch.LongTensor | None = None,
    attention_mask: torch.LongTensor | None = None,
    return_step_traces: bool = False,
) -> MambaMatEvalOutput | tuple[Any, ...]:
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else model.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else (model.config.use_cache if not model.training else False)

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if inputs_embeds is None:
        inputs_embeds = model.embeddings(input_ids)

    if use_cache:
        if cache_params is None:
            cache_params = MambaCache(
                model.config, inputs_embeds.size(0), device=inputs_embeds.device, dtype=inputs_embeds.dtype
            )
            cache_position = torch.arange(0, model.config.conv_kernel, device=inputs_embeds.device)
        elif cache_position is None:
            raise ValueError("`cache_position` must be provided when `cache_params` is passed.")
    else:
        cache_params = None

    hidden_states = inputs_embeds
    all_hidden_states = () if output_hidden_states else None
    layer_traces = [] if return_step_traces else None

    for block in model.layers:
        hidden_states, block_trace = mamba_block_mat_eval(
            block,
            hidden_states,
            cache_params=cache_params,
            cache_position=cache_position,
            attention_mask=attention_mask,
            return_step_traces=return_step_traces,
        )
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if return_step_traces:
            layer_traces.append(block_trace)

    hidden_states = model.norm_f(hidden_states)
    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

    if not return_dict:
        values = [hidden_states, cache_params if use_cache else None, all_hidden_states]
        if return_step_traces:
            values.append(tuple(layer_traces))
        return tuple(value for value in values if value is not None)

    return MambaMatEvalOutput(
        last_hidden_state=hidden_states,
        cache_params=cache_params if use_cache else None,
        hidden_states=all_hidden_states,
        layer_traces=tuple(layer_traces) if return_step_traces else None,
    )


def mamba_lm_mat_eval(
    model,
    input_ids: torch.LongTensor | None = None,
    inputs_embeds: torch.Tensor | None = None,
    cache_params: MambaCache | None = None,
    use_cache: bool | None = None,
    output_hidden_states: bool | None = None,
    return_dict: bool = True,
    cache_position: torch.LongTensor | None = None,
    attention_mask: torch.LongTensor | None = None,
    return_step_traces: bool = False,
) -> MambaCausalLMMatEvalOutput | tuple[Any, ...]:
    backbone_outputs = mamba_model_mat_eval(
        model.backbone,
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        cache_params=cache_params,
        use_cache=use_cache,
        output_hidden_states=output_hidden_states,
        return_dict=True,
        cache_position=cache_position,
        attention_mask=attention_mask,
        return_step_traces=return_step_traces,
    )
    logits = model.lm_head(backbone_outputs.last_hidden_state)

    if not return_dict:
        values = [logits, backbone_outputs.cache_params, backbone_outputs.hidden_states]
        if return_step_traces:
            values.append(backbone_outputs.layer_traces)
        return tuple(value for value in values if value is not None)

    return MambaCausalLMMatEvalOutput(
        logits=logits,
        cache_params=backbone_outputs.cache_params,
        hidden_states=backbone_outputs.hidden_states,
        layer_traces=backbone_outputs.layer_traces,
    )


def compare_tensors(lhs: torch.Tensor, rhs: torch.Tensor) -> dict[str, float]:
    diff = (lhs - rhs).abs()
    return {
        "max_abs_error": diff.max().item(),
        "mean_abs_error": diff.mean().item(),
    }
