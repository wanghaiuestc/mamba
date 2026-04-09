from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import huggingface_hub

REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_SRC = str(REPO_ROOT / "src")

if not hasattr(huggingface_hub, "is_offline_mode"):
    huggingface_hub.is_offline_mode = lambda: huggingface_hub.constants.HF_HUB_OFFLINE
dependency_versions_check = sys.modules.setdefault(
    "transformers.dependency_versions_check", types.ModuleType("dependency_versions_check")
)
if not hasattr(dependency_versions_check, "dep_version_check"):
    dependency_versions_check.dep_version_check = lambda *args, **kwargs: None

def load_transformers():
    sys.path.insert(0, REPO_SRC)
    try:
        from transformers import MambaConfig, MambaForCausalLM
        from transformers.models.mamba.modeling_mamba import MambaCache

        return MambaConfig, MambaForCausalLM, MambaCache, "repo-src"
    except Exception as repo_error:
        sys.path = [entry for entry in sys.path if entry != REPO_SRC]
        for module_name in list(sys.modules):
            if module_name == "transformers" or module_name.startswith("transformers."):
                sys.modules.pop(module_name, None)

        from transformers import MambaConfig, MambaForCausalLM
        from transformers.models.mamba.modeling_mamba import MambaCache

        return (
            MambaConfig,
            MambaForCausalLM,
            MambaCache,
            f"site-packages fallback after repo-src import failed: {repo_error}",
        )


MambaConfig, MambaForCausalLM, MambaCache, TRANSFORMERS_SOURCE = load_transformers()

if TYPE_CHECKING:
    from transformers import MambaForCausalLM as MambaForCausalLMType
else:
    MambaForCausalLMType = Any

from mambatest.mat_eval import compare_tensors, mamba_lm_mat_eval, mamba_mixer_mat_eval


def disable_optional_mamba_kernels() -> None:
    import transformers.models.mamba.modeling_mamba as modeling_mamba

    original_lazy_load_kernel = modeling_mamba.lazy_load_kernel

    def patched_lazy_load_kernel(name: str):
        if name in {"mamba-ssm", "causal-conv1d"}:
            return None
        return original_lazy_load_kernel(name)

    modeling_mamba.lazy_load_kernel = patched_lazy_load_kernel


def build_model(device: torch.device, model_id: str | None) -> tuple[MambaForCausalLMType, torch.Tensor]:
    if model_id is None:
        config = MambaConfig(
            vocab_size=64,
            hidden_size=16,
            state_size=8,
            num_hidden_layers=2,
            expand=2,
            conv_kernel=4,
            use_bias=True,
            use_conv_bias=True,
        )
        model = MambaForCausalLM(config).to(device=device, dtype=torch.float32)
        input_ids = torch.tensor([[1, 5, 7, 9, 2, 3]], device=device)
        model.eval()
        return model, input_ids

    from transformers import AutoTokenizer

    config = MambaConfig.from_pretrained(model_id)
    config.tie_word_embeddings = True
    tokenizer = AutoTokenizer.from_pretrained(model_id, config=config)
    model = MambaForCausalLM.from_pretrained(model_id, config=config, torch_dtype=torch.float32).to(device)
    model.eval()
    encoded = tokenizer("Hey how are you doing?", return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    return model, input_ids


def summarize(name: str, stats: dict[str, float], threshold: float) -> bool:
    passed = stats["max_abs_error"] < threshold
    print(
        f"{name}: max_abs_error={stats['max_abs_error']:.8e}, "
        f"mean_abs_error={stats['mean_abs_error']:.8e}, pass={passed}"
    )
    return passed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-id", type=str, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    disable_optional_mamba_kernels()
    model, input_ids = build_model(device, args.model_id)
    print(f"transformers_source={TRANSFORMERS_SOURCE}")
    prefill_cache_position = torch.arange(0, model.config.conv_kernel, device=device)
    decode_cache_position = torch.arange(model.config.conv_kernel, model.config.conv_kernel + 1, device=device)

    with torch.no_grad():
        forward_outputs = model(
            input_ids=input_ids,
            use_cache=True,
            cache_position=prefill_cache_position,
            output_hidden_states=True,
            return_dict=True,
        )
        mat_outputs = mamba_lm_mat_eval(
            model,
            input_ids=input_ids,
            use_cache=True,
            cache_position=prefill_cache_position,
            output_hidden_states=True,
            return_step_traces=True,
        )

        hidden_stats = compare_tensors(forward_outputs.hidden_states[-1], mat_outputs.hidden_states[-1])
        logits_stats = compare_tensors(forward_outputs.logits, mat_outputs.logits)

        layer0_forward_cache = MambaCache(model.config, max_batch_size=1, device=device, dtype=torch.float32)
        layer0_mat_cache = MambaCache(model.config, max_batch_size=1, device=device, dtype=torch.float32)
        layer0_input = model.backbone.embeddings(input_ids)
        normalized_layer0_input = model.backbone.layers[0].norm(
            layer0_input.to(dtype=model.backbone.layers[0].norm.weight.dtype)
        )
        forward_mixer_output = model.backbone.layers[0].mixer.slow_forward(
            normalized_layer0_input,
            cache_params=layer0_forward_cache,
            cache_position=prefill_cache_position,
        )
        mat_mixer_output, layer0_traces = mamba_mixer_mat_eval(
            model.backbone.layers[0].mixer,
            normalized_layer0_input,
            cache_params=layer0_mat_cache,
            cache_position=prefill_cache_position,
            return_step_traces=True,
        )
        mixer_stats = compare_tensors(forward_mixer_output, mat_mixer_output)
        ssm_stats = compare_tensors(
            layer0_forward_cache.ssm_states[0],
            layer0_mat_cache.ssm_states[0],
        )
        conv_stats = compare_tensors(
            layer0_forward_cache.conv_states[0],
            layer0_mat_cache.conv_states[0],
        )

        next_input_ids = torch.tensor([[11]], device=device)
        decode_forward_outputs = model(
            input_ids=next_input_ids,
            use_cache=True,
            cache_params=forward_outputs.cache_params,
            cache_position=decode_cache_position,
            return_dict=True,
        )
        decode_mat_outputs = mamba_lm_mat_eval(
            model,
            input_ids=next_input_ids,
            use_cache=True,
            cache_params=mat_outputs.cache_params,
            cache_position=decode_cache_position,
            return_step_traces=True,
        )
        decode_logits_stats = compare_tensors(decode_forward_outputs.logits, decode_mat_outputs.logits)

    passed = True
    passed &= summarize("full_model_hidden", hidden_stats, args.threshold)
    passed &= summarize("full_model_logits", logits_stats, args.threshold)
    passed &= summarize("layer0_mixer_output", mixer_stats, args.threshold)
    passed &= summarize("layer0_ssm_state", ssm_stats, args.threshold)
    passed &= summarize("layer0_conv_state", conv_stats, args.threshold)
    passed &= summarize("decode_logits", decode_logits_stats, args.threshold)

    if layer0_traces:
        first_step = layer0_traces[0]
        print(
            "trace_example:"
            f" dt_shape={tuple(first_step.dt_t.shape)},"
            f" At_shape={tuple(first_step.At.shape)},"
            f" Bt_shape={tuple(first_step.Bt.shape)},"
            f" C_t_shape={tuple(first_step.C_t.shape)},"
            f" state_shape={tuple(first_step.next_state.shape)}"
        )

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
