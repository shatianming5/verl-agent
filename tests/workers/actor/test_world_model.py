import importlib.util
import sys
import types
from pathlib import Path

import torch


def _load_world_model_module():
    repo_root = Path(__file__).resolve().parents[3]
    module_path = repo_root / "verl" / "workers" / "actor" / "world_model.py"
    spec = importlib.util.spec_from_file_location("world_model_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _package(name):
    module = types.ModuleType(name)
    module.__path__ = []
    return module


def _load_dp_actor_module(monkeypatch):
    world_model_module = _load_world_model_module()

    def gpu_memory_logger(*args, **kwargs):
        def decorator(fn):
            return fn

        return decorator

    class BasePPOActor:
        def __init__(self, config):
            self.config = config

    stubs = {
        "verl": types.ModuleType("verl"),
        "verl.trainer": _package("verl.trainer"),
        "verl.trainer.ppo": _package("verl.trainer.ppo"),
        "verl.trainer.ppo.core_algos": types.ModuleType("verl.trainer.ppo.core_algos"),
        "verl.utils": _package("verl.utils"),
        "verl.utils.debug": types.ModuleType("verl.utils.debug"),
        "verl.utils.device": types.ModuleType("verl.utils.device"),
        "verl.utils.fsdp_utils": types.ModuleType("verl.utils.fsdp_utils"),
        "verl.utils.py_functional": types.ModuleType("verl.utils.py_functional"),
        "verl.utils.seqlen_balancing": types.ModuleType("verl.utils.seqlen_balancing"),
        "verl.utils.torch_functional": types.ModuleType("verl.utils.torch_functional"),
        "verl.utils.ulysses": types.ModuleType("verl.utils.ulysses"),
        "verl.workers": _package("verl.workers"),
        "verl.workers.actor": _package("verl.workers.actor"),
        "verl.workers.actor.world_model": world_model_module,
    }
    stubs["verl"].DataProto = object
    stubs["verl.trainer.ppo.core_algos"].agg_loss = lambda loss_mat, loss_mask, loss_agg_mode: (
        loss_mat * loss_mask
    ).sum() / loss_mask.sum().clamp_min(1.0)
    stubs["verl.trainer.ppo.core_algos"].compute_policy_loss = lambda **kwargs: None
    stubs["verl.trainer.ppo.core_algos"].compute_policy_loss_gspo = lambda **kwargs: None
    stubs["verl.trainer.ppo.core_algos"].kl_penalty = lambda **kwargs: None
    stubs["verl.utils.debug"].GPUMemoryLogger = gpu_memory_logger
    stubs["verl.utils.device"].get_device_name = lambda: "cpu"
    stubs["verl.utils.device"].get_torch_device = lambda: torch
    stubs["verl.utils.device"].is_cuda_available = False
    stubs["verl.utils.device"].is_npu_available = False
    stubs["verl.utils.fsdp_utils"].FSDPModule = torch.nn.Module
    stubs["verl.utils.fsdp_utils"].fsdp2_clip_grad_norm_ = torch.nn.utils.clip_grad_norm_
    stubs["verl.utils.py_functional"].append_to_dict = lambda dst, src: dst.update(src)
    stubs["verl.utils.seqlen_balancing"].get_reverse_idx = lambda indices: indices
    stubs["verl.utils.seqlen_balancing"].rearrange_micro_batches = lambda batch, max_token_len: ([batch], [0])
    stubs["verl.utils.torch_functional"].entropy_from_logits = lambda logits: logits.sum(dim=-1) * 0.0
    stubs["verl.utils.torch_functional"].logprobs_from_logits = lambda logits, labels, inplace_backward=True: logits[
        ..., 0
    ] * 0.0
    stubs["verl.utils.ulysses"].gather_outpus_and_unpad = lambda value, **kwargs: value
    stubs["verl.utils.ulysses"].ulysses_pad_and_slice_inputs = lambda value, **kwargs: (value, None, 0)
    stubs["verl.utils.ulysses"].ulysses_pad = lambda value, **kwargs: (value, None, 0)
    stubs["verl.workers.actor"].BasePPOActor = BasePPOActor

    for name, module in stubs.items():
        monkeypatch.setitem(sys.modules, name, module)

    repo_root = Path(__file__).resolve().parents[3]
    module_path = repo_root / "verl" / "workers" / "actor" / "dp_actor.py"
    spec = importlib.util.spec_from_file_location("dp_actor_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_residual_predictor_starts_as_identity():
    module = _load_world_model_module()
    predictor = module.LatentTransitionPredictor(hidden_size=8, bottleneck_size=4, residual=True)

    hidden_states = torch.randn(3, 8)

    torch.testing.assert_close(predictor(hidden_states), hidden_states)


def test_non_residual_predictor_preserves_shape_and_gradients():
    module = _load_world_model_module()
    predictor = module.LatentTransitionPredictor(hidden_size=8, bottleneck_size=4, residual=False)
    hidden_states = torch.randn(3, 8, requires_grad=True)

    output = predictor(hidden_states)
    output.sum().backward()

    assert output.shape == hidden_states.shape
    assert hidden_states.grad is not None


def test_latent_loss_stop_gradient_target_and_masks_rows(monkeypatch):
    module = _load_dp_actor_module(monkeypatch)

    class ToyActor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(16, 4)

        def forward(self, input_ids, attention_mask=None, position_ids=None, **kwargs):
            return types.SimpleNamespace(hidden_states=(self.embedding(input_ids),))

    actor = object.__new__(module.DataParallelPPOActor)
    actor.actor_module = ToyActor()
    actor.world_model_predictor = torch.nn.Linear(4, 4, bias=False)
    actor.device_name = "cpu"
    with torch.no_grad():
        actor.world_model_predictor.weight.copy_(torch.eye(4))

    micro_batch = {
        "wm_input_ids": torch.tensor([[2, 3, 4, 5], [6, 7, 8, 9]]),
        "wm_attention_mask": torch.ones(2, 4, dtype=torch.long),
        "wm_position_ids": torch.arange(4).expand(2, -1),
        "wm_action_end_idx": torch.tensor([1, 1]),
        "wm_obs_end_idx": torch.tensor([3, 3]),
        "wm_loss_mask": torch.tensor([1.0, 0.0]),
    }

    latent_loss, metrics = actor._compute_latent_world_model_loss(micro_batch)
    latent_loss.backward()

    embedding_grad = actor.actor_module.embedding.weight.grad
    predictor_grad = actor.world_model_predictor.weight.grad

    assert metrics["actor/wm_valid"] == 1.0
    assert embedding_grad[3].norm().item() > 0.0
    torch.testing.assert_close(embedding_grad[5], torch.zeros_like(embedding_grad[5]))
    torch.testing.assert_close(embedding_grad[7], torch.zeros_like(embedding_grad[7]))
    torch.testing.assert_close(embedding_grad[9], torch.zeros_like(embedding_grad[9]))
    assert predictor_grad.norm().item() > 0.0


def test_extra_state_dict_saves_world_model_predictor_config(monkeypatch):
    module = _load_dp_actor_module(monkeypatch)
    actor = object.__new__(module.DataParallelPPOActor)
    actor.world_model_predictor = torch.nn.Linear(2, 2)
    actor.world_model_predictor_config = {
        "hidden_size": 2,
        "bottleneck_size": 2,
        "dropout": 0.0,
        "residual": False,
    }

    state = actor.extra_state_dict()

    assert state["world_model_predictor_config"]["residual"] is False
    assert "world_model_predictor" in state
