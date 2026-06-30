import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import torch


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def apply_chat_template(self, chat, add_generation_prompt=False, tokenize=False, **kwargs):
        text = "".join(f"<{message['role']}>{message['content']}</{message['role']}>" for message in chat)
        if add_generation_prompt:
            text += "<assistant>"
        if tokenize:
            return self.encode(text, add_special_tokens=False)
        return text

    def encode(self, text, add_special_tokens=False):
        return [ord(char) + 2 for char in text]

    def decode(self, token_ids):
        return "".join(chr(int(token_id) - 2) for token_id in token_ids if int(token_id) > 1)


def _package(name):
    module = types.ModuleType(name)
    module.__path__ = []
    return module


def _load_rollout_module(monkeypatch):
    def compute_position_id_with_mask(attention_mask):
        return torch.clamp(torch.cumsum(attention_mask, dim=-1) - 1, min=0)

    stubs = {
        "verl": types.ModuleType("verl"),
        "verl.utils": _package("verl.utils"),
        "verl.utils.dataset": _package("verl.utils.dataset"),
        "verl.utils.dataset.rl_dataset": types.ModuleType("verl.utils.dataset.rl_dataset"),
        "verl.utils.model": types.ModuleType("verl.utils.model"),
        "verl.utils.torch_functional": types.ModuleType("verl.utils.torch_functional"),
        "verl.protocol": types.ModuleType("verl.protocol"),
        "transformers": types.ModuleType("transformers"),
        "agent_system": _package("agent_system"),
        "agent_system.multi_turn_rollout": _package("agent_system.multi_turn_rollout"),
        "agent_system.multi_turn_rollout.utils": types.ModuleType("agent_system.multi_turn_rollout.utils"),
        "agent_system.environments": types.ModuleType("agent_system.environments"),
    }

    stubs["verl"].DataProto = object
    stubs["verl.utils.dataset.rl_dataset"].collate_fn = lambda samples: samples
    stubs["verl.utils.model"].compute_position_id_with_mask = compute_position_id_with_mask
    stubs["verl.protocol"].pad_dataproto_to_divisor = lambda data, size_divisor: (data, 0)
    stubs["verl.protocol"].unpad_dataproto = lambda data, pad_size: data
    stubs["transformers"].PreTrainedTokenizer = object
    stubs["agent_system.multi_turn_rollout.utils"].process_image = lambda image: image
    stubs["agent_system.multi_turn_rollout.utils"].to_list_of_dict = lambda batch: batch
    stubs["agent_system.multi_turn_rollout.utils"].torch_to_numpy = lambda value, is_object=False: value
    stubs["agent_system.multi_turn_rollout.utils"].filter_group_data = lambda **kwargs: kwargs
    stubs["agent_system.environments"].EnvironmentManagerBase = object

    for name, module in stubs.items():
        monkeypatch.setitem(sys.modules, name, module)

    repo_root = Path(__file__).resolve().parents[3]
    module_path = repo_root / "agent_system" / "multi_turn_rollout" / "rollout_loop.py"
    spec = importlib.util.spec_from_file_location("rollout_loop_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _collector(monkeypatch, world_model_config):
    module = _load_rollout_module(monkeypatch)
    config = types.SimpleNamespace(
        data={"apply_chat_template_kwargs": {}},
        actor_rollout_ref={"actor": {"world_model": world_model_config}},
    )
    return module.TrajectoryCollector(config=config, tokenizer=FakeTokenizer())


def test_chat_transition_ids_use_chat_template_suffix(monkeypatch):
    collector = _collector(monkeypatch, {})

    prior_ids, target_ids = collector._chat_transition_ids(
        prev_obs_text="room",
        action_text="take key",
        next_obs_text="inventory has key",
    )

    assert collector.tokenizer.decode(prior_ids) == "<user>room</user><assistant>take key</assistant>"
    assert collector.tokenizer.decode(target_ids) == "<user>inventory has key</user>"
    assert collector.tokenizer.decode(target_ids) != "inventory has key"


def test_obs_ce_batch_masks_only_next_observation_suffix(monkeypatch):
    collector = _collector(monkeypatch, {"lambda_obs": 0.01, "obs_ce_max_length": 96})

    batch = collector._build_obs_ce_batch(
        prev_obs={"text": ["room", "hall"]},
        actions=["take key", "open door"],
        next_obs={"text": ["inventory has key", "door is closed"]},
        active_masks=np.array([True, False]),
        batch_size=2,
    )

    prior_ids, target_ids = collector._chat_transition_ids("room", "take key", "inventory has key")
    prefix_len = len(prior_ids)
    seq_len = prefix_len + len(target_ids)

    assert batch["wm_obs_input_ids"].shape == (2, 96)
    assert collector.tokenizer.decode(batch["wm_obs_input_ids"][0, :seq_len]) == (
        "<user>room</user><assistant>take key</assistant><user>inventory has key</user>"
    )
    assert batch["wm_obs_loss_mask"][0, :prefix_len].sum().item() == 0
    assert torch.all(batch["wm_obs_loss_mask"][0, prefix_len:seq_len] == 1)
    assert batch["wm_obs_loss_mask"][0, seq_len:].sum().item() == 0
    assert batch["wm_obs_loss_mask"][1].sum().item() == 0


def test_latent_world_model_tensors_mark_action_and_observation_positions(monkeypatch):
    collector = _collector(monkeypatch, {"lambda_latent": 0.001, "latent_max_length": 96})

    batch = collector._build_latent_world_model_tensors(
        prev_obs={"text": ["room", "hall"]},
        actions=["take key", "open door"],
        next_obs={"text": ["inventory has key", "door is closed"]},
        active_masks=np.array([True, False]),
        batch_size=2,
    )

    prior_ids, target_ids = collector._chat_transition_ids("room", "take key", "inventory has key")
    seq_len = len(prior_ids) + len(target_ids)

    assert batch["wm_input_ids"].shape == (2, 96)
    assert collector.tokenizer.decode(batch["wm_input_ids"][0, :seq_len]) == (
        "<user>room</user><assistant>take key</assistant><user>inventory has key</user>"
    )
    assert batch["wm_action_end_idx"][0].item() == len(prior_ids) - 1
    assert batch["wm_obs_end_idx"][0].item() == seq_len - 1
    assert batch["wm_action_end_idx"][0].item() < batch["wm_obs_end_idx"][0].item()
    torch.testing.assert_close(batch["wm_loss_mask"], torch.tensor([1.0, 0.0]))
