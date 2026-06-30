import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest


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


def _load_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "wm_score_transition_dump.py"
    spec = importlib.util.spec_from_file_location("wm_score_transition_dump_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _row(**overrides):
    row = {
        "schema_version": "wm_transition_v1",
        "wm_prev_obs_text": "room",
        "wm_action_text": "take key",
        "wm_next_obs_text": "inventory has key",
        "active_masks": True,
        "episode_rewards": 10.0,
    }
    row.update(overrides)
    return row


def test_transition_encoding_uses_chat_template_suffix():
    module = _load_module()
    tokenizer = FakeTokenizer()

    prior_ids, target_ids = module.chat_transition_ids(
        tokenizer=tokenizer,
        prev_obs_text="room",
        action_text="take key",
        next_obs_text="inventory has key",
    )

    assert tokenizer.decode(prior_ids) == "<user>room</user><assistant>take key</assistant>"
    assert tokenizer.decode(target_ids) == "<user>inventory has key</user>"
    assert tokenizer.decode(target_ids) != "inventory has key"


def test_encode_transitions_masks_only_active_next_observation_tokens():
    module = _load_module()
    tokenizer = FakeTokenizer()

    encoded = module.encode_transitions(
        [_row(), _row(active_masks=False, episode_rewards=0.0)],
        tokenizer=tokenizer,
        max_length=96,
    )

    prior_ids, target_ids = module.chat_transition_ids(tokenizer, "room", "take key", "inventory has key")
    seq_len = len(prior_ids) + len(target_ids)

    assert tokenizer.decode(encoded[0].input_ids[:seq_len]) == (
        "<user>room</user><assistant>take key</assistant><user>inventory has key</user>"
    )
    assert sum(encoded[0].loss_mask[: len(prior_ids)]) == 0.0
    assert all(value == 1.0 for value in encoded[0].loss_mask[len(prior_ids) : seq_len])
    assert sum(encoded[0].loss_mask[seq_len:]) == 0.0
    assert encoded[0].action_end_pos == len(prior_ids) - 1
    assert encoded[0].obs_end_pos == seq_len - 1
    assert encoded[0].target_tokens == len(target_ids)
    assert encoded[0].episode_success is True
    assert encoded[1].target_tokens == 0
    assert encoded[1].episode_success is False


def test_string_boolean_fields_are_normalized():
    module = _load_module()
    tokenizer = FakeTokenizer()

    encoded = module.encode_transitions(
        [
            _row(active_masks="false", episode_success="false", episode_rewards=1.0),
            _row(active_masks="true", episode_success="true", episode_rewards=0.0),
        ],
        tokenizer=tokenizer,
        max_length=96,
    )

    assert encoded[0].target_tokens == 0
    assert encoded[0].episode_success is False
    assert encoded[1].target_tokens > 0
    assert encoded[1].episode_success is True

    summary = module.summarize(
        [
            {
                "checkpoint_label": "step150",
                "checkpoint_path": "/ckpt/global_step_150",
                "checkpoint_step": "150",
                "target_tokens": 2,
                "nll_sum": 2.0,
                "ce": 1.0,
                "target_confidence_mean": 0.3,
                "target_entropy_mean": 1.1,
                "action_hidden_norm": 2.0,
                "obs_hidden_norm": 3.0,
                "action_obs_cosine": 0.5,
                "episode_success": "true",
            },
            {
                "checkpoint_label": "step150",
                "checkpoint_path": "/ckpt/global_step_150",
                "checkpoint_step": "150",
                "target_tokens": 2,
                "nll_sum": 4.0,
                "ce": 2.0,
                "target_confidence_mean": 0.2,
                "target_entropy_mean": 1.4,
                "action_hidden_norm": 2.0,
                "obs_hidden_norm": 3.0,
                "action_obs_cosine": 0.1,
                "episode_success": "false",
            },
        ]
    )

    buckets = {bucket["episode_success"]: bucket["token_mean_ce"] for bucket in summary["success_buckets"]}
    assert buckets == {True: 1.0, False: 2.0}


def test_load_transitions_and_summary_success_buckets(tmp_path):
    module = _load_module()
    transition_path = tmp_path / "transitions.jsonl"
    transition_path.write_text(json.dumps(_row()) + "\n", encoding="utf-8")

    rows = module.load_transitions(str(transition_path))
    assert rows[0]["wm_action_text"] == "take key"

    summary = module.summarize(
        [
            {
                "checkpoint_label": "step0",
                "checkpoint_path": "base",
                "checkpoint_step": "init",
                "target_tokens": 2,
                "nll_sum": 4.0,
                "ce": 2.0,
                "target_confidence_mean": 0.2,
                "target_entropy_mean": 1.0,
                "action_hidden_norm": 2.0,
                "obs_hidden_norm": 3.0,
                "action_obs_cosine": 0.5,
                "episode_success": True,
            },
            {
                "checkpoint_label": "step0",
                "checkpoint_path": "base",
                "checkpoint_step": "init",
                "target_tokens": 2,
                "nll_sum": 6.0,
                "ce": 3.0,
                "target_confidence_mean": 0.1,
                "target_entropy_mean": 1.5,
                "action_hidden_norm": 4.0,
                "obs_hidden_norm": 5.0,
                "action_obs_cosine": 0.1,
                "episode_success": False,
            },
        ]
    )

    assert summary["checkpoints"][0]["target_tokens"] == 4
    assert summary["checkpoints"][0]["token_mean_ce"] == 2.5
    assert {bucket["episode_success"] for bucket in summary["success_buckets"]} == {True, False}


def test_build_provenance_records_scoring_config(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(sys, "argv", ["wm_score_transition_dump.py", "--model-path", "/model"])
    args = module.argparse.Namespace(
        model_path="/model",
        transition_jsonl="/work/transitions.jsonl",
        output_csv="/work/scores.csv",
        summary_json="/work/summary.json",
        max_length=512,
        batch_size=4,
        max_rows=128,
        device="cuda:0",
        dtype="bfloat16",
        skip_entropy=True,
        chat_template_kwargs_json='{"enable_thinking": false}',
        chat_template_kwargs={"enable_thinking": False},
    )
    specs = [
        module.CheckpointSpec(label="init", path="base"),
        module.CheckpointSpec(label="step150", path="/ckpt/global_step_150"),
    ]

    provenance = module.build_provenance(args, specs)

    assert provenance["command"] == "wm_score_transition_dump.py --model-path /model"
    assert provenance["model_path"] == "/model"
    assert provenance["transition_jsonl"] == "/work/transitions.jsonl"
    assert provenance["checkpoint_count"] == 2
    assert provenance["checkpoints"][0] == {"label": "init", "path": "base", "step": "init"}
    assert provenance["checkpoints"][1]["step"] == "150"
    assert provenance["batch_size"] == 4
    assert provenance["skip_entropy"] is True
    assert provenance["chat_template_kwargs"] == {"enable_thinking": False}


def test_atomic_write_preserves_existing_summary_on_replace_failure(tmp_path, monkeypatch):
    module = _load_module()
    output_path = tmp_path / "checkpoint_scores_summary.json"
    output_path.write_text('{"old": true}\n', encoding="utf-8")

    def fail_replace(src, dst):
        raise RuntimeError("replace failed")

    monkeypatch.setattr(module.os, "replace", fail_replace)

    with pytest.raises(RuntimeError, match="replace failed"):
        module.write_json(str(output_path), {"old": False})

    assert json.loads(output_path.read_text(encoding="utf-8")) == {"old": True}
    assert list(tmp_path.glob(".checkpoint_scores_summary.json.*.tmp")) == []


def test_load_world_model_predictor_from_checkpoint_extra_state(tmp_path):
    torch = pytest.importorskip("torch")
    module = _load_module()
    actor_dir = tmp_path / "actor"
    actor_dir.mkdir()
    predictor_state = {
        "net.0.weight": torch.ones(4),
        "net.0.bias": torch.zeros(4),
        "net.1.weight": torch.zeros(3, 4),
        "net.1.bias": torch.zeros(3),
        "net.4.weight": torch.zeros(4, 3),
        "net.4.bias": torch.tensor([0.0, 1.0, 0.0, 0.0]),
    }
    torch.save(
        {"custom_extra_state": {"world_model_predictor": predictor_state}},
        actor_dir / "extra_state_world_size_1_rank_0.pt",
    )

    predictor = module.load_world_model_predictor(str(actor_dir), device="cpu", dtype_name="float32")

    assert predictor is not None
    hidden = torch.zeros(1, 4)
    assert torch.allclose(predictor(hidden), torch.tensor([[0.0, 1.0, 0.0, 0.0]]))


def test_load_world_model_predictor_respects_non_residual_checkpoint_config(tmp_path):
    torch = pytest.importorskip("torch")
    module = _load_module()
    actor_dir = tmp_path / "actor"
    actor_dir.mkdir()
    predictor_state = {
        "net.0.weight": torch.ones(4),
        "net.0.bias": torch.zeros(4),
        "net.1.weight": torch.zeros(3, 4),
        "net.1.bias": torch.zeros(3),
        "net.4.weight": torch.zeros(4, 3),
        "net.4.bias": torch.tensor([0.0, 1.0, 0.0, 0.0]),
    }
    torch.save(
        {
            "custom_extra_state": {
                "world_model_predictor": predictor_state,
                "world_model_predictor_config": {"residual": False},
            }
        },
        actor_dir / "extra_state_world_size_1_rank_0.pt",
    )

    predictor = module.load_world_model_predictor(str(actor_dir), device="cpu", dtype_name="float32")

    assert predictor is not None
    hidden = torch.tensor([[2.0, 0.0, 0.0, 0.0]])
    assert torch.allclose(predictor(hidden), torch.tensor([[0.0, 1.0, 0.0, 0.0]]))


def test_score_batch_uses_loaded_world_model_predictor_for_cosine():
    torch = pytest.importorskip("torch")
    module = _load_module()

    class FakeModel:
        def __call__(self, input_ids, attention_mask, use_cache, output_hidden_states):
            logits = torch.zeros(input_ids.size(0), input_ids.size(1), 8, device=input_ids.device)
            hidden = torch.zeros(input_ids.size(0), input_ids.size(1), 2, device=input_ids.device)
            hidden[:, 1, :] = torch.tensor([1.0, 0.0], device=input_ids.device)
            hidden[:, 2, :] = torch.tensor([0.0, 1.0], device=input_ids.device)
            return types.SimpleNamespace(logits=logits, hidden_states=[hidden])

    predictor = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        predictor.weight.copy_(torch.tensor([[0.0, 0.0], [1.0, 0.0]]))
    encoded = module.EncodedTransition(
        transition_index=0,
        row=_row(),
        input_ids=[0, 2, 3, 0],
        attention_mask=[1, 1, 1, 0],
        loss_mask=[0.0, 0.0, 0.0, 0.0],
        seq_len=3,
        target_tokens=0,
        target_start=2,
        action_end_pos=1,
        obs_end_pos=2,
        episode_success=True,
    )

    rows = module.score_batch(
        model=FakeModel(),
        batch=[encoded],
        checkpoint=module.CheckpointSpec(label="step1", path="/ckpt/global_step_1"),
        device="cpu",
        skip_entropy=True,
        world_model_predictor=predictor,
    )

    assert rows[0]["raw_action_obs_cosine"] == 0.0
    assert rows[0]["pred_obs_cosine"] == 1.0
    assert rows[0]["action_obs_cosine"] == 1.0
    assert rows[0]["world_model_predictor_loaded"] is True
