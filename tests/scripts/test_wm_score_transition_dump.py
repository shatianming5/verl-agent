import importlib.util
import json
import sys
from pathlib import Path


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
