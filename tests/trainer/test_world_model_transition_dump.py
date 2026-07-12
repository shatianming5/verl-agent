import importlib.util
import json
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch


def _package(name):
    module = types.ModuleType(name)
    module.__path__ = []
    return module


def _load_ray_trainer_module(monkeypatch):
    @contextmanager
    def open_dict(config):
        yield config

    class Timer:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    class DataProto:
        pass

    stubs = {
        "ray": types.ModuleType("ray"),
        "codetiming": types.ModuleType("codetiming"),
        "omegaconf": types.ModuleType("omegaconf"),
        "torchdata": _package("torchdata"),
        "torchdata.stateful_dataloader": types.ModuleType("torchdata.stateful_dataloader"),
        "tqdm": types.ModuleType("tqdm"),
        "verl": types.ModuleType("verl"),
        "verl.protocol": types.ModuleType("verl.protocol"),
        "verl.single_controller": _package("verl.single_controller"),
        "verl.single_controller.base": types.ModuleType("verl.single_controller.base"),
        "verl.single_controller.ray": types.ModuleType("verl.single_controller.ray"),
        "verl.single_controller.ray.base": types.ModuleType("verl.single_controller.ray.base"),
        "verl.trainer": _package("verl.trainer"),
        "verl.trainer.ppo": _package("verl.trainer.ppo"),
        "verl.trainer.ppo.core_algos": types.ModuleType("verl.trainer.ppo.core_algos"),
        "verl.trainer.ppo.metric_utils": types.ModuleType("verl.trainer.ppo.metric_utils"),
        "verl.trainer.ppo.reward": types.ModuleType("verl.trainer.ppo.reward"),
        "verl.utils": _package("verl.utils"),
        "verl.utils.checkpoint": _package("verl.utils.checkpoint"),
        "verl.utils.checkpoint.checkpoint_manager": types.ModuleType("verl.utils.checkpoint.checkpoint_manager"),
        "verl.utils.metric": types.ModuleType("verl.utils.metric"),
        "verl.utils.seqlen_balancing": types.ModuleType("verl.utils.seqlen_balancing"),
        "verl.utils.torch_functional": types.ModuleType("verl.utils.torch_functional"),
        "verl.utils.tracking": types.ModuleType("verl.utils.tracking"),
        "verl.workers": _package("verl.workers"),
        "verl.workers.rollout": _package("verl.workers.rollout"),
        "verl.workers.rollout.async_server": types.ModuleType("verl.workers.rollout.async_server"),
        "gigpo": _package("gigpo"),
        "gigpo.core_gigpo": types.ModuleType("gigpo.core_gigpo"),
        "agent_system": _package("agent_system"),
        "agent_system.multi_turn_rollout": types.ModuleType("agent_system.multi_turn_rollout"),
    }
    stubs["codetiming"].Timer = Timer
    stubs["omegaconf"].OmegaConf = types.SimpleNamespace(set_struct=lambda *args, **kwargs: None, select=lambda *args, **kwargs: None)
    stubs["omegaconf"].open_dict = open_dict
    stubs["torchdata.stateful_dataloader"].StatefulDataLoader = object
    stubs["tqdm"].tqdm = lambda iterable=None, *args, **kwargs: iterable
    stubs["ray"].state = types.SimpleNamespace(available_resources_per_node=lambda: {})
    stubs["ray"].get = lambda value: value
    stubs["verl"].DataProto = DataProto
    stubs["verl.protocol"].pad_dataproto_to_divisor = lambda batch, divisor: (batch, 0)
    stubs["verl.protocol"].unpad_dataproto = lambda batch, pad_size: batch
    stubs["verl.single_controller.base"].Worker = object
    stubs["verl.single_controller.ray"].RayClassWithInitArgs = object
    stubs["verl.single_controller.ray"].RayResourcePool = object
    stubs["verl.single_controller.ray"].RayWorkerGroup = object
    stubs["verl.single_controller.ray.base"].create_colocated_worker_cls = lambda *args, **kwargs: object
    stubs["verl.trainer.ppo"].core_algos = stubs["verl.trainer.ppo.core_algos"]
    stubs["verl.trainer.ppo.core_algos"].AdaptiveKLController = object
    stubs["verl.trainer.ppo.core_algos"].agg_loss = lambda *args, **kwargs: torch.tensor(0.0)
    metric_utils = stubs["verl.trainer.ppo.metric_utils"]
    metric_utils.compute_data_metrics = lambda *args, **kwargs: {}
    metric_utils.compute_throughout_metrics = lambda *args, **kwargs: {}
    metric_utils.compute_timing_metrics = lambda *args, **kwargs: {}
    metric_utils.process_validation_metrics = lambda *args, **kwargs: {}
    reward_module = stubs["verl.trainer.ppo.reward"]
    reward_module.compute_reward = lambda *args, **kwargs: (None, {})
    reward_module.compute_reward_async = types.SimpleNamespace(remote=lambda *args, **kwargs: (None, {}))
    stubs["verl.utils.checkpoint.checkpoint_manager"].find_latest_ckpt_path = lambda *args, **kwargs: None
    stubs["verl.utils.metric"].reduce_metrics = lambda metrics: metrics
    stubs["verl.utils.seqlen_balancing"].get_seqlen_balanced_partitions = lambda *args, **kwargs: []
    stubs["verl.utils.seqlen_balancing"].log_seqlen_unbalance = lambda *args, **kwargs: None
    stubs["verl.utils.torch_functional"].masked_mean = lambda tensor, *args, **kwargs: tensor.mean()
    stubs["verl.utils.tracking"].ValidationGenerationsLogger = object
    stubs["verl.workers.rollout.async_server"].AsyncLLMServerManager = object
    stubs["agent_system.multi_turn_rollout"].TrajectoryCollector = object
    stubs["agent_system.multi_turn_rollout"].adjust_batch = lambda batch, *args, **kwargs: batch

    for name, module in stubs.items():
        monkeypatch.setitem(sys.modules, name, module)

    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "verl" / "trainer" / "ppo" / "ray_trainer.py"
    spec = importlib.util.spec_from_file_location("ray_trainer_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_dump_world_model_transitions_writes_scorer_schema(tmp_path, monkeypatch):
    module = _load_ray_trainer_module(monkeypatch)
    trainer = object.__new__(module.RayPPOTrainer)
    trainer.global_steps = 7
    batch = types.SimpleNamespace(
        batch={"token_level_scores": torch.tensor([[0.0, 2.0], [0.5, 0.5]])},
        non_tensor_batch={
            "uid": np.array(["uid-a", "uid-b"], dtype=object),
            "traj_uid": np.array(["traj-a", "traj-b"], dtype=object),
            "wm_step_idx": np.array([0, 1], dtype=np.int32),
            "wm_prev_obs_text": np.array(["room", "hall"], dtype=object),
            "wm_action_text": np.array(["take key", "open door"], dtype=object),
            "wm_next_obs_text": np.array(["inventory has key", "door is closed"], dtype=object),
            "wm_done_after_action": np.array([False, True]),
            "active_masks": np.array([True, False]),
            "rewards": np.array([1.0, 0.0]),
            "episode_rewards": np.array([1.0, 0.0]),
            "episode_lengths": np.array([3, 4]),
            "is_action_valid": np.array([True, False]),
            "task_success_rate": np.array([1.0, 0.0]),
        },
    )

    filename = trainer._dump_world_model_transitions(batch=batch, dump_path=str(tmp_path))

    assert filename == str(tmp_path / "7.wm_transitions.jsonl")
    rows = [json.loads(line) for line in Path(filename).read_text(encoding="utf-8").splitlines()]
    assert rows[0]["schema_version"] == "wm_transition_v1"
    assert rows[0]["split"] == "train"
    assert rows[0]["wm_prev_obs_text"] == "room"
    assert rows[0]["wm_action_text"] == "take key"
    assert rows[0]["wm_next_obs_text"] == "inventory has key"
    assert rows[0]["score"] == 2.0
    assert rows[0]["episode_success"] is True
    assert rows[1]["active_masks"] is False
    assert rows[1]["episode_success"] is False


def test_dump_world_model_transitions_writes_validation_split_with_append(tmp_path, monkeypatch):
    module = _load_ray_trainer_module(monkeypatch)
    trainer = object.__new__(module.RayPPOTrainer)
    trainer.global_steps = 150

    first_batch = types.SimpleNamespace(
        batch={},
        non_tensor_batch={
            "wm_prev_obs_text": np.array(["room"], dtype=object),
            "wm_action_text": np.array(["take key"], dtype=object),
            "wm_next_obs_text": np.array(["inventory has key"], dtype=object),
            "episode_rewards": np.array([1.0]),
        },
    )
    second_batch = types.SimpleNamespace(
        batch={},
        non_tensor_batch={
            "wm_prev_obs_text": np.array(["hall"], dtype=object),
            "wm_action_text": np.array(["open door"], dtype=object),
            "wm_next_obs_text": np.array(["door opens"], dtype=object),
            "episode_rewards": np.array([0.0]),
        },
    )

    first = trainer._dump_world_model_transitions(
        batch=first_batch,
        dump_path=str(tmp_path),
        split="val",
        batch_idx=0,
        score_tensor=torch.tensor([[0.25, 0.75]]),
    )
    second = trainer._dump_world_model_transitions(
        batch=second_batch,
        dump_path=str(tmp_path),
        split="val",
        append=True,
        row_offset=1,
        batch_idx=1,
        score_tensor=torch.tensor([0.5]),
    )

    assert first == second == str(tmp_path / "150.val.wm_transitions.jsonl")
    rows = [json.loads(line) for line in Path(first).read_text(encoding="utf-8").splitlines()]
    assert [row["split"] for row in rows] == ["val", "val"]
    assert [row["row_idx"] for row in rows] == [0, 1]
    assert [row["batch_idx"] for row in rows] == [0, 1]
    assert [row["score"] for row in rows] == [1.0, 0.5]
    assert rows[0]["episode_success"] is True
    assert rows[1]["episode_success"] is False


def test_dump_world_model_transitions_skips_non_world_model_batches(tmp_path, monkeypatch):
    module = _load_ray_trainer_module(monkeypatch)
    trainer = object.__new__(module.RayPPOTrainer)
    trainer.global_steps = 8
    batch = types.SimpleNamespace(batch={}, non_tensor_batch={"uid": np.array(["uid-a"], dtype=object)})

    assert trainer._dump_world_model_transitions(batch=batch, dump_path=str(tmp_path)) is None
    assert list(tmp_path.iterdir()) == []


def test_full_protocol_dump_requires_and_preserves_manifest_metadata(tmp_path, monkeypatch):
    module = _load_ray_trainer_module(monkeypatch)
    trainer = object.__new__(module.RayPPOTrainer)
    trainer.global_steps = 15
    trainer.config = {
        "trainer": {
            "world_model_dump_protocol": "workstream_b_full_train_v2",
            "world_model_diagnostic_checkpoint_step": "15",
        },
        "actor_rollout_ref": {
            "rollout": {
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": -1,
                "do_sample": True,
                "val_kwargs": {
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "top_k": -1,
                    "do_sample": True,
                    "n": 1,
                },
            }
        },
    }
    metadata = {
        "wm_game_id": "game-a",
        "wm_gamefile": "/data/game-a/game.tw-pddl",
        "wm_task_type": "pick_and_place_simple",
        "wm_game_sha256": "a" * 64,
        "wm_manifest_sha256": "b" * 64,
        "wm_schedule_index": 0,
        "wm_schedule_padding": False,
        "wm_trajectory_index": 0,
        "wm_episode_id": "episode-a",
    }
    batch = types.SimpleNamespace(
        batch={},
        non_tensor_batch={
            "traj_uid": np.array(["episode-a"], dtype=object),
            "wm_step_idx": np.array([0]),
            "wm_prev_obs_text": np.array(["room"], dtype=object),
            "wm_action_text": np.array(["look"], dtype=object),
            "wm_next_obs_text": np.array(["room"], dtype=object),
            "episode_rewards": np.array([10.0]),
            **{key: np.array([value], dtype=object) for key, value in metadata.items()},
        },
    )

    filename = trainer._dump_world_model_transitions(
        batch=batch,
        dump_path=str(tmp_path),
        split="train",
        score_tensor=torch.tensor([10.0]),
    )

    row = json.loads(Path(filename).read_text(encoding="utf-8"))
    assert row["schema_version"] == "wm_transition_v2"
    assert row["workstream_b_protocol"] == "workstream_b_full_train_v2"
    assert row["wm_game_id"] == "game-a"
    assert row["wm_episode_id"] == row["traj_uid"] == "episode-a"
    assert row["rollout_checkpoint_step"] == "15"
    assert row["rollout_temperature"] == 1.0
    assert row["rollout_top_k"] == -1
    assert row["score"] == 10.0


def test_episode_success_prefers_trajectory_reward_over_batch_success_rate(monkeypatch):
    module = _load_ray_trainer_module(monkeypatch)

    assert module.RayPPOTrainer._infer_episode_success(
        {"episode_rewards": 0.0, "success_rate": 0.5}
    ) is False
    assert module.RayPPOTrainer._infer_episode_success(
        {"episode_rewards": 10.0, "success_rate": 0.0}
    ) is True
