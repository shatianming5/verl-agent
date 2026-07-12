import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _modules():
    manifest = _load(
        REPO_ROOT / "agent_system" / "alfworld_game_manifest.py",
        "game_manifest_under_test",
    )
    schedule = _load(
        REPO_ROOT / "scripts" / "wm_alfworld_train_manifest.py",
        "wm_alfworld_train_manifest_under_test",
    )
    coverage = _load(
        REPO_ROOT / "scripts" / "wm_validate_rollout_coverage.py",
        "wm_validate_rollout_coverage_under_test",
    )
    return manifest, schedule, coverage


def _dataset(tmp_path, count=3):
    data_root = tmp_path / "alfworld" / "train"
    for index in reversed(range(count)):
        game_dir = data_root / f"pick_and_place_simple-{index:03d}"
        game_dir.mkdir(parents=True)
        (game_dir / "traj_data.json").write_text(
            json.dumps({"task_type": "pick_and_place_simple"}),
            encoding="utf-8",
        )
        (game_dir / "game.tw-pddl").write_text(
            json.dumps({"solvable": True, "index": index}),
            encoding="utf-8",
        )
    config = {
        "dataset": {
            "data_path": str(data_root),
            "eval_id_data_path": str(data_root),
            "eval_ood_data_path": str(data_root),
            "num_train_games": -1,
            "num_eval_games": -1,
        },
        "env": {"task_types": [1]},
    }
    return data_root, config


def test_manifest_is_sorted_exact_and_reproducible(tmp_path):
    manifest_module, schedule_module, _ = _modules()
    _, config = _dataset(tmp_path)

    first = manifest_module.build_manifest(config)
    second = manifest_module.build_manifest(config)

    assert first == second
    assert first["raw_traj_data_count"] == 3
    assert first["game_count"] == 3
    assert [game["game_id"] for game in first["games"]] == sorted(game["game_id"] for game in first["games"])
    assert all(len(game["sha256"]) == 64 for game in first["games"])
    manifest_module.validate_manifest(
        first,
        expected_games=3,
        expected_raw_trajectories=3,
        verify_files=True,
    )
    with pytest.raises(ValueError, match="exactly 3553"):
        manifest_module.validate_manifest(first, expected_games=3553)

    rows = schedule_module.build_schedule_rows(first, batch_size=2)
    assert len(rows) == 4
    assert [row["env_kwargs"]["wm_schedule_padding"] for row in rows] == [
        False,
        False,
        False,
        True,
    ]
    assert {row["env_kwargs"]["wm_game_id"] for row in rows[:3]} == {game["game_id"] for game in first["games"]}
    assert len({row["env_kwargs"]["wm_episode_id"] for row in rows}) == 4
    canonical = manifest_module.canonicalize_schedule(
        first,
        [row["env_kwargs"] for row in rows[:2]],
        env_num=2,
        group_n=1,
        require_schedule=True,
    )
    assert [entry["wm_schedule_index"] for entry in canonical] == [0, 1]
    tampered = [dict(row["env_kwargs"]) for row in rows[:2]]
    tampered[0]["wm_game_id"] = "not-in-manifest"
    with pytest.raises(ValueError, match="Schedule metadata mismatch"):
        manifest_module.canonicalize_schedule(
            first,
            tampered,
            env_num=2,
            group_n=1,
            require_schedule=True,
        )


def test_manifest_records_raw_trajectory_count_before_filtering(tmp_path):
    manifest_module, _, _ = _modules()
    data_root, config = _dataset(tmp_path)
    excluded = data_root / "movable-unsupported"
    excluded.mkdir()
    (excluded / "traj_data.json").write_text(
        json.dumps({"task_type": "pick_and_place_simple"}),
        encoding="utf-8",
    )
    (excluded / "game.tw-pddl").write_text(
        json.dumps({"solvable": True}),
        encoding="utf-8",
    )

    manifest = manifest_module.build_manifest(config)

    assert manifest["raw_traj_data_count"] == 4
    assert manifest["game_count"] == 3
    manifest_module.validate_manifest(
        manifest,
        expected_games=3,
        expected_raw_trajectories=4,
    )


def _transition(manifest, game, schedule_index, *, score=10.0):
    episode_id = f"{manifest['manifest_sha256']}:{game['game_id']}:trajectory0:schedule{schedule_index}"
    return {
        "schema_version": "wm_transition_v2",
        "workstream_b_protocol": "workstream_b_full_train_v2",
        "split": "train",
        "wm_game_id": game["game_id"],
        "wm_gamefile": game["gamefile"],
        "wm_task_type": game["task_type"],
        "wm_game_sha256": game["sha256"],
        "wm_manifest_sha256": manifest["manifest_sha256"],
        "wm_schedule_index": schedule_index,
        "wm_schedule_padding": False,
        "wm_trajectory_index": 0,
        "wm_episode_id": episode_id,
        "traj_uid": episode_id,
        "wm_step_idx": 0,
        "wm_prev_obs_text": "room",
        "wm_action_text": "look",
        "wm_next_obs_text": "room",
        "episode_success": score > 0,
        "episode_rewards": score,
        "score": score,
        "rollout_checkpoint_step": "15",
        "rollout_temperature": 1.0,
        "rollout_top_p": 1.0,
        "rollout_top_k": -1,
        "rollout_do_sample": True,
        "rollout_n": 1,
    }


def test_rollout_coverage_is_fail_closed_for_games_and_scores(tmp_path):
    manifest_module, _, coverage_module = _modules()
    _, config = _dataset(tmp_path)
    manifest = manifest_module.build_manifest(config)
    rows = [_transition(manifest, game, index, score=10.0 if index % 2 else 0.0) for index, game in enumerate(manifest["games"])]

    summary = coverage_module.validate_rollout(
        manifest,
        rows,
        expected_checkpoint_step="15",
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        do_sample=True,
    )

    assert summary["covered_games"] == summary["manifest_games"] == 3
    assert summary["stable_transition_ids"] == 3

    with pytest.raises(ValueError, match="coverage incomplete"):
        coverage_module.validate_rollout(
            manifest,
            rows[:-1],
            expected_checkpoint_step="15",
            temperature=1.0,
            top_p=1.0,
            top_k=-1,
            do_sample=True,
        )
    missing_score = [dict(row) for row in rows]
    missing_score[0].pop("score")
    with pytest.raises(ValueError, match="missing full-protocol fields"):
        coverage_module.validate_rollout(
            manifest,
            missing_score,
            expected_checkpoint_step="15",
            temperature=1.0,
            top_p=1.0,
            top_k=-1,
            do_sample=True,
        )
