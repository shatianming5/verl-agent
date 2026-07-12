import csv
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
STEPS = ["init", "15", "30", "45", "60", "75", "90", "105", "120", "135", "150"]


def _load_analyzer():
    path = REPO_ROOT / "scripts" / "bdiag_analyze.py"
    spec = importlib.util.spec_from_file_location("bdiag_analyze_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_episode_perplexity_uses_total_nll_not_mean_transition_ppl():
    module = _load_analyzer()
    base = {
        "game_id": "game",
        "episode_id": "episode",
        "success": True,
        "target_confidence_mean": 0.5,
        "raw_action_obs_cosine": 0.2,
    }
    episode = module.episode_aggregate(
        [
            {**base, "target_tokens": 1, "nll_sum": 1.0},
            {**base, "target_tokens": 9, "nll_sum": 9.0},
        ]
    )[0]

    assert episode["ce"] == 1.0
    assert episode["perplexity"] == pytest.approx(math.e)


def _transition_id(manifest_hash, episode_id, wm_step_idx):
    return hashlib.sha256(f"{manifest_hash}\0{episode_id}\0{wm_step_idx}".encode()).hexdigest()


def _write_synthetic_step(root, step, game_count=12):
    step_dir = root / f"step{step}"
    step_dir.mkdir(parents=True)
    manifest_hash = "a" * 64
    dump_rows = []
    score_rows = []
    step_number = 0 if step == "init" else int(step)
    for game_index in range(game_count):
        success = game_index % 2 == 1
        game_id = f"game-{game_index:03d}"
        episode_id = f"episode-{game_index:03d}"
        transition_id = _transition_id(manifest_hash, episode_id, 0)
        ce = 1.8 - 0.002 * step_number + (0.25 if success else -0.1) + game_index * 0.001
        confidence = math.exp(-ce)
        cosine = -0.2 + 0.001 * step_number + (0.15 if success else 0.0) + game_index * 0.002
        dump = {
            "schema_version": "wm_transition_v2",
            "wm_manifest_sha256": manifest_hash,
            "wm_game_id": game_id,
            "wm_episode_id": episode_id,
            "traj_uid": episode_id,
            "wm_task_type": "pick_and_place_simple",
            "wm_schedule_padding": False,
            "wm_step_idx": 0,
            "wm_prev_obs_text": "room",
            "wm_action_text": "look",
            "wm_next_obs_text": "same room",
            "episode_success": success,
        }
        dump_rows.append(dump)
        score_rows.append(
            {
                "checkpoint_step": step,
                "stable_transition_id": transition_id,
                "wm_manifest_sha256": manifest_hash,
                "wm_game_id": game_id,
                "wm_episode_id": episode_id,
                "traj_uid": episode_id,
                "wm_task_type": "pick_and_place_simple",
                "wm_step_idx": 0,
                "episode_success": success,
                "target_tokens": 1,
                "nll_sum": ce,
                "ce": ce,
                "perplexity": math.exp(ce),
                "target_confidence_mean": confidence,
                "raw_action_obs_cosine": cosine,
                "action_obs_cosine": cosine,
                "action_obs_cosine_semantics": "raw_action_end_to_observation_end",
                "prev_next_token_jaccard": 0.5 + game_index * 0.001,
                "action_next_token_jaccard": 0.1 + game_index * 0.001,
                "prev_obs_tokens": 5,
                "action_tokens": 2,
                "next_obs_tokens": 6,
                "token_nlls_json": json.dumps([ce]),
                "target_token_confidences_json": json.dumps([confidence]),
                "predicted_token_confidences_json": json.dumps([0.65 if success else 0.55]),
                "predicted_token_correct_json": json.dumps([1 if success else 0]),
            }
        )
    dump_path = step_dir / f"{step}.wm_transitions.jsonl"
    dump_path.write_text(
        "".join(json.dumps(row) + "\n" for row in dump_rows),
        encoding="utf-8",
    )
    with (step_dir / "scores.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(score_rows[0]))
        writer.writeheader()
        writer.writerows(score_rows)
    (step_dir / "coverage.json").write_text(
        json.dumps(
            {
                "covered_games": game_count,
                "manifest_games": game_count,
                "checkpoint_step": step,
            }
        ),
        encoding="utf-8",
    )


def test_small_or_legacy_analysis_is_rejected(tmp_path):
    dump_root = tmp_path / "rollouts"
    for step in STEPS:
        _write_synthetic_step(dump_root, step)
    out_dir = tmp_path / "analysis"
    manifest = {
        "schema_version": "alfworld_train_manifest_v1",
        "split": "train",
        "dataset_root": "/data",
        "task_type_ids": [1],
        "raw_traj_data_count": 12,
        "game_count": 12,
        "games": [
            {
                "game_id": f"game-{index:03d}",
                "gamefile": f"/data/game-{index:03d}",
                "task_type": "pick_and_place_simple",
                "sha256": f"{index:064x}",
            }
            for index in range(12)
        ],
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "bdiag_analyze.py"),
            "--dump-root",
            str(dump_root),
            "--manifest",
            str(manifest_path),
            "--exp",
            "synthetic",
            "--out-dir",
            str(out_dir),
            "--bootstrap",
            "20",
            "--gmm-bootstrap",
            "10",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    assert result.returncode != 0
    assert "exactly 3553" in result.stdout
    assert not (out_dir / "workstream_b_report_synthetic.md").exists()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("raw_action_obs_cosine", "", "Non-raw"),
        ("target_confidence_mean", "", "Missing teacher-forced"),
        ("wm_game_id", "tampered-game", "Score/dump immutable"),
        ("prev_next_token_jaccard", "", "invalid tokenizer overlap"),
        ("action_tokens", -1, "invalid tokenizer count"),
    ],
)
def test_analyzer_rejects_missing_required_transition_metric(
    tmp_path,
    monkeypatch,
    field,
    value,
    message,
):
    module = _load_analyzer()
    monkeypatch.setattr(module, "validate_step_provenance", lambda **kwargs: None)
    step_dir = tmp_path / "stepinit"
    step_dir.mkdir()
    episode_id = "episode"
    manifest_hash = "a" * 64
    transition_id = _transition_id(manifest_hash, episode_id, 0)
    source = {
        "workstream_b_protocol": "workstream_b_full_train_v2",
        "split": "train",
        "wm_manifest_sha256": manifest_hash,
        "wm_episode_id": episode_id,
        "traj_uid": episode_id,
        "wm_game_id": "game",
        "wm_gamefile": "/data/game",
        "wm_game_sha256": "b" * 64,
        "wm_task_type": "pick_and_place_simple",
        "wm_schedule_index": 0,
        "wm_schedule_padding": False,
        "wm_trajectory_index": 0,
        "wm_step_idx": 0,
        "episode_success": True,
        "episode_rewards": 10.0,
        "score": 10.0,
        "rollout_checkpoint_step": "init",
        "rollout_temperature": 1.0,
        "rollout_top_p": 1.0,
        "rollout_top_k": -1,
        "rollout_do_sample": True,
        "rollout_n": 1,
    }
    (step_dir / "init.wm_transitions.jsonl").write_text(
        json.dumps(source) + "\n",
        encoding="utf-8",
    )
    (step_dir / "coverage.json").write_text("{}\n", encoding="utf-8")
    score = {
        **source,
        "checkpoint_step": "init",
        "stable_transition_id": transition_id,
        "target_tokens": 1,
        "target_tokens_original": 1,
        "target_truncated": False,
        "target_token_coverage": 1.0,
        "nll_sum": 1.0,
        "ce": 1.0,
        "perplexity": math.e,
        "target_confidence_mean": math.exp(-1),
        "raw_action_obs_cosine": 0.2,
        "action_obs_cosine": 0.2,
        "action_obs_cosine_semantics": "raw_action_end_to_observation_end",
        "cosine_endpoint_valid": True,
        "target_token_ids_json": "[1]",
        "token_nlls_json": "[1.0]",
        "target_token_confidences_json": f"[{math.exp(-1)}]",
        "predicted_token_confidences_json": "[0.5]",
        "predicted_token_correct_json": "[1]",
        "prev_next_token_jaccard": 0.2,
        "action_next_token_jaccard": 0.1,
        "prev_obs_tokens": 3,
        "action_tokens": 1,
        "next_obs_tokens": 3,
    }
    score[field] = value
    with (step_dir / "scores.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(score))
        writer.writeheader()
        writer.writerow(score)

    with pytest.raises(ValueError, match=message):
        module.load_step(
            str(tmp_path),
            "init",
            manifest={"manifest_sha256": manifest_hash},
            manifest_path=str(tmp_path / "manifest.json"),
        )


def test_transition_gmm_bootstrap_keeps_all_rows_with_game_cluster_weights(
    monkeypatch,
):
    module = _load_analyzer()
    rows = [{"game_id": f"game-{game}", "ce": game + transition / 100} for game in range(10) for transition in range(game + 1)]
    calls = []

    def fake_fit(values, max_iter=200, sample_weights=None):
        values = list(values)
        weights = list(sample_weights)
        calls.append((len(values), weights))
        return {"ashman_d": 2.0, "overlap": 0.1}

    monkeypatch.setattr(module, "fit_gmm2", fake_fit)
    result = module.gmm_cluster_ci(
        rows,
        "ce",
        bootstrap=10,
        seed=0,
    )

    assert len(calls) == 10
    assert all(length == len(rows) for length, _ in calls)
    assert any(len(set(weights)) > 1 for _, weights in calls)
    assert result["gmm_ashman_d_ci_lo"] == 2.0


def test_analyzer_accepts_exact_full_step_with_owned_provenance(tmp_path):
    module = _load_analyzer()
    games = [
        {
            "game_id": f"game-{index:04d}",
            "gamefile": f"/data/game-{index:04d}/game.tw-pddl",
            "task_type": "pick_and_place_simple",
            "sha256": f"{index:064x}",
        }
        for index in range(3553)
    ]
    manifest = {
        "schema_version": "alfworld_train_manifest_v1",
        "split": "train",
        "dataset_root": "/data",
        "task_type_ids": [1],
        "raw_traj_data_count": 6374,
        "game_count": 3553,
        "games": games,
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    step_dir = tmp_path / "stepinit"
    step_dir.mkdir()
    dump_rows = []
    score_rows = []
    confidence = math.exp(-1.0)
    for index, game in enumerate(games):
        success = bool(index % 2)
        episode_id = f"{manifest['manifest_sha256']}:{game['game_id']}:trajectory0:schedule{index}"
        row = {
            "schema_version": "wm_transition_v2",
            "workstream_b_protocol": "workstream_b_full_train_v2",
            "split": "train",
            "wm_game_id": game["game_id"],
            "wm_gamefile": game["gamefile"],
            "wm_task_type": game["task_type"],
            "wm_game_sha256": game["sha256"],
            "wm_manifest_sha256": manifest["manifest_sha256"],
            "wm_schedule_index": index,
            "wm_schedule_padding": False,
            "wm_trajectory_index": 0,
            "wm_episode_id": episode_id,
            "traj_uid": episode_id,
            "wm_step_idx": 0,
            "wm_prev_obs_text": "room",
            "wm_action_text": "look",
            "wm_next_obs_text": "same room",
            "episode_success": success,
            "episode_rewards": 10.0 if success else 0.0,
            "score": 10.0 if success else 0.0,
            "rollout_checkpoint_step": "init",
            "rollout_temperature": 1.0,
            "rollout_top_p": 1.0,
            "rollout_top_k": -1,
            "rollout_do_sample": True,
            "rollout_n": 1,
        }
        dump_rows.append(row)
        score_rows.append(
            {
                **{
                    key: row[key]
                    for key in (
                        "workstream_b_protocol",
                        "split",
                        "wm_game_id",
                        "wm_gamefile",
                        "wm_task_type",
                        "wm_game_sha256",
                        "wm_manifest_sha256",
                        "wm_schedule_index",
                        "wm_schedule_padding",
                        "wm_trajectory_index",
                        "wm_episode_id",
                        "traj_uid",
                        "wm_step_idx",
                        "episode_success",
                        "episode_rewards",
                        "score",
                        "rollout_checkpoint_step",
                        "rollout_temperature",
                        "rollout_top_p",
                        "rollout_top_k",
                        "rollout_do_sample",
                        "rollout_n",
                    )
                },
                "checkpoint_step": "init",
                "stable_transition_id": _transition_id(
                    manifest["manifest_sha256"],
                    episode_id,
                    0,
                ),
                "episode_success": success,
                "target_tokens": 1,
                "target_tokens_original": 1,
                "target_truncated": False,
                "target_token_coverage": 1.0,
                "nll_sum": 1.0,
                "ce": 1.0,
                "perplexity": math.e,
                "target_confidence_mean": confidence,
                "raw_action_obs_cosine": 0.1 + index / 100000,
                "action_obs_cosine": 0.1 + index / 100000,
                "action_obs_cosine_semantics": "raw_action_end_to_observation_end",
                "cosine_endpoint_valid": True,
                "target_token_ids_json": "[1]",
                "token_nlls_json": "[1.0]",
                "target_token_confidences_json": json.dumps([confidence]),
                "predicted_token_confidences_json": "[0.5]",
                "predicted_token_correct_json": "[1]",
                "wm_manifest_sha256": manifest["manifest_sha256"],
                "wm_game_id": game["game_id"],
                "wm_episode_id": episode_id,
                "wm_task_type": game["task_type"],
                "wm_step_idx": 0,
                "prev_next_token_jaccard": 0.5,
                "action_next_token_jaccard": 0.1,
                "prev_obs_tokens": 3,
                "action_tokens": 1,
                "next_obs_tokens": 3,
            }
        )
    dump_path = step_dir / "init.wm_transitions.jsonl"
    dump_path.write_text(
        "".join(json.dumps(row) + "\n" for row in dump_rows),
        encoding="utf-8",
    )
    with (step_dir / "scores.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(score_rows[0]))
        writer.writeheader()
        writer.writerows(score_rows)
    coverage = module.validate_rollout(
        manifest,
        dump_rows,
        expected_checkpoint_step="init",
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        do_sample=True,
    )
    recorded_coverage = {
        **coverage,
        "manifest_path": str(manifest_path.resolve()),
        "dump_path": str(dump_path.resolve()),
    }
    (step_dir / "coverage.json").write_text(
        json.dumps(recorded_coverage),
        encoding="utf-8",
    )
    score_csv = step_dir / "scores.csv"
    (step_dir / "score_summary.json").write_text(
        json.dumps(
            {
                "raw_cosine_only": True,
                "rows": len(dump_rows),
                "transition_jsonl": str(dump_path.resolve()),
                "coverage": coverage,
                "provenance": {
                    "require_full_protocol": True,
                    "max_rows": 0,
                    "expected_checkpoint_step": "init",
                    "manifest": str(manifest_path.resolve()),
                    "expected_games": 3553,
                    "expected_raw_trajectories": 6374,
                    "checkpoints": [{"step": "init"}],
                    "output_csv": str(score_csv.resolve()),
                    "output_csv_sha256": hashlib.sha256(score_csv.read_bytes()).hexdigest(),
                    "transition_jsonl": str(dump_path.resolve()),
                    "transition_jsonl_sha256": hashlib.sha256(dump_path.read_bytes()).hexdigest(),
                },
            }
        ),
        encoding="utf-8",
    )

    rows = module.load_step(
        str(tmp_path),
        "init",
        manifest=manifest,
        manifest_path=str(manifest_path),
    )

    assert len(rows) == 3553
    with score_csv.open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="output CSV SHA256 mismatch"):
        module.load_step(
            str(tmp_path),
            "init",
            manifest=manifest,
            manifest_path=str(manifest_path),
        )
