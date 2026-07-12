import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _load_module():
    repo_root = Path(__file__).resolve().parents[2]
    path = repo_root / "scripts" / "wm_results_table.py"
    spec = importlib.util.spec_from_file_location(
        "wm_results_table_full_protocol_under_test",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _full_protocol_tree(tmp_path, module):
    out_root = tmp_path / "full"
    manifest_path = tmp_path / "manifest.json"
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
        "dataset_root": "/data/train",
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
    _write_json(manifest_path, manifest)

    inventory_dir = out_root / "preflight"
    separation_rows = []
    for step in module.FULL_PROTOCOL_STEPS[1:]:
        fingerprints = {}
        for seed in (0, 1):
            fingerprint = f"{seed + 1:01x}{int(step):04x}".ljust(64, "a")
            fingerprints[seed] = fingerprint
            _write_json(
                inventory_dir / f"seed{seed}_step{step}_actor_inventory.json",
                {
                    "schema_version": "wm_actor_shard_inventory_v1",
                    "checkpoint_step": int(step),
                    "actor_dir": str(tmp_path / f"seed{seed}" / f"global_step_{step}" / "actor"),
                    "content_sha256": fingerprint,
                },
            )
        separation_rows.append(
            {
                "step": int(step),
                "seed0_content_sha256": fingerprints[0],
                "seed1_content_sha256": fingerprints[1],
            }
        )
    separation_path = inventory_dir / "seed_checkpoint_separation.json"
    _write_json(
        separation_path,
        {
            "schema_version": "wm_seed_checkpoint_separation_v1",
            "seed0_checkpoint_root": str(tmp_path / "seed0"),
            "seed1_checkpoint_root": str(tmp_path / "seed1"),
            "checkpoints": separation_rows,
        },
    )

    for seed in (0, 1):
        for step in module.FULL_PROTOCOL_STEPS:
            step_dir = out_root / f"seed{seed}" / f"step{step}"
            dump_path = step_dir / f"{step}.wm_transitions.jsonl"
            dump_path.parent.mkdir(parents=True, exist_ok=True)
            dump_path.write_text("{}\n", encoding="utf-8")
            coverage = {
                "schema_version": "workstream_b_rollout_coverage_v1",
                "protocol": "workstream_b_full_train_v2",
                "manifest_sha256": manifest["manifest_sha256"],
                "raw_traj_data_count": 6374,
                "manifest_games": 3553,
                "covered_games": 3553,
                "min_trajectories_per_game": 1,
                "episodes": 3553,
                "padding_episodes": 31,
                "success_episodes": 1000,
                "failure_episodes": 2553,
                "transitions": 1,
                "stable_transition_ids": 1,
                "checkpoint_step": step,
                "decoding": {
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "top_k": -1,
                    "do_sample": True,
                },
                "manifest_path": str(manifest_path.resolve()),
                "dump_path": str(dump_path.resolve()),
            }
            _write_json(step_dir / "coverage.json", coverage)
            scores = step_dir / "scores.csv"
            scores.write_text("score\n1\n", encoding="utf-8")
            summary_coverage = {key: value for key, value in coverage.items() if key not in {"manifest_path", "dump_path"}}
            _write_json(
                step_dir / "score_summary.json",
                {
                    "raw_cosine_only": True,
                    "rows": 1,
                    "transition_jsonl": str(dump_path.resolve()),
                    "coverage": summary_coverage,
                    "provenance": {
                        "require_full_protocol": True,
                        "max_rows": 0,
                        "checkpoint_count": 1,
                        "expected_games": 3553,
                        "expected_raw_trajectories": 6374,
                        "expected_checkpoint_step": step,
                        "manifest": str(manifest_path.resolve()),
                        "output_csv": str(scores.resolve()),
                        "output_csv_sha256": hashlib.sha256(scores.read_bytes()).hexdigest(),
                        "transition_jsonl": str(dump_path.resolve()),
                        "transition_jsonl_sha256": hashlib.sha256(dump_path.read_bytes()).hexdigest(),
                        "checkpoints": [{"step": step}],
                    },
                },
            )
        analysis = out_root / f"seed{seed}" / "analysis"
        for name in (
            f"workstream_b_report_baseline_seed{seed}.md",
            f"bdiag_stats_baseline_seed{seed}.csv",
            f"paired_game_trends_baseline_seed{seed}.csv",
            f"token_calibration_baseline_seed{seed}.csv",
            f"tokenizer_overlap_control_baseline_seed{seed}.csv",
            "grouped_nested_hidden_probe.csv",
        ):
            path = analysis / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("artifact\n", encoding="utf-8")
        for level in ("episode", "transition"):
            for metric in module.FULL_PROTOCOL_METRICS:
                (analysis / f"line_{level}_{metric}_baseline_seed{seed}.png").write_bytes(b"png")
        for metric in module.FULL_PROTOCOL_METRICS:
            (analysis / f"hist_init_mid_150_{metric}_baseline_seed{seed}.png").write_bytes(b"png")
        (analysis / f"calibration_init_mid_150_baseline_seed{seed}.png").write_bytes(b"png")

    cross_csv = out_root / "cross_seed_trend_consistency.csv"
    cross_report = out_root / "cross_seed_trend_consistency.md"
    final_report = out_root / "WORKSTREAM_B_FULL_REPORT.md"
    for path in (cross_csv, cross_report, final_report):
        path.write_text("artifact\n", encoding="utf-8")
    marker_path = out_root / "FULL_PROTOCOL_DONE.json"
    _write_json(
        marker_path,
        {
            "schema_version": "workstream_b_full_driver_v1",
            "status": "complete",
            "seeds": [0, 1],
            "steps": list(module.FULL_PROTOCOL_STEPS),
            "manifest": str(manifest_path.resolve()),
            "expected_raw_trajectories": 6374,
            "expected_games": 3553,
            "actor_inventory_dir": str(inventory_dir.resolve()),
            "seed_checkpoint_separation": str(separation_path.resolve()),
            "cross_seed_trend_csv": str(cross_csv.resolve()),
            "cross_seed_trend_report": str(cross_report.resolve()),
        },
    )
    return marker_path


def test_results_table_accepts_only_complete_full_protocol_marker(tmp_path):
    module = _load_module()
    marker = _full_protocol_tree(tmp_path, module)

    rows = module.parse_full_protocol_done(str(marker))

    assert [row["run_key"] for row in rows] == [
        "grpo_baseline_s0",
        "grpo_baseline_s1",
    ]
    assert all(module.has_complete_diagnostic_result(row) for row in rows)
    assert all(not row["diagnostic_transition_jsonl"] for row in rows)
    assert all(not row["diagnostic_command"] for row in rows)
    for row in rows:
        row["expected"] = "yes"
    module.annotate_artifact_coverage(rows)
    assert all(row["has_diagnostic"] == "yes" for row in rows)
    assert all("diagnostic_report_csv" not in row["missing_artifacts"] for row in rows)
    assert all("diagnostic_report_svg" not in row["missing_artifacts"] for row in rows)
    assert all("diagnostic_required_figures" not in row["missing_artifacts"] for row in rows)
    built = module.build_records(
        eval_paths=[],
        train_logs=[],
        diagnostic_paths=[str(marker)],
    )
    assert len(built) == 2
    assert all(module.row_has_diagnostic(row) for row in built)


def test_results_table_rejects_missing_step_coverage(tmp_path):
    module = _load_module()
    marker = _full_protocol_tree(tmp_path, module)
    (marker.parent / "seed1" / "step75" / "coverage.json").unlink()

    with pytest.raises(ValueError, match="coverage"):
        module.parse_full_protocol_done(str(marker))


def test_results_table_rejects_legacy_shared_diagnostic_command(tmp_path):
    module = _load_module()
    row = {
        "run_key": "grpo_baseline_s0",
        "diagnostic_summary_path": "/legacy/checkpoint_scores_summary.json",
        "diagnostic_final_step": "150",
        "diagnostic_token_mean_ce": 1.2,
        "diagnostic_action_obs_cosine": 0.3,
        "diagnostic_command": "TRANSITIONS_JSONL=/legacy/shared.jsonl old-runner",
    }

    module.annotate_diagnostic_commands(
        [row],
        work_root="/work",
        diagnostic_script="scripts/run_wm_checkpoint_diagnostics.sh",
        diagnostic_steps="init 30 60 90 120 150",
        transition_step="150",
    )

    assert not module.has_complete_diagnostic_result(row)
    assert row["diagnostic_readiness"] == "legacy_diagnostic_rejected"
    assert row["diagnostic_command"] == ""
    assert row["diagnostic_transition_jsonl"] == ""
    legacy = tmp_path / "checkpoint_scores_summary.json"
    _write_json(legacy, {"checkpoints": [{"checkpoint_step": "150"}]})
    with pytest.raises(ValueError, match="Legacy diagnostic summary"):
        module.build_records(
            eval_paths=[],
            train_logs=[],
            diagnostic_paths=[str(legacy)],
        )
