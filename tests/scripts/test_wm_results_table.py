import csv
import importlib.util
import json
import os
import pytest
import sys
from pathlib import Path


def _load_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "wm_results_table.py"
    spec = importlib.util.spec_from_file_location("wm_results_table_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_eval_result_infers_obs_ce_metadata(tmp_path):
    module = _load_module()
    result_path = tmp_path / "eval10x_wm_obs_ce_l0p01_s0_results.txt"
    result_path.write_text(
        "\n".join(
            [
                "EVAL10X_START label=wm_obs_ce_l0p01_s0 ckpt=/work/checkpoints/grpo_qwen2.5_1.5b_alfworld_seed0_wm_obs_ce_l0p01/global_step_150 n=10 val_size=128 dataset=eval_in_distribution cuda=4,5 Tue",
                "eval 0 env.seed=0 success_rate=0.7500 log=/work/logs/eval0.log",
                "EVAL10X_RESULT n=10 mean=0.7400 std=0.0300",
                "EVAL10X_DONE label=wm_obs_ce_l0p01_s0 Tue",
            ]
        ),
        encoding="utf-8",
    )

    row = module.parse_eval_result(str(result_path))

    assert row["run_key"] == "obs_ce_l0p01_s0"
    assert row["objective"] == "obs_ce"
    assert row["seed"] == "0"
    assert row["lambda_obs"] == "0.01"
    assert row["eval_mean"] == 0.74
    assert row["eval_std"] == 0.03
    assert row["eval_n"] == "10"
    assert row["eval_checkpoint_path"].endswith("global_step_150")


def test_incomplete_eval_file_does_not_count_as_eval_result(tmp_path):
    module = _load_module()
    result_path = tmp_path / "eval10x_wm_obs_ce_l0p01_s0_results.txt"
    result_path.write_text(
        "\n".join(
            [
                "EVAL10X_START label=wm_obs_ce_l0p01_s0 ckpt=/work/checkpoints/grpo_qwen2.5_1.5b_alfworld_seed0_wm_obs_ce_l0p01/global_step_150 n=10 val_size=128 dataset=eval_in_distribution cuda=4,5 Tue",
                "eval 0 env.seed=0 success_rate=0.7500 log=/work/logs/eval0.log",
            ]
        ),
        encoding="utf-8",
    )

    row = module.parse_eval_result(str(result_path))
    row["expected"] = "yes"
    row["train_log_path"] = "/work/logs/train.log"
    rows = [row]
    module.annotate_eval_readiness(rows, eval_cuda="4,5", eval_n="10", eval_script="/root/grpo/eval10x_alfworld.sh")
    module.annotate_artifact_coverage(rows)
    status = dict(module.goal_rd_deliverable_status(rows))

    assert row["status"] == "eval_incomplete"
    assert row["eval_readiness"] == "eval_incomplete"
    assert row["has_eval"] == "no"
    assert row["missing_artifacts"] == "eval,diagnostic"
    assert row["final_report_readiness"] == "missing:eval,diagnostic"
    assert status["Result table status"].startswith("0/1 tracked run(s) have eval results")
    assert "eval_incomplete" in status["Result table status"]


def test_parse_train_log_collects_progress_val_and_world_model_metrics(tmp_path):
    module = _load_module()
    log_path = tmp_path / "grpo_qwen2.5_1.5b_alfworld_seed1_wmlat_l0p001_20260630_010203.log"
    log_path.write_text(
        "\n".join(
            [
                "RUN_ALFWORLD_OFFICIAL seed=1 tag=wmlat_l0p001 cuda=6,7 tp=2 micro=16 gmu=0.6 ckpt=/work/checkpoints/grpo_qwen2.5_1.5b_alfworld_seed1_wmlat_l0p001",
                "actor_rollout_ref.actor.world_model.lambda_latent=0.001",
                "Training Progress: 145/150",
                "{'val/success_rate': 0.7100, 'actor/wm_latent_loss': 0.1800, 'actor/wm_cosine': 0.8200}",
                "training/global_step:148.000",
                "saved checkpoint /work/checkpoints/grpo_qwen2.5_1.5b_alfworld_seed1_wmlat_l0p001/global_step_135",
                "Training Progress: 150/150",
                "{'val/success_rate': 0.7300, 'actor/wm_latent_loss': 0.1200, 'actor/wm_cosine': 0.8600}",
                "saved checkpoint /work/checkpoints/grpo_qwen2.5_1.5b_alfworld_seed1_wmlat_l0p001/global_step_150",
            ]
        ),
        encoding="utf-8",
    )

    row = module.parse_train_log(str(log_path))

    assert row["run_key"] == "latent_l0p001_s1"
    assert row["objective"] == "latent"
    assert row["lambda_latent"] == "0.001"
    assert row["train_step"] == 150
    assert row["train_total_steps"] == 150
    assert row["latest_checkpoint_step"] == 150
    assert row["latest_checkpoint_path"].endswith("global_step_150")
    assert row["val_success_last"] == 0.73
    assert row["val_success_best"] == 0.73
    assert row["actor/wm_latent_loss"] == 0.12
    assert row["actor/wm_cosine"] == 0.86
    assert row["wm_loss_last"] == 0.12
    assert row["wm_cosine_last"] == 0.86
    assert row["wm_metric_last"] == "latent_loss=0.120, cosine=0.860"


def test_parse_train_log_handles_live_tqdm_step_lines(tmp_path):
    module = _load_module()
    log_path = tmp_path / "grpo_qwen2.5_1.5b_alfworld_seed0_wmlat_l0p001_s0_20260630_010203.log"
    log_path.write_text(
        "\n".join(
            [
                "RUN_WM_LATENT_SEED seed=0 tag=wmlat_l0p001_s0 cuda=0,3 lambda_latent=0.001 total_epochs=150 rollout_data_dir=/work/rollouts",
                "RUN_ALFWORLD_OFFICIAL seed=0 tag=wmlat_l0p001_s0 cuda=0,3 ckpt=/work/checkpoints/grpo_qwen2.5_1.5b_alfworld_seed0_wmlat_l0p001_s0",
                "Training Progress:  80%|████████  | 120/150 [28:10:17<8:28:26, 1016.88s/it]",
                "step:120 - world_model/latent_loss:0.211 - world_model/latent_cosine:0.789 - val/success_rate:0.609 - training/global_step:120.000",
                "Saving to /work/checkpoints/grpo_qwen2.5_1.5b_alfworld_seed0_wmlat_l0p001_s0/global_step_120",
                "Training Progress:  81%|████████  | 121/150 [28:25:05<7:52:45, 978.12s/it]",
                "step:121 - world_model/latent_loss:0.257 - world_model/latent_cosine:0.743 - training/global_step:121.000",
            ]
        ),
        encoding="utf-8",
    )

    row = module.parse_train_log(str(log_path))

    assert row["run_key"] == "latent_l0p001_s0"
    assert row["train_step"] == 121
    assert row["train_total_steps"] == 150
    assert row["latest_checkpoint_step"] == 120
    assert row["latest_checkpoint_path"].endswith("global_step_120")
    assert row["val_success_last"] == 0.609
    assert row["world_model/latent_loss"] == 0.257
    assert row["wm_loss_last"] == 0.257
    assert row["wm_cosine_last"] == 0.743
    assert row["wm_metric_last"] == "latent_loss=0.257, cosine=0.743"


def test_obs_ce_l_token_is_not_misclassified_by_disabled_latent_config(tmp_path):
    module = _load_module()
    log_path = tmp_path / "grpo_qwen2.5_1.5b_alfworld_seed0_wm_obs_ce_l0p03_s0_20260630_010203.log"
    log_path.write_text(
        "\n".join(
            [
                "RUN_WM_OBS_CE_SEED seed=0 tag=wm_obs_ce_l0p03_s0 cuda=2,3 lambda_obs=0.03 total_epochs=150",
                "RUN_ALFWORLD_OFFICIAL seed=0 tag=wm_obs_ce_l0p03_s0 cuda=2,3 ckpt=/work/checkpoints/grpo_qwen2.5_1.5b_alfworld_seed0_wm_obs_ce_l0p03_s0",
                "actor_rollout_ref.actor.world_model.obs_ce_enable=True",
                "actor_rollout_ref.actor.world_model.lambda_obs=0.03",
                "actor_rollout_ref.actor.world_model.latent_enable=False",
                "Training Progress:   0%|          | 0/150 [00:00<?, ?it/s]",
                "step:0 - val/success_rate:0.078 - training/global_step:0.000",
            ]
        ),
        encoding="utf-8",
    )

    rows = module.build_records(
        eval_paths=[],
        train_logs=[str(log_path)],
        diagnostic_paths=[],
        expected_runs=module.GOAL_RD_EXPECTED_RUNS,
    )
    module.annotate_artifact_coverage(rows)
    by_key = {row["run_key"]: row for row in rows}

    assert len(rows) == 11
    assert "latent_l0p03_s0" not in by_key
    row = by_key["obs_ce_l0p03_s0"]
    assert row["objective"] == "obs_ce"
    assert row["lambda_obs"] == "0.03"
    assert row["lambda_latent"] == ""
    assert row["has_train_log"] == "yes"


def test_annotate_eval_readiness_generates_command_only_when_ready(tmp_path):
    module = _load_module()
    ready = {
        "run_key": "latent_l0p001_s1",
        "tag": "wmlat_l0p001_s1",
        "latest_checkpoint_step": 150,
        "latest_checkpoint_path": "/work/checkpoints/grpo_qwen2.5_1.5b_alfworld_seed1_wmlat_l0p001_s1/global_step_150",
    }
    waiting = {
        "run_key": "obs_ce_l0p01_s0",
        "latest_checkpoint_step": 120,
        "latest_checkpoint_path": "/work/checkpoints/grpo_qwen2.5_1.5b_alfworld_seed0_wm_obs_ce_l0p01_s0/global_step_120",
        "train_log_path": "/work/logs/train.log",
    }
    evaluated = {
        "run_key": "grpo_baseline_s0",
        "eval_mean": 0.729,
        "latest_checkpoint_step": 150,
        "latest_checkpoint_path": "/work/checkpoints/grpo/global_step_150",
    }

    rows = [ready, waiting, evaluated]
    module.annotate_eval_readiness(rows, eval_cuda="4,5", eval_n="10", eval_script="scripts/eval10x_alfworld.sh")

    assert ready["eval_readiness"] == "ready_for_eval"
    assert ready["eval_command"] == (
        "CKPT=/work/checkpoints/grpo_qwen2.5_1.5b_alfworld_seed1_wmlat_l0p001_s1/global_step_150 "
        "LABEL=wmlat_l0p001_s1 CUDA_VISIBLE_DEVICES=4,5 N_EVALS=10 bash scripts/eval10x_alfworld.sh"
    )
    assert ready["eval_target_checkpoint_path"].endswith("global_step_150")
    assert waiting["eval_readiness"] == "waiting_for_checkpoint"
    assert waiting["eval_command"] == ""
    assert waiting["eval_target_checkpoint_path"] == ""
    assert evaluated["eval_readiness"] == "evaluated"
    assert evaluated["eval_command"] == ""
    assert evaluated["eval_target_checkpoint_path"] == ""

    markdown = module.render_markdown(rows)
    assert "## Eval Readiness" in markdown
    assert "- ready_for_eval: `1`" in markdown
    assert "- waiting_for_checkpoint: `1`" in markdown
    assert "- evaluated: `1`" in markdown
    assert "Ready eval commands:" in markdown
    assert "latent_l0p001_s1" in markdown
    assert "checkpoint `/work/checkpoints/grpo_qwen2.5_1.5b_alfworld_seed1_wmlat_l0p001_s1/global_step_150`" in markdown
    assert "CUDA_VISIBLE_DEVICES=4,5" in markdown
    assert "CKPT=/work/checkpoints/grpo_qwen2.5_1.5b_alfworld_seed1_wmlat_l0p001_s1/global_step_150" in markdown


def test_merged_train_logs_keep_latest_checkpoint_step_and_path_paired(tmp_path):
    module = _load_module()
    checkpoint_root = "/work/checkpoints/grpo_qwen2.5_1.5b_alfworld_seed0_wmlat_l0p001_s0"
    older_log = tmp_path / "older.log"
    older_log.write_text(
        "\n".join(
            [
                f"RUN_ALFWORLD_OFFICIAL seed=0 tag=wmlat_l0p001_s0 cuda=0,3 ckpt={checkpoint_root}",
                "Training Progress: 120/150",
                f"saved {checkpoint_root}/global_step_120",
            ]
        ),
        encoding="utf-8",
    )
    newer_log = tmp_path / "newer.log"
    newer_log.write_text(
        "\n".join(
            [
                f"RUN_ALFWORLD_OFFICIAL seed=0 tag=wmlat_l0p001_s0 cuda=0,3 ckpt={checkpoint_root}",
                "Training Progress: 150/150",
                f"saved {checkpoint_root}/global_step_150",
            ]
        ),
        encoding="utf-8",
    )

    for train_logs in ([str(older_log), str(newer_log)], [str(newer_log), str(older_log)]):
        rows = module.build_records(eval_paths=[], train_logs=train_logs, diagnostic_paths=[])
        module.annotate_eval_readiness(
            rows,
            eval_cuda="4,5",
            eval_n="10",
            eval_script="/root/grpo/eval10x_alfworld.sh",
        )

        row = rows[0]
        assert row["latest_checkpoint_step"] == 150
        assert row["latest_checkpoint_path"] == f"{checkpoint_root}/global_step_150"
        assert row["eval_target_checkpoint_path"] == f"{checkpoint_root}/global_step_150"
        assert row["eval_readiness"] == "ready_for_eval"
        assert "global_step_150" in row["eval_command"]
        assert "global_step_120" not in row["eval_command"]
        assert ";" not in row["latest_checkpoint_path"]
        assert ";" not in row["eval_target_checkpoint_path"]
        assert ";" not in row["eval_command"]


def test_annotate_checkpoint_backups_marks_backup_coverage(tmp_path):
    module = _load_module()
    work_root = tmp_path / "work"
    backed_checkpoint = work_root / "checkpoints" / "run_a" / "global_step_150"
    missing_checkpoint = work_root / "checkpoints" / "run_b" / "global_step_135"
    backed_backup = work_root / "checkpoints_backup" / "run_a" / "global_step_150"
    backed_checkpoint.mkdir(parents=True)
    missing_checkpoint.mkdir(parents=True)
    backed_backup.mkdir(parents=True)
    rows = [
        {
            "run_key": "run_a",
            "latest_checkpoint_step": 150,
            "latest_checkpoint_path": str(backed_checkpoint),
        },
        {
            "run_key": "run_b",
            "latest_checkpoint_step": 135,
            "latest_checkpoint_path": str(missing_checkpoint),
        },
        {"run_key": "run_c"},
    ]

    module.annotate_checkpoint_backups(rows, work_root=str(work_root))

    assert rows[0]["checkpoint_backup_status"] == "backed_up"
    assert rows[0]["checkpoint_backup_path"] == str(backed_backup)
    assert rows[1]["checkpoint_backup_status"] == "missing_backup"
    assert rows[1]["checkpoint_backup_path"] == str(work_root / "checkpoints_backup" / "run_b" / "global_step_135")
    assert rows[2]["checkpoint_backup_status"] == "no_checkpoint"
    assert rows[2]["checkpoint_backup_path"] == ""


def test_annotate_checkpoint_backups_falls_back_to_eval_checkpoint(tmp_path):
    module = _load_module()
    work_root = tmp_path / "work"
    eval_checkpoint = work_root / "checkpoints" / "run_eval" / "global_step_150"
    eval_backup = work_root / "checkpoints_backup" / "run_eval" / "global_step_150"
    eval_checkpoint.mkdir(parents=True)
    eval_backup.mkdir(parents=True)
    rows = [
        {
            "run_key": "run_eval",
            "latest_checkpoint_step": 150,
            "latest_checkpoint_path": "",
            "eval_target_checkpoint_path": str(eval_checkpoint),
        }
    ]

    module.annotate_checkpoint_backups(rows, work_root=str(work_root))

    assert rows[0]["checkpoint_backup_status"] == "backed_up"
    assert rows[0]["checkpoint_backup_path"] == str(eval_backup)


def test_annotate_train_commands_generates_standard_goal_rd_commands():
    module = _load_module()
    rows = [
        {
            "run_key": "grpo_baseline_s0",
            "objective": "grpo_baseline",
            "tag": "official_4to5",
            "seed": "0",
        },
        {
            "run_key": "obs_ce_l0p03_s1",
            "objective": "obs_ce",
            "tag": "wm_obs_ce_l0p03_s1",
            "seed": "1",
            "lambda_obs": "0.03",
        },
        {
            "run_key": "latent_l0p001_s1",
            "objective": "latent",
            "tag": "wmlat_l0p001_s1",
            "seed": "1",
            "lambda_latent": "0.001",
            "train_cuda": "4,5",
        },
    ]

    module.annotate_train_commands(rows, train_cuda="6,7", train_script="/root/grpo/run_seed_alfworld_official.sh")

    assert rows[0]["train_command"] == "TAG=official_4to5 CUDA_VISIBLE_DEVICES=6,7 bash /root/grpo/run_seed_alfworld_official.sh 0"
    assert rows[1]["train_command"] == (
        "TAG=wm_obs_ce_l0p03_s1 LAMBDA_OBS=0.03 CUDA_VISIBLE_DEVICES=6,7 "
        "bash /root/grpo/run_seed_alfworld_official.sh 1"
    )
    assert rows[2]["train_command"] == (
        "TAG=wmlat_l0p001_s1 LAMBDA_LATENT=0.001 CUDA_VISIBLE_DEVICES=4,5 "
        "bash /root/grpo/run_seed_alfworld_official.sh 1"
    )


def test_annotate_train_commands_can_request_rollout_dumps():
    module = _load_module()
    rows = [
        {
            "run_key": "latent_l0p001_s1",
            "objective": "latent",
            "tag": "wmlat_l0p001_s1",
            "seed": "1",
            "lambda_latent": "0.001",
        }
    ]

    module.annotate_train_commands(
        rows,
        train_cuda="4,5",
        train_script="/root/grpo/run_seed_alfworld_official.sh",
        dump_rollouts=True,
    )

    assert rows[0]["train_command"] == (
        "TAG=wmlat_l0p001_s1 WM_DUMP_ROLLOUTS=1 LAMBDA_LATENT=0.001 CUDA_VISIBLE_DEVICES=4,5 "
        "bash /root/grpo/run_seed_alfworld_official.sh 1"
    )


def test_annotate_diagnostic_commands_uses_rollout_dir_when_available(tmp_path):
    module = _load_module()
    work_root = tmp_path / "work"
    checkpoint_root = work_root / "checkpoints" / "grpo_qwen2.5_1.5b_alfworld_seed1_wmlat_l0p001_s1"
    rollout_dir = work_root / "logs" / "world_model_rollouts" / "grpo_qwen2.5_1.5b_alfworld_seed1_wmlat_l0p001_s1"
    rollout_dir.mkdir(parents=True)
    transition_jsonl = rollout_dir / "150.wm_transitions.jsonl"
    transition_jsonl.write_text('{"ok": true}\n', encoding="utf-8")
    rows = [
        {
            "run_key": "latent_l0p001_s1",
            "tag": "wmlat_l0p001_s1",
            "latest_checkpoint_path": str(checkpoint_root / "global_step_150"),
            "rollout_data_dir": str(rollout_dir),
        }
    ]

    module.annotate_diagnostic_commands(
        rows,
        work_root=str(work_root),
        diagnostic_script="scripts/run_wm_checkpoint_diagnostics.sh",
        diagnostic_steps="init 30 60 90 120 150",
        transition_step="150",
    )

    assert rows[0]["diagnostic_checkpoint_root"] == str(checkpoint_root)
    assert rows[0]["diagnostic_transition_jsonl"] == str(transition_jsonl)
    assert rows[0]["diagnostic_readiness"] == "ready_for_diagnostic"
    assert rows[0]["diagnostic_command"] == (
        f"TRANSITIONS_JSONL={transition_jsonl} "
        f"CKPT_ROOT={checkpoint_root} "
        "TAG=wmlat_l0p001_s1 STEPS='init 30 60 90 120 150' bash scripts/run_wm_checkpoint_diagnostics.sh"
    )

    markdown = module.render_markdown(rows)
    assert "## Diagnostic Commands" in markdown
    assert "latent_l0p001_s1" in markdown
    assert "150.wm_transitions.jsonl" in markdown


def test_annotate_diagnostic_commands_infers_standard_rollout_path_and_waits_for_step150(tmp_path):
    module = _load_module()
    work_root = tmp_path / "work"
    ready_checkpoint_root = work_root / "checkpoints" / "grpo_qwen2.5_1.5b_alfworld_seed0_wm_obs_ce_l0p05_s0"
    ready_rollout_dir = work_root / "logs" / "world_model_rollouts" / ready_checkpoint_root.name
    ready_rollout_dir.mkdir(parents=True)
    ready_transition_jsonl = ready_rollout_dir / "150.wm_transitions.jsonl"
    ready_transition_jsonl.write_text('{"ok": true}\n', encoding="utf-8")
    missing_checkpoint_root = work_root / "checkpoints" / "grpo_qwen2.5_1.5b_alfworld_seed2_official_s2"
    rows = [
        {
            "run_key": "obs_ce_l0p05_s0",
            "tag": "wm_obs_ce_l0p05_s0",
            "latest_checkpoint_path": str(ready_checkpoint_root / "global_step_150"),
        },
        {
            "run_key": "obs_ce_l0p03_s0",
            "tag": "wm_obs_ce_l0p03_s0",
            "latest_checkpoint_path": str(work_root / "checkpoints" / "grpo_qwen2.5_1.5b_alfworld_seed0_wm_obs_ce_l0p03_s0" / "global_step_150"),
            "diagnostic_summary_path": "/work/logs/world_model_diagnostics/wm_obs_ce_l0p03_s0/checkpoint_scores_summary.json",
            "diagnostic_final_step": "150",
            "diagnostic_token_mean_ce": 1.4,
            "diagnostic_action_obs_cosine": 0.4,
            "diagnostic_command": "python existing.py",
        },
        {
            "run_key": "latent_l0p001_s0",
            "tag": "wmlat_l0p001_s0",
            "latest_checkpoint_step": 120,
            "latest_checkpoint_path": "/work/checkpoints/grpo_qwen2.5_1.5b_alfworld_seed0_wmlat_l0p001_s0/global_step_120",
        },
        {
            "run_key": "grpo_baseline_s2",
            "tag": "official_s2",
            "latest_checkpoint_path": str(missing_checkpoint_root / "global_step_150"),
        },
    ]

    module.annotate_diagnostic_commands(
        rows,
        work_root=str(work_root),
        diagnostic_script="scripts/run_wm_checkpoint_diagnostics.sh",
        diagnostic_steps="init 60 120 150",
        transition_step="150",
    )

    assert rows[0]["diagnostic_transition_jsonl"] == str(ready_transition_jsonl)
    assert rows[0]["diagnostic_readiness"] == "ready_for_diagnostic"
    assert "STEPS='init 60 120 150'" in rows[0]["diagnostic_command"]
    assert rows[1]["diagnostic_command"] == "python existing.py"
    assert rows[1]["diagnostic_readiness"] == "diagnosed"
    assert rows[2].get("diagnostic_command") in ("", None)
    assert rows[2]["diagnostic_readiness"] == "waiting_for_checkpoint"
    assert rows[3]["diagnostic_transition_jsonl"] == str(
        work_root / "logs" / "world_model_rollouts" / missing_checkpoint_root.name / "150.wm_transitions.jsonl"
    )
    assert rows[3].get("diagnostic_command") in ("", None)
    assert rows[3]["diagnostic_readiness"] == "missing_transition_dump"


def test_objective_coverage_summarizes_eval_and_diagnostics():
    module = _load_module()
    rows = [
        {
            "run_key": "grpo_baseline_s0",
            "objective": "grpo_baseline",
            "seed": "0",
            "eval_readiness": "evaluated",
            "diagnostic_summary_path": "/work/diag/base.json",
            "diagnostic_final_step": "150",
            "diagnostic_token_mean_ce": 1.5,
            "diagnostic_action_obs_cosine": 0.3,
        },
        {
            "run_key": "obs_ce_l0p01_s0",
            "objective": "obs_ce",
            "seed": "0",
            "eval_readiness": "waiting_for_checkpoint",
        },
        {
            "run_key": "obs_ce_l0p01_s1",
            "objective": "obs_ce",
            "seed": "1",
            "eval_readiness": "ready_for_eval",
        },
        {
            "run_key": "latent_l0p001_s1",
            "objective": "latent",
            "seed": "1",
            "eval_readiness": "eval_incomplete",
            "diagnostic_final_step": "150",
            "diagnostic_token_mean_ce": 1.4,
            "diagnostic_action_obs_cosine": 0.4,
        },
    ]

    coverage = module.objective_coverage(rows)

    assert coverage[0]["objective"] == "grpo_baseline"
    assert coverage[0]["evaluated"] == 1
    assert coverage[0]["diagnosed"] == 1
    assert coverage[1]["objective"] == "obs_ce"
    assert coverage[1]["seeds"] == "0,1"
    assert coverage[1]["ready_for_eval"] == 1
    assert coverage[1]["waiting_for_checkpoint"] == 1
    assert coverage[2]["objective"] == "latent"
    assert coverage[2]["eval_incomplete"] == 1
    assert coverage[2]["diagnosed"] == 1

    markdown = module.render_markdown(rows)
    assert "## Objective Coverage" in markdown
    assert "| grpo_baseline | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 1 |" in markdown
    assert "| obs_ce | 2 | 0,1 | 0 | 1 | 1 | 0 | 0 | 0 |" in markdown
    assert "| latent | 1 | 1 | 0 | 0 | 0 | 1 | 0 | 1 |" in markdown


def test_result_summary_groups_eval_by_objective_and_lambda():
    module = _load_module()
    rows = [
        {"run_key": "grpo_baseline_s0", "objective": "grpo_baseline", "seed": "0", "eval_mean": 0.70, "eval_std": 0.02},
        {"run_key": "grpo_baseline_s1", "objective": "grpo_baseline", "seed": "1", "eval_mean": 0.72, "eval_std": 0.04},
        {"run_key": "obs_ce_l0p01_s0", "objective": "obs_ce", "seed": "0", "lambda_obs": "0.01", "eval_mean": 0.73, "eval_std": 0.03},
        {"run_key": "obs_ce_l0p01_s1", "objective": "obs_ce", "seed": "1", "lambda_obs": "0.01", "eval_mean": 0.75, "eval_std": 0.05},
        {"run_key": "latent_l0p001_s0", "objective": "latent", "seed": "0", "lambda_latent": "0.001"},
    ]

    summary = module.result_summary(rows)

    assert summary[0]["condition"] == "GRPO baseline"
    assert summary[0]["evaluated"] == 2
    assert summary[0]["eval_mean"] == "0.71"
    assert summary[0]["delta_vs_baseline"] == "0"
    assert summary[1]["condition"] == "obs_ce lambda_obs=0.01"
    assert summary[1]["evaluated"] == 2
    assert summary[1]["eval_mean"] == "0.74"
    assert summary[1]["mean_eval_std"] == "0.04"
    assert summary[1]["delta_vs_baseline"] == "0.03"
    assert summary[2]["condition"] == "latent lambda_latent=0.001"
    assert summary[2]["evaluated"] == 0

    markdown = module.render_markdown(rows)
    assert "## Result Summary" in markdown
    assert "| obs_ce lambda_obs=0.01 | obs_ce | 0.01 |  | 2 | 2 | 0,1 | 0.7400 | 0.0141 | 0.0400 | 0.0300 |" in markdown


def test_goal_rd_deliverable_status_maps_report_requirements():
    module = _load_module()
    rows = [
        {
            "run_key": "grpo_baseline_s0",
            "objective": "grpo_baseline",
            "seed": "0",
            "expected": "yes",
            "eval_mean": 0.70,
            "train_log_path": "/work/logs/baseline.log",
            "eval_readiness": "evaluated",
            "diagnostic_readiness": "diagnosed",
            "diagnostic_summary_path": "/work/logs/world_model_diagnostics/base/checkpoint_scores_summary.json",
            "diagnostic_final_step": "150",
            "diagnostic_token_mean_ce": 1.5,
            "diagnostic_action_obs_cosine": 0.3,
        },
        {
            "run_key": "obs_ce_l0p01_s0",
            "objective": "obs_ce",
            "seed": "0",
            "expected": "yes",
            "lambda_obs": "0.01",
            "eval_mean": 0.73,
            "train_log_path": "/work/logs/obs.log",
            "eval_readiness": "evaluated",
            "diagnostic_readiness": "diagnosed",
            "diagnostic_final_step": "150",
            "diagnostic_token_mean_ce": 1.4,
            "diagnostic_action_obs_cosine": 0.4,
            "diagnostic_success_failure_ce_gap": 0.2,
        },
        {
            "run_key": "latent_l0p001_s0",
            "objective": "latent",
            "seed": "0",
            "lambda_latent": "0.001",
            "expected": "yes",
            "eval_readiness": "waiting_for_checkpoint",
            "diagnostic_readiness": "waiting_for_checkpoint",
        },
        {
            "run_key": "latent_s0",
            "objective": "latent",
            "seed": "0",
            "eval_mean": 0.99,
            "eval_readiness": "evaluated",
            "diagnostic_readiness": "diagnosed",
        },
    ]

    status = dict(module.goal_rd_deliverable_status(rows, branch="world-model-latent-objective"))

    assert status["Branch name"] == "world-model-latent-objective"
    assert status["Result table status"].startswith("2/3 tracked run(s) have eval results")
    assert "waiting_for_checkpoint" in status["Result table status"]
    assert "Raw observation CE: positive so far" in status["Raw observation CE interpretation"]
    assert "evidence available" in status["World-model feature interpretation"]
    assert status["Latent alignment interpretation"] == "Latent alignment: pending; no latent eval10x result is available yet."

    markdown = module.render_markdown(rows, branch="world-model-latent-objective")
    assert "## GOAL_RD Deliverable Status" in markdown
    assert "- GOAL_RD summary scope: `3` expected run(s)" in markdown
    assert "- Additional observed runs: `1`" in markdown
    assert "- Branch name: `world-model-latent-objective`" in markdown
    assert "- Exact commands/configs: `per-run launch lines" in markdown
    assert "- Training log coverage: `2/3 tracked run(s) have training logs`" in markdown
    assert "| latent_s0 |" in markdown
    assert "| latent | latent |  |  | 1 | 1 | 0 | 0.9900 |" not in markdown


def test_expected_run_coverage_reports_missing_artifacts():
    module = _load_module()

    rows = module.build_records(
        eval_paths=[],
        train_logs=[],
        diagnostic_paths=[],
        expected_runs=["obs_ce_l0p01_s0:objective=obs_ce,seed=0,lambda_obs=0.01,tag=wm_obs_ce_l0p01_s0"],
    )
    module.annotate_eval_readiness(rows, eval_cuda="4,5", eval_n="10", eval_script="/root/grpo/eval10x_alfworld.sh")
    module.annotate_diagnostic_commands(
        rows,
        work_root="/work",
        diagnostic_script="scripts/run_wm_checkpoint_diagnostics.sh",
        diagnostic_steps="init 30 60 90 120 150",
        transition_step="150",
    )
    module.annotate_artifact_coverage(rows)

    assert len(rows) == 1
    assert rows[0]["run_key"] == "obs_ce_l0p01_s0"
    assert rows[0]["expected"] == "yes"
    assert rows[0]["eval_readiness"] == "missing_training_log"
    assert rows[0]["has_train_log"] == "no"
    assert rows[0]["has_eval"] == "no"
    assert rows[0]["has_diagnostic"] == "no"
    assert rows[0]["diagnostic_readiness"] == "missing_training_log"
    assert rows[0]["missing_artifacts"] == "train_log,eval,diagnostic"
    assert rows[0]["final_report_readiness"] == "missing:train_log,eval,diagnostic"

    markdown = module.render_markdown(rows)
    assert "## Expected Run Coverage" in markdown
    assert "| obs_ce_l0p01_s0 | obs_ce | 0 | no | no | no | missing_training_log | missing_training_log | train_log,eval,diagnostic | missing:train_log,eval,diagnostic |" in markdown


def test_expected_run_merges_with_existing_artifacts(tmp_path):
    module = _load_module()
    log_path = tmp_path / "grpo_qwen2.5_1.5b_alfworld_seed1_wmlat_l0p001_s1_20260630_010203.log"
    log_path.write_text(
        "\n".join(
            [
                "RUN_ALFWORLD_OFFICIAL seed=1 tag=wmlat_l0p001_s1 cuda=4,5 ckpt=/work/checkpoints/grpo_qwen2.5_1.5b_alfworld_seed1_wmlat_l0p001_s1",
                "Training Progress: 150/150",
                "saved /work/checkpoints/grpo_qwen2.5_1.5b_alfworld_seed1_wmlat_l0p001_s1/global_step_150",
            ]
        ),
        encoding="utf-8",
    )
    eval_path = tmp_path / "eval10x_wmlat_l0p001_s1_results.txt"
    eval_path.write_text(
        "\n".join(
            [
                "EVAL10X_START label=wmlat_l0p001_s1 ckpt=/work/checkpoints/grpo_qwen2.5_1.5b_alfworld_seed1_wmlat_l0p001_s1/global_step_150 n=10 val_size=128 dataset=eval_in_distribution cuda=4,5 Tue",
                "EVAL10X_RESULT n=10 mean=0.7500 std=0.0200",
            ]
        ),
        encoding="utf-8",
    )

    rows = module.build_records(
        eval_paths=[str(eval_path)],
        train_logs=[str(log_path)],
        diagnostic_paths=[],
        expected_runs=["latent_l0p001_s1"],
    )
    module.annotate_eval_readiness(rows, eval_cuda="4,5", eval_n="10", eval_script="/root/grpo/eval10x_alfworld.sh")
    module.annotate_diagnostic_commands(
        rows,
        work_root=str(tmp_path),
        diagnostic_script="scripts/run_wm_checkpoint_diagnostics.sh",
        diagnostic_steps="init 30 60 90 120 150",
        transition_step="150",
    )
    module.annotate_artifact_coverage(rows)

    assert len(rows) == 1
    assert rows[0]["expected"] == "yes"
    assert rows[0]["run_key"] == "latent_l0p001_s1"
    assert rows[0]["eval_readiness"] == "evaluated"
    assert rows[0]["has_train_log"] == "yes"
    assert rows[0]["has_eval"] == "yes"
    assert rows[0]["has_diagnostic"] == "no"
    assert rows[0]["diagnostic_readiness"] == "missing_transition_dump"
    assert rows[0]["missing_artifacts"] == "diagnostic"
    assert rows[0]["final_report_readiness"] == "missing:diagnostic"


def test_expected_run_allows_tag_variants_without_blocking_conflict(tmp_path):
    module = _load_module()
    eval_path = tmp_path / "eval10x_obs_ce_l0p01_s0_results.txt"
    eval_path.write_text(
        "\n".join(
            [
                "EVAL10X_START label=obs_ce_l0p01_s0 ckpt=/work/checkpoints/grpo_qwen2.5_1.5b_alfworld_seed0_wrong_tag/global_step_150 n=10 val_size=128 dataset=eval_in_distribution cuda=4,5 Tue",
                "EVAL10X_RESULT n=10 mean=0.7400 std=0.0300",
            ]
        ),
        encoding="utf-8",
    )

    rows = module.build_records(
        eval_paths=[str(eval_path)],
        train_logs=[],
        diagnostic_paths=[],
        expected_runs=["obs_ce_l0p01_s0:objective=obs_ce,seed=0,lambda_obs=0.01,tag=wm_obs_ce_l0p01_s0"],
    )
    module.annotate_eval_readiness(rows, eval_cuda="4,5", eval_n="10", eval_script="/root/grpo/eval10x_alfworld.sh")
    module.annotate_artifact_coverage(rows)

    assert len(rows) == 1
    assert rows[0]["metadata_conflicts"] == ""
    assert rows[0]["has_eval"] == "yes"
    assert rows[0]["missing_artifacts"] == "train_log,diagnostic"
    assert rows[0]["final_report_readiness"] == "missing:train_log,diagnostic"

    markdown = module.render_markdown(rows)
    assert "- Metadata conflicts:" not in markdown


def test_expected_run_flags_blocking_metadata_conflicts():
    module = _load_module()
    records = {}
    module.merge_record(
        records,
        {
            "run_key": "obs_ce_l0p01_s0",
            "objective": "obs_ce",
            "seed": "0",
            "lambda_obs": "0.01",
            "expected": "yes",
        },
    )
    module.merge_record(
        records,
        {
            "run_key": "obs_ce_l0p01_s0",
            "objective": "obs_ce",
            "seed": "1",
            "lambda_obs": "0.03",
            "eval_mean": 0.74,
            "diagnostic_final_step": "150",
            "diagnostic_token_mean_ce": 1.4,
            "diagnostic_action_obs_cosine": 0.3,
            "diagnostic_report_md_path": "/work/diag/checkpoint_diagnostics_report.md",
            "diagnostic_report_csv_path": "/work/diag/checkpoint_diagnostics_report.csv",
            "diagnostic_report_svg_path": "/work/diag/checkpoint_diagnostics_report.svg",
            "train_log_path": "/work/logs/train.log",
        },
    )
    rows = list(records.values())
    module.annotate_artifact_coverage(rows)

    assert rows[0]["metadata_conflicts"] == "seed=0!=1;lambda_obs=0.01!=0.03"
    assert rows[0]["missing_artifacts"] == "metadata_conflict"
    assert rows[0]["final_report_readiness"] == "missing:metadata_conflict"

    markdown = module.render_markdown(rows)
    assert "- Metadata conflicts: `seed=0!=1;lambda_obs=0.01!=0.03`" in markdown


def test_goal_rd_expected_runs_cover_full_matrix():
    module = _load_module()

    rows = module.build_records(
        eval_paths=[],
        train_logs=[],
        diagnostic_paths=[],
        expected_runs=module.GOAL_RD_EXPECTED_RUNS,
    )
    module.annotate_eval_readiness(rows, eval_cuda="4,5", eval_n="10", eval_script="/root/grpo/eval10x_alfworld.sh")
    module.annotate_diagnostic_commands(
        rows,
        work_root="/work",
        diagnostic_script="scripts/run_wm_checkpoint_diagnostics.sh",
        diagnostic_steps="init 30 60 90 120 150",
        transition_step="150",
    )
    module.annotate_artifact_coverage(rows)

    run_keys = {row["run_key"] for row in rows}
    assert len(rows) == 11
    assert {
        "grpo_baseline_s0",
        "grpo_baseline_s1",
        "grpo_baseline_s2",
        "obs_ce_l0p01_s0",
        "obs_ce_l0p01_s1",
        "obs_ce_l0p03_s0",
        "obs_ce_l0p03_s1",
        "obs_ce_l0p05_s0",
        "obs_ce_l0p05_s1",
        "latent_l0p001_s0",
        "latent_l0p001_s1",
    } == run_keys

    obs_l0p05_s1 = next(row for row in rows if row["run_key"] == "obs_ce_l0p05_s1")
    assert obs_l0p05_s1["objective"] == "obs_ce"
    assert obs_l0p05_s1["seed"] == "1"
    assert obs_l0p05_s1["lambda_obs"] == "0.05"
    assert obs_l0p05_s1["tag"] == "wm_obs_ce_l0p05_s1"
    assert obs_l0p05_s1["diagnostic_readiness"] == "missing_training_log"
    assert obs_l0p05_s1["final_report_readiness"] == "missing:train_log,eval,diagnostic"

    coverage = {row["objective"]: row for row in module.objective_coverage(rows)}
    assert coverage["grpo_baseline"]["runs"] == 3
    assert coverage["obs_ce"]["runs"] == 6
    assert coverage["latent"]["runs"] == 2


def test_main_adds_goal_rd_expected_runs_without_duplicate_rows(tmp_path, monkeypatch):
    module = _load_module()
    eval_path = tmp_path / "eval10x_wm_obs_ce_l0p01_s0_results.txt"
    eval_path.write_text(
        "\n".join(
            [
                "EVAL10X_START label=wm_obs_ce_l0p01_s0 ckpt=/work/checkpoints/grpo_qwen2.5_1.5b_alfworld_seed0_wm_obs_ce_l0p01_s0/global_step_150 n=10 val_size=128 dataset=eval_in_distribution cuda=4,5 Tue",
                "EVAL10X_RESULT n=10 mean=0.7500 std=0.0200",
            ]
        ),
        encoding="utf-8",
    )
    log_path = tmp_path / "grpo_qwen2.5_1.5b_alfworld_seed0_wm_obs_ce_l0p01_s0_20260630_010203.log"
    log_path.write_text(
        "\n".join(
            [
                "RUN_ALFWORLD_OFFICIAL seed=0 tag=wm_obs_ce_l0p01_s0 cuda=4,5 ckpt=/work/checkpoints/grpo_qwen2.5_1.5b_alfworld_seed0_wm_obs_ce_l0p01_s0",
                "Training Progress: 150/150",
                "val/success_rate:0.734",
                "world_model/obs_ce_loss:0.144",
                "saved /work/checkpoints/grpo_qwen2.5_1.5b_alfworld_seed0_wm_obs_ce_l0p01_s0/global_step_150",
            ]
        ),
        encoding="utf-8",
    )
    output_md = tmp_path / "report.md"
    output_csv = tmp_path / "report.csv"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wm_results_table.py",
            "--eval-result",
            str(eval_path),
            "--train-log",
            str(log_path),
            "--expected-goal-rd-runs",
            "--report-revision",
            "report-test-rev",
            "--train-cuda",
            "6,7",
            "--output-md",
            str(output_md),
            "--output-csv",
            str(output_csv),
        ],
    )
    module.main()

    markdown = output_md.read_text(encoding="utf-8")
    assert "## Report Generation" in markdown
    assert "- Enabled flags: `--expected-goal-rd-runs`" in markdown
    assert "- Report revision: `report-test-rev`" in markdown
    assert "- Expected run inputs: `11`" in markdown
    assert "- Train CUDA: `6,7`" in markdown
    assert "## Expected Run Coverage" in markdown
    with output_csv.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 11
    by_key = {row["run_key"]: row for row in rows}
    assert len(by_key) == 11
    assert by_key["obs_ce_l0p01_s0"]["expected"] == "yes"
    assert by_key["obs_ce_l0p01_s0"]["eval_readiness"] == "evaluated"
    assert by_key["obs_ce_l0p01_s0"]["has_train_log"] == "yes"
    assert by_key["obs_ce_l0p01_s0"]["has_eval"] == "yes"
    assert by_key["obs_ce_l0p01_s0"]["final_report_readiness"] == "missing:diagnostic"
    assert by_key["obs_ce_l0p03_s1"]["final_report_readiness"] == "missing:train_log,eval,diagnostic"
    assert by_key["obs_ce_l0p03_s1"]["train_command"] == (
        "TAG=wm_obs_ce_l0p03_s1 LAMBDA_OBS=0.03 CUDA_VISIBLE_DEVICES=6,7 "
        "bash /root/grpo/run_seed_alfworld_official.sh 1"
    )


def test_goal_rd_report_preset_discovers_layout_and_expected_runs(tmp_path, monkeypatch):
    module = _load_module()
    work_root = tmp_path / "work"
    logs_dir = work_root / "logs"
    logs_dir.mkdir(parents=True)

    eval_path = logs_dir / "eval10x_wm_obs_ce_l0p01_s0_results.txt"
    eval_path.write_text(
        "\n".join(
            [
                "EVAL10X_START label=wm_obs_ce_l0p01_s0 ckpt=/work/checkpoints/grpo_qwen2.5_1.5b_alfworld_seed0_wm_obs_ce_l0p01_s0/global_step_150 n=10 val_size=128 dataset=eval_in_distribution cuda=4,5 Tue",
                "EVAL10X_RESULT n=10 mean=0.7500 std=0.0200",
            ]
        ),
        encoding="utf-8",
    )
    log_path = logs_dir / "grpo_qwen2.5_1.5b_alfworld_seed0_wm_obs_ce_l0p01_s0_20260630_010203.log"
    log_path.write_text(
        "\n".join(
            [
                "RUN_ALFWORLD_OFFICIAL seed=0 tag=wm_obs_ce_l0p01_s0 cuda=4,5 ckpt=/work/checkpoints/grpo_qwen2.5_1.5b_alfworld_seed0_wm_obs_ce_l0p01_s0",
                "Training Progress: 150/150",
                "saved /work/checkpoints/grpo_qwen2.5_1.5b_alfworld_seed0_wm_obs_ce_l0p01_s0/global_step_150",
            ]
        ),
        encoding="utf-8",
    )
    smoke_log = logs_dir / "grpo_qwen2.5_1.5b_alfworld_seed0_wm_obs_ce_smoke_20260630_010203.log"
    smoke_log.write_text(
        "\n".join(
            [
                "RUN_ALFWORLD_OFFICIAL seed=0 tag=wm_obs_ce_smoke cuda=4,5 ckpt=/work/checkpoints/grpo_qwen2.5_1.5b_alfworld_seed0_wm_obs_ce_smoke",
                "Training Progress: 3/150",
            ]
        ),
        encoding="utf-8",
    )
    eval_detail_log = logs_dir / "eval10x_wm_obs_ce_l0p01_s0_0_20260630_090243.log"
    eval_detail_log.write_text(
        "\n".join(
            [
                "trainer.experiment_name=eval10x_wm_obs_ce_l0p01_s0_0",
                "val/success_rate:0.750",
                "global_step_150",
            ]
        ),
        encoding="utf-8",
    )
    eval_launch_log = logs_dir / "eval10x_wm_obs_ce_l0p01_s0_launch_20260630_090241.log"
    eval_launch_log.write_text("EVAL_STARTED label=wm_obs_ce_l0p01_s0\n", encoding="utf-8")
    output_md = tmp_path / "report.md"
    output_csv = tmp_path / "report.csv"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wm_results_table.py",
            "--work-root",
            str(work_root),
            "--goal-rd-report",
            "--report-revision",
            "preset-test-rev",
            "--train-cuda",
            "6,7",
            "--output-md",
            str(output_md),
            "--output-csv",
            str(output_csv),
        ],
    )
    module.main()

    markdown = output_md.read_text(encoding="utf-8")
    assert "- Enabled flags: `--goal-rd-report --discover-standard-layout --expected-goal-rd-runs`" in markdown
    assert "- Report revision: `preset-test-rev`" in markdown
    assert "- Eval result inputs: `1`" in markdown
    assert "- Train log inputs: `1`" in markdown
    assert "- Expected run inputs: `11`" in markdown
    assert "- Train dump rollouts: `True`" in markdown
    assert "## Expected Run Coverage" in markdown
    with output_csv.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 11
    by_key = {row["run_key"]: row for row in rows}
    assert "latent_l0p01_s0" not in by_key
    assert by_key["obs_ce_l0p01_s0"]["has_train_log"] == "yes"
    assert by_key["obs_ce_l0p01_s0"]["has_eval"] == "yes"
    assert by_key["obs_ce_l0p03_s1"]["train_command"] == (
        "TAG=wm_obs_ce_l0p03_s1 WM_DUMP_ROLLOUTS=1 LAMBDA_OBS=0.03 CUDA_VISIBLE_DEVICES=6,7 "
        "bash /root/grpo/run_seed_alfworld_official.sh 1"
    )
    assert all("smoke" not in row["train_log_path"] for row in rows)


def test_parse_diagnostic_summary_uses_step150_and_success_gaps(tmp_path):
    module = _load_module()
    summary_dir = tmp_path / "wm_obs_ce_l0p01_s0"
    summary_dir.mkdir()
    summary_path = summary_dir / "checkpoint_scores_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "transition_jsonl": "/work/logs/rollouts/wm_obs_ce_l0p01_s0/150.wm_transitions.jsonl",
                "checkpoints": [
                    {
                        "checkpoint_label": "obs_init",
                        "checkpoint_step": "init",
                        "checkpoint_path": "base",
                        "token_mean_ce": 2.0,
                        "row_mean_action_obs_cosine": 0.10,
                    },
                    {
                        "checkpoint_label": "obs_step150",
                        "checkpoint_step": "150",
                        "checkpoint_path": "/ckpt/global_step_150",
                        "token_mean_ce": 1.5,
                        "row_mean_action_obs_cosine": 0.35,
                    },
                ],
                "provenance": {
                    "argv": [
                        "wm_score_transition_dump.py",
                        "--model-path",
                        "/model",
                    ],
                    "command": "python scripts/wm_score_transition_dump.py --model-path /model",
                    "cwd": "/work/verl-agent",
                    "model_path": "/model",
                    "output_csv": "/work/logs/world_model_diagnostics/wm_obs_ce_l0p01_s0/checkpoint_scores.csv",
                    "checkpoint_count": 2,
                    "max_length": 512,
                    "batch_size": 4,
                    "max_rows": 0,
                    "device": "cuda:0",
                    "dtype": "bfloat16",
                    "skip_entropy": True,
                    "chat_template_kwargs_json": '{"enable_thinking": false}',
                    "chat_template_kwargs": {"enable_thinking": False},
                },
                "success_buckets": [
                    {
                        "checkpoint_label": "obs_step150",
                        "checkpoint_step": "150",
                        "episode_success": True,
                        "token_mean_ce": 1.2,
                        "row_mean_action_obs_cosine": 0.50,
                    },
                    {
                        "checkpoint_label": "obs_step150",
                        "checkpoint_step": "150",
                        "episode_success": False,
                        "token_mean_ce": 1.9,
                        "row_mean_action_obs_cosine": 0.20,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    for report_name in (
        "checkpoint_diagnostics_report.md",
        "checkpoint_diagnostics_report.csv",
        "checkpoint_diagnostics_report.svg",
    ):
        (summary_dir / report_name).write_text("diagnostic report\n", encoding="utf-8")

    row = module.parse_diagnostic_summary(str(summary_path))

    assert row["run_key"] == "obs_ce_l0p01_s0"
    assert row["diagnostic_final_step"] == "150"
    assert row["diagnostic_token_mean_ce"] == 1.5
    assert row["diagnostic_delta_token_mean_ce"] == "-0.5"
    assert row["diagnostic_action_obs_cosine"] == 0.35
    assert row["diagnostic_delta_action_obs_cosine"] == "0.25"
    assert row["diagnostic_success_failure_ce_gap"] == "0.7"
    assert row["diagnostic_success_failure_cosine_gap"] == "0.3"
    assert row["diagnostic_command"].startswith("python scripts/wm_score_transition_dump.py")
    assert row["diagnostic_model_path"] == "/model"
    assert row["diagnostic_output_csv_path"].endswith("checkpoint_scores.csv")
    assert row["diagnostic_report_md_path"].endswith("checkpoint_diagnostics_report.md")
    assert row["diagnostic_report_csv_path"].endswith("checkpoint_diagnostics_report.csv")
    assert row["diagnostic_report_svg_path"].endswith("checkpoint_diagnostics_report.svg")
    assert row["diagnostic_transition_jsonl"] == "/work/logs/rollouts/wm_obs_ce_l0p01_s0/150.wm_transitions.jsonl"
    assert row["diagnostic_checkpoint_root"] == "/ckpt"
    assert row["diagnostic_checkpoint_count"] == 2
    assert row["diagnostic_max_length"] == 512
    assert row["diagnostic_batch_size"] == 4
    assert row["diagnostic_device"] == "cuda:0"
    assert row["diagnostic_dtype"] == "bfloat16"
    assert row["diagnostic_skip_entropy"] is True
    assert row["diagnostic_chat_template_kwargs_json"] == '{"enable_thinking": false}'
    assert row["diagnostic_chat_template_kwargs"] == '{"enable_thinking":false}'
    assert row["diagnostic_cwd"] == "/work/verl-agent"
    assert row["diagnostic_argv"] == '["wm_score_transition_dump.py","--model-path","/model"]'

    markdown = module.render_markdown([row])
    assert "- Diagnostic cwd: `/work/verl-agent`" in markdown
    assert "- Diagnostic chat template kwargs: `{\"enable_thinking\": false}`" in markdown
    assert '- Diagnostic argv: `["wm_score_transition_dump.py","--model-path","/model"]`' in markdown


def test_parse_diagnostic_summary_coerces_string_success_buckets(tmp_path):
    module = _load_module()
    summary_dir = tmp_path / "wm_obs_ce_l0p01_s0"
    summary_dir.mkdir()
    summary_path = summary_dir / "checkpoint_scores_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "checkpoints": [
                    {
                        "checkpoint_label": "init",
                        "checkpoint_step": "init",
                        "token_mean_ce": 2.0,
                        "row_mean_action_obs_cosine": 0.10,
                    },
                    {
                        "checkpoint_label": "step150",
                        "checkpoint_step": "150",
                        "token_mean_ce": 1.5,
                        "row_mean_action_obs_cosine": 0.35,
                    },
                ],
                "success_buckets": [
                    {
                        "checkpoint_label": "step150",
                        "checkpoint_step": "150",
                        "episode_success": "true",
                        "token_mean_ce": 1.2,
                        "row_mean_action_obs_cosine": 0.50,
                    },
                    {
                        "checkpoint_label": "step150",
                        "checkpoint_step": "150",
                        "episode_success": "false",
                        "token_mean_ce": 1.9,
                        "row_mean_action_obs_cosine": 0.20,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    row = module.parse_diagnostic_summary(str(summary_path))

    assert row["diagnostic_success_failure_ce_gap"] == "0.7"
    assert row["diagnostic_success_failure_cosine_gap"] == "0.3"


def test_parse_diagnostic_summary_backfills_success_buckets_from_score_csv(tmp_path):
    module = _load_module()
    summary_dir = tmp_path / "wm_obs_ce_l0p01_s0"
    summary_dir.mkdir()
    score_csv = summary_dir / "checkpoint_scores.csv"
    with score_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "checkpoint_label",
                "checkpoint_step",
                "target_tokens",
                "nll_sum",
                "action_obs_cosine",
                "episode_rewards",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "checkpoint_label": "step150",
                    "checkpoint_step": "150",
                    "target_tokens": "10",
                    "nll_sum": "11",
                    "action_obs_cosine": "0.5",
                    "episode_rewards": "1",
                },
                {
                    "checkpoint_label": "step150",
                    "checkpoint_step": "150",
                    "target_tokens": "10",
                    "nll_sum": "19",
                    "action_obs_cosine": "0.0",
                    "episode_rewards": "0",
                },
            ]
        )
    summary_path = summary_dir / "checkpoint_scores_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "checkpoints": [
                    {
                        "checkpoint_label": "init",
                        "checkpoint_step": "init",
                        "token_mean_ce": 2.0,
                        "row_mean_action_obs_cosine": 0.10,
                    },
                    {
                        "checkpoint_label": "step150",
                        "checkpoint_step": "150",
                        "token_mean_ce": 1.5,
                        "row_mean_action_obs_cosine": 0.35,
                    },
                ],
                "provenance": {
                    "output_csv": str(score_csv),
                },
            }
        ),
        encoding="utf-8",
    )

    row = module.parse_diagnostic_summary(str(summary_path))

    assert row["diagnostic_success_failure_ce_gap"] == "0.8"
    assert row["diagnostic_success_failure_cosine_gap"] == "0.5"


def test_parse_diagnostic_summary_does_not_invent_missing_report_paths(tmp_path):
    module = _load_module()
    summary_dir = tmp_path / "wm_obs_ce_l0p01_s0"
    summary_dir.mkdir()
    summary_path = summary_dir / "checkpoint_scores_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "transition_jsonl": "/work/logs/rollouts/wm_obs_ce_l0p01_s0/150.wm_transitions.jsonl",
                "checkpoints": [
                    {
                        "checkpoint_label": "init",
                        "checkpoint_step": "init",
                        "token_mean_ce": 2.0,
                        "row_mean_action_obs_cosine": 0.10,
                    },
                    {
                        "checkpoint_label": "step150",
                        "checkpoint_step": "150",
                        "token_mean_ce": 1.5,
                        "row_mean_action_obs_cosine": 0.35,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    row = module.parse_diagnostic_summary(str(summary_path))

    assert row["diagnostic_report_md_path"] == ""
    assert row["diagnostic_report_csv_path"] == ""
    assert row["diagnostic_report_svg_path"] == ""

    row["expected"] = "yes"
    row["train_log_path"] = "/work/logs/train.log"
    row["eval_mean"] = 0.72
    module.annotate_artifact_coverage([row])

    assert row["has_diagnostic"] == "yes"
    assert row["missing_artifacts"] == "diagnostic_report_md,diagnostic_report_csv,diagnostic_report_svg"
    assert row["final_report_readiness"] == "missing:diagnostic_report_md,diagnostic_report_csv,diagnostic_report_svg"


def test_parse_diagnostic_summary_ignores_empty_or_stale_report_paths(tmp_path):
    module = _load_module()
    summary_dir = tmp_path / "wm_obs_ce_l0p01_s0"
    summary_dir.mkdir()
    summary_path = summary_dir / "checkpoint_scores_summary.json"
    summary = {
        "transition_jsonl": "/work/logs/rollouts/wm_obs_ce_l0p01_s0/150.wm_transitions.jsonl",
        "checkpoints": [
            {
                "checkpoint_label": "init",
                "checkpoint_step": "init",
                "token_mean_ce": 2.0,
                "row_mean_action_obs_cosine": 0.10,
            },
            {
                "checkpoint_label": "step150",
                "checkpoint_step": "150",
                "token_mean_ce": 1.5,
                "row_mean_action_obs_cosine": 0.35,
            },
        ],
    }
    report_names = (
        "checkpoint_diagnostics_report.md",
        "checkpoint_diagnostics_report.csv",
        "checkpoint_diagnostics_report.svg",
    )
    for report_name in report_names:
        report_path = summary_dir / report_name
        report_path.write_text("stale report\n", encoding="utf-8")
        os.utime(report_path, (1000, 1000))
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    os.utime(summary_path, (2000, 2000))

    row = module.parse_diagnostic_summary(str(summary_path))

    assert row["diagnostic_report_md_path"] == ""
    assert row["diagnostic_report_csv_path"] == ""
    assert row["diagnostic_report_svg_path"] == ""

    for report_name in report_names:
        (summary_dir / report_name).write_text("", encoding="utf-8")
        os.utime(summary_dir / report_name, (3000, 3000))

    row = module.parse_diagnostic_summary(str(summary_path))

    assert row["diagnostic_report_md_path"] == ""
    assert row["diagnostic_report_csv_path"] == ""
    assert row["diagnostic_report_svg_path"] == ""


def test_incomplete_diagnostic_summary_does_not_count_as_complete(tmp_path):
    module = _load_module()
    work_root = tmp_path / "work"
    checkpoint_root = work_root / "checkpoints" / "grpo_qwen2.5_1.5b_alfworld_seed0_wm_obs_ce_l0p01_s0"
    rollout_dir = work_root / "logs" / "world_model_rollouts" / checkpoint_root.name
    rollout_dir.mkdir(parents=True)
    transition_jsonl = rollout_dir / "150.wm_transitions.jsonl"
    transition_jsonl.write_text('{"ok": true}\n', encoding="utf-8")

    summary_dir = tmp_path / "wm_obs_ce_l0p01_s0"
    summary_dir.mkdir()
    summary_path = summary_dir / "checkpoint_scores_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "transition_jsonl": str(transition_jsonl),
                "checkpoints": [
                    {
                        "checkpoint_label": "init",
                        "checkpoint_step": "init",
                        "token_mean_ce": 2.0,
                        "row_mean_action_obs_cosine": 0.10,
                        "rows_with_targets": 2,
                        "target_tokens": 20,
                    },
                    {
                        "checkpoint_label": "step60",
                        "checkpoint_step": "60",
                        "checkpoint_path": str(checkpoint_root / "global_step_60"),
                        "token_mean_ce": 1.7,
                        "row_mean_action_obs_cosine": 0.20,
                        "rows_with_targets": 2,
                        "target_tokens": 20,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    row = module.parse_diagnostic_summary(str(summary_path))
    row.update(
        {
            "expected": "yes",
            "train_log_path": "/work/logs/train.log",
            "eval_mean": 0.72,
            "tag": "wm_obs_ce_l0p01_s0",
            "latest_checkpoint_step": 150,
            "latest_checkpoint_path": str(checkpoint_root / "global_step_150"),
        }
    )

    assert row["diagnostic_final_step"] == "60"
    assert row["status"] == "diagnostic_incomplete"
    assert not module.row_has_diagnostic(row)

    module.annotate_diagnostic_commands(
        [row],
        work_root=str(work_root),
        diagnostic_script="scripts/run_wm_checkpoint_diagnostics.sh",
        diagnostic_steps="init 60 120 150",
        transition_step="150",
    )
    module.annotate_artifact_coverage([row])

    assert row["diagnostic_readiness"] == "ready_for_diagnostic"
    assert "run_wm_checkpoint_diagnostics.sh" in row["diagnostic_command"]
    assert row["has_diagnostic"] == "no"
    assert row["missing_artifacts"] == "diagnostic"
    assert row["final_report_readiness"] == "missing:diagnostic"


def test_zero_token_or_missing_cosine_diagnostic_is_incomplete():
    module = _load_module()
    base = {
        "diagnostic_summary_path": "/work/diag/checkpoint_scores_summary.json",
        "diagnostic_final_step": "150",
        "diagnostic_token_mean_ce": 1.4,
    }

    assert not module.row_has_diagnostic(base | {"diagnostic_action_obs_cosine": 0.3, "diagnostic_target_tokens": 0})
    assert not module.row_has_diagnostic(base | {"diagnostic_target_tokens": 20})
    assert module.row_has_diagnostic(base | {"diagnostic_action_obs_cosine": 0.0, "diagnostic_target_tokens": 20})


def test_expand_paths_expands_explicit_glob_paths(tmp_path):
    module = _load_module()
    first = tmp_path / "eval10x_seed0_results.txt"
    second = tmp_path / "eval10x_seed1_results.txt"
    first.write_text("seed0", encoding="utf-8")
    second.write_text("seed1", encoding="utf-8")

    paths = module.expand_paths([str(tmp_path / "eval10x_*_results.txt")], [])

    assert paths == [str(first), str(second)]


def test_main_writes_markdown_and_csv(tmp_path, monkeypatch):
    module = _load_module()
    eval_path = tmp_path / "eval10x_seed0_results.txt"
    eval_path.write_text(
        "\n".join(
            [
                "EVAL10X_START label=seed0 ckpt=/work/checkpoints/grpo_qwen2.5_1.5b_alfworld_seed0_official_4to5/global_step_150 n=10 val_size=128 dataset=eval_in_distribution cuda=4,5 Tue",
                "EVAL10X_RESULT n=10 mean=0.7290 std=0.0280",
            ]
        ),
        encoding="utf-8",
    )
    log_path = tmp_path / "grpo_qwen2.5_1.5b_alfworld_seed0_official_4to5_20260630_010203.log"
    log_path.write_text(
        "\n".join(
            [
                "RUN_ALFWORLD_OFFICIAL seed=0 tag=official_4to5 cuda=4,5 tp=2 micro=16 gmu=0.6 ckpt=/work/checkpoints/grpo_qwen2.5_1.5b_alfworld_seed0_official_4to5",
                "Training Progress: 150/150",
                "val/success_rate:0.734",
                "saved /work/checkpoints/grpo_qwen2.5_1.5b_alfworld_seed0_official_4to5/global_step_150",
            ]
        ),
        encoding="utf-8",
    )
    output_md = tmp_path / "report.md"
    output_csv = tmp_path / "report.csv"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wm_results_table.py",
            "--eval-result",
            str(eval_path),
            "--train-log",
            str(log_path),
            "--branch",
            "world-model-latent-objective",
            "--output-md",
            str(output_md),
            "--output-csv",
            str(output_csv),
        ],
    )
    module.main()

    markdown = output_md.read_text(encoding="utf-8")
    assert "ALFWorld World-Model Results" in markdown
    assert "| grpo_baseline_s0 | official_4to5 | grpo_baseline | 0 |" in markdown
    assert "0.7290 +/- 0.0280 (n=10)" in markdown
    with output_csv.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["run_key"] == "grpo_baseline_s0"
    assert rows[0]["eval_mean"] == "0.729"
    assert rows[0]["eval_readiness"] == "evaluated"
    assert rows[0]["eval_target_checkpoint_path"].endswith("global_step_150")
    assert rows[0]["train_step"] == "150"


def test_atomic_write_preserves_existing_results_on_replace_failure(tmp_path, monkeypatch):
    module = _load_module()
    output_path = tmp_path / "world_model_results.md"
    output_path.write_text("old report\n", encoding="utf-8")

    def fail_replace(src, dst):
        raise RuntimeError("replace failed")

    monkeypatch.setattr(module.os, "replace", fail_replace)

    with pytest.raises(RuntimeError, match="replace failed"):
        module.write_text(str(output_path), "new report\n")

    assert output_path.read_text(encoding="utf-8") == "old report\n"
    assert list(tmp_path.glob(".world_model_results.md.*.tmp")) == []


def test_main_uses_bootstrap_eval_script_default_for_ready_runs(tmp_path, monkeypatch):
    module = _load_module()
    log_path = tmp_path / "grpo_qwen2.5_1.5b_alfworld_seed1_wmlat_l0p001_s1_20260630_010203.log"
    log_path.write_text(
        "\n".join(
            [
                "RUN_ALFWORLD_OFFICIAL seed=1 tag=wmlat_l0p001_s1 cuda=4,5 ckpt=/work/checkpoints/grpo_qwen2.5_1.5b_alfworld_seed1_wmlat_l0p001_s1",
                "Training Progress: 150/150",
                "val/success_rate:0.734",
                "saved /work/checkpoints/grpo_qwen2.5_1.5b_alfworld_seed1_wmlat_l0p001_s1/global_step_150",
            ]
        ),
        encoding="utf-8",
    )
    output_md = tmp_path / "report.md"
    output_csv = tmp_path / "report.csv"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wm_results_table.py",
            "--run-log",
            str(log_path),
            "--eval-cuda",
            "6,7",
            "--output-md",
            str(output_md),
            "--output-csv",
            str(output_csv),
        ],
    )
    module.main()

    markdown = output_md.read_text(encoding="utf-8")
    assert "ready_for_eval" in markdown
    assert "bash /root/grpo/eval10x_alfworld.sh" in markdown
    with output_csv.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["eval_readiness"] == "ready_for_eval"
    assert rows[0]["eval_command"].endswith("bash /root/grpo/eval10x_alfworld.sh")
    assert rows[0]["eval_target_checkpoint_path"].endswith("global_step_150")


def test_main_discovers_standard_layout(tmp_path, monkeypatch):
    module = _load_module()
    work_root = tmp_path / "work"
    logs_dir = work_root / "logs"
    diagnostics_dir = logs_dir / "world_model_diagnostics" / "wm_obs_ce_l0p01_s0"
    diagnostics_dir.mkdir(parents=True)

    eval_path = logs_dir / "eval10x_wm_obs_ce_l0p01_s0_results.txt"
    eval_path.write_text(
        "\n".join(
            [
                "EVAL10X_START label=wm_obs_ce_l0p01_s0 ckpt=/work/checkpoints/grpo_qwen2.5_1.5b_alfworld_seed0_wm_obs_ce_l0p01_s0/global_step_150 n=10 val_size=128 dataset=eval_in_distribution cuda=4,5 Tue",
                "EVAL10X_RESULT n=10 mean=0.7500 std=0.0200",
            ]
        ),
        encoding="utf-8",
    )
    log_path = logs_dir / "grpo_qwen2.5_1.5b_alfworld_seed0_wm_obs_ce_l0p01_s0_20260630_010203.log"
    log_path.write_text(
        "\n".join(
            [
                "RUN_ALFWORLD_OFFICIAL seed=0 tag=wm_obs_ce_l0p01_s0 cuda=4,5 ckpt=/work/checkpoints/grpo_qwen2.5_1.5b_alfworld_seed0_wm_obs_ce_l0p01_s0",
                "Training Progress: 150/150",
                "val/success_rate:0.734",
                "world_model/obs_ce_loss:0.144",
                "saved /work/checkpoints/grpo_qwen2.5_1.5b_alfworld_seed0_wm_obs_ce_l0p01_s0/global_step_150",
            ]
        ),
        encoding="utf-8",
    )
    summary_path = diagnostics_dir / "checkpoint_scores_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "transition_jsonl": "/work/logs/world_model_rollouts/wm_obs_ce_l0p01_s0/150.wm_transitions.jsonl",
                "rows": 8,
                "checkpoints": [
                    {
                        "checkpoint_label": "init",
                        "checkpoint_step": "init",
                        "token_mean_ce": 1.8,
                        "row_mean_action_obs_cosine": 0.1,
                    },
                    {
                        "checkpoint_label": "step150",
                        "checkpoint_step": "150",
                        "token_mean_ce": 1.4,
                        "row_mean_action_obs_cosine": 0.3,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    for report_name in (
        "checkpoint_diagnostics_report.md",
        "checkpoint_diagnostics_report.csv",
        "checkpoint_diagnostics_report.svg",
    ):
        (diagnostics_dir / report_name).write_text("diagnostic report\n", encoding="utf-8")
    smoke_log = logs_dir / "grpo_qwen2.5_1.5b_alfworld_seed0_wm_obs_ce_smoke_20260630_010203.log"
    smoke_log.write_text(
        "\n".join(
            [
                "RUN_ALFWORLD_OFFICIAL seed=0 tag=wm_obs_ce_smoke cuda=4,5 ckpt=/work/checkpoints/grpo_qwen2.5_1.5b_alfworld_seed0_wm_obs_ce_smoke",
                "Training Progress: 3/150",
                "val/success_rate:0.100",
            ]
        ),
        encoding="utf-8",
    )
    smoke_diagnostics_dir = logs_dir / "world_model_diagnostics" / "wm_ckpt_diag_seed0_smoke_20260630"
    smoke_diagnostics_dir.mkdir()
    (smoke_diagnostics_dir / "checkpoint_scores_summary.json").write_text(
        json.dumps(
            {
                "transition_jsonl": "/work/logs/world_model_rollouts/smoke/3.wm_transitions.jsonl",
                "checkpoints": [
                    {
                        "checkpoint_label": "smoke",
                        "checkpoint_step": "3",
                        "token_mean_ce": 9.9,
                        "row_mean_action_obs_cosine": 0.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output_md = tmp_path / "report.md"
    output_csv = tmp_path / "report.csv"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wm_results_table.py",
            "--work-root",
            str(work_root),
            "--discover-standard-layout",
            "--branch",
            "world-model-latent-objective",
            "--output-md",
            str(output_md),
            "--output-csv",
            str(output_csv),
        ],
    )
    module.main()

    markdown = output_md.read_text(encoding="utf-8")
    assert f"- Work root: `{work_root}`" in markdown
    assert "0.7500 +/- 0.0200 (n=10)" in markdown
    assert "- Diagnostic report: `" in markdown
    assert "checkpoint_diagnostics_report.md" in markdown
    assert "- Diagnostic report CSV: `" in markdown
    assert "checkpoint_diagnostics_report.csv" in markdown
    assert "- Diagnostic report SVG: `" in markdown
    assert "checkpoint_diagnostics_report.svg" in markdown
    with output_csv.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert b"\r\n" not in output_csv.read_bytes()
    row = rows[0]
    assert row["run_key"] == "obs_ce_l0p01_s0"
    assert row["eval_mean"] == "0.75"
    assert row["eval_readiness"] == "evaluated"
    assert row["eval_target_checkpoint_path"].endswith("global_step_150")
    assert row["train_step"] == "150"
    assert row["diagnostic_token_mean_ce"] == "1.4"
    assert row["diagnostic_delta_token_mean_ce"] == "-0.4"
    assert row["diagnostic_report_md_path"].endswith("checkpoint_diagnostics_report.md")
    assert row["diagnostic_report_csv_path"].endswith("checkpoint_diagnostics_report.csv")
    assert row["diagnostic_report_svg_path"].endswith("checkpoint_diagnostics_report.svg")
    assert "smoke" not in row["train_log_path"]
    assert "smoke" not in row["diagnostic_summary_path"]
