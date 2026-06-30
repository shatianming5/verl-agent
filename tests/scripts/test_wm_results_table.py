import csv
import importlib.util
import json
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


def test_objective_coverage_summarizes_eval_and_diagnostics():
    module = _load_module()
    rows = [
        {
            "run_key": "grpo_baseline_s0",
            "objective": "grpo_baseline",
            "seed": "0",
            "eval_readiness": "evaluated",
            "diagnostic_summary_path": "/work/diag/base.json",
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
            "diagnostic_token_mean_ce": 1.4,
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


def test_expected_run_coverage_reports_missing_artifacts():
    module = _load_module()

    rows = module.build_records(
        eval_paths=[],
        train_logs=[],
        diagnostic_paths=[],
        expected_runs=["obs_ce_l0p01_s0:objective=obs_ce,seed=0,lambda_obs=0.01,tag=wm_obs_ce_l0p01_s0"],
    )
    module.annotate_eval_readiness(rows, eval_cuda="4,5", eval_n="10", eval_script="/root/grpo/eval10x_alfworld.sh")
    module.annotate_artifact_coverage(rows)

    assert len(rows) == 1
    assert rows[0]["run_key"] == "obs_ce_l0p01_s0"
    assert rows[0]["expected"] == "yes"
    assert rows[0]["eval_readiness"] == "missing_training_log"
    assert rows[0]["has_train_log"] == "no"
    assert rows[0]["has_eval"] == "no"
    assert rows[0]["has_diagnostic"] == "no"
    assert rows[0]["missing_artifacts"] == "train_log,eval,diagnostic"
    assert rows[0]["final_report_readiness"] == "missing:train_log,eval,diagnostic"

    markdown = module.render_markdown(rows)
    assert "## Expected Run Coverage" in markdown
    assert "| obs_ce_l0p01_s0 | obs_ce | 0 | no | no | no | missing_training_log | train_log,eval,diagnostic | missing:train_log,eval,diagnostic |" in markdown


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
    module.annotate_artifact_coverage(rows)

    assert len(rows) == 1
    assert rows[0]["expected"] == "yes"
    assert rows[0]["run_key"] == "latent_l0p001_s1"
    assert rows[0]["eval_readiness"] == "evaluated"
    assert rows[0]["has_train_log"] == "yes"
    assert rows[0]["has_eval"] == "yes"
    assert rows[0]["has_diagnostic"] == "no"
    assert rows[0]["missing_artifacts"] == "diagnostic"
    assert rows[0]["final_report_readiness"] == "missing:diagnostic"


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
                    "command": "python scripts/wm_score_transition_dump.py --model-path /model",
                    "model_path": "/model",
                    "output_csv": "/work/logs/world_model_diagnostics/wm_obs_ce_l0p01_s0/checkpoint_scores.csv",
                    "checkpoint_count": 2,
                    "max_length": 512,
                    "batch_size": 4,
                    "max_rows": 0,
                    "device": "cuda:0",
                    "dtype": "bfloat16",
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
    assert row["diagnostic_checkpoint_count"] == 2
    assert row["diagnostic_max_length"] == 512
    assert row["diagnostic_batch_size"] == 4
    assert row["diagnostic_device"] == "cuda:0"
    assert row["diagnostic_dtype"] == "bfloat16"


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
    with output_csv.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    row = rows[0]
    assert row["run_key"] == "obs_ce_l0p01_s0"
    assert row["eval_mean"] == "0.75"
    assert row["eval_readiness"] == "evaluated"
    assert row["eval_target_checkpoint_path"].endswith("global_step_150")
    assert row["train_step"] == "150"
    assert row["diagnostic_token_mean_ce"] == "1.4"
    assert row["diagnostic_delta_token_mean_ce"] == "-0.4"
    assert "smoke" not in row["train_log_path"]
    assert "smoke" not in row["diagnostic_summary_path"]
