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
    assert rows[0]["train_step"] == "150"
