import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _load_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "wm_diagnostics_report.py"
    spec = importlib.util.spec_from_file_location("wm_diagnostics_report_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _summary(path="/tmp/wm_diag/checkpoint_scores_summary.json"):
    return {
        "_summary_path": path,
        "transition_jsonl": "/tmp/transitions.jsonl",
        "rows": 12,
        "max_length": 512,
        "provenance": {
            "command": "python scripts/wm_score_transition_dump.py --model-path /model",
            "model_path": "/model",
            "output_csv": "/tmp/wm_diag/checkpoint_scores.csv",
            "checkpoint_count": 2,
            "max_length": 512,
            "batch_size": 4,
            "max_rows": 0,
            "device": "cuda:0",
            "dtype": "bfloat16",
            "skip_entropy": False,
            "checkpoints": [
                {"label": "run_init", "path": "base", "step": "init"},
                {"label": "run_step60", "path": "/ckpt/global_step_60", "step": "60"},
            ],
        },
        "checkpoints": [
            {
                "checkpoint_label": "run_step60",
                "checkpoint_step": "60",
                "checkpoint_path": "/ckpt/global_step_60",
                "rows": 2,
                "rows_with_targets": 2,
                "target_tokens": 20,
                "token_mean_ce": 1.5,
                "row_mean_target_confidence": 0.25,
                "row_mean_target_entropy": 1.8,
                "row_mean_action_obs_cosine": 0.4,
            },
            {
                "checkpoint_label": "run_init",
                "checkpoint_step": "init",
                "checkpoint_path": "base",
                "rows": 2,
                "rows_with_targets": 2,
                "target_tokens": 20,
                "token_mean_ce": 2.0,
                "row_mean_target_confidence": 0.2,
                "row_mean_target_entropy": 2.1,
                "row_mean_action_obs_cosine": 0.1,
            },
        ],
        "success_buckets": [
            {
                "checkpoint_label": "run_init",
                "checkpoint_step": "init",
                "episode_success": True,
                "token_mean_ce": 1.8,
                "row_mean_action_obs_cosine": 0.2,
            },
            {
                "checkpoint_label": "run_init",
                "checkpoint_step": "init",
                "episode_success": False,
                "token_mean_ce": 2.4,
                "row_mean_action_obs_cosine": -0.1,
            },
            {
                "checkpoint_label": "run_step60",
                "checkpoint_step": "60",
                "episode_success": True,
                "token_mean_ce": 1.1,
                "row_mean_action_obs_cosine": 0.5,
            },
            {
                "checkpoint_label": "run_step60",
                "checkpoint_step": "60",
                "episode_success": False,
                "token_mean_ce": 1.9,
                "row_mean_action_obs_cosine": 0.0,
            },
        ],
    }


def test_build_rows_sorts_checkpoints_and_computes_deltas():
    module = _load_module()

    rows = module.build_rows(_summary())

    assert [row["checkpoint_label"] for row in rows] == ["run_init", "run_step60"]
    assert rows[0]["delta_token_mean_ce"] == 0.0
    assert rows[1]["delta_token_mean_ce"] == -0.5
    assert rows[1]["delta_row_mean_action_obs_cosine"] == 0.30000000000000004
    assert rows[1]["success_failure_ce_gap"] == 0.7999999999999998
    assert rows[1]["success_failure_cosine_gap"] == 0.5
    assert rows[1]["diagnostic_command"].startswith("python scripts/wm_score_transition_dump.py")
    assert rows[1]["diagnostic_model_path"] == "/model"
    assert rows[1]["diagnostic_checkpoint_count"] == 2


def test_load_summary_backfills_success_buckets_from_score_csv(tmp_path):
    module = _load_module()
    summary_path = tmp_path / "checkpoint_scores_summary.json"
    score_csv = tmp_path / "checkpoint_scores.csv"
    with score_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "checkpoint_label",
                "checkpoint_step",
                "target_tokens",
                "nll_sum",
                "target_confidence_mean",
                "target_entropy_mean",
                "action_obs_cosine",
                "episode_rewards",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "checkpoint_label": "run_init",
                    "checkpoint_step": "init",
                    "target_tokens": "10",
                    "nll_sum": "18",
                    "target_confidence_mean": "0.3",
                    "target_entropy_mean": "1.7",
                    "action_obs_cosine": "0.2",
                    "episode_rewards": "1",
                },
                {
                    "checkpoint_label": "run_init",
                    "checkpoint_step": "init",
                    "target_tokens": "10",
                    "nll_sum": "24",
                    "target_confidence_mean": "0.1",
                    "target_entropy_mean": "2.2",
                    "action_obs_cosine": "-0.1",
                    "episode_rewards": "0",
                },
                {
                    "checkpoint_label": "run_step60",
                    "checkpoint_step": "60",
                    "target_tokens": "10",
                    "nll_sum": "11",
                    "target_confidence_mean": "0.4",
                    "target_entropy_mean": "1.4",
                    "action_obs_cosine": "0.5",
                    "episode_rewards": "1",
                },
                {
                    "checkpoint_label": "run_step60",
                    "checkpoint_step": "60",
                    "target_tokens": "10",
                    "nll_sum": "19",
                    "target_confidence_mean": "0.2",
                    "target_entropy_mean": "2.0",
                    "action_obs_cosine": "0.0",
                    "episode_rewards": "0",
                },
            ]
        )
    summary = _summary(path=str(summary_path))
    summary.pop("_summary_path")
    summary.pop("success_buckets")
    summary["provenance"]["output_csv"] = str(score_csv)
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    loaded = module.load_summary(str(summary_path))
    rows = module.build_rows(loaded)
    markdown = module.render_markdown([loaded], [rows])

    assert loaded["_success_buckets_source"] == str(score_csv)
    assert len(loaded["success_buckets"]) == 4
    assert rows[1]["success_token_mean_ce"] == 1.1
    assert rows[1]["failure_token_mean_ce"] == 1.9
    assert rows[1]["success_failure_ce_gap"] == 0.7999999999999998
    assert rows[1]["success_failure_cosine_gap"] == 0.5
    assert f"- Success/failure buckets: `backfilled from {score_csv}`" in markdown


def test_render_markdown_includes_paths_and_comparison_table():
    module = _load_module()
    summary = _summary()
    rows = module.build_rows(summary)

    markdown = module.render_markdown([summary], [rows])

    assert "World-Model Checkpoint Diagnostics" in markdown
    assert "`/tmp/transitions.jsonl`" in markdown
    assert "- Diagnostic command: `python scripts/wm_score_transition_dump.py --model-path /model`" in markdown
    assert "- Model path: `/model`" in markdown
    assert "- Scored checkpoints: `2` (run_init, run_step60)" in markdown
    assert "| run_init | init | 20 | 2.0000 | 0.0000 |" in markdown
    assert "| run_step60 | 60 | 20 | 1.5000 | -0.5000 |" in markdown


def test_render_svg_plots_ce_and_cosine_panels():
    module = _load_module()
    summary = _summary()
    rows = module.build_rows(summary)

    svg = module.render_svg([summary], [rows])

    assert svg.startswith("<svg")
    assert "token_mean_ce" in svg
    assert "action_obs_cosine" in svg
    assert "run_init" not in svg
    assert "<polyline" in svg


def test_main_writes_markdown_csv_and_svg(tmp_path, monkeypatch):
    module = _load_module()
    summary_path = tmp_path / "checkpoint_scores_summary.json"
    summary = _summary(path=str(summary_path))
    summary.pop("_summary_path")
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    output_md = tmp_path / "report.md"
    output_csv = tmp_path / "report.csv"
    output_svg = tmp_path / "report.svg"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wm_diagnostics_report.py",
            "--summary-json",
            str(summary_path),
            "--output-md",
            str(output_md),
            "--output-csv",
            str(output_csv),
            "--output-svg",
            str(output_svg),
        ],
    )
    module.main()

    assert "checkpoint_scores_summary.json" in output_md.read_text(encoding="utf-8")
    assert output_svg.read_text(encoding="utf-8").startswith("<svg")
    with output_csv.open(encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert csv_rows[0]["checkpoint_label"] == "run_init"
    assert csv_rows[0]["diagnostic_model_path"] == "/model"
    assert csv_rows[0]["diagnostic_dtype"] == "bfloat16"
    assert csv_rows[1]["delta_token_mean_ce"] == "-0.5"


def test_atomic_write_preserves_existing_file_on_replace_failure(tmp_path, monkeypatch):
    module = _load_module()
    output_path = tmp_path / "report.md"
    output_path.write_text("old report\n", encoding="utf-8")

    def fail_replace(src, dst):
        raise RuntimeError("replace failed")

    monkeypatch.setattr(module.os, "replace", fail_replace)

    with pytest.raises(RuntimeError, match="replace failed"):
        module.write_text(str(output_path), "new report\n")

    assert output_path.read_text(encoding="utf-8") == "old report\n"
    assert list(tmp_path.glob(".report.md.*.tmp")) == []
