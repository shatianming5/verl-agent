import csv
import importlib.util
import json
import sys
from pathlib import Path


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


def test_render_markdown_includes_paths_and_comparison_table():
    module = _load_module()
    summary = _summary()
    rows = module.build_rows(summary)

    markdown = module.render_markdown([summary], [rows])

    assert "World-Model Checkpoint Diagnostics" in markdown
    assert "`/tmp/transitions.jsonl`" in markdown
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
    assert csv_rows[1]["delta_token_mean_ce"] == "-0.5"
