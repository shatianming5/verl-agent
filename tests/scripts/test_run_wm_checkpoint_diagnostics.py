import json
import os
import subprocess
import sys
from pathlib import Path


def test_run_wm_checkpoint_diagnostics_generates_report_artifacts(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "run_wm_checkpoint_diagnostics.sh"
    transitions = tmp_path / "transitions.wm_transitions.jsonl"
    transitions.write_text("{}\n", encoding="utf-8")
    ckpt_root = tmp_path / "exp"
    for step in (30, 150):
        (ckpt_root / f"global_step_{step}" / "actor").mkdir(parents=True)

    scorer_args = tmp_path / "scorer_args.json"
    reporter_args = tmp_path / "reporter_args.json"
    fake_scorer = tmp_path / "fake_scorer.py"
    fake_scorer.write_text(
        "\n".join(
            [
                "import csv, json, pathlib, sys",
                f"pathlib.Path({str(scorer_args)!r}).write_text(json.dumps(sys.argv[1:]), encoding='utf-8')",
                "args = sys.argv[1:]",
                "out_csv = pathlib.Path(args[args.index('--output-csv') + 1])",
                "summary_json = pathlib.Path(args[args.index('--summary-json') + 1])",
                "out_csv.parent.mkdir(parents=True, exist_ok=True)",
                "with out_csv.open('w', newline='', encoding='utf-8') as handle:",
                "    writer = csv.writer(handle)",
                "    writer.writerow(['checkpoint_label', 'token_mean_ce'])",
                "    writer.writerow(['exp_init', '2.0'])",
                "summary_json.write_text(json.dumps({'checkpoints': [{'checkpoint_label': 'exp_init', 'checkpoint_step': 'init'}]}), encoding='utf-8')",
            ]
        ),
        encoding="utf-8",
    )
    fake_reporter = tmp_path / "fake_reporter.py"
    fake_reporter.write_text(
        "\n".join(
            [
                "import json, pathlib, sys",
                f"pathlib.Path({str(reporter_args)!r}).write_text(json.dumps(sys.argv[1:]), encoding='utf-8')",
                "args = sys.argv[1:]",
                "for flag, text in (('--output-md', '# report\\n'), ('--output-csv', 'checkpoint_label\\n'), ('--output-svg', '<svg/>\\n')):",
                "    path = pathlib.Path(args[args.index(flag) + 1])",
                "    path.parent.mkdir(parents=True, exist_ok=True)",
                "    path.write_text(text, encoding='utf-8')",
            ]
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"
    env = {
        **os.environ,
        "TRANSITIONS_JSONL": str(transitions),
        "CKPT_ROOT": str(ckpt_root),
        "COMMON_SH": str(tmp_path / "missing_common.sh"),
        "MODEL": "/model",
        "PYTHON": sys.executable,
        "SCORER": str(fake_scorer),
        "REPORTER": str(fake_reporter),
        "WORK": str(tmp_path / "work"),
        "OUT_DIR": str(out_dir),
        "STEPS": "init 30 90 150",
        "DEVICE": "cpu",
        "DTYPE": "float32",
    }

    result = subprocess.run(["bash", str(script)], env=env, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    assert "WM_CHECKPOINT_DIAGNOSTICS_SKIP" in result.stdout
    assert "global_step_90" in result.stdout
    assert "WM_CHECKPOINT_DIAGNOSTICS_DONE" in result.stdout
    assert f"report_md={out_dir / 'checkpoint_diagnostics_report.md'}" in result.stdout
    assert f"report_csv={out_dir / 'checkpoint_diagnostics_report.csv'}" in result.stdout
    assert f"report_svg={out_dir / 'checkpoint_diagnostics_report.svg'}" in result.stdout
    for name in (
        "checkpoint_scores.csv",
        "checkpoint_scores_summary.json",
        "checkpoint_diagnostics_report.md",
        "checkpoint_diagnostics_report.csv",
        "checkpoint_diagnostics_report.svg",
    ):
        assert (out_dir / name).stat().st_size > 0

    scorer_argv = json.loads(scorer_args.read_text(encoding="utf-8"))
    assert scorer_argv[scorer_argv.index("--checkpoint") + 1] == "exp_init=base"
    assert f"exp_step30={ckpt_root / 'global_step_30'}" in scorer_argv
    assert f"exp_step150={ckpt_root / 'global_step_150'}" in scorer_argv
    assert all("global_step_90" not in item for item in scorer_argv)

    reporter_argv = json.loads(reporter_args.read_text(encoding="utf-8"))
    assert reporter_argv[reporter_argv.index("--summary-json") + 1] == str(out_dir / "checkpoint_scores_summary.json")
    assert reporter_argv[reporter_argv.index("--output-md") + 1] == str(out_dir / "checkpoint_diagnostics_report.md")


def test_run_wm_checkpoint_diagnostics_can_skip_report_generation(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "run_wm_checkpoint_diagnostics.sh"
    transitions = tmp_path / "transitions.wm_transitions.jsonl"
    transitions.write_text("{}\n", encoding="utf-8")
    ckpt_root = tmp_path / "exp"
    ckpt_root.mkdir()

    fake_scorer = tmp_path / "fake_scorer.py"
    fake_scorer.write_text(
        "\n".join(
            [
                "import csv, json, pathlib, sys",
                "args = sys.argv[1:]",
                "out_csv = pathlib.Path(args[args.index('--output-csv') + 1])",
                "summary_json = pathlib.Path(args[args.index('--summary-json') + 1])",
                "out_csv.parent.mkdir(parents=True, exist_ok=True)",
                "with out_csv.open('w', newline='', encoding='utf-8') as handle:",
                "    writer = csv.writer(handle)",
                "    writer.writerow(['checkpoint_label'])",
                "    writer.writerow(['exp_init'])",
                "summary_json.write_text(json.dumps({'checkpoints': [{'checkpoint_label': 'exp_init', 'checkpoint_step': 'init'}]}), encoding='utf-8')",
            ]
        ),
        encoding="utf-8",
    )
    fake_reporter = tmp_path / "fake_reporter.py"
    fake_reporter.write_text("raise SystemExit('reporter should not run')\n", encoding="utf-8")
    out_dir = tmp_path / "out"
    env = {
        **os.environ,
        "TRANSITIONS_JSONL": str(transitions),
        "CKPT_ROOT": str(ckpt_root),
        "COMMON_SH": str(tmp_path / "missing_common.sh"),
        "MODEL": "/model",
        "PYTHON": sys.executable,
        "SCORER": str(fake_scorer),
        "REPORTER": str(fake_reporter),
        "OUT_DIR": str(out_dir),
        "STEPS": "init",
        "GENERATE_REPORT": "0",
    }

    result = subprocess.run(["bash", str(script)], env=env, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    assert "WM_CHECKPOINT_DIAGNOSTICS_DONE" in result.stdout
    assert "report_md=" not in result.stdout
    assert (out_dir / "checkpoint_scores.csv").stat().st_size > 0
    assert not (out_dir / "checkpoint_diagnostics_report.md").exists()
