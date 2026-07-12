import json
import os
import subprocess
import sys
from pathlib import Path

STEPS = ("15", "30", "45", "60", "75", "90", "105", "120", "135", "150")


def _checkpoint_root(root, name):
    checkpoint_root = root / name
    for step in STEPS:
        actor = checkpoint_root / f"global_step_{step}" / "actor"
        actor.mkdir(parents=True)
        for kind in ("model", "optim", "extra_state"):
            (actor / f"{kind}_world_size_1_rank_0.pt").write_bytes(f"{name}-{kind}-{step}".encode())
    return checkpoint_root


def _tools(tmp_path):
    call_log = tmp_path / "rollout_calls.txt"
    manifest_tool = tmp_path / "manifest_tool.py"
    manifest_tool.write_text(
        "import sys\nassert sys.argv[1] == 'verify'\n",
        encoding="utf-8",
    )
    rollout = tmp_path / "rollout.sh"
    rollout.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f'printf \'%s,%s\\n\' "$ENV_SEED" "$CKPT_STEP" >> {call_log!s}\n'
        'printf \'{"schema_version":"wm_transition_v2"}\\n\' > "$DUMP_DIR/$CKPT_STEP.wm_transitions.jsonl"\n'
        'printf \'{"covered_games":3,"manifest_games":3,"checkpoint_step":"%s"}\\n\' "$CKPT_STEP" > "$DUMP_DIR/coverage.json"\n',
        encoding="utf-8",
    )
    rollout.chmod(0o755)
    scorer = tmp_path / "scorer.py"
    scorer.write_text(
        "import json, pathlib, sys\n"
        "args=sys.argv[1:]\n"
        "csv=pathlib.Path(args[args.index('--output-csv')+1])\n"
        "summary=pathlib.Path(args[args.index('--summary-json')+1])\n"
        "csv.write_text('checkpoint_step\\n15\\n', encoding='utf-8')\n"
        "summary.write_text(json.dumps({'raw_cosine_only': True}), encoding='utf-8')\n",
        encoding="utf-8",
    )
    analyzer = tmp_path / "analyzer.py"
    analyzer.write_text(
        "import csv, pathlib, sys\n"
        "args=sys.argv[1:]\n"
        "out=pathlib.Path(args[args.index('--out-dir')+1]); out.mkdir(parents=True, exist_ok=True)\n"
        "exp=args[args.index('--exp')+1]\n"
        "(out/f'workstream_b_report_{exp}.md').write_text('# report\\n', encoding='utf-8')\n"
        "path=out/f'paired_game_trends_{exp}.csv'\n"
        "fields=['level','metric','statistic','paired_games','slope_per_step','slope_ci_lo','slope_ci_hi']\n"
        "with path.open('w', newline='', encoding='utf-8') as handle:\n"
        " writer=csv.DictWriter(handle, fieldnames=fields); writer.writeheader()\n"
        " for level in ('episode','transition'):\n"
        "  for metric in ('ce','nll','perplexity','target_confidence_mean','raw_action_obs_cosine'):\n"
        "   for statistic in ('mean_all','mean_succ','mean_fail','gap'):\n"
        "    writer.writerow({'level':level,'metric':metric,'statistic':statistic,'paired_games':3553,'slope_per_step':0.01,'slope_ci_lo':0.005,'slope_ci_hi':0.015})\n",
        encoding="utf-8",
    )
    probe = tmp_path / "probe.py"
    probe.write_text(
        "import pathlib, sys\nargs=sys.argv[1:]\npath=pathlib.Path(args[args.index('--out-csv')+1]); path.write_text('feature,auc\\naction,0.5\\n', encoding='utf-8')\n",
        encoding="utf-8",
    )
    return call_log, manifest_tool, rollout, scorer, analyzer, probe


def test_full_driver_runs_own_rollout_for_every_seed_checkpoint(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    driver = repo_root / "scripts" / "run_wm_full_train_diagnostics.sh"
    seed0 = _checkpoint_root(tmp_path, "seed0")
    seed1 = _checkpoint_root(tmp_path, "seed1")
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}\n", encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"model")
    call_log, manifest_tool, rollout, scorer, analyzer, probe = _tools(tmp_path)
    out_root = tmp_path / "out"
    env = {
        **os.environ,
        "MODEL": str(model),
        "MANIFEST": str(manifest),
        "SEED0_CKPT_ROOT": str(seed0),
        "SEED1_CKPT_ROOT": str(seed1),
        "OUT_ROOT": str(out_root),
        "CUDA_VISIBLE_DEVICES": "0",
        "PYTHON": sys.executable,
        "PREPARE_PYTHON": sys.executable,
        "MANIFEST_TOOL": str(manifest_tool),
        "ROLLOUT_SCRIPT": str(rollout),
        "SCORER": str(scorer),
        "ANALYZER": str(analyzer),
        "PROBE": str(probe),
        "RUN_PROBE": "1",
        "N_GPUS": "1",
        "ROLLOUT_TP": "1",
        "DEVICE": "cpu",
        "DTYPE": "float32",
    }

    result = subprocess.run(
        ["bash", str(driver)],
        env=env,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    calls = call_log.read_text(encoding="utf-8").splitlines()
    assert len(calls) == 22
    assert calls[:3] == ["0,init", "0,15", "0,30"]
    assert calls[-2:] == ["1,135", "1,150"]
    done = json.loads((out_root / "FULL_PROTOCOL_DONE.json").read_text(encoding="utf-8"))
    assert done["status"] == "complete"
    assert done["seeds"] == [0, 1]
    assert len(done["steps"]) == 11
    assert done["expected_raw_trajectories"] == 6374
    assert done["expected_games"] == 3553
    assert len(list((out_root / "preflight").glob("*_actor_inventory.json"))) == 20
    assert (out_root / "cross_seed_trend_consistency.csv").is_file()
    assert (out_root / "cross_seed_trend_consistency.md").is_file()
    assert "WM_FULL_TRAIN_DIAGNOSTICS_DONE" in result.stdout
    script_text = driver.read_text(encoding="utf-8")
    assert script_text.index("os.replace(report_staging, report)") < script_text.index("os.replace(staging, path)")


def test_full_driver_preflights_all_checkpoints_before_rollout(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    driver = repo_root / "scripts" / "run_wm_full_train_diagnostics.sh"
    seed0 = _checkpoint_root(tmp_path, "seed0")
    seed1 = _checkpoint_root(tmp_path, "seed1")
    actor = seed1 / "global_step_75" / "actor"
    for kind in ("model", "optim", "extra_state"):
        (actor / f"{kind}_world_size_1_rank_0.pt").unlink()
    actor.rmdir()
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}\n", encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"model")
    call_log, manifest_tool, rollout, scorer, analyzer, probe = _tools(tmp_path)
    env = {
        **os.environ,
        "MODEL": str(model),
        "MANIFEST": str(manifest),
        "SEED0_CKPT_ROOT": str(seed0),
        "SEED1_CKPT_ROOT": str(seed1),
        "OUT_ROOT": str(tmp_path / "out"),
        "CUDA_VISIBLE_DEVICES": "0",
        "PYTHON": sys.executable,
        "PREPARE_PYTHON": sys.executable,
        "MANIFEST_TOOL": str(manifest_tool),
        "ROLLOUT_SCRIPT": str(rollout),
        "SCORER": str(scorer),
        "ANALYZER": str(analyzer),
        "PROBE": str(probe),
        "N_GPUS": "1",
        "ROLLOUT_TP": "1",
    }

    result = subprocess.run(
        ["bash", str(driver)],
        env=env,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    assert result.returncode != 0
    assert "Actor directory does not exist" in result.stdout
    assert not call_log.exists()
    assert not (tmp_path / "out" / "FULL_PROTOCOL_DONE.json").exists()


def test_full_driver_can_stage_archived_checkpoints_before_rollout(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    driver = repo_root / "scripts" / "run_wm_full_train_diagnostics.sh"
    archive0 = _checkpoint_root(tmp_path, "archive0")
    archive1 = _checkpoint_root(tmp_path, "archive1")
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}\n", encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"model")
    call_log, manifest_tool, rollout, scorer, analyzer, probe = _tools(tmp_path)
    out_root = tmp_path / "out"
    stage_root = tmp_path / "staged"
    env = {
        **os.environ,
        "MODEL": str(model),
        "MANIFEST": str(manifest),
        "SEED0_ARCHIVE_ROOT": str(archive0),
        "SEED1_ARCHIVE_ROOT": str(archive1),
        "STAGE_CHECKPOINTS": "1",
        "STAGE_ROOT": str(stage_root),
        "OUT_ROOT": str(out_root),
        "CUDA_VISIBLE_DEVICES": "0",
        "PYTHON": sys.executable,
        "PREPARE_PYTHON": sys.executable,
        "MANIFEST_TOOL": str(manifest_tool),
        "ROLLOUT_SCRIPT": str(rollout),
        "SCORER": str(scorer),
        "ANALYZER": str(analyzer),
        "PROBE": str(probe),
        "N_GPUS": "1",
        "ROLLOUT_TP": "1",
        "DEVICE": "cpu",
        "DTYPE": "float32",
    }

    subprocess.run(
        ["bash", str(driver)],
        env=env,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    assert len(call_log.read_text(encoding="utf-8").splitlines()) == 22
    assert (stage_root / "seed0" / "actor_stage_receipt.json").is_file()
    assert (stage_root / "seed1" / "actor_stage_receipt.json").is_file()
    done = json.loads((out_root / "FULL_PROTOCOL_DONE.json").read_text(encoding="utf-8"))
    assert done["checkpoint_staging_enabled"] is True
    assert done["checkpoint_stage_root"] == str(stage_root)
