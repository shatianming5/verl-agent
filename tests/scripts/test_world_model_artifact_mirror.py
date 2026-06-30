import csv
import re
from pathlib import Path


def test_world_model_artifact_mirror_has_goal_rd_rows_and_diagnostics():
    repo_root = Path(__file__).resolve().parents[2]
    artifact_root = repo_root / "remote_docs" / "world_model"
    results_md = artifact_root / "world_model_results.md"
    results_csv = artifact_root / "world_model_results.csv"
    diagnostic_csv = (
        artifact_root
        / "world_model_diagnostics"
        / "wm_ckpt_diag_seed0_official_full_20260628"
        / "checkpoint_diagnostics_report.csv"
    )

    markdown = results_md.read_text(encoding="utf-8")
    revision = re.search(r"^- Report revision: `([^`]+)`", markdown, re.MULTILINE)
    assert revision and len(revision.group(1)) >= 7

    with results_csv.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    expected_rows = [row for row in rows if row.get("expected") == "yes"]
    assert len(rows) == 12
    assert len(expected_rows) == 11

    by_key = {row["run_key"]: row for row in rows}
    assert by_key["grpo_baseline_s0"]["final_report_readiness"] == "complete"
    assert by_key["grpo_baseline_s0"]["checkpoint_backup_status"] == "backed_up"
    assert by_key["grpo_baseline_s0"]["diagnostic_success_failure_ce_gap"]
    assert by_key["obs_ce_l0p01_s0"]["eval_readiness"] == "waiting_for_checkpoint"
    assert by_key["latent_l0p001_s1"]["eval_readiness"] == "waiting_for_checkpoint"

    with diagnostic_csv.open(encoding="utf-8") as handle:
        diagnostic_rows = list(csv.DictReader(handle))
    assert len(diagnostic_rows) == 6
    assert diagnostic_rows[-1]["checkpoint_step"] == "150"
    assert diagnostic_rows[-1]["success_failure_ce_gap"]
    assert diagnostic_rows[-1]["success_failure_cosine_gap"]
