from pathlib import Path


def test_full_train_rollout_script_is_explicit_and_fail_closed():
    repo_root = Path(__file__).resolve().parents[2]
    script = (repo_root / "scripts" / "wm_rollout_trainset_dump.sh").read_text(encoding="utf-8")

    assert "set -euo pipefail" in script
    assert "examples.data_preprocess.prepare" not in script
    assert "geometry3k" not in script.lower()
    assert "MANIFEST=${MANIFEST:?" in script
    assert "EXPECTED_GAMES=${EXPECTED_GAMES:-3553}" in script
    assert "EXPECTED_RAW_TRAJECTORIES=${EXPECTED_RAW_TRAJECTORIES:-6374}" in script
    assert "wm_alfworld_train_manifest.py" in script
    assert "wm_validate_rollout_coverage.py" in script
    assert "actor_rollout_ref.rollout.temperature=" in script
    assert "actor_rollout_ref.rollout.top_p=" in script
    assert "actor_rollout_ref.rollout.top_k=" in script
    assert "actor_rollout_ref.rollout.do_sample=True" in script
    assert "actor_rollout_ref.rollout.val_kwargs.temperature=" in script
    assert "actor_rollout_ref.rollout.val_kwargs.top_p=" in script
    assert "actor_rollout_ref.rollout.val_kwargs.top_k=" in script
    assert "actor_rollout_ref.rollout.val_kwargs.do_sample=True" in script
    assert "workstream_b_full_train_v2" in script
    assert '  trainer.validation_data_dir="$DUMP_DIR"' in script
    assert "+trainer.validation_data_dir" not in script
    assert script.index("wm_validate_rollout_coverage.py") < script.index("BDIAG_ROLLOUT_DONE")
