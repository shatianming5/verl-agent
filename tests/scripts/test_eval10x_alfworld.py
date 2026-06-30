from pathlib import Path
import subprocess


def test_eval10x_script_uses_short_ray_tmpdir_for_long_labels():
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "eval10x_alfworld.sh"
    text = script.read_text(encoding="utf-8")

    assert "LABEL_HASH=$(printf" in text
    assert "export RAY_TMPDIR=/tmp/ray_eval_${LABEL_HASH}_${i}" in text
    assert "RAY_TMPDIR=/root/grpo/ray_tmp_${TAG}" not in text
    assert "LAMBDA_LATENT" in text
    assert "actor_rollout_ref.actor.world_model.lambda_latent=${LAMBDA_LATENT}" in text
    assert '"${extra_hydra_args[@]}"' in text

    label = "wm_obs_ce_l0p01_s0"
    label_hash = subprocess.check_output(["cksum"], input=label.encode("utf-8"), text=False).decode().split()[0]
    ray_tmpdir = f"/tmp/ray_eval_{label_hash}_0"
    plasma_path = f"{ray_tmpdir}/ray/session_2026-06-30_08-59-55_600172_681974/sockets/plasma_store"

    assert len(plasma_path) <= 107
