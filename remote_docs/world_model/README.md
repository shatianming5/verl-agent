# World-Model GOAL_RD Artifacts

This directory mirrors the current small report artifacts from gpudev for offline review.

- Source root: `/mnt/cephfs_home_tianming.sha/grpo_alfworld/logs`
- Branch: `world-model-latent-objective`
- Report revision: `a5addc05273c040132c05f7748a7a569de4e0794`
- Rows: `12` total, `11` expected GOAL_RD rows
- Mirrored files:
  - `world_model_diagnostics/wm_ckpt_diag_seed0_official_full_20260628/checkpoint_diagnostics_report.csv`
  - `world_model_diagnostics/wm_ckpt_diag_seed0_official_full_20260628/checkpoint_diagnostics_report.md`
  - `world_model_diagnostics/wm_ckpt_diag_seed0_official_full_20260628/checkpoint_diagnostics_report.svg`
  - `world_model_results.csv`
  - `world_model_results.md`

The mirrored reports are snapshots. The authoritative live artifacts remain on gpudev under the source root above.

Regenerate with `python scripts/mirror_world_model_artifacts.py` after refreshing the gpudev reports.
