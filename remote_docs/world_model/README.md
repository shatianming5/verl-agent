# World-Model GOAL_RD Artifacts

This directory mirrors the current small report artifacts from gpudev for offline review.

- Source root: `/mnt/cephfs_home_tianming.sha/grpo_alfworld/logs`
- Branch: `world-model-latent-objective`
- Report revision: `83c98aeaaa233da3a84fd073b60e0b2b55dac098`
- Rows: `12` total, `11` expected GOAL_RD rows
- Mirrored files:
  - `world_model_diagnostics/wm_ckpt_diag_seed0_official_full_20260628/checkpoint_diagnostics_report.csv`
  - `world_model_diagnostics/wm_ckpt_diag_seed0_official_full_20260628/checkpoint_diagnostics_report.md`
  - `world_model_diagnostics/wm_ckpt_diag_seed0_official_full_20260628/checkpoint_diagnostics_report.svg`
  - `world_model_diagnostics/wm_ckpt_diag_seed0_official_full_20260628/checkpoint_scores_summary.json`
  - `world_model_diagnostics/wm_ckpt_diag_seed0_smoke_20260628/checkpoint_scores_summary.json`
  - `world_model_diagnostics/wm_obs_ce_l0p01_s0/checkpoint_diagnostics_report.csv`
  - `world_model_diagnostics/wm_obs_ce_l0p01_s0/checkpoint_diagnostics_report.md`
  - `world_model_diagnostics/wm_obs_ce_l0p01_s0/checkpoint_diagnostics_report.svg`
  - `world_model_diagnostics/wm_obs_ce_l0p01_s0/checkpoint_scores_summary.json`
  - `world_model_diagnostics/wm_obs_ce_l0p01_s1/checkpoint_diagnostics_report.csv`
  - `world_model_diagnostics/wm_obs_ce_l0p01_s1/checkpoint_diagnostics_report.md`
  - `world_model_diagnostics/wm_obs_ce_l0p01_s1/checkpoint_diagnostics_report.svg`
  - `world_model_diagnostics/wm_obs_ce_l0p01_s1/checkpoint_scores_summary.json`
  - `world_model_diagnostics/wmlat_l0p001_s0/checkpoint_diagnostics_report.csv`
  - `world_model_diagnostics/wmlat_l0p001_s0/checkpoint_diagnostics_report.md`
  - `world_model_diagnostics/wmlat_l0p001_s0/checkpoint_diagnostics_report.svg`
  - `world_model_diagnostics/wmlat_l0p001_s0/checkpoint_scores_summary.json`
  - `world_model_results.csv`
  - `world_model_results.md`

The mirrored reports are snapshots. The authoritative live artifacts remain on gpudev under the source root above.

Regenerate with `python scripts/mirror_world_model_artifacts.py` after refreshing the gpudev reports.
