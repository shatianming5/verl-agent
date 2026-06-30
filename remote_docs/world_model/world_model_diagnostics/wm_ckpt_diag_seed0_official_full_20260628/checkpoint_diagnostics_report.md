# World-Model Checkpoint Diagnostics

## wm_ckpt_diag_seed0_official_full_20260628

- Summary JSON: `/mnt/cephfs_home_tianming.sha/grpo_alfworld/logs/world_model_diagnostics/wm_ckpt_diag_seed0_official_full_20260628/checkpoint_scores_summary.json`
- Transition JSONL: `/mnt/cephfs_home_tianming.sha/grpo_alfworld/logs/world_model_diagnostics/wm_valdump_smoke_s0_step150/150.val.wm_transitions.jsonl`
- Rows: `277`
- Max length: `512`
- Success/failure buckets: `backfilled from /mnt/cephfs_home_tianming.sha/grpo_alfworld/logs/world_model_diagnostics/wm_ckpt_diag_seed0_official_full_20260628/checkpoint_scores.csv`

| checkpoint | step | target tokens | CE | delta CE | confidence | cosine | delta cosine | failure-success CE | success-failure cosine |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grpo_qwen2.5_1.5b_alfworld_seed0_official_4to5_init | init | 119991 | 1.3793 | 0.0000 | 0.6722 | 0.6144 | 0.0000 | -0.0341 | 0.0141 |
| grpo_qwen2.5_1.5b_alfworld_seed0_official_4to5_step30 | 30 | 119991 | 1.3528 | -0.0265 | 0.6654 | 0.6655 | 0.0511 | -0.0277 | 0.0030 |
| grpo_qwen2.5_1.5b_alfworld_seed0_official_4to5_step60 | 60 | 119991 | 1.3421 | -0.0372 | 0.6721 | 0.6872 | 0.0728 | -0.0362 | 0.0035 |
| grpo_qwen2.5_1.5b_alfworld_seed0_official_4to5_step90 | 90 | 119991 | 1.3568 | -0.0226 | 0.6648 | 0.6754 | 0.0610 | -0.0364 | 0.0022 |
| grpo_qwen2.5_1.5b_alfworld_seed0_official_4to5_step120 | 120 | 119991 | 1.3638 | -0.0155 | 0.6643 | 0.6881 | 0.0737 | -0.0475 | 0.0022 |
| grpo_qwen2.5_1.5b_alfworld_seed0_official_4to5_step150 | 150 | 119991 | 1.4330 | 0.0537 | 0.6668 | 0.6901 | 0.0757 | -0.2286 | -0.0012 |
