# World-Model Checkpoint Diagnostics

## wm_obs_ce_l0p01_s1

- Summary JSON: `/mnt/cephfs_home_tianming.sha/grpo_alfworld/logs/world_model_diagnostics/wm_obs_ce_l0p01_s1/checkpoint_scores_summary.json`
- Transition JSONL: `/mnt/cephfs_home_tianming.sha/grpo_alfworld/logs/world_model_rollouts/wm_obs_ce_l0p01_s1_seed1/150.wm_transitions.jsonl`
- Rows: `4394`
- Max length: `512`
- Diagnostic command: `/tmp/wm_diag_scripts_21838b85edd9/wm_score_transition_dump.py --model-path /mnt/cephfs_home_tianming.sha/grpo_alfworld/models/Qwen2.5-1.5B-Instruct --transition-jsonl /mnt/cephfs_home_tianming.sha/grpo_alfworld/logs/world_model_rollouts/wm_obs_ce_l0p01_s1_seed1/150.wm_transitions.jsonl --output-csv /mnt/cephfs_home_tianming.sha/grpo_alfworld/logs/world_model_diagnostics/wm_obs_ce_l0p01_s1/checkpoint_scores.csv --summary-json /mnt/cephfs_home_tianming.sha/grpo_alfworld/logs/world_model_diagnostics/wm_obs_ce_l0p01_s1/checkpoint_scores_summary.json --max-length 512 --batch-size 1 --device cuda --dtype bfloat16 --chat-template-kwargs-json '{}' --checkpoint grpo_qwen2.5_1.5b_alfworld_seed1_wm_obs_ce_l0p01_s1_init=base --checkpoint grpo_qwen2.5_1.5b_alfworld_seed1_wm_obs_ce_l0p01_s1_step30=/mnt/cephfs_home_tianming.sha/grpo_alfworld/checkpoints/grpo_qwen2.5_1.5b_alfworld_seed1_wm_obs_ce_l0p01_s1/global_step_30 --checkpoint grpo_qwen2.5_1.5b_alfworld_seed1_wm_obs_ce_l0p01_s1_step60=/mnt/cephfs_home_tianming.sha/grpo_alfworld/checkpoints/grpo_qwen2.5_1.5b_alfworld_seed1_wm_obs_ce_l0p01_s1/global_step_60 --checkpoint grpo_qwen2.5_1.5b_alfworld_seed1_wm_obs_ce_l0p01_s1_step90=/mnt/cephfs_home_tianming.sha/grpo_alfworld/checkpoints/grpo_qwen2.5_1.5b_alfworld_seed1_wm_obs_ce_l0p01_s1/global_step_90 --checkpoint grpo_qwen2.5_1.5b_alfworld_seed1_wm_obs_ce_l0p01_s1_step120=/mnt/cephfs_home_tianming.sha/grpo_alfworld/checkpoints/grpo_qwen2.5_1.5b_alfworld_seed1_wm_obs_ce_l0p01_s1/global_step_120 --checkpoint grpo_qwen2.5_1.5b_alfworld_seed1_wm_obs_ce_l0p01_s1_step150=/mnt/cephfs_home_tianming.sha/grpo_alfworld/checkpoints/grpo_qwen2.5_1.5b_alfworld_seed1_wm_obs_ce_l0p01_s1/global_step_150`
- Model path: `/mnt/cephfs_home_tianming.sha/grpo_alfworld/models/Qwen2.5-1.5B-Instruct`
- Per-transition CSV: `/mnt/cephfs_home_tianming.sha/grpo_alfworld/logs/world_model_diagnostics/wm_obs_ce_l0p01_s1/checkpoint_scores.csv`
- Device: `cuda`
- Dtype: `bfloat16`
- Batch size: `1`
- Max rows: `0`
- Scored checkpoints: `6` (grpo_qwen2.5_1.5b_alfworld_seed1_wm_obs_ce_l0p01_s1_init, grpo_qwen2.5_1.5b_alfworld_seed1_wm_obs_ce_l0p01_s1_step30, grpo_qwen2.5_1.5b_alfworld_seed1_wm_obs_ce_l0p01_s1_step60, grpo_qwen2.5_1.5b_alfworld_seed1_wm_obs_ce_l0p01_s1_step90, grpo_qwen2.5_1.5b_alfworld_seed1_wm_obs_ce_l0p01_s1_step120, grpo_qwen2.5_1.5b_alfworld_seed1_wm_obs_ce_l0p01_s1_step150)

| checkpoint | step | target tokens | CE | delta CE | confidence | cosine | delta cosine | failure-success CE | success-failure cosine |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grpo_qwen2.5_1.5b_alfworld_seed1_wm_obs_ce_l0p01_s1_init | init | 2084119 | 1.4289 | 0.0000 | 0.6849 | 0.2122 | 0.0000 | -0.0239 | 0.0398 |
| grpo_qwen2.5_1.5b_alfworld_seed1_wm_obs_ce_l0p01_s1_step30 | 30 | 2084119 | 0.4563 | -0.9726 | 0.8668 | 0.1951 | -0.0171 | -0.0129 | 0.0311 |
| grpo_qwen2.5_1.5b_alfworld_seed1_wm_obs_ce_l0p01_s1_step60 | 60 | 2084119 | 0.3470 | -1.0819 | 0.8941 | 0.1930 | -0.0192 | -0.0029 | 0.0255 |
| grpo_qwen2.5_1.5b_alfworld_seed1_wm_obs_ce_l0p01_s1_step90 | 90 | 2084119 | 0.3114 | -1.1176 | 0.9063 | 0.1788 | -0.0334 | 0.0017 | 0.0255 |
| grpo_qwen2.5_1.5b_alfworld_seed1_wm_obs_ce_l0p01_s1_step120 | 120 | 2084119 | 0.2838 | -1.1451 | 0.9138 | 0.1947 | -0.0175 | 0.0027 | 0.0234 |
| grpo_qwen2.5_1.5b_alfworld_seed1_wm_obs_ce_l0p01_s1_step150 | 150 | 2084119 | 0.2688 | -1.1601 | 0.9190 | 0.1626 | -0.0495 | 0.0021 | 0.0244 |
