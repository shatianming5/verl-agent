# World-Model Checkpoint Diagnostics

## wm_obs_ce_l0p01_s0

- Summary JSON: `/mnt/cephfs_home_tianming.sha/grpo_alfworld/logs/world_model_diagnostics/wm_obs_ce_l0p01_s0/checkpoint_scores_summary.json`
- Transition JSONL: `/mnt/cephfs_home_tianming.sha/grpo_alfworld/logs/world_model_rollouts/wm_obs_ce_l0p01_s0_seed0/150.wm_transitions.jsonl`
- Rows: `4173`
- Max length: `512`
- Diagnostic command: `/tmp/wm_diag_scripts_21838b85edd9/wm_score_transition_dump.py --model-path /mnt/cephfs_home_tianming.sha/grpo_alfworld/models/Qwen2.5-1.5B-Instruct --transition-jsonl /mnt/cephfs_home_tianming.sha/grpo_alfworld/logs/world_model_rollouts/wm_obs_ce_l0p01_s0_seed0/150.wm_transitions.jsonl --output-csv /mnt/cephfs_home_tianming.sha/grpo_alfworld/logs/world_model_diagnostics/wm_obs_ce_l0p01_s0/checkpoint_scores.csv --summary-json /mnt/cephfs_home_tianming.sha/grpo_alfworld/logs/world_model_diagnostics/wm_obs_ce_l0p01_s0/checkpoint_scores_summary.json --max-length 512 --batch-size 1 --device cuda --dtype bfloat16 --chat-template-kwargs-json '{}' --checkpoint grpo_qwen2.5_1.5b_alfworld_seed0_wm_obs_ce_l0p01_s0_init=base --checkpoint grpo_qwen2.5_1.5b_alfworld_seed0_wm_obs_ce_l0p01_s0_step30=/mnt/cephfs_home_tianming.sha/grpo_alfworld/checkpoints/grpo_qwen2.5_1.5b_alfworld_seed0_wm_obs_ce_l0p01_s0/global_step_30 --checkpoint grpo_qwen2.5_1.5b_alfworld_seed0_wm_obs_ce_l0p01_s0_step60=/mnt/cephfs_home_tianming.sha/grpo_alfworld/checkpoints/grpo_qwen2.5_1.5b_alfworld_seed0_wm_obs_ce_l0p01_s0/global_step_60 --checkpoint grpo_qwen2.5_1.5b_alfworld_seed0_wm_obs_ce_l0p01_s0_step90=/mnt/cephfs_home_tianming.sha/grpo_alfworld/checkpoints/grpo_qwen2.5_1.5b_alfworld_seed0_wm_obs_ce_l0p01_s0/global_step_90 --checkpoint grpo_qwen2.5_1.5b_alfworld_seed0_wm_obs_ce_l0p01_s0_step120=/mnt/cephfs_home_tianming.sha/grpo_alfworld/checkpoints/grpo_qwen2.5_1.5b_alfworld_seed0_wm_obs_ce_l0p01_s0/global_step_120 --checkpoint grpo_qwen2.5_1.5b_alfworld_seed0_wm_obs_ce_l0p01_s0_step150=/mnt/cephfs_home_tianming.sha/grpo_alfworld/checkpoints/grpo_qwen2.5_1.5b_alfworld_seed0_wm_obs_ce_l0p01_s0/global_step_150`
- Model path: `/mnt/cephfs_home_tianming.sha/grpo_alfworld/models/Qwen2.5-1.5B-Instruct`
- Per-transition CSV: `/mnt/cephfs_home_tianming.sha/grpo_alfworld/logs/world_model_diagnostics/wm_obs_ce_l0p01_s0/checkpoint_scores.csv`
- Device: `cuda`
- Dtype: `bfloat16`
- Batch size: `1`
- Max rows: `0`
- Scored checkpoints: `6` (grpo_qwen2.5_1.5b_alfworld_seed0_wm_obs_ce_l0p01_s0_init, grpo_qwen2.5_1.5b_alfworld_seed0_wm_obs_ce_l0p01_s0_step30, grpo_qwen2.5_1.5b_alfworld_seed0_wm_obs_ce_l0p01_s0_step60, grpo_qwen2.5_1.5b_alfworld_seed0_wm_obs_ce_l0p01_s0_step90, grpo_qwen2.5_1.5b_alfworld_seed0_wm_obs_ce_l0p01_s0_step120, grpo_qwen2.5_1.5b_alfworld_seed0_wm_obs_ce_l0p01_s0_step150)

| checkpoint | step | target tokens | CE | delta CE | confidence | cosine | delta cosine | failure-success CE | success-failure cosine |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grpo_qwen2.5_1.5b_alfworld_seed0_wm_obs_ce_l0p01_s0_init | init | 2019486 | 1.4352 | 0.0000 | 0.6838 | 0.1687 | 0.0000 | -0.0120 | 0.0101 |
| grpo_qwen2.5_1.5b_alfworld_seed0_wm_obs_ce_l0p01_s0_step30 | 30 | 2019486 | 0.4773 | -0.9579 | 0.8612 | 0.1522 | -0.0164 | -0.0123 | 0.0087 |
| grpo_qwen2.5_1.5b_alfworld_seed0_wm_obs_ce_l0p01_s0_step60 | 60 | 2019486 | 0.3600 | -1.0752 | 0.8923 | 0.1583 | -0.0104 | 0.0022 | 0.0048 |
| grpo_qwen2.5_1.5b_alfworld_seed0_wm_obs_ce_l0p01_s0_step90 | 90 | 2019486 | 0.3178 | -1.1174 | 0.9044 | 0.1358 | -0.0328 | 0.0045 | 0.0071 |
| grpo_qwen2.5_1.5b_alfworld_seed0_wm_obs_ce_l0p01_s0_step120 | 120 | 2019486 | 0.2910 | -1.1442 | 0.9137 | 0.1371 | -0.0316 | 0.0031 | 0.0058 |
| grpo_qwen2.5_1.5b_alfworld_seed0_wm_obs_ce_l0p01_s0_step150 | 150 | 2019486 | 0.2732 | -1.1620 | 0.9171 | 0.1509 | -0.0177 | 0.0007 | 0.0028 |
