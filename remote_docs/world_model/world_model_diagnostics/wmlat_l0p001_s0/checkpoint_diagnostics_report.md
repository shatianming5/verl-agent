# World-Model Checkpoint Diagnostics

## wmlat_l0p001_s0

- Summary JSON: `/mnt/cephfs_home_tianming.sha/grpo_alfworld/logs/world_model_diagnostics/wmlat_l0p001_s0/checkpoint_scores_summary.json`
- Transition JSONL: `/mnt/cephfs_home_tianming.sha/grpo_alfworld/logs/world_model_rollouts/wmlat_l0p001_s0_seed0/150.wm_transitions.jsonl`
- Rows: `4119`
- Max length: `512`
- Diagnostic command: `/tmp/wm_diag_scripts_21838b85edd9/wm_score_transition_dump.py --model-path /mnt/cephfs_home_tianming.sha/grpo_alfworld/models/Qwen2.5-1.5B-Instruct --transition-jsonl /mnt/cephfs_home_tianming.sha/grpo_alfworld/logs/world_model_rollouts/wmlat_l0p001_s0_seed0/150.wm_transitions.jsonl --output-csv /mnt/cephfs_home_tianming.sha/grpo_alfworld/logs/world_model_diagnostics/wmlat_l0p001_s0/checkpoint_scores.csv --summary-json /mnt/cephfs_home_tianming.sha/grpo_alfworld/logs/world_model_diagnostics/wmlat_l0p001_s0/checkpoint_scores_summary.json --max-length 512 --batch-size 1 --device cuda --dtype bfloat16 --chat-template-kwargs-json '{}' --checkpoint grpo_qwen2.5_1.5b_alfworld_seed0_wmlat_l0p001_s0_init=base --checkpoint grpo_qwen2.5_1.5b_alfworld_seed0_wmlat_l0p001_s0_step30=/mnt/cephfs_home_tianming.sha/grpo_alfworld/checkpoints/grpo_qwen2.5_1.5b_alfworld_seed0_wmlat_l0p001_s0/global_step_30 --checkpoint grpo_qwen2.5_1.5b_alfworld_seed0_wmlat_l0p001_s0_step60=/mnt/cephfs_home_tianming.sha/grpo_alfworld/checkpoints/grpo_qwen2.5_1.5b_alfworld_seed0_wmlat_l0p001_s0/global_step_60 --checkpoint grpo_qwen2.5_1.5b_alfworld_seed0_wmlat_l0p001_s0_step90=/mnt/cephfs_home_tianming.sha/grpo_alfworld/checkpoints/grpo_qwen2.5_1.5b_alfworld_seed0_wmlat_l0p001_s0/global_step_90 --checkpoint grpo_qwen2.5_1.5b_alfworld_seed0_wmlat_l0p001_s0_step120=/mnt/cephfs_home_tianming.sha/grpo_alfworld/checkpoints/grpo_qwen2.5_1.5b_alfworld_seed0_wmlat_l0p001_s0/global_step_120 --checkpoint grpo_qwen2.5_1.5b_alfworld_seed0_wmlat_l0p001_s0_step150=/mnt/cephfs_home_tianming.sha/grpo_alfworld/checkpoints/grpo_qwen2.5_1.5b_alfworld_seed0_wmlat_l0p001_s0/global_step_150`
- Model path: `/mnt/cephfs_home_tianming.sha/grpo_alfworld/models/Qwen2.5-1.5B-Instruct`
- Per-transition CSV: `/mnt/cephfs_home_tianming.sha/grpo_alfworld/logs/world_model_diagnostics/wmlat_l0p001_s0/checkpoint_scores.csv`
- Device: `cuda`
- Dtype: `bfloat16`
- Batch size: `1`
- Max rows: `0`
- Scored checkpoints: `6` (grpo_qwen2.5_1.5b_alfworld_seed0_wmlat_l0p001_s0_init, grpo_qwen2.5_1.5b_alfworld_seed0_wmlat_l0p001_s0_step30, grpo_qwen2.5_1.5b_alfworld_seed0_wmlat_l0p001_s0_step60, grpo_qwen2.5_1.5b_alfworld_seed0_wmlat_l0p001_s0_step90, grpo_qwen2.5_1.5b_alfworld_seed0_wmlat_l0p001_s0_step120, grpo_qwen2.5_1.5b_alfworld_seed0_wmlat_l0p001_s0_step150)

| checkpoint | step | target tokens | CE | delta CE | confidence | cosine | delta cosine | failure-success CE | success-failure cosine |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grpo_qwen2.5_1.5b_alfworld_seed0_wmlat_l0p001_s0_init | init | 1986513 | 1.4250 | 0.0000 | 0.6861 | 0.1762 | 0.0000 | 0.0423 | -0.0787 |
| grpo_qwen2.5_1.5b_alfworld_seed0_wmlat_l0p001_s0_step30 | 30 | 1986513 | 1.3786 | -0.0464 | 0.6794 | 0.1543 | -0.0218 | 0.0304 | -0.0901 |
| grpo_qwen2.5_1.5b_alfworld_seed0_wmlat_l0p001_s0_step60 | 60 | 1986513 | 1.3576 | -0.0674 | 0.6773 | 0.1461 | -0.0300 | 0.0248 | -0.0949 |
| grpo_qwen2.5_1.5b_alfworld_seed0_wmlat_l0p001_s0_step90 | 90 | 1986513 | 1.3678 | -0.0573 | 0.6737 | 0.1158 | -0.0603 | 0.0232 | -0.0996 |
| grpo_qwen2.5_1.5b_alfworld_seed0_wmlat_l0p001_s0_step120 | 120 | 1986513 | 1.3580 | -0.0671 | 0.6760 | 0.1330 | -0.0432 | 0.0184 | -0.0982 |
| grpo_qwen2.5_1.5b_alfworld_seed0_wmlat_l0p001_s0_step150 | 150 | 1986513 | 1.4187 | -0.0064 | 0.6767 | 0.1378 | -0.0384 | -0.0674 | -0.0923 |
