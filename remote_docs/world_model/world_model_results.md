# ALFWorld World-Model Results

- Branch: `world-model-latent-objective`
- Work root: `/mnt/cephfs_home_tianming.sha/grpo_alfworld`
- Runs in table: `12`
- GOAL_RD summary scope: `11` expected run(s)
- Additional observed runs: `1`

## Report Generation

- Enabled flags: `--goal-rd-report --discover-standard-layout --expected-goal-rd-runs`
- Report revision: `2d53f57ea5883bd3e895e96a2b8a84430154df51`
- Eval result inputs: `3`
- Train log inputs: `43`
- Diagnostic summary inputs: `1`
- Expected run inputs: `11`
- Train CUDA: `<free_2gpu_pair>`
- Train script: `/root/grpo/run_seed_alfworld_official.sh`
- Train dump rollouts: `True`
- Eval CUDA: `<free_2gpu_pair>`
- Eval n: `10`
- Eval script: `/root/grpo/eval10x_alfworld.sh`
- Diagnostic script: `scripts/run_wm_checkpoint_diagnostics.sh`
- Diagnostic steps: `init 30 60 90 120 150`
- Diagnostic transition step: `150`

## GOAL_RD Deliverable Status

- Branch name: `world-model-latent-objective`
- Code/config summary: `obs CE objective, latent hidden-state objective, rollout transition dumps, checkpoint diagnostics, and report aggregation are tracked in this branch`
- Exact commands/configs: `per-run launch lines and generated train/eval/diagnostic commands are listed in Artifact Paths`
- Result table status: `3/11 tracked run(s) have eval results; eval readiness {'evaluated': 3, 'missing_training_log': 4, 'waiting_for_checkpoint': 4}`
- Diagnostic paths status: `1/11 tracked run(s) have checkpoint diagnostics; diagnostic readiness {'diagnosed': 1, 'missing_training_log': 4, 'missing_transition_dump': 2, 'waiting_for_checkpoint': 4}`
- Raw observation CE interpretation: `Raw observation CE: pending; evaluated baseline runs=3, evaluated obs_ce runs=0.`
- World-model feature interpretation: `Observation prediction features: evidence available; success/failure CE or cosine gaps are reported for 1 diagnostic run(s).`
- Latent alignment interpretation: `Latent alignment: pending; no latent eval10x result is available yet.`
- Training log coverage: `7/11 tracked run(s) have training logs`

| run | tag | objective | seed | lambda_obs | lambda_latent | eval mean +/- std | eval n | eval readiness | diag readiness | last online val | last WM metric | train step | ckpt backup | diag CE | delta CE | diag cosine | delta cosine | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grpo_baseline_s0 | official_4to5 | grpo_baseline | 0 |  |  | 0.7290 +/- 0.0282 (n=10) | 10 | evaluated | diagnosed | 0.7810 |  | 150 | backed_up | CE 1.4330 | 0.0537 | 0.6901 | 0.0757 | evaluated;training_seen;training_complete;diagnosed |
| grpo_baseline_s1 | official_6to7 | grpo_baseline | 1 |  |  | 0.6984 +/- 0.0375 (n=10) | 10 | evaluated | missing_transition_dump | 0.7340 |  | 150 | backed_up |  |  |  |  | evaluated;training_seen;training_complete |
| grpo_baseline_s2 | official_s2 | grpo_baseline | 2 |  |  | 0.6922 +/- 0.0321 (n=10) | 10 | evaluated | missing_transition_dump | 0.7190 |  | 150 | missing_backup |  |  |  |  | evaluated;training_seen;training_complete |
| obs_ce_l0p01_s0 | wm_obs_ce_l0p01_s0 | obs_ce | 0 | 0.01 |  |  |  | waiting_for_checkpoint | waiting_for_checkpoint | 0.6800 | obs_ce_loss=0.144 | 148 | missing_backup |  |  |  |  | training_seen |
| obs_ce_l0p01_s1 | wm_obs_ce_l0p01_s1 | obs_ce | 1 | 0.01 |  |  |  | waiting_for_checkpoint | waiting_for_checkpoint | 0.6560 | obs_ce_loss=0.150 | 142 | missing_backup |  |  |  |  | training_seen |
| obs_ce_l0p03_s0 | wm_obs_ce_l0p03_s0 | obs_ce | 0 | 0.03 |  |  |  | missing_training_log | missing_training_log |  |  |  | no_checkpoint |  |  |  |  |  |
| obs_ce_l0p03_s1 | wm_obs_ce_l0p03_s1 | obs_ce | 1 | 0.03 |  |  |  | missing_training_log | missing_training_log |  |  |  | no_checkpoint |  |  |  |  |  |
| obs_ce_l0p05_s0 | wm_obs_ce_l0p05_s0 | obs_ce | 0 | 0.05 |  |  |  | missing_training_log | missing_training_log |  |  |  | no_checkpoint |  |  |  |  |  |
| obs_ce_l0p05_s1 | wm_obs_ce_l0p05_s1 | obs_ce | 1 | 0.05 |  |  |  | missing_training_log | missing_training_log |  |  |  | no_checkpoint |  |  |  |  |  |
| latent_s0 | wmls_l1 | latent | 0 |  |  |  |  | waiting_for_checkpoint | waiting_for_checkpoint |  | latent_loss=1.007, cosine=-0.007 | 1 | no_checkpoint |  |  |  |  | training_complete |
| latent_l0p001_s0 | wmlat_l0p001_s0 | latent | 0 |  | 0.001 |  |  | waiting_for_checkpoint | waiting_for_checkpoint | 0.5780 | latent_loss=0.264, cosine=0.736 | 139 | missing_backup |  |  |  |  | training_seen |
| latent_l0p001_s1 | wmlat_l0p001_s1 | latent | 1 |  | 0.001 |  |  | waiting_for_checkpoint | waiting_for_checkpoint | 0.5780 | latent_loss=0.494, cosine=0.506 | 75 | missing_backup |  |  |  |  | training_seen |

## Result Summary

| condition | objective | lambda_obs | lambda_latent | runs | evaluated | seeds | mean eval | run std | mean eval std | delta vs baseline |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GRPO baseline | grpo_baseline |  |  | 3 | 3 | 0,1,2 | 0.7065 | 0.0197 | 0.0326 | 0.0000 |
| obs_ce lambda_obs=0.01 | obs_ce | 0.01 |  | 2 | 0 | 0,1 |  |  |  |  |
| obs_ce lambda_obs=0.03 | obs_ce | 0.03 |  | 2 | 0 | 0,1 |  |  |  |  |
| obs_ce lambda_obs=0.05 | obs_ce | 0.05 |  | 2 | 0 | 0,1 |  |  |  |  |
| latent lambda_latent=0.001 | latent |  | 0.001 | 2 | 0 | 0,1 |  |  |  |  |

## Objective Coverage

| objective | runs | seeds | evaluated | ready | waiting | eval incomplete | missing train log | diagnosed | diag ready | diag waiting | missing transitions |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grpo_baseline | 3 | 0,1,2 | 3 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 2 |
| obs_ce | 6 | 0,1 | 0 | 0 | 2 | 0 | 4 | 0 | 0 | 2 | 0 |
| latent | 2 | 0,1 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 2 | 0 |

## Eval Readiness

- evaluated: `3`
- missing_training_log: `4`
- waiting_for_checkpoint: `4`

## Diagnostic Readiness

- diagnosed: `1`
- missing_training_log: `4`
- missing_transition_dump: `2`
- waiting_for_checkpoint: `4`

## Expected Run Coverage

| run | objective | seed | train log | eval | diagnostic | eval readiness | diagnostic readiness | missing artifacts | final readiness |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grpo_baseline_s0 | grpo_baseline | 0 | yes | yes | yes | evaluated | diagnosed |  | complete |
| grpo_baseline_s1 | grpo_baseline | 1 | yes | yes | no | evaluated | missing_transition_dump | diagnostic | missing:diagnostic |
| grpo_baseline_s2 | grpo_baseline | 2 | yes | yes | no | evaluated | missing_transition_dump | diagnostic | missing:diagnostic |
| obs_ce_l0p01_s0 | obs_ce | 0 | yes | no | no | waiting_for_checkpoint | waiting_for_checkpoint | eval,diagnostic | missing:eval,diagnostic |
| obs_ce_l0p01_s1 | obs_ce | 1 | yes | no | no | waiting_for_checkpoint | waiting_for_checkpoint | eval,diagnostic | missing:eval,diagnostic |
| obs_ce_l0p03_s0 | obs_ce | 0 | no | no | no | missing_training_log | missing_training_log | train_log,eval,diagnostic | missing:train_log,eval,diagnostic |
| obs_ce_l0p03_s1 | obs_ce | 1 | no | no | no | missing_training_log | missing_training_log | train_log,eval,diagnostic | missing:train_log,eval,diagnostic |
| obs_ce_l0p05_s0 | obs_ce | 0 | no | no | no | missing_training_log | missing_training_log | train_log,eval,diagnostic | missing:train_log,eval,diagnostic |
| obs_ce_l0p05_s1 | obs_ce | 1 | no | no | no | missing_training_log | missing_training_log | train_log,eval,diagnostic | missing:train_log,eval,diagnostic |
| latent_l0p001_s0 | latent | 0 | yes | no | no | waiting_for_checkpoint | waiting_for_checkpoint | eval,diagnostic | missing:eval,diagnostic |
| latent_l0p001_s1 | latent | 1 | yes | no | no | waiting_for_checkpoint | waiting_for_checkpoint | eval,diagnostic | missing:eval,diagnostic |

## Artifact Paths

### grpo_baseline_s0
- Tag: `official_4to5`
- Eval target checkpoint: `/mnt/cephfs_home_tianming.sha/grpo_alfworld/checkpoints/grpo_qwen2.5_1.5b_alfworld_seed0_official_4to5/global_step_150`
- Eval result: `/mnt/cephfs_home_tianming.sha/grpo_alfworld/logs/eval10x_seed0_results.txt`
- Eval checkpoint: `/mnt/cephfs_home_tianming.sha/grpo_alfworld/checkpoints/grpo_qwen2.5_1.5b_alfworld_seed0_official_4to5/global_step_150`
- Latest checkpoint backup: `/mnt/cephfs_home_tianming.sha/grpo_alfworld/checkpoints_backup/grpo_qwen2.5_1.5b_alfworld_seed0_official_4to5/global_step_150`
- Diagnostic report: `/mnt/cephfs_home_tianming.sha/grpo_alfworld/logs/world_model_diagnostics/wm_ckpt_diag_seed0_official_full_20260628/checkpoint_diagnostics_report.md`
- Diagnostic report CSV: `/mnt/cephfs_home_tianming.sha/grpo_alfworld/logs/world_model_diagnostics/wm_ckpt_diag_seed0_official_full_20260628/checkpoint_diagnostics_report.csv`
- Diagnostic report SVG: `/mnt/cephfs_home_tianming.sha/grpo_alfworld/logs/world_model_diagnostics/wm_ckpt_diag_seed0_official_full_20260628/checkpoint_diagnostics_report.svg`
- Diagnostic summary: `/mnt/cephfs_home_tianming.sha/grpo_alfworld/logs/world_model_diagnostics/wm_ckpt_diag_seed0_official_full_20260628/checkpoint_scores_summary.json`
- Diagnostic transitions: `/mnt/cephfs_home_tianming.sha/grpo_alfworld/logs/world_model_diagnostics/wm_valdump_smoke_s0_step150/150.val.wm_transitions.jsonl`
- Diagnostic checkpoint root: `/mnt/cephfs_home_tianming.sha/grpo_alfworld/checkpoints/grpo_qwen2.5_1.5b_alfworld_seed0_official_4to5`
- Train log: `/mnt/cephfs_home_tianming.sha/grpo_alfworld/logs/grpo_qwen2.5_1.5b_alfworld_seed0_gpudev_h200x4_4to7_maxmem_20260624_062831.log;/mnt/cephfs_home_tianming.sha/grpo_alfworld/logs/grpo_qwen2.5_1.5b_alfworld_seed0_gpudev_h200x4_4to7_maxmem_20260624_103921.log;/mnt/cephfs_home_tianming.sha/grpo_alfworld/logs/grpo_qwen2.5_1.5b_alfworld_seed0_gpudev_h200x4_4to7_maxmem_20260624_132619.log;/mnt/cephfs_home_tianming.sha/grpo_alfworld/logs/grpo_qwen2.5_1.5b_alfworld_seed0_gpudev_h200x4_4to7_v2_20260625_085258.log;/mnt/cephfs_home_tianming.sha/grpo_alfworld/logs/grpo_qwen2.5_1.5b_alfworld_seed0_gpudev_h200x8_maxmem_20260624_055704.log;/mnt/cephfs_home_tianming.sha/grpo_alfworld/logs/grpo_qwen2.5_1.5b_alfworld_seed0_gpudev_h200x8_maxmem_20260624_113634.log;/mnt/cephfs_home_tianming.sha/grpo_alfworld/logs/grpo_qwen2.5_1.5b_alfworld_seed0_gpudev_h200x8_maxmem_20260624_132438.log;/mnt/cephfs_home_tianming.sha/grpo_alfworld/logs/grpo_qwen2.5_1.5b_alfworld_seed0_gpudev_h200x8_rerun_20260623_213838.log;/mnt/cephfs_home_tianming.sha/grpo_alfworld/logs/grpo_qwen2.5_1.5b_alfworld_seed0_gpudev_h200x8_v2_20260625_085015.log;/mnt/cephfs_home_tianming.sha/grpo_alfworld/logs/grpo_qwen2.5_1.5b_alfworld_seed0_h200x8_20260622_170359.log;/mnt/cephfs_home_tianming.sha/grpo_alfworld/logs/grpo_qwen2.5_1.5b_alfworld_seed0_h200x8_20260622_174839.log;/mnt/cephfs_home_tianming.sha/grpo_alfworld/logs/grpo_qwen2.5_1.5b_alfworld_seed0_h200x8_20260622_180659.log;/mnt/cephfs_home_tianming.sha/grpo_alfworld/logs/grpo_qwen2.5_1.5b_alfworld_seed0_h200x8_20260622_184130.log;/mnt/cephfs_home_tianming.sha/grpo_alfworld/logs/grpo_qwen2.5_1.5b_alfworld_seed0_official_4to5_20260626_225827.log;/mnt/cephfs_home_tianming.sha/grpo_alfworld/logs/grpo_qwen2.5_1.5b_alfworld_seed0_official_4to5_20260626_233942.log;/mnt/cephfs_home_tianming.sha/grpo_alfworld/logs/grpo_qwen2.5_1.5b_alfworld_seed0_official_4to5_20260626_234850.log;/mnt/cephfs_home_tianming.sha/grpo_alfworld/logs/grpo_qwen2.5_1.5b_alfworld_seed0_official_4to5_20260627_003942.log`
- Eval line: `EVAL10X_START label=seed0 ckpt=/mnt/cephfs_home_tianming.sha/grpo_alfworld/checkpoints/grpo_qwen2.5_1.5b_alfworld_seed0_official_4to5/global_step_150 n=10 val_size=128 dataset=eval_in_distribution cuda=6,7 Sun Jun 28 07:56:34 UTC 2026`
- Train command: `TAG=official_4to5 WM_DUMP_ROLLOUTS=1 CUDA_VISIBLE_DEVICES='<free_2gpu_pair>' bash /root/grpo/run_seed_alfworld_official.sh 0`

### grpo_baseline_s1
- Tag: `official_6to7`
- Eval target checkpoint: `/mnt/cephfs_home_tianming.sha/grpo_alfworld/checkpoints/grpo_qwen2.5_1.5b_alfworld_seed1_official_6to7/global_step_150`
- Eval result: `/mnt/cephfs_home_tianming.sha/grpo_alfworld/logs/eval10x_seed1_results.txt`
- Eval checkpoint: `/mnt/cephfs_home_tianming.sha/grpo_alfworld/checkpoints/grpo_qwen2.5_1.5b_alfworld_seed1_official_6to7/global_step_150`
- Latest checkpoint backup: `/mnt/cephfs_home_tianming.sha/grpo_alfworld/checkpoints_backup/grpo_qwen2.5_1.5b_alfworld_seed1_official_6to7/global_step_150`
- Diagnostic transitions: `/mnt/cephfs_home_tianming.sha/grpo_alfworld/logs/world_model_rollouts/grpo_qwen2.5_1.5b_alfworld_seed1_official_6to7/150.wm_transitions.jsonl`
- Diagnostic checkpoint root: `/mnt/cephfs_home_tianming.sha/grpo_alfworld/checkpoints/grpo_qwen2.5_1.5b_alfworld_seed1_official_6to7`
- Train log: `/mnt/cephfs_home_tianming.sha/grpo_alfworld/logs/grpo_qwen2.5_1.5b_alfworld_seed1_gpudev_2gpu_4to5_20260625_090345.log;/mnt/cephfs_home_tianming.sha/grpo_alfworld/logs/grpo_qwen2.5_1.5b_alfworld_seed1_gpudev_2gpu_4to5_20260625_102928.log;/mnt/cephfs_home_tianming.sha/grpo_alfworld/logs/grpo_qwen2.5_1.5b_alfworld_seed1_gpudev_h200x4_4to7_maxmem_20260625_045101.log;/mnt/cephfs_home_tianming.sha/grpo_alfworld/logs/grpo_qwen2.5_1.5b_alfworld_seed1_gpudev_h200x4_4to7_maxmem_20260625_045642.log;/mnt/cephfs_home_tianming.sha/grpo_alfworld/logs/grpo_qwen2.5_1.5b_alfworld_seed1_gpudev_h200x4_4to7_maxmem_20260625_050052.log;/mnt/cephfs_home_tianming.sha/grpo_alfworld/logs/grpo_qwen2.5_1.5b_alfworld_seed1_gpudev_h200x4_4to7_maxmem_20260625_050848.log;/mnt/cephfs_home_tianming.sha/grpo_alfworld/logs/grpo_qwen2.5_1.5b_alfworld_seed1_h200x4_gpus4_7_20260623_081252.log;/mnt/cephfs_home_tianming.sha/grpo_alfworld/logs/grpo_qwen2.5_1.5b_alfworld_seed1_h200x4_gpus4_7_20260623_082634.log;/mnt/cephfs_home_tianming.sha/grpo_alfworld/logs/grpo_qwen2.5_1.5b_alfworld_seed1_h200x4_gpus4_7_20260623_112835.log;/mnt/cephfs_home_tianming.sha/grpo_alfworld/logs/grpo_qwen2.5_1.5b_alfworld_seed1_h200x8_20260623_074616.log;/mnt/cephfs_home_tianming.sha/grpo_alfworld/logs/grpo_qwen2.5_1.5b_alfworld_seed1_official_6to7_20260626_225927.log;/mnt/cephfs_home_tianming.sha/grpo_alfworld/logs/grpo_qwen2.5_1.5b_alfworld_seed1_official_6to7_20260626_234042.log;/mnt/cephfs_home_tianming.sha/grpo_alfworld/logs/grpo_qwen2.5_1.5b_alfworld_seed1_official_6to7_20260626_234950.log;/mnt/cephfs_home_tianming.sha/grpo_alfworld/logs/grpo_qwen2.5_1.5b_alfworld_seed1_official_6to7_20260627_004042.log`
- Launch line: `RUN_SEED_GPUDEV_MAXMEM seed=1 cuda_visible_devices=4,5,6,7 n_gpus=4 exp=grpo_qwen2.5_1.5b_alfworld_seed1_gpudev_h200x4_4to7_maxmem`
- Eval line: `EVAL10X_START label=seed1 ckpt=/mnt/cephfs_home_tianming.sha/grpo_alfworld/checkpoints/grpo_qwen2.5_1.5b_alfworld_seed1_official_6to7/global_step_150 n=10 val_size=128 dataset=eval_in_distribution cuda=6,7 Sun Jun 28 08:46:08 UTC 2026`
- Train command: `TAG=official_6to7 WM_DUMP_ROLLOUTS=1 CUDA_VISIBLE_DEVICES='<free_2gpu_pair>' bash /root/grpo/run_seed_alfworld_official.sh 1`

### grpo_baseline_s2
- Tag: `official_s2`
- Eval target checkpoint: `/mnt/cephfs_home_tianming.sha/grpo_alfworld/checkpoints/grpo_qwen2.5_1.5b_alfworld_seed2_official_s2/global_step_150`
- Eval result: `/mnt/cephfs_home_tianming.sha/grpo_alfworld/logs/eval10x_seed2_results.txt`
- Eval checkpoint: `/mnt/cephfs_home_tianming.sha/grpo_alfworld/checkpoints/grpo_qwen2.5_1.5b_alfworld_seed2_official_s2/global_step_150`
- Latest checkpoint backup: `/mnt/cephfs_home_tianming.sha/grpo_alfworld/checkpoints_backup/grpo_qwen2.5_1.5b_alfworld_seed2_official_s2/global_step_150`
- Diagnostic transitions: `/mnt/cephfs_home_tianming.sha/grpo_alfworld/logs/world_model_rollouts/grpo_qwen2.5_1.5b_alfworld_seed2_official_s2/150.wm_transitions.jsonl`
- Diagnostic checkpoint root: `/mnt/cephfs_home_tianming.sha/grpo_alfworld/checkpoints/grpo_qwen2.5_1.5b_alfworld_seed2_official_s2`
- Train log: `/mnt/cephfs_home_tianming.sha/grpo_alfworld/logs/grpo_qwen2.5_1.5b_alfworld_seed2_gpudev_2gpu_6to7_20260625_090648.log;/mnt/cephfs_home_tianming.sha/grpo_alfworld/logs/grpo_qwen2.5_1.5b_alfworld_seed2_gpudev_2gpu_6to7_20260625_103118.log;/mnt/cephfs_home_tianming.sha/grpo_alfworld/logs/grpo_qwen2.5_1.5b_alfworld_seed2_official_s2_20260628_055217.log`
- Eval line: `EVAL10X_START label=seed2 ckpt=/mnt/cephfs_home_tianming.sha/grpo_alfworld/checkpoints/grpo_qwen2.5_1.5b_alfworld_seed2_official_s2/global_step_150 n=10 val_size=128 dataset=eval_in_distribution cuda=4,5 Mon Jun 29 12:11:34 UTC 2026`
- Train command: `TAG=official_s2 WM_DUMP_ROLLOUTS=1 CUDA_VISIBLE_DEVICES='<free_2gpu_pair>' bash /root/grpo/run_seed_alfworld_official.sh 2`

### obs_ce_l0p01_s0
- Tag: `wm_obs_ce_l0p01_s0`
- Latest checkpoint: `/mnt/cephfs_home_tianming.sha/grpo_alfworld/checkpoints/grpo_qwen2.5_1.5b_alfworld_seed0_wm_obs_ce_l0p01_s0/global_step_135`
- Latest checkpoint backup: `/mnt/cephfs_home_tianming.sha/grpo_alfworld/checkpoints_backup/grpo_qwen2.5_1.5b_alfworld_seed0_wm_obs_ce_l0p01_s0/global_step_135`
- Train log: `/mnt/cephfs_home_tianming.sha/grpo_alfworld/logs/grpo_qwen2.5_1.5b_alfworld_seed0_wm_obs_ce_l0p01_s0_20260628_214934.log;/mnt/cephfs_home_tianming.sha/grpo_alfworld/logs/wm_obs_ce_l0p01_s0_launch_20260628_214933.log`
- Launch line: `RUN_WM_OBS_CE_SEED seed=0 tag=wm_obs_ce_l0p01_s0 cuda=6,7 lambda_obs=0.01 total_epochs=150 ray_tmp=/root/grpo/ray_tmp_wm_obs_ce_l0p01_s0 rollout_data_dir=/mnt/cephfs_home_tianming.sha/grpo_alfworld/logs/world_model_rollouts/wm_obs_ce_l0p01_s0_seed0 ; RUN_ALFWORLD_OFFICIAL seed=0 tag=wm_obs_ce_l0p01_s0 cuda=6,7 tp=2 micro=16 gmu=0.6 enforce_eager=True free_cache=True use_remove_padding=True ckpt=/mnt/cephfs_home_tianming.sha/grpo_alfworld/checkpoints/grpo_qwen2.5_1.5b_alfworld_seed0_wm_obs_ce_l0p01_s0`
- Train command: `TAG=wm_obs_ce_l0p01_s0 WM_DUMP_ROLLOUTS=1 LAMBDA_OBS=0.01 CUDA_VISIBLE_DEVICES=6,7 bash /root/grpo/run_seed_alfworld_official.sh 0`

### obs_ce_l0p01_s1
- Tag: `wm_obs_ce_l0p01_s1`
- Latest checkpoint: `/mnt/cephfs_home_tianming.sha/grpo_alfworld/checkpoints/grpo_qwen2.5_1.5b_alfworld_seed1_wm_obs_ce_l0p01_s1/global_step_135`
- Latest checkpoint backup: `/mnt/cephfs_home_tianming.sha/grpo_alfworld/checkpoints_backup/grpo_qwen2.5_1.5b_alfworld_seed1_wm_obs_ce_l0p01_s1/global_step_135`
- Train log: `/mnt/cephfs_home_tianming.sha/grpo_alfworld/logs/grpo_qwen2.5_1.5b_alfworld_seed1_wm_obs_ce_l0p01_s1_20260628_214935.log;/mnt/cephfs_home_tianming.sha/grpo_alfworld/logs/wm_obs_ce_l0p01_s1_launch_20260628_214933.log`
- Launch line: `RUN_WM_OBS_CE_SEED seed=1 tag=wm_obs_ce_l0p01_s1 cuda=1,2 lambda_obs=0.01 total_epochs=150 ray_tmp=/root/grpo/ray_tmp_wm_obs_ce_l0p01_s1 rollout_data_dir=/mnt/cephfs_home_tianming.sha/grpo_alfworld/logs/world_model_rollouts/wm_obs_ce_l0p01_s1_seed1 ; RUN_ALFWORLD_OFFICIAL seed=1 tag=wm_obs_ce_l0p01_s1 cuda=1,2 tp=2 micro=16 gmu=0.6 enforce_eager=True free_cache=True use_remove_padding=True ckpt=/mnt/cephfs_home_tianming.sha/grpo_alfworld/checkpoints/grpo_qwen2.5_1.5b_alfworld_seed1_wm_obs_ce_l0p01_s1`
- Train command: `TAG=wm_obs_ce_l0p01_s1 WM_DUMP_ROLLOUTS=1 LAMBDA_OBS=0.01 CUDA_VISIBLE_DEVICES=1,2 bash /root/grpo/run_seed_alfworld_official.sh 1`

### obs_ce_l0p03_s0
- Tag: `wm_obs_ce_l0p03_s0`
- Train command: `TAG=wm_obs_ce_l0p03_s0 WM_DUMP_ROLLOUTS=1 LAMBDA_OBS=0.03 CUDA_VISIBLE_DEVICES='<free_2gpu_pair>' bash /root/grpo/run_seed_alfworld_official.sh 0`

### obs_ce_l0p03_s1
- Tag: `wm_obs_ce_l0p03_s1`
- Train command: `TAG=wm_obs_ce_l0p03_s1 WM_DUMP_ROLLOUTS=1 LAMBDA_OBS=0.03 CUDA_VISIBLE_DEVICES='<free_2gpu_pair>' bash /root/grpo/run_seed_alfworld_official.sh 1`

### obs_ce_l0p05_s0
- Tag: `wm_obs_ce_l0p05_s0`
- Train command: `TAG=wm_obs_ce_l0p05_s0 WM_DUMP_ROLLOUTS=1 LAMBDA_OBS=0.05 CUDA_VISIBLE_DEVICES='<free_2gpu_pair>' bash /root/grpo/run_seed_alfworld_official.sh 0`

### obs_ce_l0p05_s1
- Tag: `wm_obs_ce_l0p05_s1`
- Train command: `TAG=wm_obs_ce_l0p05_s1 WM_DUMP_ROLLOUTS=1 LAMBDA_OBS=0.05 CUDA_VISIBLE_DEVICES='<free_2gpu_pair>' bash /root/grpo/run_seed_alfworld_official.sh 1`

### latent_s0
- Tag: `wmls_l1`
- Train log: `/mnt/cephfs_home_tianming.sha/grpo_alfworld/logs/grpo_qwen2.5_1.5b_alfworld_seed0_wmls_l1_20260628_224238.log`
- Train command: `TAG=wmls_l1 WM_DUMP_ROLLOUTS=1 CUDA_VISIBLE_DEVICES='<free_2gpu_pair>' bash /root/grpo/run_seed_alfworld_official.sh 0`

### latent_l0p001_s0
- Tag: `wmlat_l0p001_s0`
- Latest checkpoint: `/mnt/cephfs_home_tianming.sha/grpo_alfworld/checkpoints/grpo_qwen2.5_1.5b_alfworld_seed0_wmlat_l0p001_s0/global_step_135`
- Latest checkpoint backup: `/mnt/cephfs_home_tianming.sha/grpo_alfworld/checkpoints_backup/grpo_qwen2.5_1.5b_alfworld_seed0_wmlat_l0p001_s0/global_step_135`
- Train log: `/mnt/cephfs_home_tianming.sha/grpo_alfworld/logs/grpo_qwen2.5_1.5b_alfworld_seed0_wmlat_l0p001_s0_20260628_230641.log;/mnt/cephfs_home_tianming.sha/grpo_alfworld/logs/wmlat_l0p001_s0_launch_20260628_230640.log`
- Launch line: `RUN_WM_LATENT_SEED seed=0 tag=wmlat_l0p001_s0 cuda=0,3 lambda_latent=0.001 total_epochs=150 rollout_data_dir=/mnt/cephfs_home_tianming.sha/grpo_alfworld/logs/world_model_rollouts/wmlat_l0p001_s0_seed0 ; RUN_ALFWORLD_OFFICIAL seed=0 tag=wmlat_l0p001_s0 cuda=0,3 tp=2 micro=16 gmu=0.6 enforce_eager=True free_cache=True use_remove_padding=True ray_tmp=/root/grpo/ray_tmp_wmlat_l0p001_s0 ckpt=/mnt/cephfs_home_tianming.sha/grpo_alfworld/checkpoints/grpo_qwen2.5_1.5b_alfworld_seed0_wmlat_l0p001_s0`
- Train command: `TAG=wmlat_l0p001_s0 WM_DUMP_ROLLOUTS=1 LAMBDA_LATENT=0.001 CUDA_VISIBLE_DEVICES=0,3 bash /root/grpo/run_seed_alfworld_official.sh 0`

### latent_l0p001_s1
- Tag: `wmlat_l0p001_s1`
- Latest checkpoint: `/mnt/cephfs_home_tianming.sha/grpo_alfworld/checkpoints/grpo_qwen2.5_1.5b_alfworld_seed1_wmlat_l0p001_s1/global_step_75`
- Latest checkpoint backup: `/mnt/cephfs_home_tianming.sha/grpo_alfworld/checkpoints_backup/grpo_qwen2.5_1.5b_alfworld_seed1_wmlat_l0p001_s1/global_step_75`
- Train log: `/mnt/cephfs_home_tianming.sha/grpo_alfworld/logs/grpo_qwen2.5_1.5b_alfworld_seed1_wmlat_l0p001_s1_20260629_134056.log;/mnt/cephfs_home_tianming.sha/grpo_alfworld/logs/wmlat_l0p001_s1_launch_20260629_134055.log`
- Launch line: `RUN_WM_LATENT_SEED seed=1 tag=wmlat_l0p001_s1 cuda=4,5 lambda_latent=0.001 total_epochs=150 rollout_data_dir=/mnt/cephfs_home_tianming.sha/grpo_alfworld/logs/world_model_rollouts/wmlat_l0p001_s1_seed1 ; RUN_ALFWORLD_OFFICIAL seed=1 tag=wmlat_l0p001_s1 cuda=4,5 tp=2 micro=16 gmu=0.6 enforce_eager=True free_cache=True use_remove_padding=True ray_tmp=/root/grpo/ray_tmp_wmlat_l0p001_s1 ckpt=/mnt/cephfs_home_tianming.sha/grpo_alfworld/checkpoints/grpo_qwen2.5_1.5b_alfworld_seed1_wmlat_l0p001_s1`
- Train command: `TAG=wmlat_l0p001_s1 WM_DUMP_ROLLOUTS=1 LAMBDA_LATENT=0.001 CUDA_VISIBLE_DEVICES=4,5 bash /root/grpo/run_seed_alfworld_official.sh 1`
