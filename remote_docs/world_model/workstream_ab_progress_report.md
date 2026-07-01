# Workstream A/B Progress Report

- Timestamp: 2026-07-01 13:31 CST
- Branch: `world-model-latent-objective`
- Artifact snapshot: `remote_docs/world_model/world_model_results.{md,csv}` mirrored from gpudev
- Live status source: read-only gpudev process/GPU checks on 2026-07-01

## Scope

This report uses the naming present in the current scripts and mirrored artifacts.

- Workstream A: observation-prediction auxiliary loss (`obs_ce`) on ALFWorld GRPO.
- Workstream B: checkpoint/transition diagnostics and world-model feature analysis, including rollout dumps, checkpoint scoring, success/failure separation metrics, and report aggregation.
- Latent hidden-state alignment is tracked separately in the code as Workstream C; it is mentioned only where it affects shared operational status.

## Executive Summary

Workstream A is implemented and has one complete lambda setting. The complete `lambda_obs=0.01` condition has 2/2 seeds evaluated with eval10x, but the mean is below the GRPO baseline: `0.6714` versus baseline `0.7065`, delta `-0.0351`. Seed 0 is competitive (`0.7311`), while seed 1 is poor (`0.6118`), so the current evidence does not support `lambda_obs=0.01` as a reliable improvement.

Workstream B is partially complete. The diagnostic pipeline is working for completed checkpoints and has already produced obs-CE checkpoint diagnostics for both `lambda_obs=0.01` seeds. These diagnostics show the obs-CE head is learning the next-observation target in the narrow sense: token CE drops by about `-1.16` versus initialization. However, the success/failure separation signal is weak: the mean failure-success CE gap is only `+0.0014`, and the mean success-failure action-observation cosine gap is `+0.0136` across the two diagnosed obs-CE runs.

The main current blocker is infrastructure, not GPU availability. The queued remaining runs are stuck in Ceph-related D-state during data preprocessing or metadata access. GPUs `2,3,4,5` are currently effectively idle, but the jobs have not reached training because the repo/model/data paths under Ceph are not reliably readable.

## Workstream A: Observation CE Objective

### Completed Evidence

Baseline is complete for three seeds:

| condition | evaluated seeds | mean eval | run std |
| --- | ---: | ---: | ---: |
| GRPO baseline | 3/3 | 0.7065 | 0.0197 |

The only complete obs-CE condition is `lambda_obs=0.01`:

| run | seed | eval10x mean +/- std | train step | last WM metric | diagnostic |
| --- | ---: | ---: | ---: | --- | --- |
| `obs_ce_l0p01_s0` | 0 | `0.7311 +/- 0.0244` | 150 | `obs_ce_loss=0.161` | diagnosed |
| `obs_ce_l0p01_s1` | 1 | `0.6118 +/- 0.0324` | 150 | `obs_ce_loss=0.147` | diagnosed |
| `lambda_obs=0.01` aggregate | 0,1 | `0.6714` | complete |  | delta vs baseline `-0.0351` |

Interpretation: `lambda_obs=0.01` is not a reliable win so far. It can coexist with a strong seed (`s0`), but the second seed drops enough to make the mean negative. Since each finished seed already has eval10x, the remaining uncertainty is mostly across seeds and lambda settings, not within a single eval launch.

### Incomplete Sweep

The higher obs-CE lambda settings are not yet evaluable:

| condition | seeds | latest known progress | eval status | notes |
| --- | --- | --- | --- | --- |
| `lambda_obs=0.03` | 0,1 | step 15 to 21 range | waiting for checkpoint | early training seen, no final checkpoint/eval |
| `lambda_obs=0.05` | 0 | step 15 | waiting for checkpoint | early training seen, no final checkpoint/eval |
| `lambda_obs=0.05` | 1 | queued, no completed train log in mirrored snapshot | blocked live in data preprocessing | current queued wrapper has not reached PPO training |

Live check on 2026-07-01 shows the queued `wm_obs_ce_l0p05_s1` process is stuck in:

```text
/root/grpo/venv/bin/python -m examples.data_preprocess.prepare --mode text --train_data_size 16 --val_data_size 128 --local_dir /root/data/verl-agent_wm_obs_ce_l0p05_s1
```

with D-state wait channel `d_alloc_parallel`, consistent with the broader Ceph metadata stall.

## Workstream B: Diagnostics and Feature Evidence

### Implemented Pieces

The following diagnostic/reporting pieces are implemented and represented in the mirrored artifact set:

- rollout transition dumps via `WM_DUMP_ROLLOUTS=1`;
- checkpoint diagnostics over `init 30 60 90 120 150`;
- token-level observation CE scoring;
- action-observation cosine scoring;
- success/failure separation summaries;
- artifact mirroring into `remote_docs/world_model`;
- aggregate result tables and readiness tracking.

### Current Diagnostic Coverage

Across the full GOAL_RD tracked set, the mirrored report shows:

| category | complete |
| --- | ---: |
| eval results | 6/11 tracked runs |
| training logs | 10/11 tracked runs |
| checkpoint diagnostics | 4/11 tracked runs |
| obs-CE diagnostics | 2/6 obs-CE runs |

For Workstream A specifically, both completed `lambda_obs=0.01` seeds have diagnostics:

| run | final diag CE | delta CE vs init | final cosine | delta cosine | CE gap | cosine gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `obs_ce_l0p01_s0` | 0.2732 | -1.1620 | 0.1509 | -0.0177 | +0.0007 | +0.0028 |
| `obs_ce_l0p01_s1` | 0.2688 | -1.1601 | 0.1626 | -0.0495 | +0.0021 | +0.0244 |

Interpretation:

- The auxiliary head is learning the observation target: CE falls sharply from initialization for both seeds.
- The feature-level signal is directionally sensible but small: failure trajectories have slightly higher CE than success trajectories, and success trajectories have slightly higher action-observation cosine.
- These diagnostic improvements do not currently translate into better downstream ALFWorld success for `lambda_obs=0.01`.

### Diagnostic Gaps

Remaining diagnostic gaps are mostly downstream of missing or inaccessible checkpoints:

- obs-CE `lambda_obs=0.03` and `0.05` are waiting for usable final checkpoints.
- baseline seeds 1 and 2 are ready for diagnostics but previous diagnostic processes are stuck in Ceph D-state.
- current live jobs cannot progress to new checkpoints because data/model/repo reads under Ceph are unreliable.

## Current Operational Status

The relaxed GPU queueing change has been pushed and allows using assigned pairs when visible memory and `pmon` compute are clear. The current live GPU state is favorable:

| GPU | memory used MiB | utilization |
| ---: | ---: | ---: |
| 0 | 15615 | 0 |
| 1 | 19245 | 0 |
| 2 | 4 | 0 |
| 3 | 4 | 0 |
| 4 | 4 | 0 |
| 5 | 4 | 0 |
| 6 | 16055 | 0 |
| 7 | 18029 | 0 |

However, the remaining queued jobs are not training:

- `wmlat_l0p001_s1` is stuck in `examples.data_preprocess.prepare` with `ceph_mdsc_wait_request`.
- `wm_obs_ce_l0p05_s1` is stuck in `examples.data_preprocess.prepare` with `d_alloc_parallel`.
- reading Ceph repo/model files such as `verl-agent/pyproject.toml` and `Qwen2.5-1.5B-Instruct/config.json` has timed out in recent probes.

A local-disk runner has been added at `scripts/gpudev_run_world_model_local.sh` and synced to gpudev, but it still needs a local copy of `Qwen2.5-1.5B-Instruct/model.safetensors`. Neither gpudev nor cpudev2 currently has that model in local cache, and direct Hugging Face large-file download from the local machine was too slow/unreliable to use as a workaround.

## Assessment

Workstream A is technically functional but scientifically inconclusive-to-negative so far. The completed `lambda_obs=0.01` evidence argues against using that setting as-is. The stronger lambda settings are still important because the first completed setting may be under- or mis-weighted, but they cannot be judged until they reach final checkpoints and eval10x.

Workstream B is useful and already caught the key distinction: the model can improve observation CE without improving task success. This makes B valuable as an explanatory layer rather than just a reporting add-on. The next diagnostic value will come from comparing completed higher-lambda obs-CE checkpoints and adding the pending baseline diagnostics, but that is blocked by storage.

## Recommended Next Steps

1. Restore a reliable model/repo path, either by fixing Ceph or staging `Qwen2.5-1.5B-Instruct` under `/root/grpo/models` on gpudev.
2. Restart or continue `obs_ce_l0p03_s0/s1`, `obs_ce_l0p05_s0/s1` to step 150.
3. Run eval10x for each completed obs-CE checkpoint before interpreting lambda effects.
4. Run Workstream B diagnostics for all completed obs-CE checkpoints and pending baseline seeds 1/2.
5. Treat `lambda_obs=0.01` as negative unless additional seeds reverse the current mean; do not present it as an improvement.
