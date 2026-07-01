# Workstream A/B 进展报告

- 时间戳：2026-07-01 13:31 CST
- 分支：`world-model-latent-objective`
- 结果快照：从 gpudev 镜像过来的 `remote_docs/world_model/world_model_results.{md,csv}`
- 实时状态来源：2026-07-01 对 gpudev 进程和 GPU 状态做的只读检查

## 范围

本报告按下面的口径重新整理：

- Workstream A：ALFWorld GRPO 上的 observation-prediction 辅助损失，也就是 `obs_ce`。当前核心问题是 `lambda_obs` 的方向和强度。
- Workstream B：只分析 GRPO baseline 训练过程中 world-model 相关 state/representation 的变化，包括 baseline checkpoint 上的 token CE、action-observation cosine，以及成功/失败轨迹分离。B 不应把 Workstream A 的 obs-CE ablation 结果当作主分析对象。
- latent hidden-state alignment 在代码里单独作为 Workstream C 跟踪；本报告只在它影响共享运行状态时提到。

## 执行摘要

Workstream A 已经实现，但目前完成的 `lambda_obs=0.01` 不是稳定提升。两个 seed 的 eval10x 均值是 `0.6714`，低于当前镜像结果里按 3 个 baseline seed 聚合出来的 eval10x baseline `0.7065`，delta 为 `-0.0351`。这里的 `0.7065` 是 seed0/1/2 的 eval10x 均值，不是单个 seed 或 online validation 数字；如果和之前报备的 baseline 对比，必须先确认用的是同一个口径。

因为 `0.01` 已经造成负向结果，下一步 sweep 不应继续往更大的 `0.03`、`0.05` 推，而应该优先评测更小的 `lambda_obs=0.001`。`0.03` 和 `0.05` 已经有部分早期训练/排队痕迹，但它们不再应被视为 evidence-driven 的下一步主线。

Workstream B 目前只有 baseline seed0 的 checkpoint/state 诊断完成。这个诊断显示，在 vanilla GRPO baseline 上，从 init 到 step 150，token CE 从 `1.3793` 变到 `1.4330`，delta `+0.0537`；action-observation cosine 从 `0.6144` 变到 `0.6901`，delta `+0.0757`。也就是说，baseline 训练过程中 action-observation 表征相似度增强，但 next-observation token CE 并没有变好。baseline seed1/seed2 已 ready for diagnostic，但之前诊断进程卡在 Ceph D-state。

当前主要阻塞仍是基础设施，不是 GPU 可用性。剩余排队 run 卡在和 Ceph 相关的 D-state，位置在数据预处理或 metadata 访问阶段。GPU `2,3,4,5` 实际上空闲，但由于 Ceph 上的 repo/model/data 路径无法稳定读取，任务还没有进入训练。

## Baseline 口径校正

当前 artifact 里的 baseline 数字如下：

| baseline run | seed | eval10x 均值 +/- 标准差 | online val last | online val best | 备注 |
| --- | ---: | ---: | ---: | ---: | --- |
| `grpo_baseline_s0` / `official_4to5` | 0 | `0.7290 +/- 0.0282` | 0.7810 | 0.7890 | 已诊断 |
| `grpo_baseline_s1` / `official_6to7` | 1 | `0.6984 +/- 0.0375` | 0.7340 | 0.7660 | 等待诊断 |
| `grpo_baseline_s2` / `official_s2` | 2 | `0.6922 +/- 0.0321` | 0.7190 | 0.7190 | 等待诊断 |
| 3-seed eval10x aggregate | 0,1,2 | `0.7065` |  |  | run std `0.0197` |

解读：报告里用于计算 delta 的 `0.7065` 是 3-seed eval10x aggregate。它会低于 seed0 单独的 `0.7290`，也明显低于 online validation 的 `0.7810/0.7890`。后续报告应同时列出 per-seed eval10x 和 aggregate，避免把不同口径的 baseline 混用。

## Workstream A：Observation CE 目标

### 已完成证据

目前唯一完整的 obs-CE 条件是 `lambda_obs=0.01`：

| run | seed | eval10x 均值 +/- 标准差 | 训练步数 | 最后 WM 指标 | 诊断状态 |
| --- | ---: | ---: | ---: | --- | --- |
| `obs_ce_l0p01_s0` | 0 | `0.7311 +/- 0.0244` | 150 | `obs_ce_loss=0.161` | 已诊断 |
| `obs_ce_l0p01_s1` | 1 | `0.6118 +/- 0.0324` | 150 | `obs_ce_loss=0.147` | 已诊断 |
| `lambda_obs=0.01` 聚合 | 0,1 | `0.6714` | 已完成 |  | 相对 3-seed eval10x baseline delta `-0.0351` |

解读：`lambda_obs=0.01` 不是一个可靠提升。它可以和一个较强 seed（`s0`）共存，但第二个 seed 下滑太多，使整体均值变成负向。由于每个已完成 seed 都已经做过 eval10x，剩余不确定性主要来自不同 seed 和不同 lambda 设置，而不是单次 eval launch 内部的随机性。

### 正确的下一步 sweep

因为 `0.01` 已经偏负，下一步应该先降低 auxiliary loss 权重：

| 条件 | seeds | 当前状态 | 优先级 | 理由 |
| --- | --- | --- | --- | --- |
| `lambda_obs=0.001` | 0,1 | 尚未完成训练/eval | 最高 | 检查更弱 obs-CE regularization 是否能避免 `0.01` 的性能下降 |
| `lambda_obs=0.03` | 0,1 | 早期训练到 step 15-21，未 eval | 低 | 已经不是基于当前结果的合理下一步，只能作为后续非单调性检查 |
| `lambda_obs=0.05` | 0,1 | seed0 早期训练，seed1 排队卡住 | 低 | 同上，且当前排队进程还卡在 Ceph 数据预处理 |

2026-07-01 的实时检查显示，排队的 `wm_obs_ce_l0p05_s1` 进程卡在：

```text
/root/grpo/venv/bin/python -m examples.data_preprocess.prepare --mode text --train_data_size 16 --val_data_size 128 --local_dir /root/data/verl-agent_wm_obs_ce_l0p05_s1
```

其 D-state wait channel 是 `d_alloc_parallel`，和更广泛的 Ceph metadata 卡顿一致。这个 job 即使恢复，也不应替代 `lambda_obs=0.001` 作为下一步主评测。

### A 的辅助诊断

obs-CE checkpoint 诊断可以作为 Workstream A 的辅助解释，但不应混入 Workstream B 的主结论。两个 `lambda_obs=0.01` seed 的 obs-CE head 都确实学到了 next-observation 目标：

| run | final diag CE | 相对 init 的 delta CE | final cosine | delta cosine | failure-success CE | success-failure cosine |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `obs_ce_l0p01_s0` | 0.2732 | -1.1620 | 0.1509 | -0.0177 | +0.0007 | +0.0028 |
| `obs_ce_l0p01_s1` | 0.2688 | -1.1601 | 0.1626 | -0.0495 | +0.0021 | +0.0244 |

解读：auxiliary head 的 CE 大幅下降，但这没有转化成 `lambda_obs=0.01` 的任务成功率提升。这说明 A 的问题不是“head 完全没学到”，而是该辅助目标的权重或作用位置可能干扰了策略优化。

## Workstream B：GRPO Baseline State 变化

### 已实现部分

下面这些诊断/报告组件可用于 B：

- 通过 `WM_DUMP_ROLLOUTS=1` dump baseline rollout transition；
- 对 baseline checkpoint 的 `init 30 60 90 120 150` 做诊断；
- token 级 observation CE 打分；
- action-observation cosine 打分；
- 成功/失败轨迹分离摘要；
- artifact 镜像到 `remote_docs/world_model`；
- 聚合结果表和 readiness 跟踪。

### 当前 B 覆盖率

| baseline 项 | 完成情况 |
| --- | ---: |
| baseline eval10x | 3/3 seeds |
| baseline checkpoint/state diagnostics | 1/3 seeds |
| baseline seed1/seed2 diagnostics | ready but blocked/stuck |

当前只有 `grpo_baseline_s0` / `official_4to5` 完成 B 诊断：

| checkpoint | step | CE | delta CE | cosine | delta cosine | failure-success CE | success-failure cosine |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| init | init | 1.3793 | 0.0000 | 0.6144 | 0.0000 | -0.0341 | +0.0141 |
| step30 | 30 | 1.3528 | -0.0265 | 0.6655 | +0.0511 | -0.0277 | +0.0030 |
| step60 | 60 | 1.3421 | -0.0372 | 0.6872 | +0.0728 | -0.0362 | +0.0035 |
| step90 | 90 | 1.3568 | -0.0226 | 0.6754 | +0.0610 | -0.0364 | +0.0022 |
| step120 | 120 | 1.3638 | -0.0155 | 0.6881 | +0.0737 | -0.0475 | +0.0022 |
| step150 | 150 | 1.4330 | +0.0537 | 0.6901 | +0.0757 | -0.2286 | -0.0012 |

解读：

- baseline GRPO 训练会显著提高 action-observation cosine，这说明 state/representation 与动作-观测结构的关系发生了变化。
- token CE 在中间 step 短暂下降，但 step150 反而高于 init；因此不能说 baseline 训练自然提升了 next-observation 预测能力。
- step150 的 failure-success CE 为负，说明在这个 seed 的诊断集上，失败轨迹 CE 没有高于成功轨迹；这个信号需要 seed1/seed2 诊断确认，不应过度解读单 seed。

### B 的诊断缺口

剩余缺口主要来自 baseline seed1/seed2 的诊断：

- `grpo_baseline_s1` 和 `grpo_baseline_s2` 都已有 eval10x 和 checkpoint root，理论上可以跑 B 诊断。
- 之前针对 `official_6to7` 和 `official_s2` 的诊断进程卡在 Ceph D-state。
- 当前 live jobs 也无法推进，因为 Ceph 上的 data/model/repo 读取不可靠。

## 当前运行状态

放宽 GPU 排队限制的改动已经推送；当显存和 `pmon` compute 都显示空闲时，可以使用原本分配给其他任务的 GPU pair。当前实时 GPU 状态是有利的：

| GPU | 已用显存 MiB | 利用率 |
| ---: | ---: | ---: |
| 0 | 15615 | 0 |
| 1 | 19245 | 0 |
| 2 | 4 | 0 |
| 3 | 4 | 0 |
| 4 | 4 | 0 |
| 5 | 4 | 0 |
| 6 | 16055 | 0 |
| 7 | 18029 | 0 |

但是，剩余排队任务并没有真正开始训练：

- `wmlat_l0p001_s1` 卡在 `examples.data_preprocess.prepare`，wait channel 是 `ceph_mdsc_wait_request`。
- `wm_obs_ce_l0p05_s1` 卡在 `examples.data_preprocess.prepare`，wait channel 是 `d_alloc_parallel`。
- 最近探测中，读取 Ceph repo/model 文件会超时，例如 `verl-agent/pyproject.toml` 和 `Qwen2.5-1.5B-Instruct/config.json`。

已经新增了本地磁盘 runner：`scripts/gpudev_run_world_model_local.sh`，并同步到了 gpudev。但它还需要本地有一份 `Qwen2.5-1.5B-Instruct/model.safetensors`。目前 gpudev 和 cpudev2 的本地 cache 里都没有这个模型权重；从本机直接下载 Hugging Face 大文件也太慢且不稳定，暂时不能作为可靠 workaround。

## 判断

Workstream A 在技术上已经跑通，但当前 `lambda_obs=0.01` 的科学结论是负向。下一步应优先做 `lambda_obs=0.001` 的 2-seed eval10x，而不是继续把 `0.03/0.05` 作为主线。

Workstream B 应回到 baseline state trajectory：先补齐 GRPO baseline seed1/seed2 的 checkpoint diagnostics，再判断 vanilla GRPO 是否系统性改变 world-model 相关 state。当前 seed0 结果显示 cosine 增强，但 token CE 不稳定，且成功/失败分离信号不能靠单 seed 下结论。

## 建议下一步

1. 恢复可靠的 model/repo 路径：要么修复 Ceph，要么把 `Qwen2.5-1.5B-Instruct` staging 到 gpudev 的 `/root/grpo/models` 下。
2. 对 Workstream A，优先启动 `obs_ce_l0p001_s0/s1`，跑到 step 150 后做 eval10x。
3. 暂时不要把 `obs_ce_l0p03_s0/s1`、`obs_ce_l0p05_s0/s1` 作为主线结论来源；它们可以保留为后续非单调性检查。
4. 对 Workstream B，优先补跑 `grpo_baseline_s1/s2` 的 checkpoint/state diagnostics。
5. 后续报告中明确区分三类数字：per-seed eval10x、multi-seed aggregate eval10x、online validation last/best。
