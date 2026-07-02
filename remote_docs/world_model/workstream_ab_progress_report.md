# Workstream A/B 进展报告

- 时间戳：2026-07-02 17:15 CST
- 分支：`world-model-latent-objective`
- 结果快照：从 gpudev 镜像过来的 `remote_docs/world_model/world_model_results.{md,csv}`
- 实时状态来源：2026-07-02 对 gpudev eval10x 结果文件、launcher 进程和 GPU 状态做的只读检查

## 范围

本报告按下面的口径重新整理：

- Workstream A：ALFWorld GRPO 上的 observation-prediction 辅助损失，也就是 `obs_ce`。当前核心问题是 `lambda_obs` 的方向和强度。
- Workstream B：只分析 GRPO baseline 训练过程中 world-model 相关 state/representation 的变化，包括 baseline checkpoint 上的 token CE、action-observation cosine，以及成功/失败轨迹分离。B 不应把 Workstream A 的 obs-CE ablation 结果当作主分析对象。
- latent hidden-state alignment 在代码里单独作为 Workstream C 跟踪；本报告只在它影响共享运行状态时提到。

## 执行摘要

Workstream A 已经实现，并且现在有两个完整的 obs-CE 条件：`lambda_obs=0.01` 和 `lambda_obs=0.001`。两者都不是稳定提升。`lambda_obs=0.01` 两个 seed 的 eval10x 均值是 `0.6714`，低于当前镜像结果里按 3 个 baseline seed 聚合出来的 eval10x baseline `0.7065`，delta 为 `-0.0351`。`lambda_obs=0.001` 两个 seed 的 eval10x 聚合均值是 `0.5470`，delta 为 `-0.1595`。这里的 `0.7065` 是 seed0/1/2 的 eval10x 均值，不是单个 seed 或 online validation 数字；如果和之前报备的 baseline 对比，必须先确认用的是同一个口径。

因为 `0.001` 也明显负向，下一步 sweep 不应继续往更大的 `0.03`、`0.05` 推。`0.03` 和 `0.05` 已经有部分早期训练/排队痕迹，但它们不再应被视为 evidence-driven 的下一步主线。当前更合理的 A 方向是停止把 obs-CE lambda sweep 当作主线，转为分析为什么 auxiliary head 学到 next-observation 目标后没有转化成任务成功率。

Workstream B 目前只有 baseline seed0 的 checkpoint/state 诊断完成。这个诊断显示，在 vanilla GRPO baseline 上，从 init 到 step 150，token CE 从 `1.3793` 变到 `1.4330`，delta `+0.0537`；action-observation cosine 从 `0.6144` 变到 `0.6901`，delta `+0.0757`。也就是说，baseline 训练过程中 action-observation 表征相似度增强，但 next-observation token CE 并没有变好。baseline seed1/seed2 已 ready for diagnostic，但之前诊断进程卡在 Ceph D-state。

基础设施状态已经改善：Workstream A 的 `lambda_obs=0.001` 训练和 eval10x 已通过 gpudev 本地盘路径完成，不再被 Ceph 阻塞。Workstream B 的 seed1/seed2 baseline diagnostics 仍应避免依赖 Ceph repo/model/data 路径，优先使用本地 staging 后的 model/repo/data。

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

目前完整的 obs-CE 条件包括 `lambda_obs=0.01` 和 `lambda_obs=0.001`：

| run | seed | eval10x 均值 +/- 标准差 | 训练步数 | 最后可用指标 | 诊断状态 |
| --- | ---: | ---: | ---: | --- | --- |
| `obs_ce_l0p01_s0` | 0 | `0.7311 +/- 0.0244` | 150 | `obs_ce_loss=0.161` | 已诊断 |
| `obs_ce_l0p01_s1` | 1 | `0.6118 +/- 0.0324` | 150 | `obs_ce_loss=0.147` | 已诊断 |
| `lambda_obs=0.01` 聚合 | 0,1 | `0.6714` | 已完成 |  | 相对 3-seed eval10x baseline delta `-0.0351` |
| `obs_ce_l0p001_s0` | 0 | `0.4869 +/- 0.0188` | 150 | online val success `0.594` | 待诊断 |
| `obs_ce_l0p001_s1` | 1 | `0.6071 +/- 0.0309` | 150 | online val success `0.617` | 待诊断 |
| `lambda_obs=0.001` 聚合 | 0,1 | `0.5470 +/- 0.0665` | 已完成 | 20 次 eval 合并 | 相对 3-seed eval10x baseline delta `-0.1595` |

解读：`lambda_obs=0.01` 不是一个可靠提升；它可以和一个较强 seed（`s0`）共存，但第二个 seed 下滑太多，使整体均值变成负向。`lambda_obs=0.001` 没有缓解这个问题，反而更差，尤其 seed0 的 10x 均值只有 `0.4869`。由于每个已完成 seed 都已经做过 eval10x，剩余不确定性主要来自不同 seed 和不同 objective 设计，而不是单次 eval launch 内部的随机性。

### Sweep 状态更新

因为 `0.001` 已经完成且偏负，继续沿当前 obs-CE 设计做更大 lambda sweep 的优先级很低：

| 条件 | seeds | 当前状态 | 优先级 | 理由 |
| --- | --- | --- | --- | --- |
| `lambda_obs=0.001` | 0,1 | 训练完成，eval10x 完成，聚合 `0.5470 +/- 0.0665` | 已完成 | 更弱 obs-CE regularization 没有避免性能下降 |
| `lambda_obs=0.03` | 0,1 | 早期训练到 step 15-21，未 eval | 低 | 已经不是基于当前结果的合理下一步，只能作为后续非单调性检查 |
| `lambda_obs=0.05` | 0,1 | seed0 早期训练，seed1 排队卡住 | 低 | 同上，且当前排队进程还卡在 Ceph 数据预处理 |

`lambda_obs=0.001` 的两条训练和 eval10x 均通过 gpudev 本地盘路径完成，主要路径为 `/root/grpo/local_alfworld`，prepared data 使用 `/root/data/verl-agent_wm_obs_ce_l0p001_s0` 和 `/root/data/verl-agent_wm_obs_ce_l0p001_s1`。本次 eval10x 没有留下 launcher 子进程。

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
- 如果继续从 Ceph 上读取 data/model/repo，诊断仍可能不可靠；后续应优先使用本地 staging 后的 model/repo/data 路径。

## 当前运行状态

`lambda_obs=0.001` 的 eval10x 已全部结束：

| run | eval10x 完成度 | mean | std | 状态 |
| --- | ---: | ---: | ---: | --- |
| `wm_obs_ce_l0p001_s0` | 10/10 | 0.4869 | 0.0188 | done |
| `wm_obs_ce_l0p001_s1` | 10/10 | 0.6071 | 0.0309 | done |
| 合并 20 次 eval | 20/20 | 0.5470 | 0.0665 | done |

本地磁盘 runner 已经实际用于绕开 Ceph 路径问题：`scripts/gpudev_run_world_model_local.sh` 和 `scripts/eval10x_alfworld_local.sh` 已同步到 gpudev，训练/eval 使用 `/root/grpo/local_alfworld`、`/root/grpo/models/Qwen2.5-1.5B-Instruct` 和 `/root/data/...`。截至 2026-07-02 17:15 CST，两条 eval10x launcher 都已退出，无残留子进程。

## 判断

Workstream A 在技术上已经跑通，但当前 `lambda_obs=0.01` 和 `lambda_obs=0.001` 的科学结论都是负向。`0.001` 的下降幅度更大，因此继续把 `0.03/0.05` 作为主线没有充分依据。

Workstream B 应回到 baseline state trajectory：先补齐 GRPO baseline seed1/seed2 的 checkpoint diagnostics，再判断 vanilla GRPO 是否系统性改变 world-model 相关 state。当前 seed0 结果显示 cosine 增强，但 token CE 不稳定，且成功/失败分离信号不能靠单 seed 下结论。

## 建议下一步

1. 对 Workstream A，暂时停止把 obs-CE lambda sweep 作为主线；`0.01` 和 `0.001` 已经足够支持“当前设计负向”的结论。
2. 如果仍要继续 A，应先做机制分析：为什么 obs-CE head 的 next-observation CE 能下降，但策略成功率下降；再考虑更改作用位置、annealing、loss normalization 或只在特定 token/span 上施加辅助目标。
3. 暂时不要把 `obs_ce_l0p03_s0/s1`、`obs_ce_l0p05_s0/s1` 作为主线结论来源；它们最多保留为后续非单调性检查。
4. 对 Workstream B，优先补跑 `grpo_baseline_s1/s2` 的 checkpoint/state diagnostics，使用本地 model/repo/data 路径规避 Ceph metadata 卡顿。
5. 后续报告中明确区分三类数字：per-seed eval10x、multi-seed aggregate eval10x、online validation last/best。

## 2026-07-02 诊断深挖 + Workstream C 目标定型

### action-observation cosine 是怎么算的（确认口径）
`scripts/wm_score_transition_dump.py:596-633`：取 **最后一层** hidden（`hidden_states[-1]`），
`action_hidden` = action-turn 结束位置（`prefix_len-1`），`obs_hidden` = next-obs 结束位置（`seq_len-1`），
直接 `cos(action_hidden, obs_hidden)`。所有 checkpoint 用**同一批固定 transition**；报告里的
`row_mean_action_obs_cosine` 是对所有 transition 的简单平均。**baseline 无 saved predictor →
`action_obs_cosine == raw_action_obs_cosine`，即纯 raw hidden cosine，未经任何 predictor/projection。**

### 成功/失败可分性（seed0 baseline，per-trajectory）
诊断固定集来自 `wm_valdump_smoke_s0_step150`，实际只有 **8 条轨迹（3 成功 / 5 失败）**；
"277 transition" 是因为失败轨迹跑满 50 步、成功轨迹 ~9 步。**样本量过小，不足以支撑可分性结论。**
per-episode 平均 cosine（成功−失败 gap）随训练：init +0.0284 → 30 +0.0086 → 60 +0.0124 →
90 +0.0072 → 120 +0.0070 → 150 +0.0062（gap 很小且**随训练收缩**）。step150 成功={0.683,0.693}
被失败={0.641,0.685,0.692,0.692,0.698}完全覆盖 → **重叠、不可分**。GMM(2) 在 step150 cosine 上
最优 cluster→label acc≈base rate → 分不开。cosine 随训练升高是**成功与失败几乎同步**
（init→150：all +0.076 / success +0.062 / failure +0.077），**不是成功轨迹驱动**，是全局效应。
其他指标（next-obs CE、target-confidence、hidden norm）在此样本上同样不可分（step150 CE gap +0.21
由 1-2 条轨迹 + step150 CE 异常驱动，不可信）。**结论：需在大 dump（`world_model_rollouts/*/1.jsonl`
约 6144 行 / 数百 episode）上重跑 B 诊断，才能对可分性下结论。**

### Workstream C latent objective 定型：无 predictor 直接版
按用户确认，C 线 latent loss 改为直接版，不经过任何 predictor / projection：
```
L_latent = 1 - cosine(h_action, stop_gradient(h_obs))
```
- 实现：`verl/workers/actor/dp_actor.py`。默认 `latent_use_predictor=False` → 不建 predictor，
  直接对 action-end hidden 与 **stop-gradient 的** obs-end hidden 做 cosine；梯度只经 action 侧回传共享 Transformer。
- posterior(obs 端)保持 `.detach()` 不变。已单测验证：grad@obs=0、grad@action>0。
- collapse 监控：训练日志的 `world_model/latent_cosine`、`latent_action_norm`、`latent_obs_norm`、
  `latent_action_feature_var`、`latent_obs_feature_var`（若 cosine 过快饱和到 1 或 feature_var→0 即预警）。

### ⚠️ 代码分叉（需处理）
- git 交付分支 `world-model-latent-objective` 的 latent 实现用 `world_model_predictor`（已提交的重构）。
- **实际在跑的 cephfs `$WORK/verl-agent`（分支 `wm-cotrain-goal-rd`）用 `self.latent_predictor` + feature_var，是未提交的本地改动。**
- 即：已产出的 λ=0.001 / λ=0.005 结果来自 cephfs 的 predictor 版；**pushed 仓库不复现这些 run**。
- 无 predictor 改动已落到 git 交付分支;要让**未来的 run** 真正无 predictor，需同步到 cephfs 运行副本(未做,待定)。
