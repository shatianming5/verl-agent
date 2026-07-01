# Workstream A/B 进展报告

- 时间戳：2026-07-01 13:31 CST
- 分支：`world-model-latent-objective`
- 结果快照：从 gpudev 镜像过来的 `remote_docs/world_model/world_model_results.{md,csv}`
- 实时状态来源：2026-07-01 对 gpudev 进程和 GPU 状态做的只读检查

## 范围

本报告沿用当前脚本和镜像 artifacts 里的命名。

- Workstream A：ALFWorld GRPO 上的 observation-prediction 辅助损失，也就是 `obs_ce`。
- Workstream B：checkpoint/transition 诊断和 world-model 特征分析，包括 rollout dump、checkpoint 打分、成功/失败轨迹分离指标，以及报告聚合。
- latent hidden-state alignment 在代码里单独作为 Workstream C 跟踪；本报告只在它影响共享运行状态时提到。

## 执行摘要

Workstream A 已经实现，并且有一个 lambda 设置完整跑完。完整的 `lambda_obs=0.01` 条件已经完成 2/2 个 seed 的 eval10x，但均值低于 GRPO baseline：`0.6714`，baseline 为 `0.7065`，delta 为 `-0.0351`。Seed 0 表现有竞争力（`0.7311`），但 seed 1 表现较差（`0.6118`），所以当前证据不支持把 `lambda_obs=0.01` 视为一个稳定提升。

Workstream B 部分完成。诊断流水线对已经完成的 checkpoint 可以正常工作，并且已经为两个 `lambda_obs=0.01` seed 生成了 obs-CE checkpoint 诊断。这些诊断说明 obs-CE head 在狭义上确实学到了 next-observation 目标：相对于初始化，token CE 大约下降了 `-1.16`。但是成功/失败轨迹的分离信号较弱：两个已诊断 obs-CE run 的平均 failure-success CE gap 只有 `+0.0014`，平均 success-failure action-observation cosine gap 为 `+0.0136`。

当前主要阻塞是基础设施，不是 GPU 可用性。剩余排队 run 卡在和 Ceph 相关的 D-state，位置在数据预处理或 metadata 访问阶段。GPU `2,3,4,5` 目前实际上空闲，但由于 Ceph 上的 repo/model/data 路径无法稳定读取，任务还没有进入训练。

## Workstream A：Observation CE 目标

### 已完成证据

Baseline 已完成三个 seed：

| 条件 | 已评测 seed 数 | eval 均值 | run 标准差 |
| --- | ---: | ---: | ---: |
| GRPO baseline | 3/3 | 0.7065 | 0.0197 |

目前唯一完整的 obs-CE 条件是 `lambda_obs=0.01`：

| run | seed | eval10x 均值 +/- 标准差 | 训练步数 | 最后 WM 指标 | 诊断状态 |
| --- | ---: | ---: | ---: | --- | --- |
| `obs_ce_l0p01_s0` | 0 | `0.7311 +/- 0.0244` | 150 | `obs_ce_loss=0.161` | 已诊断 |
| `obs_ce_l0p01_s1` | 1 | `0.6118 +/- 0.0324` | 150 | `obs_ce_loss=0.147` | 已诊断 |
| `lambda_obs=0.01` 聚合 | 0,1 | `0.6714` | 已完成 |  | 相对 baseline delta `-0.0351` |

解读：到目前为止，`lambda_obs=0.01` 不是一个可靠提升。它可以和一个较强 seed（`s0`）共存，但第二个 seed 下滑太多，使整体均值变成负向。由于每个已完成 seed 都已经做过 eval10x，剩余不确定性主要来自不同 seed 和不同 lambda 设置，而不是单次 eval launch 内部的随机性。

### 尚未完成的 sweep

更高的 obs-CE lambda 设置目前还不能评估：

| 条件 | seeds | 最新已知进度 | eval 状态 | 备注 |
| --- | --- | --- | --- | --- |
| `lambda_obs=0.03` | 0,1 | step 15 到 21 区间 | 等待 checkpoint | 看到早期训练，但没有最终 checkpoint/eval |
| `lambda_obs=0.05` | 0 | step 15 | 等待 checkpoint | 看到早期训练，但没有最终 checkpoint/eval |
| `lambda_obs=0.05` | 1 | 已排队，镜像快照里没有完成的 train log | 实时状态卡在数据预处理 | 当前排队 wrapper 尚未进入 PPO 训练 |

2026-07-01 的实时检查显示，排队的 `wm_obs_ce_l0p05_s1` 进程卡在：

```text
/root/grpo/venv/bin/python -m examples.data_preprocess.prepare --mode text --train_data_size 16 --val_data_size 128 --local_dir /root/data/verl-agent_wm_obs_ce_l0p05_s1
```

其 D-state wait channel 是 `d_alloc_parallel`，和更广泛的 Ceph metadata 卡顿一致。

## Workstream B：诊断和特征证据

### 已实现部分

下面这些诊断/报告组件已经实现，并且反映在镜像 artifact 集合里：

- 通过 `WM_DUMP_ROLLOUTS=1` dump rollout transition；
- 对 `init 30 60 90 120 150` 做 checkpoint 诊断；
- token 级 observation CE 打分；
- action-observation cosine 打分；
- 成功/失败轨迹分离摘要；
- artifact 镜像到 `remote_docs/world_model`；
- 聚合结果表和 readiness 跟踪。

### 当前诊断覆盖率

在完整的 GOAL_RD 跟踪集合上，镜像报告显示：

| 类别 | 完成情况 |
| --- | ---: |
| eval 结果 | 6/11 tracked runs |
| 训练日志 | 10/11 tracked runs |
| checkpoint 诊断 | 4/11 tracked runs |
| obs-CE 诊断 | 2/6 obs-CE runs |

具体到 Workstream A，两个已完成的 `lambda_obs=0.01` seed 都已有诊断：

| run | final diag CE | 相对 init 的 delta CE | final cosine | delta cosine | CE gap | cosine gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `obs_ce_l0p01_s0` | 0.2732 | -1.1620 | 0.1509 | -0.0177 | +0.0007 | +0.0028 |
| `obs_ce_l0p01_s1` | 0.2688 | -1.1601 | 0.1626 | -0.0495 | +0.0021 | +0.0244 |

解读：

- 辅助 head 确实在学习 observation target：两个 seed 的 CE 都相对于初始化大幅下降。
- 特征层面的信号方向合理但幅度很小：失败轨迹的 CE 略高于成功轨迹，而成功轨迹的 action-observation cosine 略高。
- 这些诊断指标上的改善目前还没有转化为 `lambda_obs=0.01` 在下游 ALFWorld 成功率上的提升。

### 诊断缺口

剩余诊断缺口主要来自缺失或无法访问的 checkpoint：

- obs-CE `lambda_obs=0.03` 和 `0.05` 还在等待可用的最终 checkpoint。
- baseline seed 1 和 seed 2 已经可以做诊断，但之前的诊断进程卡在 Ceph D-state。
- 当前 live jobs 无法推进到新 checkpoint，因为 Ceph 上的 data/model/repo 读取不可靠。

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

Workstream A 在技术上已经跑通，但科学结论目前偏“不确定到负向”。已完成的 `lambda_obs=0.01` 证据不支持直接使用这个设置。更高 lambda 设置仍然重要，因为第一个完整设置可能权重过低或权重不合适；但在它们跑到最终 checkpoint 并完成 eval10x 之前，无法判断效果。

Workstream B 是有价值的，并且已经抓住了关键区别：模型可以提升 observation CE，但不一定提升任务成功率。因此 B 不只是报告附属项，而是一个解释层。下一步诊断价值主要来自对比已完成的更高 lambda obs-CE checkpoint，并补上 pending 的 baseline 诊断；不过这一步目前被存储问题阻塞。

## 建议下一步

1. 恢复可靠的 model/repo 路径：要么修复 Ceph，要么把 `Qwen2.5-1.5B-Instruct` staging 到 gpudev 的 `/root/grpo/models` 下。
2. 重启或继续 `obs_ce_l0p03_s0/s1`、`obs_ce_l0p05_s0/s1` 到 step 150。
3. 每个完成的 obs-CE checkpoint 都先跑 eval10x，再解释 lambda 效果。
4. 对所有完成的 obs-CE checkpoint 和 pending 的 baseline seed 1/2 跑 Workstream B 诊断。
5. 除非更多 seed 反转当前均值，否则应把 `lambda_obs=0.01` 视为负向结果，不要把它表述成提升。
