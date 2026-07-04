# 世界模型协同训练 —— 全项目状态台账

**快照时间：** 2026-07-04 约 08:20 UTC（16:20 北京时间）。本文件在 **gpudev 集群故障期间**编写（见 §5）—— 各数字为故障前最后一次确认的良好状态；cephfs 侧的原始日志当前不可达。

**要超越的基线：** GRPO `eval10x = 0.7065 ± 0.0197`（3 seed 聚合；单 seed seed0=0.729)。**目前所有世界模型变体都低于此值。**

---

## 1. Workstream A —— obs-CE 协同训练（已完成，干净负结果）
辅助损失 = teacher-forced 的 next-obs token 交叉熵,权重 `λ_obs`。已核实**未**被任何 latent 项污染(A 线日志中 latent loss = 0;证据见 `PROVENANCE.md`)。

| run（全名） | λ_obs | eval10x | 相对基线 Δ | 备注 |
|---|---|---|---|---|
| `wm_obs_ce_l0p001_s0` | 0.001 | **0.5470** | −0.16 | |
| `wm_obs_ce_l0p01_s0` | 0.01 | **0.6714** | −0.035 | |
| `wm_obs_ce_l0p05_s1` | 0.05 | 已杀 | — | step10 val 0.18,发散 |

**结论：** λ_obs 越大越差(单调)。obs-CE 协同训练无法超越基线。这与 B 线发现一致——成功轨迹对应*更高*的 obs-CE(novelty),所以把 CE 往下压是反效果的。

## 2. Predictor-latent（已废弃 —— 仅作"对照",不进主线)
按用户指令保留但**不得进入主线结论**。实现使用了 `world_model_predictor`/`latent_predictor`(一个 predictor 头),用户已在设计中移除它。

| run（全名） | λ_latent | eval10x | 备注 |
|---|---|---|---|
| `wmlat_l0p001_s0` | 0.001 | **0.6735 ± 0.0267** | 低于基线;predictor 版 |
| `wmlat_l0p005_s0` | 0.005 | 已杀（T1) | predictor 版 |
| `wmlatct_l0p01_s0` | 0.01 | 已杀（T1) | 偏离计划的 contrastive |

## 3. Workstream C —— no-predictor latent（最终方法）—— 进行中
最终目标（用户确认）：**`L_latent = 1 − cos(h_action, stop_gradient(h_obs))`**,**无 predictor / projection head**,obs 端 stop-gradient。代码：`verl/workers/actor/dp_actor.py`(`latent_use_predictor=False` 默认,commit `cf5610f`/`51c40b1`)。所有 run 通过 `latent_pred_norm == latent_action_norm`(精确相等)在线核实 no-predictor 生效。

| run（全名） | λ_latent | seed | GPU | 最后已知 step | 最后已知 train-val | eval10x |
|---|---|---|---|---|---|---|
| `wmlatnp_l0p001_s0` | 0.001 | 0 | 2,3 | 113/150 | 0.469（峰值 0.539） | 待定（未到 150） |
| `wmlatnp_l0p001_s1` | 0.001 | 1 | 0,1 | 121/150 | **0.641**（领先） | 待定 |
| `wmlatnp_l0p005_s0` | 0.005 | 0 | 4,5 | 102/150 | 0.602 | 待定 |
| `wmlatnp_l0p005_s1` | 0.005 | 1 | — | 未启动（延后） | — | — |

- 最后确认时间 **2026-07-03 21:48 UTC（05:48 北京时间）**,距本快照约 10 小时。无 run 到达 `global_step_150`,因此 no-predictor 方法**尚无 eval10x**。
- **注意：** 训练时的 `val/success_rate` 偏乐观且有噪声;与 0.7065 的真正对比是 step150 的 `eval10x`。不要用 train-val 下结论。
- 故障前所有 C run 均无 NaN/OOM。

## 4. Workstream B —— 可分性诊断（seed0 完成,seed1 被中断)
完整协议：11 个 checkpoint(init,15,…,150) × 每个 2048 个 train 任务,温度 1.0,原始指标(无 predictor),bootstrap 置信区间,episode+transition 分开,GMM(2),obs 重叠控制,hidden probe。**取代**之前的 8 条轨迹冒烟分析。

- **seed0 = `official_4to5`：已完成。** 报告 `workstream_b_report.md` + `bdiag_official_4to5/`(12 张图 + `bdiag_stats.csv` + `probe.csv` + `addon_overlap_control.csv`)。已 push `e7ef115`。
  - action-obs **cosine**：不可分(AUC ~0.5 两级),gap 中段翻负,**不随训练增长**;e2 证明这**不是** token 重叠假象(residual ≈ raw gap)。
  - next-obs **CE / perplexity**：均值漂移**随训练增长**(CE episode AUC 0.55→0.836)**但方向是 success = 更高 CE** ⇒ novelty/任务推进,而非预测技能。
  - **GMM(2) 精度 ≤ base rate**(所有指标)⇒ 分布**高度重叠**,并非干净的簇。
  - **hidden-state probe**：success 可强线性解码(obs_hidden AUC **0.88–0.94** ≫ chance)但**在 init 就已很高**、训练后略降 ⇒ 是表征固有的,非 GRPO 造就。
  - 对账了之前的 `+0.028`(episode)与 `+0.0141`(transition)init-cosine 数字——是同一 checkpoint 的两个聚合级。
- **seed1 = `official_6to7`：被故障中断。** rollout 曾到 10/11(`init…120` 已 dump,`step135` 跑中);watcher 在等 score→analyze→probe。cephfs 上：`logs/bdiag_rollouts/official_6to7/step{init..120}/*.wm_transitions.jsonl`。恢复后需补 step135+150 再跑 score/analyze/probe 链。

## 5. 故障（基础设施层 —— 非本项目 job 造成)
- **我的外网：正常**(github.com ping ~95 ms)。
- **gpudev（10.100.2.64:24187)：不可用** —— 症状波动:TCP 超时 / ssh banner-exchange 卡住 / "connection closed by remote"。典型的主机崩溃 / 重启 / 存储卡死特征。
- **gpudev2（10.100.2.40)：同样 banner-exchange 卡住** ⇒ 集群/子网/共享存储层面的问题,非 gpudev 单机、非 job 造成。(cephfs 挂起会同时卡住 ssh 登录——home 在 cephfs——和训练 checkpoint 写入。)
- 客户端侧无法修复;等集群/存储恢复。**建议询问集群管理员 gpudev / cephfs 是否宕机或在维护。**

## 6. 数据安全
- ✅ **在 GitHub(`shatianming5/verl-agent`,`master` 分支):** 全部代码(dp_actor latent objective、B 流水线脚本)、B seed0 报告 + 图表 + CSV、`PROVENANCE.md`、进展报告。**与集群状态无关,安全。**
- ⏳ **在 cephfs(共享盘,当前不可达):** 原始训练日志、所有 checkpoint(每 15 step 存)、rollout dumps、每 checkpoint 的 `scores.csv`。**若 cephfs 恢复后完好,则无任何损失** —— 只丢在途算力,可从最近的 15-step checkpoint 续跑。

## 7. 恢复 runbook（gpudev 回来后）
1. **核验 cephfs 完整性：** `ls` 各 C run(`wmlatnp_l0p001_s0/s1`、`wmlatnp_l0p005_s0`)与 seed1 rollout dumps 的 checkpoint;确认每个 C run 最后的 `global_step_*`。
2. **从最近 checkpoint 续跑死掉的 C run**(不要从头重启):用 `resume_from_path=<最新 global_step>` 重启各 `wmlatnp_*`;保持 no-predictor 门控(`+trainer.latent_use_predictor=False`)。
3. **完成 B seed1：** 续跑 `bdiag_rollout_all.sh EXP=official_6to7 SEED=1`(跳过已 dump 的 10 个,补 step135+150),再跑 score→analyze→probe watcher;对账 seed0 vs seed1 趋势。
4. **首个到 `global_step_150` 的 C run 做 eval10x**(在其释放的卡上跑)→ 写结果(标 no-predictor、对比基线 0.7065)→ push。随后若有空 pair 则补 `wmlatnp_l0p005_s1`。
5. 汇报纪律不变：用全名、标 commit+路径、明说每次启动/杀进程、勿动 rlvr、只按 PID 杀。

## 8. Commit 轨迹（GitHub）
`e7ef115` B 完整协议报告 · `5ccd0f2` sklearn-free GMM · `2681f7d` sklearn-free probe · `4a646e7` e3 probe · `3a76bb0` B 流水线 · `6552a72` provenance+WS-A 无污染 · `51c40b1` contrastive latent · `cf5610f` no-predictor objective + B 深挖。
