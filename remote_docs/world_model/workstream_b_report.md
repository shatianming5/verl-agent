# Workstream B —— 世界模型指标可分性与趋势（完整协议)

**状态：** 基线 **seed0 = `official_4to5`** 已完成(11 个 checkpoint)。seed1 验证已单独启动。
**本报告取代**之前在 8 条轨迹冒烟集(`wm_valdump_smoke_s0_step150`)上做的全部 B 分析;那份已作废、不得引用。

## 1. 协议与数据溯源

- **Checkpoint（全集,每 15 步):** `init, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150`
  (`init` = 基座 `Qwen2.5-1.5B-Instruct`;其余 = `checkpoints/grpo_qwen2.5_1.5b_alfworld_seed0_official_4to5/global_step_<s>`)。
- **Rollout：** 每个 checkpoint 在 **N_TASKS=2048** 个 TRAIN 游戏上 rollout,每任务 1 条轨迹,采样 `temperature=1.0`(与训练时一致)。
  脚本 `scripts/wm_rollout_trainset_dump.sh`(GMU 0.45),驱动 `bdiag_rollout_all.sh`。
  **基础设施上限已标注：** 每 checkpoint 2048(而非全部约 3553 个 train 游戏)—— 这是算力上限,非全体样本。
- **每条轨迹的标签：** `episode_rewards > 0` ⇒ 成功。train 游戏上的成功率随 checkpoint 单调上升:
  init≈0.07 → 90=0.46 → 120=0.63 → 135=0.70 → **150=0.79**(sanity:基线 GRPO 训练确实有效)。
- **每个 transition 的指标（teacher-forced,原始 —— 无 predictor/projection):** next-obs CE / NLL、perplexity、target-token 置信度、
  action-end↔obs-end 末层 hidden **cosine**(`h_action` 取 `prefix_len-1`,`h_obs` 取 `seq_len-1`)。打分器 `scripts/wm_score_transition_dump.py`。
- **聚合：** 分别在 **episode 级** 和 **transition 级** 报告(见下方对账)。每个 gap 都带 bootstrap 95% 置信区间。
- **代码：** 分析 `scripts/bdiag_analyze.py`(`5ccd0f2`,numpy GMM),probe `scripts/bdiag_hidden_probe.py`(`2681f7d`,sklearn-free)。
- **产物：** `remote_docs/world_model/bdiag_official_4to5/`(图 + `bdiag_stats.csv`、`addon_overlap_control.csv`、`probe.csv`)。

### 对账之前两个 init-cosine 数字
用户指出 `+0.028` 和 `+0.0141` 两个 init cosine gap 都出现过却没对账。它们是**同一 checkpoint、两个聚合级**:
- **episode 级** init action-obs cosine gap = **+0.0280**(对每个 episode 的均值再求均值),CI [−0.0023, +0.0559]。
- **transition 级** init gap = **+0.0141**,CI [+0.0051, +0.0232]。
现在全部按聚合级标注。以下所有数字来自 `bdiag_stats.csv`。

## 2. 各指标的可分性、gap（95% CI)与趋势

`gap = 成功均值 − 失败均值`。`AUC` = 把原始指标当作成功分数时的 per-episode/transition ROC-AUC。`GMM` = 单指标 2 分量一维高斯混合的"簇→标签"精度,对比多数类 **base rate**。

### 2a. Action–obs hidden cosine（C 线的目标指标)—— 不是成功信号
| step | episode gap [CI] | ep AUC | transition gap [CI] | tr AUC |
|---|---|---|---|---|
| init | +0.0280 [−0.002,+0.056] | 0.526 | +0.0141 [+0.005,+0.023] | 0.510 |
| 15 | +0.0544 [+0.032,+0.077] | 0.564 | +0.0506 [+0.044,+0.058] | 0.545 |
| 45 | −0.0069 [−0.037,+0.021] | 0.478 | −0.0068 [−0.016,+0.002] | 0.496 |
| 75 | −0.0436 [−0.063,−0.022] | 0.441 | −0.0465 [−0.052,−0.041] | 0.465 |
| 120 | −0.0245 [−0.045,−0.005] | 0.453 | −0.0309 [−0.036,−0.025] | 0.473 |
| 150 | −0.0056 [−0.028,+0.015] | 0.483 | −0.0072 [−0.013,−0.002] | 0.492 |

- **在任何 checkpoint 都不可分**(AUC ∈ [0.44, 0.56],即 ~随机,两级都是)。
- gap 早期小幅为正(峰值 +0.054 @15),**到 step45 穿零、随后转负**(失败轨迹 cosine *更高*)贯穿中后期,末尾 ≈0。
- **不随训练增长** —— 反而退化。这在每 checkpoint ~2k episode / 45–98k transition 上稳健地证实了之前小样本的提示:action-obs cosine 不追踪成功。

### 2b. Next-obs CE / perplexity —— 均值漂移随训练增长,但方向不对
| step | CE ep gap [CI] | CE ep AUC | ppl ep gap | ppl ep AUC |
|---|---|---|---|---|
| init | +0.0189 [−0.000,+0.039] | 0.550 | +0.090 | 0.555 |
| 60 | +0.0409 [+0.031,+0.051] | 0.618 | +0.176 | 0.623 |
| 90 | +0.0572 [+0.048,+0.066] | 0.665 | +0.256 | 0.673 |
| 135 | +0.1268 [+0.116,+0.138] | 0.771 | +0.659 | 0.774 |
| 150 | +0.2148 [+0.201,+0.229] | **0.836** | +1.240 | 0.830 |

- **可分性随 checkpoint 单调增长：** CE episode AUC **0.55 → 0.836**(transition 0.51 → 0.649)。gap 的 CI 从 step15 起就不含 0。
- **但符号是 `success = 更高` CE**(step150:成功均值 1.678 vs 失败 1.463)。成功 episode 里 next-obs 更"意外"。
- 最简约的解读:**novelty / 任务推进**,而非世界模型技能。成功轨迹推进任务、进入新观测(更高 CE);失败轨迹在重复、可自我预测的观测上打转(更低 CE)。`target_confidence` 佐证:成功 = target-token 置信度略*低*(gap −0.004→−0.010,AUC 0.37–0.48,即低于 0.5)。

### 2c. GMM 表明:是均值漂移,但高度重叠（并非干净的簇)
对每个指标,**GMM 2 分量的簇精度 ≤ 多数类 base rate**(例:CE episode step150 GMM 0.588 vs base 0.788;cosine step120 GMM 0.503 vs base 0.633)。任何单指标上的 2 高斯混合**都无法比"猜多数类"更好地还原成功/失败划分。** 所以即便存在 gap(CE),两类分布也是**均值有漂移但高度重叠、非双峰可分** —— 见直方图(`hist_*.png`)。这直接回答了"真分开还是高度重叠":**高度重叠。**

## 3. 附加分析（在这批完整数据上,而非 8 条集)

- **e1 —— transition 级 NLL 校准：** 即上文的 transition 行;CE gap 真实但在 transition 级偏小(即使 step150 AUC ≤ 0.65),且随训练朝同一(成功更高)方向增长。
- **e2 —— obs 重叠控制（shortcut 检查):** prev↔next obs token 的 Jaccard 在每个 checkpoint 都 **~0.90**(0.9065→0.8913)。回归掉每对的重叠后,cosine gap 几乎不动:`residual_cos_gap ≈ raw_cos_gap`(step75 raw −0.0465 vs residual −0.0442;step150 raw −0.0072 vs residual −0.0061)。所以 cosine 结果**不是** token 重叠假象 —— 这种(不)可分性是真实的。(`addon_overlap_control.csv`。)
- **e3 —— hidden-state 成功线性 probe（5 折 CV ROC-AUC,sklearn-free):**
  | step | obs_hidden AUC | action_hidden AUC | chance |
  |---|---|---|---|
  | init | 0.887 | 0.808 | ~0.49 |
  | 30 | **0.940** | 0.787 | ~0.46 |
  | 45 | 0.932 | 0.857 | ~0.51 |
  | 150 | 0.883 | 0.726 | ~0.49 |

  **轨迹成功可以从策略的 obs-end hidden state 强线性解码(AUC 0.88–0.94 ≫ chance)。** 但它**在 `init` 就已 ~0.89**,并且在 step45 后略有*下降* —— 所以这个丰富的成功信号是**表征固有的,不是 GRPO 造就的。**

## 4. 三个 B 问题的回答
1. **每个 checkpoint 上可分吗?** cosine:**否**(AUC~0.5)。CE/perplexity:**只是均值漂移、高度重叠**(step150 AUC 高达 0.84 但 GMM ≤ base rate)。target_confidence:弱、方向相反。**Hidden-state probe:是,强可分(AUC ~0.88–0.94)** —— 但见第(3)点。
2. **gap = 成功 − 失败(95% CI):** 按 指标/级别/step 列在 `bdiag_stats.csv`(`gap`、`gap_ci_lo/hi`)。
3. **可分性随训练增长吗?** cosine:**否**(持平→退化)。CE/perplexity:**是,增长**(AUC 0.55→0.84)但方向是 novelty(成功=更高 CE)。hidden probe:**否**(init 就高,略降)。

## 5. 对 Workstream A / C 的决策含义
- C 线 no-predictor latent 目标优化的是 **action-obs cosine** —— 在这里它**与成功无关、也不随训练增长**。提高它未必助任务成功;直接看 eval,别把 latent 指标当成功代理。
- 最小化 **next-obs CE**(Workstream A obs-CE 协同训练)把模型推向*更低*的 obs 意外度,但在这批数据里**成功恰恰伴随更高的 obs 意外度(novelty/推进)。** 天真的 CE-min 辅助有奖励"重复、不推进"行为的风险 —— 这是需要标出的真实隐患,也与 obs_ce 变体至今低于基线(0.547 / 0.671 vs **0.7065**)一致。
- 最强的成功信号在 **hidden state** 里、且 init 就存在。*塑造/暴露*这个已有表征的目标,可能比 next-obs 预测目标更有前途。

## 6. 产物（全部在 `remote_docs/world_model/bdiag_official_4to5/`)
- 折线图(成功/失败/全体,随 step):`line_episode_*.png`、`line_transition_*.png`,涵盖 {action_obs_cosine, ce, perplexity, target_confidence_mean}。
- 成功 vs 失败分布直方图:`hist_*.png`(init / 中期 / step150 面板)。
- 数据:`bdiag_stats.csv`(每 step×级别×指标:gap+CI、AUC、GMM)、`addon_overlap_control.csv`(e2)、`probe.csv`(e3)。
- 生成脚本:`scripts/bdiag_analyze.py`(`5ccd0f2`)、`scripts/bdiag_hidden_probe.py`(`2681f7d`)、`scripts/wm_score_transition_dump.py`、`scripts/wm_rollout_trainset_dump.sh`。
- 每 checkpoint 的原始分数:cephfs 上的 `logs/bdiag_rollouts/official_4to5/step*/scores.csv`(+ `*.wm_transitions.jsonl` dumps)。

## 7. 下一步
- **seed1 = `official_6to7`** 验证:同一协议(每 checkpoint 2048 任务,温度 1.0),在把任何趋势定为最终结论前,先验证它在多个 seed 上可复现。
