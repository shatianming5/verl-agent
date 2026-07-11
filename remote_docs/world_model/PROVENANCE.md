# World-Model 结果溯源表 (2026-07-03)

约定:C 线用 `λ_latent`,A 线用 `λ_obs`,永不混称。所有 run 用完整 tag。

## 代码版本(全部快照在 rescue 分支)
运行副本 = cephfs `wm-cotrain-goal-rd`(未提交改动),已原样冻结为:
**`shatianming5/verl-agent` 分支 `rescue/cephfs-running-copy-20260703` @ `ca85772`**(verbatim,无清理,含 3 个 .bak)。

| 代码版本 | 文件(在 ca85772 内) | latent 结构 | 说明 |
|---|---|---|---|
| **V1 predictor** | `verl/workers/actor/dp_actor.py.bak_nopred_20260702_151924` | `self.latent_predictor(action_hidden)` | 原始未提交版,产出**所有** obs-CE 结果 + predictor-latent 结果 |
| **V2 no-predictor** | `dp_actor.py.bak_contrastive_20260702_155323` | `pred_hidden=action_hidden`(bypass) | 产出 `wmlatnp_l0p005_s0` |
| **V3 no-pred+contrastive** | `dp_actor.py`(HEAD) | centered InfoNCE(flag `latent_contrastive`) | 产出 `wmlatct_l0p01_s0`(已 kill,无结果) |

> git 交付分支 `world-model-latent-objective`(`cf5610f` no-pred / `51c40b1` contrastive)是 **另一套结构**(`world_model_predictor`),**从未实际运行**;跑出结果的是上面 cephfs 的 V1/V2/V3。这是已知的 git↔cephfs 分叉。

## 结果溯源

| workstream | run tag | 代码 | latent | λ | predictor | 关键 artifacts |
|---|---|---|---|---|---|---|
| A obs-CE | `wm_obs_ce_l0p01_s0` / `_s1` | V1 | **OFF(0)** | λ_obs=0.01 | — | `logs/eval10x_wm_obs_ce_l0p01_s{0,1}_results.txt` |
| A obs-CE | `wm_obs_ce_l0p001_s0` / `_s1` | V1 | **OFF(0)** | λ_obs=0.001 | — | `local_alfworld/logs/eval10x_wm_obs_ce_l0p001_s{0,1}_results.txt` |
| A obs-CE | `wm_obs_ce_l0p03_s0`,`l0p05_s0/s1` | V1 | **OFF(0)** | λ_obs=0.03/0.05 | — | 部分/已 kill |
| B 诊断 | `wm_ckpt_diag_seed0_official_full_20260628` | 诊断脚本(raw cosine,无 predictor) | — | — | — | `remote_docs/world_model/world_model_diagnostics/...` **⚠️作废(仅8轨迹)** |
| C latent | `wmlat_l0p001_s0` / `_s1` | V1 | ON | λ_latent=0.001 | **启用** | `checkpoints/...wmlat_l0p001_s{0,1}/`;`eval10x_wmlat_l0p001_s0=0.6735±0.0267` |
| C latent | `wmlat_l0p005_s0` | V1 | ON | λ_latent=0.005 | **启用** | 已 kill,部分 checkpoint |
| C latent | `wmlatnp_l0p005_s0` | V2 | ON | λ_latent=0.005 | 停用 | cephfs 历史部分 run；gpudev 当前不可达 |
| C latent | `wmlatct_l0p01_s0` | V3 | ON | λ_latent=0.01,τ0.1 | 停用+contrastive | 已 kill(~step2,无结果) |

**标注**:所有 V1 predictor 版 C 结果(`wmlat_l0p001_s0/s1`、`wmlat_l0p005_s0`)= **"predictor 版,仅作对照",不进入主线结论**。

## Workstream A 未被 latent 污染 — 日志证据
每个 obs-CE run 训练日志里 `world_model/latent*` 计数为 0,`world_model/obs_ce_loss` 为 150(每 step 一条):

```
wm_obs_ce_l0p01_s0    : latent=0  obs_ce=150  (actor/wm_obs_ce_loss:1.272 - actor/wm_obs_ce_coef:0.010)
wm_obs_ce_l0p01_s1    : latent=0  obs_ce=150  (actor/wm_obs_ce_loss:1.265 - actor/wm_obs_ce_coef:0.010)
wm_obs_ce_l0p001_s0   : latent=0  obs_ce=150  (log _20260701_101628)
wm_obs_ce_l0p001_s1   : latent=0  obs_ce=150  (log _20260701_101628)
wm_obs_ce_l0p03_s0    : latent=0
wm_obs_ce_l0p05_s0    : latent=0
wm_obs_ce_l0p05_s1    : latent=0
```
结论:A 的 obs-CE 结果干净,latent loss 项不存在。

## 2026-07-12 更新：predictor 对照与 no-predictor 主线

### Predictor 版只作对照

权威机器可读清单为
`remote_docs/world_model/PREDICTOR_ONLY_ARTIFACTS.csv`。`wmlat_l0p001_s0`、
`wmlat_l0p001_s1`、`wmlat_l0p005_s0` 均来自 rescue 快照 `ca85772` 中的 V1
predictor 实现，不得进入主线结论。原始 gpudev/cephfs 当前不可达，未删除或改写任何
原始 artifact；可访问的镜像目录另放置 `PREDICTOR_ONLY.md` 标记。

### `.136` no-predictor 重跑

`.136` 运行副本的 `dp_actor.py` blob
`5a495ea15b1a30b28c12e41e598913c8bd05b4bb` 与已推送
`5bbbbb986be3a068ed572066976dbcf8c5c22306` 完全一致；runner 工作树 blob
`1cfd06b179b6e54c143e3b62a81811e1769a8454` 也与该提交一致。训练日志 resolved
config 明确记录 `latent_use_predictor: False`。

| workstream | 完整 run 名 | 训练代码 | λ | predictor | 状态与数字 | artifacts |
|---|---|---|---|---|---|---|
| C latent | `wmlatnp_l0p001_s0` | `f809747c22fab41109bf54dc5d14d95c8c3e2922` + 上述两个 worktree blob（等价于 `5bbbbb9` 的训练实现） | λ_latent=0.001 | OFF | 150/150；最终 val success rate=0.750 | `/mnt/SSD_8TB/zechuan/grpo_alfworld_wm/logs/full_wmlatnp_l0p001_s0.log`；`/mnt/SSD_8TB/zechuan/grpo_alfworld_wm/checkpoints/grpo_qwen2.5_1.5b_alfworld_seed0_wmlatnp_l0p001_s0/global_step_150` |
| C latent | `wmlatnp_l0p001_s1` | 同上 | λ_latent=0.001 | OFF | 150/150；最终 val success rate=0.640625 | `/mnt/SSD_8TB/zechuan/grpo_alfworld_wm/logs/full_wmlatnp_l0p001_s1.log`；`/mnt/SSD_8TB/zechuan/grpo_alfworld_wm/checkpoints/grpo_qwen2.5_1.5b_alfworld_seed1_wmlatnp_l0p001_s1/global_step_150` |
| C latent | `wmlatnp_l0p005_s0` | `208560695c0dbc39d76bcff57a4c94720a1cd34d` | λ_latent=0.005 | OFF（强制 guard） | 2026-07-12 07:32 CST 从零重启；此前被主管叫停的无 checkpoint 部分日志已保留 | `/mnt/SSD_8TB/zechuan/grpo_alfworld_wm/logs/full_wmlatnp_l0p005_s0.log`；停止副本 `full_wmlatnp_l0p005_s0.stopped_by_supervisor_20260712_071414_nopredictor.log`；checkpoint root `/mnt/SSD_8TB/zechuan/grpo_alfworld_wm/checkpoints/grpo_qwen2.5_1.5b_alfworld_seed0_wmlatnp_l0p005_s0` |
| C latent | `wmlatnp_l0p005_s1` | `208560695c0dbc39d76bcff57a4c94720a1cd34d` | λ_latent=0.005 | OFF（强制 guard） | 串行排队在 `wmlatnp_l0p005_s0` 后 | checkpoint root `/mnt/SSD_8TB/zechuan/grpo_alfworld_wm/checkpoints/grpo_qwen2.5_1.5b_alfworld_seed1_wmlatnp_l0p005_s1` |

上述四个 run 的 exact command、checkpoint root、eval10x result 路径由
`/mnt/SSD_8TB/zechuan/grpo_alfworld_wm/logs/c_nopredictor_runs_2085606.tsv`
与 `c_nopredictor_seq_136.log` 记录。eval10x 统一由代码
`208560695c0dbc39d76bcff57a4c94720a1cd34d` 执行。

### Workstream B 旧冒烟诊断作废

`wm_ckpt_diag_seed0_official_full_20260628` 仅来自
`wm_valdump_smoke_s0_step150` 的 8 条 episode（3 success / 5 failure），不得再作为
可分性、gap 或训练趋势证据。镜像目录内的 `DEPRECATED.md` 是强制弃用标记；新版 B
必须使用全训练任务 manifest、每个 checkpoint 自己的 rollout、全 11 个 checkpoint
及至少两个训练 seed。
