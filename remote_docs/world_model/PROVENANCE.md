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
| C latent | `wmlatnp_l0p005_s0` | V2 | ON | λ_latent=0.005 | 停用 | 运行中(GPU4,5) |
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
