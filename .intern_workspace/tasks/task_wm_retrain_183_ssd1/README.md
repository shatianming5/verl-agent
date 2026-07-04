# task_wm_retrain_183_ssd1 - 在 .183 从头重训 world-model 实验

<!-- METADATA:STATUS=InProgress,ASSIGNEE=intern_123 -->

## 背景

gpudev(10.100.2.64) 集群/cephfs 于 2026-07-04 故障,world-model 未完成实验的 checkpoint 断点均在 cephfs 上,当前 UNREACHABLE，无法续跑。主管决定迁移到新 GPU 机器 .183 (`7352-10x4090-183`, 10×RTX4090D-49GB) 从头重训。

## 目标机器 / 落盘

- 机器：`ssh -p 20183 zechuan@1.14.177.180`（`7352-10x4090-183`）
- 落盘根目录：`/mnt/SSD1_8TB/zechuan/grpo_alfworld_wm/`（SSD1 根属 root 不可写，只能写 zechuan 子目录）
- 复用：`/mnt/SSD1_8TB/zechuan/models/Qwen2.5-1.5B-Instruct`、`/mnt/SSD1_8TB/zechuan/.cache/alfworld`
- Python 环境：复用 `gdpo` conda env（torch2.4+cu121, flash_attn 2.8.3, verl, vllm）+ `PYTHONPATH=$REPO` 覆盖为本仓库
- 空闲卡：GPU 1,3,4,5,7,8,9 各余 ~37GB（占卡邻居 jusheng latent_mas 负载轻，需错卡）

## 重训清单（gpudev 断点拿不到，全部从头 TOTAL_EPOCHS=150）

WS-C no-predictor latent（`L_latent = 1 − cos(h_action, sg(h_obs))`，`latent_use_predictor=False`）：
- `wmlatnp_l0p001_s0`（λ=0.001, seed0）
- `wmlatnp_l0p001_s1`（λ=0.001, seed1）
- `wmlatnp_l0p005_s0`（λ=0.005, seed0）
- （延后）`wmlatnp_l0p005_s1`

## 执行策略（主管已拍板 2026-07-05）

1. 落盘 `/mnt/SSD1_8TB/zechuan`，复用现有 models/alfworld 结构
2. 直接复用 `gdpo` env
3. **先跑 1 条冒烟（小 epoch）验证环境/数据/显存/no-predictor 门控全通，再铺开完整 150-epoch 多 run**

## 约束

- 代码改动走 PR，禁止直接 push master
- 冒烟通过后回报主管，等确认再铺开完整 run
- 复用他人共享 env（gdpo）与 zechuan 名下资产，注意不破坏、不冲突
