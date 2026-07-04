# task_wm_retrain_183_ssd1 - Task Knowledge

<!-- METADATA:SESSION=5 -->

## Knowledge Entries

0. **gdpo env 补齐清单**（.183，纯增量 pip，dry-run 均确认不动 torch/vllm/transformers）：`torchdata alfworld cachetools gymnasium peft`。补齐后 main_ppo+peft+alfworld env 全链 import OK。
1. **HF 不可达**：huggingface.co timeout；`hf-mirror.com` 可达。prepare.py 的 geometry3k 只用于指示 modality+size、内容不用 → text parquet 可直接按 schema 生成绕过 HF。
2. **ALFWorld 数据下载**：源=github releases（可达）；`alfworld-download` 默认 temp=/tmp/alfworld 常被他人占用 → 必须 `export TMPDIR=<可写路径>`。

1. **.183 落盘限制**：`/mnt/SSD1_8TB` 根目录属 root，zechuan 账号只能写 `/mnt/SSD1_8TB/zechuan/` 子目录。
2. **可复用 env**：`gdpo`(verl+flash_attn2.8.3+vllm dev, torch2.4/cu121)、`goodhart`(verl)、`autorl`(alfworld+vllm0.18)。用 `PYTHONPATH=$REPO` 覆盖为本仓库 verl。
3. **已缓存资产**：`/mnt/SSD1_8TB/zechuan/models/Qwen2.5-1.5B-Instruct`、`/mnt/SSD1_8TB/zechuan/.cache/alfworld`。
4. **占卡邻居**：jusheng 的 12 个 `run.py --method latent_mas`（Qwen3-4B/DeepSeek-R1-8B），负载轻但占着全部 10 卡的一半显存；需按空闲显存错卡（每卡 ~37GB 空的：1,3,4,5,7,8,9）。
5. **runner 模板**：`scripts/gpudev_run_world_model_local.sh`（原 /root/grpo/* 路径）；关键超参 TRAIN_DATA_SIZE=16, VAL=128, GROUP_SIZE=8, N_GPUS=2, TP=2, GMU=0.6, TOTAL_EPOCHS=150, SAVE_FREQ=15, TEST_FREQ=5。
6. **no-predictor 门控**：latent run 必须 `latent_use_predictor=False`，在线核实 `pred_norm==act_norm` 精确相等。
7. **gpudev/cephfs**：故障中，断点不可达；恢复后原机 resume 才是最省路径，本任务是备用重训路线。
8. **OOM 教训（launch3）**：冒烟必须错开 jusheng 占卡；jusheng 20GB spiker 会瞬时把某卡 free 从 ~37GB 压到 ~16GB，正好撞 FSDP actor-update 峰值 → CUDA OOM。选卡看**瞬时最坏**而非当前 free；卡 6 有常驻 spiker，避开；4,5 稳。降显存后备手段：param_offload/optimizer_offload=True 或 GMU 0.45→0.3。
9. **checkpoint 保留机制（verl）**：`trainer.max_actor_ckpt_to_keep=N` 是滚动窗口，verl 自动删更旧的（ray_trainer.py:1050-1053）；当前 full 配置 =null=永不删。本实验 `adv_estimator=grpo` **无 critic**，`max_critic_ckpt_to_keep` 无关。
10. **ckpt 内容/体积**：默认 `checkpoint.contents=['model','optimizer','extra']` 分片存（省空间，非 HF 全量）；加 `hf_model` 可另存 HF 便携格式（更占空间，仅评测需要时开）。base 权重 5.8G；分片 ckpt 实测体积待 smoke step3 落盘确认。SSD1 现 2.4T 空闲。
11. **权重管理方案（待主管定稿，full 前置）**：(a) `max_actor_ckpt_to_keep=3` 只留最近 3 个滚动；(b) 每次落 ckpt 后 rsync 到异地（我本地 or 另一台机）做备份，破单盘单点；(c) critic 知识无关不用配。理由：本任务正因 cephfs 单点丢权重而生，full=null+单盘=同构复现风险。
12. **权重管理 diff 已备**（本地分支 `intern_123/wm183-weight-mgmt`，基于任务分支）：脚本加 `MAX_ACTOR_CKPT_TO_KEEP`(默认3)+`CKPT_BACKUP_DEST`(默认空/关)，`bash -n` 过。**未开 PR/未 merge**，等冒烟绿 + 主管批准。
13. **备份设计未决点（主管拍板）**：现 diff 的 rsync 是**跑完后**镜像，只保完成态，**不防跑中掉盘**（正是 cephfs 威胁模型）。真正抗单点需**跑中逐 ckpt 备份**（后台 watcher 盯 `global_step_*` 落盘即 rsync，或 verl callback）+ 备份目的地（我本地/另一台机）。此为架构选择，待主管定。
14. **诊断法：日志静默 ≠ 卡死**。verl 按 step 边界攒着 flush，val 之后到首个 step 落地可长时间无新日志。判"真算 vs 死锁"硬证据：对 GPU worker pid 读 `/proc/<pid>/stat` 第 14+15 字段(utime+stime)间隔 ~12s 双采样，增长≈满核 tick 即真算；配合 GPU util 持续高、进程状态 `R`。别只看日志 mtime 下结论。
