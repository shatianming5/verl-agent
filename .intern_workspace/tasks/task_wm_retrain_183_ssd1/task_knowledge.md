# task_wm_retrain_183_ssd1 - Task Knowledge

<!-- METADATA:SESSION=1 -->

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
