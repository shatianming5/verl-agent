# task_wm_retrain_183_ssd1 - History Log

<!-- METADATA:SESSION=2 -->

## Session 2 - 持续监督:冒烟健康跑 val_before_train

- 复核冒烟进度：进程存活 42min，2 个 `ray::WorkerDict`(FSDP actor)~95% CPU + 多个 `ray::AlfworldWo` env worker 在跑；GPU 7,8 util 91-98% 波动、显存 15-33GB 随 rollout/prefill 起落 = 真算非卡死。
- 一度日志 14min 无新增，排查确认是 verl 按 step flush 进度条 + 首次验证 rollout（128 游戏×≤50 步、TP=2 4090D）本就慢，非卡住；已到 `Training Progress: 0/6` + `Initializing AlfredTWEnv`。
- 无报错/OOM/NaN；持续监督 monitor（be05pg3pm）在盯，首个 val/step/checkpoint 或报错即通知。
- 教训修正：上次读 CPU 时间读的是父启动器 PID（19s），真正在算的是 ray worker（各 39min CPU）。

## Session 1 - 冒烟真正跑起来 + 持续监督

- 冒烟第 3 次（peft 补齐后）**越过全部依赖问题**：模型加载成功（Qwen2ForCausalLM 1.54B params）、FSDP wrap、Total steps=6、wandb offline 起、GPU 7,8 各 33GB/100% util 在算，进到 `val_before_train` 首次 rollout。
- `vllm._version` 仅无害 warning，非报错。
- 主管指令"持续监督"：挂 persistent monitor（be05pg3pm）盯到冒烟完成（WM_183_DONE / 存 checkpoint）或报错。
- 依赖补齐总账（gdpo env 纯增量、dry-run 确认不动 torch/vllm）：torchdata、alfworld、cachetools、gymnasium、peft。

## Session 0 - 2026-07-05 创建 + 启动

- 主管指令：gpudev 故障续跑不可行，迁移到 .183 从头重训 world-model，落盘 `/mnt/SSD1_8TB/zechuan`。
- 前置探测：7 台机器 GPU 占用普查，.183 选定（10×4090D，6-7 卡各余 ~37GB，96核/503GB，外网通）。
- 主管拍板：落盘 zechuan 子目录、复用 gdpo env、先冒烟再铺开。
- 复用资产已确认在机：`zechuan/models/Qwen2.5-1.5B-Instruct`、`zechuan/.cache/alfworld`、`gdpo` env（verl+flash_attn2.8.3+vllm）。
- gpudev 仍 UNREACHABLE（多次探测），确认为从头重训而非续跑。

### 环境搭建实录（.183）

- **代码**：clone github.com/shatianming5/verl-agent → `/mnt/SSD1_8TB/zechuan/grpo_alfworld_wm/verl-agent`。
- **环境（复用 gdpo，纯增量补 4 个缺失依赖，全部 dry-run 确认不动 torch/vllm/transformers）**：
  - `torchdata`、`alfworld`（gdpo 原本没有）
  - `cachetools`（main_ppo 缺）
  - `gymnasium`（ray worker 运行期缺，alfworld env wrapper 需要）
  - gdpo 关键版本保持：torch2.4+cu121 / transformers4.47.1 / tensordict0.12.2 / flash_attn2.8.3 / vllm dev —— 与本仓库 pin 有差异但 import 链全通。
- **模型**：复用 `/mnt/SSD1_8TB/zechuan/models/Qwen2.5-1.5B-Instruct`（已缓存，未下载）。
- **ALFWorld 数据**：`.cache/alfworld` 原为空；`alfworld-download` 因 `/tmp/alfworld` 被 jusheng 占用而 PermissionError → 设 `TMPDIR=SSD1/zechuan/.../tmp` 后台下载成功（2.3G，json_2.1.1 全集）。数据源 = github releases（可达）。
- **parquet**：HF 不可达（huggingface.co timeout；hf-mirror.com 可达）。prepare.py 只用 geometry3k 指示 modality+size、内容不用 → 直接按 text schema 生成 train16/test128 parquet，绕过 HF。
- **runner**：`scripts/run_world_model_183.sh`（PR commit f809747），gpudev 脚本的 .183 适配版，no-predictor 门控 `+latent_use_predictor=False`，GMU 0.45。
- **冒烟**：GPU 7,8，TOTAL_EPOCHS=6。第 1 次因缺 gymnasium 失败 → 补装后第 2 次重启，进行中观察。

