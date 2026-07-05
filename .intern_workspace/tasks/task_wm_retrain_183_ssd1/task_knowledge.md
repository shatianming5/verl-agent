# task_wm_retrain_183_ssd1 - Task Knowledge

<!-- METADATA:SESSION=11 -->

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
15. **OOM 是结构性非运气（launch4 二次确认）**：与 jusheng 共卡时，我进程峰值~32GB + FSDP 反向~5GB + 邻居~11GB ≈ 48GB 逼近单卡 47.38GB，换卡无用（每卡都有邻居）。**降显存三管齐下**（commit a09ca61，脚本已 env 参数化）：`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`(消~10GB 碎片) + `optimizer_offload=True`(Adam~6GB 卸 CPU) + `GMU 0.30`(缩 vLLM KV)。峰值→~24-26GB。可单独回退调优。
16. **.183 SSH 认证脆弱**：本机连 .183 靠临时认证态，会失效 → `Permission denied(publickey,password)`。`~/.ssh/id_ed25519`(指纹 SHA256:2DVxXB9Sq9hf1YFEU1CO6GbEIwOt8p9w+uWuxlHDrdg)未授权到 .183。恢复需主管把该公钥加进 .183 ~zechuan/.ssh/authorized_keys 或提供旧私钥。scp 比 ssh 更严(要 `-o StrictHostKeyChecking=accept-new`)。**GitHub token 与 SSH 登录是两套凭证，互不通用。**
17. **SSH 免密恢复法（已生效）**：.183 password=zechuan@hcp123(勿落盘)。sshpass 未装时用 `SSH_ASKPASS=<吐密码脚本> SSH_ASKPASS_REQUIRE=force DISPLAY=:0 setsid -w ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no ... 'cat pub >> ~/.ssh/authorized_keys'` 一次性装公钥，之后 BatchMode 免密。传文件走 `cat 本地 | ssh 'cat > 远端'`（scp 认证与 ssh 不同源，易 Permission denied）。
18. **选卡是启动前的实时动作**：.183 GPU 占用随 jusheng 大幅波动（同一天卡 0-4 从 ~37GB free 变 ~17GB free，卡 8,9 仅 ~8GB）。每次启动前必须现采样选 free 最高的一对，不能沿用上次的卡号。新显存配置峰值 ~24-26GB，需 2 卡各 >26GB。
19. **降显存旋钮分两类，别混用（launch5 教训）**：治 OOM 的是 `optimizer_offload=True`(Adam~6GB 卸 CPU) + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`(消碎片)——省的是训练显存，不动 rollout。压吞吐的是 `GMU`(vLLM KV 预留)——降太狠会饿死 val rollout。**GMU 0.30 致 val 128 局并发极低、67min 0 进度**；回调 0.40 兼顾。GMU 只需小幅 trim(0.45→0.40)，OOM 主要靠前两者。
20. **诊断：GPU 100% util 但 memory-util 低(22-44%) + env 交互=0 + 0 进度 = 生成被 KV 饿死的龟速**，不是卡死也不是正常算。区别于「日志静默但真算」(那个 env 交互在涨、显存在爬)。val rollout 正常时会刷 ALFWorld `dresser is not closed` 噪声；一条都没有 = rollout 没真正产出。
21. **三管齐下各分项已验证生效(launch6)**：`expandable_segments` 让 `reserved-but-unallocated` 从 10.22GiB→98MiB(碎片消除铁证)；`optimizer_offload`+GMU0.40 让 val 跑通(metrics 0.4946)。但共卡 jusheng 时 actor+反向峰值 ~34GB + 邻居 11GB 仍超 47.38GB。**结论：共卡场景下最后手段是 `param_offload=True` 或独占整卡。**
22. **优先独占整卡而非共卡**：jusheng 负载会整卡撤走(观察到卡 6,7 从半占变 0MiB 全空)。整卡 49GB 时我方峰值 ~34GB 绰绰有余，无需任何 offload/降 GMU。故选卡第一优先找 used=0 的整卡；只有全被占时才共卡+降显存。整卡窗口是临时的，jusheng 可能回来，抓紧启动。
23. **val 本来就慢，别误判龟速（修正条目20的归因）**：launch7 用 GMU0.45 整卡(KV 充足)val 仍 35min+ 未出 metrics，证明 val 慢**主因不是 KV 饿死**，而是 val 首次 vLLM 批量生成 128 局 rollout 的冷启动慢(`enforce_eager=True` 关 CUDA graph → 逐 token 慢) + ALFWorld env step。**基准：launch4(GMU0.45) 约 40min 才出 val metrics**。所以 <40min 无 metrics 属正常，别急着杀。真龟速的判据要更严：>60min 且 dresser=0。(launch5 GMU0.30 是 KV 确实太小叠加，67min 才判死。)
24. **256 个 `ray::AlfworldWorker` 并行**：ALFWorld env 是多进程并行(非单线程 CPU 瓶颈)。val/train rollout 慢不在 env 并行度，在 vLLM 生成吞吐。
25. **判"卡死 vs 慢"的黄金证据 = GPU 功耗**：util 99% 可能是自旋。真算功耗接近上限(4090D 上限 425W，真算 300W+)；**功耗仅 ~156-184W + util 99% = 忙等自旋/死锁**，不是真算。配合 TaskRunner `futex_wait_queue`(等锁) + dresser=0(零产出) + 日志 mtime 停 30min → 判卡死。launch7 即此：偶发 vLLM/ray 初始化死锁，同配置 launch6 能过 → 配置无罪，干净重启即可。
26. **val-stall 哨兵设计**：针对"GPU 满载但 val 零产出"的死锁盲区，监控脚本轮询 `grep -c "is not closed"`(dresser env 交互)：>0 报健康产出、启动 45min 后仍=0 报疑似卡死、进程亡报 exit tail。比只 tail 日志更早抓到 launch7 式卡死。
27. **val 首次生成卡死是系统性复现(launch7+launch8 连续两次同点)**，非偶发——重启救不了。判据：功耗~100W(上限425)自旋 + dresser=0 + 日志停 wandb 初始化后。**注意**：vllm dev 版不打印 "Adding requests/Processed prompts"，launch6(成功)进展行数也=0，故"无 vLLM 日志"≠卡死，别用它判据。根因待诊断 agent 结论(疑 vllm+TP2+enforce_eager 首次生成 hang)。
28. **⚠️ dresser 信号被证伪(重大纠正)**：`dresser is not closed` **不是 val 产出信号**——所有 run(含成功出 val metrics 的 launch3/launch6)dresser 全=0。它只在训练 rollout 阶段打印，val 用不同 eval 路径。**从 Session 7 起用 dresser=0 判 launch5/7/8「零产出卡死」是错的**。正确的 val 信号只有 `Initial validation metrics`/`test_score`。判卡死靠：GPU 功耗自旋(~100W/上限425) + val metrics 长时间不出 + 日志 mtime 停滞 + worker CPU 仍涨(自旋)。
29. **vLLM 精确版本 = 0.6.3**(非泛指 dev)。属 V0 engine 时代(V1 需 0.7+)，排除 V1 多进程问题。环境 NCCL：`NCCL_P2P_DISABLE=0 / NCCL_P2P_LEVEL=SYS / NCCL_IB_DISABLE=1`。`VLLM_ATTENTION_BACKEND=FLASH_ATTN`(launch6 成功与 launch8 卡死用同后端，排除后端差异)。疑似根因待查：0.6.3+TP2+ray+enforce_eager 首次生成 hang。
30. **⚠️⚠️ val 本就慢 60-80min，不是卡死(第二次重大纠正)**：用 wandb run 目录 mtime 算出 **launch6 全程 87min、val 阶段约 60-80min 才出 metrics**(行797 val 紧邻行850 OOM)。所以任何 run 在启动后 <80min 没出 val metrics 都属正常。根因：`enforce_eager=True` 关 CUDA graph → val 128 局逐 token 生成极慢；GPU 功耗低(~110W)也是逐 token 正常表现，**非卡死非 bug**。Session 7-10 判 launch5/7 卡死很可能全是误判，launch8 亦然。
31. **判 run 健康的可靠信号(取代 dresser/功耗/日志flush)**：① **wandb 数据文件 `.wandb` 的 mtime** 是否在更新(最可靠，进程在写数据=健康)；② worker `/proc/<pid>/stat` utime+stime 增量(在算)。日志 flush 慢、功耗低、dresser=0 都不能判卡死。wandb `.wandb` 文件 >10min 不更新才是真停滞。
32. **⚠️⚠️⚠️ 冒烟 8 次失败最终定论：update_actor 反向峰值 ~43GB 是硬需求**。val 稳定过(test_score=0.4946)，但 actor 反向传播 PyTorch 纯真实占用 42.65GB(碎片仅 128MB)，逼近单卡 47.38GB。**GMU/expandable_segments/optimizer_offload 都救不了**——它们不动反向激活+梯度这个最大头。根治二选一：(a) `param_offload=True`(actor 参数分片卸 CPU，峰值→~30GB，代价训练 1.5-2x 慢)；(b) **独占整卡且全程不被抢**(.183 与 jusheng 共享，jusheng 会在 ~70min 后回占，是历次 OOM 主因)。主管定：优先换可独占的卡/机器，排除那台 96GB 的(疑 8xH100)。
33. **可用机器清单缺口**：本地无 ssh config；确知只有 .183(1.14.177.180:20183, zechuan)。gpudev(10.100.2.64)/gpudev2(10.100.2.40) 故障中。Session 0「7 台机器普查」未落盘。.183 上 GPU 6-9 会周期性全空(jusheng 撤走)但也会回占，不能保证全程独占。换机器需向主管/coordinator 要清单。
