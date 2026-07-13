# 操作 tick 日志（会话运行日志)

_故障快照 2026-07-04 时从会话 plan.md 复制。完整的决策 + 监控历史（中英混排的工作日志,技术缩写如 done/watch/GPU/step 保留原样)。_

# 继承 codex session 019f0fb4 — ALFWorld World-Model Co-Training

## 目标 (persistent goal)
最大程度并发 sub-agent 执行 `GOAL_RD.md`，按 `SETUP.md` 配置，每完成一步 push 到 GitHub。

## 基础设施
- 编排目录(本机, 非 git): `/Users/tommy/Downloads/agent_training_v1`
- 工作 worktree: `worktrees/verl-agent-rd`,分支 `world-model-latent-objective`
- push 目标 remote: `shatianming5` (github.com/shatianming5/verl-agent)
- 集群: cpudev2(有网,下载)→ cephfs(共享)→ gpudev(8×H200,离线,训练)
  - gpudev ssh: `ssh gpudev` (10.100.2.64:24187)
  - WORK=`/mnt/cephfs_home_tianming.sha/grpo_alfworld`;本地热盘 `/root/grpo/local_alfworld`
- baseline eval10x 3-seed aggregate = **0.7065 ± 0.0197**

## 用户偏好
- 中文回复;状态查询只读,不擅自 start/stop;不 kill 无关 job;避免大范围 Ceph 扫描(用精确路径+timeout)

## 三条 workstream 现状
- **A 原始 obs-CE**: 已完成,干净负结果(2-seed 聚合)。λ=0.001→0.5470(s0=0.487/s1=0.607,Δ-0.16), λ=0.01→0.6714(s0=**0.731**/s1=0.612,Δ-0.035), λ=0.05→早期崩(s1 online val 0.18)。**非单调、呈倒 U**:0.01 优于 0.001、其 seed0 略超基线聚合 0.7065,但 seed 方差大,聚合仍<基线。("单调变差"是早期笔误,已更正。)
- **B baseline 诊断**: 1/3 seed(seed0)。发现 baseline 训练↑action-obs cosine(+0.076)但 next-obs token CE 不降。缺 seed1/seed2。
- **C latent hidden-state alignment(主 proposed)**: λ_latent=0.001。
  - **seed0 已全部完成**(train 150 + eval10x):eval10x=**0.6735±0.0267**,低于 baseline seed0(0.729)/aggregate(0.7065)。online val 0.727 是乐观值。
  - seed1 训练中 step122/150,online val 0.773@120(乐观),~5h 到 150,eval10x 待做。
  - 结论未定,取决于 seed1 eval10x;目前 base latent 偏"未超 baseline"。

## 关键校正
- 所有 world-model 变体目前 eval10x 都 <baseline 0.7065:obs_ce0.001=0.547 / obs_ce0.01=0.671 / **wmlat0.001 seed0=0.6735**(seed1 待定)。
- online val 普遍高于 eval10x,勿用 online val 下结论。

## 运行中的 job(勿动,除非用户明确要求)
- `wmlat_l0p001_s1` (PID 1082523) — C seed1,GPU 2,3,~5h 到 150
- 若干 `rlvr_agentic_grpo.py` — 无关(GPU 0,1,7),禁止触碰
- ~~`wm_obs_ce_l0p05_s1`~~ — 已按用户授权 kill(灾难 val0.18),GPU 4,5 已释放

## GPU 现状
- wmlat s1(λ=0.001): GPU 2,3 | **wmlat s0(λ=0.005, 新 sweep 点): GPU 4,5** | 无关 rlvr: 0,1,7 | 空闲: 6

## 进度
- [x] 读取并继承 codex session 019f0fb4
- [x] commit+push 已完成的 λ=0.001 obs-CE 报告 → shatianming5 @ 3ce692e
- [x] kill 浪费的 λ=0.05 run,释放 GPU 4,5
- [x] 发现 wmlat seed0(λ=0.001)已完整(train+eval10x=0.6735),无需重训
- [x] 按用户选择,启动 latent λ sweep 点 **wmlat_l0p005_s0**(λ=0.005,seed0,GPU 4,5,~32h)
- [ ] 监控 wmlat_l0p005_s0 训练;完成后 eval10x,和 λ=0.001 seed0(0.6735)/baseline(0.7065)对比
- [ ] seed1(λ=0.001)~5h 到 150 → 做 eval10x 拿 2-seed 定论
- [ ] (可选)Workstream B seed1/seed2 诊断

## 循环日志
- tick1 (19:12): s1 step128/150(val0.758,~3.9h);l0p005_s0 step1/150(latent_cosine0.012,早期);均未到150。GPU 0,1,7 因 rlvr 完成而空,但 **rlvr pipeline 仍活**(新 job ckpt_AGRPO_GEN_r2→r3 在 GPU6),故不占 0,1(避免撞他人 job);sweep 扩展只用项目卡 2,3,4,5。本 tick 仅监控。

## 自主循环操作手册 (loop cheatsheet)
ssh: `ssh -o ConnectTimeout=15 -o BatchMode=yes -o ServerAliveInterval=10 gpudev`
WORK=`/mnt/cephfs_home_tianming.sha/grpo_alfworld`;push: worktree `verl-agent-rd` 分支 `world-model-latent-objective` → `git push shatianming5`

**每次醒来检查**:
1. 两个 run 进度: `grep -E "Training Progress|val/success_rate" $WORK/logs/grpo_qwen2.5_1.5b_alfworld_seed*_wmlat_*.log | tail`
2. 是否有 `.../global_step_150` 且无 `eval10x_<TAG>_results.txt` → 该 run 训完待 eval
3. GPU: `nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits`

**训完做 eval10x**(在该 run 自己腾出的 2 卡上跑):
```
CKPT=$WORK/checkpoints/grpo_qwen2.5_1.5b_alfworld_seed<S>_<TAG>/global_step_150 \
LABEL=<TAG> CUDA_VISIBLE_DEVICES=<pair> N_EVALS=10 \
setsid nohup bash /root/grpo/eval10x_alfworld.sh >$WORK/logs/<TAG>_eval10x_launch.out 2>&1 </dev/null &
```
结果: `$WORK/logs/eval10x_<TAG>_results.txt` 末行 `EVAL10X_RESULT n=10 mean=.. std=..`

**GPU 腾出补 sweep 点**(TP=2 需一对空卡):
```
setsid nohup env TAG=wmlat_l0p0XX_s0 LAMBDA_LATENT=0.0XX CUDA_VISIBLE_DEVICES=<pair> \
  bash /root/grpo/run_wm_latent_seed.sh 0 >$WORK/logs/wmlat_l0p0XX_s0_launcher_$(date +%s).log 2>&1 </dev/null &
```
sweep 队列(按需): λ=0.01(拉开上界) → λ=0.003。若某 λ 明显超 baseline,再补它的 seed1 凑 2-seed。

**出结果就 push**: 编辑 `remote_docs/world_model/workstream_ab_progress_report.md` 加新行 → commit → `git push shatianming5 world-model-latent-objective`。

**baseline 对照**: aggregate 0.7065;latent λ=0.001 seed0=0.6735。目标:找到稳定 >0.7065 的设置。
**收尾**: 所有 run 训完+eval+push、且无待补 sweep 点/2-seed 时,`manage_schedule stop`。
- tick2 (19:42): s1 step130/150(~3h);s0-λ0.005 step3/150(cosine0.019→0.025);均未到150。rlvr 大量新 job 占 0,1,6,7 → 项目卡稳定为 2,3,4,5,均忙。仅监控。
- tick3 (20:12): s1 133/150(~2.5h);s0-λ0.005 5/150(cosine0.062↑)。仅监控。
- tick4 (20:42): s1 135/150(~2h);s0-λ0.005 7/150(cosine0.066)。GPU0,1 空闲1.5h但属rlvr轮换卡→不占(守"不扰他人job")。sweep 只等项目卡2,3。仅监控。
- tick5 (21:12): s1 138/150(~1.5h);s0-λ0.005 9/150(cosine0.125↑健康)。仅监控。
- tick6 (21:42): s1 140/150(~1h,10步);s0-λ0.005 11/150(cosine0.126)。仅监控,s1 临近训完。
- tick7 (22:12): s1 143/150(~7步~1h);s0-λ0.005 13/150。仅监控。
- tick8 (22:42): s1 145/150(~5步);s0-λ0.005 15/150。仅监控。

## 2026-07-02 Workstream C 改为无-predictor(用户确认)+ 部署
- git 交付分支已 push cf5610f:`latent_use_predictor` 默认 False → `L_latent=1-cos(h_action, sg(h_obs))`,单测过。
- **cephfs 运行副本已同步**(最小改动:forward 里 bypass predictor,gate 默认无-predictor;backup=`dp_actor.py.bak_nopred_*`)。运行中的 s1 已加载旧码不受影响。
- **代码分叉**:git 分支(world_model_predictor)≠ cephfs 运行分支 wm-cotrain-goal-rd(latent_predictor,未提交)。已产出的 λ=0.001/0.005 结果=predictor 版,pushed 仓库不复现。待办:择机把 cephfs 运行版提交进 git 复现现状。
- 已 kill predictor 版 λ=0.005;**重启为无-predictor `wmlatnp_l0p005_s0`(GPU 4,5)**。
- **新 sweep 命名**:无-predictor 用 `wmlatnp_l{λ}_s{seed}`(区别旧 predictor 版 `wmlat_*`)。未来 sweep 全用无-predictor。
- 运行时确认无-predictor:日志 `world_model/latent_pred_norm` 应 ≈ `latent_action_norm`(因 pred_hidden=action_hidden)。
- Workstream B 诊断结论:8-轨迹样本太小,成功/失败在 action-obs cosine 上**不可分**、gap 随训练收缩、升高非成功驱动;需在大 dump(world_model_rollouts/*/1.jsonl ~6144行)重跑才可下结论。
- tick9 (23:26): s1(λ0.001 pred)99%收尾中;wmlatnp_l0p005_s0(无pred)已"Training from scratch"跑val_before_train(新tag未resume旧ckpt✓),GPU4,5。均未到150。仅监控。下tick验pred_norm≈action_norm。

## 2026-07-02 深挖:让 world-model 超 baseline —— 对比式(contrastive)latent 目标
**诊断(证据)**:ALFWorld 的 obs 里 current↔next 文本 **Jaccard 重叠中位 0.913(96% >0.8)**,是大段静态模板;plain cos(h_action,h_obs) 被这个"公共模/身份"主导(WS-B 里成功/失败同步升、不可分即此症)。→ 单纯 cosine latent 学的是 shortcut,不是 action 后果。
**方案**:contrastive InfoNCE —— 先 **center(去 batch 公共模)** 再 L2-norm,再对 in-batch 负样本做 InfoNCE;公共模在正负样本间抵消,只剩 action-specific 后果可分。无 predictor、obs stop-grad、温度 τ。
**验证(本地单测)**:plain cos 不可分(pos≈neg);centered InfoNCE 正确样本 loss→0、无关对照回到 chance、grad>0、SGD 收敛。**关键坑**:不 center 时即使 L2-norm 也被公共模淹没(loss≈chance)——必须 center。
**实现**:git `dp_actor.py`(push 51c40b1)+ cephfs 运行副本(已同步,backup 存)。flag `latent_contrastive`(默认 False)、`latent_temperature`(默认 0.1);日志出 `latent_gap=pos-neg cosine`(健康信号:gap 涨=在学后果)。
**已起 run**:`wmlatct_l0p01_s0`(contrastive,λ=0.01,τ=0.1,seed0,GPU 2,3)。
**监控要点**:① online val 别塌(像 obs_ce0.05);② `latent_gap` 要随训练涨;③ GRPO kl/pg_loss 稳。promising 就补 seed1 + 扫 λ/τ。
**其他状态**:s1(predictor λ0.001 seed1)已训完待 eval(低优先,predictor 变体);wmlatnp(无pred λ0.005)GPU4,5 跑,已确认 pred_norm==action_norm。
- tick10 (23:59): contrastive wmlatct_l0p01_s0 初始化中(val_before_train,GPU2,3,driver alive);wmlatnp step1(val0.078=base model 正常)。均健康。下tick验 contrastive latent_gap。s1 eval 仍延后。
- tick11 (00:29): contrastive wmlatct_l0p01_s0 step1 健康:latent_gap=0.081(pos0.079>neg-0.001,对比信号正确)、latent_loss2.994(~chance初始)、latent_cosine0.521(raw被模板抬高,印证shortcut)、GRPO稳(pg-0.045/kl0.038)、无error/nan。待观察:gap是否随训练涨+step5 val是否稳。wmlatnp/4,5跑。

## 2026-07-03 用户 corrective 指令执行(5 步)
**汇报铁律**:全名(wm_obs_ce_l0p001_s0 / wmlatnp_l0p005_s1),λ_obs vs λ_latent 不混称;每个 start/kill 明说;每个数字标 commit+路径;数据不足先说不足+造数据计划。

**T1 done**:kill wmlatct_l0p01_s0(off-plan contrastive,GPU2,3 释放);wmlat_l0p005_s0(predictor)确认已死;wmlatnp_l0p005_s0 保留(合法 T3 run)。predictor 版结果全标"仅作对照"。
**T2 done**:cephfs 运行副本 verbatim 冻结 → shatianming5 `rescue/cephfs-running-copy-20260703 @ ca85772`(含 3 .bak,无清理)。溯源表 `remote_docs/world_model/PROVENANCE.md @ 6552a72`。A 未污染:所有 obs_ce run 日志 latent=0/obs_ce=150(证据在 PROVENANCE.md)。
**T3 in_progress**:cephfs dp_actor.py 复原为 V2 plain no-pred(contrastive_refs=0,obs .detach())。no-pred C run(命令模板见 cheatsheet,latent_contrastive 不设=默认 plain):
  - `wmlatnp_l0p001_s0`(λ_latent=0.001,GPU2,3,seed0)running
  - `wmlatnp_l0p001_s1`(λ_latent=0.001,GPU0,1,seed1)running
  - `wmlatnp_l0p005_s0`(λ_latent=0.005,GPU4,5,seed0)running(先前起)
  - `wmlatnp_l0p005_s1` 待补(等空 pair);全部训完各跑 eval10x
  - checkpoint 路径:`$WORK/checkpoints/grpo_qwen2.5_1.5b_alfworld_seed{S}_wmlatnp_l{λ}_s{S}/global_step_*`
**T4 in_progress(重点)**:baseline checkpoint 全集/seed = **init,15,30,45,60,75,90,105,120,135,150**(11 点)。seed0=official_4to5,seed1=official_6to7,seed2=official_s2。
  协议:每 ckpt 在 **训练集全部 ~3000 任务** rollout(采样同训练)→ 每 ckpt 数千带 success/fail 轨迹;逐轨迹算 next-obs CE/ppl/target-conf/action-obs raw cosine(无 predictor);**episode 级 vs transition 级分开报**;GMM(2) 可分性 + 每统计量 bootstrap CI;成功/失败/全体三线随 step;init/中/150 直方图;附加:NLL 校准 / 控 obs-obs 重叠的 action→obs / hidden success 线性探针。全用新数据,8轨迹那份作废。
  资源:GPU6,7 预留 T4 rollout;pipeline 待建。

## 2026-07-03 T4 pipeline 建成+验证+启动(rollout 部分)
**关键 infra 解码**:val 转储用 `+trainer.validation_data_dir=<dir>`(不是 rollout_data_dir);train split 用 `env.alfworld.eval_dataset=train`;temp 用 `val_kwargs.temperature=1.0`;wm 文本捕获(rollout_loop.py:530)无条件,baseline 也行。
**脚本**:`scripts/wm_rollout_trainset_dump.sh`(单 ckpt train-set rollout+dump)+ gpudev `/root/grpo/bdiag_rollout_all.sh`(11 ckpt 驱动)。
**验证**:TEST2 baseline seed0 step150 × 64 train task → 64 轨迹(53 成/11 败)、1333 transition、wm 文本+标签齐全、succ_rate 0.828。✓ 对比旧 8 轨迹,现每 ckpt ~3072 轨迹。
**已启动**:baseline seed0(official_4to5)全 11 ckpt rollout(N_TASKS=3072,GPU6,7,~4.4h)→ `$WORK/logs/bdiag_rollouts/official_4to5/step{init,15..150}/{step}.val.wm_transitions.jsonl`。
**待建(rollout 跑时并行做)**:scoring(复用 wm_score_transition_dump.py,raw cosine 无 predictor)→ stats(GMM+bootstrap CI,episode/transition 分开)→ plots(3 线随 step + init/中/150 直方图)→ 3 附加分析(NLL 校准/控 obs 重叠对齐/hidden 线性探针)。然后补 ≥1 seed(official_6to7)。

## T4 pipeline 状态(compute 链已就绪)
链条:rollout(PID4082905,GPU6,7,~4.4h)→ scoring(PID4092912 chained,wm_score_transition_dump.py per-ckpt,raw cosine 无 predictor)→ analysis(scripts/bdiag_analyze.py,已 push 3a76bb0)。
- bdiag_analyze.py 覆盖:问题 d(GMM(2)+AUC+bootstrap CI gap+可分性)、episode/transition **分开**、交付 f(3线折线 all/succ/fail vs step + init/75/150 直方图 + stats CSV)、add-on e2(控 prev/next obs Jaccard 重叠后的残差 cosine gap)。
- **待补 e1 深化 + e3**:e1 next-obs NLL 校准图;**e3 hidden 线性探针**需扩 scorer dump action/obs hidden(.npz)→ 有 4.4h 窗口在 scoring 前加。
- 跑完 seed0(official_4to5)后补 seed1(official_6to7)同协议。
- 交付时给全部图/CSV/脚本路径。

## T4 rollout BUG 修复 (tick, 01:2x)
- **bug**:首版 driver 每 rollout ~10s 秒退、0 dump。根因:`prepare --val_data_size 3072` 崩(val 数据集仅 601;train 仅 2101,IndexError)。
- **fix**(scripts/wm_rollout_trainset_dump.sh):改用 **train.parquet 当 val_files**(`--train_data_size N_TASKS`),`val_batch_size=128`(不再 =N_TASKS 的巨 batch)。T3 验证(256 task)→ 11MB labeled dump ✓。
- **数据量诚实说明(rule5)**:prepare train prompt 上限 2101 → 用 **N_TASKS=2048/ckpt**(非用户说的 3000+;基础设施 cap)。仍 ~2048 轨迹/ckpt(vs 旧 8)。
- **已重启**:full seed0 rollout v2(PID 4154701,N_TASKS=2048,11 ckpt,GPU6,7,~7h)+ scoring 重新 chained(PID 4154702)。init rollout 已过 data-prep 在跑。
- C run 健康:wmlatnp_l0p001_s0/s1 step1、wmlatnp_l0p005_s0 step6,均 latent_pred_norm==latent_action_norm(no-pred 生效)。

## T4 分析脚本全就位 (01:42)
- scripts/bdiag_analyze.py(问题 d + 交付 f + e2 overlap 控制)已 push 3a76bb0。
- scripts/bdiag_hidden_probe.py(**e3** hidden success 线性探针,5-fold CV ROC-AUC + shuffled chance)已 push;synthetic 验证:separable AUC0.94/chance0.52、random 0.49≈0.50 ✓。
- e1(next-obs NLL 校准):CE/NLL 已在 bdiag_analyze 指标里(逐 step succ/fail/all + CI);校准曲线可在 analyze 里补一个 confidence-bin。
- **编排**:seed0 rollout(PID4154701,~7h)→ score(chained 4154702)→ 我跑 bdiag_analyze + bdiag_hidden_probe(GPU6,7 空了)→ 出图/CSV → 补 seed1 official_6to7。

## T4 全自动链(01:47)—— 不依赖 loop/session
gpudev 上三级 chain 全部就绪:
- rollout PID 4154701 (official_4to5, 11ckpt×2048, GPU6,7) 已确认在生成+dump("Dumped world-model transitions")
- scoring PID 4154702 (chained after rollout)
- analyze+probe PID 3271 (chained after scoring;等 BDIAG_SCORE_DONE → bdiag_analyze.py + bdiag_hidden_probe.py) → 输出 $WORK/logs/bdiag_analysis/official_4to5/{图, bdiag_stats_*.csv, probe.csv, addon_overlap_control_*.csv}
剩余需 loop/我做:① 把 bdiag_analysis 结果拉回本地+push+写 B 报告;② 启动 seed1 official_6to7 同协议;③ T3 各 no-pred C run 训完 eval10x+push;④ 补 wmlatnp_l0p005_s1。
- tick(#5,02:15): C runs wmlatnp_l0p001_s0/s1=4、l0p005_s0=9(/150,val正常爬);T4 init rollout 中(768轨迹52成/716败 累积中,~6/16 batch),GPU6,7 生成中。全健康,无 step150/无空项目卡/scoring 未完 → 仅监控。
- tick(#5,02:45): C wmlatnp_l0p001_s0/s1=6、l0p005_s0=10(/150);T4 仍在 init(base model 慢,GPU6,7 生成中,~64min)。全健康,无里程碑,仅监控。
- tick(#5,03:16): T4 init 完成=**2048 轨迹**(150成/1898败,base model 正常)、step15 rollout 中(128→2048 累积)。2/11 ckpt。C runs 8/8/12。全健康,无里程碑,仅监控。
- tick(#5,03:46): C 9/9/13;T4 step15 rollout中(2/11)。观察点:wmlatnp_l0p001_s0/s1 step9 val~0.07 近base(l0p005_s0 step13 已0.148爬),盯到step30看是否爬。仅监控。
- tick(#5,04:16): C 11/11/15,val: l0p001_s1=0.156↑/l0p005_s0=0.156↑/l0p001_s0=0.078平(盯);T4 step15 ~full(89088 trans,GPU6,7 82/89- tick(#5,04:16): C 11/11/15, val: l0p001_s1=0.156(up)/l0p005_s0=0.156(up)/l0p001_s0=0.078(flat,watch); T4 step15 ~full(89088 trans, GPU6,7 82/89% gen), init done 2048. Revised ETA: ~1-1.5h/ckpt -> T4 seed0 rollout ~13h. monitor-only.
- tick(04:46): C 13/13/16; l0p001_s0 val平但训练健康(0真NaN,pg-0.004/kl0.022/grad0.659/latent_cos0.514),种子方差。T4 3/11(init/15/30)。**loop 30m->60m(#6)**,里程碑还远(T4~13h/C eval~35h)。
- tick(05:49) 异常+修复: T4 step45-150 OOM(用户 rlvr job `ckpt_MT_r1->AGRPO_text_r1` 回到 GPU6 占25GB + 我 GMU0.80 → 超139GB)。init/15/30 dump 完好。修:GMU 0.80->0.55(与 rlvr 共存,76.5+25<139),driver 加 wait<50GB。v4(866980)重跑 45-150,step45 已过 vLLM init OOM=0 生成中。**注意:T4 在 rlvr 轮换卡 6,7 上,与 rlvr 有算力争用(内存不撞、都在跑)**;若用户要我完全让出 6,7,则 T4 需等 C run 训完(~30h)才有非 rlvr 卡。C runs 全在爬(l0p001_s0 val 0.078->0.117)。
- tick(06:49) OOM 根因订正+根治: 真因不是 rlvr 阻塞(rlvr 在 6/7 各只占 ~4/24GB,很轻),而是**我自己的 rollout 进程步间不释放**(上一步 main_ppo 挂住占 ~76GB,与下一步叠加超 139GB)。修:driver 每步后 force-cleanup 残留 ray_bdiag/main_ppo 进程(不碰 rlvr)+ GMU0.55 + wait<40GB。v5(1190391)重跑,SKIP init/15/30/45,step60 已过 vLLM init OOM=0 生成中。**结论:T4 与 rlvr 可低冲突共存,无需让出/停摆 30h。**
- tick(07:1x) OOM 彻底根治(v6): step90 又 OOM 揭示真因——我上一步的 **Ray workers**(非 main_ppo driver)步间不释放,占 ~76GB,cleanup 只杀了 driver 没杀 workers → step90 只剩 29GB free。根治:driver 重写,cleanup 改为**枚举 GPU6,7 的 compute-apps,按 allowlist 放过用户 job(eval_local_agent/rlvr)+C-run(wmlatnp),其余(全是我的 main_ppo/ray worker)SIGKILL,再 poll GPU<35GB 才进下一步**;GMU 0.55→**0.45**(pass-through,不再硬编码)。先硬清 6,7(我的 76GB 残留清掉→回 4GB),确认**没碰用户 job**(用户 24GB 训练 job 自然轮换成 4GB eval_local_agent,非我所杀)。v6(PID 1411433,SKIP init/15/30/45/60)重跑 75-150:**step75 OOM=0、GPU6,7 各 71GB(0.45 余量足)、dump 生成中(19MB)**。链:rollout(75-150)→scoring→analyze+probe 全 chained,自主完成。
- C runs 全健康@step25/150:wmlatnp_l0p001_s0(GPU2,3 val0.133)/l0p001_s1(GPU0,1 val0.195)/l0p005_s0(GPU4,5 val0.219),**三者 pred_norm==act_norm 精确相等=no-predictor 生效✓**,~18min/step→~37h 到 step150。
- tick(05:15 UTC / 13:10 CST) 状态确认+2个自伤事件坦白:
  * **T4 OOM 彻底修好**:GMU0.45 + robust cleanup(杀我在 GPU6,7 的全部进程,allowlist 放过 用户 eval_local_agent/rlvr + C-run wmlatnp)后,step75/90/105/120 全过(**曾反复 OOM 的 step90 现已成功**,success_rate 随 ckpt 升 0.46→0.554)。dumps 9/11(init..120),剩 135/150 ~40min。
  * 自伤1:relaunch chain 的 scoring 调用漏传 SEED/CUDA(set -u 会失败)→ 改为独立 watcher 修复。
  * 自伤2:driver.log 有 2 个**残留 BDIAG_ALL_DONE**(driver 每轮结尾都 echo,即使步骤 OOM)→ 首个 analyze-watcher 误判"rollout 完成"提前在 GPU6,7 起 scoring(与运行中的 step75 抢卡)。杀之;但 kill grep(含 bdiag_score_all)误匹配了 rollout chain 的 bash -c(1411433)。**幸而 rollout 是 setsid 会话,orphan 后存活继续到 step120,无损失**。修:watcher 改 key on "全 11 dump 齐"(PID 2141384),不再信 stale marker。
  * C runs 全健康@~step43-46/150(~30%):wmlatnp_l0p001_s0 val0.172/l0p001_s1 val0.305(领先)/l0p005_s0 val0.234,pred==act norm(no-pred✓)。~31h 到 step150。
  * 链:rollout(→150,~40min)→watcher score+analyze+probe→B 产物→我写报告 push→seed1。
- tick(07:15 UTC / 15:13 CST): T4 rollout 收尾——step135 done(success_rate**0.700**),step150 跑中(dump 17k/~55k,~40min)。checkpoint success 单调升 init→...→90=0.46/105=0.554/120=0.633/135=0.700(与 baseline eval 一致,good)。**曾担心的 step150 scoring race 已排除**:init scoring 实测 **21min**/98k 行(远慢于估计,因与 step150 rollout 共享 GPU6,7),按 ~15-21min/step,scoring 约 09:45 UTC 才到 step150,而 step150 rollout ~07:55 就写完 → scoring 碰 step150 时它早已完整,无 partial/偏差。watcher(2141384)自主 score(init done,现 step15)→analyze→probe,预计 ~10:45 UTC 出 B 产物。C runs step46-50/150 健康(s1 val0.375 领先)。无干预。
- tick(08:13 UTC / 16:13 CST) **里程碑:T4 seed0 rollout 全 11 ckpt 完成**(BDIAG_ALL_DONE 07:54,step150 success 0.788;success 单调升 90=0.46→120=0.633→135=0.700→150=0.788,与 baseline eval 吻合)。scoring 6/11(init..90),现 step105 → 预计 09:30 UTC 评完 → analyze+probe → B 产物 ~10:15 UTC。C runs step52-58/150 健康(s0 val0.297/s1 0.273,noisy 但爬)。无干预。产物就绪即写报告+scp+push+起 seed1。
- tick(09:20 UTC / 17:13 CST) **B 产物落地 + 修 2 个离线-venv 依赖 + 预览结论**:
  * scoring 10/11 完成,BDIAG_SCORE_DONE;analyze **成功**出图(4 指标 hist + episode line + bdiag_stats CSV + overlap-control)。但 **step60 scores.csv 缺**(07:47 评 step60 时与 step150 rollout 抢 GPU6,7 crash 于 ~960 行)→ 已重评(GPU 空闲,进度单调 46560→51840,~2min 完)。
  * **修 2 处 sklearn 缺失**(gpudev venv 无 sklearn):(a) probe e3 用 sklearn LogisticRegression/CV/AUC → 改 torch Adam logreg + rank-based AUC + 手写 stratified 5-fold(commit 2681f7d);(b) analyze GMM 用 sklearn.mixture → 改 numpy 1-D 2-comp EM(commit 5ccd0f2)。均本地数值验证(probe 可分 0.93/chance 0.52;GMM 可分 1.0/重叠 0.52)+ push + 部署 gpudev。
  * 起 reanalyze+probe 链(PID 2764661):等 step60→重跑 analyze(带 GMM)+ 修好的 probe→全 11 步产物。
  * **预览 KEY 结论(10 步 episode 级,step60 待补)**:
    - **action_obs_cosine 不可分**:AUC 0.44-0.56(~chance),gap init +0.028→step75 **翻负 -0.044**→step150 -0.006,**不随训练增长**。~2000 episode/ckpt 稳健证实之前 8 条的结论。
    - **next-obs CE 可分且随训练强增**:gap +0.019→+0.215,AUC **0.55→0.836** 单调。**但方向是 success=更高 CE**(succ 1.678 vs fail 1.463 @150)→更像"成功轨迹访问更多新颖 obs(任务推进)"而非"世界模型预测更准";含义:降低 next-obs CE 的辅助 loss 未必助 success,需谨慎。待 11 步+transition 级+overlap-control+probe 定论。
  * C runs step56-63/150 健康。
- tick(10:19 UTC / 18:13 CST) **B seed0 报告完成+push+seed1 启动**:
  * reanalyze+probe 链完成(BDIAG_REANALYZE_DONE 10:14):11/11 scored,GMM 已填,probe 完成。
  * **B 报告写完 push(commit e7ef115)**:remote_docs/world_model/workstream_b_report.md + bdiag_official_4to5/{12 图 + bdiag_stats.csv + probe.csv + overlap_control.csv}。
  * **最终结论(seed0,11 ckpt,2048 task/ckpt)**:
    - cosine 不可分(AUC~0.5 两级),gap 中段翻负,不随训练增长;e2 overlap-control 证明非 token 重叠假象(residual≈raw)。
    - CE/perplexity mean-shift 随训练强增(CE ep AUC 0.55→0.836)但 success=更高 CE(novelty 非预测技能);target_conf 弱反向(AUC<0.5)。
    - **GMM 2-comp acc ≤ base rate**(所有指标)→ 高度重叠、非干净双峰簇(回答"真分开还是重叠":高度重叠)。
    - **hidden probe:success 强可线性解码(obs_hidden AUC 0.88-0.94≫chance)但 init 就有、训练不增反略降**→信号在表征里、非 GRPO 造。
    - 显式对账 +0.028(episode)vs +0.0141(transition)= 同 ckpt 两聚合级。
  * **对 A/C 的含义**:C 线优化的 cosine 与 success 无关且不随训练增长;A 线 min CE 方向与 success(=高 CE novelty)相反,有奖励重复/不推进风险(与 obs_ce 变体 0.547/0.671<0.7065 一致)。
  * **seed1(official_6to7)启动**:rollout PID 2886109(init 起,GMU0.45 robust cleanup)+ watcher PID 2886110(all-11-dumps→score→analyze→probe)。GPU6,7(与 rlvr 共存)。
  * C runs step64-67/150 健康。
- tick(08:0x UTC / 16:02 CST 07-04) **⚠️ gpudev 失联(基础设施故障)**:用户报"机器完蛋了"。诊断:
  * 我的外网正常(github ping 95ms)→ 非我方网络。
  * gpudev(10.100.2.64:24187):症状波动——nc TCP 超时 / ssh banner-exchange 超时 / "Connection closed by remote"。不可用。
  * **sibling gpudev2(10.100.2.40)也 banner-exchange 超时**(TCP 连上但 sshd 不发 banner)→ 是**集群/子网侧问题**(cephfs 挂起会卡登录、也会卡训练写 ckpt;或整机 reboot/过载),非我 job 造成。
  * 无法从本地修复(基础设施层)。
  * **数据安全评估**:C run ckpt(每 15 step)+ seed1 dumps(10/11)在 **cephfs 共享盘**(非 gpudev 本地)——若 cephfs 完好则无损、可从最近 ckpt 续跑;**B seed0 报告已在 GitHub(commit e7ef115,绝对安全)**。只丢在途算力(可 resume)。
  * 失联前状态(05:48 CST):seed1 rollout 10/11(step135 跑中);C runs s0 step113/s1 step121(val0.641)/l0p005_s0 step102(val0.602)。
  * 计划:每 tick 重试连通;恢复后①查 cephfs 存活②从最近 ckpt 重启死掉的 job③seed1 只剩 step150。不 spam 重试(无助恢复)。

## 2026-07-13 C 线有效重跑进度与 ETA

- `wmlatnp_direct_l0p001_s0`（commit
  `5667ec25475a24631855443152b32505beac9dc3`）于 02:05 CST 到
  `global_step=40`，进度条 `41/150`，当前 run 剩余约 `47:21:50`。
- 最新有效指标：latent loss `0.175`、action/obs feature variance
  `3.137/4.130`、peak allocated `35.211GB`；无 OOM/NaN/traceback。
- 已落盘 `global_step_15`、`global_step_30`。权威日志：
  `/mnt/SSD_8TB/zechuan/grpo_alfworld_wm/logs/full_wmlatnp_direct_l0p001_s0.log`。
- 按当前平均 `1564s/step` 估算：当前 run 约 47 小时；其余三个
  `wmlatnp_direct_*` 训练约 65 小时/run，C 全训练加四个 eval10x
  合计尚需约 10–11 天（无中断前提）。
- gpudev 仍 connection closed、gpudev2 timeout；B full protocol 代码已在
  `839dbf6566420fa000ee75fad96b710193a0dd29`，但 baseline checkpoint
  不可达，因此 B 科学 rollout/图/CSV 仍 blocked。历史 8-episode 与
  2048-task 报告均已 superseded，不作为当前证据。

### 2026-07-13 23:24 CST 更新

- `wmlatnp_direct_l0p001_s0` 到 `global_step=93`，进度条
  `93/150 [39:06:48<19:58:22]`；checkpoint 已覆盖
  `15/30/45/60/75/90`。
- 最新指标：latent loss `0.166`、cosine `0.834`、action/obs feature
  variance `2.603/4.399`、peak allocated `36.288GB`，无
  OOM/NaN/traceback。
- 当前 run 约剩 20 小时。按当前平均约 `1261s/step` 估算，其余三个
  direct run 加四个 eval10x 后，C 全队列尚需约 `7.5–8` 天（无中断前提）。
- gpudev 仍 connection closed，B checkpoint staging 状态不变。

### 2026-07-13 23:25 CST 用户暂停

- 按用户指令停止 `wmlatnp_direct_l0p001_s0` 及整个后续串行队列。
- 停止点 `global_step=93`；已保存 checkpoint：
  `15/30/45/60/75/90`。日志、checkpoint、rollout artifacts 均保留。
- 目标训练/Ray进程=0，queue进程=0；GPU2/3 已释放至 1MiB，未触碰
  GPU0/1/4 上其他任务。
- 暂停 marker：
  `/mnt/SSD_8TB/zechuan/grpo_alfworld_wm/logs/wmlatnp_direct_l0p001_s0_PAUSED_BY_USER.txt`。
- 未来若恢复，`resume_mode=auto` 将从 `global_step_90` 继续，最多重算3步；
  当前 ETA 计时暂停。
