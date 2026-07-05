# ERROR_BOOK — intern_123 错题本

## E1 - 共享 GPU 机满占所有卡跑多 run → 机器资源耗尽拖垮所有 run（2026-07-06）

**场景**：.183(10×4090D)做配置搜索，一次起 4 个测试 run + 主 run(launch10）= 满占全 10 卡跑 5 个 verl/ray 训练。

**后果**：机器共享资源耗尽 —— `raylet memory_monitor: Got negative`，ray object store（每 run 预留 64GB）+ 64 核 CPU + 内存被 5 个 run 瓜分，**所有 run 被拖垮，包括已通过冒烟的主 run（崩在 step4）**。SSH 也因机器过载反复断连。

**根因**：verl 每个 run 起 `object_store_memory=64G` + 256 个 ALFWorld env worker + FSDP/vLLM，单机跑 2-3 个已接近极限，5 个必爆。GPU 显存够 ≠ 机器整体资源够。

**教训**：
- **共享机绝不满占所有卡跑多 run**。配置搜索/多 run 应**串行**，或**最多 2 个并行**，且留 CPU/内存余量。
- 起多 run 前算总资源：N×object_store(64G) + N×env_workers 是否超机器内存/CPU。
- 主 run（承载重要成果的）**单独跑，不与实验 run 混在同机满负荷**。

**幸免**：冒烟核心成果（step3 ckpt 19GB）在崩溃前已落盘，零损失。但这是运气，不是设计——重要 run 该隔离保护。
