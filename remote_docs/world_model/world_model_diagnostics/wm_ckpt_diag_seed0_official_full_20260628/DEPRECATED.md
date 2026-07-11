# 作废：8 条轨迹冒烟诊断

该目录基于 `wm_valdump_smoke_s0_step150`，只有 8 条 episode（3 success / 5
failure）。不得引用其中的可分性、success-failure gap 或随 checkpoint 变化趋势。

替代协议必须覆盖训练 split 全部任务、`init/15/30/45/60/75/90/105/120/135/150`
全部 checkpoint，并至少复核两个训练 seed。
