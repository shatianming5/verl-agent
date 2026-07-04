# World-Model Co-Training — Consolidated Status Ledger

**Snapshot:** 2026-07-04 ~08:20 UTC (16:20 CST). Compiled during a **gpudev cluster outage** (see §5) — figures are last-known-good from before the outage; cephfs-side raw logs are currently unreachable.

**Baseline to beat:** GRPO `eval10x = 0.7065 ± 0.0197` (3-seed aggregate; per-seed seed0=0.729). **All world-model variants so far are below this.**

---

## 1. Workstream A — obs-CE co-training (DONE, clean negative)
Auxiliary loss = teacher-forced next-obs token cross-entropy, weight `λ_obs`. Verified **not** polluted by any latent term (WS-A logs show latent loss = 0; evidence in `PROVENANCE.md`).

| run (full name) | λ_obs | eval10x | Δ vs baseline | note |
|---|---|---|---|---|
| `wm_obs_ce_l0p001_s0` | 0.001 | **0.5470** | −0.16 | |
| `wm_obs_ce_l0p01_s0` | 0.01 | **0.6714** | −0.035 | |
| `wm_obs_ce_l0p05_s1` | 0.05 | killed | — | step10 val 0.18, diverged |

**Verdict:** monotonically worse as λ_obs rises. obs-CE co-training does not beat baseline. Consistent with the B finding that success correlates with *higher* obs-CE (novelty), so pushing CE down is counterproductive.

## 2. Predictor-latent (DEPRECATED — "对照 only", not main line)
Per user directive these are retained but **must not enter main-line conclusions**. Implementation used `world_model_predictor`/`latent_predictor` (a predictor head), which the user has since removed from the design.

| run (full name) | λ_latent | eval10x | note |
|---|---|---|---|
| `wmlat_l0p001_s0` | 0.001 | **0.6735 ± 0.0267** | below baseline; predictor version |
| `wmlat_l0p005_s0` | 0.005 | killed (T1) | predictor version |
| `wmlatct_l0p01_s0` | 0.01 | killed (T1) | off-plan contrastive |

## 3. Workstream C — no-predictor latent (THE FINAL method) — IN PROGRESS
Final objective (user-confirmed): **`L_latent = 1 − cos(h_action, stop_gradient(h_obs))`**, **no predictor / projection head**, obs-end stop-gradient. Code: `verl/workers/actor/dp_actor.py` (`latent_use_predictor=False` default, commits `cf5610f`/`51c40b1`). All runs verified no-predictor live via `latent_pred_norm == latent_action_norm` (exact equality).

| run (full name) | λ_latent | seed | GPU | last-known step | last-known train-val | eval10x |
|---|---|---|---|---|---|---|
| `wmlatnp_l0p001_s0` | 0.001 | 0 | 2,3 | 113/150 | 0.469 (peaked 0.539) | pending (not @150 yet) |
| `wmlatnp_l0p001_s1` | 0.001 | 1 | 0,1 | 121/150 | **0.641** (leading) | pending |
| `wmlatnp_l0p005_s0` | 0.005 | 0 | 4,5 | 102/150 | 0.602 | pending |
| `wmlatnp_l0p005_s1` | 0.005 | 1 | — | not launched (deferred) | — | — |

- Last confirmed at **2026-07-03 21:48 UTC (05:48 CST)**, ~10 h before this snapshot. No run had reached `global_step_150`, so **no eval10x exists yet** for the no-predictor method.
- **Caution:** train-time `val/success_rate` runs optimistic and noisy; the real comparison to 0.7065 is `eval10x` at step150. Do not conclude from train-val.
- No NaN/OOM observed in any C run up to the outage.

## 4. Workstream B — separability diagnostic (seed0 DONE, seed1 interrupted)
Full protocol: 11 checkpoints (init,15,…,150) × 2048 train tasks each, temp 1.0, raw metrics (no predictor), bootstrap CI, episode+transition split, GMM(2), obs-overlap control, hidden probe. **Retires** the earlier 8-trajectory smoke analysis.

- **seed0 = `official_4to5`: COMPLETE.** Report `workstream_b_report.md` + `bdiag_official_4to5/` (12 figures + `bdiag_stats.csv` + `probe.csv` + `addon_overlap_control.csv`). Pushed `e7ef115`.
  - action-obs **cosine**: not separable (AUC ~0.5 both levels), gap flips negative mid-training, **no growth**; e2 shows it is **not** a token-overlap artifact (residual ≈ raw gap).
  - next-obs **CE / perplexity**: mean-shift **grows** with training (CE episode AUC 0.55→0.836) **but direction is success = HIGHER CE** ⇒ novelty/progress, not prediction skill.
  - **GMM(2) accuracy ≤ base rate** for every metric ⇒ distributions **highly overlap**, not clean clusters.
  - **hidden-state probe**: success strongly linearly decodable (obs_hidden AUC **0.88–0.94** ≫ chance) but **already high at init**, slightly declining with training ⇒ inherent to representation, not GRPO-induced.
  - Reconciled the earlier `+0.028` (episode) vs `+0.0141` (transition) init-cosine numbers as two aggregation levels of the same checkpoint.
- **seed1 = `official_6to7`: INTERRUPTED by outage.** Rollout was 10/11 (`init…120` dumped, `step135` running); watcher waiting to score→analyze→probe. On cephfs: `logs/bdiag_rollouts/official_6to7/step{init..120}/*.wm_transitions.jsonl`. Needs step135+150 then the score/analyze/probe chain.

## 5. Outage (infrastructure — not caused by these jobs)
- **My internet: healthy** (github.com ping ~95 ms).
- **gpudev (10.100.2.64:24187): unusable** — fluctuating failures: TCP timeout / ssh banner-exchange hang / "connection closed by remote". Classic host crash / reboot / stalled-storage signature.
- **gpudev2 (10.100.2.40): also banner-exchange hang** ⇒ cluster-/subnet-/shared-storage-level issue, not gpudev-single-host, not job-induced. (A hung **cephfs** stalls both ssh logins — home on cephfs — and training checkpoint writes.)
- Cannot be fixed from the client side; awaiting cluster/storage recovery. **Suggest asking the cluster admin whether gpudev / cephfs is down or under maintenance.**

## 6. Data safety
- ✅ **On GitHub (`shatianming5/verl-agent`, branch `world-model-latent-objective`, HEAD `e7ef115`):** all code (dp_actor latent objective, B pipeline scripts), B seed0 report + figures + CSVs, `PROVENANCE.md`, progress reports. **Safe regardless of cluster state.**
- ⏳ **On cephfs (shared filer, currently unreachable):** raw training logs, all checkpoints (saved every 15 steps), rollout dumps, per-checkpoint `scores.csv`. **If cephfs is intact after recovery, nothing is lost** — only in-flight compute, resumable from the last 15-step checkpoint.

## 7. Recovery runbook (when gpudev returns)
1. **Verify cephfs integrity:** `ls` checkpoints for `wmlatnp_l0p001_s0/s1`, `wmlatnp_l0p005_s0` and seed1 rollout dumps; confirm last `global_step_*` per C run.
2. **Resume dead C runs from last checkpoint** (do not restart from scratch): relaunch each `wmlatnp_*` with `resume_from_path=<latest global_step>`; keep no-predictor gating (`+trainer.latent_use_predictor=False`).
3. **Finish B seed1:** resume `bdiag_rollout_all.sh EXP=official_6to7 SEED=1` (skips the 10 dumped, does step135+150), then the score→analyze→probe watcher; reconcile seed0 vs seed1 trends.
4. **eval10x the first C run to hit `global_step_150`** on its freed cards → write result (mark no-predictor, vs baseline 0.7065) → push. Then `wmlatnp_l0p005_s1` if a pair frees.
5. Reporting discipline unchanged: full run names, cite commit+path, announce every start/kill, never touch rlvr, kill by PID only.

## 8. Commit trail (GitHub)
`e7ef115` B full-protocol report · `5ccd0f2` sklearn-free GMM · `2681f7d` sklearn-free probe · `4a646e7` e3 probe · `3a76bb0` B pipeline · `6552a72` provenance+WS-A no-pollution · `51c40b1` contrastive latent · `cf5610f` no-predictor objective + B deep-dive.
