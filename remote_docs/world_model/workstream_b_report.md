# Workstream B — World-Model Metric Separability & Trend (proper protocol)

**Status:** baseline **seed0 = `official_4to5`** complete (11 checkpoints). seed1 confirmation launched separately.
**This supersedes** all prior B analysis done on the 8-trajectory smoke set (`wm_valdump_smoke_s0_step150`); that is retired and must not be cited.

## 1. Protocol & data provenance

- **Checkpoints (full set, every 15 steps):** `init, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150`
  (`init` = base `Qwen2.5-1.5B-Instruct`; others = `checkpoints/grpo_qwen2.5_1.5b_alfworld_seed0_official_4to5/global_step_<s>`).
- **Rollout:** each checkpoint rolled out on **N_TASKS=2048** TRAIN games, 1 traj/task, sampling `temperature=1.0` (train-time setting).
  Script `scripts/wm_rollout_trainset_dump.sh` (GMU 0.45), driver `bdiag_rollout_all.sh`.
  **Infra cap flagged:** 2048/checkpoint (not the full ~3553 train games) — a compute cap, not the full population.
- **Per-trajectory label:** `episode_rewards > 0` ⇒ success. Success rate on train games rises monotonically with the checkpoint:
  init≈0.07 → 90=0.46 → 120=0.63 → 135=0.70 → **150=0.79** (sanity: baseline GRPO training worked).
- **Metrics per transition (teacher-forced, raw — NO predictor/projection):** next-obs CE / NLL, perplexity, target-token confidence,
  action-end↔obs-end last-layer hidden **cosine** (`h_action` at `prefix_len-1`, `h_obs` at `seq_len-1`). Scorer `scripts/wm_score_transition_dump.py`.
- **Aggregation:** reported at **episode level** and **transition level** separately (see reconciliation below). Bootstrap 95% CI on every gap.
- **Code:** analysis `scripts/bdiag_analyze.py` (`5ccd0f2`, numpy GMM), probe `scripts/bdiag_hidden_probe.py` (`2681f7d`, sklearn-free).
- **Artifacts:** `remote_docs/world_model/bdiag_official_4to5/` (figures + `bdiag_stats.csv`, `addon_overlap_control.csv`, `probe.csv`).

### Reconciliation of the two earlier init-cosine numbers
The user flagged that a `+0.028` and a `+0.0141` init cosine gap both appeared without reconciliation. They are the **same checkpoint, two aggregation levels**:
- **episode-level** init action-obs cosine gap = **+0.0280** (mean over per-episode means), CI [−0.0023, +0.0559].
- **transition-level** init gap = **+0.0141**, CI [+0.0051, +0.0232].
Both are now always labeled by level. All numbers below come from `bdiag_stats.csv`.

## 2. Per-metric separability, gap (95% CI), and trend

`gap = mean_success − mean_failure`. `AUC` = per-episode/transition ROC-AUC of the raw metric as a success score. `GMM` = 2-component 1-D Gaussian-mixture cluster→label accuracy vs the majority-class **base rate**.

### 2a. Action–obs hidden cosine (the C-line's target metric) — NOT a success signal
| step | episode gap [CI] | ep AUC | transition gap [CI] | tr AUC |
|---|---|---|---|---|
| init | +0.0280 [−0.002,+0.056] | 0.526 | +0.0141 [+0.005,+0.023] | 0.510 |
| 15 | +0.0544 [+0.032,+0.077] | 0.564 | +0.0506 [+0.044,+0.058] | 0.545 |
| 45 | −0.0069 [−0.037,+0.021] | 0.478 | −0.0068 [−0.016,+0.002] | 0.496 |
| 75 | −0.0436 [−0.063,−0.022] | 0.441 | −0.0465 [−0.052,−0.041] | 0.465 |
| 120 | −0.0245 [−0.045,−0.005] | 0.453 | −0.0309 [−0.036,−0.025] | 0.473 |
| 150 | −0.0056 [−0.028,+0.015] | 0.483 | −0.0072 [−0.013,−0.002] | 0.492 |

- **Not separable at any checkpoint** (AUC ∈ [0.44, 0.56], i.e. ~chance, both levels).
- Gap is small-positive early (peak +0.054 @15), **crosses zero by step45, goes negative** (failures have *higher* cosine) through mid/late training, ends ≈0.
- **Does NOT grow with training** — if anything it degrades. This robustly confirms (on ~2k episodes / 45–98k transitions per checkpoint) the earlier small-sample hint that action-obs cosine does not track success.

### 2b. Next-obs CE / perplexity — mean-shift GROWS with training, but wrong direction
| step | CE ep gap [CI] | CE ep AUC | ppl ep gap | ppl ep AUC |
|---|---|---|---|---|
| init | +0.0189 [−0.000,+0.039] | 0.550 | +0.090 | 0.555 |
| 60 | +0.0409 [+0.031,+0.051] | 0.618 | +0.176 | 0.623 |
| 90 | +0.0572 [+0.048,+0.066] | 0.665 | +0.256 | 0.673 |
| 135 | +0.1268 [+0.116,+0.138] | 0.771 | +0.659 | 0.774 |
| 150 | +0.2148 [+0.201,+0.229] | **0.836** | +1.240 | 0.830 |

- **Separability GROWS monotonically** with the checkpoint: CE episode AUC **0.55 → 0.836** (transition 0.51 → 0.649). Gap CI excludes 0 from step15 on.
- **BUT the sign is `success = HIGHER` CE** (step150: succ mean 1.678 vs fail 1.463). Higher next-obs surprise in *successful* episodes.
- Most parsimonious reading: **novelty / task-progress**, not world-model skill. Successful trajectories advance through the task into new observations (higher CE); failures loop on repetitive, self-predictable observations (lower CE). `target_confidence` corroborates: success = slightly *lower* target-token confidence (gap −0.004→−0.010, AUC 0.37–0.48, i.e. below 0.5).

### 2c. GMM says: mean-shift, but heavily OVERLAPPING (not clean clusters)
For every metric, **GMM 2-component cluster accuracy ≤ the majority base rate** (e.g. CE episode step150 GMM 0.588 vs base 0.788; cosine step120 GMM 0.503 vs base 0.633). A 2-Gaussian mixture on any single metric **cannot recover the success/failure split better than guessing the majority class.** So even where a gap exists (CE), the two class distributions are **highly overlapping with shifted means, not bimodally separable** — see the histograms (`hist_*.png`). This directly answers "真分开还是高度重叠": **highly overlapping.**

## 3. Add-on analyses (on this full dataset, not the 8-traj set)

- **e1 — transition-level NLL calibration:** reported above as the transition rows; CE gap is real but small at transition level (AUC ≤ 0.65 even at step150), and grows with training in the same (success-higher) direction.
- **e2 — obs-overlap control (shortcut check):** prev↔next obs token Jaccard is **~0.90** at every checkpoint (0.9065→0.8913). Regressing out per-pair overlap barely moves the cosine gap: `residual_cos_gap ≈ raw_cos_gap` (step75 raw −0.0465 vs residual −0.0442; step150 raw −0.0072 vs residual −0.0061). So the cosine result is **not** a token-overlap artifact — the (non-)separability is genuine. (`addon_overlap_control.csv`.)
- **e3 — hidden-state success linear probe (5-fold CV ROC-AUC, sklearn-free):**
  | step | obs_hidden AUC | action_hidden AUC | chance |
  |---|---|---|---|
  | init | 0.887 | 0.808 | ~0.49 |
  | 30 | **0.940** | 0.787 | ~0.46 |
  | 45 | 0.932 | 0.857 | ~0.51 |
  | 150 | 0.883 | 0.726 | ~0.49 |

  **Trajectory success is strongly, linearly decodable from the policy's obs-end hidden state (AUC 0.88–0.94 ≫ chance).** But it is **already ~0.89 at `init`** and, if anything, *declines* slightly after step45 — so this rich success signal is **inherent to the representation, not created by GRPO.**

## 4. Answers to the three B questions
1. **Separable per checkpoint?** cosine: **no** (AUC~0.5). CE/perplexity: **only as a shifted mean, heavily overlapping** (AUC up to 0.84 @150 but GMM ≤ base rate). target_confidence: weak, inverted. **Hidden-state probe: yes, strongly (AUC ~0.88–0.94)** — but see (3).
2. **gap = succ − fail (95% CI):** tabulated per metric/level/step in `bdiag_stats.csv` (`gap`, `gap_ci_lo/hi`).
3. **Does separability grow with training?** cosine: **no** (flat→degrading). CE/perplexity: **yes, grows** (AUC 0.55→0.84) but in the novelty direction (success=higher CE). hidden probe: **no** (high from init, slight decline).

## 5. Decision-relevant implications for Workstreams A / C
- The C-line no-predictor latent objective optimizes **action-obs cosine** — a quantity that here is **uncorrelated with success and does not grow with training**. Raising it need not help task success; monitor eval directly, do not assume the latent metric is a success proxy.
- Minimizing **next-obs CE** (Workstream A obs-CE co-training) pushes toward *lower* obs-surprise, but in this data **success co-occurs with HIGHER obs-surprise (novelty/progress).** A naive CE-min auxiliary risks rewarding repetitive, low-progress behavior — a real hazard to flag, consistent with obs_ce variants so far underperforming baseline (0.547 / 0.671 vs **0.7065**).
- The strongest success signal lives in the **hidden state** and is present at init. Objectives that *shape/expose* that existing representation may be more promising than next-obs prediction objectives.

## 6. Artifacts (all under `remote_docs/world_model/bdiag_official_4to5/`)
- Line charts (success/failure/all, per step): `line_episode_*.png`, `line_transition_*.png` for {action_obs_cosine, ce, perplexity, target_confidence_mean}.
- Success-vs-failure distribution histograms: `hist_*.png` (init / mid / step150 panels).
- Data: `bdiag_stats.csv` (per step×level×metric: gap+CI, AUC, GMM), `addon_overlap_control.csv` (e2), `probe.csv` (e3).
- Generators: `scripts/bdiag_analyze.py` (`5ccd0f2`), `scripts/bdiag_hidden_probe.py` (`2681f7d`), `scripts/wm_score_transition_dump.py`, `scripts/wm_rollout_trainset_dump.sh`.
- Raw scores per checkpoint: `logs/bdiag_rollouts/official_4to5/step*/scores.csv` (+ `*.wm_transitions.jsonl` dumps) on cephfs.

## 7. Next
- **seed1 = `official_6to7`** confirmation: same protocol (2048 tasks/checkpoint, temp 1.0), to verify the trends replicate across seeds before any trend is claimed final.
