#!/usr/bin/env bash
# Run an ALFWorld world-model experiment on the .183 box (10x RTX4090D) from
# /mnt/SSD1_8TB/zechuan local disk, reusing the shared `gdpo` conda env.
#
# This is the .183 migration of scripts/gpudev_run_world_model_local.sh after the
# 2026-07-04 gpudev/cephfs outage made the original checkpoints unreachable.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=4,5 scripts/run_world_model_183.sh latent 0 0.001 wmlatnp_l0p001_s0
#   CUDA_VISIBLE_DEVICES=7,8 scripts/run_world_model_183.sh obs_ce 0 0.001 wm_obs_ce_l0p001_s0
#
# Smoke test (short run):
#   TOTAL_EPOCHS=6 SAVE_FREQ=3 TEST_FREQ=3 CUDA_VISIBLE_DEVICES=4,5 \
#     scripts/run_world_model_183.sh latent 0 0.001 smoke_wmlatnp_l0p001_s0
set -euo pipefail

KIND=${1:?usage: CUDA_VISIBLE_DEVICES=4,5 run_world_model_183.sh latent|obs_ce SEED VALUE TAG}
SEED=${2:?usage: CUDA_VISIBLE_DEVICES=4,5 run_world_model_183.sh latent|obs_ce SEED VALUE TAG}
VALUE=${3:?usage: CUDA_VISIBLE_DEVICES=4,5 run_world_model_183.sh latent|obs_ce SEED VALUE TAG}
TAG=${4:?usage: CUDA_VISIBLE_DEVICES=4,5 run_world_model_183.sh latent|obs_ce SEED VALUE TAG}
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES, e.g. 4,5}"

# --- .183 / zechuan local layout ---------------------------------------------
ROOT=${ROOT:-/mnt/SSD1_8TB/zechuan/grpo_alfworld_wm}
WORK=${WORK:-$ROOT}
REPO=${REPO:-$ROOT/verl-agent}
MODEL=${MODEL:-/mnt/SSD1_8TB/zechuan/models/Qwen2.5-1.5B-Instruct}
PYBIN=${PYBIN:-$HOME/anaconda3/envs/gdpo/bin/python}
ALFWORLD_DATA=${ALFWORLD_DATA:-/mnt/SSD1_8TB/zechuan/.cache/alfworld}

export WORK REPO MODEL ALFWORLD_DATA TAG CUDA_VISIBLE_DEVICES
export PYTHONPATH=$REPO:${PYTHONPATH:-}
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1
# Defragment CUDA allocator: reclaims the ~10GB reserved-but-unallocated blocks
# that caused OOM in the actor-update backward pass when sharing cards with jusheng.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export VERL_DISABLE_FLASH_ATTN_CE=${VERL_DISABLE_FLASH_ATTN_CE:-1}
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME=$WORK/hf_home
export HF_DATASETS_CACHE=$WORK/hf_datasets
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export WANDB_DIR=$WORK/wandb
export WANDB_MODE=${WANDB_MODE:-offline}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export WM_DUMP_ROLLOUTS=${WM_DUMP_ROLLOUTS:-1}
export TMPDIR=${TMPDIR:-$WORK/tmp}
mkdir -p "$TMPDIR"

raise_file_limit() { ulimit -n 65535 2>/dev/null || ulimit -n 4096 2>/dev/null || true; }

disable_proxy_for_ray() {
  local base_no_proxy host_name host_ips
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
  base_no_proxy=localhost,127.0.0.1,::1,169.254.169.254
  host_name=$(hostname 2>/dev/null || true)
  host_ips=$(hostname -I 2>/dev/null | tr ' ' ',' | sed 's/,$//' || true)
  export no_proxy="${base_no_proxy}${host_name:+,$host_name}${host_ips:+,$host_ips}"
  export NO_PROXY="$no_proxy"
  export RAY_USAGE_STATS_ENABLED=0
  export RAY_BACKEND_LOG_LEVEL=${RAY_BACKEND_LOG_LEVEL:-warning}
  export RAY_DEDUP_LOGS=${RAY_DEDUP_LOGS:-0}
  raise_file_limit
}

EXP=grpo_qwen2.5_1.5b_alfworld_seed${SEED}_${TAG}
RAY_TMPDIR=${RUN_RAY_TMPDIR:-$WORK/ray_tmp_${TAG}}
ray_tmp_hash=$(printf '%s_%s' "$TAG" "$SEED" | cksum | awk '{print $1}')
# ray plasma socket path must stay < 108 chars; fall back to a short hashed dir.
if (( ${#RAY_TMPDIR} > 60 )); then RAY_TMPDIR=/tmp/ray_z_${ray_tmp_hash}; fi
export RAY_TMPDIR

mkdir -p "$WORK/logs" "$WORK/checkpoints" "$WANDB_DIR" "$HF_HOME" "$HF_DATASETS_CACHE" "$RAY_TMPDIR"
cd "$REPO"

ROLLOUT_DATA_DIR=${ROLLOUT_DATA_DIR:-$WORK/logs/world_model_rollouts/${TAG}_seed${SEED}}
mkdir -p "$ROLLOUT_DATA_DIR"

WM_ARGS=()
case "$KIND" in
  latent)
    LATENT_MAX_LENGTH=${LATENT_MAX_LENGTH:-512}
    LATENT_TARGET=${LATENT_TARGET:-text}
    # no-predictor objective:  L_latent = 1 - cos(h_action, sg(h_obs))
    WM_ARGS+=(
      actor_rollout_ref.actor.world_model.lambda_latent="$VALUE"
      actor_rollout_ref.actor.world_model.latent_max_length="$LATENT_MAX_LENGTH"
      actor_rollout_ref.actor.world_model.latent_target="$LATENT_TARGET"
      +actor_rollout_ref.actor.world_model.latent_use_predictor=False
      trainer.rollout_data_dir="$ROLLOUT_DATA_DIR"
    )
    ;;
  obs_ce)
    OBS_CE_MAX_LENGTH=${OBS_CE_MAX_LENGTH:-512}
    OBS_CE_TARGET=${OBS_CE_TARGET:-text}
    WM_ARGS+=(
      actor_rollout_ref.actor.world_model.lambda_obs="$VALUE"
      actor_rollout_ref.actor.world_model.obs_ce_coef="$VALUE"
      actor_rollout_ref.actor.world_model.obs_ce_max_length="$OBS_CE_MAX_LENGTH"
      actor_rollout_ref.actor.world_model.obs_ce_target="$OBS_CE_TARGET"
      trainer.rollout_data_dir="$ROLLOUT_DATA_DIR"
    )
    ;;
  *)
    echo "unknown KIND=$KIND; expected latent or obs_ce" >&2
    exit 2
    ;;
esac

EXTRA_HYDRA_ARGS=()
if [[ -n "${EXTRA_HYDRA_OVERRIDES:-}" ]]; then
  read -r -a EXTRA_HYDRA_ARGS <<< "$EXTRA_HYDRA_OVERRIDES"
fi

TRAIN_DATA_SIZE=${TRAIN_DATA_SIZE:-16}
VAL_DATA_SIZE=${VAL_DATA_SIZE:-128}
GROUP_SIZE=${GROUP_SIZE:-8}
N_GPUS=2
ROLLOUT_TP=2
PPO_MINI=${PPO_MINI:-256}
PPO_MICRO=${PPO_MICRO:-16}
LOGPROB_MICRO=${LOGPROB_MICRO:-16}
REF_MICRO=${REF_MICRO:-16}
# GMU 0.30 (was 0.45): shrink vLLM KV-cache reservation so the actor-update
# backward pass has headroom when sharing 49GB cards with jusheng (~11GB neighbor).
GMU=${GMU:-0.30}
# Offload optimizer state (Adam moments, ~6GB) to CPU during actor update — the
# backward-pass peak is what OOM'd on cards 4,5. param stays on GPU for speed.
OPTIMIZER_OFFLOAD=${OPTIMIZER_OFFLOAD:-True}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-150}
SAVE_FREQ=${SAVE_FREQ:-15}
TEST_FREQ=${TEST_FREQ:-5}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-True}
RAY_OBJECT_STORE_MEMORY=${RAY_OBJECT_STORE_MEMORY:-64000000000}
# Weight management: keep a rolling window of the most recent N actor checkpoints
# (verl auto-deletes older ones) instead of retaining every ckpt forever. Single
# SSD1 disk with no backup was a cephfs-outage-shaped single point of failure.
# GRPO has no critic, so only the actor knob matters. Set to null to disable.
MAX_ACTOR_CKPT_TO_KEEP=${MAX_ACTOR_CKPT_TO_KEEP:-3}
# Optional off-box mirror of each checkpoint as it lands. Empty = disabled.
# Format: rsync-style destination, e.g. user@host:/path or /mnt/other/backup.
CKPT_BACKUP_DEST=${CKPT_BACKUP_DEST:-}

LOG=$WORK/logs/${EXP}_$(date +%Y%m%d_%H%M%S).log
CKPT_DIR=$WORK/checkpoints/$EXP
mkdir -p "$CKPT_DIR"
printf '%s\n' "$RAY_TMPDIR" > "$WORK/logs/${EXP}_ray_tmpdir.txt"
DATA_DIR=${DATA_DIR:-$WORK/data/verl-agent_wmretrain}

echo "RUN_WM_183 kind=$KIND seed=$SEED tag=$TAG cuda=$CUDA_VISIBLE_DEVICES value=$VALUE work=$WORK model=$MODEL ckpt=$CKPT_DIR rollout_data_dir=$ROLLOUT_DATA_DIR epochs=$TOTAL_EPOCHS gmu=$GMU opt_offload=$OPTIMIZER_OFFLOAD alloc_conf=$PYTORCH_CUDA_ALLOC_CONF"

if [[ -s "$DATA_DIR/text/train.parquet" && -s "$DATA_DIR/text/test.parquet" ]]; then
  echo "REUSE_PREPARED_DATA data_dir=$DATA_DIR"
else
  echo "ERROR: missing parquet under $DATA_DIR/text (generate it first)" >&2
  exit 3
fi

disable_proxy_for_ray

"$PYBIN" -m verl.trainer.main_ppo \
  ray_init.num_cpus=64 \
  +ray_init.include_dashboard=False \
  +ray_init.object_store_memory="$RAY_OBJECT_STORE_MEMORY" \
  algorithm.adv_estimator=grpo \
  data.train_files="$DATA_DIR/text/train.parquet" \
  data.val_files="$DATA_DIR/text/test.parquet" \
  data.train_batch_size="$TRAIN_DATA_SIZE" \
  data.val_batch_size="$VAL_DATA_SIZE" \
  data.max_prompt_length=2048 \
  data.max_response_length=512 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.return_raw_chat=True \
  actor_rollout_ref.model.path="$MODEL" \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$PPO_MICRO" \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.01 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload="$OPTIMIZER_OFFLOAD" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$LOGPROB_MICRO" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP" \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization="$GMU" \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$REF_MICRO" \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.use_invalid_action_penalty=True \
  actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
  algorithm.use_kl_in_reward=False \
  env.env_name=alfworld/AlfredTWEnv \
  env.seed="$SEED" \
  env.max_steps=50 \
  env.rollout.n="$GROUP_SIZE" \
  env.resources_per_worker.num_cpus=0.1 \
  env.alfworld.eval_dataset=eval_in_distribution \
  trainer.critic_warmup=0 \
  "trainer.logger=['console','wandb']" \
  trainer.project_name=verl_agent_alfworld \
  trainer.experiment_name="$EXP" \
  trainer.n_gpus_per_node="$N_GPUS" \
  trainer.nnodes=1 \
  trainer.save_freq="$SAVE_FREQ" \
  trainer.test_freq="$TEST_FREQ" \
  trainer.total_epochs="$TOTAL_EPOCHS" \
  trainer.val_before_train="$VAL_BEFORE_TRAIN" \
  trainer.default_local_dir="$CKPT_DIR" \
  trainer.default_hdfs_dir=null \
  trainer.max_actor_ckpt_to_keep="$MAX_ACTOR_CKPT_TO_KEEP" \
  trainer.max_critic_ckpt_to_keep=null \
  "${WM_ARGS[@]}" \
  "${EXTRA_HYDRA_ARGS[@]}" \
  2>&1 | tee -a "$LOG"

if [[ -n "$CKPT_BACKUP_DEST" ]]; then
  echo "CKPT_BACKUP start dest=$CKPT_BACKUP_DEST src=$CKPT_DIR"
  rsync -a --delete "$CKPT_DIR/" "$CKPT_BACKUP_DEST/$EXP/" \
    && echo "CKPT_BACKUP done dest=$CKPT_BACKUP_DEST/$EXP" \
    || echo "CKPT_BACKUP FAILED dest=$CKPT_BACKUP_DEST (non-fatal; local ckpt intact)"
fi

echo "WM_183_DONE kind=$KIND seed=$SEED tag=$TAG log=$LOG ckpt=$CKPT_DIR rollout_data_dir=$ROLLOUT_DATA_DIR"
