#!/usr/bin/env bash
# Run a GOAL_RD ALFWorld world-model experiment fully from gpudev local disk.
set -euo pipefail

KIND=${1:?usage: CUDA_VISIBLE_DEVICES=4,5 gpudev_run_world_model_local.sh latent|obs_ce SEED VALUE TAG}
SEED=${2:?usage: CUDA_VISIBLE_DEVICES=4,5 gpudev_run_world_model_local.sh latent|obs_ce SEED VALUE TAG}
VALUE=${3:?usage: CUDA_VISIBLE_DEVICES=4,5 gpudev_run_world_model_local.sh latent|obs_ce SEED VALUE TAG}
TAG=${4:?usage: CUDA_VISIBLE_DEVICES=4,5 gpudev_run_world_model_local.sh latent|obs_ce SEED VALUE TAG}
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES, e.g. 4,5}"

WORK=${WORK:-/root/grpo/local_alfworld}
REPO=${REPO:-/root/grpo/verl-agent}
MODEL=${MODEL:-/root/grpo/models/Qwen2.5-1.5B-Instruct}
VENV=${VENV:-/root/grpo/venv}
ALFWORLD_DATA=${ALFWORLD_DATA:-/root/grpo/alfworld_data}

source /root/grpo/env.sh

export WORK REPO MODEL VENV ALFWORLD_DATA TAG CUDA_VISIBLE_DEVICES
export PYTHONPATH=$REPO:${PYTHONPATH:-}
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1
export VERL_DISABLE_FLASH_ATTN_CE=${VERL_DISABLE_FLASH_ATTN_CE:-1}
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME=$WORK/hf_home
export HF_DATASETS_CACHE=$WORK/hf_datasets
export WANDB_DIR=$WORK/wandb
export WANDB_MODE=${WANDB_MODE:-offline}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export WM_DUMP_ROLLOUTS=${WM_DUMP_ROLLOUTS:-1}

raise_file_limit() {
  ulimit -n 65535 2>/dev/null || ulimit -n 4096 2>/dev/null || true
}

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
RAY_TMPDIR=${RUN_RAY_TMPDIR:-/root/grpo/ray_tmp_${TAG}_local}
ray_socket_probe="${RAY_TMPDIR}/ray/session_2026-06-28_22-40-18_939895_12345678/sockets/plasma_store"
if (( ${#ray_socket_probe} > 107 )); then
  ray_tmp_hash=$(printf '%s_%s' "$TAG" "$SEED" | cksum | awk '{print $1}')
  RAY_TMPDIR=/root/grpo/ray_${ray_tmp_hash}
fi
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
    LATENT_PREDICTOR_HIDDEN_SIZE=${LATENT_PREDICTOR_HIDDEN_SIZE:-0}
    LATENT_PREDICTOR_DROPOUT=${LATENT_PREDICTOR_DROPOUT:-0.0}
    WM_ARGS+=(
      actor_rollout_ref.actor.world_model.obs_ce_enable=False
      actor_rollout_ref.actor.world_model.latent_enable=True
      actor_rollout_ref.actor.world_model.lambda_latent="$VALUE"
      actor_rollout_ref.actor.world_model.latent_max_length="$LATENT_MAX_LENGTH"
      actor_rollout_ref.actor.world_model.latent_target="$LATENT_TARGET"
      actor_rollout_ref.actor.world_model.latent_predictor_hidden_size="$LATENT_PREDICTOR_HIDDEN_SIZE"
      actor_rollout_ref.actor.world_model.latent_predictor_dropout="$LATENT_PREDICTOR_DROPOUT"
      trainer.rollout_data_dir="$ROLLOUT_DATA_DIR"
    )
    ;;
  obs_ce)
    OBS_CE_MAX_LENGTH=${OBS_CE_MAX_LENGTH:-512}
    OBS_CE_TARGET=${OBS_CE_TARGET:-text}
    WM_ARGS+=(
      actor_rollout_ref.actor.world_model.obs_ce_enable=True
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

TRAIN_DATA_SIZE=16
VAL_DATA_SIZE=128
GROUP_SIZE=8
N_GPUS=2
ROLLOUT_TP=2
PPO_MINI=256
PPO_MICRO=16
LOGPROB_MICRO=16
REF_MICRO=16
GMU=0.6
TOTAL_EPOCHS=${TOTAL_EPOCHS:-150}
SAVE_FREQ=${SAVE_FREQ:-15}
TEST_FREQ=${TEST_FREQ:-5}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-True}
RAY_OBJECT_STORE_MEMORY=${RAY_OBJECT_STORE_MEMORY:-64000000000}

LOG=$WORK/logs/${EXP}_$(date +%Y%m%d_%H%M%S).log
CKPT_DIR=$WORK/checkpoints/$EXP
mkdir -p "$CKPT_DIR"
printf '%s\n' "$RAY_TMPDIR" > "$WORK/logs/${EXP}_ray_tmpdir.txt"
DATA_DIR=/root/data/verl-agent_${TAG}

echo "RUN_WM_LOCAL kind=$KIND seed=$SEED tag=$TAG cuda=$CUDA_VISIBLE_DEVICES value=$VALUE work=$WORK model=$MODEL ckpt=$CKPT_DIR rollout_data_dir=$ROLLOUT_DATA_DIR"

"$VENV/bin/python" -m examples.data_preprocess.prepare \
  --mode text --train_data_size "$TRAIN_DATA_SIZE" --val_data_size "$VAL_DATA_SIZE" --local_dir "$DATA_DIR"

disable_proxy_for_ray

"$VENV/bin/python" -m verl.trainer.main_ppo \
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
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
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
  trainer.max_actor_ckpt_to_keep=null \
  trainer.max_critic_ckpt_to_keep=null \
  "${WM_ARGS[@]}" \
  "${EXTRA_HYDRA_ARGS[@]}" \
  2>&1 | tee -a "$LOG"

echo "WM_LOCAL_DONE kind=$KIND seed=$SEED tag=$TAG log=$LOG ckpt=$CKPT_DIR rollout_data_dir=$ROLLOUT_DATA_DIR"
