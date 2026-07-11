#!/usr/bin/env bash
# 10x ALFWorld evaluation against checkpoints stored on gpudev local disk.
#
# This mirrors eval10x_alfworld.sh but avoids the Ceph-backed common config so
# eval logs, Ray temp files, and cleanup checkpoints remain under WORK.
set -uo pipefail

CKPT=${CKPT:?set CKPT=/path/to/global_step_150}
LABEL=${LABEL:?set LABEL, e.g. wm_obs_ce_l0p001_s0}
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES, e.g. 6,7}"

WORK=${WORK:-/root/grpo/local_alfworld}
REPO=${REPO:-/root/grpo/verl-agent}
MODEL=${MODEL:-/root/grpo/models/Qwen2.5-1.5B-Instruct}
VENV=${VENV:-/root/grpo/venv}
ALFWORLD_DATA=${ALFWORLD_DATA:-/root/grpo/alfworld_data}

N_EVALS=${N_EVALS:-10}
VAL_DATA_SIZE=${VAL_DATA_SIZE:-128}
EVAL_DATASET=${EVAL_DATASET:-eval_in_distribution}
N_GPUS=${N_GPUS:-2}
ROLLOUT_TP=${ROLLOUT_TP:-2}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.70}
EXTRA_HYDRA_OVERRIDES=${EXTRA_HYDRA_OVERRIDES:-}
OUT=${OUT:-$WORK/logs/eval10x_${LABEL}_results.txt}
PREPARED_DATA_DIR=${PREPARED_DATA_DIR:-}

append_hydra_override() {
  if [[ -n "$EXTRA_HYDRA_OVERRIDES" ]]; then
    EXTRA_HYDRA_OVERRIDES+=" $1"
  else
    EXTRA_HYDRA_OVERRIDES="$1"
  fi
}

if [[ -n "${LAMBDA_OBS:-}" ]]; then
  OBS_CE_MAX_LENGTH=${OBS_CE_MAX_LENGTH:-512}
  OBS_CE_TARGET=${OBS_CE_TARGET:-text}
  OBS_CE_LOSS_AGG_MODE=${OBS_CE_LOSS_AGG_MODE:-token-mean}
  append_hydra_override "actor_rollout_ref.actor.world_model.lambda_obs=${LAMBDA_OBS}"
  append_hydra_override "actor_rollout_ref.actor.world_model.obs_ce_coef=${LAMBDA_OBS}"
  append_hydra_override "actor_rollout_ref.actor.world_model.obs_ce_max_length=${OBS_CE_MAX_LENGTH}"
  append_hydra_override "actor_rollout_ref.actor.world_model.obs_ce_target=${OBS_CE_TARGET}"
  append_hydra_override "actor_rollout_ref.actor.world_model.obs_ce_loss_agg_mode=${OBS_CE_LOSS_AGG_MODE}"
fi

if [[ -n "${LAMBDA_LATENT:-}" ]]; then
  LATENT_MAX_LENGTH=${LATENT_MAX_LENGTH:-512}
  LATENT_TARGET=${LATENT_TARGET:-text}
  append_hydra_override "actor_rollout_ref.actor.world_model.lambda_latent=${LAMBDA_LATENT}"
  append_hydra_override "actor_rollout_ref.actor.world_model.latent_max_length=${LATENT_MAX_LENGTH}"
  append_hydra_override "actor_rollout_ref.actor.world_model.latent_target=${LATENT_TARGET}"
  append_hydra_override "+actor_rollout_ref.actor.world_model.latent_use_predictor=False"
  append_hydra_override "+actor_rollout_ref.actor.world_model.latent_contrastive=False"
fi

extra_hydra_args=()
if [[ -n "$EXTRA_HYDRA_OVERRIDES" ]]; then
  # Intentional word splitting: Hydra overrides are passed as separate CLI args.
  extra_hydra_args=($EXTRA_HYDRA_OVERRIDES)
fi

source /root/grpo/env.sh

export WORK REPO MODEL VENV ALFWORLD_DATA CUDA_VISIBLE_DEVICES
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

mkdir -p "$WORK/logs" "$WORK/checkpoints" "$WANDB_DIR" "$HF_HOME" "$HF_DATASETS_CACHE"
cd "$REPO"
disable_proxy_for_ray

: > "$OUT"
echo "EVAL10X_START label=$LABEL ckpt=$CKPT n=$N_EVALS val_size=$VAL_DATA_SIZE dataset=$EVAL_DATASET cuda=$CUDA_VISIBLE_DEVICES work=$WORK model=$MODEL $(date -u)" | tee -a "$OUT"
LABEL_HASH=$(printf "%s" "$LABEL" | cksum | awk '{print $1}')

for i in $(seq 0 $((N_EVALS - 1))); do
  TAG=eval10x_${LABEL}_${i}
  export RAY_TMPDIR=/root/grpo/ray_eval_local_${LABEL_HASH}_${i}
  mkdir -p "$RAY_TMPDIR"
  CLEAN_DATA_DIR=0
  if [[ -n "$PREPARED_DATA_DIR" ]]; then
    DATA_DIR=$PREPARED_DATA_DIR
  else
    DATA_DIR=/root/data/verl-agent_${TAG}
    "$VENV/bin/python" -m examples.data_preprocess.prepare \
      --mode text --train_data_size 16 --val_data_size "$VAL_DATA_SIZE" --local_dir "$DATA_DIR" >/dev/null 2>&1
    CLEAN_DATA_DIR=1
  fi
  LOG=$WORK/logs/${TAG}_$(date +%Y%m%d_%H%M%S).log

  if [[ ! -s "$DATA_DIR/text/train.parquet" || ! -s "$DATA_DIR/text/test.parquet" ]]; then
    echo "eval $i env.seed=$i status=DATA_MISSING data_dir=$DATA_DIR log=$LOG" | tee -a "$OUT"
    exit 1
  fi

  if ! "$VENV/bin/python" -m verl.trainer.main_ppo \
    ray_init.num_cpus=32 \
    +ray_init.include_dashboard=False \
    +ray_init.object_store_memory=48000000000 \
    algorithm.adv_estimator=grpo \
    data.train_files="$DATA_DIR/text/train.parquet" \
    data.val_files="$DATA_DIR/text/test.parquet" \
    data.train_batch_size=16 \
    data.val_batch_size="$VAL_DATA_SIZE" \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="$MODEL" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization="$ROLLOUT_GPU_MEMORY_UTILIZATION" \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed="$i" \
    env.max_steps=50 \
    env.rollout.n=8 \
    env.resources_per_worker.num_cpus=0.1 \
    env.alfworld.eval_dataset="$EVAL_DATASET" \
    trainer.critic_warmup=0 \
    "trainer.logger=['console']" \
    trainer.project_name=verl_agent_alfworld_eval10x \
    trainer.experiment_name="$TAG" \
    trainer.n_gpus_per_node="$N_GPUS" \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=1 \
    trainer.total_epochs=150 \
    trainer.val_before_train=True \
    trainer.val_only=True \
    trainer.resume_mode=resume_path \
    trainer.resume_from_path="$CKPT" \
    trainer.default_local_dir="$WORK/checkpoints/${TAG}" \
    trainer.default_hdfs_dir=null \
    "${extra_hydra_args[@]}" \
    > "$LOG" 2>&1; then
    echo "eval $i env.seed=$i status=FAILED log=$LOG" | tee -a "$OUT"
    exit 1
  fi

  sr=$(grep -aoE "val/success_rate:[0-9.]+" "$LOG" | tail -1 | grep -oE "[0-9.]+")
  echo "eval $i env.seed=$i success_rate=${sr:-NA} log=$LOG" | tee -a "$OUT"
  if [[ "$CLEAN_DATA_DIR" == "1" ]]; then
    rm -rf "$DATA_DIR" 2>/dev/null
  fi
  rm -rf "$RAY_TMPDIR" "$WORK/checkpoints/${TAG}" 2>/dev/null
done

echo "--- summary $LABEL ---" | tee -a "$OUT"
grep -oE "success_rate=[0-9.]+" "$OUT" | grep -oE "[0-9.]+" | \
  awk '{n++; s+=$1; ss+=$1*$1} END{ if(n>0){m=s/n; var=(ss-s*s/n)/(n>1?n-1:1); sd=sqrt(var>0?var:0); printf "EVAL10X_%s n=%d mean=%.4f std=%.4f\n","RESULT",n,m,sd} }' | tee -a "$OUT"
echo "EVAL10X_DONE label=$LABEL $(date -u)" | tee -a "$OUT"
