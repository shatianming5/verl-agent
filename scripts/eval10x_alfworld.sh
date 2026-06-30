#!/usr/bin/env bash
# Robust 10x evaluation of a finished ALFWorld step-150 checkpoint to reduce the
# high single-pass eval variance.
#
# Safe for concurrent training: this does not call ray stop. Each iteration uses
# its own short RAY_TMPDIR and runs on the caller-provided GPUs.
set -uo pipefail

CKPT=${CKPT:?set CKPT=/path/to/global_step_150}
LABEL=${LABEL:?set LABEL, e.g. seed0}
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES, e.g. 6,7}"
N_EVALS=${N_EVALS:-10}
VAL_DATA_SIZE=${VAL_DATA_SIZE:-128}
EVAL_DATASET=${EVAL_DATASET:-eval_in_distribution}
N_GPUS=${N_GPUS:-2}
ROLLOUT_TP=${ROLLOUT_TP:-2}
EXTRA_HYDRA_OVERRIDES=${EXTRA_HYDRA_OVERRIDES:-}

append_hydra_override() {
  if [[ -n "$EXTRA_HYDRA_OVERRIDES" ]]; then
    EXTRA_HYDRA_OVERRIDES+=" $1"
  else
    EXTRA_HYDRA_OVERRIDES="$1"
  fi
}

if [[ -n "${LAMBDA_OBS:-}" ]]; then
  OBS_CE_MAX_LENGTH=${OBS_CE_MAX_LENGTH:-512}
  OBS_CE_LOSS_AGG_MODE=${OBS_CE_LOSS_AGG_MODE:-token-mean}
  append_hydra_override "actor_rollout_ref.actor.world_model.obs_ce_enable=True"
  append_hydra_override "actor_rollout_ref.actor.world_model.lambda_obs=${LAMBDA_OBS}"
  append_hydra_override "actor_rollout_ref.actor.world_model.obs_ce_max_length=${OBS_CE_MAX_LENGTH}"
  append_hydra_override "actor_rollout_ref.actor.world_model.obs_ce_loss_agg_mode=${OBS_CE_LOSS_AGG_MODE}"
fi

if [[ -n "${LAMBDA_LATENT:-}" ]]; then
  LATENT_MAX_LENGTH=${LATENT_MAX_LENGTH:-512}
  LATENT_TARGET=${LATENT_TARGET:-text}
  LATENT_PREDICTOR_HIDDEN_SIZE=${LATENT_PREDICTOR_HIDDEN_SIZE:-0}
  LATENT_PREDICTOR_DROPOUT=${LATENT_PREDICTOR_DROPOUT:-0.0}
  append_hydra_override "actor_rollout_ref.actor.world_model.latent_enable=True"
  append_hydra_override "actor_rollout_ref.actor.world_model.lambda_latent=${LAMBDA_LATENT}"
  append_hydra_override "actor_rollout_ref.actor.world_model.latent_max_length=${LATENT_MAX_LENGTH}"
  append_hydra_override "actor_rollout_ref.actor.world_model.latent_target=${LATENT_TARGET}"
  append_hydra_override "actor_rollout_ref.actor.world_model.latent_predictor_hidden_size=${LATENT_PREDICTOR_HIDDEN_SIZE}"
  append_hydra_override "actor_rollout_ref.actor.world_model.latent_predictor_dropout=${LATENT_PREDICTOR_DROPOUT}"
fi

extra_hydra_args=()
if [[ -n "$EXTRA_HYDRA_OVERRIDES" ]]; then
  # Intentional word splitting: Hydra overrides are passed as separate CLI args.
  extra_hydra_args=($EXTRA_HYDRA_OVERRIDES)
fi

source /root/grpo/grpo_alfworld_common.sh
export CUDA_VISIBLE_DEVICES
disable_proxy_for_ray

OUT=/mnt/cephfs_home_tianming.sha/grpo_alfworld/logs/eval10x_${LABEL}_results.txt
: > "$OUT"
echo "EVAL10X_START label=$LABEL ckpt=$CKPT n=$N_EVALS val_size=$VAL_DATA_SIZE dataset=$EVAL_DATASET cuda=$CUDA_VISIBLE_DEVICES $(date -u)" | tee -a "$OUT"
LABEL_HASH=$(printf "%s" "$LABEL" | cksum | awk '{print $1}')

for i in $(seq 0 $((N_EVALS - 1))); do
  TAG=eval10x_${LABEL}_${i}
  export RAY_TMPDIR=/tmp/ray_eval_${LABEL_HASH}_${i}
  mkdir -p "$RAY_TMPDIR"
  DATA_DIR=/root/data/verl-agent_${TAG}
  LOG=$WORK/logs/${TAG}_$(date +%Y%m%d_%H%M%S).log

  "$VENV/bin/python" -m examples.data_preprocess.prepare \
    --mode text --train_data_size 16 --val_data_size "$VAL_DATA_SIZE" --local_dir "$DATA_DIR" >/dev/null 2>&1

  "$VENV/bin/python" -m verl.trainer.main_ppo \
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
    actor_rollout_ref.rollout.gpu_memory_utilization=0.80 \
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
    > "$LOG" 2>&1

  sr=$(grep -aoE "val/success_rate:[0-9.]+" "$LOG" | tail -1 | grep -oE "[0-9.]+")
  echo "eval $i env.seed=$i success_rate=${sr:-NA} log=$LOG" | tee -a "$OUT"
  rm -rf "$DATA_DIR" "$RAY_TMPDIR" "$WORK/checkpoints/${TAG}" 2>/dev/null
done

echo "--- summary $LABEL ---" | tee -a "$OUT"
grep -oE "success_rate=[0-9.]+" "$OUT" | grep -oE "[0-9.]+" | \
  awk '{n++; s+=$1; ss+=$1*$1; v[n]=$1} END{ if(n>0){m=s/n; var=(ss-s*s/n)/(n>1?n-1:1); sd=sqrt(var>0?var:0); printf "EVAL10X_%s n=%d mean=%.4f std=%.4f\n","RESULT",n,m,sd} }' | tee -a "$OUT"
echo "EVAL10X_DONE label=$LABEL $(date -u)" | tee -a "$OUT"
