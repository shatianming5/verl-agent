#!/usr/bin/env bash
# Workstream B (proper protocol): roll out one baseline checkpoint on the TRAIN games
# with training sampling (temperature=1.0) and dump world-model transitions
# ({step}.val.wm_transitions.jsonl with success/failure labels) for offline scoring.
#
# Unlike eval10x_alfworld.sh (val split, temp 0.4, no dump) this uses the train split,
# temp 1.0, and trainer.rollout_data_dir to dump transitions. Safe for concurrent training:
# no ray stop, isolated RAY_TMPDIR, val_only, save_freq=-1.
set -uo pipefail

CKPT=${CKPT:?set CKPT=/path/to/global_step_N or CKPT=base for init}
LABEL=${LABEL:?set LABEL, e.g. bdiag_official_4to5_step150}
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES, e.g. 6,7}"
DUMP_DIR=${DUMP_DIR:?set DUMP_DIR=/path/to/rollout_dump_dir}
N_TASKS=${N_TASKS:-3072}          # number of train episodes to roll out
N_TRAJ=${N_TRAJ:-1}               # trajectories per task (env.rollout.n); >=1
TEMP=${TEMP:-1.0}                 # train-consistent sampling
N_GPUS=${N_GPUS:-2}
ROLLOUT_TP=${ROLLOUT_TP:-2}
GMU=${GMU:-0.80}

source /root/grpo/grpo_alfworld_common.sh
export CUDA_VISIBLE_DEVICES
disable_proxy_for_ray

LABEL_HASH=$(printf "%s" "$LABEL" | cksum | awk '{print $1}')
export RAY_TMPDIR=/root/grpo/ray_bdiag_${LABEL_HASH}
mkdir -p "$RAY_TMPDIR" "$DUMP_DIR"
DATA_DIR=/root/data/verl-agent_bdiag_${LABEL}
LOG=$WORK/logs/bdiag_rollout_${LABEL}_$(date +%Y%m%d_%H%M%S).log

# val.parquet drives how many episodes are rolled out (val_data_size); train.parquet unused here.
"$VENV/bin/python" -m examples.data_preprocess.prepare \
  --mode text --train_data_size 16 --val_data_size "$N_TASKS" --local_dir "$DATA_DIR" >/dev/null 2>&1

resume_args=()
if [[ "$CKPT" == "base" || "$CKPT" == "init" ]]; then
  : # no resume: base model = init checkpoint
else
  resume_args=(trainer.resume_mode=resume_path trainer.resume_from_path="$CKPT")
fi

echo "BDIAG_ROLLOUT_START label=$LABEL ckpt=$CKPT n_tasks=$N_TASKS n_traj=$N_TRAJ temp=$TEMP dump=$DUMP_DIR cuda=$CUDA_VISIBLE_DEVICES $(date -u)" | tee -a "$LOG"

"$VENV/bin/python" -m verl.trainer.main_ppo \
  ray_init.num_cpus=32 \
  +ray_init.include_dashboard=False \
  +ray_init.object_store_memory=48000000000 \
  algorithm.adv_estimator=grpo \
  data.train_files="$DATA_DIR/text/train.parquet" \
  data.val_files="$DATA_DIR/text/test.parquet" \
  data.train_batch_size=16 \
  data.val_batch_size="$N_TASKS" \
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
  actor_rollout_ref.rollout.gpu_memory_utilization="$GMU" \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.val_kwargs.temperature="$TEMP" \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.use_invalid_action_penalty=True \
  actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
  algorithm.use_kl_in_reward=False \
  env.env_name=alfworld/AlfredTWEnv \
  env.seed=0 \
  env.max_steps=50 \
  env.rollout.n="$N_TRAJ" \
  env.resources_per_worker.num_cpus=0.1 \
  env.alfworld.eval_dataset=train \
  trainer.critic_warmup=0 \
  "trainer.logger=['console']" \
  trainer.project_name=verl_agent_alfworld_bdiag \
  trainer.experiment_name="bdiag_${LABEL}" \
  trainer.n_gpus_per_node="$N_GPUS" \
  trainer.nnodes=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=1 \
  trainer.total_epochs=150 \
  trainer.val_before_train=True \
  trainer.val_only=True \
  +trainer.validation_data_dir="$DUMP_DIR" \
  "${resume_args[@]}" \
  trainer.default_local_dir="$WORK/checkpoints/bdiag_${LABEL}" \
  trainer.default_hdfs_dir=null \
  >> "$LOG" 2>&1

sr=$(grep -aoE "val/success_rate:[0-9.]+" "$LOG" | tail -1 | grep -oE "[0-9.]+")
dumped=$(ls -t "$DUMP_DIR"/*.wm_transitions.jsonl 2>/dev/null | head -1)
n_lines=$( [ -n "$dumped" ] && wc -l < "$dumped" || echo 0 )
echo "BDIAG_ROLLOUT_DONE label=$LABEL success_rate=${sr:-NA} dump_file=${dumped:-NONE} transitions=${n_lines} log=$LOG" | tee -a "$LOG"
rm -rf "$DATA_DIR" "$RAY_TMPDIR" "$WORK/checkpoints/bdiag_${LABEL}" 2>/dev/null
