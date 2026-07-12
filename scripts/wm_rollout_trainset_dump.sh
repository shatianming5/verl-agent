#!/usr/bin/env bash
# Roll out one checkpoint on every game in a validated ALFWorld train manifest.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CKPT=${CKPT:?set CKPT=base or /path/to/global_step_N}
CKPT_STEP=${CKPT_STEP:?set CKPT_STEP=init,15,...,150}
LABEL=${LABEL:?set a unique LABEL}
DUMP_DIR=${DUMP_DIR:?set DUMP_DIR to a dedicated checkpoint output directory}
MANIFEST=${MANIFEST:?set MANIFEST to an alfworld_train_manifest_v1 JSON file}
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES explicitly}"

VAL_BATCH=${VAL_BATCH:-128}
N_TRAJ=${N_TRAJ:-1}
TEMP=${TEMP:-1.0}
TOP_P=${TOP_P:-1.0}
TOP_K=${TOP_K:--1}
DO_SAMPLE=${DO_SAMPLE:-true}
ENV_SEED=${ENV_SEED:-0}
EXPECTED_GAMES=${EXPECTED_GAMES:-3553}
EXPECTED_RAW_TRAJECTORIES=${EXPECTED_RAW_TRAJECTORIES:-6374}
N_GPUS=${N_GPUS:-2}
ROLLOUT_TP=${ROLLOUT_TP:-2}
GMU=${GMU:-0.45}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-512}
COMMON_SH=${COMMON_SH:-/root/grpo/grpo_alfworld_common.sh}
PREPARE_PYTHON=${PREPARE_PYTHON:-}

if [[ "$N_TRAJ" != "1" ]]; then
  echo "Full protocol currently requires exactly one scheduled trajectory per manifest row (N_TRAJ=1)" >&2
  exit 2
fi
if [[ ! "$EXPECTED_GAMES" =~ ^[0-9]+$ ]] \
  || [[ ! "$EXPECTED_RAW_TRAJECTORIES" =~ ^[0-9]+$ ]] \
  || (( EXPECTED_GAMES != 3553 || EXPECTED_RAW_TRAJECTORIES != 6374 )); then
  echo "Full Workstream B requires exactly 6374 raw trajectories and 3553 filtered train games" >&2
  exit 2
fi
if [[ "$DO_SAMPLE" != "true" ]]; then
  echo "Training-matched Workstream B decoding requires DO_SAMPLE=true" >&2
  exit 2
fi
if [[ "$TEMP" != "1" && "$TEMP" != "1.0" ]] \
  || [[ "$TOP_P" != "1" && "$TOP_P" != "1.0" ]] \
  || [[ "$TOP_K" != "-1" ]]; then
  echo "Baseline training decoding is temperature=1.0, top_p=1.0, top_k=-1; overrides are rejected" >&2
  exit 2
fi
if [[ "$CKPT_STEP" == "init" ]]; then
  if [[ "$CKPT" != "base" && "$CKPT" != "init" ]]; then
    echo "CKPT_STEP=init requires CKPT=base or CKPT=init" >&2
    exit 2
  fi
elif [[ ! -d "$CKPT/actor" ]]; then
  echo "Missing checkpoint actor directory: $CKPT/actor" >&2
  exit 2
fi

if [[ -f "$COMMON_SH" ]]; then
  # shellcheck source=/dev/null
  source "$COMMON_SH"
  if declare -F disable_proxy_for_ray >/dev/null; then
    disable_proxy_for_ray
  fi
fi

MODEL=${MODEL:?set MODEL or provide it through COMMON_SH}
WORK=${WORK:?set WORK or provide it through COMMON_SH}
TRAIN_PYTHON=${TRAIN_PYTHON:-${VENV:+$VENV/bin/python}}
TRAIN_PYTHON=${TRAIN_PYTHON:-python}
PREPARE_PYTHON=${PREPARE_PYTHON:-$TRAIN_PYTHON}
MANIFEST_TOOL=${MANIFEST_TOOL:-"$SCRIPT_DIR/wm_alfworld_train_manifest.py"}
COVERAGE_TOOL=${COVERAGE_TOOL:-"$SCRIPT_DIR/wm_validate_rollout_coverage.py"}

mkdir -p "$DUMP_DIR" "$WORK/logs" "$WORK/.wm_b_protocol"
if find "$DUMP_DIR" -maxdepth 1 -type f \( -name '*.wm_transitions.jsonl' -o -name 'coverage.json' \) -print -quit | grep -q .; then
  echo "Refusing to reuse a checkpoint dump directory containing protocol outputs: $DUMP_DIR" >&2
  exit 2
fi

"$PREPARE_PYTHON" "$MANIFEST_TOOL" verify \
  --manifest "$MANIFEST" \
  --expected-games "$EXPECTED_GAMES" \
  --expected-raw-trajectories "$EXPECTED_RAW_TRAJECTORIES" \
  --verify-files

LABEL_HASH=$(printf "%s" "$LABEL" | cksum | awk '{print $1}')
RUN_DIR="$WORK/.wm_b_protocol/${LABEL_HASH}"
DATA_DIR="$RUN_DIR/data"
RAY_TMPDIR=${RAY_TMPDIR:-"$RUN_DIR/ray"}
LOG=${LOG:-"$WORK/logs/bdiag_rollout_${LABEL}_$(date +%Y%m%d_%H%M%S).log"}
SCHEDULE_PARQUET="$DATA_DIR/text/train.parquet"
export RAY_TMPDIR CUDA_VISIBLE_DEVICES
mkdir -p "$(dirname "$SCHEDULE_PARQUET")" "$RAY_TMPDIR"

"$PREPARE_PYTHON" "$MANIFEST_TOOL" schedule \
  --manifest "$MANIFEST" \
  --output-parquet "$SCHEDULE_PARQUET" \
  --batch-size "$VAL_BATCH" \
  --expected-games "$EXPECTED_GAMES" \
  --expected-raw-trajectories "$EXPECTED_RAW_TRAJECTORIES"
test -s "$SCHEDULE_PARQUET"

resume_args=()
if [[ "$CKPT_STEP" != "init" ]]; then
  resume_args=(trainer.resume_mode=resume_path trainer.resume_from_path="$CKPT")
fi

echo \
  "BDIAG_ROLLOUT_START label=$LABEL ckpt=$CKPT checkpoint_step=$CKPT_STEP " \
  "manifest=$MANIFEST val_batch=$VAL_BATCH n_traj=$N_TRAJ temperature=$TEMP " \
  "top_p=$TOP_P top_k=$TOP_K do_sample=$DO_SAMPLE dump=$DUMP_DIR cuda=$CUDA_VISIBLE_DEVICES" \
  | tee -a "$LOG"

"$TRAIN_PYTHON" -m verl.trainer.main_ppo \
  ray_init.num_cpus=32 \
  +ray_init.include_dashboard=False \
  +ray_init.object_store_memory=48000000000 \
  algorithm.adv_estimator=grpo \
  data.train_files="$SCHEDULE_PARQUET" \
  data.val_files="$SCHEDULE_PARQUET" \
  data.train_batch_size=16 \
  data.val_batch_size="$VAL_BATCH" \
  data.max_prompt_length="$MAX_PROMPT_LENGTH" \
  data.max_response_length="$MAX_RESPONSE_LENGTH" \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.return_raw_chat=True \
  data.shuffle=False \
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
  actor_rollout_ref.rollout.temperature="$TEMP" \
  actor_rollout_ref.rollout.top_p="$TOP_P" \
  actor_rollout_ref.rollout.top_k="$TOP_K" \
  actor_rollout_ref.rollout.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.temperature="$TEMP" \
  actor_rollout_ref.rollout.val_kwargs.top_p="$TOP_P" \
  actor_rollout_ref.rollout.val_kwargs.top_k="$TOP_K" \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.n="$N_TRAJ" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.use_invalid_action_penalty=True \
  actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
  algorithm.use_kl_in_reward=False \
  env.env_name=alfworld/AlfredTWEnv \
  env.seed="$ENV_SEED" \
  env.max_steps=50 \
  env.rollout.n="$N_TRAJ" \
  env.resources_per_worker.num_cpus=0.1 \
  env.alfworld.eval_dataset=train \
  +env.alfworld.manifest_path="$MANIFEST" \
  +env.alfworld.manifest_expected_games="$EXPECTED_GAMES" \
  +env.alfworld.manifest_expected_raw_trajectories="$EXPECTED_RAW_TRAJECTORIES" \
  +env.alfworld.require_manifest_schedule=True \
  +env.alfworld.verify_manifest_files=True \
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
  trainer.validation_data_dir="$DUMP_DIR" \
  +trainer.validation_dump_split=train \
  +trainer.world_model_dump_protocol=workstream_b_full_train_v2 \
  +trainer.world_model_diagnostic_checkpoint_step="$CKPT_STEP" \
  "${resume_args[@]}" \
  trainer.default_local_dir="$RUN_DIR/checkpoints" \
  trainer.default_hdfs_dir=null \
  >>"$LOG" 2>&1

mapfile -t dump_files < <(find "$DUMP_DIR" -maxdepth 1 -type f -name '*.wm_transitions.jsonl' -print | sort)
if [[ ${#dump_files[@]} -ne 1 ]]; then
  echo "Expected exactly one fresh transition dump in $DUMP_DIR, found ${#dump_files[@]}" >&2
  exit 1
fi
DUMP_FILE=${dump_files[0]}
COVERAGE_JSON="$DUMP_DIR/coverage.json"

"$PREPARE_PYTHON" "$COVERAGE_TOOL" \
  --manifest "$MANIFEST" \
  --dump "$DUMP_FILE" \
  --expected-checkpoint-step "$CKPT_STEP" \
  --temperature "$TEMP" \
  --top-p "$TOP_P" \
  --top-k "$TOP_K" \
  --do-sample "$DO_SAMPLE" \
  --expected-games "$EXPECTED_GAMES" \
  --expected-raw-trajectories "$EXPECTED_RAW_TRAJECTORIES" \
  --min-trajectories-per-game 1 \
  --output-json "$COVERAGE_JSON"

test -s "$DUMP_FILE"
test -s "$COVERAGE_JSON"
echo \
  "BDIAG_ROLLOUT_DONE label=$LABEL checkpoint_step=$CKPT_STEP dump_file=$DUMP_FILE " \
  "coverage=$COVERAGE_JSON log=$LOG"
