#!/usr/bin/env bash
# Queue remaining GOAL_RD world-model training runs and launch only on a safe 2-GPU pair.
set -euo pipefail

WORK=${WORK:-/mnt/cephfs_home_tianming.sha/grpo_alfworld}
STAMP=${STAMP:-$(date -u +%Y%m%d_%H%M%S)}
LOG_DIR=${LOG_DIR:-$WORK/logs/queue_launchers}
STATUS_DIR=${STATUS_DIR:-$WORK/logs/run_status}
JOB_LOG_DIR=${JOB_LOG_DIR:-$WORK/logs}
mkdir -p "$LOG_DIR" "$STATUS_DIR" "$JOB_LOG_DIR"

QUEUE_LOG=${QUEUE_LOG:-$LOG_DIR/world_model_queue_${STAMP}.log}
CANDIDATE_PAIRS=(${CANDIDATE_PAIRS:-"0,1 2,3 4,5 6,7"})
SLEEP_SECONDS=${SLEEP_SECONDS:-60}
MAX_GPU_MEM_MIB=${MAX_GPU_MEM_MIB:-500}
ALLOW_ASSIGNED_PAIRS=${ALLOW_ASSIGNED_PAIRS:-0}

JOBS=(
  "latent_l0p001_s1|wmlat_l0p001_s1|latent|1|0.001"
  "obs_ce_l0p05_s1|wm_obs_ce_l0p05_s1|obs_ce|1|0.05"
)

log() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$QUEUE_LOG"
}

refresh_snapshots() {
  MEM_FILE=$(mktemp /tmp/wm_queue_mem.XXXXXX)
  PMON_FILE=$(mktemp /tmp/wm_queue_pmon.XXXXXX)
  PS_FILE=$(mktemp /tmp/wm_queue_ps.XXXXXX)
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits > "$MEM_FILE"
  nvidia-smi pmon -c 1 > "$PMON_FILE"
  ps -eo pid,stat,cmd > "$PS_FILE"
}

cleanup_snapshots() {
  rm -f "${MEM_FILE:-}" "${PMON_FILE:-}" "${PS_FILE:-}"
}

gpu_mem_used() {
  local gpu=$1
  awk -F, -v g="$gpu" '$1 + 0 == g { gsub(/ /, "", $2); print $2 + 0 }' "$MEM_FILE"
}

gpu_has_pmon_compute() {
  local gpu=$1
  awk -v g="$gpu" '$1 == g && $2 != "-" { found=1 } END { exit found ? 0 : 1 }' "$PMON_FILE"
}

pair_has_known_assignment() {
  local pair=$1
  case "$pair" in
    0,1)
      grep -E 'wm_obs_ce_l0p05_s0|CUDA_VISIBLE_DEVICES=0,1' "$PS_FILE" >/dev/null && return 0
      ;;
    2,3)
      grep -E 'official_6to7|wm_obs_ce_l0p03_s0|CUDA_VISIBLE_DEVICES=2,3' "$PS_FILE" >/dev/null && return 0
      ;;
    4,5)
      grep -E 'official_s2|wmlat_l0p001_s1|CUDA_VISIBLE_DEVICES=4,5' "$PS_FILE" >/dev/null && return 0
      ;;
    6,7)
      grep -E 'wm_obs_ce_l0p03_s1|CUDA_VISIBLE_DEVICES=6,7' "$PS_FILE" >/dev/null && return 0
      ;;
  esac
  return 1
}

pair_is_safe() {
  local pair=$1
  local left=${pair%,*}
  local right=${pair#*,}
  local left_mem right_mem
  left_mem=$(gpu_mem_used "$left")
  right_mem=$(gpu_mem_used "$right")

  [[ -n "$left_mem" && -n "$right_mem" ]] || return 1
  (( left_mem < MAX_GPU_MEM_MIB && right_mem < MAX_GPU_MEM_MIB )) || return 1
  ! gpu_has_pmon_compute "$left" || return 1
  ! gpu_has_pmon_compute "$right" || return 1
  [[ "$ALLOW_ASSIGNED_PAIRS" == "1" ]] || ! pair_has_known_assignment "$pair" || return 1
}

first_safe_pair() {
  local pair
  refresh_snapshots
  for pair in "${CANDIDATE_PAIRS[@]}"; do
    if pair_is_safe "$pair"; then
      printf '%s\n' "$pair"
      cleanup_snapshots
      return 0
    fi
  done
  cleanup_snapshots
  return 1
}

job_active() {
  local tag=$1
  ps -eo cmd | grep -F "$tag" | grep -E 'run_wm|run_seed_alfworld|verl.trainer.main_ppo|ray::' | grep -v grep >/dev/null
}

launch_job() {
  local job_id=$1 tag=$2 kind=$3 seed=$4 value=$5 pair=$6
  local log_file="$JOB_LOG_DIR/${tag}_queued_${STAMP}.log"
  local pid_file="$STATUS_DIR/${tag}.queued_pid"

  if job_active "$tag"; then
    log "SKIP already-active job=$job_id tag=$tag"
    return 0
  fi

  log "LAUNCH job=$job_id tag=$tag kind=$kind seed=$seed value=$value cuda=$pair log=$log_file"
  if [[ "$kind" == "latent" ]]; then
    nohup env TAG="$tag" WM_DUMP_ROLLOUTS=1 LAMBDA_LATENT="$value" CUDA_VISIBLE_DEVICES="$pair" \
      bash /root/grpo/run_wm_latent_seed.sh "$seed" > "$log_file" 2>&1 < /dev/null &
  elif [[ "$kind" == "obs_ce" ]]; then
    nohup env TAG="$tag" WM_DUMP_ROLLOUTS=1 LAMBDA_OBS="$value" CUDA_VISIBLE_DEVICES="$pair" \
      bash /root/grpo/run_wm_obs_ce_seed.sh "$seed" > "$log_file" 2>&1 < /dev/null &
  else
    log "ERROR unknown kind=$kind for job=$job_id"
    return 1
  fi
  printf '%s\n' "$!" > "$pid_file"
  log "STARTED job=$job_id tag=$tag pid=$! pid_file=$pid_file"
  sleep 90
}

main() {
  local launched=0
  log "QUEUE_START jobs=${#JOBS[@]} candidate_pairs=${CANDIDATE_PAIRS[*]} max_gpu_mem_mib=$MAX_GPU_MEM_MIB allow_assigned_pairs=$ALLOW_ASSIGNED_PAIRS"

  local spec job_id tag kind seed value pair
  for spec in "${JOBS[@]}"; do
    IFS='|' read -r job_id tag kind seed value <<< "$spec"
    while true; do
      if pair=$(first_safe_pair); then
        launch_job "$job_id" "$tag" "$kind" "$seed" "$value" "$pair"
        launched=$((launched + 1))
        break
      fi
      log "WAIT no_safe_pair next_job=$job_id sleep=${SLEEP_SECONDS}s"
      sleep "$SLEEP_SECONDS"
    done
  done

  log "QUEUE_DONE launched=$launched"
}

main "$@"
