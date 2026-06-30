#!/usr/bin/env bash
# Score a fixed wm_transition_v1 JSONL dump across a checkpoint series.
set -euo pipefail

TRANSITIONS_JSONL=${TRANSITIONS_JSONL:?set TRANSITIONS_JSONL=/path/to/*.wm_transitions.jsonl}
CKPT_ROOT=${CKPT_ROOT:?set CKPT_ROOT=/path/to/checkpoints/<experiment>}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SCORER=${SCORER:-"$SCRIPT_DIR/wm_score_transition_dump.py"}
WORK=${WORK:-/mnt/cephfs_home_tianming.sha/grpo_alfworld}
COMMON_SH=${COMMON_SH:-/root/grpo/grpo_alfworld_common.sh}

if [[ -f "$COMMON_SH" ]]; then
  # Provides MODEL, VENV, WORK and Ray/proxy helpers on gpudev.
  # shellcheck source=/dev/null
  source "$COMMON_SH"
  disable_proxy_for_ray || true
fi

MODEL=${MODEL:?set MODEL=/path/to/base/model or source COMMON_SH}
PYTHON=${PYTHON:-${VENV:+$VENV/bin/python}}
PYTHON=${PYTHON:-python}

TAG=${TAG:-wm_checkpoint_diagnostics}
LABEL_PREFIX=${LABEL_PREFIX:-$(basename "$CKPT_ROOT")}
STEPS=${STEPS:-init 30 60 90 120 150}
MAX_LENGTH=${MAX_LENGTH:-512}
BATCH_SIZE=${BATCH_SIZE:-1}
MAX_ROWS=${MAX_ROWS:-0}
DEVICE=${DEVICE:-cuda}
DTYPE=${DTYPE:-bfloat16}
OUT_DIR=${OUT_DIR:-$WORK/logs/world_model_diagnostics/$TAG}
OUT_CSV=${OUT_CSV:-$OUT_DIR/checkpoint_scores.csv}
SUMMARY_JSON=${SUMMARY_JSON:-$OUT_DIR/checkpoint_scores_summary.json}
LOG=${LOG:-$OUT_DIR/run_$(date +%Y%m%d_%H%M%S).log}
CHAT_TEMPLATE_KWARGS_JSON=${CHAT_TEMPLATE_KWARGS_JSON:-{}}

mkdir -p "$OUT_DIR"

checkpoint_args=()
for step in $STEPS; do
  if [[ "$step" == "init" || "$step" == "0" ]]; then
    checkpoint_args+=(--checkpoint "${LABEL_PREFIX}_init=base")
    continue
  fi
  ckpt=$CKPT_ROOT/global_step_$step
  if [[ -d "$ckpt/actor" ]]; then
    checkpoint_args+=(--checkpoint "${LABEL_PREFIX}_step${step}=$ckpt")
  else
    echo "WM_CHECKPOINT_DIAGNOSTICS_SKIP missing=$ckpt" | tee -a "$LOG"
  fi
done

if [[ ${#checkpoint_args[@]} -eq 0 ]]; then
  echo "No checkpoints selected from STEPS='$STEPS' under $CKPT_ROOT" >&2
  exit 1
fi

max_rows_args=()
if [[ "$MAX_ROWS" != "0" ]]; then
  max_rows_args=(--max-rows "$MAX_ROWS")
fi

skip_entropy_args=()
if [[ "${SKIP_ENTROPY:-0}" == "1" ]]; then
  skip_entropy_args=(--skip-entropy)
fi

echo "RUN_WM_CHECKPOINT_DIAGNOSTICS tag=$TAG transitions=$TRANSITIONS_JSONL ckpt_root=$CKPT_ROOT steps=$STEPS out_dir=$OUT_DIR device=$DEVICE dtype=$DTYPE"

"$PYTHON" "$SCORER" \
  --model-path "$MODEL" \
  --transition-jsonl "$TRANSITIONS_JSONL" \
  --output-csv "$OUT_CSV" \
  --summary-json "$SUMMARY_JSON" \
  --max-length "$MAX_LENGTH" \
  --batch-size "$BATCH_SIZE" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --chat-template-kwargs-json "$CHAT_TEMPLATE_KWARGS_JSON" \
  "${max_rows_args[@]}" \
  "${skip_entropy_args[@]}" \
  "${checkpoint_args[@]}" \
  2>&1 | tee -a "$LOG"

test -s "$OUT_CSV"
test -s "$SUMMARY_JSON"
grep -q '"checkpoints"' "$SUMMARY_JSON"

echo "WM_CHECKPOINT_DIAGNOSTICS_DONE csv=$OUT_CSV summary=$SUMMARY_JSON log=$LOG"
