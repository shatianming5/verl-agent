#!/usr/bin/env bash
# Full Workstream B driver: two baseline seeds × eleven checkpoint-specific rollouts.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MODEL=${MODEL:?set MODEL to the baseline Hugging Face model}
MANIFEST=${MANIFEST:?set MANIFEST to the validated full ALFWorld train manifest}
SEED0_CKPT_ROOT=${SEED0_CKPT_ROOT:-}
SEED1_CKPT_ROOT=${SEED1_CKPT_ROOT:-}
SEED0_ARCHIVE_ROOT=${SEED0_ARCHIVE_ROOT:-}
SEED1_ARCHIVE_ROOT=${SEED1_ARCHIVE_ROOT:-}
STAGE_CHECKPOINTS=${STAGE_CHECKPOINTS:-0}
STAGE_ROOT=${STAGE_ROOT:-}
STAGE_REUSE_VERIFIED=${STAGE_REUSE_VERIFIED:-0}
OUT_ROOT=${OUT_ROOT:?set OUT_ROOT to a new full-protocol result directory}
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES explicitly}"

PYTHON=${PYTHON:-python}
PREPARE_PYTHON=${PREPARE_PYTHON:-"$PYTHON"}
ROLLOUT_SCRIPT=${ROLLOUT_SCRIPT:-"$SCRIPT_DIR/wm_rollout_trainset_dump.sh"}
SCORER=${SCORER:-"$SCRIPT_DIR/wm_score_transition_dump.py"}
ANALYZER=${ANALYZER:-"$SCRIPT_DIR/bdiag_analyze.py"}
PROBE=${PROBE:-"$SCRIPT_DIR/bdiag_hidden_probe.py"}
MANIFEST_TOOL=${MANIFEST_TOOL:-"$SCRIPT_DIR/wm_alfworld_train_manifest.py"}
INVENTORY_TOOL=${INVENTORY_TOOL:-"$SCRIPT_DIR/wm_checkpoint_actor_inventory.py"}
STAGE_TOOL=${STAGE_TOOL:-"$SCRIPT_DIR/wm_stage_checkpoint_actors.py"}
CROSS_SEED_TOOL=${CROSS_SEED_TOOL:-"$SCRIPT_DIR/wm_cross_seed_trends.py"}
SEED_SEPARATION_TOOL=${SEED_SEPARATION_TOOL:-"$SCRIPT_DIR/wm_validate_seed_separation.py"}
RUN_PROBE=${RUN_PROBE:-1}
EXPECTED_GAMES=${EXPECTED_GAMES:-3553}
EXPECTED_RAW_TRAJECTORIES=${EXPECTED_RAW_TRAJECTORIES:-6374}
VAL_BATCH=${VAL_BATCH:-128}
TEMP=${TEMP:-1.0}
TOP_P=${TOP_P:-1.0}
TOP_K=${TOP_K:--1}
DO_SAMPLE=${DO_SAMPLE:-true}
MAX_LENGTH=${MAX_LENGTH:-2048}
BATCH_SIZE=${BATCH_SIZE:-1}
BOOTSTRAP=${BOOTSTRAP:-1000}
GMM_BOOTSTRAP=${GMM_BOOTSTRAP:-50}
PROBE_BATCH_SIZE=${PROBE_BATCH_SIZE:-16}
DEVICE=${DEVICE:-cuda}
DTYPE=${DTYPE:-bfloat16}
N_GPUS=${N_GPUS:-2}
ROLLOUT_TP=${ROLLOUT_TP:-2}
STEPS=(init 15 30 45 60 75 90 105 120 135 150)
SEEDS=(0 1)

if [[ "$RUN_PROBE" != "1" ]]; then
  echo "The full Workstream B protocol requires grouped/nested hidden probes (RUN_PROBE=1)" >&2
  exit 2
fi
if [[ ! "$EXPECTED_GAMES" =~ ^[0-9]+$ ]] \
  || [[ ! "$EXPECTED_RAW_TRAJECTORIES" =~ ^[0-9]+$ ]] \
  || (( EXPECTED_GAMES != 3553 || EXPECTED_RAW_TRAJECTORIES != 6374 )); then
  echo "Full Workstream B requires exactly 6374 raw trajectories and 3553 filtered train games" >&2
  exit 2
fi
if [[ "$STAGE_CHECKPOINTS" != "0" && "$STAGE_CHECKPOINTS" != "1" ]]; then
  echo "STAGE_CHECKPOINTS must be 0 or 1" >&2
  exit 2
fi
if [[ ! "$N_GPUS" =~ ^[0-9]+$ || ! "$ROLLOUT_TP" =~ ^[0-9]+$ ]] \
  || (( N_GPUS < 1 || ROLLOUT_TP < 1 || N_GPUS % ROLLOUT_TP != 0 )); then
  echo "N_GPUS must be positive and divisible by ROLLOUT_TP" >&2
  exit 2
fi
if [[ -e "$OUT_ROOT" ]] && find "$OUT_ROOT" -mindepth 1 -print -quit | grep -q .; then
  echo "Refusing to reuse a non-empty full-protocol output root: $OUT_ROOT" >&2
  exit 2
fi
mkdir -p "$OUT_ROOT"

base_model_complete() {
  [[ -s "$MODEL/config.json" ]] || return 1
  find "$MODEL" -maxdepth 1 -type f \
    \( -name '*.safetensors' -o -name 'pytorch_model*.bin' \) -print -quit | grep -q .
}

if ! base_model_complete; then
  echo "Missing or incomplete init/base model: $MODEL" >&2
  exit 1
fi

"$PREPARE_PYTHON" "$MANIFEST_TOOL" verify \
  --manifest "$MANIFEST" \
  --expected-games "$EXPECTED_GAMES" \
  --expected-raw-trajectories "$EXPECTED_RAW_TRAJECTORIES" \
  --verify-files

if [[ "$STAGE_CHECKPOINTS" == "1" ]]; then
  : "${STAGE_ROOT:?STAGE_CHECKPOINTS=1 requires STAGE_ROOT}"
  : "${SEED0_ARCHIVE_ROOT:?STAGE_CHECKPOINTS=1 requires SEED0_ARCHIVE_ROOT}"
  : "${SEED1_ARCHIVE_ROOT:?STAGE_CHECKPOINTS=1 requires SEED1_ARCHIVE_ROOT}"
  if [[ "$SEED0_ARCHIVE_ROOT" == "$SEED1_ARCHIVE_ROOT" ]]; then
    echo "Seed 0 and seed 1 archive roots must differ" >&2
    exit 2
  fi
  SEED0_CKPT_ROOT="$STAGE_ROOT/seed0"
  SEED1_CKPT_ROOT="$STAGE_ROOT/seed1"
  stage_reuse_args=()
  if [[ "$STAGE_REUSE_VERIFIED" == "1" ]]; then
    stage_reuse_args+=(--reuse-verified)
  fi
  "$PYTHON" "$STAGE_TOOL" \
    --source-root "$SEED0_ARCHIVE_ROOT" \
    --destination-root "$SEED0_CKPT_ROOT" \
    --steps "${STEPS[@]:1}" \
    --expected-world-size "$N_GPUS" \
    "${stage_reuse_args[@]}"
  "$PYTHON" "$STAGE_TOOL" \
    --source-root "$SEED1_ARCHIVE_ROOT" \
    --destination-root "$SEED1_CKPT_ROOT" \
    --steps "${STEPS[@]:1}" \
    --expected-world-size "$N_GPUS" \
    "${stage_reuse_args[@]}"
else
  : "${SEED0_CKPT_ROOT:?set SEED0_CKPT_ROOT or enable STAGE_CHECKPOINTS}"
  : "${SEED1_CKPT_ROOT:?set SEED1_CKPT_ROOT or enable STAGE_CHECKPOINTS}"
fi
if [[ "$SEED0_CKPT_ROOT" == "$SEED1_CKPT_ROOT" ]]; then
  echo "Seed 0 and seed 1 checkpoint roots must differ" >&2
  exit 2
fi
CKPT_ROOTS=("$SEED0_CKPT_ROOT" "$SEED1_CKPT_ROOT")

# Inventory every model/optimizer/extra-state shard before any GPU process.
preflight_dir="$OUT_ROOT/preflight"
mkdir -p "$preflight_dir"
for seed_index in 0 1; do
  seed=${SEEDS[$seed_index]}
  root=${CKPT_ROOTS[$seed_index]}
  for step in "${STEPS[@]:1}"; do
    actor_dir="$root/global_step_${step}/actor"
    "$PYTHON" "$INVENTORY_TOOL" \
      --actor-dir "$actor_dir" \
      --expected-step "$step" \
      --expected-world-size "$N_GPUS" \
      --output-json "$preflight_dir/seed${seed}_step${step}_actor_inventory.json"
  done
done
"$PYTHON" "$SEED_SEPARATION_TOOL" \
  --seed0-root "$SEED0_CKPT_ROOT" \
  --seed1-root "$SEED1_CKPT_ROOT" \
  --seed0-archive-root "$SEED0_ARCHIVE_ROOT" \
  --seed1-archive-root "$SEED1_ARCHIVE_ROOT" \
  --inventory-dir "$preflight_dir" \
  --output-json "$preflight_dir/seed_checkpoint_separation.json"
test -s "$preflight_dir/seed_checkpoint_separation.json"

for seed_index in 0 1; do
  seed=${SEEDS[$seed_index]}
  ckpt_root=${CKPT_ROOTS[$seed_index]}
  seed_dir="$OUT_ROOT/seed${seed}"
  mkdir -p "$seed_dir"
  for step in "${STEPS[@]}"; do
    step_dir="$seed_dir/step${step}"
    mkdir -p "$step_dir"
    if [[ "$step" == "init" ]]; then
      checkpoint=base
    else
      checkpoint="$ckpt_root/global_step_${step}"
    fi
    label="wm_b_full_seed${seed}_step${step}"
    echo "WM_FULL_CHECKPOINT_START seed=$seed step=$step checkpoint=$checkpoint"
    CKPT="$checkpoint" \
      CKPT_STEP="$step" \
      LABEL="$label" \
      DUMP_DIR="$step_dir" \
      MANIFEST="$MANIFEST" \
      MODEL="$MODEL" \
      ENV_SEED="$seed" \
      EXPECTED_GAMES="$EXPECTED_GAMES" \
      EXPECTED_RAW_TRAJECTORIES="$EXPECTED_RAW_TRAJECTORIES" \
      VAL_BATCH="$VAL_BATCH" \
      N_GPUS="$N_GPUS" \
      ROLLOUT_TP="$ROLLOUT_TP" \
      TEMP="$TEMP" \
      TOP_P="$TOP_P" \
      TOP_K="$TOP_K" \
      DO_SAMPLE="$DO_SAMPLE" \
      PREPARE_PYTHON="$PREPARE_PYTHON" \
      CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
      bash "$ROLLOUT_SCRIPT"

    mapfile -t dumps < <(find "$step_dir" -maxdepth 1 -type f -name '*.wm_transitions.jsonl' -print | sort)
    if [[ ${#dumps[@]} -ne 1 || ! -s "$step_dir/coverage.json" ]]; then
      echo "Checkpoint rollout did not produce one dump plus coverage: seed=$seed step=$step" >&2
      exit 1
    fi
    dump=${dumps[0]}
    "$PYTHON" "$SCORER" \
      --model-path "$MODEL" \
      --transition-jsonl "$dump" \
      --output-csv "$step_dir/scores.csv" \
      --summary-json "$step_dir/score_summary.json" \
      --max-length "$MAX_LENGTH" \
      --batch-size "$BATCH_SIZE" \
      --device "$DEVICE" \
      --dtype "$DTYPE" \
      --skip-entropy \
      --checkpoint "${label}=$checkpoint" \
      --require-full-protocol \
      --manifest "$MANIFEST" \
      --expected-checkpoint-step "$step" \
      --rollout-temperature "$TEMP" \
      --rollout-top-p "$TOP_P" \
      --rollout-top-k "$TOP_K" \
      --rollout-do-sample "$DO_SAMPLE" \
      --expected-games "$EXPECTED_GAMES" \
      --expected-raw-trajectories "$EXPECTED_RAW_TRAJECTORIES"
    test -s "$step_dir/scores.csv"
    test -s "$step_dir/score_summary.json"
    grep -q '"raw_cosine_only": true' "$step_dir/score_summary.json"
    echo "WM_FULL_CHECKPOINT_DONE seed=$seed step=$step dump=$dump"
  done

  analysis_dir="$seed_dir/analysis"
  "$PYTHON" "$ANALYZER" \
    --dump-root "$seed_dir" \
    --manifest "$MANIFEST" \
    --exp "baseline_seed${seed}" \
    --out-dir "$analysis_dir" \
    --bootstrap "$BOOTSTRAP" \
    --gmm-bootstrap "$GMM_BOOTSTRAP" \
    --seed "$seed"
  test -s "$analysis_dir/workstream_b_report_baseline_seed${seed}.md"

  if [[ "$RUN_PROBE" == "1" ]]; then
    "$PYTHON" "$PROBE" \
      --model-path "$MODEL" \
      --ckpt-root "$ckpt_root" \
      --dump-root "$seed_dir" \
      --manifest "$MANIFEST" \
      --exp "baseline_seed${seed}" \
      --out-csv "$analysis_dir/grouped_nested_hidden_probe.csv" \
      --device "$DEVICE" \
      --dtype "$DTYPE" \
      --batch-size "$PROBE_BATCH_SIZE" \
      --max-length "$MAX_LENGTH" \
      --fit-device "$DEVICE" \
      --seed "$seed"
    test -s "$analysis_dir/grouped_nested_hidden_probe.csv"
  fi
done

cross_seed_csv="$OUT_ROOT/cross_seed_trend_consistency.csv"
cross_seed_report="$OUT_ROOT/cross_seed_trend_consistency.md"
"$PYTHON" "$CROSS_SEED_TOOL" \
  --seed0-trends "$OUT_ROOT/seed0/analysis/paired_game_trends_baseline_seed0.csv" \
  --seed1-trends "$OUT_ROOT/seed1/analysis/paired_game_trends_baseline_seed1.csv" \
  --output-csv "$cross_seed_csv" \
  --output-report "$cross_seed_report"
test -s "$cross_seed_csv"
test -s "$cross_seed_report"

"$PYTHON" - \
  "$OUT_ROOT" \
  "$MANIFEST" \
  "$RUN_PROBE" \
  "$STAGE_CHECKPOINTS" \
  "$STAGE_ROOT" \
  "$SEED0_ARCHIVE_ROOT" \
  "$SEED1_ARCHIVE_ROOT" \
  "$EXPECTED_RAW_TRAJECTORIES" \
  "$EXPECTED_GAMES" <<'PY'
import json
import os
import sys
from pathlib import Path

out_root = Path(sys.argv[1]).resolve()
manifest = Path(sys.argv[2]).resolve()
run_probe = sys.argv[3] == "1"
stage_checkpoints = sys.argv[4] == "1"
stage_root = sys.argv[5]
seed0_archive_root = sys.argv[6]
seed1_archive_root = sys.argv[7]
expected_raw_trajectories = int(sys.argv[8])
expected_games = int(sys.argv[9])
value = {
    "schema_version": "workstream_b_full_driver_v1",
    "status": "complete",
    "seeds": [0, 1],
    "steps": ["init", "15", "30", "45", "60", "75", "90", "105", "120", "135", "150"],
    "manifest": str(manifest),
    "expected_raw_trajectories": expected_raw_trajectories,
    "expected_games": expected_games,
    "checkpoint_staging_enabled": stage_checkpoints,
    "checkpoint_stage_root": stage_root if stage_checkpoints else None,
    "checkpoint_archive_roots": (
        {"seed0": seed0_archive_root, "seed1": seed1_archive_root}
        if stage_checkpoints
        else None
    ),
    "actor_inventory_dir": str(out_root / "preflight"),
    "seed_checkpoint_separation": str(
        out_root / "preflight" / "seed_checkpoint_separation.json"
    ),
    "cross_seed_trend_csv": str(out_root / "cross_seed_trend_consistency.csv"),
    "cross_seed_trend_report": str(out_root / "cross_seed_trend_consistency.md"),
    "probe_complete": run_probe,
}
report = out_root / "WORKSTREAM_B_FULL_REPORT.md"
report_lines = [
    "# Workstream B full-train checkpoint diagnostics",
    "",
    "Status: **complete**. This run rejects legacy shared/eight-trajectory dumps.",
    "",
    f"- ALFWorld train manifest: `{manifest}`",
    f"- Authoritative discovery: `{expected_raw_trajectories}` raw trajectories → `{expected_games}` filtered games",
    "- Seeds: `0`, `1`",
    "- Checkpoints per seed: `init, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150`",
    "- Each checkpoint was scored only on its own manifest-scheduled rollout.",
    "- Geometry is raw action-end ↔ observation-end cosine; no predictor is loaded.",
    f"- Actor shard inventories: `{out_root / 'preflight'}`",
    f"- Checkpoint staging: `{'enabled' if stage_checkpoints else 'pre-staged input'}`",
    f"- Cross-seed trend consistency CSV: `{out_root / 'cross_seed_trend_consistency.csv'}`",
    f"- Cross-seed trend report: `{out_root / 'cross_seed_trend_consistency.md'}`",
    "",
    "## Seed reports",
    "",
]
for seed in (0, 1):
    report_lines.append(
        f"- Seed {seed}: `{out_root / f'seed{seed}' / 'analysis' / f'workstream_b_report_baseline_seed{seed}.md'}`"
    )
    if run_probe:
        report_lines.append(
            f"- Seed {seed} grouped/nested probes: "
            f"`{out_root / f'seed{seed}' / 'analysis' / 'grouped_nested_hidden_probe.csv'}`"
        )
report_staging = out_root / f".WORKSTREAM_B_FULL_REPORT.{os.getpid()}.staging"
report_staging.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
os.replace(report_staging, report)
if not report.is_file() or report.stat().st_size == 0:
    raise RuntimeError(f"Final report was not published: {report}")
path = out_root / "FULL_PROTOCOL_DONE.json"
staging = out_root / f".FULL_PROTOCOL_DONE.{os.getpid()}.staging"
staging.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(staging, path)
PY
test -s "$OUT_ROOT/WORKSTREAM_B_FULL_REPORT.md"
test -s "$OUT_ROOT/FULL_PROTOCOL_DONE.json"
echo "WM_FULL_TRAIN_DIAGNOSTICS_DONE out_root=$OUT_ROOT seeds=0,1 steps=${STEPS[*]}"
