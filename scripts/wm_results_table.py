#!/usr/bin/env python3
"""Aggregate ALFWorld world-model run artifacts into final result tables."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
import shlex
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_WORK = "/mnt/cephfs_home_tianming.sha/grpo_alfworld"
DEFAULT_EVAL_SCRIPT = "/root/grpo/eval10x_alfworld.sh"
RUN_PREFIX_RE = re.compile(r"grpo_qwen2\.5_1\.5b_alfworld_(.+?)(?:_\d{8}_\d{6})?(?:\.log)?$")

CSV_COLUMNS = [
    "run_key",
    "method",
    "objective",
    "tag",
    "seed",
    "lambda_obs",
    "lambda_latent",
    "eval_mean",
    "eval_std",
    "eval_n",
    "eval_readiness",
    "eval_command",
    "eval_target_checkpoint_path",
    "eval_result_path",
    "eval_checkpoint_path",
    "train_step",
    "train_total_steps",
    "val_success_last",
    "val_success_best",
    "wm_loss_last",
    "wm_cosine_last",
    "wm_metric_last",
    "latest_checkpoint_step",
    "latest_checkpoint_path",
    "diagnostic_final_step",
    "diagnostic_token_mean_ce",
    "diagnostic_delta_token_mean_ce",
    "diagnostic_action_obs_cosine",
    "diagnostic_delta_action_obs_cosine",
    "diagnostic_success_failure_ce_gap",
    "diagnostic_success_failure_cosine_gap",
    "diagnostic_rows",
    "diagnostic_init_ce",
    "diagnostic_final_ce",
    "diagnostic_final_cosine",
    "diagnostic_best_step",
    "diagnostic_report_md_path",
    "diagnostic_summary_path",
    "train_log_path",
    "launch_line",
    "command_summary",
    "eval_start_line",
    "status",
]

NUMERIC_COLUMNS = {
    "eval_mean",
    "eval_std",
    "val_success_last",
    "val_success_best",
    "wm_loss_last",
    "wm_cosine_last",
    "diagnostic_token_mean_ce",
    "diagnostic_delta_token_mean_ce",
    "diagnostic_action_obs_cosine",
    "diagnostic_delta_action_obs_cosine",
    "diagnostic_success_failure_ce_gap",
    "diagnostic_success_failure_cosine_gap",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-result", action="append", default=[], help="Path to eval10x_*_results.txt. Repeat as needed.")
    parser.add_argument("--eval-glob", action="append", default=[], help="Glob for eval10x result files.")
    parser.add_argument("--train-log", action="append", default=[], help="Path to a training log. Repeat as needed.")
    parser.add_argument("--run-log", action="append", dest="train_log", default=argparse.SUPPRESS, help="Alias for --train-log.")
    parser.add_argument("--train-log-glob", action="append", default=[], help="Glob for training logs.")
    parser.add_argument("--run-log-glob", action="append", dest="train_log_glob", default=argparse.SUPPRESS, help="Alias for --train-log-glob.")
    parser.add_argument(
        "--diagnostic-summary",
        action="append",
        default=[],
        help="Path to checkpoint_scores_summary.json from wm_score_transition_dump.py. Repeat as needed.",
    )
    parser.add_argument("--diagnostic-glob", action="append", default=[], help="Glob for checkpoint diagnostic summaries.")
    parser.add_argument("--work-root", default=DEFAULT_WORK, help="Remote work root used in example commands.")
    parser.add_argument(
        "--discover-standard-layout",
        action="store_true",
        help="Auto-discover standard artifacts under --work-root/logs and add them to explicit inputs.",
    )
    parser.add_argument("--branch", default=None, help="Branch name to show in the report. Defaults to current git branch if available.")
    parser.add_argument("--eval-cuda", default="<free_2gpu_pair>", help="CUDA_VISIBLE_DEVICES value to use in generated eval commands.")
    parser.add_argument("--eval-n", default="10", help="N_EVALS value to use in generated eval commands.")
    parser.add_argument(
        "--eval-script",
        default=DEFAULT_EVAL_SCRIPT,
        help="Eval script path to use in generated eval commands.",
    )
    parser.add_argument("--output-md", help="Markdown report path. Prints to stdout if no output path is provided.")
    parser.add_argument("--output-csv", help="Machine-readable table path.")
    return parser.parse_args()


def read_text(path: str) -> str:
    with open(path, encoding="utf-8", errors="replace") as handle:
        return handle.read()


def expand_paths(paths: list[str], patterns: list[str]) -> list[str]:
    expanded = []
    for path in paths:
        matches = glob.glob(path, recursive=True) if glob.has_magic(path) else []
        expanded.extend(matches or [path])
    for pattern in patterns:
        expanded.extend(glob.glob(pattern, recursive=True))
    return sorted(dict.fromkeys(str(Path(path)) for path in expanded))


def standard_layout_globs(work_root: str) -> tuple[list[str], list[str], list[str]]:
    logs = Path(work_root) / "logs"
    return (
        [str(logs / "eval10x_*_results.txt")],
        [
            str(logs / "grpo_qwen2.5_1.5b_alfworld_seed*.log"),
            str(logs / "*wm_obs*.log"),
            str(logs / "*wmlat*.log"),
            str(logs / "*latent*.log"),
        ],
        [str(logs / "world_model_diagnostics" / "**" / "checkpoint_scores_summary.json")],
    )


def discover_standard_layout_paths(work_root: str) -> tuple[list[str], list[str], list[str]]:
    eval_globs, train_log_globs, diagnostic_globs = standard_layout_globs(work_root)
    return (
        exclude_smoke_paths(expand_paths([], eval_globs)),
        exclude_smoke_paths(expand_paths([], train_log_globs)),
        exclude_smoke_paths(expand_paths([], diagnostic_globs)),
    )


def exclude_smoke_paths(paths: list[str]) -> list[str]:
    return [path for path in paths if "smoke" not in str(Path(path)).lower()]


def coerce_float(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def coerce_int(value: Any) -> int | None:
    if value in ("", None):
        return None
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def format_number(value: Any, digits: int = 4) -> str:
    number = coerce_float(value)
    if number is None:
        return ""
    return f"{number:.{digits}f}"


def decimal_from_l_token(token: str) -> str:
    match = re.fullmatch(r"l(\d+)p(\d+)", token)
    if not match:
        return ""
    left, right = match.groups()
    return f"{int(left)}.{right}"


def lambda_label(value: str) -> str:
    number = coerce_float(value)
    if number is None:
        return ""
    text = f"{number:g}"
    if "e" in text or "E" in text:
        text = f"{number:.8f}".rstrip("0").rstrip(".")
    return "l" + text.replace(".", "p")


def run_fragment_from_path(path: str) -> str:
    path_obj = Path(path)
    name = path_obj.name
    if name.startswith("eval10x_") and name.endswith("_results.txt"):
        return name[len("eval10x_") : -len("_results.txt")]
    match = RUN_PREFIX_RE.search(name)
    if match:
        return match.group(1)
    if name == "checkpoint_scores_summary.json":
        return path_obj.parent.name
    return path_obj.stem


def normalize_fragment(fragment: str) -> str:
    normalized = fragment.strip()
    normalized = re.sub(r"^grpo_qwen2\.5_1\.5b_alfworld_", "", normalized)
    normalized = re.sub(r"_\d{8}_\d{6}$", "", normalized)
    normalized = normalized.replace("wm_latent", "wmlat")
    normalized = normalized.replace("latent_hidden", "latent")
    return normalized


def infer_seed(text: str) -> str:
    for pattern in (
        r"(?:^|[^A-Za-z0-9])seed(?:=|_)?(\d+)(?:[^A-Za-z0-9]|$)",
        r"(?:^|[^A-Za-z0-9])s(\d+)(?:[^A-Za-z0-9]|$)",
    ):
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return ""


def infer_lambda(text: str, names: tuple[str, ...]) -> str:
    for name in names:
        match = re.search(rf"{re.escape(name)}\s*[=:]\s*([0-9]+(?:\.[0-9]+)?(?:e[-+]?\d+)?)", text, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return ""


def infer_l_token(text: str) -> str:
    for token in re.findall(r"(?:^|[_-])(l\d+p\d+)(?:[_-]|$)", text, flags=re.IGNORECASE):
        return decimal_from_l_token(token.lower())
    return ""


def infer_tag(text: str) -> str:
    tag_match = re.search(r"(?:^|\s)tag=([^ \t]+)", text)
    if tag_match:
        return tag_match.group(1)

    checkpoint_match = re.search(r"grpo_qwen2\.5_1\.5b_alfworld_seed\d+_([A-Za-z0-9_.-]+)", text)
    if checkpoint_match:
        tag = checkpoint_match.group(1)
        return re.sub(r"_(?:init|step\d+)$", "", tag)

    seed_match = re.search(r"(?:^|\s)seed\d+_([A-Za-z0-9_.-]+)", text)
    if seed_match:
        return seed_match.group(1)
    return ""


def infer_objective(text: str, lambda_obs: str = "", lambda_latent: str = "") -> str:
    lowered = text.lower()
    if coerce_float(lambda_latent) and coerce_float(lambda_latent) > 0:
        return "latent"
    if coerce_float(lambda_obs) and coerce_float(lambda_obs) > 0:
        return "obs_ce"
    if any(token in lowered for token in ("wmlat", "latent")):
        return "latent"
    if any(token in lowered for token in ("obs_ce", "wm_obs", "lambda_obs")):
        return "obs_ce"
    if any(token in lowered for token in ("official", "baseline", "seed")):
        return "grpo_baseline"
    return "unknown"


def method_from_objective(objective: str) -> str:
    return {"grpo_baseline": "grpo", "obs_ce": "obs_ce", "latent": "latent"}.get(objective, objective)


def infer_metadata(fragment: str, text: str = "") -> dict[str, str]:
    combined = normalize_fragment(f"{fragment} {text}")
    seed = infer_seed(combined)
    lambda_obs = infer_lambda(combined, ("actor_rollout_ref.actor.world_model.lambda_obs", "lambda_obs", "obs_ce_coef"))
    lambda_latent = infer_lambda(
        combined,
        ("actor_rollout_ref.actor.world_model.lambda_latent", "lambda_latent", "latent_loss_coef"),
    )
    lowered = combined.lower()
    if not lambda_obs and any(token in lowered for token in ("obs_ce", "wm_obs", "lambda_obs")):
        lambda_obs = infer_l_token(combined)
    if not lambda_latent and any(token in lowered for token in ("wmlat", "latent", "lambda_latent")):
        lambda_latent = infer_l_token(combined)
    objective = infer_objective(combined, lambda_obs=lambda_obs, lambda_latent=lambda_latent)

    parts = [objective]
    if objective == "obs_ce" and lambda_obs:
        parts.append(lambda_label(lambda_obs))
    elif objective == "latent" and lambda_latent:
        parts.append(lambda_label(lambda_latent))
    if seed:
        parts.append(f"s{seed}")
    run_key = "_".join(part for part in parts if part and part != "unknown")
    if not run_key:
        run_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", normalize_fragment(fragment)).strip("_") or "unknown"
    tag = infer_tag(combined)
    return {
        "run_key": run_key,
        "method": method_from_objective(objective),
        "objective": objective,
        "tag": tag,
        "seed": seed,
        "lambda_obs": lambda_obs if objective == "obs_ce" else "",
        "lambda_latent": lambda_latent if objective == "latent" else "",
    }


def extract_metric_values(text: str, metric_name: str) -> list[float]:
    escaped = re.escape(metric_name)
    values = []
    for match in re.finditer(rf"['\"]?{escaped}['\"]?\s*[:=]\s*([-+]?[0-9]*\.?[0-9]+(?:e[-+]?\d+)?)", text, flags=re.IGNORECASE):
        value = coerce_float(match.group(1))
        if value is not None:
            values.append(value)
    return values


def parse_eval_result(path: str) -> dict[str, Any]:
    text = read_text(path)
    fragment = run_fragment_from_path(path)
    start_match = re.search(r"^EVAL10X_START\s+(.+)$", text, flags=re.MULTILINE)
    result_match = re.search(
        r"^EVAL10X_RESULT\s+n=(\d+)\s+mean=([-+]?[0-9]*\.?[0-9]+)\s+std=([-+]?[0-9]*\.?[0-9]+)",
        text,
        flags=re.MULTILINE,
    )
    start_line = start_match.group(0) if start_match else ""
    if start_match:
        label_match = re.search(r"(?:^|\s)label=(\S+)", start_match.group(1))
        if label_match:
            fragment = label_match.group(1)
    ckpt_match = re.search(r"(?:^|\s)ckpt=(\S+)", start_line)
    ckpt_path = ckpt_match.group(1) if ckpt_match else ""
    checkpoint_step_match = re.search(r"/global_step_(\d+)(?:/|$)", ckpt_path)
    meta = infer_metadata(fragment, f"{start_line} {ckpt_path}")
    row: dict[str, Any] = {
        **meta,
        "eval_result_path": path,
        "eval_start_line": start_line,
        "eval_checkpoint_path": ckpt_path,
        "checkpoint_step": checkpoint_step_match.group(1) if checkpoint_step_match else "",
    }
    start_values = parse_key_values(start_line)
    if start_values.get("dataset"):
        row["eval_dataset"] = start_values["dataset"]
    if start_values.get("val_size"):
        row["eval_val_size"] = start_values["val_size"]
    if result_match:
        row.update(
            {
                "eval_n": result_match.group(1),
                "eval_mean": coerce_float(result_match.group(2)),
                "eval_std": coerce_float(result_match.group(3)),
            }
        )
        row["status"] = "evaluated"
    else:
        success_values = extract_metric_values(text, "success_rate")
        row["eval_n"] = str(len(success_values)) if success_values else ""
        row["status"] = "eval_incomplete"
    return row


def parse_key_values(line: str) -> dict[str, str]:
    result = {}
    for key, value in re.findall(r"([A-Za-z_][A-Za-z0-9_./-]*)=([^ \t]+)", line):
        result[key] = value
    return result


def parse_train_log(path: str) -> dict[str, Any]:
    text = read_text(path)
    fragment = run_fragment_from_path(path)
    launch_lines = re.findall(r"^(RUN_[A-Z0-9_]+.+)$", text, flags=re.MULTILINE)
    launch_line = " ; ".join(launch_lines)
    values: dict[str, str] = {}
    for line in launch_lines:
        values.update(parse_key_values(line))
    tag = values.get("tag", "")
    if values.get("seed") and tag:
        fragment = f"seed{values['seed']}_{tag}"

    meta = infer_metadata(fragment, text)
    progress = [(int(step), int(total)) for step, total in re.findall(r"Training Progress:[^\r\n]*(\d+)/(\d+)", text)]
    checkpoint_steps = [int(step) for step in re.findall(r"global_step_(\d+)", text)]
    logged_steps = [int(value) for value in extract_metric_values(text, "training/global_step")]
    line_steps = [int(step) for step in re.findall(r"(?:^|\s)step:(\d+)", text)]
    val_values = extract_metric_values(text, "val/success_rate")
    latest_step = max([step for step, _ in progress] + checkpoint_steps + logged_steps + line_steps, default=None)
    latest_ckpt_step = max(checkpoint_steps, default=None)
    total_steps = max([total for _, total in progress], default="")
    ckpt_path = values.get("ckpt", "")
    if ckpt_path and latest_ckpt_step is not None:
        ckpt_path = str(Path(ckpt_path) / f"global_step_{latest_ckpt_step}")

    row: dict[str, Any] = {
        **meta,
        "train_log_path": path,
        "launch_line": launch_line,
        "command_summary": launch_line,
        "train_cuda": values.get("cuda", ""),
        "rollout_data_dir": values.get("rollout_data_dir", ""),
        "train_step": "" if latest_step is None else latest_step,
        "train_total_steps": total_steps,
        "latest_checkpoint_step": "" if latest_ckpt_step is None else latest_ckpt_step,
        "latest_checkpoint_path": ckpt_path,
        "val_success_last": val_values[-1] if val_values else "",
        "val_success_best": max(val_values) if val_values else "",
        "status": "training_complete"
        if any(step >= total for step, total in progress)
        or (latest_step is not None and isinstance(total_steps, int) and latest_step >= total_steps)
        else "training_seen",
    }
    for metric in (
        "actor/wm_obs_ce_loss",
        "actor/wm_obs_ce_tokens",
        "actor/wm_latent_loss",
        "actor/wm_cosine",
        "actor/wm_grad_norm",
        "world_model/obs_ce_loss",
        "world_model/obs_ce_tokens",
        "world_model/latent_loss",
        "world_model/latent_cosine",
        "world_model/latent_rows",
        "world_model/latent_action_feature_var",
        "world_model/latent_obs_feature_var",
    ):
        values_for_metric = extract_metric_values(text, metric)
        if values_for_metric:
            row[metric] = values_for_metric[-1]
    if row.get("world_model/obs_ce_loss") not in ("", None):
        row["wm_loss_last"] = row["world_model/obs_ce_loss"]
        row["wm_metric_last"] = f"obs_ce_loss={format_number(row['wm_loss_last'], digits=3)}"
    elif row.get("actor/wm_obs_ce_loss") not in ("", None):
        row["wm_loss_last"] = row["actor/wm_obs_ce_loss"]
        row["wm_metric_last"] = f"obs_ce_loss={format_number(row['wm_loss_last'], digits=3)}"
    if row.get("world_model/latent_loss") not in ("", None):
        row["wm_loss_last"] = row["world_model/latent_loss"]
    elif row.get("actor/wm_latent_loss") not in ("", None):
        row["wm_loss_last"] = row["actor/wm_latent_loss"]
    if row.get("world_model/latent_cosine") not in ("", None):
        row["wm_cosine_last"] = row["world_model/latent_cosine"]
    elif row.get("actor/wm_cosine") not in ("", None):
        row["wm_cosine_last"] = row["actor/wm_cosine"]
    if row.get("wm_cosine_last") not in ("", None):
        row["wm_metric_last"] = (
            f"latent_loss={format_number(row.get('wm_loss_last'), digits=3)}, "
            f"cosine={format_number(row.get('wm_cosine_last'), digits=3)}"
        )
    return row


parse_run_log = parse_train_log


def checkpoint_sort_key(row: dict[str, Any]) -> tuple[int, int, str]:
    step = str(row.get("checkpoint_step", "")).strip().lower()
    label = str(row.get("checkpoint_label", ""))
    if step in {"init", "base"} or "init" in label.lower():
        return (0, -1, label)
    try:
        return (1, int(step), label)
    except ValueError:
        return (2, 0, label)


def choose_final_checkpoint(checkpoints: list[dict[str, Any]]) -> dict[str, Any]:
    sorted_checkpoints = sorted(checkpoints, key=checkpoint_sort_key)
    for checkpoint in reversed(sorted_checkpoints):
        if str(checkpoint.get("checkpoint_step", "")) == "150":
            return checkpoint
    return sorted_checkpoints[-1]


def success_bucket(summary: dict[str, Any], label: str, success: bool) -> dict[str, Any]:
    for bucket in summary.get("success_buckets", []):
        if str(bucket.get("checkpoint_label", "")) == label and bucket.get("episode_success") is success:
            return bucket
    return {}


def numeric_delta(row: dict[str, Any], baseline: dict[str, Any], key: str) -> str:
    value = coerce_float(row.get(key))
    base = coerce_float(baseline.get(key))
    if value is None or base is None:
        return ""
    return f"{value - base:.10g}"


def parse_diagnostic_summary(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        summary = json.load(handle)
    checkpoints = summary.get("checkpoints", [])
    if not isinstance(checkpoints, list) or not checkpoints:
        raise ValueError(f"{path} does not contain checkpoint summaries")

    fragment = run_fragment_from_path(path)
    transition_path = str(summary.get("transition_jsonl", ""))
    meta = infer_metadata(fragment, f"{transition_path} {json.dumps(summary)[:4000]}")
    baseline = sorted(checkpoints, key=checkpoint_sort_key)[0]
    final = choose_final_checkpoint(checkpoints)
    checkpoints_with_ce = [checkpoint for checkpoint in checkpoints if coerce_float(checkpoint.get("token_mean_ce")) is not None]
    best = min(checkpoints_with_ce, key=lambda checkpoint: coerce_float(checkpoint.get("token_mean_ce")), default={})
    final_label = str(final.get("checkpoint_label", ""))
    success = success_bucket(summary, final_label, True)
    failure = success_bucket(summary, final_label, False)

    success_ce = coerce_float(success.get("token_mean_ce"))
    failure_ce = coerce_float(failure.get("token_mean_ce"))
    success_cosine = coerce_float(success.get("row_mean_action_obs_cosine"))
    failure_cosine = coerce_float(failure.get("row_mean_action_obs_cosine"))

    return {
        **meta,
        "diagnostic_summary_path": path,
        "diagnostic_rows": summary.get("rows", ""),
        "diagnostic_init_ce": baseline.get("token_mean_ce", ""),
        "diagnostic_final_step": final.get("checkpoint_step", ""),
        "diagnostic_token_mean_ce": final.get("token_mean_ce", ""),
        "diagnostic_final_ce": final.get("token_mean_ce", ""),
        "diagnostic_delta_token_mean_ce": numeric_delta(final, baseline, "token_mean_ce"),
        "diagnostic_action_obs_cosine": final.get("row_mean_action_obs_cosine", ""),
        "diagnostic_final_cosine": final.get("row_mean_action_obs_cosine", ""),
        "diagnostic_delta_action_obs_cosine": numeric_delta(final, baseline, "row_mean_action_obs_cosine"),
        "diagnostic_best_step": best.get("checkpoint_step", ""),
        "diagnostic_report_md_path": str(Path(path).with_name("checkpoint_diagnostics_report.md")),
        "diagnostic_success_failure_ce_gap": ""
        if success_ce is None or failure_ce is None
        else f"{failure_ce - success_ce:.10g}",
        "diagnostic_success_failure_cosine_gap": ""
        if success_cosine is None or failure_cosine is None
        else f"{success_cosine - failure_cosine:.10g}",
        "status": "diagnosed",
    }


def empty_record(run_key: str) -> dict[str, Any]:
    return {column: "" for column in CSV_COLUMNS} | {"run_key": run_key}


def append_unique(existing: Any, value: Any) -> str:
    if value in ("", None):
        return "" if existing in ("", None) else str(existing)
    parts = [part for part in str(existing).split(";") if part] if existing not in ("", None) else []
    if str(value) not in parts:
        parts.append(str(value))
    return ";".join(parts)


def merge_record(records: dict[str, dict[str, Any]], row: dict[str, Any]) -> None:
    run_key = str(row.get("run_key") or "unknown")
    record = records.setdefault(run_key, empty_record(run_key))
    for key in ("method", "objective", "tag", "seed", "lambda_obs", "lambda_latent"):
        if not record.get(key) and row.get(key):
            record[key] = row[key]

    for key, value in row.items():
        if key in {"run_key", "method", "objective", "tag", "seed", "lambda_obs", "lambda_latent"} or value in ("", None):
            continue
        if key.endswith("_path") or key in {"train_log_path", "diagnostic_summary_path"}:
            record[key] = append_unique(record.get(key, ""), value)
        elif key in {"train_step", "train_total_steps", "latest_checkpoint_step", "eval_n", "diagnostic_final_step"}:
            current = coerce_int(record.get(key))
            incoming = coerce_int(value)
            if current is None or (incoming is not None and incoming > current):
                record[key] = value
        elif key == "val_success_best":
            current = coerce_float(record.get(key))
            incoming = coerce_float(value)
            if current is None or (incoming is not None and incoming > current):
                record[key] = value
        elif key in {"launch_line", "command_summary", "eval_start_line", "status"}:
            record[key] = append_unique(record.get(key, ""), value)
        else:
            record[key] = value


def objective_sort_key(row: dict[str, Any]) -> tuple[int, float, int, str]:
    order = {"grpo_baseline": 0, "obs_ce": 1, "latent": 2, "unknown": 9}
    objective = str(row.get("objective", "unknown"))
    lambda_value = coerce_float(row.get("lambda_obs") or row.get("lambda_latent")) or 0.0
    seed = coerce_int(row.get("seed")) or -1
    return (order.get(objective, 8), lambda_value, seed, str(row.get("run_key", "")))


def build_records(eval_paths: list[str], train_logs: list[str], diagnostic_paths: list[str]) -> list[dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for path in eval_paths:
        merge_record(records, parse_eval_result(path))
    for path in train_logs:
        merge_record(records, parse_train_log(path))
    for path in sorted(diagnostic_paths, key=lambda item: (0 if "smoke" in item.lower() else 1, item)):
        merge_record(records, parse_diagnostic_summary(path))
    return sorted(records.values(), key=objective_sort_key)


def build_eval_command(row: dict[str, Any], eval_cuda: str, eval_n: str, eval_script: str) -> str:
    checkpoint_path = str(row.get("latest_checkpoint_path") or "")
    if not checkpoint_path:
        return ""
    label = str(row.get("tag") or row.get("run_key") or "world_model_run")
    assignments = {
        "CKPT": checkpoint_path,
        "LABEL": label,
        "CUDA_VISIBLE_DEVICES": eval_cuda,
        "N_EVALS": str(eval_n),
    }
    prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in assignments.items())
    return f"{prefix} bash {shlex.quote(eval_script)}"


def annotate_eval_readiness(rows: list[dict[str, Any]], eval_cuda: str, eval_n: str, eval_script: str) -> None:
    for row in rows:
        row["eval_target_checkpoint_path"] = ""
        if coerce_float(row.get("eval_mean")) is not None:
            row["eval_readiness"] = "evaluated"
            row["eval_command"] = ""
            row["eval_target_checkpoint_path"] = str(row.get("eval_checkpoint_path") or "")
            continue
        if row.get("eval_result_path"):
            row["eval_readiness"] = "eval_incomplete"
            row["eval_command"] = ""
            row["eval_target_checkpoint_path"] = str(row.get("eval_checkpoint_path") or "")
            continue

        latest_checkpoint_step = coerce_int(row.get("latest_checkpoint_step"))
        latest_checkpoint_path = str(row.get("latest_checkpoint_path") or "")
        if latest_checkpoint_step is not None and latest_checkpoint_step >= 150 and latest_checkpoint_path:
            row["eval_readiness"] = "ready_for_eval"
            row["eval_command"] = build_eval_command(row, eval_cuda=eval_cuda, eval_n=eval_n, eval_script=eval_script)
            row["eval_target_checkpoint_path"] = latest_checkpoint_path
        elif latest_checkpoint_step is not None or row.get("train_log_path"):
            row["eval_readiness"] = "waiting_for_checkpoint"
            row["eval_command"] = ""
        else:
            row["eval_readiness"] = "missing_training_log"
            row["eval_command"] = ""


def git_branch() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return ""
    return result.stdout.strip()


def render_eval_cell(row: dict[str, Any]) -> str:
    mean = format_number(row.get("eval_mean"))
    std = format_number(row.get("eval_std"))
    eval_n = row.get("eval_n")
    if mean and std:
        suffix = f" (n={eval_n})" if eval_n not in ("", None) else ""
        return f"{mean} +/- {std}{suffix}"
    return mean


def format_checkpoint(row: dict[str, Any]) -> str:
    return str(row.get("latest_checkpoint_path") or row.get("eval_checkpoint_path") or "")


def render_markdown(rows: list[dict[str, Any]], branch: str = "", work_root: str = DEFAULT_WORK) -> str:
    lines = ["# ALFWorld World-Model Results", ""]
    if branch:
        lines.append(f"- Branch: `{branch}`")
    lines.append(f"- Work root: `{work_root}`")
    lines.append(f"- Runs in table: `{len(rows)}`")
    lines.append("")

    columns = [
        ("run_key", "run"),
        ("tag", "tag"),
        ("objective", "objective"),
        ("seed", "seed"),
        ("lambda_obs", "lambda_obs"),
        ("lambda_latent", "lambda_latent"),
        ("eval", "eval mean +/- std"),
        ("eval_n", "eval n"),
        ("eval_readiness", "eval readiness"),
        ("val_success_last", "last online val"),
        ("wm_metric_last", "last WM metric"),
        ("train_step", "train step"),
        ("diagnostic_token_mean_ce", "diag CE"),
        ("diagnostic_delta_token_mean_ce", "delta CE"),
        ("diagnostic_action_obs_cosine", "diag cosine"),
        ("diagnostic_delta_action_obs_cosine", "delta cosine"),
        ("status", "status"),
    ]
    lines.append("| " + " | ".join(title for _, title in columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        cells = []
        for key, _ in columns:
            if key == "eval":
                cells.append(render_eval_cell(row))
            elif key == "diagnostic_token_mean_ce":
                ce = format_number(row.get(key))
                cells.append(f"CE {ce}" if ce else "")
            elif key in NUMERIC_COLUMNS:
                cells.append(format_number(row.get(key)))
            else:
                cells.append(str(row.get(key, "")))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    readiness_counts = Counter(str(row.get("eval_readiness") or "unknown") for row in rows)
    if readiness_counts:
        lines.append("## Eval Readiness")
        lines.append("")
        for readiness, count in sorted(readiness_counts.items()):
            lines.append(f"- {readiness}: `{count}`")
        ready_rows = [row for row in rows if row.get("eval_readiness") == "ready_for_eval" and row.get("eval_command")]
        if ready_rows:
            lines.append("")
            lines.append("Ready eval commands:")
            lines.append("")
            for row in ready_rows:
                target = str(row.get("eval_target_checkpoint_path") or row.get("latest_checkpoint_path") or "")
                lines.append(f"- `{row.get('run_key', '')}` checkpoint `{target}`: `{row['eval_command']}`")
        lines.append("")

    lines.append("## Artifact Paths")
    lines.append("")
    for row in rows:
        lines.append(f"### {row.get('run_key', '')}")
        if row.get("tag"):
            lines.append(f"- Tag: `{row['tag']}`")
        for key, title in (
            ("eval_target_checkpoint_path", "Eval target checkpoint"),
            ("eval_result_path", "Eval result"),
            ("eval_checkpoint_path", "Eval checkpoint"),
            ("latest_checkpoint_path", "Latest checkpoint"),
            ("diagnostic_summary_path", "Diagnostic summary"),
            ("train_log_path", "Train log"),
        ):
            value = row.get(key, "")
            if value:
                lines.append(f"- {title}: `{value}`")
        if row.get("launch_line"):
            lines.append(f"- Launch line: `{row['launch_line']}`")
        if row.get("command_summary") and row.get("command_summary") != row.get("launch_line"):
            lines.append(f"- Command summary: `{row['command_summary']}`")
        if row.get("eval_start_line"):
            lines.append(f"- Eval line: `{row['eval_start_line']}`")
        if row.get("eval_command"):
            lines.append(f"- Eval command: `{row['eval_command']}`")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(CSV_COLUMNS)
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: str, text: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def main() -> None:
    args = parse_args()
    if args.discover_standard_layout:
        eval_paths, train_log_paths, diagnostic_paths = discover_standard_layout_paths(args.work_root)
        args.eval_result.extend(eval_paths)
        args.train_log.extend(train_log_paths)
        args.diagnostic_summary.extend(diagnostic_paths)
    eval_paths = expand_paths(args.eval_result, args.eval_glob)
    train_logs = expand_paths(args.train_log, args.train_log_glob)
    diagnostic_paths = expand_paths(args.diagnostic_summary, args.diagnostic_glob)
    rows = build_records(eval_paths, train_logs, diagnostic_paths)
    annotate_eval_readiness(rows, eval_cuda=args.eval_cuda, eval_n=args.eval_n, eval_script=args.eval_script)
    branch = args.branch if args.branch is not None else git_branch()
    markdown = render_markdown(rows, branch=branch, work_root=args.work_root)

    if args.output_csv:
        write_csv(args.output_csv, rows)
    if args.output_md:
        write_text(args.output_md, markdown)
    if not args.output_csv and not args.output_md:
        sys.stdout.write(markdown)


if __name__ == "__main__":
    main()
