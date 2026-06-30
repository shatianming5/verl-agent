#!/usr/bin/env python3
"""Build a compact report from world-model checkpoint diagnostic summaries."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import sys
from pathlib import Path
from typing import Any


NUMERIC_COLUMNS = [
    "token_mean_ce",
    "delta_token_mean_ce",
    "row_mean_target_confidence",
    "delta_row_mean_target_confidence",
    "row_mean_target_entropy",
    "row_mean_action_obs_cosine",
    "delta_row_mean_action_obs_cosine",
    "success_token_mean_ce",
    "failure_token_mean_ce",
    "success_failure_ce_gap",
    "success_action_obs_cosine",
    "failure_action_obs_cosine",
    "success_failure_cosine_gap",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-json",
        action="append",
        required=True,
        help="Path to a checkpoint_scores_summary.json produced by wm_score_transition_dump.py. Repeat for multiple runs.",
    )
    parser.add_argument("--output-md", help="Markdown report path. Prints to stdout if no output path is provided.")
    parser.add_argument("--output-csv", help="Machine-readable comparison table path.")
    parser.add_argument("--output-svg", help="Self-contained SVG plot path for CE and cosine checkpoint dynamics.")
    parser.add_argument(
        "--baseline-label",
        help="Checkpoint label to use for deltas. Defaults to the earliest checkpoint in each summary.",
    )
    return parser.parse_args()


def load_summary(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        summary = json.load(handle)
    if not isinstance(summary.get("checkpoints"), list):
        raise ValueError(f"{path} does not contain a checkpoints list")
    if not summary["checkpoints"]:
        raise ValueError(f"{path} contains no checkpoint summaries")
    summary["_summary_path"] = path
    return summary


def coerce_float(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    return None


def checkpoint_sort_key(row: dict[str, Any]) -> tuple[int, int, str]:
    step = str(row.get("checkpoint_step", "")).strip().lower()
    label = str(row.get("checkpoint_label", ""))
    if step in {"init", "base"} or "init" in label.lower():
        return (0, -1, label)
    try:
        return (1, int(step), label)
    except ValueError:
        return (2, 0, label)


def bucket_lookup(summary: dict[str, Any]) -> dict[tuple[str, bool], dict[str, Any]]:
    buckets = {}
    for bucket in summary.get("success_buckets", []):
        label = str(bucket.get("checkpoint_label", ""))
        success = coerce_bool(bucket.get("episode_success"))
        if label and success is not None:
            buckets[(label, success)] = bucket
    return buckets


def diagnostic_provenance_fields(summary: dict[str, Any]) -> dict[str, Any]:
    provenance = summary.get("provenance")
    if not isinstance(provenance, dict):
        return {}
    checkpoints = provenance.get("checkpoints", [])
    checkpoint_count = provenance.get("checkpoint_count", "")
    if checkpoint_count == "" and isinstance(checkpoints, list):
        checkpoint_count = len(checkpoints)
    return {
        "diagnostic_command": provenance.get("command", ""),
        "diagnostic_model_path": provenance.get("model_path", ""),
        "diagnostic_output_csv_path": provenance.get("output_csv", ""),
        "diagnostic_checkpoint_count": checkpoint_count,
        "diagnostic_max_length": provenance.get("max_length", ""),
        "diagnostic_batch_size": provenance.get("batch_size", ""),
        "diagnostic_max_rows": provenance.get("max_rows", ""),
        "diagnostic_device": provenance.get("device", ""),
        "diagnostic_dtype": provenance.get("dtype", ""),
        "diagnostic_skip_entropy": provenance.get("skip_entropy", ""),
    }


def choose_baseline(checkpoints: list[dict[str, Any]], baseline_label: str | None) -> dict[str, Any]:
    if baseline_label:
        for checkpoint in checkpoints:
            if str(checkpoint.get("checkpoint_label", "")) == baseline_label:
                return checkpoint
        raise ValueError(f"baseline label {baseline_label!r} was not found")
    return sorted(checkpoints, key=checkpoint_sort_key)[0]


def add_delta(row: dict[str, Any], key: str, baseline: dict[str, Any]) -> None:
    value = coerce_float(row.get(key))
    base_value = coerce_float(baseline.get(key))
    row[f"delta_{key}"] = "" if value is None or base_value is None else value - base_value


def build_rows(summary: dict[str, Any], baseline_label: str | None = None) -> list[dict[str, Any]]:
    checkpoints = sorted(summary["checkpoints"], key=checkpoint_sort_key)
    baseline = choose_baseline(checkpoints, baseline_label)
    buckets = bucket_lookup(summary)
    run_name = Path(str(summary.get("_summary_path", "summary"))).parent.name or "summary"
    provenance_fields = diagnostic_provenance_fields(summary)

    rows = []
    for checkpoint in checkpoints:
        label = str(checkpoint.get("checkpoint_label", ""))
        success_bucket = buckets.get((label, True), {})
        failure_bucket = buckets.get((label, False), {})
        row = {
            "run": run_name,
            "checkpoint_label": label,
            "checkpoint_step": checkpoint.get("checkpoint_step", ""),
            "checkpoint_path": checkpoint.get("checkpoint_path", ""),
            "rows": checkpoint.get("rows", ""),
            "rows_with_targets": checkpoint.get("rows_with_targets", ""),
            "target_tokens": checkpoint.get("target_tokens", ""),
            "token_mean_ce": checkpoint.get("token_mean_ce", ""),
            "row_mean_target_confidence": checkpoint.get("row_mean_target_confidence", ""),
            "row_mean_target_entropy": checkpoint.get("row_mean_target_entropy", ""),
            "row_mean_action_obs_cosine": checkpoint.get("row_mean_action_obs_cosine", ""),
            "success_token_mean_ce": success_bucket.get("token_mean_ce", ""),
            "failure_token_mean_ce": failure_bucket.get("token_mean_ce", ""),
            "success_action_obs_cosine": success_bucket.get("row_mean_action_obs_cosine", ""),
            "failure_action_obs_cosine": failure_bucket.get("row_mean_action_obs_cosine", ""),
            **provenance_fields,
        }
        add_delta(row, "token_mean_ce", baseline)
        add_delta(row, "row_mean_target_confidence", baseline)
        add_delta(row, "row_mean_action_obs_cosine", baseline)

        success_ce = coerce_float(row["success_token_mean_ce"])
        failure_ce = coerce_float(row["failure_token_mean_ce"])
        row["success_failure_ce_gap"] = "" if success_ce is None or failure_ce is None else failure_ce - success_ce
        success_cosine = coerce_float(row["success_action_obs_cosine"])
        failure_cosine = coerce_float(row["failure_action_obs_cosine"])
        row["success_failure_cosine_gap"] = "" if success_cosine is None or failure_cosine is None else success_cosine - failure_cosine
        rows.append(row)
    return rows


def format_cell(value: Any, key: str) -> str:
    if value in ("", None):
        return ""
    number = coerce_float(value)
    if key in NUMERIC_COLUMNS and number is not None:
        return f"{number:.4f}"
    return str(value)


def render_markdown(summaries: list[dict[str, Any]], rows_by_summary: list[list[dict[str, Any]]]) -> str:
    lines = ["# World-Model Checkpoint Diagnostics", ""]
    for summary, rows in zip(summaries, rows_by_summary):
        summary_path = summary.get("_summary_path", "")
        lines.append(f"## {Path(str(summary_path)).parent.name or Path(str(summary_path)).name}")
        lines.append("")
        lines.append(f"- Summary JSON: `{summary_path}`")
        if summary.get("transition_jsonl"):
            lines.append(f"- Transition JSONL: `{summary['transition_jsonl']}`")
        if summary.get("rows") != "":
            lines.append(f"- Rows: `{summary['rows']}`")
        if summary.get("max_length") != "":
            lines.append(f"- Max length: `{summary['max_length']}`")
        provenance = summary.get("provenance")
        if isinstance(provenance, dict):
            for key, title in (
                ("command", "Diagnostic command"),
                ("model_path", "Model path"),
                ("output_csv", "Per-transition CSV"),
                ("device", "Device"),
                ("dtype", "Dtype"),
                ("batch_size", "Batch size"),
                ("max_rows", "Max rows"),
            ):
                value = provenance.get(key, "")
                if value != "":
                    lines.append(f"- {title}: `{value}`")
            checkpoints = provenance.get("checkpoints", [])
            if isinstance(checkpoints, list) and checkpoints:
                checkpoint_labels = ", ".join(
                    str(item.get("label") or item.get("path") or "") for item in checkpoints if isinstance(item, dict)
                )
                lines.append(f"- Scored checkpoints: `{len(checkpoints)}` ({checkpoint_labels})")
        lines.append("")

        columns = [
            ("checkpoint_label", "checkpoint"),
            ("checkpoint_step", "step"),
            ("target_tokens", "target tokens"),
            ("token_mean_ce", "CE"),
            ("delta_token_mean_ce", "delta CE"),
            ("row_mean_target_confidence", "confidence"),
            ("row_mean_action_obs_cosine", "cosine"),
            ("delta_row_mean_action_obs_cosine", "delta cosine"),
            ("success_failure_ce_gap", "failure-success CE"),
            ("success_failure_cosine_gap", "success-failure cosine"),
        ]
        lines.append("| " + " | ".join(title for _, title in columns) + " |")
        lines.append("| " + " | ".join("---" for _ in columns) + " |")
        for row in rows:
            lines.append("| " + " | ".join(format_cell(row.get(key, ""), key) for key, _ in columns) + " |")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _plot_points(rows: list[dict[str, Any]], key: str, x: int, y: int, width: int, height: int) -> tuple[list[tuple[float, float, str]], str, str]:
    values = [coerce_float(row.get(key)) for row in rows]
    finite = [value for value in values if value is not None]
    if not finite:
        return [], "", ""
    min_value = min(finite)
    max_value = max(finite)
    if math.isclose(min_value, max_value):
        pad = max(abs(min_value) * 0.05, 0.1)
        min_value -= pad
        max_value += pad
    points = []
    for idx, (row, value) in enumerate(zip(rows, values)):
        if value is None:
            continue
        px = x + (width * idx / max(len(rows) - 1, 1))
        py = y + height - ((value - min_value) / (max_value - min_value) * height)
        points.append((px, py, str(row.get("checkpoint_step") or row.get("checkpoint_label") or idx)))
    return points, f"{min_value:.3f}", f"{max_value:.3f}"


def _polyline(points: list[tuple[float, float, str]]) -> str:
    return " ".join(f"{px:.1f},{py:.1f}" for px, py, _ in points)


def _render_panel(rows: list[dict[str, Any]], key: str, title: str, color: str, x: int, y: int, width: int, height: int) -> list[str]:
    points, min_label, max_label = _plot_points(rows, key, x, y, width, height)
    lines = [
        f'<text x="{x}" y="{y - 16}" class="panel-title">{html.escape(title)}</text>',
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" class="plot-bg"/>',
        f'<line x1="{x}" y1="{y + height}" x2="{x + width}" y2="{y + height}" class="axis"/>',
        f'<line x1="{x}" y1="{y}" x2="{x}" y2="{y + height}" class="axis"/>',
    ]
    if not points:
        lines.append(f'<text x="{x + width / 2:.1f}" y="{y + height / 2:.1f}" text-anchor="middle" class="empty">no numeric data</text>')
        return lines
    lines.extend(
        [
            f'<text x="{x - 10}" y="{y + 4}" text-anchor="end" class="tick">{max_label}</text>',
            f'<text x="{x - 10}" y="{y + height}" text-anchor="end" class="tick">{min_label}</text>',
            f'<polyline points="{_polyline(points)}" fill="none" stroke="{color}" stroke-width="2.5"/>',
        ]
    )
    for px, py, label in points:
        lines.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4" fill="{color}"/>')
        lines.append(f'<text x="{px:.1f}" y="{y + height + 18}" text-anchor="middle" class="tick">{html.escape(label)}</text>')
    return lines


def render_svg(summaries: list[dict[str, Any]], rows_by_summary: list[list[dict[str, Any]]]) -> str:
    width = 920
    run_height = 390
    height = 40 + run_height * len(rows_by_summary)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; fill: #111827; }",
        ".title { font-size: 18px; font-weight: 700; }",
        ".panel-title { font-size: 13px; font-weight: 650; }",
        ".tick { font-size: 11px; fill: #4b5563; }",
        ".empty { font-size: 12px; fill: #6b7280; }",
        ".plot-bg { fill: #f9fafb; stroke: #d1d5db; }",
        ".axis { stroke: #6b7280; stroke-width: 1; }",
        "</style>",
        '<text x="30" y="26" class="title">World-Model Checkpoint Diagnostics</text>',
    ]
    for idx, (summary, rows) in enumerate(zip(summaries, rows_by_summary)):
        y0 = 60 + idx * run_height
        run_name = Path(str(summary.get("_summary_path", "summary"))).parent.name or "summary"
        lines.append(f'<text x="30" y="{y0 - 12}" class="panel-title">{html.escape(run_name)}</text>')
        lines.extend(_render_panel(rows, "token_mean_ce", "token_mean_ce (lower is better)", "#2563eb", 85, y0, 780, 120))
        lines.extend(_render_panel(rows, "row_mean_action_obs_cosine", "action_obs_cosine (higher is better)", "#16a34a", 85, y0 + 190, 780, 120))
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    fieldnames = []
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
    summaries = [load_summary(path) for path in args.summary_json]
    rows_by_summary = [build_rows(summary, baseline_label=args.baseline_label) for summary in summaries]
    all_rows = [row for rows in rows_by_summary for row in rows]
    markdown = render_markdown(summaries, rows_by_summary)

    if args.output_csv:
        write_csv(args.output_csv, all_rows)
    if args.output_md:
        write_text(args.output_md, markdown)
    if args.output_svg:
        write_text(args.output_svg, render_svg(summaries, rows_by_summary))
    if not args.output_csv and not args.output_md and not args.output_svg:
        sys.stdout.write(markdown)


if __name__ == "__main__":
    main()
