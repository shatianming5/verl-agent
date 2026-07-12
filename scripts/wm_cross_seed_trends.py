#!/usr/bin/env python3
"""Compare paired-game checkpoint trends between baseline seeds 0 and 1."""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
from typing import Any

LEVELS = ("episode", "transition")
METRICS = (
    "ce",
    "nll",
    "perplexity",
    "target_confidence_mean",
    "raw_action_obs_cosine",
)
STATISTICS = ("mean_all", "mean_succ", "mean_fail", "gap")
EXPECTED_GAMES = 3553


def finite(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite: {value!r}")
    return result


def load_trends(path: str | os.PathLike[str]) -> dict[tuple[str, str, str], dict[str, Any]]:
    trend_path = Path(path)
    if not trend_path.is_file():
        raise FileNotFoundError(f"Missing paired-game trend CSV: {trend_path}")
    rows: dict[tuple[str, str, str], dict[str, Any]] = {}
    with trend_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row.get("level", ""), row.get("metric", ""), row.get("statistic", ""))
            if key in rows:
                raise ValueError(f"Duplicate trend row in {trend_path}: {key}")
            if key[0] not in LEVELS or key[1] not in METRICS or key[2] not in STATISTICS:
                raise ValueError(f"Unexpected trend row in {trend_path}: {key}")
            paired_games = int(row.get("paired_games") or 0)
            if paired_games != EXPECTED_GAMES:
                raise ValueError(f"Trend {key} has paired_games={paired_games}; expected exactly {EXPECTED_GAMES}")
            slope = finite(row.get("slope_per_step"), "slope_per_step")
            ci_lo = finite(row.get("slope_ci_lo"), "slope_ci_lo")
            ci_hi = finite(row.get("slope_ci_hi"), "slope_ci_hi")
            if ci_lo > ci_hi:
                raise ValueError(f"Trend CI is reversed for {key}")
            rows[key] = {
                "paired_games": paired_games,
                "slope": slope,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
            }
    expected = {(level, metric, statistic) for level in LEVELS for metric in METRICS for statistic in STATISTICS}
    if set(rows) != expected:
        missing = sorted(expected - set(rows))
        extra = sorted(set(rows) - expected)
        raise ValueError(f"Trend matrix is incomplete: missing={missing} extra={extra}")
    return rows


def slope_direction(value: float, tolerance: float = 1e-12) -> str:
    if value > tolerance:
        return "positive"
    if value < -tolerance:
        return "negative"
    return "flat"


def compare_trends(
    seed0: dict[tuple[str, str, str], dict[str, Any]],
    seed1: dict[tuple[str, str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    if set(seed0) != set(seed1):
        raise ValueError("Seed trend matrices differ")
    output = []
    for level in LEVELS:
        for metric in METRICS:
            for statistic in STATISTICS:
                key = (level, metric, statistic)
                first = seed0[key]
                second = seed1[key]
                direction0 = slope_direction(first["slope"])
                direction1 = slope_direction(second["slope"])
                direction_consistent = direction0 == direction1 and direction0 != "flat"
                ci_overlap = max(first["ci_lo"], second["ci_lo"]) <= min(first["ci_hi"], second["ci_hi"])
                seed0_excludes_zero = first["ci_lo"] > 0 or first["ci_hi"] < 0
                seed1_excludes_zero = second["ci_lo"] > 0 or second["ci_hi"] < 0
                positive_confirmed = first["slope"] > 0 and second["slope"] > 0 and first["ci_lo"] > 0 and second["ci_lo"] > 0
                negative_confirmed = first["slope"] < 0 and second["slope"] < 0 and first["ci_hi"] < 0 and second["ci_hi"] < 0
                confirmed = positive_confirmed or negative_confirmed
                output.append(
                    {
                        "level": level,
                        "metric": metric,
                        "statistic": statistic,
                        "paired_games_seed0": first["paired_games"],
                        "paired_games_seed1": second["paired_games"],
                        "seed0_slope_per_step": first["slope"],
                        "seed0_ci_lo": first["ci_lo"],
                        "seed0_ci_hi": first["ci_hi"],
                        "seed0_direction": direction0,
                        "seed1_slope_per_step": second["slope"],
                        "seed1_ci_lo": second["ci_lo"],
                        "seed1_ci_hi": second["ci_hi"],
                        "seed1_direction": direction1,
                        "direction_consistent": direction_consistent,
                        "ci_overlap": ci_overlap,
                        "both_cis_exclude_zero": (seed0_excludes_zero and seed1_excludes_zero),
                        "ci_and_point_direction_consistent": confirmed,
                        "confirmed_by_second_seed": confirmed,
                    }
                )
    return output


def atomic_write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    staging = path.with_name(f".{path.name}.{os.getpid()}.staging")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with staging.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(staging, path)
    finally:
        try:
            staging.unlink()
        except FileNotFoundError:
            pass


def atomic_write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    confirmed = [row for row in rows if row["confirmed_by_second_seed"]]
    consistent = [row for row in rows if row["direction_consistent"]]
    lines = [
        "# Cross-seed Workstream B trend consistency",
        "",
        f"- Expected paired games per seed: `{EXPECTED_GAMES}`",
        f"- Complete trend comparisons: `{len(rows)}`",
        f"- Direction-consistent comparisons: `{len(consistent)}`",
        f"- Confirmed by seed 1 (same direction; both 95% CIs exclude zero): `{len(confirmed)}`",
        "",
        "## Confirmed trends",
        "",
        "| Level | Metric | Statistic | Seed 0 slope | Seed 1 slope |",
        "|---|---|---|---:|---:|",
    ]
    for row in confirmed:
        lines.append(f"| {row['level']} | {row['metric']} | {row['statistic']} | {row['seed0_slope_per_step']:.6g} | {row['seed1_slope_per_step']:.6g} |")
    if not confirmed:
        lines.append("| — | — | — | — | — |")
    staging = path.with_name(f".{path.name}.{os.getpid()}.staging")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with staging.open("w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(staging, path)
    finally:
        try:
            staging.unlink()
        except FileNotFoundError:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed0-trends", required=True)
    parser.add_argument("--seed1-trends", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-report", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed0 = load_trends(args.seed0_trends)
    seed1 = load_trends(args.seed1_trends)
    rows = compare_trends(seed0, seed1)
    output_csv = Path(args.output_csv)
    output_report = Path(args.output_report)
    atomic_write_csv(output_csv, rows)
    atomic_write_report(output_report, rows)
    print(f"WM_CROSS_SEED_TRENDS_DONE comparisons={len(rows)} confirmed={sum(row['confirmed_by_second_seed'] for row in rows)} csv={output_csv} report={output_report}")


if __name__ == "__main__":
    main()
