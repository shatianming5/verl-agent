#!/usr/bin/env python3
"""Generate, verify, and materialize the full ALFWorld train-game manifest."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agent_system.alfworld_game_manifest import (  # noqa: E402
    AUTHORITATIVE_RAW_TRAIN_TRAJECTORY_COUNT,
    AUTHORITATIVE_TRAIN_GAME_COUNT,
    build_manifest,
    load_manifest,
    validate_manifest,
)


def atomic_json_write(path: str, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_name(f".{target.name}.{os.getpid()}.staging")
    try:
        with staging.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(staging, target)
    finally:
        try:
            staging.unlink()
        except FileNotFoundError:
            pass


def build_schedule_rows(manifest: dict[str, Any], batch_size: int) -> list[dict[str, Any]]:
    """Create one unpadded row per game and pad only the final validation batch."""

    validate_manifest(manifest, require_train=True)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    games = manifest["games"]
    if not games:
        raise ValueError("Cannot schedule an empty manifest")
    padded_count = int(math.ceil(len(games) / batch_size) * batch_size)
    rows = []
    for schedule_index in range(padded_count):
        game = games[schedule_index % len(games)]
        is_padding = schedule_index >= len(games)
        episode_id = f"{manifest['manifest_sha256']}:{game['game_id']}:trajectory0:schedule{schedule_index}"
        env_kwargs = {
            "wm_game_id": game["game_id"],
            "wm_gamefile": game["gamefile"],
            "wm_task_type": game["task_type"],
            "wm_game_sha256": game["sha256"],
            "wm_manifest_sha256": manifest["manifest_sha256"],
            "wm_schedule_index": schedule_index,
            "wm_schedule_padding": is_padding,
            "wm_trajectory_index": 0,
            "wm_episode_id": episode_id,
        }
        rows.append(
            {
                "data_source": "text",
                "prompt": [{"role": "user", "content": ""}],
                "ability": "agent",
                "extra_info": {
                    "split": "train",
                    "index": schedule_index,
                    "game_id": game["game_id"],
                    "schedule_padding": is_padding,
                },
                "env_kwargs": env_kwargs,
            }
        )
    return rows


def write_schedule_parquet(path: str, rows: list[dict[str, Any]]) -> None:
    try:
        from datasets import Dataset
    except ImportError as exc:
        raise RuntimeError("The existing 'datasets' dependency is required to write the rollout parquet") from exc
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(rows).to_parquet(str(target))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate")
    generate.add_argument("--config", required=True)
    generate.add_argument("--output", required=True)
    generate.add_argument(
        "--expected-games",
        type=int,
        default=AUTHORITATIVE_TRAIN_GAME_COUNT,
    )
    generate.add_argument(
        "--expected-raw-trajectories",
        type=int,
        default=AUTHORITATIVE_RAW_TRAIN_TRAJECTORY_COUNT,
    )

    verify = subparsers.add_parser("verify")
    verify.add_argument("--manifest", required=True)
    verify.add_argument(
        "--expected-games",
        type=int,
        default=AUTHORITATIVE_TRAIN_GAME_COUNT,
    )
    verify.add_argument(
        "--expected-raw-trajectories",
        type=int,
        default=AUTHORITATIVE_RAW_TRAIN_TRAJECTORY_COUNT,
    )
    verify.add_argument("--verify-files", action="store_true")

    schedule = subparsers.add_parser("schedule")
    schedule.add_argument("--manifest", required=True)
    schedule.add_argument("--output-parquet", required=True)
    schedule.add_argument("--batch-size", type=int, required=True)
    schedule.add_argument(
        "--expected-games",
        type=int,
        default=AUTHORITATIVE_TRAIN_GAME_COUNT,
    )
    schedule.add_argument(
        "--expected-raw-trajectories",
        type=int,
        default=AUTHORITATIVE_RAW_TRAIN_TRAJECTORY_COUNT,
    )
    schedule.add_argument("--verify-files", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.expected_games != AUTHORITATIVE_TRAIN_GAME_COUNT or args.expected_raw_trajectories != AUTHORITATIVE_RAW_TRAIN_TRAJECTORY_COUNT:
        raise ValueError(f"Authoritative ALFWorld train discovery is fixed at {AUTHORITATIVE_RAW_TRAIN_TRAJECTORY_COUNT} raw trajectories and {AUTHORITATIVE_TRAIN_GAME_COUNT} filtered games")
    if args.command == "generate":
        with open(args.config, encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        manifest = build_manifest(config, split="train")
        validate_manifest(
            manifest,
            expected_games=args.expected_games,
            expected_raw_trajectories=args.expected_raw_trajectories,
            require_train=True,
            verify_files=True,
        )
        atomic_json_write(args.output, manifest)
        print(f"WM_MANIFEST_GENERATED path={args.output} raw_trajectories={manifest['raw_traj_data_count']} games={manifest['game_count']} sha256={manifest['manifest_sha256']}")
        return

    manifest = load_manifest(
        args.manifest,
        expected_games=args.expected_games,
        expected_raw_trajectories=args.expected_raw_trajectories,
        require_train=True,
        verify_files=args.verify_files,
    )
    if args.command == "verify":
        print(f"WM_MANIFEST_VERIFIED path={args.manifest} raw_trajectories={manifest['raw_traj_data_count']} games={manifest['game_count']} sha256={manifest['manifest_sha256']}")
        return

    rows = build_schedule_rows(manifest, args.batch_size)
    write_schedule_parquet(args.output_parquet, rows)
    print(f"WM_MANIFEST_SCHEDULE_WRITTEN path={args.output_parquet} games={manifest['game_count']} rows={len(rows)} padding={len(rows) - manifest['game_count']} batch_size={args.batch_size}")


if __name__ == "__main__":
    main()
