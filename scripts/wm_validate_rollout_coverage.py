#!/usr/bin/env python3
"""Fail-closed coverage and provenance validation for Workstream B rollouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agent_system.alfworld_game_manifest import (  # noqa: E402
    AUTHORITATIVE_RAW_TRAIN_TRAJECTORY_COUNT,
    AUTHORITATIVE_TRAIN_GAME_COUNT,
    load_manifest,
)

PROTOCOL = "workstream_b_full_train_v2"
SCHEMA = "wm_transition_v2"


def _bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value.lower() in {"true", "1"}:
            return True
        if value.lower() in {"false", "0"}:
            return False
    if value in (0, 1):
        return bool(value)
    return None


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def stable_transition_id(row: dict[str, Any]) -> str:
    payload = (f"{row['wm_manifest_sha256']}\0{row['wm_episode_id']}\0{int(row['wm_step_idx'])}").encode()
    return hashlib.sha256(payload).hexdigest()


def load_dump(path: str) -> list[dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: row must be a JSON object")
            row["_line_number"] = line_number
            rows.append(row)
    if not rows:
        raise ValueError(f"Rollout dump is empty: {path}")
    return rows


def validate_rollout(
    manifest: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    expected_checkpoint_step: str,
    temperature: float,
    top_p: float,
    top_k: int,
    do_sample: bool,
    min_trajectories_per_game: int = 1,
) -> dict[str, Any]:
    game_by_id = {game["game_id"]: game for game in manifest["games"]}
    required = {
        "schema_version",
        "workstream_b_protocol",
        "split",
        "wm_game_id",
        "wm_gamefile",
        "wm_task_type",
        "wm_game_sha256",
        "wm_manifest_sha256",
        "wm_schedule_index",
        "wm_schedule_padding",
        "wm_trajectory_index",
        "wm_episode_id",
        "traj_uid",
        "wm_step_idx",
        "wm_prev_obs_text",
        "wm_action_text",
        "wm_next_obs_text",
        "episode_success",
        "episode_rewards",
        "score",
        "rollout_checkpoint_step",
        "rollout_temperature",
        "rollout_top_p",
        "rollout_top_k",
        "rollout_do_sample",
        "rollout_n",
    }
    episode_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    transition_ids: set[str] = set()
    schedule_to_episode: dict[int, str] = {}

    for row_index, row in enumerate(rows, start=1):
        where = f"line {row.get('_line_number', row_index)}"
        missing = required - set(row)
        if missing:
            raise ValueError(f"{where}: missing full-protocol fields: {sorted(missing)}")
        if row["schema_version"] != SCHEMA or row["workstream_b_protocol"] != PROTOCOL:
            raise ValueError(f"{where}: legacy or unknown Workstream B schema/protocol")
        if row["split"] != "train":
            raise ValueError(f"{where}: expected split='train', got {row['split']!r}")
        if str(row["rollout_checkpoint_step"]) != str(expected_checkpoint_step):
            raise ValueError(f"{where}: rollout checkpoint {row['rollout_checkpoint_step']!r} does not match expected {expected_checkpoint_step!r}")
        decoding_actual = (
            _finite(row["rollout_temperature"]),
            _finite(row["rollout_top_p"]),
            int(row["rollout_top_k"]),
            _bool(row["rollout_do_sample"]),
        )
        decoding_expected = (float(temperature), float(top_p), int(top_k), bool(do_sample))
        if decoding_actual != decoding_expected:
            raise ValueError(f"{where}: decoding mismatch actual={decoding_actual} expected={decoding_expected}")
        if int(row["rollout_n"]) != 1:
            raise ValueError(f"{where}: full protocol requires rollout_n=1")

        game_id = str(row["wm_game_id"])
        game = game_by_id.get(game_id)
        if game is None:
            raise ValueError(f"{where}: game_id is not in the manifest: {game_id!r}")
        expected_game_values = {
            "wm_gamefile": game["gamefile"],
            "wm_task_type": game["task_type"],
            "wm_game_sha256": game["sha256"],
            "wm_manifest_sha256": manifest["manifest_sha256"],
        }
        for key, expected in expected_game_values.items():
            if row[key] != expected:
                raise ValueError(f"{where}: {key} mismatch expected={expected!r} actual={row[key]!r}")

        episode_id = str(row["wm_episode_id"])
        if not episode_id or str(row["traj_uid"]) != episode_id:
            raise ValueError(f"{where}: traj_uid must equal the stable wm_episode_id")
        try:
            schedule_index = int(row["wm_schedule_index"])
            step_index = int(row["wm_step_idx"])
            trajectory_index = int(row["wm_trajectory_index"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{where}: schedule/trajectory/step indices must be integers") from exc
        if schedule_index < 0 or step_index < 0 or trajectory_index < 0:
            raise ValueError(f"{where}: schedule/trajectory/step indices must be non-negative")
        scheduled_game = manifest["games"][schedule_index % manifest["game_count"]]
        schedule_padding = _bool(row["wm_schedule_padding"])
        expected_padding = schedule_index >= manifest["game_count"]
        if game_id != scheduled_game["game_id"] or schedule_padding is not expected_padding:
            raise ValueError(f"{where}: schedule index does not map to the canonical manifest game/padding")
        if trajectory_index != 0:
            raise ValueError(f"{where}: full protocol requires trajectory index 0")
        expected_episode_id = f"{manifest['manifest_sha256']}:{game_id}:trajectory0:schedule{schedule_index}"
        if episode_id != expected_episode_id:
            raise ValueError(f"{where}: non-canonical stable episode id")
        prior_episode = schedule_to_episode.setdefault(schedule_index, episode_id)
        if prior_episode != episode_id:
            raise ValueError(f"{where}: schedule index {schedule_index} maps to multiple episodes")

        transition_id = stable_transition_id(row)
        if transition_id in transition_ids:
            raise ValueError(f"{where}: duplicate stable transition id {transition_id}")
        transition_ids.add(transition_id)

        success = _bool(row["episode_success"])
        reward = _finite(row["episode_rewards"])
        score = _finite(row["score"])
        if success is None or reward is None or score is None:
            raise ValueError(f"{where}: score, episode reward, and success label must be finite/present")
        if (score > 0.0) != success or (reward > 0.0) != success:
            raise ValueError(f"{where}: score/reward disagree with episode_success")
        episode_rows[episode_id].append(row)

    game_episodes: dict[str, set[str]] = defaultdict(set)
    success_episodes = 0
    padding_episodes = 0
    for episode_id, group in episode_rows.items():
        first = group[0]
        invariant_keys = (
            "wm_game_id",
            "wm_gamefile",
            "wm_game_sha256",
            "wm_manifest_sha256",
            "wm_schedule_index",
            "wm_schedule_padding",
            "wm_trajectory_index",
            "episode_success",
            "episode_rewards",
            "score",
        )
        for row in group[1:]:
            for key in invariant_keys:
                if row[key] != first[key]:
                    raise ValueError(f"Episode {episode_id!r} has inconsistent {key}")
        steps = sorted(int(row["wm_step_idx"]) for row in group)
        if steps != list(range(len(steps))):
            raise ValueError(f"Episode {episode_id!r} has non-contiguous steps: {steps[:10]}")
        is_padding = _bool(first["wm_schedule_padding"])
        if is_padding is None:
            raise ValueError(f"Episode {episode_id!r} has invalid wm_schedule_padding")
        if is_padding:
            padding_episodes += 1
        else:
            game_episodes[str(first["wm_game_id"])].add(episode_id)
            if _bool(first["episode_success"]):
                success_episodes += 1

    missing_games = [game_id for game_id in game_by_id if len(game_episodes.get(game_id, set())) < min_trajectories_per_game]
    if missing_games:
        preview = ", ".join(missing_games[:10])
        raise ValueError(f"Manifest coverage incomplete: {len(missing_games)} games have fewer than {min_trajectories_per_game} trajectories; first={preview}")

    return {
        "schema_version": "workstream_b_rollout_coverage_v1",
        "protocol": PROTOCOL,
        "manifest_sha256": manifest["manifest_sha256"],
        "raw_traj_data_count": manifest["raw_traj_data_count"],
        "manifest_games": manifest["game_count"],
        "covered_games": len(game_episodes),
        "min_trajectories_per_game": min(len(value) for value in game_episodes.values()),
        "episodes": len(episode_rows) - padding_episodes,
        "total_episodes_with_padding": len(episode_rows),
        "padding_episodes": padding_episodes,
        "success_episodes": success_episodes,
        "failure_episodes": len(episode_rows) - padding_episodes - success_episodes,
        "transitions": len(rows),
        "stable_transition_ids": len(transition_ids),
        "checkpoint_step": str(expected_checkpoint_step),
        "decoding": {
            "temperature": float(temperature),
            "top_p": float(top_p),
            "top_k": int(top_k),
            "do_sample": bool(do_sample),
        },
    }


def atomic_write_json(path: str, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_name(f".{target.name}.{os.getpid()}.staging")
    try:
        with staging.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(staging, target)
    finally:
        try:
            staging.unlink()
        except FileNotFoundError:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dump", required=True)
    parser.add_argument("--expected-checkpoint-step", required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--top-p", type=float, required=True)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--do-sample", choices=("true", "false"), required=True)
    parser.add_argument(
        "--expected-games",
        type=int,
        default=AUTHORITATIVE_TRAIN_GAME_COUNT,
    )
    parser.add_argument(
        "--expected-raw-trajectories",
        type=int,
        default=AUTHORITATIVE_RAW_TRAIN_TRAJECTORY_COUNT,
    )
    parser.add_argument("--min-trajectories-per-game", type=int, default=1)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--verify-files", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.expected_games != AUTHORITATIVE_TRAIN_GAME_COUNT or args.expected_raw_trajectories != AUTHORITATIVE_RAW_TRAIN_TRAJECTORY_COUNT:
        raise ValueError("Coverage validation requires the authoritative 6374→3553 ALFWorld train discovery")
    manifest = load_manifest(
        args.manifest,
        expected_games=args.expected_games,
        expected_raw_trajectories=args.expected_raw_trajectories,
        require_train=True,
        verify_files=args.verify_files,
    )
    rows = load_dump(args.dump)
    summary = validate_rollout(
        manifest,
        rows,
        expected_checkpoint_step=args.expected_checkpoint_step,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        do_sample=args.do_sample == "true",
        min_trajectories_per_game=args.min_trajectories_per_game,
    )
    summary["manifest_path"] = os.path.realpath(args.manifest)
    summary["dump_path"] = os.path.realpath(args.dump)
    atomic_write_json(args.output_json, summary)
    print(f"WM_ROLLOUT_COVERAGE_VERIFIED games={summary['covered_games']}/{summary['manifest_games']} episodes={summary['episodes']} transitions={summary['transitions']} output={args.output_json}")


if __name__ == "__main__":
    main()
