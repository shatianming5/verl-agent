"""Deterministic ALFWorld game discovery and manifest validation.

This module is intentionally free of TextWorld imports so manifest generation can
run on CPU-only hosts.  ``AlfredTWEnv.collect_game_files`` uses the same discovery
function, making the manifest the exact, auditable environment game list.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

TASK_TYPES = {
    1: "pick_and_place_simple",
    2: "look_at_obj_in_light",
    3: "pick_clean_then_place_in_recep",
    4: "pick_heat_then_place_in_recep",
    5: "pick_cool_then_place_in_recep",
    6: "pick_two_obj_and_place",
}
MANIFEST_SCHEMA = "alfworld_train_manifest_v1"
AUTHORITATIVE_RAW_TRAIN_TRAJECTORY_COUNT = 6374
AUTHORITATIVE_TRAIN_GAME_COUNT = 3553
WORKSTREAM_B_INFO_KEYS = (
    "wm_game_id",
    "wm_gamefile",
    "wm_task_type",
    "wm_game_sha256",
    "wm_manifest_sha256",
    "wm_schedule_index",
    "wm_schedule_padding",
    "wm_trajectory_index",
    "wm_episode_id",
)


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split_data_path(config: Mapping[str, Any], split: str) -> str:
    key_by_split = {
        "train": "data_path",
        "eval_in_distribution": "eval_id_data_path",
        "eval_out_of_distribution": "eval_ood_data_path",
    }
    try:
        key = key_by_split[split]
    except KeyError as exc:
        raise ValueError(f"Unsupported ALFWorld split: {split!r}") from exc
    value = os.path.expandvars(str(config["dataset"][key]))
    if not value or not os.path.isdir(value):
        raise FileNotFoundError(f"ALFWorld split directory does not exist: {value}")
    return os.path.realpath(value)


def collect_game_records_with_stats(
    config: Mapping[str, Any],
    split: str = "train",
    *,
    apply_config_limit: bool = True,
) -> tuple[list[dict[str, Any]], int]:
    """Collect the solvable games accepted by ``AlfredTWEnv``.

    The filtering rules mirror the historical environment implementation:
    unsupported movable/sliced tasks, task types outside ``env.task_types``,
    missing game files, and games not explicitly marked solvable are excluded.
    Results are sorted before applying any configured limit.
    """

    data_path = _split_data_path(config, split)
    task_type_ids = list(config["env"]["task_types"])
    if not task_type_ids:
        raise ValueError("ALFWorld env.task_types must not be empty")
    unknown_ids = sorted(set(task_type_ids) - set(TASK_TYPES))
    if unknown_ids:
        raise ValueError(f"Unknown ALFWorld task type ids: {unknown_ids}")
    allowed_task_types = {TASK_TYPES[int(task_id)] for task_id in task_type_ids}

    records: list[dict[str, Any]] = []
    raw_trajectory_count = 0
    for root, dirs, files in os.walk(data_path, topdown=False):
        dirs.sort()
        files = sorted(files)
        if "traj_data.json" not in files:
            continue
        raw_trajectory_count += 1
        if "movable" in root or "Sliced" in root:
            continue

        traj_path = os.path.join(root, "traj_data.json")
        gamefile = os.path.join(root, "game.tw-pddl")
        with open(traj_path, encoding="utf-8") as handle:
            traj_data = json.load(handle)
        task_type = traj_data.get("task_type")
        if task_type not in allowed_task_types or not os.path.isfile(gamefile):
            continue
        with open(gamefile, encoding="utf-8") as handle:
            game_data = json.load(handle)
        if game_data.get("solvable") is not True:
            continue

        real_gamefile = os.path.realpath(gamefile)
        game_id = Path(root).relative_to(data_path).as_posix()
        records.append(
            {
                "game_id": game_id,
                "gamefile": real_gamefile,
                "task_type": str(task_type),
                "sha256": sha256_file(real_gamefile),
            }
        )

    records.sort(key=lambda row: (row["game_id"], row["gamefile"], row["task_type"], row["sha256"]))
    if apply_config_limit:
        limit_key = "num_train_games" if split == "train" else "num_eval_games"
        limit = int(config["dataset"].get(limit_key, -1))
        if limit > 0:
            records = records[:limit]
    return records, raw_trajectory_count


def collect_game_records(
    config: Mapping[str, Any],
    split: str = "train",
    *,
    apply_config_limit: bool = True,
) -> list[dict[str, Any]]:
    records, _ = collect_game_records_with_stats(
        config,
        split,
        apply_config_limit=apply_config_limit,
    )
    return records


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def build_manifest(config: Mapping[str, Any], split: str = "train") -> dict[str, Any]:
    records, raw_trajectory_count = collect_game_records_with_stats(config, split)
    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA,
        "split": split,
        "dataset_root": _split_data_path(config, split),
        "task_type_ids": sorted(int(task_id) for task_id in config["env"]["task_types"]),
        "raw_traj_data_count": raw_trajectory_count,
        "game_count": len(records),
        "games": records,
    }
    manifest["manifest_sha256"] = hashlib.sha256(_canonical_json(manifest)).hexdigest()
    return manifest


def validate_manifest(
    manifest: Mapping[str, Any],
    *,
    min_games: int = 0,
    expected_games: int | None = None,
    expected_raw_trajectories: int | None = None,
    require_train: bool = True,
    verify_files: bool = False,
) -> dict[str, Any]:
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError(f"Expected schema_version={MANIFEST_SCHEMA!r}")
    if require_train and manifest.get("split") != "train":
        raise ValueError("Workstream B requires an ALFWorld train-split manifest")
    games_value = manifest.get("games")
    if not isinstance(games_value, list):
        raise ValueError("Manifest games must be a list")
    games = [dict(game) for game in games_value]
    if manifest.get("game_count") != len(games):
        raise ValueError(f"Manifest game_count={manifest.get('game_count')!r} does not equal games length={len(games)}")
    if len(games) < min_games:
        raise ValueError(f"Manifest has {len(games)} games; at least {min_games} are required")
    if expected_games is not None and len(games) != expected_games:
        raise ValueError(f"Manifest has {len(games)} games; exactly {expected_games} are required")
    raw_trajectory_count = manifest.get("raw_traj_data_count")
    if not isinstance(raw_trajectory_count, int) or raw_trajectory_count < len(games):
        raise ValueError(f"Manifest raw_traj_data_count is invalid: {raw_trajectory_count!r}")
    if expected_raw_trajectories is not None and raw_trajectory_count != expected_raw_trajectories:
        raise ValueError(f"Manifest has {raw_trajectory_count} raw trajectories; exactly {expected_raw_trajectories} are required")

    expected_order = sorted(
        games,
        key=lambda row: (
            str(row.get("game_id", "")),
            str(row.get("gamefile", "")),
            str(row.get("task_type", "")),
            str(row.get("sha256", "")),
        ),
    )
    if games != expected_order:
        raise ValueError("Manifest games are not deterministically sorted")

    ids: set[str] = set()
    gamefiles: set[str] = set()
    required = {"game_id", "gamefile", "task_type", "sha256"}
    for index, game in enumerate(games):
        missing = required - set(game)
        if missing:
            raise ValueError(f"Manifest game {index} is missing fields: {sorted(missing)}")
        game_id = str(game["game_id"])
        gamefile = os.path.realpath(str(game["gamefile"]))
        digest = str(game["sha256"])
        if not game_id or game_id in ids:
            raise ValueError(f"Duplicate or empty game_id: {game_id!r}")
        if gamefile in gamefiles:
            raise ValueError(f"Duplicate gamefile: {gamefile!r}")
        if not re_full_sha256(digest):
            raise ValueError(f"Invalid SHA256 for game_id={game_id!r}: {digest!r}")
        if gamefile != str(game["gamefile"]):
            raise ValueError(f"gamefile must be canonical for game_id={game_id!r}: {game['gamefile']!r}")
        if verify_files:
            if not os.path.isfile(gamefile):
                raise FileNotFoundError(f"Manifest gamefile does not exist: {gamefile}")
            actual_digest = sha256_file(gamefile)
            if actual_digest != digest:
                raise ValueError(f"SHA256 mismatch for game_id={game_id!r}: expected={digest} actual={actual_digest}")
        ids.add(game_id)
        gamefiles.add(gamefile)

    unsigned = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    expected_sha = hashlib.sha256(_canonical_json(unsigned)).hexdigest()
    if manifest.get("manifest_sha256") != expected_sha:
        raise ValueError(f"Manifest fingerprint mismatch: expected={expected_sha} actual={manifest.get('manifest_sha256')!r}")
    return dict(manifest)


def re_full_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def load_manifest(
    path: str | os.PathLike[str],
    *,
    min_games: int = 0,
    expected_games: int | None = None,
    expected_raw_trajectories: int | None = None,
    require_train: bool = True,
    verify_files: bool = False,
) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    return validate_manifest(
        manifest,
        min_games=min_games,
        expected_games=expected_games,
        expected_raw_trajectories=expected_raw_trajectories,
        require_train=require_train,
        verify_files=verify_files,
    )


def canonicalize_schedule(
    manifest: Mapping[str, Any] | None,
    schedule: Any,
    *,
    env_num: int,
    group_n: int,
    require_schedule: bool,
) -> list[dict[str, Any] | None]:
    process_count = env_num * group_n
    if schedule is None:
        if require_schedule:
            raise ValueError("Workstream B ALFWorld reset requires an explicit manifest schedule")
        return [None] * process_count
    if hasattr(schedule, "tolist"):
        schedule = schedule.tolist()
    elif isinstance(schedule, tuple):
        schedule = list(schedule)
    elif isinstance(schedule, dict):
        schedule = [schedule]
    if not isinstance(schedule, list):
        raise TypeError(f"ALFWorld schedule must be a list of mappings, got {type(schedule).__name__}")
    if group_n != 1:
        raise ValueError("Explicit Workstream B schedules require group_n=1")
    if len(schedule) != process_count:
        raise ValueError(f"ALFWorld schedule has {len(schedule)} entries; expected {process_count}")
    if manifest is None:
        raise ValueError("Explicit ALFWorld schedules require a validated manifest")

    games = manifest["games"]
    if not games:
        raise ValueError("Explicit ALFWorld schedules require a non-empty manifest")
    canonical: list[dict[str, Any] | None] = []
    seen_schedule_indices: set[int] = set()
    for position, raw_entry in enumerate(schedule):
        if not isinstance(raw_entry, dict):
            raise TypeError(f"ALFWorld schedule entry {position} must be a mapping")
        entry = dict(raw_entry)
        missing = set(WORKSTREAM_B_INFO_KEYS) - set(entry)
        if missing:
            raise ValueError(f"ALFWorld schedule entry {position} is missing {sorted(missing)}")
        schedule_index = int(entry["wm_schedule_index"])
        trajectory_index = int(entry["wm_trajectory_index"])
        if schedule_index < 0 or schedule_index in seen_schedule_indices:
            raise ValueError(f"Invalid or duplicate ALFWorld schedule index: {schedule_index}")
        if trajectory_index != 0:
            raise ValueError("Full Workstream B schedule requires wm_trajectory_index=0")
        expected_game = games[schedule_index % len(games)]
        expected_padding = schedule_index >= len(games)
        expected = {
            "wm_game_id": expected_game["game_id"],
            "wm_gamefile": expected_game["gamefile"],
            "wm_task_type": expected_game["task_type"],
            "wm_game_sha256": expected_game["sha256"],
            "wm_manifest_sha256": manifest["manifest_sha256"],
            "wm_schedule_padding": expected_padding,
            "wm_episode_id": (f"{manifest['manifest_sha256']}:{expected_game['game_id']}:trajectory0:schedule{schedule_index}"),
        }
        for key, value in expected.items():
            if entry[key] != value:
                raise ValueError(f"Schedule metadata mismatch at index={schedule_index}: {key} expected={value!r} actual={entry[key]!r}")
        entry["wm_gamefile"] = os.path.realpath(str(entry["wm_gamefile"]))
        entry["wm_schedule_index"] = schedule_index
        entry["wm_trajectory_index"] = trajectory_index
        entry["wm_schedule_padding"] = expected_padding
        canonical.append(entry)
        seen_schedule_indices.add(schedule_index)
    return canonical
