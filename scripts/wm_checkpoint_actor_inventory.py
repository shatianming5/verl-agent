#!/usr/bin/env python3
"""Inventory and validate every FSDP actor shard for one checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

SHARD_PATTERN = re.compile(r"^(model|optim|extra_state)_world_size_(\d+)_rank_(\d+)\.pt$")
REQUIRED_SHARD_KINDS = ("model", "optim", "extra_state")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def inventory_actor(
    actor_dir: str | os.PathLike[str],
    *,
    expected_step: int | None = None,
    expected_world_size: int | None = None,
) -> dict[str, Any]:
    actor_input = Path(actor_dir)
    if actor_input.is_symlink():
        raise ValueError(f"Actor directory must be staged data, not a symlink: {actor_input}")
    actor = actor_input.resolve()
    if not actor.is_dir():
        raise FileNotFoundError(f"Actor directory does not exist: {actor}")

    shard_rows: list[dict[str, Any]] = []
    malformed_shards = []
    for path in sorted(actor.iterdir(), key=lambda item: item.name):
        if path.is_symlink():
            raise ValueError(f"Actor inventory rejects symlinks: {path}")
        if not path.is_file():
            continue
        match = SHARD_PATTERN.fullmatch(path.name)
        if match is None:
            if path.name.startswith(("model_world_size_", "optim_world_size_", "extra_state_world_size_")) and path.suffix == ".pt":
                malformed_shards.append(path.name)
            continue
        kind, world_size_text, rank_text = match.groups()
        size = path.stat().st_size
        if size <= 0:
            raise ValueError(f"Actor shard is empty: {path}")
        shard_rows.append(
            {
                "name": path.name,
                "kind": kind,
                "world_size": int(world_size_text),
                "rank": int(rank_text),
                "bytes": size,
                "sha256": sha256_file(path),
            }
        )
    if malformed_shards:
        raise ValueError(f"Malformed actor shard names: {malformed_shards}")
    if not shard_rows:
        raise ValueError(f"No FSDP actor shards found in {actor}")

    world_sizes = {row["world_size"] for row in shard_rows}
    if len(world_sizes) != 1:
        raise ValueError(f"Actor shards mix world sizes: {sorted(world_sizes)}")
    world_size = next(iter(world_sizes))
    if world_size < 1:
        raise ValueError(f"Actor shard world_size must be positive: {world_size}")
    if expected_world_size is not None and world_size != expected_world_size:
        raise ValueError(f"Actor world_size={world_size}; expected {expected_world_size}")
    expected_ranks = set(range(world_size))
    for kind in REQUIRED_SHARD_KINDS:
        ranks = [row["rank"] for row in shard_rows if row["kind"] == kind]
        if len(ranks) != len(set(ranks)):
            raise ValueError(f"Duplicate {kind} actor shard ranks: {ranks}")
        if set(ranks) != expected_ranks:
            raise ValueError(f"Incomplete {kind} actor shards: ranks={sorted(ranks)} expected={sorted(expected_ranks)}")

    step_match = re.fullmatch(r"global_step_(\d+)", actor.parent.name)
    checkpoint_step = int(step_match.group(1)) if step_match else None
    if expected_step is not None and checkpoint_step != expected_step:
        raise ValueError(f"Actor path checkpoint step={checkpoint_step}; expected {expected_step}")

    content = {
        "checkpoint_step": checkpoint_step,
        "world_size": world_size,
        "shards": shard_rows,
    }
    return {
        "schema_version": "wm_actor_shard_inventory_v1",
        "actor_dir": str(actor),
        **content,
        "shard_count": len(shard_rows),
        "total_shard_bytes": sum(row["bytes"] for row in shard_rows),
        "content_sha256": canonical_sha256(content),
    }


def atomic_write_json(path: str | os.PathLike[str], value: Any) -> None:
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
    parser.add_argument("--actor-dir", required=True)
    parser.add_argument("--expected-step", type=int)
    parser.add_argument("--expected-world-size", type=int)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inventory = inventory_actor(
        args.actor_dir,
        expected_step=args.expected_step,
        expected_world_size=args.expected_world_size,
    )
    atomic_write_json(args.output_json, inventory)
    print(f"WM_ACTOR_INVENTORY_VERIFIED step={inventory['checkpoint_step']} world_size={inventory['world_size']} shards={inventory['shard_count']} bytes={inventory['total_shard_bytes']} sha256={inventory['content_sha256']} output={args.output_json}")


if __name__ == "__main__":
    main()
