#!/usr/bin/env python3
"""Atomically stage and verify checkpoint actor directories from an archive."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Any

from wm_checkpoint_actor_inventory import (
    atomic_write_json,
    inventory_actor,
)

DEFAULT_STEPS = (15, 30, 45, 60, 75, 90, 105, 120, 135, 150)


def comparable_inventory(inventory: dict[str, Any]) -> dict[str, Any]:
    return {
        "checkpoint_step": inventory["checkpoint_step"],
        "world_size": inventory["world_size"],
        "shards": inventory["shards"],
        "shard_count": inventory["shard_count"],
        "total_shard_bytes": inventory["total_shard_bytes"],
        "content_sha256": inventory["content_sha256"],
    }


def source_inventories(
    source_root: Path,
    steps: tuple[int, ...],
    expected_world_size: int | None,
) -> dict[int, dict[str, Any]]:
    if not source_root.is_dir():
        raise FileNotFoundError(f"Checkpoint archive root is unavailable: {source_root}")
    return {
        step: inventory_actor(
            source_root / f"global_step_{step}" / "actor",
            expected_step=step,
            expected_world_size=expected_world_size,
        )
        for step in steps
    }


def required_stage_bytes(
    inventories: dict[int, dict[str, Any]],
    destination_root: Path,
    reuse_verified: bool,
) -> int:
    required = 0
    for step, source_inventory in inventories.items():
        destination_actor = destination_root / f"global_step_{step}" / "actor"
        if destination_actor.exists() and reuse_verified:
            destination_inventory = inventory_actor(
                destination_actor,
                expected_step=step,
                expected_world_size=source_inventory["world_size"],
            )
            if comparable_inventory(destination_inventory) != comparable_inventory(source_inventory):
                raise ValueError(f"Existing staged actor differs from archive: {destination_actor}")
        elif destination_actor.exists():
            raise FileExistsError(f"Destination actor already exists; use --reuse-verified: {destination_actor}")
        else:
            required += sum(path.stat().st_size for path in (Path(source_inventory["actor_dir"]).iterdir()) if path.is_file())
    return required


def stage_checkpoint_actors(
    source_root: str | os.PathLike[str],
    destination_root: str | os.PathLike[str],
    *,
    steps: tuple[int, ...] = DEFAULT_STEPS,
    expected_world_size: int | None = None,
    reuse_verified: bool = False,
) -> dict[str, Any]:
    source = Path(source_root).resolve()
    destination = Path(destination_root).resolve()
    if tuple(sorted(set(steps))) != steps:
        raise ValueError("Stage steps must be unique and sorted")
    if expected_world_size is not None and expected_world_size < 1:
        raise ValueError("expected_world_size must be positive")
    common = Path(os.path.commonpath((source, destination)))
    if common in {source, destination}:
        raise ValueError("Source and destination checkpoint roots must not overlap")
    inventories = source_inventories(source, steps, expected_world_size)
    destination.mkdir(parents=True, exist_ok=True)
    required_bytes = required_stage_bytes(
        inventories,
        destination,
        reuse_verified,
    )
    free_bytes = shutil.disk_usage(destination).free
    if free_bytes < int(required_bytes * 1.05):
        raise OSError(f"Insufficient staging space: required_with_margin={int(required_bytes * 1.05)} free={free_bytes}")

    checkpoint_rows = []
    for step in steps:
        source_inventory = inventories[step]
        source_actor = Path(source_inventory["actor_dir"])
        destination_step = destination / f"global_step_{step}"
        destination_actor = destination_step / "actor"
        destination_step.mkdir(parents=True, exist_ok=True)
        status = "reused"
        if not destination_actor.exists():
            status = "copied"
            staging_actor = destination_step / f".actor.staging.{os.getpid()}"
            if staging_actor.exists():
                raise FileExistsError(f"Stale actor staging directory: {staging_actor}")
            try:
                shutil.copytree(source_actor, staging_actor, symlinks=False)
                staged_inventory = inventory_actor(
                    staging_actor,
                    expected_world_size=source_inventory["world_size"],
                )
                staged_comparable = comparable_inventory(staged_inventory)
                staged_comparable["checkpoint_step"] = step
                if staged_comparable != comparable_inventory(source_inventory):
                    raise ValueError(f"Staged actor checksum/inventory mismatch at step {step}")
                os.replace(staging_actor, destination_actor)
            finally:
                if staging_actor.exists():
                    shutil.rmtree(staging_actor)

        destination_inventory = inventory_actor(
            destination_actor,
            expected_step=step,
            expected_world_size=source_inventory["world_size"],
        )
        if comparable_inventory(destination_inventory) != comparable_inventory(source_inventory):
            raise ValueError(f"Final staged actor differs from archive at step {step}")
        inventory_path = destination_step / "actor_inventory.json"
        atomic_write_json(inventory_path, destination_inventory)
        checkpoint_rows.append(
            {
                "step": step,
                "status": status,
                "actor_dir": str(destination_actor),
                "inventory": str(inventory_path),
                "world_size": destination_inventory["world_size"],
                "shards": destination_inventory["shard_count"],
                "bytes": destination_inventory["total_shard_bytes"],
                "content_sha256": destination_inventory["content_sha256"],
            }
        )

    receipt = {
        "schema_version": "wm_actor_stage_receipt_v1",
        "source_root": str(source),
        "destination_root": str(destination),
        "steps": list(steps),
        "checkpoints": checkpoint_rows,
    }
    atomic_write_json(destination / "actor_stage_receipt.json", receipt)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--destination-root", required=True)
    parser.add_argument("--steps", nargs="+", type=int, default=list(DEFAULT_STEPS))
    parser.add_argument("--expected-world-size", type=int)
    parser.add_argument("--reuse-verified", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    steps = tuple(args.steps)
    if tuple(sorted(set(steps))) != steps:
        raise ValueError("Stage steps must be unique and sorted")
    receipt = stage_checkpoint_actors(
        args.source_root,
        args.destination_root,
        steps=steps,
        expected_world_size=args.expected_world_size,
        reuse_verified=args.reuse_verified,
    )
    print(f"WM_ACTOR_STAGE_DONE source={receipt['source_root']} destination={receipt['destination_root']} checkpoints={len(receipt['checkpoints'])}")


if __name__ == "__main__":
    main()
