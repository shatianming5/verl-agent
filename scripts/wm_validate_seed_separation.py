#!/usr/bin/env python3
"""Fail if seed 0 and seed 1 resolve to the same checkpoint provenance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from wm_checkpoint_actor_inventory import atomic_write_json

STEPS = (15, 30, 45, 60, 75, 90, 105, 120, 135, 150)


def load_inventory(path: Path, expected_step: int) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing seed actor inventory: {path}")
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if value.get("schema_version") != "wm_actor_shard_inventory_v1":
        raise ValueError(f"Invalid actor inventory schema: {path}")
    if value.get("checkpoint_step") != expected_step:
        raise ValueError(f"Actor inventory step mismatch: {path}")
    fingerprint = value.get("content_sha256")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise ValueError(f"Actor inventory fingerprint is missing: {path}")
    return value


def validate_seed_separation(
    *,
    seed0_root: str,
    seed1_root: str,
    inventory_dir: str,
    seed0_archive_root: str = "",
    seed1_archive_root: str = "",
) -> dict[str, Any]:
    root0 = Path(seed0_root).resolve()
    root1 = Path(seed1_root).resolve()
    if root0 == root1:
        raise ValueError("Seed 0 and seed 1 checkpoint roots resolve to the same path")
    archive0 = Path(seed0_archive_root).resolve() if seed0_archive_root else None
    archive1 = Path(seed1_archive_root).resolve() if seed1_archive_root else None
    if archive0 is not None and archive1 is not None and archive0 == archive1:
        raise ValueError("Seed 0 and seed 1 archive roots resolve to the same path")

    inventories = Path(inventory_dir)
    rows = []
    for step in STEPS:
        first = load_inventory(
            inventories / f"seed0_step{step}_actor_inventory.json",
            step,
        )
        second = load_inventory(
            inventories / f"seed1_step{step}_actor_inventory.json",
            step,
        )
        if Path(first["actor_dir"]).resolve() == Path(second["actor_dir"]).resolve():
            raise ValueError(f"Seed actor inventories point to the same directory at step {step}")
        if first["content_sha256"] == second["content_sha256"]:
            raise ValueError(f"Seed actor shard inventories are identical at step {step}")
        rows.append(
            {
                "step": step,
                "seed0_content_sha256": first["content_sha256"],
                "seed1_content_sha256": second["content_sha256"],
            }
        )
    return {
        "schema_version": "wm_seed_checkpoint_separation_v1",
        "seed0_checkpoint_root": str(root0),
        "seed1_checkpoint_root": str(root1),
        "seed0_archive_root": str(archive0) if archive0 is not None else None,
        "seed1_archive_root": str(archive1) if archive1 is not None else None,
        "checkpoints": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed0-root", required=True)
    parser.add_argument("--seed1-root", required=True)
    parser.add_argument("--seed0-archive-root", default="")
    parser.add_argument("--seed1-archive-root", default="")
    parser.add_argument("--inventory-dir", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = validate_seed_separation(
        seed0_root=args.seed0_root,
        seed1_root=args.seed1_root,
        inventory_dir=args.inventory_dir,
        seed0_archive_root=args.seed0_archive_root,
        seed1_archive_root=args.seed1_archive_root,
    )
    atomic_write_json(args.output_json, result)
    print(f"WM_SEED_SEPARATION_VERIFIED checkpoints={len(result['checkpoints'])} output={args.output_json}")


if __name__ == "__main__":
    main()
