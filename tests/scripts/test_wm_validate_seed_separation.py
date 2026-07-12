import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _load_module():
    repo_root = Path(__file__).resolve().parents[2]
    scripts = repo_root / "scripts"
    sys.path.insert(0, str(scripts))
    path = scripts / "wm_validate_seed_separation.py"
    spec = importlib.util.spec_from_file_location(
        "wm_validate_seed_separation_under_test",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _inventories(root, *, identical=False):
    inventory_dir = root / "inventories"
    inventory_dir.mkdir()
    seed0_root = root / "seed0"
    seed1_root = root / "seed1"
    for step in (15, 30, 45, 60, 75, 90, 105, 120, 135, 150):
        for seed, checkpoint_root in ((0, seed0_root), (1, seed1_root)):
            fingerprint = f"{step:04x}{0 if identical else seed:x}".ljust(64, "a")
            value = {
                "schema_version": "wm_actor_shard_inventory_v1",
                "checkpoint_step": step,
                "actor_dir": str(checkpoint_root / f"global_step_{step}" / "actor"),
                "content_sha256": fingerprint,
            }
            (inventory_dir / f"seed{seed}_step{step}_actor_inventory.json").write_text(
                json.dumps(value),
                encoding="utf-8",
            )
    return seed0_root, seed1_root, inventory_dir


def test_seed_separation_accepts_distinct_roots_and_inventories(tmp_path):
    module = _load_module()
    seed0, seed1, inventories = _inventories(tmp_path)

    result = module.validate_seed_separation(
        seed0_root=str(seed0),
        seed1_root=str(seed1),
        inventory_dir=str(inventories),
    )

    assert len(result["checkpoints"]) == 10


def test_seed_separation_rejects_same_roots_or_shards(tmp_path):
    module = _load_module()
    seed0, seed1, inventories = _inventories(tmp_path, identical=True)
    with pytest.raises(ValueError, match="identical at step"):
        module.validate_seed_separation(
            seed0_root=str(seed0),
            seed1_root=str(seed1),
            inventory_dir=str(inventories),
        )
    with pytest.raises(ValueError, match="same path"):
        module.validate_seed_separation(
            seed0_root=str(seed0),
            seed1_root=str(seed0),
            inventory_dir=str(inventories),
        )
