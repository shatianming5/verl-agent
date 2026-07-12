import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "scripts"


def _load(name):
    sys.path.insert(0, str(SCRIPTS))
    path = SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _actor(root, step, world_size=2):
    actor = root / f"global_step_{step}" / "actor"
    actor.mkdir(parents=True)
    for kind in ("model", "optim", "extra_state"):
        for rank in range(world_size):
            (actor / f"{kind}_world_size_{world_size}_rank_{rank}.pt").write_bytes(f"{step}-{kind}-{rank}".encode())
    (actor / "config.json").write_text("{}\n", encoding="utf-8")
    return actor


def test_actor_inventory_requires_all_ranked_shard_kinds(tmp_path):
    inventory_module = _load("wm_checkpoint_actor_inventory")
    actor = _actor(tmp_path / "source", 15)

    inventory = inventory_module.inventory_actor(
        actor,
        expected_step=15,
        expected_world_size=2,
    )

    assert inventory["world_size"] == 2
    assert inventory["shard_count"] == 6
    assert len(inventory["content_sha256"]) == 64
    assert {row["kind"] for row in inventory["shards"]} == {
        "model",
        "optim",
        "extra_state",
    }

    (actor / "optim_world_size_2_rank_1.pt").unlink()
    with pytest.raises(ValueError, match="Incomplete optim"):
        inventory_module.inventory_actor(actor, expected_step=15)


def test_checkpoint_staging_is_atomic_verified_and_reusable(tmp_path):
    stage_module = _load("wm_stage_checkpoint_actors")
    source = tmp_path / "archive"
    _actor(source, 15)
    _actor(source, 30)
    destination = tmp_path / "staged"

    receipt = stage_module.stage_checkpoint_actors(
        source,
        destination,
        steps=(15, 30),
        expected_world_size=2,
    )

    assert [row["status"] for row in receipt["checkpoints"]] == [
        "copied",
        "copied",
    ]
    assert (destination / "global_step_15" / "actor_inventory.json").is_file()
    saved_receipt = json.loads((destination / "actor_stage_receipt.json").read_text(encoding="utf-8"))
    assert saved_receipt["steps"] == [15, 30]
    assert not list(destination.glob("**/.actor.staging.*"))

    reused = stage_module.stage_checkpoint_actors(
        source,
        destination,
        steps=(15, 30),
        expected_world_size=2,
        reuse_verified=True,
    )
    assert [row["status"] for row in reused["checkpoints"]] == [
        "reused",
        "reused",
    ]


def test_checkpoint_staging_fails_closed_when_archive_is_unavailable(tmp_path):
    stage_module = _load("wm_stage_checkpoint_actors")
    destination = tmp_path / "staged"

    with pytest.raises(FileNotFoundError, match="archive root is unavailable"):
        stage_module.stage_checkpoint_actors(
            tmp_path / "disconnected-cephfs",
            destination,
            steps=(15,),
        )

    assert not (destination / "actor_stage_receipt.json").exists()
