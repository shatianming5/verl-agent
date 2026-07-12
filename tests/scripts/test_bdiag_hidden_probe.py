import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


def _load_module():
    repo_root = Path(__file__).resolve().parents[2]
    scripts = repo_root / "scripts"
    sys.path.insert(0, str(scripts))
    path = scripts / "bdiag_hidden_probe.py"
    spec = importlib.util.spec_from_file_location("bdiag_hidden_probe_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_stratified_group_folds_never_leak_games():
    module = _load_module()
    groups = np.asarray(
        ["game-a", "game-a", "game-b", "game-b", "game-c", "game-c", "game-d", "game-d"],
        dtype=object,
    )
    labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)

    folds = module.stratified_group_folds(labels, groups, folds=2, seed=0)

    assert len(folds) == 2
    heldout_groups = [set(groups[fold].tolist()) for fold in folds]
    assert heldout_groups[0].isdisjoint(heldout_groups[1])
    assert heldout_groups[0] | heldout_groups[1] == set(groups.tolist())
    for fold in folds:
        assert set(labels[fold].tolist()) == {0, 1}


def test_group_label_shuffle_keeps_episode_labels_within_game():
    module = _load_module()
    groups = np.asarray(["a", "a", "b", "b", "c", "c", "d", "d"], dtype=object)
    labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)

    shuffled = module.group_label_shuffle(labels, groups, seed=3)

    for group in set(groups.tolist()):
        assert len(set(shuffled[groups == group].tolist())) == 1
    assert sorted(shuffled.tolist()) == sorted(labels.tolist())


def test_nested_group_probe_reports_disjoint_heldout_games():
    module = _load_module()
    groups = np.asarray([f"game-{index:02d}" for index in range(20)], dtype=object)
    labels = np.asarray([0] * 10 + [1] * 10, dtype=np.int64)
    features = np.column_stack(
        [
            labels * 4.0 - 2.0,
            np.linspace(-0.2, 0.2, len(labels)),
            np.ones(len(labels)),
        ]
    ).astype(np.float32)

    result = module.nested_group_probe(
        features,
        labels,
        groups,
        outer_folds=2,
        inner_folds=2,
        c_grid=[0.3],
        seed=2,
        fit_device="cpu",
    )

    assert result["probe_auc"] > 0.95
    heldout_folds = [set(values) for values in json.loads(result["heldout_game_folds_json"])]
    assert heldout_folds[0].isdisjoint(heldout_folds[1])
    assert heldout_folds[0] | heldout_folds[1] == set(groups.tolist())


def test_probe_reuses_strict_rollout_validation(tmp_path, monkeypatch):
    module = _load_module()
    step_dir = tmp_path / "stepinit"
    step_dir.mkdir()
    (step_dir / "init.wm_transitions.jsonl").write_text("{}\n", encoding="utf-8")
    (step_dir / "coverage.json").write_text("{}\n", encoding="utf-8")

    def reject(**kwargs):
        assert kwargs["step"] == "init"
        raise ValueError("strict rollout rejected")

    monkeypatch.setattr(module, "validate_step_provenance", reject)
    with pytest.raises(ValueError, match="strict rollout rejected"):
        module.load_rollout_rows(
            str(tmp_path),
            "init",
            manifest={},
            manifest_path=str(tmp_path / "manifest.json"),
        )


def test_probe_rejects_missing_padding_metadata(tmp_path, monkeypatch):
    module = _load_module()
    step_dir = tmp_path / "stepinit"
    step_dir.mkdir()
    (step_dir / "init.wm_transitions.jsonl").write_text(
        '{"wm_episode_id":"episode"}\n',
        encoding="utf-8",
    )
    (step_dir / "coverage.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(module, "validate_step_provenance", lambda **kwargs: None)

    with pytest.raises(ValueError, match="No non-padding rollout rows"):
        module.load_rollout_rows(
            str(tmp_path),
            "init",
            manifest={},
            manifest_path=str(tmp_path / "manifest.json"),
        )
