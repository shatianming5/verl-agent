#!/usr/bin/env python3
"""Grouped, nested-CV hidden-state probes for the full Workstream B protocol."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wm_score_transition_dump as scorer  # noqa: E402
from bdiag_analyze import validate_step_provenance  # noqa: E402
from wm_validate_rollout_coverage import load_dump  # noqa: E402

from agent_system.alfworld_game_manifest import load_manifest  # noqa: E402

STEP_ORDER = ["init", "15", "30", "45", "60", "75", "90", "105", "120", "135", "150"]


def extract_episode_hiddens(model, encoded, device: str, batch_size: int):
    """Mean-pool transition endpoints within episodes; retain game groups."""

    import torch

    action_by_episode: dict[str, list[np.ndarray]] = defaultdict(list)
    observation_by_episode: dict[str, list[np.ndarray]] = defaultdict(list)
    metadata: dict[str, tuple[str, int]] = {}
    expected_episode_ids = {str(item.row["wm_episode_id"]) for item in encoded}
    use_cuda = device.startswith("cuda")
    for start in range(0, len(encoded), batch_size):
        batch = encoded[start : start + batch_size]
        input_ids = torch.tensor([item.input_ids for item in batch], dtype=torch.long, device=device)
        attention_mask = torch.tensor(
            [item.attention_mask for item in batch],
            dtype=torch.long,
            device=device,
        )
        with torch.no_grad(), torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=use_cuda,
        ):
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=True,
            )
        hidden = output.hidden_states[-1]
        for index, item in enumerate(batch):
            if item.action_end_pos is None or item.obs_end_pos is None:
                continue
            row = item.row
            episode_id = str(row["wm_episode_id"])
            game_id = str(row["wm_game_id"])
            success = scorer.episode_success(row)
            if success is None:
                raise ValueError(f"Episode success is missing for {episode_id}")
            prior = metadata.setdefault(episode_id, (game_id, int(success)))
            if prior != (game_id, int(success)):
                raise ValueError(f"Inconsistent episode metadata for {episode_id}")
            action_by_episode[episode_id].append(hidden[index, item.action_end_pos].float().cpu().numpy())
            observation_by_episode[episode_id].append(hidden[index, item.obs_end_pos].float().cpu().numpy())
        del hidden, output, input_ids, attention_mask

    episode_ids = sorted(set(action_by_episode) & set(observation_by_episode))
    if expected_episode_ids != set(episode_ids):
        missing = expected_episode_ids - set(episode_ids)
        raise ValueError(f"Episodes lack valid action/observation endpoints: {len(missing)}")
    action = np.asarray(
        [np.mean(action_by_episode[episode_id], axis=0) for episode_id in episode_ids],
        dtype=np.float32,
    )
    observation = np.asarray(
        [np.mean(observation_by_episode[episode_id], axis=0) for episode_id in episode_ids],
        dtype=np.float32,
    )
    labels = np.asarray([metadata[episode_id][1] for episode_id in episode_ids], dtype=np.int64)
    games = np.asarray([metadata[episode_id][0] for episode_id in episode_ids], dtype=object)
    return action, observation, labels, games, np.asarray(episode_ids, dtype=object)


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = _rankdata(np.asarray(scores, dtype=float))
    return float((ranks[labels == 1].sum() - positives * (positives + 1) / 2.0) / (positives * negatives))


def stratified_group_folds(
    labels: np.ndarray,
    groups: np.ndarray,
    folds: int,
    seed: int,
) -> list[np.ndarray]:
    unique_groups = sorted(set(groups.tolist()))
    group_labels = {}
    for group in unique_groups:
        group_values = labels[groups == group]
        group_labels[group] = int(group_values.mean() >= 0.5)
    class_groups = {label: [group for group in unique_groups if group_labels[group] == label] for label in (0, 1)}
    effective_folds = min(folds, len(class_groups[0]), len(class_groups[1]))
    if effective_folds < 2:
        return []
    rng = np.random.default_rng(seed)
    fold_groups = [set() for _ in range(effective_folds)]
    for label in (0, 1):
        values = class_groups[label]
        rng.shuffle(values)
        for index, group in enumerate(values):
            fold_groups[index % effective_folds].add(group)
    result = []
    for heldout_groups in fold_groups:
        indices = np.asarray(
            [index for index, group in enumerate(groups) if group in heldout_groups],
            dtype=np.int64,
        )
        result.append(indices)
    return result


def fit_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    regularization_c: float,
    epochs: int = 150,
    learning_rate: float = 0.03,
    device: str = "cpu",
) -> tuple[np.ndarray, float]:
    import torch

    x_tensor = torch.tensor(features, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(labels, dtype=torch.float32, device=device)
    weights = torch.zeros(features.shape[1], device=device, requires_grad=True)
    bias = torch.zeros(1, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([weights, bias], lr=learning_rate)
    penalty = 1.0 / (regularization_c * max(len(features), 1))
    best = float("inf")
    stale = 0
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = x_tensor @ weights + bias
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y_tensor)
        loss = loss + penalty * (weights * weights).sum()
        loss.backward()
        optimizer.step()
        current = float(loss.detach())
        if current < best - 1e-6:
            best = current
            stale = 0
        else:
            stale += 1
            if stale >= 20:
                break
    return weights.detach().cpu().numpy(), float(bias.detach().cpu().item())


def standardize(
    train: np.ndarray,
    test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (train - mean) / std, (test - mean) / std


def choose_c_nested(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    train_indices: np.ndarray,
    *,
    c_grid: list[float],
    inner_folds: int,
    seed: int,
    fit_device: str,
) -> float:
    inner = stratified_group_folds(
        labels[train_indices],
        groups[train_indices],
        inner_folds,
        seed,
    )
    if not inner:
        return c_grid[len(c_grid) // 2]
    mean_auc_by_c = []
    all_local = np.arange(len(train_indices))
    for regularization_c in c_grid:
        fold_aucs = []
        for heldout_local in inner:
            training_local = np.setdiff1d(all_local, heldout_local, assume_unique=True)
            if len(set(labels[train_indices[training_local]].tolist())) < 2:
                continue
            x_train, x_test = standardize(
                features[train_indices[training_local]],
                features[train_indices[heldout_local]],
            )
            weights, bias = fit_logistic(
                x_train,
                labels[train_indices[training_local]],
                regularization_c=regularization_c,
                device=fit_device,
            )
            fold_auc = roc_auc(
                x_test @ weights + bias,
                labels[train_indices[heldout_local]],
            )
            if math.isfinite(fold_auc):
                fold_aucs.append(fold_auc)
        mean_auc_by_c.append(float(np.mean(fold_aucs)) if fold_aucs else float("-inf"))
    return c_grid[int(np.argmax(mean_auc_by_c))]


def nested_group_probe(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    outer_folds: int = 5,
    inner_folds: int = 3,
    c_grid: list[float] | None = None,
    seed: int = 0,
    fit_device: str = "cpu",
) -> dict[str, Any]:
    c_grid = c_grid or [0.03, 0.3, 3.0]
    folds = stratified_group_folds(labels, groups, outer_folds, seed)
    if not folds:
        raise ValueError("Probe requires at least two success and two failure game groups")
    all_indices = np.arange(len(labels))
    aucs = []
    selected_cs = []
    heldout_group_sets = []
    for fold_index, test_indices in enumerate(folds):
        train_indices = np.setdiff1d(all_indices, test_indices, assume_unique=True)
        train_groups = set(groups[train_indices].tolist())
        test_groups = set(groups[test_indices].tolist())
        if train_groups & test_groups:
            raise AssertionError("Game leakage between probe train and test folds")
        selected_c = choose_c_nested(
            features,
            labels,
            groups,
            train_indices,
            c_grid=c_grid,
            inner_folds=inner_folds,
            seed=seed + fold_index + 1,
            fit_device=fit_device,
        )
        x_train, x_test = standardize(features[train_indices], features[test_indices])
        weights, bias = fit_logistic(
            x_train,
            labels[train_indices],
            regularization_c=selected_c,
            device=fit_device,
        )
        fold_auc = roc_auc(x_test @ weights + bias, labels[test_indices])
        if math.isfinite(fold_auc):
            aucs.append(fold_auc)
            selected_cs.append(selected_c)
            heldout_group_sets.append(sorted(test_groups))
    if not aucs:
        raise ValueError("No valid grouped outer probe folds")
    return {
        "probe_auc": float(np.mean(aucs)),
        "outer_fold_aucs_json": json.dumps(aucs),
        "selected_cs_json": json.dumps(selected_cs),
        "outer_folds": len(aucs),
        "heldout_game_folds_json": json.dumps(heldout_group_sets),
    }


def group_label_shuffle(labels: np.ndarray, groups: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    unique_groups = sorted(set(groups.tolist()))
    group_labels = [int(labels[groups == group].mean() >= 0.5) for group in unique_groups]
    rng.shuffle(group_labels)
    mapping = dict(zip(unique_groups, group_labels))
    return np.asarray([mapping[group] for group in groups], dtype=np.int64)


def load_rollout_rows(
    dump_root: str,
    step: str,
    *,
    manifest: dict[str, Any],
    manifest_path: str,
) -> list[dict[str, Any]]:
    step_dir = Path(dump_root) / f"step{step}"
    coverage_path = step_dir / "coverage.json"
    dumps = sorted(step_dir.glob("*.wm_transitions.jsonl"))
    if len(dumps) != 1 or not coverage_path.is_file():
        raise FileNotFoundError(f"Missing unique rollout/coverage for probe checkpoint {step}")
    dump_rows = load_dump(str(dumps[0]))
    validate_step_provenance(
        step=step,
        manifest_path=manifest_path,
        dump_path=dumps[0],
        scores_path=step_dir / "scores.csv",
        coverage_path=coverage_path,
        score_summary_path=step_dir / "score_summary.json",
        manifest=manifest,
        dump_rows=dump_rows,
    )
    rows = [row for row in dump_rows if scorer._to_bool(row.get("wm_schedule_padding")) is False]
    if not rows:
        raise ValueError(f"No non-padding rollout rows for probe checkpoint {step}")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--ckpt-root", required=True)
    parser.add_argument("--dump-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--exp", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--c-grid", default="0.03,0.3,3.0")
    parser.add_argument("--fit-device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    c_grid = [float(value) for value in args.c_grid.split(",")]
    if any(value <= 0 for value in c_grid):
        raise ValueError("All probe C values must be positive")

    manifest = load_manifest(
        args.manifest,
        expected_games=3553,
        expected_raw_trajectories=6374,
        require_train=True,
        verify_files=False,
    )
    rollout_rows = {
        step: load_rollout_rows(
            args.dump_root,
            step,
            manifest=manifest,
            manifest_path=args.manifest,
        )
        for step in STEP_ORDER
    }
    for step in STEP_ORDER[1:]:
        actor = Path(args.ckpt_root) / f"global_step_{step}" / "actor"
        if not actor.is_dir():
            raise FileNotFoundError(f"Missing required probe checkpoint: {actor}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=False)
    output_rows = []
    for step_index, step in enumerate(STEP_ORDER):
        checkpoint_path = "base" if step == "init" else str(Path(args.ckpt_root) / f"global_step_{step}")
        checkpoint = scorer.CheckpointSpec(label=f"{args.exp}_step{step}", path=checkpoint_path)
        print(f"PROBE_LOAD step={step} checkpoint={checkpoint_path}", flush=True)
        model = scorer.load_model(
            args.model_path,
            checkpoint,
            device=args.device,
            dtype_name=args.dtype,
        )
        encoded = scorer.encode_transitions(
            rollout_rows[step],
            tokenizer,
            args.max_length,
        )
        action, observation, labels, games, episode_ids = extract_episode_hiddens(
            model,
            encoded,
            args.device,
            args.batch_size,
        )
        del model, encoded
        if args.device.startswith("cuda"):
            import torch

            torch.cuda.empty_cache()
        gc.collect()

        shuffled_labels = group_label_shuffle(labels, games, args.seed + step_index)
        for feature_name, interpretation, features in (
            ("action_end_hidden", "predictive", action),
            ("observation_end_hidden", "descriptive_only", observation),
        ):
            result = nested_group_probe(
                features,
                labels,
                games,
                outer_folds=args.outer_folds,
                inner_folds=args.inner_folds,
                c_grid=c_grid,
                seed=args.seed + step_index,
                fit_device=args.fit_device,
            )
            chance = nested_group_probe(
                features,
                shuffled_labels,
                games,
                outer_folds=args.outer_folds,
                inner_folds=args.inner_folds,
                c_grid=c_grid,
                seed=args.seed + 1000 + step_index,
                fit_device=args.fit_device,
            )
            output_rows.append(
                {
                    "exp": args.exp,
                    "step": step,
                    "feature": feature_name,
                    "interpretation": interpretation,
                    "probe_auc": result["probe_auc"],
                    "group_shuffled_auc": chance["probe_auc"],
                    "episodes": len(episode_ids),
                    "games": len(set(games.tolist())),
                    "n_success": int(labels.sum()),
                    "n_failure": int(len(labels) - labels.sum()),
                    "outer_folds": result["outer_folds"],
                    "inner_folds": args.inner_folds,
                    "outer_fold_aucs_json": result["outer_fold_aucs_json"],
                    "selected_cs_json": result["selected_cs_json"],
                    "heldout_game_folds_json": result["heldout_game_folds_json"],
                }
            )
            print(
                f"PROBE step={step} feature={feature_name} auc={result['probe_auc']:.4f} group_shuffled={chance['probe_auc']:.4f} games={len(set(games.tolist()))}",
                flush=True,
            )

    output_path = Path(args.out_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(f"Probe output was not created: {output_path}")
    print(f"BDIAG_PROBE_DONE out_csv={output_path}")


if __name__ == "__main__":
    main()
