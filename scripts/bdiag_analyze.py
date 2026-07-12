#!/usr/bin/env python3
"""Fail-closed Workstream B analysis over checkpoint-specific full-train rollouts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from wm_validate_rollout_coverage import (  # noqa: E402
    load_dump,
    validate_rollout,
)

from agent_system.alfworld_game_manifest import load_manifest  # noqa: E402

STEP_ORDER = ["init", "15", "30", "45", "60", "75", "90", "105", "120", "135", "150"]
METRICS = ["ce", "nll", "perplexity", "target_confidence_mean", "raw_action_obs_cosine"]
METRIC_LABELS = {
    "ce": "teacher-forced next-observation CE / NLL per token",
    "nll": "teacher-forced next-observation total NLL",
    "perplexity": "teacher-forced next-observation perplexity",
    "target_confidence_mean": "target-token confidence",
    "raw_action_obs_cosine": "raw action-end ↔ observation-end cosine",
}


def finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_bool(value: Any) -> bool | None:
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


def bind_score_to_dump(
    score: dict[str, Any],
    source: dict[str, Any],
    transition_id: str,
) -> None:
    string_fields = (
        "workstream_b_protocol",
        "split",
        "wm_game_id",
        "wm_gamefile",
        "wm_task_type",
        "wm_game_sha256",
        "wm_manifest_sha256",
        "wm_episode_id",
        "traj_uid",
        "rollout_checkpoint_step",
    )
    integer_fields = (
        "wm_schedule_index",
        "wm_trajectory_index",
        "wm_step_idx",
        "rollout_top_k",
        "rollout_n",
    )
    boolean_fields = (
        "wm_schedule_padding",
        "episode_success",
        "rollout_do_sample",
    )
    float_fields = (
        "episode_rewards",
        "score",
        "rollout_temperature",
        "rollout_top_p",
    )
    for field in string_fields:
        if field not in score or field not in source or str(score[field]) != str(source[field]):
            raise ValueError(f"Score/dump immutable field mismatch for {transition_id}: {field}")
    for field in integer_fields:
        try:
            score_value = int(score[field])
            source_value = int(source[field])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Score/dump integer field missing for {transition_id}: {field}") from exc
        if score_value != source_value:
            raise ValueError(f"Score/dump immutable field mismatch for {transition_id}: {field}")
    for field in boolean_fields:
        score_value = parse_bool(score.get(field))
        source_value = parse_bool(source.get(field))
        if score_value is None or source_value is None or score_value != source_value:
            raise ValueError(f"Score/dump immutable field mismatch for {transition_id}: {field}")
    for field in float_fields:
        score_value = finite_float(score.get(field))
        source_value = finite_float(source.get(field))
        if score_value is None or source_value is None or score_value != source_value:
            raise ValueError(f"Score/dump immutable field mismatch for {transition_id}: {field}")


def parse_json_list(value: Any, field: str) -> list[float]:
    if value in (None, ""):
        return []
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
        result = [float(item) for item in parsed]
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid JSON numeric list in {field}: {value!r}") from exc
    if any(not math.isfinite(item) for item in result):
        raise ValueError(f"Non-finite value in {field}")
    return result


def _stable_transition_id(row: dict[str, Any], fallback_index: int) -> str:
    if row.get("stable_transition_id"):
        return str(row["stable_transition_id"])
    payload = (f"{row.get('wm_manifest_sha256', 'legacy')}\0{row.get('wm_episode_id') or row.get('traj_uid') or fallback_index}\0{row.get('wm_step_idx', fallback_index)}").encode()
    return hashlib.sha256(payload).hexdigest()


def validate_step_provenance(
    *,
    step: str,
    manifest_path: str,
    dump_path: Path,
    scores_path: Path,
    coverage_path: Path,
    score_summary_path: Path,
    manifest: dict[str, Any],
    dump_rows: list[dict[str, Any]],
) -> None:
    recomputed = validate_rollout(
        manifest,
        dump_rows,
        expected_checkpoint_step=step,
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        do_sample=True,
    )
    with coverage_path.open(encoding="utf-8") as handle:
        recorded = json.load(handle)
    required_coverage_fields = (
        "schema_version",
        "protocol",
        "manifest_sha256",
        "raw_traj_data_count",
        "manifest_games",
        "covered_games",
        "min_trajectories_per_game",
        "episodes",
        "total_episodes_with_padding",
        "padding_episodes",
        "success_episodes",
        "failure_episodes",
        "transitions",
        "stable_transition_ids",
        "checkpoint_step",
        "decoding",
    )
    missing = [field for field in required_coverage_fields if field not in recorded]
    if missing:
        raise ValueError(f"Coverage proof is incomplete at step {step}: missing={missing}")
    for field in required_coverage_fields:
        if recorded[field] != recomputed[field]:
            raise ValueError(f"Coverage proof mismatch at step {step}: {field} recorded={recorded[field]!r} recomputed={recomputed[field]!r}")
    if os.path.realpath(str(recorded.get("manifest_path", ""))) != os.path.realpath(manifest_path):
        raise ValueError(f"Coverage manifest provenance mismatch at step {step}")
    if os.path.realpath(str(recorded.get("dump_path", ""))) != os.path.realpath(dump_path):
        raise ValueError(f"Coverage dump provenance mismatch at step {step}")

    if not score_summary_path.is_file():
        raise FileNotFoundError(f"Missing full-protocol score summary for checkpoint {step}: {score_summary_path}")
    with score_summary_path.open(encoding="utf-8") as handle:
        score_summary = json.load(handle)
    provenance = score_summary.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError(f"Missing scorer provenance at step {step}")
    if score_summary.get("raw_cosine_only") is not True:
        raise ValueError(f"Scorer is not raw-cosine-only at step {step}")
    if score_summary.get("rows") != len(dump_rows):
        raise ValueError(f"Scorer row count mismatch at step {step}")
    if score_summary.get("coverage") != recomputed:
        raise ValueError(f"Scorer coverage proof mismatch at step {step}")
    if os.path.realpath(str(score_summary.get("transition_jsonl", ""))) != os.path.realpath(dump_path):
        raise ValueError(f"Scorer transition provenance mismatch at step {step}")
    if provenance.get("require_full_protocol") is not True:
        raise ValueError(f"Scorer did not require the full protocol at step {step}")
    if provenance.get("max_rows") not in (0, None):
        raise ValueError(f"Scorer used a row cap at step {step}")
    if str(provenance.get("expected_checkpoint_step")) != str(step):
        raise ValueError(f"Scorer checkpoint provenance mismatch at step {step}")
    if os.path.realpath(str(provenance.get("manifest", ""))) != os.path.realpath(manifest_path):
        raise ValueError(f"Scorer manifest provenance mismatch at step {step}")
    if provenance.get("expected_games") != 3553:
        raise ValueError(f"Scorer game-count provenance mismatch at step {step}")
    if provenance.get("expected_raw_trajectories") != 6374:
        raise ValueError(f"Scorer raw-count provenance mismatch at step {step}")
    if os.path.realpath(str(provenance.get("output_csv", ""))) != os.path.realpath(scores_path):
        raise ValueError(f"Scorer output CSV provenance mismatch at step {step}")
    if provenance.get("output_csv_sha256") != sha256_file(scores_path):
        raise ValueError(f"Scorer output CSV SHA256 mismatch at step {step}")
    if os.path.realpath(str(provenance.get("transition_jsonl", ""))) != os.path.realpath(dump_path):
        raise ValueError(f"Scorer dump provenance mismatch at step {step}")
    if provenance.get("transition_jsonl_sha256") != sha256_file(dump_path):
        raise ValueError(f"Scorer dump SHA256 mismatch at step {step}")
    checkpoints = provenance.get("checkpoints")
    if not isinstance(checkpoints, list) or len(checkpoints) != 1:
        raise ValueError(f"Scorer must contain exactly one checkpoint at step {step}")
    if str(checkpoints[0].get("step")) != str(step):
        raise ValueError(f"Scored checkpoint does not match rollout step {step}")


def load_step(
    dump_root: str,
    step: str,
    *,
    manifest: dict[str, Any],
    manifest_path: str,
) -> list[dict[str, Any]]:
    step_dir = Path(dump_root) / f"step{step}"
    scores_path = step_dir / "scores.csv"
    coverage_path = step_dir / "coverage.json"
    score_summary_path = step_dir / "score_summary.json"
    dumps = sorted(step_dir.glob("*.wm_transitions.jsonl"))
    if not scores_path.is_file():
        raise FileNotFoundError(f"Missing score CSV for checkpoint {step}: {scores_path}")
    if not coverage_path.is_file():
        raise FileNotFoundError(f"Missing coverage proof for checkpoint {step}: {coverage_path}")
    if len(dumps) != 1:
        raise ValueError(f"Expected exactly one rollout dump for checkpoint {step}, found {len(dumps)}")
    dump_rows_list = load_dump(str(dumps[0]))
    validate_step_provenance(
        step=step,
        manifest_path=manifest_path,
        dump_path=dumps[0],
        scores_path=scores_path,
        coverage_path=coverage_path,
        score_summary_path=score_summary_path,
        manifest=manifest,
        dump_rows=dump_rows_list,
    )
    dump_rows: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(dump_rows_list):
        transition_id = _stable_transition_id(row, index)
        if transition_id in dump_rows:
            raise ValueError(f"Duplicate transition id in {dumps[0]}: {transition_id}")
        dump_rows[transition_id] = row

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with scores_path.open(encoding="utf-8", newline="") as handle:
        for score_index, score in enumerate(csv.DictReader(handle)):
            if str(score.get("checkpoint_step")) != str(step):
                raise ValueError(f"{scores_path}: score checkpoint={score.get('checkpoint_step')!r}, expected={step!r}")
            transition_id = _stable_transition_id(score, score_index)
            if transition_id in seen:
                raise ValueError(f"Duplicate scored transition id: {transition_id}")
            seen.add(transition_id)
            source = dump_rows.get(transition_id)
            if source is None:
                raise ValueError(f"Score row has no matching rollout transition: {transition_id}")
            if parse_bool(source.get("wm_schedule_padding")):
                continue
            bind_score_to_dump(score, source, transition_id)
            success = parse_bool(source.get("episode_success"))
            if success is None:
                raise ValueError(f"Missing success label for transition {transition_id}")
            raw_cosine = finite_float(score.get("raw_action_obs_cosine"))
            alias_cosine = finite_float(score.get("action_obs_cosine"))
            semantics = score.get("action_obs_cosine_semantics")
            if raw_cosine is None or not -1.0 <= raw_cosine <= 1.0 or alias_cosine != raw_cosine:
                raise ValueError(f"Non-raw action_obs_cosine alias for transition {transition_id}")
            if semantics != "raw_action_end_to_observation_end":
                raise ValueError(f"Unexpected cosine semantics for transition {transition_id}: {semantics!r}")
            if parse_bool(score.get("cosine_endpoint_valid")) is not True:
                raise ValueError(f"Invalid cosine endpoint for transition {transition_id}")

            target_tokens = int(score.get("target_tokens") or 0)
            target_tokens_original = int(score.get("target_tokens_original") or 0)
            nll_sum = finite_float(score.get("nll_sum"))
            ce = finite_float(score.get("ce"))
            perplexity = finite_float(score.get("perplexity"))
            confidence = finite_float(score.get("target_confidence_mean"))
            if target_tokens <= 0 or target_tokens_original != target_tokens or parse_bool(score.get("target_truncated")) is not False or nll_sum is None or nll_sum < 0 or ce is None or ce < 0 or perplexity is None or confidence is None or not 0.0 <= confidence <= 1.0:
                raise ValueError(f"Missing teacher-forced score for active transition {transition_id}")
            coverage = finite_float(score.get("target_token_coverage"))
            if coverage is None or not math.isclose(
                coverage,
                1.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(f"Incomplete target-token coverage for {transition_id}")
            token_nlls = parse_json_list(score.get("token_nlls_json"), "token_nlls_json")
            target_token_ids = parse_json_list(
                score.get("target_token_ids_json"),
                "target_token_ids_json",
            )
            target_confidences = parse_json_list(
                score.get("target_token_confidences_json"),
                "target_token_confidences_json",
            )
            predicted_confidences = parse_json_list(
                score.get("predicted_token_confidences_json"),
                "predicted_token_confidences_json",
            )
            predicted_correct = parse_json_list(
                score.get("predicted_token_correct_json"),
                "predicted_token_correct_json",
            )
            token_lengths = {
                len(token_nlls),
                len(target_token_ids),
                len(target_confidences),
                len(predicted_confidences),
                len(predicted_correct),
                target_tokens,
            }
            if len(token_lengths) != 1:
                raise ValueError(f"Per-token calibration arrays disagree for transition {transition_id}")
            if any(value < 0 for value in token_nlls):
                raise ValueError(f"Negative token NLL for transition {transition_id}")
            if any(value < 0 or not value.is_integer() for value in target_token_ids):
                raise ValueError(f"Invalid target token id for transition {transition_id}")
            if any(not 0.0 <= value <= 1.0 for value in target_confidences):
                raise ValueError(f"Invalid target-token confidence for transition {transition_id}")
            if any(not 0.0 <= value <= 1.0 for value in predicted_confidences):
                raise ValueError(f"Invalid predicted-token confidence for transition {transition_id}")
            if any(value not in (0.0, 1.0) for value in predicted_correct):
                raise ValueError(f"Invalid predicted-token correctness for transition {transition_id}")
            if not math.isclose(
                nll_sum,
                sum(token_nlls),
                rel_tol=1e-5,
                abs_tol=1e-4,
            ):
                raise ValueError(f"Transition/token NLL mismatch for {transition_id}")
            if not math.isclose(
                ce,
                nll_sum / target_tokens,
                rel_tol=1e-5,
                abs_tol=1e-6,
            ):
                raise ValueError(f"CE/NLL mismatch for transition {transition_id}")
            if not math.isclose(
                confidence,
                float(np.mean(target_confidences)),
                rel_tol=1e-5,
                abs_tol=1e-6,
            ):
                raise ValueError(f"Target confidence/token mismatch for transition {transition_id}")
            if not math.isclose(
                perplexity,
                math.exp(min(ce, 80.0)),
                rel_tol=1e-5,
                abs_tol=1e-6,
            ):
                raise ValueError(f"Perplexity/CE mismatch for transition {transition_id}")
            prev_next_jaccard = finite_float(score.get("prev_next_token_jaccard"))
            action_next_jaccard = finite_float(score.get("action_next_token_jaccard"))
            if prev_next_jaccard is None or not 0.0 <= prev_next_jaccard <= 1.0 or action_next_jaccard is None or not 0.0 <= action_next_jaccard <= 1.0:
                raise ValueError(f"Missing or invalid tokenizer overlap for {transition_id}")
            token_count_fields = {}
            for token_field in (
                "prev_obs_tokens",
                "action_tokens",
                "next_obs_tokens",
            ):
                token_count = finite_float(score.get(token_field))
                if token_count is None or token_count < 0 or not token_count.is_integer():
                    raise ValueError(f"Missing or invalid tokenizer count for {transition_id}: {token_field}")
                token_count_fields[token_field] = int(token_count)

            row = {
                "stable_transition_id": transition_id,
                "game_id": str(source["wm_game_id"]),
                "episode_id": str(source["wm_episode_id"]),
                "success": success,
                "task_type": str(source["wm_task_type"]),
                "wm_step_idx": int(source["wm_step_idx"]),
                "nll_sum": nll_sum,
                "nll": nll_sum,
                "target_tokens": target_tokens,
                "ce": ce,
                "perplexity": perplexity,
                "target_confidence_mean": confidence,
                "raw_action_obs_cosine": raw_cosine,
                "prev_next_token_jaccard": prev_next_jaccard,
                "action_next_token_jaccard": action_next_jaccard,
                **token_count_fields,
                "token_nlls": token_nlls,
                "target_token_confidences": target_confidences,
                "predicted_token_confidences": predicted_confidences,
                "predicted_token_correct": predicted_correct,
                "manifest_sha256": str(source["wm_manifest_sha256"]),
            }
            if not row["game_id"] or not row["episode_id"] or not row["manifest_sha256"]:
                raise ValueError(f"Missing stable game/episode/manifest identity for {transition_id}")
            rows.append(row)

    expected_nonpadding = {transition_id for transition_id, row in dump_rows.items() if not parse_bool(row.get("wm_schedule_padding"))}
    if seen != set(dump_rows):
        missing = set(dump_rows) - seen
        extra = seen - set(dump_rows)
        raise ValueError(f"Score/dump transition mismatch at step {step}: missing={len(missing)} extra={len(extra)}")
    if len(rows) != len(expected_nonpadding):
        raise ValueError(f"Unexpected non-padding transition count at step {step}")
    return rows


def episode_aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["episode_id"]].append(row)
    episodes = []
    for episode_id, group in grouped.items():
        games = {row["game_id"] for row in group}
        labels = {row["success"] for row in group}
        if len(games) != 1 or len(labels) != 1:
            raise ValueError(f"Episode metadata is inconsistent: {episode_id}")
        total_tokens = sum(row["target_tokens"] for row in group)
        total_nll = sum(row["nll_sum"] for row in group)
        ce = total_nll / total_tokens
        confidence_numerator = sum((row["target_confidence_mean"] or 0.0) * row["target_tokens"] for row in group if row["target_confidence_mean"] is not None)
        confidence_tokens = sum(row["target_tokens"] for row in group if row["target_confidence_mean"] is not None)
        cosines = [row["raw_action_obs_cosine"] for row in group if row["raw_action_obs_cosine"] is not None]
        episodes.append(
            {
                "game_id": next(iter(games)),
                "episode_id": episode_id,
                "success": next(iter(labels)),
                "ce": ce,
                "nll": total_nll,
                "perplexity": math.exp(min(ce, 80.0)),
                "target_confidence_mean": (confidence_numerator / confidence_tokens if confidence_tokens else None),
                "raw_action_obs_cosine": float(np.mean(cosines)) if cosines else None,
                "target_tokens": total_tokens,
                "nll_sum": total_nll,
            }
        )
    return episodes


def auc(values: Iterable[float], labels: Iterable[bool]) -> float:
    values_array = np.asarray(list(values), dtype=float)
    labels_array = np.asarray(list(labels), dtype=bool)
    positives = values_array[labels_array]
    negatives = values_array[~labels_array]
    if not len(positives) or not len(negatives):
        return float("nan")
    comparisons = positives[:, None] - negatives[None, :]
    return float((np.sum(comparisons > 0) + 0.5 * np.sum(comparisons == 0)) / comparisons.size)


def clustered_mean_gap_ci(
    rows: list[dict[str, Any]],
    metric: str,
    *,
    bootstrap: int,
    seed: int,
) -> dict[str, float]:
    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get(metric) is not None:
            clusters[row["game_id"]].append(row)
    if not clusters:
        return {
            key: float("nan")
            for key in (
                "mean_all",
                "mean_all_ci_lo",
                "mean_all_ci_hi",
                "mean_succ",
                "mean_succ_ci_lo",
                "mean_succ_ci_hi",
                "mean_fail",
                "mean_fail_ci_lo",
                "mean_fail_ci_hi",
                "gap",
                "gap_ci_lo",
                "gap_ci_hi",
            )
        }
    summaries = []
    for cluster in clusters.values():
        values = np.asarray([row[metric] for row in cluster], dtype=float)
        labels = np.asarray([row["success"] for row in cluster], dtype=bool)
        summaries.append(
            (
                values.sum(),
                len(values),
                values[labels].sum(),
                int(labels.sum()),
                values[~labels].sum(),
                int((~labels).sum()),
            )
        )
    array = np.asarray(summaries, dtype=float)

    def estimates(sample: np.ndarray) -> tuple[float, float, float, float]:
        all_mean = sample[:, 0].sum() / sample[:, 1].sum()
        succ_count = sample[:, 3].sum()
        fail_count = sample[:, 5].sum()
        succ_mean = sample[:, 2].sum() / succ_count if succ_count else float("nan")
        fail_mean = sample[:, 4].sum() / fail_count if fail_count else float("nan")
        return all_mean, succ_mean, fail_mean, succ_mean - fail_mean

    point = estimates(array)
    rng = np.random.default_rng(seed)
    sampled = np.empty((bootstrap, 4), dtype=float)
    for index in range(bootstrap):
        sampled[index] = estimates(array[rng.integers(0, len(array), len(array))])
    names = ("mean_all", "mean_succ", "mean_fail", "gap")
    result: dict[str, float] = {}
    for column, name in enumerate(names):
        finite = sampled[np.isfinite(sampled[:, column]), column]
        result[name] = point[column]
        result[f"{name}_ci_lo"] = float(np.percentile(finite, 2.5)) if len(finite) else float("nan")
        result[f"{name}_ci_hi"] = float(np.percentile(finite, 97.5)) if len(finite) else float("nan")
    return result


def fit_gmm2(
    values: Iterable[float],
    max_iter: int = 200,
    sample_weights: Iterable[float] | None = None,
) -> dict[str, Any] | None:
    values_array = np.asarray(list(values), dtype=float)
    if sample_weights is None:
        row_weights = np.ones(len(values_array), dtype=float)
    else:
        row_weights = np.asarray(list(sample_weights), dtype=float)
        if row_weights.shape != values_array.shape:
            raise ValueError("GMM sample weights must match values")
        if np.any(~np.isfinite(row_weights)) or np.any(row_weights < 0):
            raise ValueError("GMM sample weights must be finite and non-negative")
    keep = np.isfinite(values_array) & (row_weights > 0)
    values_array = values_array[keep]
    row_weights = row_weights[keep]
    if len(values_array) < 10 or np.ptp(values_array) <= 1e-12:
        return None
    effective_rows = float(row_weights.sum())
    means = np.quantile(values_array, [0.25, 0.75]).astype(float)
    weighted_mean = float(np.average(values_array, weights=row_weights))
    weighted_variance = float(np.average((values_array - weighted_mean) ** 2, weights=row_weights))
    variances = np.full(2, max(weighted_variance, 1e-6))
    weights = np.array([0.5, 0.5], dtype=float)
    responsibilities = np.full((len(values_array), 2), 0.5)
    previous_log_likelihood = -np.inf
    for _ in range(max_iter):
        log_density = np.column_stack([np.log(max(weights[component], 1e-12)) - 0.5 * (np.log(2.0 * np.pi * variances[component]) + (values_array - means[component]) ** 2 / variances[component]) for component in range(2)])
        maximum = log_density.max(axis=1, keepdims=True)
        log_normalizer = maximum + np.log(np.exp(log_density - maximum).sum(axis=1, keepdims=True))
        responsibilities = np.exp(log_density - log_normalizer)
        log_likelihood = float((row_weights[:, None] * log_normalizer).sum())
        weighted_responsibilities = responsibilities * row_weights[:, None]
        component_mass = weighted_responsibilities.sum(axis=0).clip(min=1e-9)
        weights = component_mass / effective_rows
        means = (weighted_responsibilities * values_array[:, None]).sum(axis=0) / component_mass
        variances = (weighted_responsibilities * (values_array[:, None] - means[None, :]) ** 2).sum(axis=0) / component_mass
        variances = variances.clip(min=1e-8)
        if abs(log_likelihood - previous_log_likelihood) <= 1e-8 * (1.0 + abs(log_likelihood)):
            break
        previous_log_likelihood = log_likelihood

    log_density = np.column_stack([np.log(max(weights[component], 1e-12)) - 0.5 * (np.log(2.0 * np.pi * variances[component]) + (values_array - means[component]) ** 2 / variances[component]) for component in range(2)])
    maximum = log_density.max(axis=1, keepdims=True)
    log_normalizer = maximum + np.log(np.exp(log_density - maximum).sum(axis=1, keepdims=True))
    responsibilities = np.exp(log_density - log_normalizer)
    log_likelihood = float((row_weights[:, None] * log_normalizer).sum())

    order = np.argsort(means)
    weights = weights[order]
    means = means[order]
    variances = variances[order]
    responsibilities = responsibilities[:, order]
    ashman_d = math.sqrt(2.0) * abs(means[1] - means[0]) / math.sqrt(variances.sum())
    standard_deviations = np.sqrt(variances)
    lower = float(np.min(means - 8.0 * standard_deviations))
    upper = float(np.max(means + 8.0 * standard_deviations))
    grid = np.linspace(lower, upper, 4096)
    densities = np.vstack([weights[component] * np.exp(-0.5 * (grid - means[component]) ** 2 / variances[component]) / math.sqrt(2.0 * math.pi * variances[component]) for component in range(2)])
    minimum_density = np.minimum(densities[0], densities[1])
    overlap = float(np.sum((minimum_density[:-1] + minimum_density[1:]) * np.diff(grid) / 2.0))
    bic = -2.0 * log_likelihood + 5.0 * math.log(effective_rows)
    return {
        "weights": weights,
        "means": means,
        "variances": variances,
        "responsibilities": responsibilities,
        "hard_components": responsibilities.argmax(axis=1),
        "log_likelihood": log_likelihood,
        "bic": bic,
        "ashman_d": float(ashman_d),
        "overlap": overlap,
    }


def gmm_with_heldout_mapping(
    rows: list[dict[str, Any]],
    metric: str,
    *,
    seed: int,
) -> dict[str, Any]:
    usable = [row for row in rows if row.get(metric) is not None]
    fit = fit_gmm2(row[metric] for row in usable)
    empty = {
        "gmm_weight_0": float("nan"),
        "gmm_weight_1": float("nan"),
        "gmm_mean_0": float("nan"),
        "gmm_mean_1": float("nan"),
        "gmm_variance_0": float("nan"),
        "gmm_variance_1": float("nan"),
        "gmm_ashman_d": float("nan"),
        "gmm_overlap": float("nan"),
        "gmm_bic": float("nan"),
        "gmm_component_0_label": "",
        "gmm_component_1_label": "",
        "gmm_mapping_train_accuracy": float("nan"),
        "gmm_mapping_heldout_accuracy": float("nan"),
        "gmm_mapping_train_games": 0,
        "gmm_mapping_heldout_games": 0,
    }
    if fit is None or len({row["success"] for row in usable}) < 2:
        return empty

    games = sorted({row["game_id"] for row in usable})
    labels_by_game: dict[str, list[bool]] = defaultdict(list)
    for row in usable:
        labels_by_game[row["game_id"]].append(row["success"])
    game_labels = {game: int(np.mean(labels_by_game[game]) >= 0.5) for game in games}
    if any(sum(game_labels[game] == label for game in games) < 2 for label in (0, 1)):
        return empty
    rng = np.random.default_rng(seed)
    train_games: set[str] = set()
    for label in (0, 1):
        class_games = [game for game in games if game_labels[game] == label]
        rng.shuffle(class_games)
        split = min(max(int(round(0.7 * len(class_games))), 1), len(class_games) - 1)
        train_games.update(class_games[:split])
    labels = np.asarray([row["success"] for row in usable], dtype=int)
    train_mask = np.asarray([row["game_id"] in train_games for row in usable], dtype=bool)
    components = fit["hard_components"]
    component_success_rates = []
    for component in range(2):
        mask = train_mask & (components == component)
        component_success_rates.append(float(labels[mask].mean()) if mask.any() else 0.5)
    if component_success_rates[0] == component_success_rates[1]:
        mapping = (0, 1)
    elif component_success_rates[0] < component_success_rates[1]:
        mapping = (0, 1)
    else:
        mapping = (1, 0)
    predicted_labels = np.asarray([mapping[component] for component in components])
    heldout_mask = ~train_mask
    result = {
        "gmm_weight_0": float(fit["weights"][0]),
        "gmm_weight_1": float(fit["weights"][1]),
        "gmm_mean_0": float(fit["means"][0]),
        "gmm_mean_1": float(fit["means"][1]),
        "gmm_variance_0": float(fit["variances"][0]),
        "gmm_variance_1": float(fit["variances"][1]),
        "gmm_ashman_d": float(fit["ashman_d"]),
        "gmm_overlap": float(fit["overlap"]),
        "gmm_bic": float(fit["bic"]),
        "gmm_component_0_label": "success" if mapping[0] else "failure",
        "gmm_component_1_label": "success" if mapping[1] else "failure",
        "gmm_mapping_train_accuracy": float((predicted_labels[train_mask] == labels[train_mask]).mean()),
        "gmm_mapping_heldout_accuracy": float((predicted_labels[heldout_mask] == labels[heldout_mask]).mean()),
        "gmm_mapping_train_games": len(train_games),
        "gmm_mapping_heldout_games": len(games) - len(train_games),
    }
    return result


def gmm_cluster_ci(
    rows: list[dict[str, Any]],
    metric: str,
    *,
    bootstrap: int,
    seed: int,
) -> dict[str, float]:
    usable = [row for row in rows if row.get(metric) is not None]
    games = sorted({row["game_id"] for row in usable})
    game_index = {game: index for index, game in enumerate(games)}
    values = np.asarray([float(row[metric]) for row in usable], dtype=float)
    value_game_indices = np.asarray(
        [game_index[row["game_id"]] for row in usable],
        dtype=np.int64,
    )
    estimates = []
    if len(games) >= 10:
        rng = np.random.default_rng(seed)
        for _ in range(bootstrap):
            sampled_game_counts = np.bincount(
                rng.integers(0, len(games), len(games)),
                minlength=len(games),
            )
            fit = fit_gmm2(
                values,
                sample_weights=sampled_game_counts[value_game_indices],
            )
            if fit is not None:
                estimates.append((fit["ashman_d"], fit["overlap"]))
    array = np.asarray(estimates, dtype=float)
    if not len(array):
        return {
            "gmm_ashman_d_ci_lo": float("nan"),
            "gmm_ashman_d_ci_hi": float("nan"),
            "gmm_overlap_ci_lo": float("nan"),
            "gmm_overlap_ci_hi": float("nan"),
        }
    return {
        "gmm_ashman_d_ci_lo": float(np.percentile(array[:, 0], 2.5)),
        "gmm_ashman_d_ci_hi": float(np.percentile(array[:, 0], 97.5)),
        "gmm_overlap_ci_lo": float(np.percentile(array[:, 1], 2.5)),
        "gmm_overlap_ci_hi": float(np.percentile(array[:, 1], 97.5)),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def calibration_rows(
    rows_by_step: dict[str, list[dict[str, Any]]],
    bins: int,
) -> list[dict[str, Any]]:
    output = []
    edges = np.linspace(0.0, 1.0, bins + 1)
    for step, rows in rows_by_step.items():
        for outcome in ("all", "success", "failure"):
            selected = [row for row in rows if outcome == "all" or row["success"] == (outcome == "success")]
            confidence = np.asarray(
                [value for row in selected for value in row["predicted_token_confidences"]],
                dtype=float,
            )
            correct = np.asarray(
                [value for row in selected for value in row["predicted_token_correct"]],
                dtype=float,
            )
            target_confidence = np.asarray(
                [value for row in selected for value in row["target_token_confidences"]],
                dtype=float,
            )
            nll = np.asarray(
                [value for row in selected for value in row["token_nlls"]],
                dtype=float,
            )
            if not len(confidence):
                raise ValueError(f"No calibration tokens for step={step} outcome={outcome}")
            bin_ids = np.minimum(np.searchsorted(edges, confidence, side="right") - 1, bins - 1)
            ece = 0.0
            per_bin = []
            for bin_index in range(bins):
                mask = bin_ids == bin_index
                if not mask.any():
                    continue
                mean_confidence = float(confidence[mask].mean())
                accuracy = float(correct[mask].mean())
                fraction = float(mask.mean())
                ece += fraction * abs(mean_confidence - accuracy)
                per_bin.append(
                    {
                        "step": step,
                        "outcome": outcome,
                        "bin": bin_index,
                        "bin_lo": edges[bin_index],
                        "bin_hi": edges[bin_index + 1],
                        "tokens": int(mask.sum()),
                        "mean_predicted_confidence": mean_confidence,
                        "top1_accuracy": accuracy,
                        "mean_target_confidence": float(target_confidence[mask].mean()),
                        "mean_token_nll": float(nll[mask].mean()),
                    }
                )
            for row in per_bin:
                row["ece"] = ece
                output.append(row)
    return output


def overlap_control(
    rows_by_step: dict[str, list[dict[str, Any]]],
    *,
    bootstrap: int,
    seed: int,
) -> list[dict[str, Any]]:
    output = []
    task_types = sorted({row["task_type"] for rows in rows_by_step.values() for row in rows})
    for step_index, (step, rows) in enumerate(rows_by_step.items()):
        usable = list(rows)
        if any(row["raw_action_obs_cosine"] is None or row["prev_next_token_jaccard"] is None or row["action_next_token_jaccard"] is None for row in usable):
            raise ValueError(f"Overlap control cannot drop transitions at step {step}")
        if len(usable) != len(rows):
            raise AssertionError("Overlap control row count changed")
        if len(usable) < 10:
            raise ValueError(f"Too few raw-cosine rows for overlap control at step {step}")
        columns = [
            np.ones(len(usable)),
            np.asarray([row["prev_next_token_jaccard"] for row in usable]),
            np.asarray([row["action_next_token_jaccard"] for row in usable]),
            np.log1p([row["prev_obs_tokens"] for row in usable]),
            np.log1p([row["action_tokens"] for row in usable]),
            np.log1p([row["next_obs_tokens"] for row in usable]),
            np.asarray([row["wm_step_idx"] for row in usable], dtype=float),
        ]
        column_names = [
            "intercept",
            "prev_next_token_jaccard",
            "action_next_token_jaccard",
            "log_prev_obs_tokens",
            "log_action_tokens",
            "log_next_obs_tokens",
            "wm_step_idx",
        ]
        for task_type in task_types[1:]:
            columns.append(np.asarray([row["task_type"] == task_type for row in usable], dtype=float))
            column_names.append(f"task_type={task_type}")
        design = np.column_stack(columns)
        cosine = np.asarray([row["raw_action_obs_cosine"] for row in usable], dtype=float)
        coefficients = np.linalg.lstsq(design, cosine, rcond=None)[0]
        residuals = cosine - design @ coefficients
        residual_rows = []
        for row, residual in zip(usable, residuals):
            residual_rows.append({**row, "residual": float(residual)})
        raw_ci = clustered_mean_gap_ci(
            usable,
            "raw_action_obs_cosine",
            bootstrap=bootstrap,
            seed=seed + step_index,
        )
        residual_ci = clustered_mean_gap_ci(
            residual_rows,
            "residual",
            bootstrap=bootstrap,
            seed=seed + 1000 + step_index,
        )
        output.append(
            {
                "step": step,
                "rows": len(usable),
                "games": len({row["game_id"] for row in usable}),
                "mean_prev_next_token_jaccard": float(np.mean([row["prev_next_token_jaccard"] for row in usable])),
                "mean_action_next_token_jaccard": float(np.mean([row["action_next_token_jaccard"] for row in usable])),
                "raw_cosine_gap": raw_ci["gap"],
                "raw_cosine_gap_ci_lo": raw_ci["gap_ci_lo"],
                "raw_cosine_gap_ci_hi": raw_ci["gap_ci_hi"],
                "residual_cosine_gap": residual_ci["gap"],
                "residual_cosine_gap_ci_lo": residual_ci["gap_ci_lo"],
                "residual_cosine_gap_ci_hi": residual_ci["gap_ci_hi"],
                "covariate_coefficients_json": json.dumps(
                    dict(zip(column_names, coefficients.tolist())),
                    sort_keys=True,
                ),
            }
        )
    return output


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(y)
    if mask.sum() < 2:
        return float("nan")
    centered = x[mask] - x[mask].mean()
    denominator = float(np.dot(centered, centered))
    return float(np.dot(centered, y[mask] - y[mask].mean()) / denominator) if denominator else float("nan")


def paired_trends(
    data_by_step_level: dict[str, dict[str, list[dict[str, Any]]]],
    *,
    bootstrap: int,
    seed: int,
) -> list[dict[str, Any]]:
    output = []
    steps = list(data_by_step_level)
    x = np.asarray([0 if step == "init" else int(step) for step in steps], dtype=float)
    for level in ("episode", "transition"):
        for metric_index, metric in enumerate(METRICS):
            per_step: dict[str, dict[str, tuple[float, bool]]] = {}
            for step in steps:
                grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
                for row in data_by_step_level[step][level]:
                    if row.get(metric) is not None:
                        grouped[row["game_id"]].append(row)
                per_step[step] = {
                    game: (
                        float(np.mean([row[metric] for row in group])),
                        bool(group[0]["success"]),
                    )
                    for game, group in grouped.items()
                }
            common_games = sorted(set.intersection(*(set(per_step[step]) for step in steps)))
            if not common_games:
                raise ValueError(f"No paired games for trend level={level} metric={metric}")

            value_matrix = np.asarray(
                [[per_step[step][game][0] for game in common_games] for step in steps],
                dtype=float,
            )
            label_matrix = np.asarray(
                [[per_step[step][game][1] for game in common_games] for step in steps],
                dtype=bool,
            )
            point_series = {
                "mean_all": value_matrix.mean(axis=1),
                "mean_succ": np.asarray([values[labels].mean() if labels.any() else float("nan") for values, labels in zip(value_matrix, label_matrix)]),
                "mean_fail": np.asarray([values[~labels].mean() if (~labels).any() else float("nan") for values, labels in zip(value_matrix, label_matrix)]),
            }
            point_series["gap"] = point_series["mean_succ"] - point_series["mean_fail"]

            rng = np.random.default_rng(seed + metric_index * 100 + (0 if level == "episode" else 10_000))
            sampled_indices = rng.integers(
                0,
                len(common_games),
                size=(bootstrap, len(common_games)),
            )
            bootstrap_series = {statistic: np.empty((bootstrap, len(steps)), dtype=float) for statistic in ("mean_all", "mean_succ", "mean_fail", "gap")}
            for step_index in range(len(steps)):
                sampled_values = value_matrix[step_index][sampled_indices]
                sampled_labels = label_matrix[step_index][sampled_indices]
                bootstrap_series["mean_all"][:, step_index] = sampled_values.mean(axis=1)
                success_counts = sampled_labels.sum(axis=1)
                failure_counts = len(common_games) - success_counts
                bootstrap_series["mean_succ"][:, step_index] = np.divide(
                    (sampled_values * sampled_labels).sum(axis=1),
                    success_counts,
                    out=np.full(bootstrap, np.nan),
                    where=success_counts > 0,
                )
                bootstrap_series["mean_fail"][:, step_index] = np.divide(
                    (sampled_values * ~sampled_labels).sum(axis=1),
                    failure_counts,
                    out=np.full(bootstrap, np.nan),
                    where=failure_counts > 0,
                )
            bootstrap_series["gap"] = bootstrap_series["mean_succ"] - bootstrap_series["mean_fail"]

            for statistic in ("mean_all", "mean_succ", "mean_fail", "gap"):
                point = _slope(x, point_series[statistic])
                slopes = np.asarray(
                    [_slope(x, values) for values in bootstrap_series[statistic]],
                    dtype=float,
                )
                finite_slopes = slopes[np.isfinite(slopes)]
                output.append(
                    {
                        "level": level,
                        "metric": metric,
                        "statistic": statistic,
                        "paired_games": len(common_games),
                        "slope_per_step": point,
                        "slope_ci_lo": float(np.percentile(finite_slopes, 2.5)),
                        "slope_ci_hi": float(np.percentile(finite_slopes, 97.5)),
                        "bootstrap_probability_nonpositive": float(np.mean(finite_slopes <= 0)),
                        "bootstrap_probability_nonnegative": float(np.mean(finite_slopes >= 0)),
                    }
                )
    return output


def generate_plots(
    out_dir: Path,
    exp: str,
    stats: list[dict[str, Any]],
    data_by_step_level: dict[str, dict[str, list[dict[str, Any]]]],
    calibration: list[dict[str, Any]],
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = []
    steps = list(data_by_step_level)
    x = [0 if step == "init" else int(step) for step in steps]
    for level in ("episode", "transition"):
        for metric in METRICS:
            selected = {row["step"]: row for row in stats if row["level"] == level and row["metric"] == metric}
            figure, axis = plt.subplots(figsize=(7, 4.5))
            for label, key, color in (
                ("all", "mean_all", "black"),
                ("success", "mean_succ", "tab:green"),
                ("failure", "mean_fail", "tab:red"),
            ):
                y = np.asarray([selected[step][key] for step in steps], dtype=float)
                lo = np.asarray([selected[step][f"{key}_ci_lo"] for step in steps], dtype=float)
                hi = np.asarray([selected[step][f"{key}_ci_hi"] for step in steps], dtype=float)
                axis.plot(x, y, marker="o", label=label, color=color)
                axis.fill_between(x, lo, hi, color=color, alpha=0.12)
            axis.set_title(f"{METRIC_LABELS[metric]} ({level}) — {exp}")
            axis.set_xlabel("checkpoint step")
            axis.set_ylabel(metric)
            axis.grid(alpha=0.25)
            axis.legend()
            path = out_dir / f"line_{level}_{metric}_{exp}.png"
            figure.tight_layout()
            figure.savefig(path, dpi=140)
            plt.close(figure)
            figures.append(path)

    histogram_steps = [step for step in ("init", "75", "150") if step in steps]
    if len(histogram_steps) != 3:
        raise ValueError("Histogram protocol requires init, 75, and 150")
    for metric in METRICS:
        figure, axes = plt.subplots(1, 3, figsize=(13, 3.8))
        for axis, step in zip(axes, histogram_steps):
            episode_rows = data_by_step_level[step]["episode"]
            success_values = [row[metric] for row in episode_rows if row["success"] and row.get(metric) is not None]
            failure_values = [row[metric] for row in episode_rows if not row["success"] and row.get(metric) is not None]
            axis.hist(success_values, bins=24, density=True, alpha=0.55, label=f"success n={len(success_values)}")
            axis.hist(failure_values, bins=24, density=True, alpha=0.55, label=f"failure n={len(failure_values)}")
            axis.set_title(f"step {step}")
            axis.legend(fontsize=7)
        figure.suptitle(f"{METRIC_LABELS[metric]} episode distributions — {exp}")
        path = out_dir / f"hist_init_mid_150_{metric}_{exp}.png"
        figure.tight_layout()
        figure.savefig(path, dpi=140)
        plt.close(figure)
        figures.append(path)

    figure, axis = plt.subplots(figsize=(6, 5))
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="ideal")
    selected_steps = {"init", "75", "150"}
    for step in ("init", "75", "150"):
        points = [row for row in calibration if row["step"] == step and row["outcome"] == "all"]
        axis.plot(
            [row["mean_predicted_confidence"] for row in points],
            [row["top1_accuracy"] for row in points],
            marker="o",
            label=f"step {step} (ECE={points[0]['ece']:.3f})",
        )
    if selected_steps - set(data_by_step_level):
        raise ValueError("Calibration plot requires init, 75, and 150")
    axis.set_xlabel("mean predicted token confidence")
    axis.set_ylabel("top-1 token accuracy")
    axis.set_title(f"Next-observation token calibration — {exp}")
    axis.grid(alpha=0.25)
    axis.legend()
    calibration_path = out_dir / f"calibration_init_mid_150_{exp}.png"
    figure.tight_layout()
    figure.savefig(calibration_path, dpi=140)
    plt.close(figure)
    figures.append(calibration_path)
    return figures


def write_report(
    path: Path,
    *,
    exp: str,
    dump_root: str,
    stats_path: Path,
    trends_path: Path,
    calibration_path: Path,
    overlap_path: Path,
    figures: list[Path],
    stats: list[dict[str, Any]],
) -> None:
    final_episode = [row for row in stats if row["step"] == "150" and row["level"] == "episode"]
    lines = [
        f"# Workstream B full-train diagnostics: {exp}",
        "",
        "This report uses checkpoint-specific rollouts over the complete validated ALFWorld train manifest.",
        "Legacy eight-trajectory validation dumps and predictor-transformed geometry are not accepted.",
        "",
        "## Protocol",
        "",
        f"- Checkpoints: {', '.join(STEP_ORDER)}",
        f"- Rollout root: `{dump_root}`",
        "- Geometry: raw last-layer action-end ↔ true encoded observation-end cosine only",
        "- Uncertainty: game-cluster bootstrap; step trends use paired-game bootstrap",
        "- Episode perplexity: `exp(sum NLL / sum target tokens)`, never mean transition perplexity",
        "- GMM label mapping: learned on grouped training games and evaluated on held-out games",
        "",
        "## Step 150 episode-level summary",
        "",
        "| Metric | all | success | failure | success−failure | 95% cluster CI | held-out GMM mapping acc |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in final_episode:
        lines.append(f"| {METRIC_LABELS[row['metric']]} | {row['mean_all']:.6g} | {row['mean_succ']:.6g} | {row['mean_fail']:.6g} | {row['gap']:.6g} | [{row['gap_ci_lo']:.6g}, {row['gap_ci_hi']:.6g}] | {row['gmm_mapping_heldout_accuracy']:.6g} |")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Statistics CSV: `{stats_path}`",
            f"- Paired trends CSV: `{trends_path}`",
            f"- Token calibration CSV: `{calibration_path}`",
            f"- Tokenizer-overlap control CSV: `{overlap_path}`",
        ]
    )
    lines.extend(f"- Figure: `{figure}`" for figure in figures)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def deterministic_seed(base: int, *parts: str) -> int:
    digest = hashlib.sha256("\0".join(parts).encode("utf-8")).digest()
    return (base + int.from_bytes(digest[:4], "big")) % (2**32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--exp", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--steps", nargs="+", default=STEP_ORDER)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--gmm-bootstrap", type=int, default=50)
    parser.add_argument("--calibration-bins", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps != STEP_ORDER:
        raise ValueError(f"Full protocol requires exactly these checkpoints: {STEP_ORDER}")
    if args.bootstrap < 20 or args.gmm_bootstrap < 10:
        raise ValueError("At least 20 cluster bootstraps and 10 GMM cluster bootstraps are required")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(
        args.manifest,
        expected_games=3553,
        expected_raw_trajectories=6374,
        require_train=True,
        verify_files=False,
    )
    rows_by_step = {
        step: load_step(
            args.dump_root,
            step,
            manifest=manifest,
            manifest_path=args.manifest,
        )
        for step in STEP_ORDER
    }
    manifest_hashes = {row["manifest_sha256"] for rows in rows_by_step.values() for row in rows}
    if manifest_hashes != {manifest["manifest_sha256"]}:
        raise ValueError(f"Checkpoint rollouts use different manifests: {manifest_hashes}")
    game_sets = [{row["game_id"] for row in rows} for rows in rows_by_step.values()]
    if any(game_set != game_sets[0] for game_set in game_sets[1:]):
        raise ValueError("Checkpoint rollouts do not cover the same game ids")

    data_by_step_level: dict[str, dict[str, list[dict[str, Any]]]] = {}
    stats = []
    for step in STEP_ORDER:
        transition_rows = rows_by_step[step]
        episode_rows = episode_aggregate(transition_rows)
        data_by_step_level[step] = {
            "transition": transition_rows,
            "episode": episode_rows,
        }
        for level, data in data_by_step_level[step].items():
            for metric in METRICS:
                usable = [row for row in data if row.get(metric) is not None]
                if len(usable) != len(data):
                    raise ValueError(f"Metric is missing rows: step={step} level={level} metric={metric}")
                labels = [row["success"] for row in usable]
                values = [row[metric] for row in usable]
                if not usable or len(set(labels)) < 2:
                    raise ValueError(f"Metric lacks both outcomes: step={step} level={level} metric={metric}")
                seed = deterministic_seed(args.seed, step, level, metric)
                cluster_stats = clustered_mean_gap_ci(
                    data,
                    metric,
                    bootstrap=args.bootstrap,
                    seed=seed,
                )
                gmm = gmm_with_heldout_mapping(data, metric, seed=seed)
                gmm_ci = gmm_cluster_ci(
                    data,
                    metric,
                    bootstrap=args.gmm_bootstrap,
                    seed=seed + 1,
                )
                stats.append(
                    {
                        "exp": args.exp,
                        "step": step,
                        "level": level,
                        "metric": metric,
                        "rows": len(usable),
                        "games": len({row["game_id"] for row in usable}),
                        "n_succ": sum(labels),
                        "n_fail": len(labels) - sum(labels),
                        **cluster_stats,
                        "auc": auc(values, labels),
                        **gmm,
                        **gmm_ci,
                    }
                )

    stats_path = out_dir / f"bdiag_stats_{args.exp}.csv"
    trends_path = out_dir / f"paired_game_trends_{args.exp}.csv"
    calibration_path = out_dir / f"token_calibration_{args.exp}.csv"
    overlap_path = out_dir / f"tokenizer_overlap_control_{args.exp}.csv"
    report_path = out_dir / f"workstream_b_report_{args.exp}.md"
    write_csv(stats_path, stats)
    trends = paired_trends(
        data_by_step_level,
        bootstrap=args.bootstrap,
        seed=args.seed + 2000,
    )
    write_csv(trends_path, trends)
    calibration = calibration_rows(rows_by_step, args.calibration_bins)
    write_csv(calibration_path, calibration)
    overlap = overlap_control(
        rows_by_step,
        bootstrap=args.bootstrap,
        seed=args.seed + 3000,
    )
    write_csv(overlap_path, overlap)
    figures = generate_plots(out_dir, args.exp, stats, data_by_step_level, calibration)
    write_report(
        report_path,
        exp=args.exp,
        dump_root=args.dump_root,
        stats_path=stats_path,
        trends_path=trends_path,
        calibration_path=calibration_path,
        overlap_path=overlap_path,
        figures=figures,
        stats=stats,
    )
    required = [stats_path, trends_path, calibration_path, overlap_path, report_path, *figures]
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError(f"Analysis artifacts are missing/empty: {missing}")
    print(f"BDIAG_ANALYZE_DONE stats={stats_path} trends={trends_path} calibration={calibration_path} overlap={overlap_path} report={report_path} figures={len(figures)}")


if __name__ == "__main__":
    main()
