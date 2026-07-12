#!/usr/bin/env python3
"""Score fixed ALFWorld world-model transition dumps across checkpoints.

The input is a JSONL file produced by the rollout code with
``schema_version == "wm_transition_v1"``.  The scorer rebuilds the same
chat-template transition used by the training objectives:

    user(current observation), assistant(action), user(next observation)

Only tokens in the next-observation user turn are scored for CE/NLL.  The
geometry metric is always the unmodified last-layer action-end to observation-
end cosine.  Predictor checkpoints are deliberately ignored by this diagnostic.
"""

from __future__ import annotations

import argparse
import csv
import gc
import glob
import hashlib
import json
import math
import os
import re
import shlex
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CheckpointSpec:
    label: str
    path: str


@dataclass(frozen=True)
class EncodedTransition:
    transition_index: int
    row: dict[str, Any]
    input_ids: list[int]
    attention_mask: list[int]
    loss_mask: list[float]
    seq_len: int
    target_tokens: int
    target_start: int
    action_end_pos: int | None
    obs_end_pos: int | None
    episode_success: bool | None
    prior_tokens_original: int = 0
    target_tokens_original: int = 0
    prefix_truncated: bool = False
    target_truncated: bool = False
    prev_obs_tokens: int = 0
    action_tokens: int = 0
    next_obs_tokens: int = 0
    prev_next_token_jaccard: float = 0.0
    action_next_token_jaccard: float = 0.0


@dataclass(frozen=True)
class TruncatedTransition:
    token_ids: list[int]
    prefix_len: int
    target_len: int
    prior_tokens_original: int
    target_tokens_original: int
    prefix_truncated: bool
    target_truncated: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="Base Hugging Face model directory.")
    parser.add_argument("--transition-jsonl", required=True, help="wm_transition_v1 JSONL dump.")
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        help="Checkpoint spec as label=path. Use label=base for the initial base model.",
    )
    parser.add_argument("--output-csv", required=True, help="Per-transition CSV output.")
    parser.add_argument("--summary-json", required=True, help="Aggregate summary JSON output.")
    parser.add_argument("--max-length", type=int, default=512, help="World-model sequence length.")
    parser.add_argument("--batch-size", type=int, default=1, help="Forward batch size.")
    parser.add_argument("--max-rows", type=int, default=0, help="Limit rows for smoke tests; 0 means all.")
    parser.add_argument("--device", default="cuda", help="cuda, cuda:0, or cpu.")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=("bfloat16", "float16", "float32"),
        help="Model dtype used for scoring.",
    )
    parser.add_argument(
        "--chat-template-kwargs-json",
        default="{}",
        help="JSON object forwarded to tokenizer.apply_chat_template.",
    )
    parser.add_argument("--skip-entropy", action="store_true", help="Skip target-token entropy.")
    parser.add_argument(
        "--require-full-protocol",
        action="store_true",
        help="Require complete Workstream B v2 metadata, coverage, scores, and decoding provenance.",
    )
    parser.add_argument("--manifest", help="ALFWorld train manifest required by --require-full-protocol.")
    parser.add_argument("--expected-checkpoint-step")
    parser.add_argument("--rollout-temperature", type=float)
    parser.add_argument("--rollout-top-p", type=float)
    parser.add_argument("--rollout-top-k", type=int)
    parser.add_argument("--rollout-do-sample", choices=("true", "false"))
    parser.add_argument("--expected-games", type=int, default=3553)
    parser.add_argument("--expected-raw-trajectories", type=int, default=6374)
    args = parser.parse_args()
    if args.max_length < 2:
        parser.error("--max-length must be at least 2")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if not args.checkpoint:
        args.checkpoint = ["base=base"]
    if args.require_full_protocol:
        required_options = {
            "--manifest": args.manifest,
            "--expected-checkpoint-step": args.expected_checkpoint_step,
            "--rollout-temperature": args.rollout_temperature,
            "--rollout-top-p": args.rollout_top_p,
            "--rollout-top-k": args.rollout_top_k,
            "--rollout-do-sample": args.rollout_do_sample,
        }
        missing_options = [name for name, value in required_options.items() if value is None]
        if missing_options:
            parser.error(f"--require-full-protocol also requires {', '.join(missing_options)}")
        if args.max_rows:
            parser.error("--max-rows is incompatible with fail-closed full-protocol coverage")
        if args.expected_games != 3553 or args.expected_raw_trajectories != 6374:
            parser.error("full protocol requires exactly 6374 raw ALFWorld train trajectories and 3553 filtered games")
    try:
        args.chat_template_kwargs = json.loads(args.chat_template_kwargs_json)
    except json.JSONDecodeError as exc:
        parser.error(f"--chat-template-kwargs-json is not valid JSON: {exc}")
    if not isinstance(args.chat_template_kwargs, dict):
        parser.error("--chat-template-kwargs-json must decode to an object")
    return args


def parse_checkpoint_specs(specs: list[str]) -> list[CheckpointSpec]:
    parsed = []
    for spec in specs:
        if "=" in spec:
            label, path = spec.split("=", 1)
        else:
            path = spec
            label = Path(path).name or "checkpoint"
        label = label.strip()
        path = path.strip()
        if not label or not path:
            raise ValueError(f"Invalid checkpoint spec: {spec!r}")
        parsed.append(CheckpointSpec(label=label, path=path))
    return parsed


def build_provenance(args: argparse.Namespace, specs: list[CheckpointSpec]) -> dict[str, Any]:
    return {
        "argv": list(sys.argv),
        "command": shlex.join(sys.argv),
        "cwd": os.getcwd(),
        "model_path": args.model_path,
        "transition_jsonl": args.transition_jsonl,
        "output_csv": args.output_csv,
        "summary_json": args.summary_json,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "max_rows": args.max_rows,
        "device": args.device,
        "dtype": args.dtype,
        "skip_entropy": bool(args.skip_entropy),
        "require_full_protocol": bool(getattr(args, "require_full_protocol", False)),
        "manifest": getattr(args, "manifest", None),
        "expected_checkpoint_step": getattr(args, "expected_checkpoint_step", None),
        "rollout_temperature": getattr(args, "rollout_temperature", None),
        "rollout_top_p": getattr(args, "rollout_top_p", None),
        "rollout_top_k": getattr(args, "rollout_top_k", None),
        "rollout_do_sample": getattr(args, "rollout_do_sample", None),
        "expected_games": getattr(args, "expected_games", None),
        "expected_raw_trajectories": getattr(args, "expected_raw_trajectories", None),
        "chat_template_kwargs_json": args.chat_template_kwargs_json,
        "chat_template_kwargs": args.chat_template_kwargs,
        "checkpoint_count": len(specs),
        "checkpoints": [
            {
                "label": spec.label,
                "path": spec.path,
                "step": checkpoint_step(spec.label, spec.path),
            }
            for spec in specs
        ],
    }


def load_transitions(path: str, max_rows: int = 0) -> list[dict[str, Any]]:
    rows = []
    required = {"schema_version", "wm_prev_obs_text", "wm_action_text", "wm_next_obs_text"}
    with open(path, encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if max_rows and len(rows) >= max_rows:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            missing = required - set(row)
            if row.get("schema_version") not in {"wm_transition_v1", "wm_transition_v2"} or missing:
                raise ValueError(f"{path}:{line_no} is not a complete supported wm_transition row; schema={row.get('schema_version')!r} missing={sorted(missing)}")
            rows.append(row)
    if not rows:
        raise ValueError(f"No transition rows loaded from {path}")
    return rows


def active_mask(row: dict[str, Any]) -> bool:
    value = row.get("active_masks", True)
    parsed = _to_bool(value)
    return bool(parsed) if parsed is not None else bool(value)


def _to_float(value: Any) -> float | None:
    if isinstance(value, list):
        value = value[0] if value else None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _to_bool(value: Any) -> bool | None:
    if isinstance(value, list):
        value = value[0] if value else None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    if value in (0, 1):
        return bool(value)
    return None


def episode_success(row: dict[str, Any]) -> bool | None:
    if "episode_success" in row:
        parsed = _to_bool(row["episode_success"])
        if parsed is not None:
            return parsed
    episode_reward = _to_float(row.get("episode_rewards"))
    if episode_reward is not None:
        return episode_reward > 0.0
    if bool(row.get("wm_done_after_action", False)):
        reward = _to_float(row.get("rewards"))
        if reward is not None:
            return reward > 0.0
    return None


def chat_transition_ids(
    tokenizer: Any,
    prev_obs_text: str,
    action_text: str,
    next_obs_text: str,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> tuple[list[int], list[int]]:
    chat_template_kwargs = chat_template_kwargs or {}
    prior_chat = [
        {"role": "user", "content": prev_obs_text},
        {"role": "assistant", "content": action_text},
    ]
    posterior_chat = [*prior_chat, {"role": "user", "content": next_obs_text}]
    prior_text = tokenizer.apply_chat_template(
        prior_chat,
        add_generation_prompt=False,
        tokenize=False,
        **chat_template_kwargs,
    )
    posterior_text = tokenizer.apply_chat_template(
        posterior_chat,
        add_generation_prompt=False,
        tokenize=False,
        **chat_template_kwargs,
    )
    prior_ids = tokenizer.encode(prior_text, add_special_tokens=False)
    if posterior_text.startswith(prior_text):
        target_text = posterior_text[len(prior_text) :]
        return prior_ids, tokenizer.encode(target_text, add_special_tokens=False)

    posterior_ids = tokenizer.encode(posterior_text, add_special_tokens=False)
    split_idx = 0
    max_prefix = min(len(prior_ids), len(posterior_ids))
    while split_idx < max_prefix and prior_ids[split_idx] == posterior_ids[split_idx]:
        split_idx += 1
    return prior_ids, posterior_ids[split_idx:]


def truncate_transition_ids(
    prior_ids: list[int],
    target_ids: list[int],
    max_length: int,
    pad_token_id: int,
) -> TruncatedTransition:
    if not prior_ids:
        prior_ids = [pad_token_id]
    prior_tokens_original = len(prior_ids)
    target_tokens_original = len(target_ids)
    target_budget = max(max_length - 1, 0)
    target_ids = list(target_ids[:target_budget])
    prefix_budget = max(max_length - len(target_ids), 1)
    prior_ids = list(prior_ids[-prefix_budget:])
    token_ids = (prior_ids + target_ids)[:max_length]
    return TruncatedTransition(
        token_ids=token_ids,
        prefix_len=len(prior_ids),
        target_len=len(target_ids),
        prior_tokens_original=prior_tokens_original,
        target_tokens_original=target_tokens_original,
        prefix_truncated=len(prior_ids) < prior_tokens_original,
        target_truncated=len(target_ids) < target_tokens_original,
    )


def token_jaccard(left: list[int], right: list[int]) -> float:
    left_set = set(left)
    right_set = set(right)
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 1.0


def encode_transitions(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    max_length: int,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> list[EncodedTransition]:
    encoded = []
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    for idx, row in enumerate(rows):
        prior_ids, target_ids = chat_transition_ids(
            tokenizer=tokenizer,
            prev_obs_text=str(row.get("wm_prev_obs_text", "")),
            action_text=str(row.get("wm_action_text", "")),
            next_obs_text=str(row.get("wm_next_obs_text", "")),
            chat_template_kwargs=chat_template_kwargs,
        )
        truncated = truncate_transition_ids(
            prior_ids=prior_ids,
            target_ids=target_ids,
            max_length=max_length,
            pad_token_id=pad_id,
        )
        token_ids = truncated.token_ids
        prefix_len = truncated.prefix_len
        target_len = truncated.target_len
        seq_len = len(token_ids)
        input_ids = [pad_id] * max_length
        attention_mask = [0] * max_length
        loss_mask = [0.0] * max_length
        if seq_len:
            input_ids[:seq_len] = token_ids
            attention_mask[:seq_len] = [1] * seq_len
            target_start = min(prefix_len, seq_len)
            if active_mask(row) and target_len > 0 and target_start < seq_len:
                for pos in range(target_start, seq_len):
                    loss_mask[pos] = 1.0
        else:
            target_start = 0

        action_end_pos = max(min(prefix_len - 1, seq_len - 1), 0) if seq_len else None
        obs_end_pos = max(seq_len - 1, 0) if seq_len > target_start and not truncated.target_truncated else None
        if action_end_pos is not None and obs_end_pos is not None and action_end_pos >= obs_end_pos:
            action_end_pos = None
            obs_end_pos = None

        prev_obs_ids = tokenizer.encode(str(row.get("wm_prev_obs_text", "")), add_special_tokens=False)
        action_ids = tokenizer.encode(str(row.get("wm_action_text", "")), add_special_tokens=False)
        next_obs_ids = tokenizer.encode(str(row.get("wm_next_obs_text", "")), add_special_tokens=False)
        encoded.append(
            EncodedTransition(
                transition_index=idx,
                row=row,
                input_ids=input_ids,
                attention_mask=attention_mask,
                loss_mask=loss_mask,
                seq_len=seq_len,
                target_tokens=int(sum(loss_mask[1:])),
                target_start=target_start,
                action_end_pos=action_end_pos,
                obs_end_pos=obs_end_pos,
                episode_success=episode_success(row),
                prior_tokens_original=truncated.prior_tokens_original,
                target_tokens_original=truncated.target_tokens_original,
                prefix_truncated=truncated.prefix_truncated,
                target_truncated=truncated.target_truncated,
                prev_obs_tokens=len(prev_obs_ids),
                action_tokens=len(action_ids),
                next_obs_tokens=len(next_obs_ids),
                prev_next_token_jaccard=token_jaccard(prev_obs_ids, next_obs_ids),
                action_next_token_jaccard=token_jaccard(action_ids, next_obs_ids),
            )
        )
    return encoded


def dtype_from_name(torch: Any, name: str) -> Any:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def resolve_actor_dir(checkpoint_path: str) -> str | None:
    if checkpoint_path.lower() in {"base", "init", "none"}:
        return None
    candidate = Path(checkpoint_path)
    actor_dir = candidate / "actor"
    if actor_dir.is_dir():
        return str(actor_dir)
    if candidate.is_dir():
        return str(candidate)
    raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")


def rank_id(path: str) -> int:
    match = re.search(r"_rank_(\d+)\.pt$", path)
    if not match:
        raise ValueError(f"Cannot parse rank id from {path}")
    return int(match.group(1))


def fsdp_shard_paths(actor_dir: str) -> list[str]:
    return sorted(glob.glob(os.path.join(actor_dir, "model_world_size_*_rank_*.pt")), key=rank_id)


def materialize_dtensor_value(values: list[Any]) -> Any:
    import torch

    first = values[0]
    if hasattr(first, "to_local") and hasattr(first, "placements"):
        placements = tuple(first.placements)
        if len(placements) != 1:
            raise NotImplementedError(f"Only one-dimensional DTensor placements are supported, got {placements}")
        placement = placements[0]
        placement_name = type(placement).__name__
        if placement_name == "Replicate":
            return first.to_local().detach().cpu()
        if placement_name == "Shard":
            dim = int(placement.dim)
            parts = [value.to_local().detach().cpu() for value in values]
            merged = torch.cat(parts, dim=dim)
            full_shape = tuple(int(size) for size in first.shape)
            if merged.shape[dim] > full_shape[dim]:
                merged = merged.narrow(dim, 0, full_shape[dim])
            return merged.contiguous()
        raise NotImplementedError(f"Unsupported DTensor placement: {placement}")
    if torch.is_tensor(first):
        return first.detach().cpu()
    return first


def load_fsdp_dtensor_state_dict(actor_dir: str) -> dict[str, Any]:
    import torch
    import torch.distributed.tensor  # noqa: F401 - required to unpickle DTensor checkpoints

    shard_paths = fsdp_shard_paths(actor_dir)
    if not shard_paths:
        return {}

    print(f"WM_SCORE_LOAD_SHARDS actor_dir={actor_dir} shards={len(shard_paths)}", flush=True)
    # These are local trusted training artifacts. PyTorch 2.6+ needs
    # weights_only=False for DTensor state-dict pickles.
    shard_dicts = [torch.load(path, map_location="cpu", weights_only=False) for path in shard_paths]
    keys = list(shard_dicts[0].keys())
    for shard_path, shard_dict in zip(shard_paths[1:], shard_dicts[1:]):
        if list(shard_dict.keys()) != keys:
            raise ValueError(f"Shard key mismatch in {shard_path}")

    merged = {}
    for key in keys:
        merged[key] = materialize_dtensor_value([shard_dict[key] for shard_dict in shard_dicts])
    del shard_dicts
    gc.collect()
    return merged


def load_model(model_path: str, checkpoint: CheckpointSpec, device: str, dtype_name: str) -> Any:
    import torch
    from transformers import AutoModelForCausalLM

    dtype = dtype_from_name(torch, dtype_name)
    actor_dir = resolve_actor_dir(checkpoint.path)
    shard_paths = fsdp_shard_paths(actor_dir) if actor_dir else []
    load_path = model_path if shard_paths else actor_dir or model_path
    print(f"WM_SCORE_LOAD_MODEL label={checkpoint.label} load_path={load_path}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        load_path,
        torch_dtype=dtype,
        trust_remote_code=False,
        low_cpu_mem_usage=True,
    )
    if actor_dir and shard_paths:
        state_dict = load_fsdp_dtensor_state_dict(actor_dir)
        if state_dict:
            missing, unexpected = model.load_state_dict(state_dict, strict=True)
            if missing or unexpected:
                raise RuntimeError(f"load_state_dict mismatch: missing={missing} unexpected={unexpected}")
            del state_dict
            gc.collect()
    model.to(device)
    model.eval()
    return model


def checkpoint_step(label: str, path: str) -> str:
    if path.lower() in {"base", "init", "none"} or "init" in label:
        return "init"
    match = re.search(r"global_step_(\d+)", path)
    return match.group(1) if match else ""


def safe_float(value: float | None) -> float | str:
    if value is None or not math.isfinite(value):
        return ""
    return float(value)


def stable_transition_id(source: dict[str, Any], transition_index: int) -> str:
    manifest_sha = str(source.get("wm_manifest_sha256", "legacy"))
    episode_id = str(source.get("wm_episode_id") or source.get("traj_uid") or transition_index)
    step_index = source.get("wm_step_idx", transition_index)
    payload = f"{manifest_sha}\0{episode_id}\0{step_index}".encode()
    return hashlib.sha256(payload).hexdigest()


def score_batch(
    model: Any,
    batch: list[EncodedTransition],
    checkpoint: CheckpointSpec,
    device: str,
    skip_entropy: bool,
) -> list[dict[str, Any]]:
    import torch
    import torch.nn.functional as F

    input_ids = torch.tensor([item.input_ids for item in batch], dtype=torch.long, device=device)
    attention_mask = torch.tensor([item.attention_mask for item in batch], dtype=torch.long, device=device)
    loss_mask = torch.tensor([item.loss_mask for item in batch], dtype=torch.float32, device=device)

    use_cuda_autocast = device.startswith("cuda")
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_cuda_autocast):
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True,
        )

    logits = outputs.logits[:, :-1, :]
    labels = input_ids[:, 1:]
    shifted_mask = loss_mask[:, 1:].bool()
    last_hidden = outputs.hidden_states[-1]

    rows = []
    for i, item in enumerate(batch):
        token_mask = shifted_mask[i]
        token_count = int(token_mask.sum().item())
        nll_sum = ce = ppl = confidence = entropy = None
        token_nlls: list[float] = []
        target_confidences: list[float] = []
        predicted_confidences: list[float] = []
        predicted_correct: list[int] = []
        target_token_ids: list[int] = []
        if token_count > 0:
            selected_logits = logits[i, token_mask, :].float()
            selected_labels = labels[i, token_mask]
            log_probs = F.log_softmax(selected_logits, dim=-1)
            token_log_probs = log_probs.gather(dim=-1, index=selected_labels.unsqueeze(-1)).squeeze(-1)
            predicted_log_probs, predicted_labels = log_probs.max(dim=-1)
            nll_sum = float((-token_log_probs).sum().item())
            ce = nll_sum / token_count
            ppl = math.exp(min(ce, 80.0))
            confidence = float(token_log_probs.exp().mean().item())
            token_nlls = (-token_log_probs).cpu().tolist()
            target_confidences = token_log_probs.exp().cpu().tolist()
            predicted_confidences = predicted_log_probs.exp().cpu().tolist()
            predicted_correct = predicted_labels.eq(selected_labels).to(torch.int64).cpu().tolist()
            target_token_ids = selected_labels.cpu().tolist()
            if not skip_entropy:
                probs = log_probs.exp()
                entropy = float((-(probs * log_probs).sum(dim=-1)).mean().item())

        action_norm = obs_norm = action_obs_cosine = raw_action_obs_cosine = None
        if item.action_end_pos is not None and item.obs_end_pos is not None:
            action_hidden = last_hidden[i, item.action_end_pos]
            obs_hidden = last_hidden[i, item.obs_end_pos]
            action_hidden_float = action_hidden.float()
            obs_hidden_float = obs_hidden.float()
            action_norm = float(action_hidden_float.norm().item())
            obs_norm = float(obs_hidden_float.norm().item())
            raw_action_obs_cosine = float(F.cosine_similarity(action_hidden_float, obs_hidden_float, dim=0).item())
            action_obs_cosine = raw_action_obs_cosine

        source = item.row
        result = {
            "checkpoint_label": checkpoint.label,
            "checkpoint_path": checkpoint.path,
            "checkpoint_step": checkpoint_step(checkpoint.label, checkpoint.path),
            "transition_index": item.transition_index,
            "stable_transition_id": stable_transition_id(source, item.transition_index),
            "global_step": source.get("global_step", ""),
            "split": source.get("split", ""),
            "batch_idx": source.get("batch_idx", ""),
            "row_idx": source.get("row_idx", ""),
            "wm_step_idx": source.get("wm_step_idx", ""),
            "wm_done_after_action": source.get("wm_done_after_action", ""),
            "episode_success": "" if item.episode_success is None else item.episode_success,
            "target_start": item.target_start,
            "seq_len": item.seq_len,
            "target_tokens": token_count,
            "prior_tokens_original": item.prior_tokens_original,
            "target_tokens_original": item.target_tokens_original,
            "prefix_truncated": item.prefix_truncated,
            "target_truncated": item.target_truncated,
            "target_token_coverage": (token_count / item.target_tokens_original if item.target_tokens_original else 1.0),
            "prev_obs_tokens": item.prev_obs_tokens,
            "action_tokens": item.action_tokens,
            "next_obs_tokens": item.next_obs_tokens,
            "prev_next_token_jaccard": item.prev_next_token_jaccard,
            "action_next_token_jaccard": item.action_next_token_jaccard,
            "nll_sum": safe_float(nll_sum),
            "ce": safe_float(ce),
            "perplexity": safe_float(ppl),
            "target_confidence_mean": safe_float(confidence),
            "target_entropy_mean": safe_float(entropy),
            "target_token_ids_json": json.dumps(target_token_ids, separators=(",", ":")),
            "token_nlls_json": json.dumps(token_nlls, separators=(",", ":")),
            "target_token_confidences_json": json.dumps(target_confidences, separators=(",", ":")),
            "predicted_token_confidences_json": json.dumps(predicted_confidences, separators=(",", ":")),
            "predicted_token_correct_json": json.dumps(predicted_correct, separators=(",", ":")),
            "action_hidden_norm": safe_float(action_norm),
            "obs_hidden_norm": safe_float(obs_norm),
            "raw_action_obs_cosine": safe_float(raw_action_obs_cosine),
            "action_obs_cosine": safe_float(action_obs_cosine),
            "action_obs_cosine_semantics": "raw_action_end_to_observation_end",
            "cosine_endpoint_valid": not item.target_truncated,
        }
        source_metadata_keys = {
            "workstream_b_protocol",
            "rollout_checkpoint_step",
            "rollout_temperature",
            "rollout_top_p",
            "rollout_top_k",
            "rollout_do_sample",
            "rollout_n",
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
            "score",
        }
        for key, value in source.items():
            if "success_rate" in key or key in {
                "rewards",
                "active_masks",
                "is_action_valid",
                "episode_rewards",
                "episode_lengths",
                *source_metadata_keys,
            }:
                result[key] = value
        rows.append(result)
    return rows


def score_checkpoint(
    args: argparse.Namespace,
    encoded: list[EncodedTransition],
    checkpoint: CheckpointSpec,
) -> list[dict[str, Any]]:
    import torch

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    model = load_model(args.model_path, checkpoint, device=device, dtype_name=args.dtype)
    results = []
    print(f"WM_SCORE_CHECKPOINT_START label={checkpoint.label} rows={len(encoded)}", flush=True)
    for start in range(0, len(encoded), args.batch_size):
        batch = encoded[start : start + args.batch_size]
        results.extend(
            score_batch(
                model,
                batch,
                checkpoint,
                device=device,
                skip_entropy=args.skip_entropy,
            )
        )
        if (start // args.batch_size + 1) % 10 == 0:
            print(f"WM_SCORE_PROGRESS label={checkpoint.label} rows={len(results)}/{len(encoded)}", flush=True)
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    gc.collect()
    print(f"WM_SCORE_CHECKPOINT_DONE label={checkpoint.label} rows={len(results)}", flush=True)
    return results


def mean(values: list[float]) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else None


def numeric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [float(row[key]) for row in rows if row.get(key) not in ("", None)]


def summarize_metric_group(group: list[dict[str, Any]]) -> dict[str, Any]:
    token_rows = [row for row in group if isinstance(row.get("target_tokens"), int) and row["target_tokens"] > 0]
    total_tokens = sum(int(row["target_tokens"]) for row in token_rows)
    total_nll = sum(numeric_values(token_rows, "nll_sum"))
    token_ce = total_nll / total_tokens if total_tokens else None
    return {
        "rows": len(group),
        "rows_with_targets": len(token_rows),
        "target_tokens": total_tokens,
        "token_mean_ce": safe_float(token_ce),
        "token_mean_perplexity": safe_float(math.exp(min(token_ce, 80.0)) if token_ce is not None else None),
        "row_mean_ce": safe_float(mean(numeric_values(token_rows, "ce"))),
        "row_mean_target_confidence": safe_float(mean(numeric_values(token_rows, "target_confidence_mean"))),
        "row_mean_target_entropy": safe_float(mean(numeric_values(token_rows, "target_entropy_mean"))),
        "row_mean_action_hidden_norm": safe_float(mean(numeric_values(group, "action_hidden_norm"))),
        "row_mean_obs_hidden_norm": safe_float(mean(numeric_values(group, "obs_hidden_norm"))),
        "row_mean_raw_action_obs_cosine": safe_float(mean(numeric_values(group, "raw_action_obs_cosine"))),
        "row_mean_action_obs_cosine": safe_float(mean(numeric_values(group, "action_obs_cosine"))),
        "prefix_truncated_rows": sum(1 for row in group if _to_bool(row.get("prefix_truncated"))),
        "target_truncated_rows": sum(1 for row in group if _to_bool(row.get("target_truncated"))),
        "raw_cosine_rows": len(numeric_values(group, "raw_action_obs_cosine")),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_label: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_label.setdefault(str(row["checkpoint_label"]), []).append(row)

    checkpoint_summaries = []
    success_buckets = []
    for label, group in by_label.items():
        summary = summarize_metric_group(group)
        summary.update(
            {
                "checkpoint_label": label,
                "checkpoint_path": group[0].get("checkpoint_path", ""),
                "checkpoint_step": group[0].get("checkpoint_step", ""),
            }
        )
        checkpoint_summaries.append(summary)

        for success_value in (True, False):
            subgroup = [row for row in group if _to_bool(row.get("episode_success")) is success_value]
            if subgroup:
                bucket = summarize_metric_group(subgroup)
                bucket.update(
                    {
                        "checkpoint_label": label,
                        "checkpoint_step": group[0].get("checkpoint_step", ""),
                        "episode_success": success_value,
                    }
                )
                success_buckets.append(bucket)

    return {"checkpoints": checkpoint_summaries, "success_buckets": success_buckets}


def atomic_write(path: str, writer: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_name = ""
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            delete=False,
            dir=str(target.parent),
            prefix=f".{target.name}.",
            suffix=".tmp",
            encoding="utf-8",
            newline="",
        ) as handle:
            temp_name = handle.name
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, target)
    finally:
        if temp_name:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass


def write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    def writer(handle: Any) -> None:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    atomic_write(path, writer)


def write_json(path: str, value: Any) -> None:
    def writer(handle: Any) -> None:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")

    atomic_write(path, writer)


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    specs = parse_checkpoint_specs(args.checkpoint)
    coverage_summary = None
    if args.require_full_protocol:
        if len(specs) != 1:
            raise ValueError("Full Workstream B scoring requires exactly one checkpoint and that checkpoint's own rollout")
        spec_step = checkpoint_step(specs[0].label, specs[0].path)
        if str(spec_step) != str(args.expected_checkpoint_step):
            raise ValueError(f"Scored checkpoint step={spec_step!r} does not match rollout step={args.expected_checkpoint_step!r}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=False)
    transitions = load_transitions(args.transition_jsonl, args.max_rows)
    if args.require_full_protocol:
        from wm_validate_rollout_coverage import validate_rollout

        from agent_system.alfworld_game_manifest import (
            load_manifest,
        )

        manifest = load_manifest(
            args.manifest,
            expected_games=args.expected_games,
            expected_raw_trajectories=args.expected_raw_trajectories,
            require_train=True,
            verify_files=False,
        )
        coverage_summary = validate_rollout(
            manifest,
            transitions,
            expected_checkpoint_step=args.expected_checkpoint_step,
            temperature=args.rollout_temperature,
            top_p=args.rollout_top_p,
            top_k=args.rollout_top_k,
            do_sample=args.rollout_do_sample == "true",
        )
    encoded = encode_transitions(
        transitions,
        tokenizer=tokenizer,
        max_length=args.max_length,
        chat_template_kwargs=args.chat_template_kwargs,
    )
    if args.require_full_protocol:
        target_truncated = [item.transition_index for item in encoded if item.target_truncated]
        missing_targets = [item.transition_index for item in encoded if item.target_tokens <= 0]
        if target_truncated or missing_targets:
            raise ValueError(f"Full protocol requires every transition's complete next-observation endpoint and score: target_truncated={len(target_truncated)} missing_targets={len(missing_targets)}")
    print(
        f"WM_SCORE_START transitions={args.transition_jsonl} rows={len(encoded)} checkpoints={len(specs)}",
        flush=True,
    )

    all_rows = []
    for checkpoint in specs:
        all_rows.extend(score_checkpoint(args, encoded, checkpoint))
    if args.require_full_protocol:
        missing_raw_cosine = [row["stable_transition_id"] for row in all_rows if row.get("raw_action_obs_cosine") in ("", None) or row.get("action_obs_cosine") != row.get("raw_action_obs_cosine")]
        if missing_raw_cosine:
            raise ValueError(f"Full protocol raw action-end/observation-end cosine is missing or aliased for {len(missing_raw_cosine)} transitions")

    write_csv(args.output_csv, all_rows)
    summary = summarize(all_rows)
    summary["transition_jsonl"] = args.transition_jsonl
    summary["max_length"] = args.max_length
    summary["rows"] = len(encoded)
    summary["provenance"] = build_provenance(args, specs)
    summary["provenance"]["output_csv"] = os.path.realpath(args.output_csv)
    summary["provenance"]["output_csv_sha256"] = sha256_file(args.output_csv)
    summary["provenance"]["transition_jsonl"] = os.path.realpath(args.transition_jsonl)
    summary["provenance"]["transition_jsonl_sha256"] = sha256_file(args.transition_jsonl)
    if coverage_summary is not None:
        summary["coverage"] = coverage_summary
        summary["raw_cosine_only"] = True
    write_json(args.summary_json, summary)
    print(f"WM_SCORE_DONE csv={args.output_csv} summary={args.summary_json}", flush=True)


if __name__ == "__main__":
    main()
