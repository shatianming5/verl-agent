# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import itertools
import logging
import os
from typing import Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, compute_policy_loss_gspo, kl_penalty
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_device_name, get_torch_device, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.actor.world_model import LatentTransitionPredictor

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input


__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.world_model_predictor = None
        self.world_model_predictor_config = {}
        self.latent_world_model_enabled = False
        self.world_model_loss_coef = 0.0
        self._last_world_model_grad_norm = None

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        print(f"Actor use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        print(f"Actor use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.compute_entropy_from_logits = (
            torch.compile(verl_F.entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else verl_F.entropy_from_logits
        )
        self.device_name = get_device_name()
        self._init_latent_world_model()

    def _actor_config(self):
        module = self.actor_module
        if hasattr(module, "_fsdp_wrapped_module"):
            module = module._fsdp_wrapped_module
        return getattr(module, "config", None)

    def _infer_hidden_size(self) -> int:
        config = self._actor_config()
        candidate_configs = [config, getattr(config, "text_config", None)] if config is not None else []
        for candidate in candidate_configs:
            hidden_size = getattr(candidate, "hidden_size", None)
            if hidden_size is not None:
                return int(hidden_size)
        raise ValueError("Cannot infer actor hidden_size for latent world-model predictor.")

    def _first_actor_parameter(self):
        return next(self.actor_module.parameters())

    def _world_model_config(self):
        return self.config.get("world_model", {})

    def _obs_ce_coef(self) -> float:
        world_model_config = self._world_model_config()
        return float(world_model_config.get("lambda_obs", world_model_config.get("obs_ce_coef", 0.0)) or 0.0)

    def _obs_ce_enabled(self) -> bool:
        return self._obs_ce_coef() > 0.0

    def _latent_loss_coef(self) -> float:
        world_model_config = self._world_model_config()
        lambda_latent = float(world_model_config.get("lambda_latent", 0.0) or 0.0)
        if lambda_latent > 0.0:
            return lambda_latent
        return float(world_model_config.get("latent_loss_coef", 0.0) or 0.0)

    def _latent_use_predictor(self) -> bool:
        """Whether to route the prior (action-end) hidden through a learned
        LatentTransitionPredictor before the cosine loss. Default False: the
        Workstream C objective is the direct L_latent = 1 - cos(h_action, sg(h_obs))."""
        return bool(self._world_model_config().get("latent_use_predictor", False))

    def _broadcast_world_model_parameters(self):
        if self.world_model_predictor is None:
            return
        if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1:
            return
        for parameter in self.world_model_predictor.parameters():
            dist.broadcast(parameter.data, src=0)

    def _init_latent_world_model(self):
        world_model_config = self._world_model_config()
        self.world_model_loss_coef = self._latent_loss_coef()
        if self.world_model_loss_coef <= 0 or self.actor_optimizer is None:
            return
        if self.use_ulysses_sp:
            raise ValueError("actor.world_model latent loss is not supported with actor Ulysses sequence parallelism yet.")

        self.latent_world_model_enabled = True
        if not self._latent_use_predictor():
            # Default Workstream C objective: L_latent = 1 - cos(h_action, stop_gradient(h_obs)).
            # No predictor / projection head; gradient flows action_hidden -> shared transformer.
            return

        hidden_size = self._infer_hidden_size()
        predictor_hidden_size = world_model_config.get("predictor_hidden_size", None)
        if predictor_hidden_size is None:
            predictor_hidden_size = world_model_config.get("latent_predictor_hidden_size", hidden_size)
        predictor_hidden_size = int(predictor_hidden_size or hidden_size)
        dropout = world_model_config.get("predictor_dropout", None)
        if dropout is None:
            dropout = world_model_config.get("latent_predictor_dropout", 0.0)
        dropout = float(dropout or 0.0)
        residual = bool(world_model_config.get("predictor_residual", True))
        first_param = self._first_actor_parameter()

        self.world_model_predictor_config = {
            "hidden_size": hidden_size,
            "bottleneck_size": predictor_hidden_size,
            "dropout": dropout,
            "residual": residual,
        }
        self.world_model_predictor = LatentTransitionPredictor(
            hidden_size=hidden_size,
            bottleneck_size=predictor_hidden_size,
            dropout=dropout,
            residual=residual,
        ).to(device=first_param.device, dtype=first_param.dtype)
        self._broadcast_world_model_parameters()

        optim_config = self.config.get("optim", {})
        base_group = self.actor_optimizer.param_groups[0]
        self.actor_optimizer.add_param_group(
            {
                "params": list(self.world_model_predictor.parameters()),
                "lr": optim_config.get("lr", base_group["lr"]),
                "weight_decay": optim_config.get("weight_decay", base_group.get("weight_decay", 0.0)),
                "name": "latent_world_model_predictor",
            }
        )

    def extra_state_dict(self):
        if self.world_model_predictor is None:
            return {}
        return {
            "world_model_predictor_config": dict(self.world_model_predictor_config),
            "world_model_predictor": {
                key: value.detach().cpu()
                for key, value in self.world_model_predictor.state_dict().items()
            }
        }

    def load_extra_state_dict(self, state_dict):
        if self.world_model_predictor is None:
            return
        predictor_state = (state_dict or {}).get("world_model_predictor", None)
        if predictor_state is None:
            print("WARN: latent world-model predictor state is missing from checkpoint; keeping current initialization.")
            return
        device = self._first_actor_parameter().device
        predictor_state = {key: value.to(device=device) for key, value in predictor_state.items()}
        self.world_model_predictor.load_state_dict(predictor_state)
        self._broadcast_world_model_parameters()

    def adapt_optimizer_state_dict(self, optimizer_state_dict):
        if optimizer_state_dict is None or self.actor_optimizer is None:
            return optimizer_state_dict

        saved_param_groups = optimizer_state_dict.get("param_groups", [])
        current_optimizer_state = self.actor_optimizer.state_dict()
        current_param_groups = current_optimizer_state.get("param_groups", [])
        if len(saved_param_groups) == len(current_param_groups):
            return optimizer_state_dict

        saved_state = optimizer_state_dict.get("state", {})
        if self.world_model_predictor is not None and len(saved_param_groups) + 1 == len(current_param_groups):
            print("WARN: optimizer checkpoint has no latent world-model param group; initializing predictor optimizer state from scratch.")
            return {
                **optimizer_state_dict,
                "state": saved_state,
                "param_groups": [*saved_param_groups, current_param_groups[-1]],
            }

        if (
            self.world_model_predictor is None
            and len(saved_param_groups) == len(current_param_groups) + 1
            and saved_param_groups[-1].get("name") == "latent_world_model_predictor"
        ):
            print("WARN: dropping latent world-model optimizer state because latent loss is disabled.")
            dropped_params = set(saved_param_groups[-1].get("params", []))
            return {
                **optimizer_state_dict,
                "state": {param_id: state for param_id, state in saved_state.items() if param_id not in dropped_params},
                "param_groups": saved_param_groups[:-1],
            }

        return optimizer_state_dict

    def _has_latent_world_model_batch(self, micro_batch) -> bool:
        if not self.latent_world_model_enabled:
            return False
        legacy_keys = [
            "wm_input_ids",
            "wm_attention_mask",
            "wm_position_ids",
            "wm_action_end_idx",
            "wm_obs_end_idx",
            "wm_loss_mask",
        ]
        named_keys = [
            "wm_latent_input_ids",
            "wm_latent_attention_mask",
            "wm_latent_position_ids",
            "wm_latent_action_pos",
            "wm_latent_obs_pos",
            "wm_latent_loss_mask",
        ]
        return all(key in micro_batch for key in legacy_keys) or all(key in micro_batch for key in named_keys)

    def _get_latent_world_model_batch(self, micro_batch):
        if "wm_input_ids" in micro_batch:
            return {
                "input_ids": micro_batch["wm_input_ids"],
                "attention_mask": micro_batch["wm_attention_mask"],
                "position_ids": micro_batch["wm_position_ids"],
                "action_pos": micro_batch["wm_action_end_idx"].long(),
                "obs_pos": micro_batch["wm_obs_end_idx"].long(),
                "loss_mask": micro_batch["wm_loss_mask"].float(),
            }
        return {
            "input_ids": micro_batch["wm_latent_input_ids"],
            "attention_mask": micro_batch["wm_latent_attention_mask"],
            "position_ids": micro_batch["wm_latent_position_ids"],
            "action_pos": micro_batch["wm_latent_action_pos"].long(),
            "obs_pos": micro_batch["wm_latent_obs_pos"].long(),
            "loss_mask": micro_batch["wm_latent_loss_mask"].float(),
        }

    def _has_obs_ce_batch(self, micro_batch) -> bool:
        required_keys = [
            "wm_obs_input_ids",
            "wm_obs_attention_mask",
            "wm_obs_position_ids",
            "wm_obs_loss_mask",
        ]
        return self._obs_ce_enabled() and all(key in micro_batch for key in required_keys)

    def _compute_obs_ce_loss(self, micro_batch, temperature, loss_agg_mode):
        if not self._has_obs_ce_batch(micro_batch):
            return None, {}

        input_ids = micro_batch["wm_obs_input_ids"]
        attention_mask = micro_batch["wm_obs_attention_mask"]
        position_ids = micro_batch["wm_obs_position_ids"]
        loss_mask = micro_batch["wm_obs_loss_mask"].float()

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            output = self.actor_module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
            )
            logits = output.logits[:, :-1, :]
            logits.div_(temperature)
            labels = input_ids[:, 1:]
            log_probs = logprobs_from_logits(logits=logits, labels=labels)

        shifted_loss_mask = loss_mask[:, 1:].to(dtype=log_probs.dtype, device=log_probs.device)
        token_count = shifted_loss_mask.sum()
        loss_mat = -log_probs
        if token_count.detach().item() <= 0:
            zero = loss_mat.sum() * 0.0
            return zero, {
                "actor/wm_obs_ce_loss": 0.0,
                "actor/wm_obs_ce_tokens": 0.0,
                "world_model/obs_ce_loss": 0.0,
                "world_model/obs_ce_tokens": 0.0,
            }

        if loss_agg_mode == "token-mean":
            obs_ce_loss = (loss_mat * shifted_loss_mask).sum() / token_count
        else:
            obs_ce_loss = agg_loss(loss_mat=loss_mat, loss_mask=shifted_loss_mask, loss_agg_mode=loss_agg_mode)

        metrics = {
            "actor/wm_obs_ce_loss": obs_ce_loss.detach().item(),
            "actor/wm_obs_ce_tokens": token_count.detach().item(),
            "world_model/obs_ce_loss": obs_ce_loss.detach().item(),
            "world_model/obs_ce_tokens": token_count.detach().item(),
        }
        return obs_ce_loss, metrics

    def _compute_latent_world_model_loss(self, micro_batch):
        if not self._has_latent_world_model_batch(micro_batch):
            return None, {}

        wm_batch = self._get_latent_world_model_batch(micro_batch)
        wm_input_ids = wm_batch["input_ids"]
        wm_attention_mask = wm_batch["attention_mask"]
        wm_position_ids = wm_batch["position_ids"]
        wm_action_end_idx = wm_batch["action_pos"]
        wm_obs_end_idx = wm_batch["obs_pos"]
        wm_loss_mask = wm_batch["loss_mask"]

        valid_count = wm_loss_mask.sum()
        if valid_count.item() == 0:
            zero = torch.zeros((), device=wm_input_ids.device, dtype=torch.float32)
            return zero, {"actor/wm_valid": 0.0}

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            output = self.actor_module(
                input_ids=wm_input_ids,
                attention_mask=wm_attention_mask,
                position_ids=wm_position_ids,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
            if output.hidden_states is None:
                raise RuntimeError("Actor forward did not return hidden_states for latent world-model loss.")
            hidden_states = output.hidden_states[-1]
            batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
            max_idx = hidden_states.size(1) - 1
            action_hidden = hidden_states[batch_idx, wm_action_end_idx.clamp(min=0, max=max_idx)]
            obs_hidden = hidden_states[batch_idx, wm_obs_end_idx.clamp(min=0, max=max_idx)].detach()
            if self.world_model_predictor is not None:
                pred_hidden = self.world_model_predictor(action_hidden)
            else:
                # Default Workstream C: direct cosine on the action-end hidden, no predictor.
                pred_hidden = action_hidden

        wm_cfg = self._world_model_config()
        use_contrastive = bool(wm_cfg.get("latent_contrastive", False))
        valid = wm_loss_mask.bool()
        pos_cos = neg_cos = None
        if use_contrastive:
            # InfoNCE: pull each action-end hidden toward its OWN next-obs hidden and push away
            # from other transitions' next-obs (in-batch negatives). ALFWorld observations share
            # ~91% tokens (static task/template), so that common-mode signal is shared by positives
            # and negatives and cancels here, forcing the representation to encode the action-specific
            # consequence rather than the shortcut of task/state identity. No predictor; obs stop-grad.
            tau = float(wm_cfg.get("latent_temperature", 0.1) or 0.1)
            anchor = pred_hidden.float()[valid]
            target = obs_hidden.float()[valid]
            # Remove the batch common-mode (the ~static observation template) before comparing,
            # otherwise the shared component swamps the per-transition consequence even after
            # L2-normalization (verified empirically). Centering is what makes InfoNCE discriminative.
            anchor = anchor - anchor.mean(dim=0, keepdim=True)
            target = target - target.mean(dim=0, keepdim=True)
            anchor = F.normalize(anchor, dim=-1)
            target = F.normalize(target, dim=-1)
            n = anchor.size(0)
            if n >= 2:
                logits = anchor @ target.t() / tau
                labels = torch.arange(n, device=logits.device)
                latent_loss = F.cross_entropy(logits, labels)
                with torch.no_grad():
                    sim = anchor @ target.t()
                    pos_cos = sim.diag().mean()
                    neg_cos = (sim.sum() - sim.diag().sum()) / (n * (n - 1))
            elif n == 1:
                latent_loss = (1.0 - (anchor * target).sum(dim=-1)).mean()
            else:
                latent_loss = pred_hidden.float().sum() * 0.0
        else:
            cosine = F.cosine_similarity(pred_hidden.float(), obs_hidden.float(), dim=-1)
            per_sample_loss = 1.0 - cosine
            latent_loss = torch.sum(per_sample_loss * wm_loss_mask) / valid_count.clamp_min(1.0)

        with torch.no_grad():
            plain_cos = F.cosine_similarity(pred_hidden.float(), obs_hidden.float(), dim=-1)
            any_valid = bool(valid.any().item())
            metrics = {
                "actor/wm_latent_loss": latent_loss.detach().item(),
                "actor/wm_cosine": plain_cos[valid].mean().detach().item() if any_valid else 0.0,
                "actor/wm_valid": valid_count.detach().item(),
                "actor/wm_pred_norm": pred_hidden.float()[valid].norm(dim=-1).mean().detach().item() if any_valid else 0.0,
                "actor/wm_target_norm": obs_hidden.float()[valid].norm(dim=-1).mean().detach().item() if any_valid else 0.0,
            }
            if use_contrastive and pos_cos is not None:
                metrics["actor/wm_pos_cosine"] = pos_cos.item()
                metrics["actor/wm_neg_cosine"] = neg_cos.item()
                metrics["actor/wm_gap"] = (pos_cos - neg_cos).item()
        return latent_loss, metrics

    def _sync_world_model_gradients(self):
        if self.world_model_predictor is None:
            return
        if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1:
            return
        world_size = dist.get_world_size()
        for parameter in self.world_model_predictor.parameters():
            if parameter.grad is None:
                continue
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
            parameter.grad.div_(world_size)

    def _clip_world_model_grad_norm(self):
        self._last_world_model_grad_norm = None
        if self.world_model_predictor is None:
            return None
        parameters = [parameter for parameter in self.world_model_predictor.parameters() if parameter.grad is not None]
        if not parameters:
            device = self._first_actor_parameter().device
            self._last_world_model_grad_norm = torch.zeros((), device=device)
            return self._last_world_model_grad_norm
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=self.config.grad_clip)
        self._last_world_model_grad_norm = grad_norm.detach()
        return grad_norm

    def _forward_micro_batch(self, micro_batch, temperature, calculate_entropy=False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch:
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0)

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices).transpose(0, 1).unsqueeze(1)  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)
                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outpus_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        grad_norm_is_finite = torch.isfinite(grad_norm)
        if self._last_world_model_grad_norm is not None:
            grad_norm_is_finite = grad_norm_is_finite & torch.isfinite(self._last_world_model_grad_norm.to(device=grad_norm.device))

        # if grad_norm is not finite, skip the update
        if not grad_norm_is_finite:
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}, world_model_grad_norm: {self._last_world_model_grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        elif use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature, calculate_entropy=calculate_entropy)
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]

        return log_probs, entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        multi_turn = data.meta_info.get("multi_turn", False)

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages"]
        obs_ce_keys = [
            "wm_obs_input_ids",
            "wm_obs_attention_mask",
            "wm_obs_position_ids",
            "wm_obs_loss_mask",
        ]
        latent_legacy_keys = [
            "wm_input_ids",
            "wm_attention_mask",
            "wm_position_ids",
            "wm_action_end_idx",
            "wm_obs_end_idx",
            "wm_loss_mask",
        ]
        latent_named_keys = [
            "wm_latent_input_ids",
            "wm_latent_attention_mask",
            "wm_latent_position_ids",
            "wm_latent_action_pos",
            "wm_latent_obs_pos",
            "wm_latent_loss_mask",
        ]
        if self._obs_ce_enabled() and all(key in data.batch.keys() for key in obs_ce_keys):
            select_keys.extend(obs_ce_keys)
        if self.world_model_predictor is not None:
            if all(key in data.batch.keys() for key in latent_legacy_keys):
                select_keys.extend(latent_legacy_keys)
            elif all(key in data.batch.keys() for key in latent_named_keys):
                select_keys.extend(latent_named_keys)
        if multi_turn:
            select_keys.append("loss_mask")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                if has_multi_modal_inputs:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    # split batch into micro_batches
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for data in micro_batches:
                    # Support all hardwares
                    if isinstance(data, DataProto):
                        data = {**data.batch.to(get_torch_device().current_device()), **data.non_tensor_batch}
                    else:
                        data = data.to(get_torch_device().current_device())  # actor device is cpu when using offload
                    responses = data["responses"]
                    response_length = responses.size(1)
                    attention_mask = data["attention_mask"]
                    if multi_turn:
                        response_mask = data["loss_mask"][:, -response_length:]
                    else:
                        response_mask = attention_mask[:, -response_length:]

                    old_log_prob = data["old_log_probs"]
                    advantages = data["advantages"]

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature, calculate_entropy=calculate_entropy)
                    
                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    if loss_mode == "vanilla":
                        policy_loss_fn = compute_policy_loss
                    elif loss_mode == "gspo":
                        policy_loss_fn = compute_policy_loss_gspo
                    else:
                        raise ValueError(f"Unsupported loss_mode: {loss_mode}")

                    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        cliprange=clip_ratio,
                        cliprange_low=clip_ratio_low,
                        cliprange_high=clip_ratio_high,
                        clip_ratio_c=clip_ratio_c,
                        loss_agg_mode=loss_agg_mode,
                    )

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = data["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] = kl_loss.detach().item()
                        metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    obs_ce_loss, obs_ce_metrics = self._compute_obs_ce_loss(
                        data,
                        temperature=temperature,
                        loss_agg_mode=self._world_model_config().get("obs_ce_loss_agg_mode", "token-mean"),
                    )
                    if obs_ce_loss is not None:
                        policy_loss = policy_loss + obs_ce_loss * self._obs_ce_coef()
                        obs_ce_metrics["actor/wm_obs_ce_coef"] = self._obs_ce_coef()
                        obs_ce_metrics["world_model/obs_ce_lambda"] = self._obs_ce_coef()
                        append_to_dict(metrics, obs_ce_metrics)

                    latent_loss, world_model_metrics = self._compute_latent_world_model_loss(data)
                    if latent_loss is not None:
                        policy_loss = policy_loss + latent_loss * self.world_model_loss_coef
                        world_model_metrics["actor/wm_coef"] = self.world_model_loss_coef
                        append_to_dict(metrics, world_model_metrics)

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
                    else:
                        loss = policy_loss / self.gradient_accumulation
                    loss.backward()

                    data = {
                        "actor/pg_loss": pg_loss.detach().item(),
                        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                        "actor/ppo_kl": ppo_kl.detach().item(),
                        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                    }
                    append_to_dict(metrics, data)

                self._sync_world_model_gradients()
                self._clip_world_model_grad_norm()
                grad_norm = self._optimizer_step()
                data = {"actor/grad_norm": grad_norm.detach().item()}
                if self._last_world_model_grad_norm is not None:
                    data["actor/wm_grad_norm"] = self._last_world_model_grad_norm.detach().item()
                append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad()
        return metrics
