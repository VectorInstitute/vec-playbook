"""Numerical logic related to GRPO."""

import logging
import math

import torch
from rich.progress import Progress
from torch.nn.functional import log_softmax, softmax
from transformers import PreTrainedModel

from templates.src.rlvr.grpo.data_types import (
    BatchForGRPO,
    BatchForInference,
    GRPOBatcher,
    GRPOMetrics,
    PerTokenProbs,
)


def get_per_token_probs(
    batch: BatchForInference | BatchForGRPO, model: "PreTrainedModel"
) -> PerTokenProbs:
    """Obtain per-token probs for a given batch and a given model.

    Params:
        batch: Tokenized _Batch (batch, length).
        model: Pretrained Causal LM.

    Returns
    -------
        PerTokenProbs storing dense torch tensors on the model device.
        (batch, length - 1) since no label is available for the last token.
    """
    device = model.device
    input_ids = torch.as_tensor(batch.input_ids, dtype=torch.long, device=device)
    attention_mask = torch.as_tensor(
        batch.attention_mask, dtype=torch.long, device=device
    )

    with torch.inference_mode():
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        # logits: (batch, length, vocab) where
        # (j, k, :) contains distribution for tokens k+1 of batch j.
        logits = output.logits

        # (batch, length - 1, vocab) skipping last token which has no label.
        # for all possible vocabs.
        probabilities_all = softmax(logits, dim=-1)[:, :-1, :]

        # (batch, length - 1, 1) skipping first token which is always given.
        target_token_ids = input_ids[:, 1:].unsqueeze(-1)

        # (batch, length - 1) for input_ids only.
        probabilities_selected = torch.gather(
            probabilities_all, dim=-1, index=target_token_ids
        ).squeeze(-1)

    return PerTokenProbs(
        attention_mask=attention_mask[: batch.num_valid],
        full=probabilities_all.detach()[: batch.num_valid],
        selected=probabilities_selected.detach()[: batch.num_valid],
    )


def compute_grpo_loss(
    *,
    logits_next: torch.Tensor,
    target_token_ids: torch.Tensor,
    valid_token_mask: torch.Tensor,
    taken_prob_old: torch.Tensor,
    taken_prob_ref: torch.Tensor,
    advantage_per_token: torch.Tensor,
    kl_coefficient: float,
    numerical_stability_eps: float = 1e-12,
) -> torch.Tensor:
    """
    Compute the GRPO loss for a mini-batch.

    The per-token objective (to MAXIMIZE) is:
        J_token = r * A  -  β * (x - log x - 1),

    where:
        • r = π_current(a|s) / π_old(a|s)
        • x = π_ref(a|s) / π_current(a|s)
        • A is the rollout-level normalized advantage
        (broadcast across tokens in that rollout)
        • β is the KL coefficient.

    This function returns the NEGATIVE batch objective (i.e., a loss to MINIMIZE).
    Reduction matches Eq. (19) of the GRPO formulation:
        1) mean over valid tokens for each rollout,
        2) mean over rollouts in the mini-batch.

    Args:
        logits_next: Logits for next-token prediction at positions [:, :-1], (B, T, V).
        target_token_ids: Gold next-token ids, (B, T).
        valid_token_mask: Boolean mask selecting generated tokens to train on, (B, T).
        taken_prob_old: Probability under the pi_frozen for the taken token, (B, T).
        taken_prob_ref: Probability under the pi_kl_ref for the taken token, (B, T).
        advantage_per_token: Token-level normalized advantages, (B, T).
        kl_coefficient: Strength of the KL penalty to the reference policy.
        numerical_stability_eps: Small epsilon to avoid log(0).

    Returns
    -------
        A scalar tensor: the loss to minimize (negative batch objective).
    """
    # Log-probabilities of all tokens under current policy
    log_prob_all = log_softmax(logits_next, dim=-1)  # (B, T, V)

    # Log-probability of the *taken* tokens under current policy
    log_prob_taken_current = torch.gather(
        log_prob_all, dim=-1, index=target_token_ids.unsqueeze(-1)
    ).squeeze(-1)  # (B, T)

    # Clamp old/ref probabilities for stability and take logs
    taken_prob_old = taken_prob_old.clamp(min=numerical_stability_eps)
    taken_prob_ref = taken_prob_ref.clamp(min=numerical_stability_eps)
    log_prob_taken_old = torch.log(taken_prob_old)  # (B, T)
    log_prob_taken_ref = torch.log(taken_prob_ref)  # (B, T)

    # Importance ratio: r = π_current / π_old  (for the taken token)
    log_ratio_current_over_old = log_prob_taken_current - log_prob_taken_old  # (B, T)
    ratio_current_over_old = torch.exp(log_ratio_current_over_old)  # (B, T)

    # KL penalty term (per taken token):  x - log x - 1, where x = pi_ref / pi_current
    log_ratio_ref_over_current = log_prob_taken_ref - log_prob_taken_current  # (B, T)
    ratio_ref_over_current = torch.exp(log_ratio_ref_over_current)  # (B, T)
    kl_scalar_taken = (
        ratio_ref_over_current - log_ratio_ref_over_current - 1.0
    )  # (B, T)

    # Per-token objective contribution
    token_objective = (
        ratio_current_over_old * advantage_per_token - kl_coefficient * kl_scalar_taken
    )  # (B, T)

    # Mask to only include valid positions
    token_objective = token_objective * valid_token_mask.float()  # (B, T)

    # Mean over valid tokens per rollout (avoid division by zero)
    tokens_per_rollout = valid_token_mask.sum(dim=1)  # (B,)
    per_rollout_mean = token_objective.sum(dim=1) / torch.clamp(
        tokens_per_rollout, min=1
    )  # (B,)

    # Batch mean; return negative to obtain a loss to minimize
    batch_objective_mean = per_rollout_mean.mean()
    return -batch_objective_mean


def optimize_grpo_one_epoch(
    batcher: GRPOBatcher,
    model: PreTrainedModel,
    model_pi_ref: PreTrainedModel,
    optimizer: torch.optim.Optimizer,
    gradient_accumulation_steps: int,
    kl_coefficient: float = 0.05,
    max_grad_norm: float = 1.0,
    use_mixed_precision: bool = True,
    numerical_stability_eps: float = 1e-12,
) -> tuple[PreTrainedModel, torch.optim.Optimizer, GRPOMetrics]:
    """
    Perform one optimization epoch of GRPO on a HuggingFace causal language model.

    This revision reduces line count by inlining the mini-batch slicing directly at
    the call sites (model forward and loss computation). See `compute_grpo_loss`
    for the objective definition and reduction details.

    Scaler is not used for bfloat16.
    """
    logger = logging.getLogger(__name__)
    model.train()
    device = next(model.parameters()).device
    loss_metrics: list[float] = []

    num_steps = len(batcher)
    expected_updates = math.ceil(num_steps / max(1, gradient_accumulation_steps))

    accumulated_microbatches = 0
    optimizer.zero_grad(set_to_none=True)

    progress = Progress().__enter__()
    step_task_id = progress.add_task(
        "[cyan]GRPO optimization[/cyan]", total=expected_updates
    )

    for batch_raw in batcher:
        batch = batch_raw.to_torch(device=device, target_class=BatchForGRPO)
        taken_prob_base = get_per_token_probs(batch, model=model)  # (B, T - 1)
        taken_prob_ref = get_per_token_probs(batch, model=model_pi_ref)  # (B, T - 1)

        with torch.amp.autocast(
            "cuda", enabled=(use_mixed_precision and torch.cuda.is_available())
        ):
            # outputs.logits: (B, T - 1, V)
            outputs = model(
                input_ids=batch.input_ids.to(torch.long),
                attention_mask=batch.attention_mask.to(torch.long),
            )
            logger.debug(
                {k: getattr(v, "shape", None) for k, v in batch.model_dump().items()}
            )

            # Note that the first input token is skipped.
            loss_to_minimize = compute_grpo_loss(
                # After slicing: (B, T - 1, V)
                logits_next=outputs.logits[:, :-1, :],
                # After slicing: (B, T - 1)
                target_token_ids=batch.input_ids[:, 1:].to(torch.long),
                valid_token_mask=batch.loss_masks[:, 1:],
                advantage_per_token=batch.per_token_advantage[:, 1:].to(torch.float32),
                # (B, T - 1)
                taken_prob_old=taken_prob_base.selected.to(torch.float32),
                taken_prob_ref=taken_prob_ref.selected.to(torch.float32),
                # Hyperparameters
                kl_coefficient=kl_coefficient,
                numerical_stability_eps=numerical_stability_eps,
            ) / max(1, gradient_accumulation_steps)

            loss_to_minimize.backward()

        loss_metrics.append(loss_to_minimize.detach().cpu().item())
        accumulated_microbatches += 1

        if accumulated_microbatches % max(1, gradient_accumulation_steps) == 0:
            if max_grad_norm and max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            progress.update(step_task_id, advance=1)

    # Flush leftover microbatches
    if accumulated_microbatches % max(1, gradient_accumulation_steps) != 0:
        if max_grad_norm and max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.zero_grad(set_to_none=True)
        progress.update(step_task_id, advance=1)

    progress.__exit__(None, None, None)

    if not loss_metrics:
        raise RuntimeError("No batch was proceed!")

    return model, optimizer, GRPOMetrics(avg_loss=sum(loss_metrics) / len(loss_metrics))
