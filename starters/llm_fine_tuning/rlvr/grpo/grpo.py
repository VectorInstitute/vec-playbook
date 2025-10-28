"""GRPO logic."""

import math

import torch
from rich.progress import Progress
from torch.nn.functional import log_softmax
from torch.optim import AdamW
from transformers import PreTrainedModel

from starters.llm_fine_tuning.rlvr.grpo.data_types import (
    BatchForGRPOTorch,
    GRPOBatcher,
    GRPOMetrics,
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
        • A is the rollout-level normalized advantage (broadcast across tokens in that rollout)
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
    ).squeeze(
        -1
    )  # (B, T)

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
    *,
    batcher: GRPOBatcher,
    model: PreTrainedModel,
    gradient_accumulation_steps: int,
    learning_rate: float = 1e-5,
    kl_coefficient: float = 0.05,
    adam_betas: tuple[float, float] = (0.9, 0.999),
    adam_weight_decay: float = 0.0,
    max_grad_norm: float = 1.0,
    use_mixed_precision: bool = True,
    numerical_stability_eps: float = 1e-12,
) -> tuple[PreTrainedModel, GRPOMetrics]:
    """
    Perform one optimization epoch of GRPO on a HuggingFace causal language model.

    This revision reduces line count by inlining the mini-batch slicing directly at
    the call sites (model forward and loss computation). See `compute_grpo_loss`
    for the objective definition and reduction details.
    """
    model.train()
    device = next(model.parameters()).device
    loss_metrics: list[float] = []

    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=adam_betas,
        weight_decay=adam_weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=(use_mixed_precision and torch.cuda.is_available())
    )

    num_steps = len(batcher)
    expected_updates = math.ceil(num_steps / max(1, gradient_accumulation_steps))

    accumulated_microbatches = 0
    optimizer.zero_grad(set_to_none=True)

    with Progress() as progress:
        step_task_id = progress.add_task(
            "[cyan]GRPO optimization[/cyan] (optimizer steps)", total=expected_updates
        )

        for batch_np in batcher:
            batch = batch_np.to_torch(device=device, target_class=BatchForGRPOTorch)
            with torch.amp.autocast(
                "cuda", enabled=(use_mixed_precision and torch.cuda.is_available())
            ):
                outputs = model(
                    input_ids=batch.input_ids.to(torch.long),
                    attention_mask=batch.attention_mask.to(torch.long),
                )
                logits_next = outputs.logits[:, :-1, :]  # (B, T, V)

                # Note that the first input token is skipped.
                loss_to_minimize = compute_grpo_loss(
                    logits_next=logits_next,
                    target_token_ids=batch.input_ids[:, 1:].to(torch.long),
                    valid_token_mask=batch.loss_masks[:, 1:],
                    taken_prob_old=batch.pi_selected_base[:, 1:].to(torch.float32),
                    taken_prob_ref=batch.pi_selected_ref[:, 1:].to(torch.float32),
                    advantage_per_token=batch.per_token_advantage[:, 1:].to(
                        torch.float32
                    ),
                    kl_coefficient=kl_coefficient,
                    numerical_stability_eps=numerical_stability_eps,
                ) / max(1, gradient_accumulation_steps)

            scaler.scale(loss_to_minimize).backward()
            loss_metrics.append(loss_to_minimize.detach().cpu().item())
            accumulated_microbatches += 1

            if accumulated_microbatches % max(1, gradient_accumulation_steps) == 0:
                if max_grad_norm and max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                progress.update(step_task_id, advance=1)

        # Flush leftover microbatches
        if accumulated_microbatches % max(1, gradient_accumulation_steps) != 0:
            if max_grad_norm and max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            progress.update(step_task_id, advance=1)

    return model, GRPOMetrics(loss_metrics=loss_metrics)
