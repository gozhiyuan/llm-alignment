from __future__ import annotations

from typing import Callable, List

import torch


def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Compute per-response rewards and normalize them within each group.

    Rewards are scored via ``reward_fn`` and grouped into contiguous chunks of
    ``group_size``. Within each group we subtract the mean reward, and optionally
    divide by the (sample) standard deviation, adding ``advantage_eps`` to avoid
    division by zero.

    Args:
        reward_fn: Callable[[str, str], dict[str, float]] Scores the rollout responses against the ground truths, producing a dict with keys "reward", "format_reward", and "answer_reward".
        rollout_responses: list[str] Rollouts from the policy. The length of this list is rollout_batch_size = n_prompts_per_rollout_batch * group_size.
        repeated_ground_truths: list[str] The ground truths for the examples. The length of this list is rollout_batch_size, because the ground truth for each example is repeated group_size times.
        group_size: int Number of responses per question (group).
        advantage_eps: float Small constant to avoid division by zero in normalization.
        normalize_by_std: bool If True, divide by the per-group standard deviation; otherwisesubtract only the group mean.
    Returns:
        tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
            advantages: torch.Tensor of shape (rollout_batch_size,): Group-normalized rewards for each rollout response.
            raw_rewards: torch.Tensor of shape (rollout_batch_size,): Unnormalized rewards for each rollout response.
            metadata: dict[str, float]: Metadata for the rewards of the rollout batch.
    """
    if len(rollout_responses) != len(repeated_ground_truths):
        raise ValueError("rollout_responses and repeated_ground_truths must have the same length")
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    rollout_batch_size = len(rollout_responses)
    if rollout_batch_size % group_size != 0:
        raise ValueError("rollout_batch_size must be divisible by group_size")

    raw_rewards: List[float] = []
    format_rewards: List[float] = []
    answer_rewards: List[float] = []
    for response, gt in zip(rollout_responses, repeated_ground_truths):
        reward_info = reward_fn(response, gt)
        raw_rewards.append(float(reward_info.get("reward", 0.0)))
        format_rewards.append(float(reward_info.get("format_reward", 0.0)))
        answer_rewards.append(float(reward_info.get("answer_reward", 0.0)))

    raw_tensor = torch.tensor(raw_rewards, dtype=torch.float32) # shape (rollout_batch_size,)
    grouped_raw = raw_tensor.view(-1, group_size) # shape (rollout_batch_size // group_size, group_size)

    group_means = grouped_raw.mean(dim=1, keepdim=True) # shape (rollout_batch_size // group_size, 1)
    if normalize_by_std:
        group_stds = grouped_raw.std(dim=1, keepdim=True, unbiased=True) # shape (rollout_batch_size // group_size, 1)
        denom = group_stds + advantage_eps
        normalized = (grouped_raw - group_means) / denom # shape (rollout_batch_size // group_size, group_size)
    else:
        normalized = grouped_raw - group_means # shape (rollout_batch_size // group_size, group_size)

    advantages = normalized.reshape(-1) # shape (rollout_batch_size,)
    raw_rewards_tensor = grouped_raw.reshape(-1) # shape (rollout_batch_size,)

    metadata = {
        "raw_reward_mean": raw_rewards_tensor.mean().item(),
        "raw_reward_std": raw_rewards_tensor.std(unbiased=True).item(),
        "normalized_reward_mean": advantages.mean().item(),
        "normalized_reward_std": advantages.std(unbiased=True).item(),
        "normalize_by_std": normalize_by_std,
        "group_means": group_means.squeeze(-1),
        "group_stds": grouped_raw.std(dim=1, unbiased=True),
        "format_reward_mean": float(torch.tensor(format_rewards).mean().item()),
        "answer_reward_mean": float(torch.tensor(answer_rewards).mean().item()),
    }

    return advantages, raw_rewards_tensor, metadata


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """Naive per-token policy gradient loss: -A * log p.
    Args:
        raw_rewards_or_advantages: torch.Tensor Shape (batch_size, 1), scalar reward/advantage for each rollout response.
        policy_log_probs: torch.Tensor Shape (batch_size, sequence_length), logprobs foreach token.
    Returns:
        torch.Tensor Shape (batch_size, sequence_length), the per-token policy-gradient loss (to
        be aggregated across the batch and sequence dimensions in the training loop).
    """
    # Ensure rewards/advantages broadcast across sequence length.
    if raw_rewards_or_advantages.ndim == 1:
        raw_rewards_or_advantages = raw_rewards_or_advantages.unsqueeze(-1) # shape (batch_size, 1)

    if raw_rewards_or_advantages.shape[0] != policy_log_probs.shape[0]:
        raise ValueError("Batch size mismatch between rewards and log probs")

    return -raw_rewards_or_advantages * policy_log_probs # shape (batch_size, sequence_length)


def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Per-token GRPO-Clip loss following PPO-style clipping.
    Args:
        advantages: torch.Tensor Shape (batch_size, 1), per-example advantages A.
        policy_log_probs: torch.Tensor Shape (batch_size, sequence_length), per-token log
        probs from the policy being trained.
        old_log_probs: torch.Tensor Shape (batch_size, sequence_length), per-token log probs
        from the old policy.
        cliprange: float Clip parameter ϵ (e.g. 0.2).
    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]].
        loss torch.Tensor of shape (batch_size, sequence_length), the per-token clipped
        loss.
        metadata dict containing whatever you want to log. We suggest logging whether each
        token was clipped or not, i.e., whether the clipped policy gradient loss on the RHS of
        the min was lower than the LHS.
    """
    if advantages.ndim == 1:
        advantages = advantages.unsqueeze(-1) # shape (batch_size, 1)
    if advantages.shape[0] != policy_log_probs.shape[0]:
        raise ValueError("Batch size mismatch between advantages and log probs")
    if policy_log_probs.shape != old_log_probs.shape:
        raise ValueError("policy_log_probs and old_log_probs must have the same shape")

    # Ratio of new to old policy probabilities at each token.
    ratio = torch.exp(policy_log_probs - old_log_probs) # shape (batch_size, sequence_length)
    unclipped_obj = ratio * advantages

    clipped_ratio = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange) # shape (batch_size, sequence_length)
    clipped_obj = clipped_ratio * advantages # shape (batch_size, sequence_length)

    objective = torch.minimum(unclipped_obj, clipped_obj) # shape (batch_size, sequence_length)
    loss = -objective # shape (batch_size, sequence_length)

    metadata = {
        "is_clipped": (objective == clipped_obj).to(policy_log_probs.dtype), # shape (batch_size, sequence_length)
    }
    return loss, metadata


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: str,
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Dispatch to the requested policy gradient loss.
    
    Args:
        policy_log_probs: torch.Tensor Shape (batch_size, sequence_length), per-token log probs from the policy being trained.
        loss_type: str One of "no_baseline", "reinforce_with_baseline", or "grpo_clip".
        raw_rewards: torch.Tensor | None Shape (batch_size, 1), raw rewards for each rollout response. Needed for loss_type="no_baseline".
        advantages: torch.Tensor | None Shape (batch_size, 1), advantages for each rollout response. Needed for loss_type in {"reinforce_with_baseline", "grpo_clip"}.
        old_log_probs: torch.Tensor | None Shape (batch_size, sequence_length), per-token log probs from the old policy. Needed for loss_type="grpo_clip".
        cliprange: float | None Scalar ϵ used for clipping. Needed for loss_type="grpo_clip".
    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
            loss: torch.Tensor Shape (batch_size, sequence_length), the per-token loss.
            metadata: dict[str, torch.Tensor] Metadata for the loss.
    """
    if loss_type == "no_baseline":
        if raw_rewards is None:
            raise ValueError("raw_rewards required for loss_type='no_baseline'")
        loss = compute_naive_policy_gradient_loss(
            raw_rewards_or_advantages=raw_rewards,
            policy_log_probs=policy_log_probs,
        )
        metadata: dict[str, torch.Tensor] = {}
        return loss, metadata

    if loss_type == "reinforce_with_baseline":
        if advantages is None:
            raise ValueError("advantages required for loss_type='reinforce_with_baseline'")
        loss = compute_naive_policy_gradient_loss(
            raw_rewards_or_advantages=advantages,
            policy_log_probs=policy_log_probs,
        )
        metadata = {}
        return loss, metadata

    if loss_type == "grpo_clip":
        if advantages is None or old_log_probs is None or cliprange is None:
            raise ValueError("advantages, old_log_probs, and cliprange required for loss_type='grpo_clip'")
        return compute_grpo_clip_loss(
            advantages=advantages,
            policy_log_probs=policy_log_probs,
            old_log_probs=old_log_probs,
            cliprange=cliprange,
        )

    raise ValueError(f"Unknown loss_type '{loss_type}'")


def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
) -> torch.Tensor:
    """Mean over masked elements along dim (or all dims if dim=None)."""
    if tensor.shape != mask.shape:
        raise ValueError("tensor and mask must have the same shape")
    mask_f = mask.to(tensor.dtype)
    masked = tensor * mask_f
    if dim is None:
        total = masked.sum()
        count = mask_f.sum()
    else:
        total = masked.sum(dim=dim)
        count = mask_f.sum(dim=dim)
    return total / count
