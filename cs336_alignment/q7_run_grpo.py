from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable

import torch
import typer
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from unittest.mock import patch

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.q3_run_zeroshot_benchmark import (
    DEFAULT_DATASET as MATH_DATASET,
    DEFAULT_PROMPT_PATH as MATH_PROMPT_PATH,
    load_math_examples,
)
from cs336_alignment.q4_sft_helpers import (
    get_response_log_probs,
    tokenize_prompt_and_output,
)
from cs336_alignment.q7_grpo import (
    compute_group_normalized_rewards,
    compute_policy_gradient_loss,
    masked_mean,
)

app = typer.Typer(pretty_exceptions_enable=False)


# -------------------------------
# Default hyperparameters
# -------------------------------
DEFAULT_MODEL = "data/a5-alignment/models/Qwen2.5-Math-1.5B"
n_grpo_steps: int = 200
learning_rate: float = 1e-5
advantage_eps: float = 1e-6
rollout_batch_size: int = 256
group_size: int = 8
sampling_temperature: float = 1.0
sampling_min_tokens: int = 4
sampling_max_tokens: int = 1024
epochs_per_rollout_batch: int = 1
train_batch_size: int = 256
gradient_accumulation_steps: int = 128
gpu_memory_utilization: float = 0.85
loss_type: str = "reinforce_with_baseline"  # or "no_baseline", "grpo_clip"
use_std_normalization: bool = True
clip_grad_value: float = 1.0
val_every: int = 10
val_examples: int = 128
device_policy = "cuda:0"
device_vllm = "cuda:1"


# -------------------------------
# Utilities
# -------------------------------
def build_sampling_params(group_size: int) -> SamplingParams:
    """Sampling params for vLLM rollouts."""
    return SamplingParams(
        temperature=sampling_temperature,
        top_p=1.0,
        max_tokens=sampling_max_tokens,
        min_tokens=sampling_min_tokens,
        n=group_size,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )


def init_vllm(model_path: str, device: str, gpu_mem_util: float) -> LLM:
    """Start vLLM on a separate device for fast batched generation."""
    vllm_set_random_seed(1234)
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None,
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_path,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_mem_util,
        )


def load_policy(model_path: str, device: str) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    policy = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    policy.to(device)
    policy.config.use_cache = False
    policy.gradient_checkpointing_enable()
    policy.train()
    return policy, tokenizer


def load_math_data(max_examples: int | None = None) -> tuple[list[str], list[str]]:
    prompt_template = Path(MATH_PROMPT_PATH).read_text()
    examples = load_math_examples(
        MATH_DATASET,
        prompt_template,
        max_examples=max_examples,
    )
    prompts = [ex.prompt for ex in examples]
    gts = [ex.ground_truth for ex in examples]
    return prompts, gts


def collect_rollouts(
    llm: LLM,
    prompts: list[str],
    ground_truths: list[str],
    sampling_params: SamplingParams,
) -> tuple[list[str], list[str], list[dict[str, float]]]:
    """Generate group_size responses per prompt and flatten."""
    outputs = llm.generate(prompts, sampling_params)
    responses: list[str] = []
    repeated_gts: list[str] = []
    reward_infos: list[dict[str, float]] = []
    for gt, out in zip(ground_truths, outputs):
        for choice in out.outputs or []:
            resp = choice.text
            responses.append(resp)
            repeated_gts.append(gt)
            reward_infos.append(r1_zero_reward_fn(resp, gt))
    return responses, repeated_gts, reward_infos


def prepare_microbatches(
    prompts: list[str],
    responses: list[str],
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
) -> Iterable[dict[str, torch.Tensor]]:
    """Tokenize and yield microbatches of prompt/response pairs."""
    loader = DataLoader(
        list(zip(prompts, responses)),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: tokenize_prompt_and_output(
            [p for p, _ in batch], [r for _, r in batch], tokenizer
        ),
    )
    for toks in loader:
        yield toks


def evaluate_rewards(policy: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, prompts: list[str], gts: list[str], n_eval: int = 128) -> dict[str, float]:
    """Lightweight eval: generate 1 sample per prompt and average rewards."""
    subset_prompts = prompts[:n_eval]
    subset_gts = gts[:n_eval]
    policy.eval()
    with torch.no_grad():
        encoded = tokenizer(
            subset_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(policy.device)
        gen = policy.generate(
            **encoded,
            max_new_tokens=256,
            temperature=0.8,
            top_p=0.9,
        )
    rewards = []
    for seq, gt in zip(gen, subset_gts):
        resp = tokenizer.decode(seq[encoded["input_ids"].shape[1] :], skip_special_tokens=True)
        rewards.append(r1_zero_reward_fn(resp, gt).get("reward", 0.0))
    policy.train()
    return {"avg_reward": float(sum(rewards) / max(len(rewards), 1))}


def grpo_step(
    policy: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    llm: LLM,
    prompts_pool: list[str],
    gts_pool: list[str],
    optimizer: torch.optim.Optimizer,
    *,
    step_idx: int,
    loss_type: str,
    advantage_eps: float,
    group_size: int,
    rollout_batch_size: int,
    train_batch_size: int,
    gradient_accumulation_steps: int,
    epochs_per_rollout_batch: int,
    use_std_normalization: bool,
) -> dict[str, float]:
    """Run one rollout + training round."""
    assert train_batch_size % gradient_accumulation_steps == 0, "train_batch_size must be divisible by gradient_accumulation_steps"
    micro_train_batch_size = train_batch_size // gradient_accumulation_steps
    assert rollout_batch_size % group_size == 0, "rollout_batch_size must be divisible by group_size"
    n_prompts_per_rollout_batch = rollout_batch_size // group_size
    assert train_batch_size >= group_size, "train_batch_size must be greater than or equal to group_size"

    # Sample prompts for this rollout batch.
    indices = random.sample(range(len(prompts_pool)), k=min(n_prompts_per_rollout_batch, len(prompts_pool)))
    batch_prompts = [prompts_pool[i] for i in indices]
    batch_gts = [gts_pool[i] for i in indices]

    sampling_params = build_sampling_params(group_size=group_size)
    rollouts, repeated_gts, reward_infos = collect_rollouts(llm, batch_prompts, batch_gts, sampling_params)

    # Compute advantages.
    advantages, raw_rewards, reward_meta = compute_group_normalized_rewards(
        reward_fn=r1_zero_reward_fn,
        rollout_responses=rollouts,
        repeated_ground_truths=repeated_gts,
        group_size=group_size,
        advantage_eps=advantage_eps,
        normalize_by_std=use_std_normalization,
    )

    # Training epochs over this rollout batch (on-policy default = 1).
    loss_log: list[float] = []
    token_entropy_log: list[float] = []
    clip_frac_log: list[float] = []

    # Slice rollouts into microbatches.
    total_examples = len(rollouts)
    assert total_examples == rollout_batch_size, "rollout_batch_size mismatch with collected rollouts"

    for epoch in range(epochs_per_rollout_batch):
        # Optionally precompute old log probs for GRPO-Clip (off-policy).
        old_log_probs_cache: list[torch.Tensor] = []
        if loss_type == "grpo_clip":
            for toks in prepare_microbatches(batch_prompts * group_size, rollouts, tokenizer, micro_train_batch_size):
                with torch.no_grad():
                    old_lp = get_response_log_probs(
                        model=policy,
                        input_ids=toks["input_ids"].to(policy.device),
                        labels=toks["labels"].to(policy.device),
                        return_token_entropy=False,
                        with_grad=False,
                    )["log_probs"]
                old_log_probs_cache.append(old_lp.cpu())
        optimizer.zero_grad()
        micro_idx = 0
        for toks in prepare_microbatches(batch_prompts * group_size, rollouts, tokenizer, micro_train_batch_size):
            toks = {k: v.to(policy.device) for k, v in toks.items()}
            lp_out = get_response_log_probs(
                model=policy,
                input_ids=toks["input_ids"],
                labels=toks["labels"],
                return_token_entropy=True,
                with_grad=True,
            )
            policy_log_probs = lp_out["log_probs"]
            token_entropy = lp_out.get("token_entropy")
            adv_slice = advantages[micro_idx : micro_idx + policy_log_probs.shape[0]].to(policy.device).unsqueeze(-1)
            raw_slice = raw_rewards[micro_idx : micro_idx + policy_log_probs.shape[0]].to(policy.device).unsqueeze(-1)
            old_lp = None
            if loss_type == "grpo_clip":
                old_lp = old_log_probs_cache[micro_idx // micro_train_batch_size].to(policy.device)

            loss_per_token, meta = compute_policy_gradient_loss(
                policy_log_probs=policy_log_probs,
                loss_type=loss_type,
                raw_rewards=raw_slice,
                advantages=adv_slice,
                old_log_probs=old_lp,
                cliprange=0.2 if loss_type == "grpo_clip" else None,
            )
            # Mask to response tokens and average per example.
            masked_loss = loss_per_token * toks["response_mask"]
            per_example = masked_mean(masked_loss, toks["response_mask"], dim=1)
            loss = per_example.mean() / gradient_accumulation_steps
            loss.backward()
            loss_log.append(loss.item() * gradient_accumulation_steps)
            if token_entropy is not None:
                te = masked_mean(token_entropy * toks["response_mask"], toks["response_mask"], dim=1)
                token_entropy_log.extend(te.detach().cpu().tolist())
            if "is_clipped" in meta:
                cf = masked_mean(meta["is_clipped"] * toks["response_mask"], toks["response_mask"], dim=1)
                clip_frac_log.extend(cf.detach().cpu().tolist())

            micro_idx += policy_log_probs.shape[0]
            if micro_idx % micro_train_batch_size == 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), clip_grad_value)
                optimizer.step()
                optimizer.zero_grad()

    stats = {
        "loss": float(sum(loss_log) / max(len(loss_log), 1)),
        "reward_mean": reward_meta["raw_reward_mean"],
        "token_entropy_mean": float(sum(token_entropy_log) / max(len(token_entropy_log), 1)) if token_entropy_log else None,
        "clip_fraction_mean": float(sum(clip_frac_log) / max(len(clip_frac_log), 1)) if clip_frac_log else None,
    }
    stats.update({k: v for k, v in reward_meta.items() if isinstance(v, (int, float))})
    return stats


@app.command()
def main(
    model_path: str = DEFAULT_MODEL,
    n_steps: int = n_grpo_steps,
    lr: float = learning_rate,
    rollout_bs: int = rollout_batch_size,
    train_bs: int = train_batch_size,
    group: int = group_size,
    grad_accum: int = gradient_accumulation_steps,
    loss: str = loss_type,
    use_std: bool = use_std_normalization,
    epochs_per_batch: int = epochs_per_rollout_batch,
    logdir: Path = Path("outputs/runs_grpo"),
) -> None:
    """Run GRPO training on MATH with vLLM rollouts."""
    logdir.mkdir(parents=True, exist_ok=True)
    prompts_all, gts_all = load_math_data()

    policy, tokenizer = load_policy(model_path, device_policy)
    llm = init_vllm(model_path, device_vllm, gpu_memory_utilization)

    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=lr,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )

    logs: list[dict[str, float]] = []
    for step in range(1, n_steps + 1):
        stats = grpo_step(
            policy=policy,
            tokenizer=tokenizer,
            llm=llm,
            prompts_pool=prompts_all,
            gts_pool=gts_all,
            optimizer=optimizer,
            step_idx=step,
            loss_type=loss,
            advantage_eps=advantage_eps,
            group_size=group,
            rollout_batch_size=rollout_bs,
            train_batch_size=train_bs,
            gradient_accumulation_steps=grad_accum,
            epochs_per_rollout_batch=epochs_per_batch,
            use_std_normalization=use_std,
        )
        logs.append({"step": step, **stats})
        if step % 1 == 0:
            print(f"[step {step}] loss={stats['loss']:.4f} reward_mean={stats['reward_mean']:.4f}")

        if step % val_every == 0:
            eval_stats = evaluate_rewards(policy, tokenizer, prompts_all, gts_all, n_eval=val_examples)
            print(f"[eval step {step}] {eval_stats}")
            logs.append({"step": step, "eval_reward": eval_stats["avg_reward"]})

    (logdir / "train_logs.json").write_text(json.dumps(logs, indent=2))
    print(f"Saved logs to {logdir/'train_logs.json'}")


if __name__ == "__main__":
    app()

