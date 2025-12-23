"""
Run expert iteration (EI) on MATH using the SFT objective.

This follows Algorithm 2 from the writeup:
1. Generate multiple reasoning traces for a batch of questions with the current policy.
2. Score each generation with the reward function.
3. Keep only correct generations and fine-tune the policy on this filtered set.
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
from unittest.mock import patch

import torch
from torch.optim import AdamW as TorchAdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.q3_run_zeroshot_benchmark import (
    DEFAULT_DATASET as MATH_DATASET,
    DEFAULT_PROMPT_PATH as MATH_PROMPT_PATH,
    load_math_examples,
)
from cs336_alignment.q4_sft_helpers import (
    get_response_log_probs,
    load_model_and_tokenizer,
    sft_microbatch_train_step,
    tokenize_prompt_and_output,
)

# -------------------------------
# Config
# -------------------------------
model_id = "data/a5-alignment/models/Qwen2.5-Math-1.5B"
policy_device = "cuda:0"
vllm_device = "cuda:1"
vllm_seed = 1234
vllm_gpu_mem_utilization = 0.4

# EI generation params
n_ei_steps = 3
ei_batch_size = 8  # number of questions sampled per EI step
generations_per_prompt = 4  # G in Algorithm 2
sampling_temperature = 0.7
sampling_max_tokens = 256
sampling_min_tokens = 4  # avoid empty generations per instructions
sampling_seed = 7

# Training params
lr = 2e-5
batch_size = 1
grad_accum = 4
clip_grad = 1.0  # gradient clipping per instructions
train_epochs_per_step = 1
max_length = 512  # truncate prompt+response during tokenization

# Data + logging
math_dataset_path = MATH_DATASET
math_prompt_template_path = Path(MATH_PROMPT_PATH)
max_math_examples = 256  # cap for quicker iterations; set None for full
log_every = 20
output_dir = Path("outputs/runs_sft_ei")
output_dir.mkdir(parents=True, exist_ok=True)


def reward_fn_response_vs_gt(response: str, ground_truth: str) -> Dict[str, float]:
    """Math-aware reward using the same logic as the zero-shot benchmark."""
    return r1_zero_reward_fn(response, ground_truth, fast=True)


def init_vllm(model_path: str, device: str, seed: int, gpu_memory_utilization: float) -> LLM:
    """Start vLLM on a separate device for fast batched generation."""
    vllm_set_random_seed(seed)

    # Monkeypatch from TRL to force single-device placement and skip profiling checks.
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
            gpu_memory_utilization=gpu_memory_utilization,
        )


def load_policy_into_vllm_instance(policy: torch.nn.Module, llm: LLM):
    """Load the current policy weights into the running vLLM instance."""
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


def build_ei_sampling_params() -> SamplingParams:
    """Sampling params for EI data collection."""
    return SamplingParams(
        temperature=sampling_temperature,
        max_tokens=sampling_max_tokens,
        min_tokens=sampling_min_tokens,
        n=generations_per_prompt,
        seed=sampling_seed,
    )


def collate_fn(batch: Sequence[Dict[str, str]], tokenizer):
    prompts = [b["prompt"] for b in batch]
    outputs = [b["response"] for b in batch]
    toks = tokenize_prompt_and_output(prompts, outputs, tokenizer, max_length=max_length)
    return {k: v.to(policy_device) for k, v in toks.items()}


def sample_math_data() -> Tuple[List[str], List[str]]:
    """Load and format MATH prompts and ground truths."""
    prompt_template = math_prompt_template_path.read_text()
    examples = load_math_examples(
        math_dataset_path,
        prompt_template,
        max_examples=max_math_examples,
    )
    prompts = [ex.prompt for ex in examples]
    ground_truths = [ex.ground_truth for ex in examples]
    return prompts, ground_truths


def sample_batch(
    prompts: Sequence[str],
    ground_truths: Sequence[str],
    batch_size: int,
    rng: random.Random,
) -> Tuple[List[str], List[str]]:
    """Uniformly sample a batch of prompts/ground truths."""
    indices = rng.sample(range(len(prompts)), k=min(batch_size, len(prompts)))
    batch_prompts = [prompts[i] for i in indices]
    batch_gts = [ground_truths[i] for i in indices]
    return batch_prompts, batch_gts


def collect_ei_dataset(
    vllm_model: LLM,
    prompts: Sequence[str],
    ground_truths: Sequence[str],
    sampling_params: SamplingParams,
) -> Tuple[List[Dict[str, str]], Dict[str, float]]:
    """
    Generate multiple outputs per prompt, score with the reward, and keep correct ones.
    """
    outputs = vllm_model.generate(prompts, sampling_params)
    correct_pairs: List[Dict[str, str]] = []
    total_generations = 0
    correct_generations = 0
    for idx, request_output in enumerate(outputs):
        gt = ground_truths[idx]
        for out in request_output.outputs or []:
            total_generations += 1
            text = out.text
            rewards = reward_fn_response_vs_gt(text, gt)
            if rewards.get("answer_reward", 0.0) >= 1.0:
                correct_generations += 1
                correct_pairs.append({"prompt": prompts[idx], "response": text})

    stats = {
        "total_generations": total_generations,
        "correct_generations": correct_generations,
        "kept_fraction": correct_generations / max(total_generations, 1),
    }
    return correct_pairs, stats


def train_on_pairs(
    model: torch.nn.Module,
    tokenizer,
    pairs: List[Dict[str, str]],
    optimizer,
):
    """Run SFT on the filtered EI dataset for a few epochs."""
    if not pairs:
        return {"losses": []}

    loader = DataLoader(
        pairs,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer),
    )

    # Number of optimizer steps per epoch after gradient accumulation.
    updates_per_epoch = math.ceil(len(loader) / grad_accum)
    total_updates = max(1, updates_per_epoch * train_epochs_per_step)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(0.1 * total_updates)),
        num_training_steps=total_updates,
    )

    optimizer.zero_grad()
    losses: List[float] = []
    micro_step = 0

    for epoch in range(train_epochs_per_step):
        for batch_idx, batch in enumerate(loader):
            logprob_out = get_response_log_probs(
                model=model,
                input_ids=batch["input_ids"],
                labels=batch["labels"],
                return_token_entropy=False,
                with_grad=True,
            )
            policy_log_probs = logprob_out["log_probs"]
            loss, _ = sft_microbatch_train_step(
                policy_log_probs=policy_log_probs,
                response_mask=batch["response_mask"],
                gradient_accumulation_steps=grad_accum,
                normalize_constant=1.0,
            )
            losses.append(loss.item())
            micro_step += 1

            should_step = (micro_step % grad_accum == 0) or (batch_idx + 1 == len(loader))
            if should_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if micro_step % log_every == 0:
                recent = losses[-log_every:]
                avg_loss = sum(recent) / len(recent)
                print(f"[train] micro_step={micro_step} avg_loss={avg_loss:.4f}")

    return {"losses": losses}


def run_expert_iteration():
    # Seeds for reproducibility.
    torch.manual_seed(sampling_seed)
    random.seed(sampling_seed)

    # Load data.
    all_prompts, all_ground_truths = sample_math_data()
    print(f"Loaded {len(all_prompts)} MATH prompts for EI.")

    # Initialize policy model.
    model, tokenizer = load_model_and_tokenizer(model_id)
    model.to(policy_device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.train()
    optimizer = TorchAdamW(model.parameters(), lr=lr)

    # Initialize vLLM for fast sampling.
    vllm_model = init_vllm(
        model_path=model_id,
        device=vllm_device,
        seed=vllm_seed,
        gpu_memory_utilization=vllm_gpu_mem_utilization,
    )
    sampling_params = build_ei_sampling_params()

    rng = random.Random(sampling_seed)

    for step in range(1, n_ei_steps + 1):
        print(f"\n=== EI step {step}/{n_ei_steps} ===")

        # Sync current policy weights to vLLM before sampling.
        load_policy_into_vllm_instance(model, vllm_model)

        # Sample a question batch and generate candidates.
        batch_prompts, batch_gts = sample_batch(all_prompts, all_ground_truths, ei_batch_size, rng)
        pairs, gen_stats = collect_ei_dataset(
            vllm_model=vllm_model,
            prompts=batch_prompts,
            ground_truths=batch_gts,
            sampling_params=sampling_params,
        )
        print(
            f"Generated {gen_stats['total_generations']} samples "
            f"kept {gen_stats['correct_generations']} "
            f"(fraction={gen_stats['kept_fraction']:.3f})"
        )

        if not pairs:
            print("No correct generations this step; skipping SFT update.")
            continue

        # Fine-tune on the filtered pairs.
        train_log = train_on_pairs(model, tokenizer, pairs, optimizer)
        if train_log["losses"]:
            print(f"Finished SFT on {len(pairs)} pairs; last loss={train_log['losses'][-1]:.4f}")

    print("\nEI finished. You can now save the model or run evaluation.")
    # Example: evaluation hook using the helper from q3 (optional).
    # eval_sampling_params = build_eval_sampling_params()
    # load_policy_into_vllm_instance(model, vllm_model)
    # metrics = evaluate_vllm(vllm_model, reward_fn_response_vs_gt, all_prompts[:64], all_ground_truths[:64], eval_sampling_params, output_jsonl=output_dir/"eval.jsonl", metrics_json=output_dir/"metrics.json")
    # print("Eval metrics:", metrics)


if __name__ == "__main__":
    run_expert_iteration()
