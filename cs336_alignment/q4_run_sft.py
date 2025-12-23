from pathlib import Path
from typing import Dict, Tuple
from unittest.mock import patch

import torch
from datasets import load_dataset, Features, Value
from torch.utils.data import DataLoader
from torch.optim import AdamW as TorchAdamW
from transformers import PreTrainedModel, get_linear_schedule_with_warmup
from vllm import LLM
from vllm.model_executor import set_random_seed as vllm_set_random_seed

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.q4_sft_helpers import (
    tokenize_prompt_and_output,
    sft_microbatch_train_step,
    log_generations,
    log_generations_vllm,
    load_model_and_tokenizer,
    get_response_log_probs,
)
from cs336_alignment.q3_run_zeroshot_benchmark import (
    DEFAULT_DATASET as MATH_DATASET,
    DEFAULT_PROMPT_PATH as MATH_PROMPT_PATH,
    build_sampling_params as build_vllm_sampling_params,
    load_math_examples,
)

# -------------------------------
# Config
# -------------------------------
model_id = "data/a5-alignment/models/Qwen2.5-Math-1.5B"
train_dataset_path = "data/a5-alignment/AM-DeepSeek-R1-Distilled-1.4M/am_0.9M_sample_1k.jsonl"
split_sizes = [512]  # None = full
max_length = 512  # cap prompt+response tokens to control memory
lr = 2e-5
batch_size = 1
num_epochs = 2
grad_accum = 4  # adjust with microbatch size to get effective batch
clip_grad = 1.0
eval_every = 300
loss_log_every = 100
train_val_split = 0  # set to 0 or None to disable holdout split
use_vllm = True  # set False for single-GPU runs to skip vLLM evaluator
policy_device = "cuda:0"
vllm_device = "cuda:1"
vllm_seed = 1234
vllm_gpu_mem_utilization = 0.4
math_eval_examples = 16
math_eval_batch_size = 1
log_filter_stats = True
log_generations_during_eval = True
log_generations_samples = 4
output_dir = Path("outputs/runs_sft_reasoning")
output_dir.mkdir(exist_ok=True, parents=True)

features = Features(
    {
        "messages": [
            {
                "role": Value("string"),
                "content": Value("string"),
                "info": {
                    "source": Value("string"),
                    "reference_answer": Value("string"),
                    "test_case": Value("string"),
                    "think_content": Value("string"),
                    "answer_content": Value("string"),
                },
            }
        ]
    }
)


def load_data(subset=None, seed=42, test_size: float | None = train_val_split):
    ds = load_dataset(
        "json",
        data_files={"train": train_dataset_path},
        features=features,
    )["train"]
    if subset:
        ds = ds.shuffle(seed=seed).select(range(subset))
    if test_size and test_size > 0:
        return ds.train_test_split(test_size=test_size, seed=seed)
    # No split requested; use the same data for train/val consumers.
    return {"train": ds, "test": ds}

def build_text_pairs(example):
    # prompt is the user message; response is the full assistant content (reasoning + answer)
    msgs = example["messages"]
    user = next(m for m in msgs if m["role"] == "user")
    assistant = next(m for m in msgs if m["role"] == "assistant")
    return {
        "prompt": user["content"],
        "response": assistant["content"],
        "reference": user["info"].get("reference_answer"),
    }

def normalize_answer(s: str) -> str:
    return " ".join(s.strip().lower().split())


def reward_fn_response_vs_gt(response: str, ground_truth: str) -> Dict[str, float]:
    """Math-aware reward using the same logic as zeroshot benchmark."""
    return r1_zero_reward_fn(response, ground_truth, fast=True)


def filter_correct_dataset(ds, *, seed: int = 42):
    """Filter dataset to only examples where provided assistant answer matches reference."""
    before = len(ds)
    missing_ref = 0
    empty_resp = 0
    debug_samples = []

    def _is_correct(example):
        msgs = example["messages"]
        user = next(m for m in msgs if m["role"] == "user")
        assistant = next(m for m in msgs if m["role"] == "assistant")
        ref = user["info"].get("reference_answer", "") or assistant["info"].get("reference_answer", "")
        if not ref:
            nonlocal missing_ref
            missing_ref += 1
            return False

        resp = assistant["content"]
        # # For debugging purposes, we can use the think and answer content instead of the full response.
        # think = assistant["info"].get("think_content", "") or ""
        # ans = assistant["info"].get("answer_content", "") or assistant["content"] or ""
        # # r1_zero_reward_fn requires the exact pattern with a space between the tags.
        # resp = f"<think>{think[:50]}</think> <answer>{ans}</answer>"
        if not resp:
            nonlocal empty_resp
            empty_resp += 1
            return False

        rewards = reward_fn_response_vs_gt(resp, ref)

        if log_filter_stats and len(debug_samples) < 5:
            debug_samples.append(
                {
                    "prompt": user["content"][:80],
                    "response_head": resp[:120],
                    "ref": ref[:120],
                    "rewards": rewards,
                }
            )

        return rewards.get("answer_reward", 0.0) >= 1.0

    filtered = ds.filter(_is_correct)
    filtered = filtered.shuffle(seed=seed)
    if log_filter_stats:
        print(f"Filter correct dataset: {before} -> {len(filtered)} examples; missing_ref={missing_ref}, empty_resp={empty_resp}")
        for idx, sample in enumerate(debug_samples, 1):
            print(f"[filter debug {idx}] rewards={sample['rewards']} prompt='{sample['prompt']}' response='{sample['response_head']}' ref='{sample['ref']}'")
    return filtered

def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85) -> LLM:
    """
    Start the inference process, placing the vLLM model on a GPU separate from the policy.
    """
    vllm_set_random_seed(seed)

    # Monkeypatch from TRL to force single-device placement and skip profiling checks.
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None,
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )


def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    """
    Load the current policy weights into the running vLLM instance.
    """
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


def load_math_validation_set(max_examples: int | None = None) -> Tuple[list[str], list[str]]:
    """
    Prepare prompts/ground-truth pairs from the MATH validation parquet for evaluation.
    """
    prompt_template = Path(MATH_PROMPT_PATH).read_text()
    examples = load_math_examples(
        MATH_DATASET,
        prompt_template,
        max_examples=max_examples,
    )
    prompts = [ex.prompt for ex in examples]
    ground_truths = [ex.ground_truth for ex in examples]
    return prompts, ground_truths

def collate_fn(batch, tokenizer):
    prompts = [b["prompt"] for b in batch]
    outputs = [b["response"] for b in batch]
    toks = tokenize_prompt_and_output(prompts, outputs, tokenizer, max_length=max_length)
    return {k: v.to(policy_device) for k, v in toks.items()}

def train_one_size(subset, *, filtered: bool = False):
    ds = load_data(subset=subset, test_size=train_val_split)
    if filtered:
        ds = {
            "train": filter_correct_dataset(ds["train"]),
            "test": filter_correct_dataset(ds["test"]),
        }
    train_ds = ds["train"].map(build_text_pairs)
    val_ds = ds["test"].map(build_text_pairs)

    model, tokenizer = load_model_and_tokenizer(model_id)
    model.to(policy_device)
    model.config.use_cache = False  # required for gradient checkpointing
    model.gradient_checkpointing_enable()
    model.train()

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=lambda b: collate_fn(b, tokenizer))
    optimizer = TorchAdamW(model.parameters(), lr=lr)
    total_steps = num_epochs * len(train_loader) // grad_accum
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.05 * total_steps), num_training_steps=total_steps
    )

    step = 0
    losses = []
    eval_steps = []
    eval_accs = []
    loss_steps = []
    loss_values = []

    # Prepare MATH validation resources on a separate GPU via vLLM.
    math_prompts, math_ground_truths = load_math_validation_set(max_examples=math_eval_examples)
    sampling_params = build_vllm_sampling_params() if use_vllm else None
    math_llm = None
    if use_vllm:
        math_llm = init_vllm(
            model_id,
            device=vllm_device,
            seed=vllm_seed,
            gpu_memory_utilization=vllm_gpu_mem_utilization,
        )

    optimizer.zero_grad()
    for epoch in range(num_epochs):
        for batch in train_loader:
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
            if (step + 1) % loss_log_every == 0:
                print(f"Step {step+1}: loss={loss.item():.4f} (avg={sum(losses[-50:])/50:.4f})")
            loss_steps.append(step + 1)
            loss_values.append(loss.item())
            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if (step + 1) % eval_every == 0:
                acc = 0.0
                summary = {}
                records = []
                if math_prompts:
                    if use_vllm:
                        gen_log = log_generations_vllm(
                            vllm_model=math_llm,
                            prompts=math_prompts,
                            ground_truths=math_ground_truths,
                            reward_fn=reward_fn_response_vs_gt,
                            sampling_params=sampling_params,
                            batch_size=math_eval_batch_size,
                        )
                    else:
                        gen_log = log_generations(
                            model=model,
                            tokenizer=tokenizer,
                            prompts=math_prompts,
                            ground_truths=math_ground_truths,
                            reward_fn=reward_fn_response_vs_gt,
                            max_new_tokens=128,
                            device=policy_device,
                        )
                    summary = gen_log.get("summary", {})
                    records = gen_log.get("records", [])
                    acc = summary.get("accuracy", 0.0)
                eval_steps.append(step + 1)
                eval_accs.append(acc)
                print(f"Step {step+1}: MATH val_acc={acc:.3f}")
                if log_generations_during_eval and records:
                    sample = records[: min(log_generations_samples, len(records))]
                    sample_summary = summary if summary is not None else {}
                    print(f"Step {step+1}: sample generations summary {sample_summary}")
                    for rec in sample:
                        print(f"- prompt: {rec['prompt'][:80]}...")
                        print(f"  response: {rec['response'][:120]}...")
                        print(f"  reward: {rec['reward']}")
                # Optional: save a checkpoint (uncomment to enable)
                # ckpt_dir = output_dir / f"checkpoints/step_{step+1}"
                # ckpt_dir.mkdir(parents=True, exist_ok=True)
                # model.save_pretrained(ckpt_dir)
                # tokenizer.save_pretrained(ckpt_dir)
            step += 1

    return {"loss": losses, "loss_steps": loss_steps, "loss_values": loss_values, "eval_steps": eval_steps, "eval_accs": eval_accs, "model": model, "tokenizer": tokenizer, "val_ds": val_ds}

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # results = {}
    # for subset in split_sizes:
    #     tag = "full" if subset is None else str(subset)
    #     print(f"=== Training on subset {tag} ===")
    #     res = train_one_size(subset)
    #     results[tag] = res
    #     plt.figure()
    #     plt.plot(res["eval_steps"], res["eval_accs"], label=f"{tag}")
    #     plt.xlabel("train steps")
    #     plt.ylabel("val accuracy")
    #     plt.title(f"SFT val acc ({tag})")
    #     plt.grid(True)
    #     plt.savefig(output_dir / f"val_acc_{tag}.png", dpi=150)
    #     plt.close()

    #     # Plot training loss
    #     plt.figure()
    #     plt.plot(res["loss_steps"], res["loss_values"], alpha=0.6, label="loss")
    #     plt.xlabel("train steps")
    #     plt.ylabel("loss")
    #     plt.title(f"SFT training loss ({tag})")
    #     plt.grid(True)
    #     plt.legend()
    #     plt.savefig(output_dir / f"loss_{tag}.png", dpi=150)
    #     plt.close()

    # # Example logging generations on a small val slice
    # any_tag = "full" if "full" in results else next(iter(results))
    # model = results[any_tag]["model"]
    # tok = results[any_tag]["tokenizer"]
    # # Sample generations on MATH validation prompts instead of the SFT train split.
    # math_prompts, math_ground_truths = load_math_validation_set(max_examples=math_eval_examples)
    # sample_prompts = math_prompts[: min(8, len(math_prompts))]
    # sample_gts = math_ground_truths[: len(sample_prompts)]
    # log = log_generations(model, tok, sample_prompts, sample_gts, reward_fn_response_vs_gt, device=policy_device)
    # print("Sample generations summary:", log["summary"])

    # Question 2: train on filtered (correct-only) dataset and plot
    print("=== Training on filtered full dataset (only correct examples) ===")
    filtered_res = train_one_size(None, filtered=True)
    plt.figure()
    plt.plot(filtered_res["eval_steps"], filtered_res["eval_accs"], label="filtered_full")
    plt.xlabel("train steps")
    plt.ylabel("val accuracy")
    plt.title("SFT val acc (filtered full)")
    plt.grid(True)
    plt.savefig(output_dir / "val_acc_filtered_full.png", dpi=150)
    plt.close()