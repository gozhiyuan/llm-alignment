from __future__ import annotations

import argparse
import os
import random
from typing import Iterable, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cs336_alignment.hh_dataset import load_hh_dataset
from cs336_alignment.sq5_dpo import (
    _format_with_alpaca_template,
    _sequence_log_prob,
    compute_per_instance_dpo_loss,
)


def _set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _maybe_pad_token(tokenizer):
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token


def classification_accuracy(
    model: torch.nn.Module,
    tokenizer,
    examples: Iterable[dict[str, str]],
) -> float:
    """Compute preference classification accuracy on a list of examples."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for ex in examples:
            chosen_text = _format_with_alpaca_template(ex["instruction"], ex["response_chosen"], tokenizer)
            rejected_text = _format_with_alpaca_template(ex["instruction"], ex["response_rejected"], tokenizer)
            lp_chosen = _sequence_log_prob(model, tokenizer, chosen_text, with_grad=False)
            lp_rejected = _sequence_log_prob(model, tokenizer, rejected_text, with_grad=False)
            correct += int(lp_chosen.item() > lp_rejected.item())
            total += 1
    return correct / max(total, 1)


def train_dpo(
    model_path: str,
    hh_root: str,
    output_dir: str,
    batch_size: int = 64,
    beta: float = 0.1,
    lr: float = 1e-6,
    epochs: int = 1,
    val_size: int = 200,
    grad_accum_steps: int = 8,
    seed: int = 42,
    policy_device: str = "cuda:0",
    ref_device: str = "cuda:1",
):
    """
    Train an instruction-tuned model with DPO on the HH dataset.
    """
    _set_seed(seed)

    os.makedirs(output_dir, exist_ok=True)
    raw_examples = load_hh_dataset(hh_root)
    random.shuffle(raw_examples)
    val_split = raw_examples[:val_size]
    train_split = raw_examples[val_size:]

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    _maybe_pad_token(tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map={"": policy_device}
    )
    model_ref = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map={"": ref_device}
    )
    model_ref.eval()

    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)

    best_val_acc = -1.0
    global_step = 0

    for epoch in range(epochs):
        model.train()
        for start in range(0, len(train_split), batch_size):
            batch = train_split[start : start + batch_size]
            for idx, ex in enumerate(batch):
                loss = compute_per_instance_dpo_loss(
                    lm=model,
                    lm_ref=model_ref,
                    tokenizer=tokenizer,
                    beta=beta,
                    prompt=ex["instruction"],
                    response_chosen=ex["response_chosen"],
                    response_rejected=ex["response_rejected"],
                )
                # Gradient accumulation to emulate larger batches.
                (loss / grad_accum_steps).backward()
                if (idx + 1) % grad_accum_steps == 0 or idx == len(batch) - 1:
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

        val_acc = classification_accuracy(model, tokenizer, val_split)
        print(f"Epoch {epoch+1}: validation accuracy={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(output_dir, "best_model")
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            print(f"Saved new best model to {save_path}")

    print(f"Training complete. Best validation accuracy: {best_val_acc:.4f}")


def main():
    parser = argparse.ArgumentParser(description="DPO training on Anthropic HH.")
    parser.add_argument("--model_path", required=True, help="Path or HF id of the instruction-tuned model.")
    parser.add_argument("--hh_root", required=True, help="Directory containing HH jsonl.gz shards.")
    parser.add_argument("--output_dir", required=True, help="Where to save checkpoints.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--val_size", type=int, default=200)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--policy_device", default="cuda:0")
    parser.add_argument("--ref_device", default="cuda:1")
    args = parser.parse_args()

    train_dpo(
        model_path=args.model_path,
        hh_root=args.hh_root,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        beta=args.beta,
        lr=args.lr,
        epochs=args.epochs,
        val_size=args.val_size,
        grad_accum_steps=args.grad_accum_steps,
        seed=args.seed,
        policy_device=args.policy_device,
        ref_device=args.ref_device,
    )


if __name__ == "__main__":
    main()

