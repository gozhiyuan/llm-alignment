from typing import Any, Callable, List
import contextlib

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

def load_model_and_tokenizer(
    model_path: str = "data/a5-alignment/models/Qwen2.5-Math-1.5B",
    prefer_flash: bool = True,
):
    """
    Load model/tokenizer with a graceful fallback if flash-attn is missing or mismatched.
    """
    attn_impl = "flash_attention_2" if prefer_flash else "sdpa"
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            attn_implementation=attn_impl,
        )
    except ImportError as e:
        if not prefer_flash:
            raise
        # Fall back to PyTorch SDPA if flash-attn is not usable in the environment.
        print(f"Flash-Attn import failed ({e}); falling back to SDPA.")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

def tokenize_prompt_and_output(
    prompt_strs: List[str],
    output_strs: List[str],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int | None = None,
) -> dict[str, torch.Tensor]:
    """Tokenize prompt/output pairs and build response mask.

    The concatenated sequence is shifted to form input_ids / labels. The
    response_mask marks label positions that correspond to the output portion of
    each example (prompt + output + padding).
    """
    if len(prompt_strs) != len(output_strs):
        raise ValueError("prompt_strs and output_strs must have the same length")

    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer must define either pad_token_id or eos_token_id")

    prompt_tokenized = tokenizer(
        prompt_strs, add_special_tokens=True, padding=False, return_attention_mask=False
    )["input_ids"]
    output_tokenized = tokenizer(
        output_strs, add_special_tokens=True, padding=False, return_attention_mask=False
    )["input_ids"]

    max_len = max(len(p_ids) + len(o_ids) for p_ids, o_ids in zip(prompt_tokenized, output_tokenized))
    if max_length is not None:
        max_len = min(max_len, max_length)

    combined_sequences: list[list[int]] = []
    response_masks: list[list[bool]] = []
    for p_ids, o_ids in zip(prompt_tokenized, output_tokenized):
        merged = p_ids + o_ids
        merged = merged[:max_len]
        merged.extend([pad_token_id] * (max_len - len(merged)))

        prompt_len = len(p_ids)
        output_len = len(o_ids)
        # When truncated, cap the mask within the merged length.
        mask = [
            (prompt_len <= token_idx < prompt_len + output_len)
            for token_idx in range(1, len(merged))
        ][: max_len - 1]

        combined_sequences.append(merged)
        response_masks.append(mask)

    combined_tensor = torch.tensor(combined_sequences, dtype=torch.long)
    input_ids = combined_tensor[:, :-1]
    labels = combined_tensor[:, 1:]
    response_mask = torch.tensor(response_masks, dtype=torch.bool)

    return {"input_ids": input_ids, "labels": labels, "response_mask": response_mask}



def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Entropy over the last dimension of logits.
    Get the entropy of the next-token predictions (i.e., entropy over the vocabulary dimension).
    Args:
    logits: torch.Tensor Tensor of shape (batch_size, sequence_length, vocab_size)
    containing unnormalized logits.
    Returns:
    torch.Tensor Shape (batch_size, sequence_length). The entropy for each next-token
    prediction.
    """
    # log_z = log(sum exp(logits)); this is log Z, the normalizer, computed stably via log-sum-exp.
    log_z = torch.logsumexp(logits, dim=-1)
    # probs is softmax(logits): p_i = exp(logit_i) / sum_j exp(logit_j),
    # i.e., the next-token distribution over the vocab.
    # We compute it as exp(logits - log_z) to avoid overflow:
    #   log p_i = logits_i - log_z  (since p_i = exp(logits_i) / Z, log Z = log_z)
    #   probs   = exp(logits - log_z) gives the normalized probabilities without large exp(logits_i).
    #   shapes: log_z is (batch_size, seq_len); logits is (batch_size, seq_len, vocab_size).
    #   we unsqueeze log_z -> (batch_size, seq_len, 1) so it broadcasts across vocab when subtracting.
    probs = torch.exp(logits - log_z.unsqueeze(-1))
    # Shannon entropy H = -sum_i p_i log p_i, with log p_i = logit_i - log_z, so:
    #   H = -sum_i p_i (logit_i - log_z)
    #     = -sum_i p_i * logit_i + log_z * sum_i p_i
    #     = log_z - sum_i p_i * logit_i  (because sum_i p_i = 1).
    return log_z - (probs * logits).sum(dim=-1)


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
    normalize_constant: float = 1.0,
) -> torch.Tensor:
    """Sum over a dimension with a mask and divide by the provided constant."""
    masked = tensor * mask
    summed = masked.sum(dim=dim) if dim is not None else masked.sum()
    return summed / normalize_constant


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Compute the masked cross-entropy loss for a single microbatch and backprop.

    The loss is the sum of -log p over response tokens, normalized by batch size,
    an optional constant, and the gradient accumulation factor.
    """
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    if policy_log_probs.shape != response_mask.shape:
        raise ValueError("policy_log_probs and response_mask must have the same shape")

    batch_size = policy_log_probs.shape[0]
    mask = response_mask.to(policy_log_probs.dtype)

    # Cross-entropy per token is simply -log p when the target token is observed.
    masked_ce_sum = masked_normalize(
        tensor=-policy_log_probs,
        mask=mask,
        dim=None,
        normalize_constant=normalize_constant,
    )

    denom = batch_size * gradient_accumulation_steps
    loss = masked_ce_sum / denom

    loss.backward()

    metadata = {
        "masked_ce_sum": masked_ce_sum.detach(),
        "num_response_tokens": mask.sum().detach(),
    }
    return loss, metadata


def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
    with_grad: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Get per-token conditional log-probabilities and optional entropy from a causal LM.
    log_probs: (batch, seq_len) is the log-probability that the model assigns to the target (label) token at each position

    Args:
        model: HuggingFace causal LM (already on the correct device).
        input_ids: (batch, seq_len) token ids for prompt + response.
        labels: (batch, seq_len) target token ids (typically input_ids shifted left).
        return_token_entropy: whether to also return per-token entropy.
    """
    ctx = torch.no_grad() if not with_grad else contextlib.nullcontext()
    with ctx:
        # logits: unnormalized scores for each vocab token at every position.
        # Shape: (batch, seq_len, vocab)
        logits = model(input_ids).logits

        # log_probs_all: log-softmax over vocab, same shape as logits.
        # We work in log-space for numerical stability.
        log_probs_all = torch.log_softmax(logits, dim=-1)

        # Gather the log-prob assigned to the target token at each position.
        # labels shape: (batch, seq_len); we add a trailing dim so gather can select
        # along vocab axis, then squeeze to drop that singleton dim back to (batch, seq_len).
        log_probs = log_probs_all.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

        output: dict[str, torch.Tensor] = {"log_probs": log_probs}
        if return_token_entropy:
            output["token_entropy"] = compute_entropy(logits)
        return output


def log_generations(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    ground_truths: list[str],
    reward_fn: Callable[[str, str], dict[str, float]],
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    """
    Generate responses for a set of prompts and log useful diagnostics.

    Logs per-example prompt/response/ground-truth, reward info, avg token entropy,
    response length stats, and accuracy.
    """
    if len(prompts) != len(ground_truths):
        raise ValueError("prompts and ground_truths must have the same length")

    if device is not None:
        model = model.to(device)

    model.eval()
    with torch.no_grad():
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        input_ids = encoded["input_ids"].to(model.device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(model.device)

        prompt_lengths = attention_mask.sum(dim=1) if attention_mask is not None else torch.full(
            (input_ids.shape[0],), input_ids.shape[1], device=model.device
        )

        gen_out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        sequences = gen_out.sequences  # (batch, seq_len_total)
        scores = gen_out.scores  # list of length num_generated, each (batch, vocab)

        # Stack scores to shape (batch, gen_len, vocab) for entropy.
        if scores:
            scores_tensor = torch.stack(scores, dim=1)  # (batch, gen_len, vocab)
            token_entropy = compute_entropy(scores_tensor)
        else:
            token_entropy = None

        records: list[dict[str, Any]] = []
        response_lengths: list[int] = []
        response_lengths_correct: list[int] = []
        response_lengths_incorrect: list[int] = []
        entropies: list[float] = []
        correct = 0

        for idx, (seq, prompt_len, gt) in enumerate(zip(sequences, prompt_lengths, ground_truths)):
            prompt_len_int = int(prompt_len.item()) if isinstance(prompt_len, torch.Tensor) else int(prompt_len)
            response_ids = seq[prompt_len_int:]

            # Truncate at eos if present.
            if tokenizer.eos_token_id is not None:
                eos_positions = (response_ids == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
                if len(eos_positions) > 0:
                    response_ids = response_ids[: eos_positions[0]]

            response_len = response_ids.numel()
            response_text = tokenizer.decode(response_ids, skip_special_tokens=True)

            reward_info = reward_fn(response_text, gt)

            # Determine correctness heuristic: positive answer_reward counts as correct.
            is_correct = reward_info.get("answer_reward", 0.0) > 0
            correct += int(is_correct)
            response_lengths.append(response_len)
            (response_lengths_correct if is_correct else response_lengths_incorrect).append(response_len)

            # Per-example average token entropy over generated tokens (if available).
            avg_entropy = None
            if token_entropy is not None and token_entropy.shape[1] > 0 and response_len > 0:
                # token_entropy shape: (batch, gen_len)
                per_tok_ent = token_entropy[idx]
                avg_entropy = per_tok_ent[:response_len].mean().item()
                entropies.append(avg_entropy)

            record = {
                "prompt": prompts[idx],
                "response": response_text,
                "ground_truth": gt,
                "reward": reward_info,
                "response_length": response_len,
                "avg_token_entropy": avg_entropy,
            }
            records.append(record)

        def _safe_mean(values: list[int | float]) -> float | None:
            return float(sum(values) / len(values)) if values else None

        summary = {
            "avg_response_length": _safe_mean(response_lengths),
            "avg_response_length_correct": _safe_mean(response_lengths_correct),
            "avg_response_length_incorrect": _safe_mean(response_lengths_incorrect),
            "avg_token_entropy": _safe_mean(entropies),
            "accuracy": correct / max(len(prompts), 1),
        }

        return {"records": records, "summary": summary}


def log_generations_vllm(
    vllm_model,
    prompts: list[str],
    ground_truths: list[str],
    reward_fn: Callable[[str, str], dict[str, float]],
    sampling_params,
    batch_size: int = 8,
) -> dict[str, Any]:
    """
    vLLM-backed generation logger that returns per-example records and summary (incl. accuracy).
    """
    if len(prompts) != len(ground_truths):
        raise ValueError("prompts and ground_truths must have the same length")

    total = len(prompts)
    correct = 0
    records: list[dict[str, Any]] = []
    response_lengths: list[int] = []
    response_lengths_correct: list[int] = []
    response_lengths_incorrect: list[int] = []

    for start in range(0, total, batch_size):
        batch_prompts = prompts[start : start + batch_size]
        batch_ground_truths = ground_truths[start : start + batch_size]
        outputs = vllm_model.generate(batch_prompts, sampling_params)
        for idx_in_batch, request_output in enumerate(outputs):
            generated_text = request_output.outputs[0].text if request_output.outputs else ""
            gt = batch_ground_truths[idx_in_batch]
            rewards = reward_fn(generated_text, gt)
            is_correct = rewards.get("answer_reward", 0.0) >= 1.0
            correct += int(is_correct)
            response_len = len(generated_text.split())
            response_lengths.append(response_len)
            (response_lengths_correct if is_correct else response_lengths_incorrect).append(response_len)
            records.append(
                {
                    "prompt": batch_prompts[idx_in_batch],
                    "response": generated_text,
                    "ground_truth": gt,
                    "reward": rewards,
                    "response_length": response_len,
                    # Token entropy not available from vLLM generate output here.
                    "avg_token_entropy": None,
                }
            )

    def _safe_mean(values: list[int | float]) -> float | None:
        return float(sum(values) / len(values)) if values else None

    summary = {
        "avg_response_length": _safe_mean(response_lengths),
        "avg_response_length_correct": _safe_mean(response_lengths_correct),
        "avg_response_length_incorrect": _safe_mean(response_lengths_incorrect),
        "avg_token_entropy": None,
        "accuracy": correct / max(total, 1),
    }
    return {"records": records, "summary": summary}


if __name__ == "__main__":
    # Example usage: load model/tokenizer for supervised fine-tuning.
    model, tokenizer = load_model_and_tokenizer()