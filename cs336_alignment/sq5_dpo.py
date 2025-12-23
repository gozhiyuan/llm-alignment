from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase

# Alpaca instruction-following template used throughout the assignment.
ALPACA_TEMPLATE = (
    "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n"
    "{instruction}\n\n"
    "### Response:\n"
    "{response}"
)


def _format_with_alpaca_template(
    prompt: str, response: str, tokenizer: PreTrainedTokenizerBase
) -> str:
    """
    Format a prompt/response pair using the Alpaca template and append EOS.
    """
    if tokenizer.eos_token is None:
        raise ValueError("Tokenizer must define an eos_token for DPO.")
    # Ensure the response is explicitly terminated so log-probs account for EOS.
    terminated_response = response + tokenizer.eos_token
    return ALPACA_TEMPLATE.format(instruction=prompt, response=terminated_response)


def _sequence_log_prob(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    formatted_text: str,
    with_grad: bool,
) -> torch.Tensor:
    """
    Compute the unconditional log-probability of a fully formatted sequence.
    """
    encoded = tokenizer(formatted_text, return_tensors="pt")
    input_ids = encoded["input_ids"].to(model.device)

    # Shift for next-token prediction: logits at position i predict token i+1.
    labels = input_ids[:, 1:]
    inputs = input_ids[:, :-1]

    ctx = torch.enable_grad() if with_grad else torch.no_grad()
    with ctx:
        logits = model(inputs).logits
        log_probs_all = torch.log_softmax(logits, dim=-1)
        token_log_probs = log_probs_all.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        # Sum over the sequence to obtain log p(sequence).
        return token_log_probs.sum(dim=-1).squeeze(-1)


def compute_per_instance_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> torch.Tensor:
    """
    Compute the per-instance DPO loss for a single preference pair.
    Returns a scalar tensor on the device of `lm`.
    """
    chosen_text = _format_with_alpaca_template(prompt, response_chosen, tokenizer)
    rejected_text = _format_with_alpaca_template(prompt, response_rejected, tokenizer)

    # Policy log-probs require gradients; reference log-probs do not.
    logprob_pi_chosen = _sequence_log_prob(lm, tokenizer, chosen_text, with_grad=True)
    logprob_pi_rejected = _sequence_log_prob(lm, tokenizer, rejected_text, with_grad=True)
    logprob_ref_chosen = _sequence_log_prob(lm_ref, tokenizer, chosen_text, with_grad=False)
    logprob_ref_rejected = _sequence_log_prob(lm_ref, tokenizer, rejected_text, with_grad=False)

    # Move reference stats to the policy device for a single-device loss.
    delta_pi = logprob_pi_chosen - logprob_pi_rejected
    delta_ref = (logprob_ref_chosen - logprob_ref_rejected).to(lm.device)

    margin = beta * (delta_pi - delta_ref)
    loss = -F.logsigmoid(margin)
    return loss

