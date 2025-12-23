from __future__ import annotations

import gzip
import json
import os
import random
from typing import Iterable, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase

from cs336_alignment.sq5_dpo import ALPACA_TEMPLATE

# Anthropic HH training shards.
HH_FILENAMES = [
    "harmless-base.jsonl.gz",
    "helpful-base.jsonl.gz",
    "helpful-online.jsonl.gz",
    "helpful-rejection-sampled.jsonl.gz",
]


def _parse_conversation(conversation: str) -> List[Tuple[str, str]]:
    """
    Parse a HH conversation string into (role, text) tuples.
    Roles are expected to start with "Human:" or "Assistant:".
    """
    turns: List[Tuple[str, str]] = []
    for block in conversation.strip().split("\n\n"):
        block = block.strip()
        if not block:
            continue
        if block.startswith("Human:"):
            turns.append(("human", block[len("Human:") :].strip()))
        elif block.startswith("Assistant:"):
            turns.append(("assistant", block[len("Assistant:") :].strip()))
    return turns


def _extract_single_turn(turns: List[Tuple[str, str]]) -> tuple[str | None, str | None]:
    """
    Return (instruction, first_assistant_reply) if there is exactly one human turn.
    Otherwise return (None, None) to signal the example should be skipped.
    """
    human_turns = [t for t in turns if t[0] == "human"]
    assistant_turns = [t for t in turns if t[0] == "assistant"]
    if len(human_turns) != 1 or len(assistant_turns) == 0:
        return None, None
    return human_turns[0][1], assistant_turns[0][1]


def load_hh_dataset(root_dir: str) -> list[dict[str, str]]:
    """
    Load and preprocess the Anthropic HH dataset from `root_dir`.

    Processing rules:
    - Combine the four provided training shards.
    - Ignore multi-turn conversations (require exactly one human message).
    - Extract the first human message as the instruction.
    - Extract the first assistant message from each of the chosen/rejected conversations
      as the responses.
    - Record the source filename for analysis.

    Returns:
        List of dicts with keys: instruction, response_chosen, response_rejected, source.
    """
    examples: list[dict[str, str]] = []

    for fname in HH_FILENAMES:
        path = os.path.join(root_dir, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"HH shard not found: {path}")

        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                chosen_turns = _parse_conversation(record["chosen"])
                rejected_turns = _parse_conversation(record["rejected"])

                instr_chosen, resp_chosen = _extract_single_turn(chosen_turns)
                instr_rejected, resp_rejected = _extract_single_turn(rejected_turns)

                # Skip if either side is multi-turn or missing.
                if instr_chosen is None or instr_rejected is None:
                    continue
                # Ensure both conversations start from the same instruction.
                if instr_chosen != instr_rejected:
                    continue

                examples.append(
                    {
                        "instruction": instr_chosen,
                        "response_chosen": resp_chosen,
                        "response_rejected": resp_rejected,
                        "source": fname,
                    }
                )

    return examples


class PackedSFTDataset(Dataset):
    """Fixed-length, packed LM examples."""

    def __init__(self, examples: list[dict[str, torch.Tensor]]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.examples[idx]


def _format_instruction(prompt: str, response: str, tokenizer: PreTrainedTokenizerBase) -> str:
    if tokenizer.eos_token is None:
        raise ValueError("Tokenizer must define an eos_token for packing.")
    return ALPACA_TEMPLATE.format(
        instruction=prompt,
        response=response + tokenizer.eos_token,
    )


def get_packed_sft_dataset(
    tokenizer: PreTrainedTokenizerBase,
    dataset_path: str | os.PathLike,
    seq_length: int,
    shuffle: bool,
) -> Dataset:
    """
    Construct a packed SFT dataset of fixed-length LM examples.
    """
    with open(dataset_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    if shuffle:
        random.shuffle(records)

    # Tokenize all formatted examples into one long stream of token ids.
    token_stream: list[int] = []
    for rec in records:
        text = _format_instruction(rec["prompt"], rec["response"], tokenizer)
        encoded = tokenizer(
            text,
            add_special_tokens=True,
            padding=False,
            truncation=False,
            return_attention_mask=False,
        )["input_ids"]
        token_stream.extend(encoded)

    examples: list[dict[str, torch.Tensor]] = []
    # Build overlapping chunks so that labels are inputs shifted by one token.
    for start in range(0, len(token_stream) - 1, seq_length):
        input_ids = token_stream[start : start + seq_length]
        labels = token_stream[start + 1 : start + seq_length + 1]
        if len(input_ids) < seq_length or len(labels) < seq_length:
            break
        examples.append(
            {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            }
        )

    return PackedSFTDataset(examples)


def iterate_batches(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """Wrap the dataset in a DataLoader for one epoch of batches."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
from __future__ import annotations

import gzip
import json
import os
from typing import Iterable, List, Tuple

# The four Anthropic HH training shards we combine.
HH_FILENAMES = [
    "harmless-base.jsonl.gz",
    "helpful-base.jsonl.gz",
    "helpful-online.jsonl.gz",
    "helpful-rejection-sampled.jsonl.gz",
]


def _parse_conversation(conversation: str) -> List[Tuple[str, str]]:
    """
    Parse a HH conversation string into (role, text) tuples.
    Roles are expected to start with \"Human:\" or \"Assistant:\".
    """
    turns: List[Tuple[str, str]] = []
    for block in conversation.strip().split("\\n\\n"):
        block = block.strip()
        if not block:
            continue
        if block.startswith("Human:"):
            turns.append(("human", block[len("Human:") :].strip()))
        elif block.startswith("Assistant:"):
            turns.append(("assistant", block[len("Assistant:") :].strip()))
    return turns


def _extract_single_turn(turns: List[Tuple[str, str]]) -> tuple[str | None, str | None]:
    """
    Return (instruction, first_assistant_reply) if there is exactly one human turn.
    Otherwise return (None, None) to signal the example should be skipped.
    """
    human_turns = [t for t in turns if t[0] == "human"]
    assistant_turns = [t for t in turns if t[0] == "assistant"]
    if len(human_turns) != 1 or len(assistant_turns) == 0:
        return None, None
    return human_turns[0][1], assistant_turns[0][1]


def load_hh_dataset(root_dir: str) -> list[dict[str, str]]:
    """
    Load and preprocess the Anthropic HH dataset from `root_dir`.

    Processing rules:
    - Combine the four provided training shards.
    - Ignore multi-turn conversations (require exactly one human message).
    - Extract the first human message as the instruction.
    - Extract the first assistant message from each of the chosen/rejected conversations
      as the responses.
    - Record the source filename for analysis.

    Returns:
        List of dicts with keys: instruction, response_chosen, response_rejected, source.
    """
    examples: list[dict[str, str]] = []

    for fname in HH_FILENAMES:
        path = os.path.join(root_dir, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"HH shard not found: {path}")

        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                chosen_turns = _parse_conversation(record["chosen"])
                rejected_turns = _parse_conversation(record["rejected"])

                instr_chosen, resp_chosen = _extract_single_turn(chosen_turns)
                instr_rejected, resp_rejected = _extract_single_turn(rejected_turns)

                # Skip if either side is multi-turn or missing.
                if instr_chosen is None or instr_rejected is None:
                    continue
                # Ensure both conversations start from the same instruction.
                if instr_chosen != instr_rejected:
                    continue

                examples.append(
                    {
                        "instruction": instr_chosen,
                        "response_chosen": resp_chosen,
                        "response_rejected": resp_rejected,
                        "source": fname,
                    }
                )

    return examples

