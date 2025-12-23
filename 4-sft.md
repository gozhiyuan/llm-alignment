# Supervised Fine-Tuning (SFT) Implementation Guide

This document provides a comprehensive guide to the SFT implementation, covering mathematical foundations, optimization strategies, and detailed code explanations.

## Table of Contents

1. [Mathematical Formulation](#mathematical-formulation)
2. [Implementation & Optimization Details](#implementation--optimization-details)
3. [Code Walkthrough](#code-walkthrough)
   - [Main Training Script (`q4_run_sft.py`)](#main-training-script-q4_run_sftpy)
   - [Helper Functions (`q4_sft_helpers.py`)](#helper-functions-q4_sft_helperspy)

---

## Mathematical Formulation

### Objective Function

Supervised Fine-Tuning (SFT) trains a language model to maximize the log-likelihood of target response tokens given the prompt. The objective is a **masked cross-entropy loss** that only applies to the response portion of each sequence.

For a batch of examples, let:
- `prompt_i` be the input prompt for example `i`
- `response_i` be the target response tokens for example `i`
- `θ` be the model parameters

The model processes the concatenated sequence `[prompt_i, response_i]` and predicts the next token at each position. We only compute loss over positions corresponding to response tokens.

### Loss Calculation

The loss for a single batch is computed as follows:

1. **Tokenization**: Each prompt-response pair is tokenized and concatenated:
   ```
   sequence_i = [prompt_tokens_i, response_tokens_i]
   ```

2. **Forward Pass**: The model produces logits `logits ∈ ℝ^(batch × seq_len × vocab_size)` for the entire sequence.

3. **Log-Probabilities**: We compute log-softmax over the vocabulary dimension:
   ```
   log_probs_all = log_softmax(logits, dim=-1)  # Shape: (batch, seq_len, vocab_size)
   ```

4. **Target Extraction**: For each position, we extract the log-probability assigned to the target token:
   ```
   log_probs = gather(log_probs_all, index=labels, dim=-1)  # Shape: (batch, seq_len)
   ```
   where `labels` is the sequence shifted by one position (teacher forcing).

5. **Masking**: We apply a `response_mask` that is `True` only for positions corresponding to response tokens:
   ```
   masked_ce_sum = sum(-log_probs * response_mask)  # Sum over all masked positions
   ```

6. **Normalization**: The loss is normalized by batch size and gradient accumulation steps:
   ```
   loss = masked_ce_sum / (batch_size × gradient_accumulation_steps)
   ```

**Key Insight**: The loss is the **mean negative log-likelihood (NLL)** over response tokens. Lower loss means the model assigns higher probability to the correct response tokens.

### Gradient Accumulation

When using gradient accumulation, we accumulate gradients over `grad_accum` microbatches before updating parameters. The normalization factor `batch_size × gradient_accumulation_steps` ensures that:
- The effective batch size is `batch_size × grad_accum`
- The loss scale is consistent regardless of accumulation steps
- Each microbatch contributes proportionally to the final gradient

---

## Implementation & Optimization Details

### Memory Management

Training large language models on limited GPU memory (e.g., A10G with 24GB) requires several optimization strategies:

#### 1. Gradient Checkpointing

**What it does**: Gradient checkpointing trades compute for memory by recomputing activations during the backward pass instead of storing them.

**How it works**:
- During forward pass: Only a few strategic checkpoints (e.g., layer inputs) are saved in GPU memory
- During backward pass: Missing activations are recomputed on-the-fly by re-running forward passes for intermediate segments
- **Trade-off**: Reduces activation memory by ~3-4x but increases training time by ~30%

**Implementation**:
```python
model.config.use_cache = False  # Disable KV cache (required for checkpointing)
model.gradient_checkpointing_enable()  # Enable recomputation
```

#### 2. Sequence Length Truncation

**Problem**: Very long prompt+response sequences can cause OOM during training.

**Solution**: We cap the total sequence length via `max_length` parameter:
- Sequences longer than `max_length` are truncated (tail tokens are lost)
- The `response_mask` is adjusted to only mark positions within the truncated sequence
- **Trade-off**: Some training examples may lose important tail information, but this is necessary for memory-constrained setups

**Note**: In production, you'd want `max_length` large enough to cover 99%+ of your data (e.g., 2048 or 4096). For A10G GPUs, we use smaller values (256-512) to fit within memory.

#### 3. Batch Size and Gradient Accumulation

- **Microbatch size**: `batch_size = 1` (smallest possible to minimize memory)
- **Effective batch size**: `batch_size × grad_accum = 1 × 4 = 4`
- This allows training with larger effective batches while keeping memory usage low

### Evaluation Strategy

The implementation supports two evaluation modes:

#### 1. vLLM Evaluation (Multi-GPU Setup)

**Architecture**:
- **Policy model**: Runs on `policy_device` (e.g., `cuda:0`) for training
- **vLLM instance**: Runs on `vllm_device` (e.g., `cuda:1`) for evaluation
- **Weight synchronization**: Before each eval, policy weights are copied to the vLLM instance via `load_policy_into_vllm_instance()`

**Advantages**:
- Training and evaluation can run on separate GPUs (no memory competition)
- vLLM's optimized serving stack provides faster inference
- No need to switch training model to eval mode

**Memory Considerations**:
- vLLM reserves `gpu_memory_utilization × total_gpu_memory` for KV cache
- For A10G (24GB), we use `gpu_memory_utilization = 0.4` to avoid OOM
- The vLLM instance holds a full copy of the model + KV cache

#### 2. Native Evaluation (Single-GPU Setup)

**When to use**: Set `use_vllm = False` for single-GPU runs

**How it works**:
- Uses the training model directly with `model.generate()`
- Temporarily switches model to eval mode
- No separate model copy needed

**Trade-offs**:
- Simpler setup, no second GPU required
- Training pauses during evaluation (synchronous)
- Risk of OOM if model + eval activations exceed GPU memory

### Data Pipeline

#### Prompt/Response Construction

The dataset contains messages with `role` (user/assistant) and `content`. The `build_text_pairs()` function extracts:
- **Prompt**: `user["content"]` (the question/instruction)
- **Response**: `assistant["content"]` (the full reasoning + answer)
- **Reference**: `user["info"]["reference_answer"]` (ground truth for evaluation)

#### Response Masking

The `tokenize_prompt_and_output()` function creates a `response_mask` that:
- Is `True` for positions corresponding to response tokens
- Is `False` for prompt tokens and padding
- Has shape `(batch, seq_len - 1)` (one less than sequence length due to shifting)

**Mask Creation Logic**:
```python
# For each example:
prompt_len = len(prompt_tokens)
output_len = len(response_tokens)
# Mask is True for positions [prompt_len, prompt_len + output_len)
mask = [prompt_len <= idx < prompt_len + output_len 
        for idx in range(1, len(merged_sequence))]
```

#### Dataset Filtering

The `filter_correct_dataset()` function filters examples where the assistant's response matches the reference answer:

1. **Format Check**: Uses `r1_zero_reward_fn()` which requires responses to have the format:
   ```
   <think>...</think> <answer>...</answer>
   ```

2. **Answer Matching**: The grader extracts the answer from the `<answer>` tag and compares it to the reference using:
   - Math-aware normalization (handles LaTeX, fractions, etc.)
   - SymPy-based equality checking (not just string matching)
   - Multiple fallback strategies for robustness

3. **Filtering**: Only examples with `answer_reward >= 1.0` are kept

**Note**: The filter uses the full `assistant["content"]` to match the format expected by the reward function.

---

## Code Walkthrough

### Main Training Script (`q4_run_sft.py`)

#### Configuration Section (Lines 29-55)

Key configuration parameters:

- **Model & Data**:
  - `model_id`: Path to base model (Qwen2.5-Math-1.5B)
  - `train_dataset_path`: Path to training JSONL file
  - `max_length`: Maximum sequence length (512 for A10G memory constraints)

- **Training Hyperparameters**:
  - `lr = 2e-5`: Learning rate
  - `batch_size = 1`: Microbatch size (minimal for memory)
  - `grad_accum = 4`: Gradient accumulation steps (effective batch = 4)
  - `num_epochs = 2`: Number of training epochs
  - `clip_grad = 1.0`: Gradient clipping threshold

- **Evaluation**:
  - `eval_every = 300`: Evaluate every N training steps
  - `use_vllm = True`: Use vLLM for evaluation (requires 2 GPUs)
  - `math_eval_examples = 16`: Number of MATH validation examples to evaluate

#### Data Loading (`load_data`, Lines 76-87)

```python
def load_data(subset=None, seed=42, test_size: float | None = train_val_split):
```

- Loads dataset from JSONL file using HuggingFace `datasets`
- If `subset` is provided, randomly samples that many examples
- If `test_size > 0`, splits into train/test; otherwise uses same data for both
- Returns a dict with `{"train": ..., "test": ...}`

#### Text Pair Construction (`build_text_pairs`, Lines 89-98)

```python
def build_text_pairs(example):
```

Extracts prompt/response pairs from the message format:
- Finds user and assistant messages by role
- Returns dict with `prompt`, `response`, and `reference` fields

#### Dataset Filtering (`filter_correct_dataset`, Lines 109-157)

```python
def filter_correct_dataset(ds, *, seed: int = 42):
```

**Process**:
1. For each example, extracts `assistant["content"]` as response
2. Extracts reference from `user["info"]["reference_answer"]` (with fallback)
3. Calls `reward_fn_response_vs_gt()` to check correctness
4. Filters examples where `answer_reward >= 1.0`
5. Logs statistics (before/after counts, debug samples)

**Debug Logging**: When `log_filter_stats = True`, prints:
- Filter statistics (before → after counts)
- Sample debug cases showing rewards, prompts, responses

#### vLLM Initialization (`init_vllm`, Lines 159-178)

```python
def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85) -> LLM:
```

**Monkeypatching** (from TRL):
- `world_size_patch`: Forces vLLM to use single-device mode (avoids distributed setup)
- `profiling_patch`: Skips memory profiling assertions that fail in our setting

**vLLM Configuration**:
- `device`: Target GPU (e.g., `"cuda:1"`)
- `dtype=torch.bfloat16`: Mixed precision for memory efficiency
- `enable_prefix_caching=True`: Cache prompt prefixes for faster inference
- `gpu_memory_utilization=0.4`: Reserve 40% of GPU for KV cache (conservative for A10G)

#### Weight Synchronization (`load_policy_into_vllm_instance`, Lines 181-187)

```python
def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
```

**Process**:
1. Extracts policy model's `state_dict()` (all parameters)
2. Accesses vLLM's internal model via `llm.llm_engine.model_executor.driver_worker.model_runner.model`
3. Calls `load_weights()` to copy parameters from policy to vLLM

**Note**: This creates a **copy** of weights (not a reference), so the two models are independent after copying.

#### Training Loop (`train_one_size`, Lines 210-326)

**Setup Phase** (Lines 220-251):
1. Load model and tokenizer, move to `policy_device`
2. Enable gradient checkpointing and disable KV cache
3. Create DataLoader with custom `collate_fn`
4. Initialize optimizer (AdamW) and learning rate scheduler
5. If `use_vllm`, initialize vLLM instance on separate GPU

**Training Iteration** (Lines 254-324):
```python
for epoch in range(num_epochs):
    for batch in train_loader:
        # 1. Forward pass with gradients
        logprob_out = get_response_log_probs(..., with_grad=True)
        
        # 2. Compute loss (only on response tokens)
        loss, _ = sft_microbatch_train_step(
            policy_log_probs=logprob_out["log_probs"],
            response_mask=batch["response_mask"],
            ...
        )
        
        # 3. Gradient accumulation
        if (step + 1) % grad_accum == 0:
            clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        # 4. Periodic evaluation
        if (step + 1) % eval_every == 0:
            # Generate and evaluate on MATH validation set
            gen_log = log_generations_vllm(...)  # or log_generations(...)
            acc = gen_log["summary"]["accuracy"]
```

**Key Points**:
- Loss is accumulated over `grad_accum` steps before updating
- Evaluation uses MATH validation set (not training data split)
- vLLM evaluation copies weights before each eval run

### Helper Functions (`q4_sft_helpers.py`)

#### Tokenization (`tokenize_prompt_and_output`, Lines 34-87)

```python
def tokenize_prompt_and_output(
    prompt_strs: List[str],
    output_strs: List[str],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int | None = None,
) -> dict[str, torch.Tensor]:
```

**Process**:
1. **Tokenize separately**: Prompt and output are tokenized independently (no special token overlap)
2. **Concatenate**: `merged = prompt_tokens + output_tokens`
3. **Truncate**: If `max_length` is set, truncate to that length
4. **Pad**: Pad to `max_len` (longest sequence in batch)
5. **Shift for labels**: 
   - `input_ids = merged[:-1]` (all but last token)
   - `labels = merged[1:]` (shifted by one for next-token prediction)
6. **Create mask**: `response_mask` marks positions corresponding to response tokens

**Output**:
- `input_ids`: Shape `(batch, seq_len - 1)`
- `labels`: Shape `(batch, seq_len - 1)`
- `response_mask`: Shape `(batch, seq_len - 1)`, boolean tensor

#### Loss Computation (`sft_microbatch_train_step`, Lines 130-167)

```python
def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,  # (batch, seq_len)
    response_mask: torch.Tensor,      # (batch, seq_len)
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
```

**Step-by-step**:
1. **Validate shapes**: `policy_log_probs` and `response_mask` must have same shape
2. **Convert mask to float**: `mask = response_mask.to(policy_log_probs.dtype)`
3. **Compute masked sum**: 
   ```python
   masked_ce_sum = sum(-policy_log_probs * mask) / normalize_constant
   ```
   This sums negative log-probs over all response token positions.
4. **Normalize**: 
   ```python
   loss = masked_ce_sum / (batch_size × gradient_accumulation_steps)
   ```
5. **Backward pass**: `loss.backward()` accumulates gradients

**Return**: `(loss, metadata)` where metadata includes `masked_ce_sum` and `num_response_tokens` for logging.

#### Log-Probability Extraction (`get_response_log_probs`, Lines 170-205)

```python
def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
    with_grad: bool = False,
) -> dict[str, torch.Tensor]:
```

**Process**:
1. **Forward pass**: `logits = model(input_ids).logits` → Shape `(batch, seq_len, vocab_size)`
2. **Log-softmax**: `log_probs_all = log_softmax(logits, dim=-1)` (numerically stable)
3. **Gather target log-probs**: 
   ```python
   log_probs = log_probs_all.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
   ```
   - `labels.unsqueeze(-1)` → `(batch, seq_len, 1)` for indexing
   - `gather()` selects the log-prob of the target token at each position
   - `squeeze(-1)` → back to `(batch, seq_len)`
4. **Optional entropy**: If `return_token_entropy=True`, computes entropy from full logits

**Gradient Control**: 
- `with_grad=False`: Uses `torch.no_grad()` context (for evaluation)
- `with_grad=True`: Uses `contextlib.nullcontext()` (for training)

#### Generation Logging (`log_generations`, Lines 208-328)

```python
def log_generations(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    ground_truths: list[str],
    reward_fn: Callable[[str, str], dict[str, float]],
    ...
) -> dict[str, Any]:
```

**Process**:
1. **Generate**: Uses `model.generate()` with sampling (temperature, top_p)
2. **Extract scores**: Gets per-token logits from `gen_out.scores`
3. **Compute entropy**: Uses `compute_entropy()` on stacked scores
4. **Decode responses**: Extracts generated text (excluding prompt)
5. **Compute rewards**: Calls `reward_fn()` for each prompt-response pair
6. **Aggregate statistics**: 
   - Accuracy (fraction with `answer_reward > 0`)
   - Average response lengths (overall, correct, incorrect)
   - Average token entropy

**Output**: Returns `{"records": [...], "summary": {...}}` with per-example details and aggregate stats.

#### vLLM Generation Logging (`log_generations_vllm`, Lines 331-387)

```python
def log_generations_vllm(
    vllm_model,
    prompts: list[str],
    ground_truths: list[str],
    reward_fn: Callable[[str, str], dict[str, float]],
    sampling_params,
    batch_size: int = 8,
) -> dict[str, Any]:
```

**Differences from `log_generations`**:
- Uses `vllm_model.generate()` instead of `model.generate()`
- Processes in batches for efficiency
- **Token entropy**: Not available from vLLM (would require top-k logprobs, which we don't currently request)
- Otherwise provides same output format (records + summary with accuracy)

**Note**: To get token entropy from vLLM, you'd need to:
1. Set `sampling_params.logprobs = k` (e.g., 5) to request top-k logprobs
2. Compute approximate entropy from the top-k distribution
3. This is an approximation since we don't have full vocabulary distribution

#### Entropy Computation (`compute_entropy`, Lines 91-115)

```python
def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
```

**Mathematical Derivation**:
- **Log-sum-exp**: `log_z = logsumexp(logits, dim=-1)` → log of partition function
- **Probabilities**: `probs = exp(logits - log_z)` (numerically stable)
- **Shannon Entropy**: `H = -sum(p_i * log(p_i)) = log_z - sum(p_i * logit_i)`

**Implementation**:
```python
log_z = torch.logsumexp(logits, dim=-1)  # (batch, seq_len)
probs = torch.exp(logits - log_z.unsqueeze(-1))  # (batch, seq_len, vocab)
entropy = log_z - (probs * logits).sum(dim=-1)  # (batch, seq_len)
```

**Output**: Shape `(batch, seq_len)` - entropy for each next-token prediction.

---

## Summary

This SFT implementation provides:

1. **Memory-efficient training** via gradient checkpointing, sequence truncation, and small batch sizes
2. **Flexible evaluation** supporting both vLLM (multi-GPU) and native (single-GPU) modes
3. **Comprehensive logging** with per-example records, accuracy metrics, and token entropy
4. **Robust data handling** with filtering, masking, and format validation

The code is designed to work within GPU memory constraints (A10G 24GB) while maintaining training efficiency and evaluation accuracy.

