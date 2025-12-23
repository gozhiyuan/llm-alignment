from __future__ import annotations

import argparse
import json
import pathlib
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence, Tuple

from vllm import LLM, SamplingParams

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - tqdm is optional at runtime
    tqdm = None

from cs336_alignment.drgrpo_grader import extract_answer, r1_zero_reward_fn


DEFAULT_DATASET = "data/a5-alignment/MATH/data/train-00000-of-00001-7320a6f3aba8ebd2.parquet"
MAX_EXAMPLES = 100
DEFAULT_MODEL = "data/a5-alignment/models/Qwen2.5-Math-1.5B"
DEFAULT_PROMPT_PATH = pathlib.Path(__file__).parent / "prompts" / "r1_zero.prompt"
DEFAULT_OUTPUT_JSONL = pathlib.Path(__file__).parent / "outputs" / "zeroshot_math_generations.jsonl"
DEFAULT_METRICS_JSON = pathlib.Path(__file__).parent / "outputs" / "zeroshot_math_metrics.json"


@dataclass
class MathExample:
    prompt: str
    ground_truth: str
    raw_question: str
    raw_answer: str


def _batched(seq: Sequence, batch_size: int) -> Iterable[Tuple[int, Sequence]]:
    for start in range(0, len(seq), batch_size):
        yield start, seq[start : start + batch_size]


def _load_prompt_template(prompt_path: pathlib.Path) -> str:
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt template not found at {prompt_path}")
    return prompt_path.read_text()


def _coerce_to_str(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return _coerce_to_str(value[0])
    return str(value)


def load_math_examples(
    parquet_path: str,
    prompt_template: str,
    *,
    question_column: str | None = None,
    answer_column: str | None = None,
    max_examples: int | None = None,
) -> list[MathExample]:
    """
    Load MATH-style examples from a parquet file and format them with r1_zero prompt.

    The loader tries a few common column names if none are provided. If the answer column
    looks like a full solution, we attempt to extract a boxed answer for grading.
    """
    df = None
    import_error: Exception | None = None
    try:
        import pandas as pd  # type: ignore

        df = pd.read_parquet(parquet_path)
    except Exception as exc:  # pragma: no cover - depends on runtime deps
        import_error = exc
    if df is None:
        try:
            import pyarrow.parquet as pq  # type: ignore

            table = pq.read_table(parquet_path)
            df = table.to_pandas()
        except Exception as exc:  # pragma: no cover - depends on runtime deps
            if import_error is not None:
                raise RuntimeError(
                    "Failed to load parquet file. Please install pandas or pyarrow in "
                    "your SageMaker environment."
                ) from exc
            raise

    columns = set(df.columns)
    resolved_question = question_column or next(
        (c for c in ("question", "problem", "prompt", "query") if c in columns), None
    )
    resolved_answer = answer_column or next(
        (c for c in ("answer", "ground_truth", "gt", "final_answer", "solution", "solutions") if c in columns),
        None,
    )
    if resolved_question is None or resolved_answer is None:
        raise KeyError(
            f"Could not infer question/answer columns from available columns {sorted(columns)}. "
            "Please pass --question-column and --answer-column."
        )

    if max_examples is not None:
        df = df.iloc[:max_examples]

    examples: list[MathExample] = []
    for _, row in df.iterrows():
        raw_q = _coerce_to_str(row[resolved_question])
        raw_a = row[resolved_answer]
        answer_str = _coerce_to_str(raw_a)
        # If the answer column contains a full solution, pull the boxed answer when possible.
        parsed_answer = extract_answer(answer_str) or answer_str
        prompt = prompt_template.format(question=raw_q)
        examples.append(
            MathExample(
                prompt=prompt,
                ground_truth=parsed_answer,
                raw_question=raw_q,
                raw_answer=answer_str,
            )
        )
    return examples


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: Sequence[str],
    ground_truths: Sequence[str],
    eval_sampling_params: SamplingParams,
    *,
    output_jsonl: pathlib.Path,
    metrics_json: pathlib.Path,
    batch_size: int = 16,
) -> dict[str, float]:
    """
    Evaluate a language model on a list of prompts, compute metrics, and serialize results.
    """
    if len(prompts) != len(ground_truths):
        raise ValueError("Prompts and ground_truths must have the same length.")

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    metrics_json.parent.mkdir(parents=True, exist_ok=True)

    total = len(prompts)
    format_reward_sum = 0.0
    answer_reward_sum = 0.0
    both_one = 0
    format_one_answer_zero = 0
    format_zero_answer_zero = 0

    iterator = _batched(list(range(total)), batch_size)
    if tqdm is not None:
        iterator = tqdm(iterator, total=(total + batch_size - 1) // batch_size, desc="Evaluating")

    with output_jsonl.open("w", encoding="utf-8") as out_f:
        for start_idx, batch_indices in iterator:
            batch_prompts = [prompts[i] for i in batch_indices]
            batch_ground_truths = [ground_truths[i] for i in batch_indices]
            outputs = vllm_model.generate(batch_prompts, eval_sampling_params)

            for idx_in_batch, request_output in enumerate(outputs):
                prompt_idx = start_idx + idx_in_batch
                generated_text = ""
                if request_output.outputs:
                    generated_text = request_output.outputs[0].text

                rewards = reward_fn(generated_text, batch_ground_truths[idx_in_batch])
                format_reward = float(rewards.get("format_reward", 0.0))
                answer_reward = float(rewards.get("answer_reward", 0.0))

                format_reward_sum += format_reward
                answer_reward_sum += answer_reward
                if format_reward >= 1.0 and answer_reward >= 1.0:
                    both_one += 1
                elif format_reward >= 1.0:
                    format_one_answer_zero += 1
                else:
                    format_zero_answer_zero += 1

                record = {
                    "prompt_index": prompt_idx,
                    "prompt": batch_prompts[idx_in_batch],
                    "ground_truth": batch_ground_truths[idx_in_batch],
                    "generation": generated_text,
                    "rewards": rewards,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    metrics = {
        "n_examples": total,
        "format_reward_rate": format_reward_sum / total if total else 0.0,
        "answer_reward_rate": answer_reward_sum / total if total else 0.0,
        "accuracy": both_one / total if total else 0.0,
        "format_and_answer_1": both_one,
        "format_1_answer_0": format_one_answer_zero,
        "format_0_answer_0": format_zero_answer_zero,
    }
    metrics_json.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    return metrics


def build_sampling_params() -> SamplingParams:
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )
    return sampling_params


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen2.5-Math-1.5B zero-shot on MATH (parquet).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-path", type=str, default=DEFAULT_DATASET, help="Path to MATH parquet file.")
    parser.add_argument(
        "--prompt-path",
        type=str,
        default=str(DEFAULT_PROMPT_PATH),
        help="Path to r1_zero prompt template with {question} placeholder.",
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Path or name for vLLM model.")
    parser.add_argument(
        "--output-jsonl",
        type=str,
        default=str(DEFAULT_OUTPUT_JSONL),
        help="Where to save generations + rewards (JSONL).",
    )
    parser.add_argument(
        "--metrics-json",
        type=str,
        default=str(DEFAULT_METRICS_JSON),
        help="Where to save aggregate metrics (JSON).",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Number of prompts per vLLM.generate call.")
    parser.add_argument("--max-examples", type=int, default=MAX_EXAMPLES, help="Optional cap on number of eval examples.")
    parser.add_argument("--question-column", type=str, default=None, help="Override question column name.")
    parser.add_argument("--answer-column", type=str, default=None, help="Override answer column name.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="tensor_parallel_size for vLLM.")
    parser.add_argument("--dtype", type=str, default="auto", help="dtype argument forwarded to vLLM.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    prompt_template = _load_prompt_template(pathlib.Path(args.prompt_path))
    examples = load_math_examples(
        args.dataset_path,
        prompt_template,
        question_column=args.question_column,
        answer_column=args.answer_column,
        max_examples=args.max_examples,
    )
    prompts = [ex.prompt for ex in examples]
    ground_truths = [ex.ground_truth for ex in examples]

    sampling_params = build_sampling_params()
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
    )

    metrics = evaluate_vllm(
        llm,
        r1_zero_reward_fn,
        prompts,
        ground_truths,
        sampling_params,
        output_jsonl=pathlib.Path(args.output_jsonl),
        metrics_json=pathlib.Path(args.metrics_json),
        batch_size=args.batch_size,
    )

    print(json.dumps(metrics, indent=2))
    print(f"Saved generations to {args.output_jsonl}")
    print(f"Saved metrics to {args.metrics_json}")


if __name__ == "__main__":
    main()

