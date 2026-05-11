from datetime import datetime
from datasets import load_dataset
from config.settings import LCB_DATASET, LCB_VERSION, N_SAMPLES


def load_livecode_bench(n_samples: int = None, after: str = None):
    """
    Load LiveCodeBench from HuggingFace.

    Each sample contains:
      - question_id:        unique identifier
      - question_title:     short title of the problem
      - question_content:   full problem statement (HTML/markdown)
      - difficulty:         'easy' | 'medium' | 'hard'
      - platform:           'leetcode' | 'atcoder' | 'codeforces'
      - contest_date:       when the problem was published
      - starter_code:       function stub to complete (may be empty)
      - public_test_cases:  example tests shown in the problem
      - private_test_cases: hidden evaluation tests (gzipped + base64)

    Args:
        n_samples: limit number of samples returned.
        after:     only include problems on or after this date e.g. '2024-01-01'
    """
    dataset = load_dataset(LCB_DATASET, name=LCB_VERSION, split="test")

    if after is not None:
        cutoff = datetime.strptime(after, "%Y-%m-%d")
        dataset = dataset.filter(
            lambda x: x["contest_date"].replace(tzinfo=None) >= cutoff
        )
        print(f"Filtered to problems on or after {after}: {len(dataset)} samples remaining")

    limit = n_samples if n_samples is not None else N_SAMPLES
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))

    print(f"Loaded {len(dataset)} samples from {LCB_DATASET} ({LCB_VERSION})")
    return dataset


def build_prompt(sample: dict) -> str:
    """
    Convert a LiveCodeBench sample into a plain-text prompt for an LLM.
    Includes the problem statement, public test cases, starter code,
    and output formatting instructions.
    """

    parts = ["Solve the following coding problem. Return ONLY the final code, no explanation, no code comments\n"]

    parts.append(f"Problem: {sample['question_title']}\n")
    parts.append(sample["question_content"])

    if sample.get("starter_code", "").strip():
        parts.append(f"\nUse this starter code:\n```python\n{sample['starter_code']}\n```")

    parts.append("\nReturn your solution as a Python code block (```python ... ```).")

    return "\n".join(parts)