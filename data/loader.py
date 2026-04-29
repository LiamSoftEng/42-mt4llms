from datasets import load_dataset
from config.settings import LCB_DATASET, LCB_VERSION, N_SAMPLES


def load_livecode_bench(n_samples: int = None):
    """
    Load LiveCodeBench from HuggingFace.

    Each sample contains:
      - question_id:      unique identifier
      - question_title:   short title of the problem
      - question_content: full problem statement (HTML/markdown)
      - difficulty:       'easy' | 'medium' | 'hard'
      - platform:         'leetcode' | 'atcoder' | 'codeforces'
      - release_date:     when the problem was published
      - starter_code:     function stub to complete (may be empty)
      - public_tests:     list of {input, output} dicts for examples shown in the problem
      - private_tests:    list of {input, output} dicts used for hidden evaluation

    Args:
        n_samples: if set, overrides N_SAMPLES from settings.
    """
    dataset = load_dataset(LCB_DATASET, name=LCB_VERSION, split="test")

    limit = n_samples if n_samples is not None else N_SAMPLES
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))

    print(f"Loaded {len(dataset)} samples from {LCB_DATASET} ({LCB_VERSION})")
    return dataset


def build_prompt(sample: dict) -> str:
    """
    Convert a LiveCodeBench sample into a plain-text prompt for an LLM.
    Includes the problem statement, any starter code, and output formatting instructions.
    """
    parts = ["Solve the following coding problem. Return ONLY the final code, no explanation.\n"]

    parts.append(f"Problem: {sample['question_title']}\n")
    parts.append(sample["question_content"])

    if sample.get("starter_code", "").strip():
        parts.append(f"\nUse this starter code:\n```python\n{sample['starter_code']}\n```")

    parts.append("\nReturn your solution as a Python code block (```python ... ```).")

    return "\n".join(parts)