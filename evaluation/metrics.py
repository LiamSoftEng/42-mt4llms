"""
Code metrics computed on extracted code blocks using radon.
"""

from functools import lru_cache
from itertools import combinations
import os
from statistics import mean, stdev
import warnings

from radon.raw import analyze
from radon.complexity import cc_visit


def _codesim_debug(message: str) -> None:
    if os.getenv("CODE_SIM_DEBUG") == "1":
        print(f"  [codesim] {message}")


@lru_cache(maxsize=1)
def _get_code_embedder():
    from sentence_transformers import SentenceTransformer
    from config.settings import CODE_EMBEDDING_MODEL

    os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")
    try:
        return SentenceTransformer(CODE_EMBEDDING_MODEL, model_kwargs={"use_safetensors": False})
    except Exception as exc:
        _codesim_debug(f"sentence-transformers init failed: {exc}")
        return None


@lru_cache(maxsize=1)
def _get_transformer_encoder():
    from config.settings import CODE_EMBEDDING_MODEL, CODE_TOKENIZER_MODEL

    os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HUGGINGFACE_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

    warnings.filterwarnings(
        "ignore",
        message="`resume_download` is deprecated*",
        category=FutureWarning,
        module="huggingface_hub",
    )

    from transformers import AutoConfig, AutoModel, AutoTokenizer, T5EncoderModel
    import torch
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()
    tokenizer_model = CODE_TOKENIZER_MODEL or CODE_EMBEDDING_MODEL
    config = AutoConfig.from_pretrained(CODE_EMBEDDING_MODEL)
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, use_fast=True)
    except Exception as exc:
        _codesim_debug(f"fast tokenizer load failed for {tokenizer_model}: {exc}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, use_fast=False)

    if config.model_type in {"t5", "mt5"}:
        model = T5EncoderModel.from_pretrained(CODE_EMBEDDING_MODEL, use_safetensors=False)
    else:
        model = AutoModel.from_pretrained(CODE_EMBEDDING_MODEL, use_safetensors=False)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device


def _encode_with_transformers(code_a: str, code_b: str):
    import torch

    tokenizer, model, device = _get_transformer_encoder()
    code_a = str(code_a)
    code_b = str(code_b)
    inputs = tokenizer([code_a, code_b], return_tensors="pt", padding=True, truncation=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    token_embeddings = outputs.last_hidden_state
    attention_mask = inputs["attention_mask"]
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = (token_embeddings * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    pooled = summed / counts
    pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
    return pooled[0], pooled[1]


def compute_code_similarity(code_a: str | None, code_b: str | None) -> float | None:
    """
    Compute cosine similarity between two code snippets using embeddings.
    Returns None when inputs are missing or embedding fails.
    """
    if not code_a or not code_b:
        return None
    try:
        emb_a, emb_b = _encode_with_transformers(code_a, code_b)
        import torch

        score = torch.nn.functional.cosine_similarity(emb_a, emb_b, dim=0).item()
        return float(score)
    except Exception as exc:
        _codesim_debug(f"transformers fallback failed: {exc}")
        return None


def compute_code_distance(code_a: str | None, code_b: str | None) -> float | None:
    """
    Compute embedding distance as 1 - cosine similarity.
    Returns None when similarity cannot be computed.
    """
    similarity = compute_code_similarity(code_a, code_b)
    if similarity is None:
        return None
    return float(1 - similarity)


def compute_pairwise_code_variability(codes: list[str | None]) -> dict:
    """
    Measure variability across repeated generated code samples.

    Each code sample is embedded and compared against every other sample.
    Standard deviation is computed over pairwise embedding distances, so
    higher values mean the prompt produced less consistent code across runs.
    """
    valid_codes = [code for code in codes if code]
    distances = []
    similarities = []

    for code_a, code_b in combinations(valid_codes, 2):
        similarity = compute_code_similarity(code_a, code_b)
        if similarity is None:
            continue
        similarities.append(float(similarity))
        distances.append(float(1 - similarity))

    if not distances:
        return {
            "n_codes": len(valid_codes),
            "n_pairs": 0,
            "mean_similarity": None,
            "std_similarity": None,
            "mean_distance": None,
            "std_distance": None,
        }

    return {
        "n_codes": len(valid_codes),
        "n_pairs": len(distances),
        "mean_similarity": round(mean(similarities), 6),
        "std_similarity": round(stdev(similarities), 6) if len(similarities) > 1 else 0.0,
        "mean_distance": round(mean(distances), 6),
        "std_distance": round(stdev(distances), 6) if len(distances) > 1 else 0.0,
    }


def compute_ncloc(code: str) -> int:
    """
    Non-comment lines of code - blank lines and comment lines excluded.
    Uses radon's lloc (logical lines of code).
    Returns -1 if code cannot be parsed.
    """
    try:
        result = analyze(code)
        return result.lloc
    except Exception:
        return -1


def compute_cyclomatic_complexity(code: str) -> float:
    """
    Average cyclomatic complexity across all functions/methods in the code.
    Returns -1 if code cannot be parsed or contains no functions.
    """
    try:
        blocks = cc_visit(code)
        if not blocks:
            return -1
        return round(sum(b.complexity for b in blocks) / len(blocks), 2)
    except Exception:
        return -1


def compute_metrics(code: str | None) -> dict:
    """
    Compute all metrics for a code block.
    Returns a dict with ncloc and cyclomatic_complexity.
    """
    if not code:
        return {"ncloc": -1, "cyclomatic_complexity": -1}
    return {
        "ncloc": compute_ncloc(code),
        "cyclomatic_complexity": compute_cyclomatic_complexity(code),
    }
