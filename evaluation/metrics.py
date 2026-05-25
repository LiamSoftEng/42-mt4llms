"""
Code metrics computed on extracted code blocks using radon.
"""

from functools import lru_cache
import os

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
    from transformers import AutoConfig, AutoModel, AutoTokenizer, T5EncoderModel, T5Tokenizer, T5TokenizerFast
    import torch
    from config.settings import CODE_EMBEDDING_MODEL, CODE_TOKENIZER_MODEL

    os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HUGGINGFACE_HUB_DISABLE_PROGRESS_BARS", "1")
    try:
        from transformers.utils import logging as hf_logging

        hf_logging.disable_progress_bar()
    except Exception:
        pass
    tokenizer_model = CODE_TOKENIZER_MODEL or CODE_EMBEDDING_MODEL
    config = AutoConfig.from_pretrained(CODE_EMBEDDING_MODEL)
    if config.model_type in {"t5", "mt5"}:
        tokenizer = None
        try:
            tokenizer = T5TokenizerFast.from_pretrained(tokenizer_model)
        except Exception as exc:
            _codesim_debug(f"fast tokenizer load failed for {tokenizer_model}: {exc}")
        if tokenizer is None:
            try:
                tokenizer = T5Tokenizer.from_pretrained(tokenizer_model)
            except Exception as exc:
                _codesim_debug(f"slow tokenizer load failed for {tokenizer_model}: {exc}")
        if tokenizer is None:
            raise RuntimeError(f"Unable to load tokenizer for {tokenizer_model}")
    else:
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
