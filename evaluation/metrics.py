"""
Code metrics computed on extracted code blocks using radon.
"""

from functools import lru_cache
import io
from itertools import combinations
import os
from statistics import mean, stdev
import textwrap
import tokenize
import warnings

from radon.raw import analyze
from radon.complexity import cc_visit

from evaluation.executor import extract_code
from config.settings import configure_huggingface_environment


def _codesim_debug(message: str) -> None:
    if os.getenv("CODE_SIM_DEBUG") == "1":
        print(f"  [codesim] {message}")


# ---------------------------------------------------------------------------
# Embedding backend
# ---------------------------------------------------------------------------

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
    configure_huggingface_environment()

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


# ---------------------------------------------------------------------------
# Similarity / distance
# ---------------------------------------------------------------------------

def _strip_comments_for_codesim(code: str) -> str:
    """Remove Python comments without changing strings or stored code."""
    try:
        filtered = []
        skip_comment_linebreak = False
        for token in tokenize.generate_tokens(io.StringIO(code).readline):
            if token.type == tokenize.COMMENT:
                skip_comment_linebreak = not token.line[:token.start[1]].strip()
                continue
            if skip_comment_linebreak and token.type in (tokenize.NL, tokenize.NEWLINE):
                skip_comment_linebreak = False
                continue
            filtered.append((token.type, token.string))
        return tokenize.untokenize(filtered)
    except (IndentationError, tokenize.TokenError):
        return code


def compute_code_similarity(code_a: str | None, code_b: str | None) -> float | None:
    """
    Compute cosine similarity between two code snippets using embeddings.
    Returns None when inputs are missing or embedding fails.
    """
    if not code_a or not code_b:
        return None
    try:
        emb_a, emb_b = _encode_with_transformers(
            _strip_comments_for_codesim(code_a),
            _strip_comments_for_codesim(code_b),
        )
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


def compute_pairwise_similarity_list(codes: list[str | None]) -> list[float]:
    """
    Raw list of pairwise cosine similarities among valid code samples, one
    entry per unique pair. compute_pairwise_code_variability() only returns
    the aggregated mean/std of these values — this exists because the t-test
    baseline for code_sim needs the individual pairwise values themselves,
    not a summary.
    """
    valid_codes = [c for c in codes if c]
    similarities = []
    for code_a, code_b in combinations(valid_codes, 2):
        similarity = compute_code_similarity(code_a, code_b)
        if similarity is not None:
            similarities.append(float(similarity))
    return similarities


# ---------------------------------------------------------------------------
# Static code metrics
# ---------------------------------------------------------------------------

def compute_ncloc(code: str) -> int:
    """
    Non-comment lines of code — blank lines and comment lines excluded.
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

    radon's cc_visit only walks function/method bodies, so it has nothing to
    measure in flat/procedural code with no def blocks (common for
    LeetCode/AtCoder-style solutions that just read input, compute, and
    print — in practice this is most of this dataset's solutions). Rather
    than return a sentinel for those, wrap the whole module body in a
    synthetic function so cc_visit has something to walk, and measure the
    program's actual branching complexity as a single unit.

    This only changes the *parsed structure* radon sees (one extra level of
    indentation) — it doesn't execute the code, so it's safe even for code
    containing multi-line string literals; any leading whitespace added
    inside a string's content has no effect on complexity.

    Returns -1 if the code cannot be parsed at all (real code, not the
    wrapped version). -2 is still possible for the rare case where even the
    wrapped code yields no measurable blocks (e.g. inconsistent
    tabs/spaces prevents wrapping from parsing).
    """
    try:
        blocks = cc_visit(code)
    except Exception:
        return -1

    if not blocks:
        try:
            wrapped = "def _synthetic_wrapper():\n" + textwrap.indent(code, "    ")
            blocks = cc_visit(wrapped)
        except Exception:
            blocks = []
        if not blocks:
            return -2

    return round(sum(b.complexity for b in blocks) / len(blocks), 2)


def compute_metrics(code: str | None) -> dict:
    """
    Compute static metrics (NCLOC, CCN) for a code block.
    Returns a dict with ncloc and cyclomatic_complexity.
    """
    if not code:
        return {"ncloc": -1, "cyclomatic_complexity": -1}
    return {
        "ncloc": compute_ncloc(code),
        "cyclomatic_complexity": compute_cyclomatic_complexity(code),
    }


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def compute_extractable_code(response: str | None) -> bool:
    """
    Return True if a Python code block can be extracted from the response.
    """
    if not response:
        return False
    return extract_code(response) is not None


# ---------------------------------------------------------------------------
# Test pass metrics
# ---------------------------------------------------------------------------

def compute_passes_tests(test_results: list[tuple]) -> bool | None:
    """
    Return True if ALL test cases passed, False if any failed.
    Each element of test_results is a tuple whose first element is a bool.
    Returns None when the list is empty.
    """
    if not test_results:
        return None
    return all(result[0] for result in test_results)


def compute_passes_same_tests(
    source_results: list[tuple],
    followup_results: list[tuple],
) -> bool | None:
    """
    Return True if the followup passes every test that the source passed.

    A followup that regresses on any individually-passing source test returns
    False, even if its overall pass count is the same or higher.
    Returns None when either list is empty or lengths differ.
    """
    if not source_results or not followup_results:
        return None
    if len(source_results) != len(followup_results):
        return None
    for (src_passed, *_), (flw_passed, *_) in zip(source_results, followup_results):
        if src_passed and not flw_passed:
            return False
    return True


# ---------------------------------------------------------------------------
# Token metrics
# ---------------------------------------------------------------------------

def compute_token_count(usage: dict | None) -> dict:
    """
    Extract token counts from a usage dict.
    Returns prompt, completion, total, and thinking token counts.
    All values are None when usage is not available.
    """
    if not usage:
        return {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "thinking_tokens": None,
        }
    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "thinking_tokens": usage.get("thinking_tokens"),
    }


def compute_token_efficiency(usage: dict | None) -> float | None:
    """
    Token efficiency = completion_tokens / thinking_tokens.

    Measures how many output tokens were produced per thinking/reasoning token
    spent. A higher value means more output relative to internal reasoning cost.
    Returns None when thinking tokens are unavailable or zero (non-reasoning
    models, or models that didn't use extended thinking on this call).
    """
    if not usage:
        return None
    thinking = usage.get("thinking_tokens")
    completion = usage.get("completion_tokens")
    if not thinking or not completion:
        return None
    return round(completion / thinking, 6)


# ---------------------------------------------------------------------------
# Repeat baseline / t-test violation detection
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def load_repeat_baseline() -> dict[str, dict[str, tuple[float, ...]]]:
    """
    Build {question_id: {metric_name: (values...)}} from the repeat-only
    runs listed in config.settings.REPEAT_BASELINE_DIRS. Each directory is
    scanned for repeat-*.json files; every file contributes one independent
    sample per question. Directories may cover disjoint question sets (here,
    split by difficulty: hard/medium/easy) — merged by question_id across
    all directories.

    ncloc/cyclomatic_complexity are computed fresh from each entry's response
    code (never stored anywhere). llm_call_time/total_tokens/token_efficiency
    are read directly since those already exist per-entry.

    code_sim is a special case: instead of one value per file, its baseline
    is the full list of pairwise CodeT5 similarities between every pair of
    extracted code samples for that question across all repeat files — i.e.
    "how similar do two independent generations of this question tend to
    be," which the source-vs-followup code_sim score gets tested against.

    Cached: computed once per process, since it re-extracts code, recomputes
    NCLOC/CC, and runs embeddings for every repeat file on first access —
    with three difficulty folders this can take a while the first time it's
    called; subsequent calls in the same process are free.
    """
    import json
    from pathlib import Path
    from config.settings import REPEAT_BASELINE_DIRS

    metric_keys = ("ncloc", "cyclomatic_complexity", "llm_call_time", "total_tokens", "token_efficiency")
    baseline: dict[str, dict[str, list[float]]] = {}
    codes_by_question: dict[str, list[str | None]] = {}

    for dir_path in REPEAT_BASELINE_DIRS:
        for file_path in sorted(Path(dir_path).glob("repeat-*.json")):
            try:
                with open(file_path, encoding="utf-8") as f:
                    entries = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(entries, list):
                continue

            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                qid = entry.get("question_id")
                if not qid:
                    continue

                q_metrics = baseline.setdefault(qid, {k: [] for k in metric_keys})
                codes_by_question.setdefault(qid, [])

                code = extract_code(entry.get("response") or "")
                codes_by_question[qid].append(code)

                code_metrics = compute_metrics(code)
                if code_metrics["ncloc"] not in (-1, -2):
                    q_metrics["ncloc"].append(code_metrics["ncloc"])
                if code_metrics["cyclomatic_complexity"] not in (-1, -2):
                    q_metrics["cyclomatic_complexity"].append(code_metrics["cyclomatic_complexity"])

                llm_call_time = entry.get("llm_call_time")
                if isinstance(llm_call_time, (int, float)):
                    q_metrics["llm_call_time"].append(llm_call_time)

                total_tokens = entry.get("total_tokens")
                if isinstance(total_tokens, (int, float)):
                    q_metrics["total_tokens"].append(total_tokens)

                token_efficiency = entry.get("token_efficiency")
                if isinstance(token_efficiency, (int, float)):
                    q_metrics["token_efficiency"].append(token_efficiency)

    result: dict[str, dict[str, tuple[float, ...]]] = {}
    for qid, metrics_dict in baseline.items():
        entry_out = {k: tuple(v) for k, v in metrics_dict.items()}
        entry_out["code_sim"] = tuple(compute_pairwise_similarity_list(codes_by_question.get(qid, [])))
        result[qid] = entry_out

    return result


def compute_ttest_violation(
    baseline_samples: list[float],
    followup_value,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> dict | None:
    """
    One-sample t-test: is followup_value consistent with the per-question
    baseline distribution built from repeat (no-transformation) runs?

    ``alternative`` uses scipy's orientation, where the repeat baseline is
    the sample and ``followup_value`` is the hypothesised population mean:
    ``less`` detects a follow-up above the baseline, while ``greater``
    detects a follow-up below it.

    Returns None if untestable (fewer than 2 baseline samples, or a
    non-numeric followup value) rather than raising or false-flagging.
    """
    if not isinstance(followup_value, (int, float)) or isinstance(followup_value, bool):
        return None
    if len(baseline_samples) < 2:
        return None
    from scipy import stats as _stats
    try:
        t_stat, p_value = _stats.ttest_1samp(
            baseline_samples,
            popmean=followup_value,
            alternative=alternative,
        )
    except Exception:
        return None

    return {
        "n_baseline": len(baseline_samples),
        "baseline_mean": round(mean(baseline_samples), 4),
        "baseline_std": round(stdev(baseline_samples), 4) if len(baseline_samples) > 1 else 0.0,
        "followup": followup_value,
        "t_stat": round(float(t_stat), 4),
        "p_value": round(float(p_value), 6),
        "alternative": alternative,
        "significant": bool(p_value < alpha),
    }
