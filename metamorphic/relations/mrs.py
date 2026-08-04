import re, random
from metamorphic.relations.base_mr import BaseMR
from metamorphic.relations import phrases
from metamorphic.relations.phrases import _pick
from llms.mr_helper import mr_llm_transform
from evaluation.executor import extract_code


def _insert_after_first_line(prompt: str, line: str) -> str:
    lines = prompt.split("\n")
    if lines and lines[0] != "":
        lines.insert(1, line)
        return "\n".join(lines)
    if lines and lines[0] == "":
        lines.insert(0, line)
        return "\n".join(lines)
    return line


def _prepend_line(prompt: str, line: str) -> str:
    if not prompt:
        return line
    return f"{line}\n{prompt}"


def _strip_list_prefix(text: str) -> str:
    if text.startswith("-") or text.startswith("*"):
        return text.lstrip("-* ").strip()
    match = re.match(r"^\d+[\.)]\s*(.*)$", text)
    if match:
        return match.group(1)
    return text


def _to_bulleted(prompt: str) -> str:
    lines = prompt.split("\n")
    out = []
    in_code = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            out.append(line)
            continue
        if in_code or stripped == "":
            out.append(line)
            continue
        out.append(f"- {_strip_list_prefix(stripped)}")
    return "Structured prompt:\n" + "\n".join(out)

def _remove_random_sentence(text: str) -> str:
    """Split on sentence boundaries and delete one at random, preserving original spacing."""
    sentences = re.findall(r"[^.!?]*[.!?]+\s*|[^.!?]+$", text)
    candidates = [i for i, s in enumerate(sentences) if s.strip()]
    if candidates:
        del sentences[random.choice(candidates)]
    return "".join(sentences)

def _clean_library_name(raw: str) -> str:
    name = raw.strip().splitlines()[0].strip()
    name = name.strip("`\"'. ")
    if not name or len(name) > 40:
        name = "a suitable standard library module"
    return name

class MR0(BaseMR):
    target = "PD"

    @property
    def name(self) -> str:
        return "do_nothing"

    def transform(self, text: str) -> str:
        return text


class MR1(BaseMR):
    target = "PD"

    @property
    def name(self) -> str:
        return "irrelevant_context"

    def transform(self, text: str) -> str:
        return _insert_after_first_line(text, _pick(phrases.IRRELEVANT_CONTEXT))


class MR2(BaseMR):
    target = "I"

    @property
    def name(self) -> str:
        return "constraint_redundant_spec"

    def transform(self, prefix: str, suffix: str) -> tuple[str, str]:
        return _insert_after_first_line(prefix, _pick(phrases.REDUNDANT_CONTEXT)), suffix


class MR3(BaseMR):
    target = "PD"

    @property
    def name(self) -> str:
        return "format_structural_equivalence"

    def transform(self, text: str) -> str:
        return _to_bulleted(text)


class MR4(BaseMR):
    target = "I"

    @property
    def name(self) -> str:
        return "persona_role_prompting"

    def transform(self, prefix: str, suffix: str) -> tuple[str, str]:
        return _prepend_line(prefix, _pick(phrases.PERSONAS)), suffix


class MR5(BaseMR):
    target = "PD"

    @property
    def name(self) -> str:
        return "polarity_negative_formulation"

    def transform(self, text: str) -> str:
        return mr_llm_transform(phrases.NEGATION_PROMPT, text)


class MR6(BaseMR):
    target = "I"

    @property
    def name(self) -> str:
        return "identifier_explicit_naming"

    def transform(self, prefix: str, suffix: str) -> tuple[str, str]:
        return (_insert_after_first_line(prefix, phrases.make_identifier_constraint(phrases.WORDS)), suffix)


class MR7(BaseMR):
    target = "PD"

    @property
    def name(self) -> str:
        return "order_requirement_permutation"

    def transform(self, text: str) -> str:
        return mr_llm_transform(phrases.PERMUTATION_PROMPT, text)


class MR8(BaseMR):
    target = "PD"

    @property
    def name(self) -> str:
        return "remove_sentences"

    def transform(self, text: str) -> str:
        return _remove_random_sentence(text)


class MR9(BaseMR):
    target = "PD"

    @property
    def name(self) -> str:
        return "antonym_substitution"

    def transform(self, text: str) -> str:
        return mr_llm_transform(phrases.ANTONYM_PROMPT_DIFF, text)


class MR10(BaseMR):
    target = "PD"

    @property
    def name(self) -> str:
        return "synonym_substitution"

    def transform(self, text: str) -> str:
        return mr_llm_transform(phrases.SYNONYM_PROMPT, text)


class MR11(BaseMR):
    target = "I"

    @property
    def name(self) -> str:
        return "loop_while_only"

    def transform(self, prefix: str, suffix: str) -> tuple[str, str]:
        return _insert_after_first_line(prefix, _pick(phrases.LOOP_TYPE_CONSTRAINT)), suffix


class MR12(BaseMR):
    target = "I"

    @property
    def name(self) -> str:
        return "full_commenting"

    def transform(self, prefix: str, suffix: str) -> tuple[str, str]:
        return _insert_after_first_line(prefix, phrases.FULL_COMMENTING_CONSTRAINT), suffix


class MR13(BaseMR):
    target = "I"

    @property
    def name(self) -> str:
        return "dummy_variable"

    def transform(self, prefix: str, suffix: str) -> tuple[str, str]:
        return _insert_after_first_line(prefix, _pick(phrases.DUMMY_VARIABLE_CONSTRAINT)), suffix


class MR14(BaseMR):
    target = "I"

    @property
    def name(self) -> str:
        return "guard_clause_conditionals"

    def transform(self, prefix: str, suffix: str) -> tuple[str, str]:
        return _insert_after_first_line(prefix, phrases.GUARD_CLAUSE_CONSTRAINT), suffix


class MR15(BaseMR):
    target = "I"

    @property
    def name(self) -> str:
        return "use_recursion"

    def transform(self, prefix: str, suffix: str) -> tuple[str, str]:
        return _insert_after_first_line(prefix, phrases.RECURSION_CONSTRAINT), suffix


class MR16(BaseMR):
    target = "I"

    @property
    def name(self) -> str:
        return "no_builtin_libraries"

    def transform(self, prefix: str, suffix: str) -> tuple[str, str]:
        return _insert_after_first_line(prefix, phrases.NO_BUILTIN_LIBS), suffix


class MR17(BaseMR):
    """LLM-assisted: ask the helper which standard-library module fits this
    problem, then require the model under test to use it."""
    target = "PD"

    @property
    def name(self) -> str:
        return "mandated_library"

    def transform(self, text: str) -> str:
        suggestion = mr_llm_transform(phrases.LIBRARY_SUGGESTION_PROMPT, text)
        library = _clean_library_name(suggestion)
        constraint = f"You must use Python's {library} in your solution."
        print(f"  [mr17] helper suggested: {library}")
        return _insert_after_first_line(text, constraint)

class MR18(BaseMR):
    target = "PD"

    @property
    def name(self) -> str:
        return "scope_input_narrowing"

    def transform(self, text: str) -> str:
        return _insert_after_first_line(text, _pick(phrases.SCOPE_NARROWING))


class MR19(BaseMR):
    """LLM-assisted: have the helper solve the problem in one call, then, in a
    separate call with no access to the original problem text, describe only
    the resulting code. That description becomes the new problem statement."""
    target = "PD"

    @property
    def name(self) -> str:
        return "round_trip_paraphrase"

    def transform(self, text: str) -> str:
        solve_response = mr_llm_transform(phrases.ROUND_TRIP_SOLVE_PROMPT, text)
        code = extract_code(solve_response)
        if not code:
            # Helper failed to produce a fenced block — fall back to the raw
            # response so the MR doesn't silently return an empty prompt.
            code = solve_response
        explanation = mr_llm_transform(phrases.ROUND_TRIP_EXPLAIN_PROMPT, code)
        return explanation