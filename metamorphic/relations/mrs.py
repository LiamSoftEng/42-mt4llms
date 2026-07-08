import re
from metamorphic.relations.base_mr import BaseMR
from metamorphic.relations import phrases as ph
from metamorphic.relations.phrases import _pick
from llms.mr_helper import mr_llm_transform


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


def _is_list_line(text: str) -> bool:
    if text.startswith("-") or text.startswith("*"):
        return True
    return re.match(r"^\d+[\.)]\s+", text) is not None


def _permute_constraints(prompt: str) -> str:
    lines = prompt.split("\n")
    out = []
    buffer = []
    in_code = False

    def flush_buffer() -> None:
        if buffer:
            out.extend(reversed(buffer))
            buffer.clear()

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            flush_buffer()
            in_code = not in_code
            out.append(line)
            continue
        if in_code:
            flush_buffer()
            out.append(line)
            continue
        if _is_list_line(stripped):
            buffer.append(line)
        else:
            flush_buffer()
            out.append(line)

    flush_buffer()
    return "\n".join(out)


def _rewrite_return_to_negative(prompt: str) -> str:
    lines = prompt.split("\n")
    for i, line in enumerate(lines):
        stripped = line.strip()
        match = re.match(r"^(Return|return)\s+(.*)$", stripped)
        if match:
            rest = match.group(2).rstrip(".")
            prefix_len = len(line) - len(line.lstrip(" "))
            prefix = line[:prefix_len]
            lines[i] = f"{prefix}Do not return anything other than {rest}."
            return "\n".join(lines)
    return _insert_after_first_line(prompt, _pick(ph.NEGATIVE_CONSTRAINT))


#indexes second to last non-blank line, and removes it, as earlier sentences are often too important to remove
def _remove_sentences(text: str) -> str:
    """Remove the second-to-last line from the prompt, skipping past any blank lines."""
    lines = text.split("\n")
    idx = len(lines) - 2
    while idx >= 0 and not lines[idx].strip():
        idx -= 1
    if idx >= 0:
        del lines[idx]
    return "\n".join(lines)


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
        return "dead_context"

    def transform(self, text: str) -> str:
        return _insert_after_first_line(text, _pick(ph.NOISE))


class MR2(BaseMR):
    target = "I"

    @property
    def name(self) -> str:
        return "constraint_redundant_spec"

    def transform(self, prefix: str, suffix: str) -> tuple[str, str]:
        return _insert_after_first_line(prefix, _pick(ph.REDUNDANT_SPEC)), suffix


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
        return _prepend_line(prefix, _pick(ph.PERSONA)), suffix


class MR5(BaseMR):
    target = "I"

    @property
    def name(self) -> str:
        return "polarity_negative_formulation"

    def transform(self, prefix: str, suffix: str) -> tuple[str, str]:
        return _rewrite_return_to_negative(prefix), suffix


class MR6(BaseMR):
    target = "I"

    @property
    def name(self) -> str:
        return "identifier_explicit_naming"

    def transform(self, prefix: str, suffix: str) -> tuple[str, str]:
        return _insert_after_first_line(prefix, _pick(ph.IDENTIFIER_CONSTRAINT)), suffix


class MR7(BaseMR):
    target = "I"

    @property
    def name(self) -> str:
        return "hint_trivial_algorithm"

    def transform(self, prefix: str, suffix: str) -> tuple[str, str]:
        return _insert_after_first_line(prefix, _pick(ph.TRIVIAL_HINT)), suffix


class MR8(BaseMR):
    target = "PD"

    @property
    def name(self) -> str:
        return "order_requirement_permutation"

    def transform(self, text: str) -> str:
        return _permute_constraints(text)


# New ones
class MR9(BaseMR):
    target = "PD"

    @property
    def name(self) -> str:
        return "remove_sentences"

    def transform(self, text: str) -> str:
        return _remove_sentences(text)


class MR10(BaseMR):
    target = "PD"

    @property
    def name(self) -> str:
        return "antonym_substitution"

    def transform(self, text: str) -> str:
        return mr_llm_transform(_pick(ph.ANTONYM_PROMPT_DIFF), text)


class MR11(BaseMR):
    target = "PD"

    @property
    def name(self) -> str:
        return "synonym_substitution"

    def transform(self, text: str) -> str:
        return mr_llm_transform(_pick(ph.SYNONYM_PROMPT), text)


class MR12(BaseMR):
    target = "I"

    @property
    def name(self) -> str:
        return "loop_while_only"

    def transform(self, prefix: str, suffix: str) -> tuple[str, str]:
        return _insert_after_first_line(prefix, _pick(ph.LOOP_TYPE_CONSTRAINT)), suffix


class MR13(BaseMR):
    target = "I"

    @property
    def name(self) -> str:
        return "full_commenting"

    def transform(self, prefix: str, suffix: str) -> tuple[str, str]:
        return _insert_after_first_line(prefix, _pick(ph.FULL_COMMENTING_CONSTRAINT)), suffix


class MR14(BaseMR):
    target = "I"

    @property
    def name(self) -> str:
        return "dummy_variable"

    def transform(self, prefix: str, suffix: str) -> tuple[str, str]:
        return _insert_after_first_line(prefix, _pick(ph.DUMMY_VARIABLE_CONSTRAINT)), suffix


class MR15(BaseMR):
    target = "I"

    @property
    def name(self) -> str:
        return "guard_clause_conditionals"

    def transform(self, prefix: str, suffix: str) -> tuple[str, str]:
        return _insert_after_first_line(prefix, _pick(ph.GUARD_CLAUSE_CONSTRAINT)), suffix


class MR16(BaseMR):
    target = "I"

    @property
    def name(self) -> str:
        return "use_recursion"

    def transform(self, prefix: str, suffix: str) -> tuple[str, str]:
        return _insert_after_first_line(prefix, _pick(ph.RECURSION_CONSTRAINT)), suffix