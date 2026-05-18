import re

NOISE = "Note: This problem was originally presented in an international programming contest."
REDUNDANT_SPEC = "Follow the programming language's required syntax rules (e.g., indentation for scope in Python)."
PERSONA = "You are an expert senior software engineer writing production-grade code."
BACKSTORY = "Backstory: Alice is building a database for her spaceship, but this story is irrelevant to the task."
IDENTIFIER_CONSTRAINT = "Use currentIndex as your loop variable and final_output as your output list."
OUTPUT_FORMAT = "Provide only the raw code with no markdown formatting or explanations."
TRIVIAL_HINT = "A straightforward loop-based solution is acceptable."
NEGATIVE_CONSTRAINT = "Do not return anything other than what the specification requires."
VERBOSITY_NOTE = (
    "Definitions: iterate means loop through each element; concatenate means append one string to the end of another."
)


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


def _append_line(prompt: str, line: str) -> str:
    if not prompt:
        return line
    return f"{prompt}\n{line}"


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
    return _insert_after_first_line(prompt, NEGATIVE_CONSTRAINT)


class MR1:

    @property
    def name(self) -> str:
        return "dead_context"

    def transform(self, prompt: str) -> str:
        """Insert an irrelevant sentence after the first line of the prompt."""
        lines = prompt.split("\n")
        lines.insert(1, NOISE)
        return "\n".join(lines)

    def verify(self, source_passed: bool, followup_passed: bool) -> bool:
        return source_passed == followup_passed


class MR2:

    @property
    def name(self) -> str:
        return "constraint_redundant_spec"

    def transform(self, prompt: str) -> str:
        return _insert_after_first_line(prompt, REDUNDANT_SPEC)

    def verify(self, source_passed: bool, followup_passed: bool) -> bool:
        return source_passed == followup_passed


class MR3:

    @property
    def name(self) -> str:
        return "format_structural_equivalence"

    def transform(self, prompt: str) -> str:
        return _to_bulleted(prompt)

    def verify(self, source_passed: bool, followup_passed: bool) -> bool:
        return source_passed == followup_passed


class MR4:

    @property
    def name(self) -> str:
        return "persona_role_prompting"

    def transform(self, prompt: str) -> str:
        return _prepend_line(prompt, PERSONA)

    def verify(self, source_passed: bool, followup_passed: bool) -> bool:
        return source_passed == followup_passed


class MR5:

    @property
    def name(self) -> str:
        return "context_extraneous_narrative"

    def transform(self, prompt: str) -> str:
        return _insert_after_first_line(prompt, BACKSTORY)

    def verify(self, source_passed: bool, followup_passed: bool) -> bool:
        return source_passed == followup_passed


class MR6:

    @property
    def name(self) -> str:
        return "polarity_negative_formulation"

    def transform(self, prompt: str) -> str:
        return _rewrite_return_to_negative(prompt)

    def verify(self, source_passed: bool, followup_passed: bool) -> bool:
        return source_passed == followup_passed


class MR7:

    @property
    def name(self) -> str:
        return "identifier_explicit_naming"

    def transform(self, prompt: str) -> str:
        return _insert_after_first_line(prompt, IDENTIFIER_CONSTRAINT)

    def verify(self, source_passed: bool, followup_passed: bool) -> bool:
        return source_passed == followup_passed


class MR8:

    @property
    def name(self) -> str:
        return "output_formatting_constraint"

    def transform(self, prompt: str) -> str:
        return _insert_after_first_line(prompt, OUTPUT_FORMAT)

    def verify(self, source_passed: bool, followup_passed: bool) -> bool:
        return source_passed == followup_passed


class MR9:

    @property
    def name(self) -> str:
        return "hint_trivial_algorithm"

    def transform(self, prompt: str) -> str:
        return _insert_after_first_line(prompt, TRIVIAL_HINT)

    def verify(self, source_passed: bool, followup_passed: bool) -> bool:
        return source_passed == followup_passed


class MR10:

    @property
    def name(self) -> str:
        return "order_requirement_permutation"

    def transform(self, prompt: str) -> str:
        return _permute_constraints(prompt)

    def verify(self, source_passed: bool, followup_passed: bool) -> bool:
        return source_passed == followup_passed


class MR11:

    @property
    def name(self) -> str:
        return "verbosity_lexical_expansion"

    def transform(self, prompt: str) -> str:
        return _append_line(prompt, VERBOSITY_NOTE)

    def verify(self, source_passed: bool, followup_passed: bool) -> bool:
        return source_passed == followup_passed