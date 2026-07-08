import random

def _pick(variants: list[str]) -> str:
    return random.choice(variants)


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
ANTONYM_PROMPT_DIFF = (
            "Rewrite the programming problem below by substituting key terms with their antonyms "
            "ensuring that the substitutions make sense in context"
            "All numeric constraints, input/output formats, and the required answer must stay identical. "
            "Output only the rewritten problem text with no preamble."
)
ANTONYM_PROMPT_EQUIV = (
    "Rewrite the programming problem below by substituting key terms with their antonyms "
    "ensuring that the substitutions make sense in context"
    "and adjusting the surrounding logic so the problem remains semantically equivalent "
    "(e.g. 'ascending order' → 'not in descending order'). "
    "All numeric constraints, input/output formats, and the required answer must stay identical. "
    "Output only the rewritten problem text with no preamble."
)
SYNONYM_PROMPT = (
    "Rewrite the programming problem below by substituting key domain terms and verbs "
    "ensuring that the substitutions make sense in context"
    "Preserve all numeric constraints, input/output formats, and the exact meaning. "
    "Do not add or remove any requirements. "
    "Output only the rewritten problem text with no preamble."
)
LOOP_TYPE_CONSTRAINT = (
    "You must use only while loops for all iteration. Do not use for loops."
)
FULL_COMMENTING_CONSTRAINT = (
    "Add a concise inline comment to every line of your solution explaining what that line does. "
    "Every non-blank line of code must have a comment."
)
DUMMY_VARIABLE_CONSTRAINT = (
    "At the start of your solution, declare a variable named _unused_flag and set it to True. "
    "Declare a variable named unused_itteration and set it to 0."
    "Do not use it these variables anywhere else in your solution."
)
GUARD_CLAUSE_CONSTRAINT = (
    "All conditional logic must be expressed as guard clauses (early returns) rather than nested if-else statements. "
    "Do not use nested if-else statements."
)
RECURSION_CONSTRAINT = (
    "You must implement your solution using recursion. "
    "Do not use any iterative loops (for or while)."
)



