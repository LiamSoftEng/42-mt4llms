import random


def _pick(variants: list[str]) -> str:
    return random.choice(variants)


def make_identifier_constraint(words: list[str]) -> str:
    var1, var2 = random.sample(words, 2)
    template = _pick(IDENTIFIER_CONSTRAINT)
    return template.format(variable1=var1, variable2=var2)


WORDS = [
    "whisper",
    "tmp",
    "garden",
    "foo",
    "copper",
    "val2",
    "q7z",
    "thunder",
    "a1",
    "ribbon",
    "cnt2",
    "marble",
    "do_thing",
    "velvet",
    "m2p",
]
IRRELEVANT_CONTEXT = [
    "Fun fact: honey never spoils if stored properly.",
    "The Great Barrier Reef is visible from space.",
    "This question was inspired by a conversation overheard at a coffee shop.",
    "Octopuses have three hearts and blue blood.",
    "This scenario is loosely based on an old folk tale from Eastern Europe.",
    "The Eiffel Tower can grow taller in summer due to heat expansion.",
    "A similar version of this puzzle once appeared in a math olympiad.",
    "Bananas are technically classified as berries.",
    "This example was adapted from a textbook published in the 1990s.",
    "The average adult has around 60,000 miles of blood vessels in their body.",
    "This dataset was collected across 12 different research sites.",
    "Researchers have long debated the optimal way to teach coding to beginners.",
    "A similar approach is sometimes used in supply chain optimization problems.",
    "Many students find this next part surprisingly tricky.",
    "The silent astronaut carefully navigated her lunar rover through the crystal caverns.",
]
REDUNDANT_CONTEXT = [
    "Note: This problem was originally presented in an international programming contest.",
    "Follow the programming language's required syntax rules (e.g., indentation for scope in Python).",
    "Make sure your code compiles or runs without syntax errors before submitting.",
    "Use meaningful variable names to improve code readability.",
    "Remember that most languages are case sensitive when it comes to variable and function names.",
    "It's good practice to test your solution on multiple inputs before finalizing it.",
    "Standard input and output methods should be used unless otherwise specified.",
    "Code style and formatting conventions may vary by language but don't change the underlying logic.",
    "Efficient solutions are appreciated, but correctness is the primary requirement.",
    "Off-by-one errors are a common source of bugs, so double check loop boundaries.",
    "Version control is recommended for tracking changes, though it has no bearing on the output.",
    "Do not return anything other than what the specification requires.",
    "Some languages require explicit type declarations, while others infer types automatically.",
    "Whitespace within expressions is generally ignored by most compilers and interpreters.",
    "Definitions: iterate means loop through each element; concatenate means append one string to the end of another.",
]
PERSONAS = [
    "You are an expert senior software engineer writing production-grade code.",
    "You are a competitive programmer optimizing for speed and efficiency under contest constraints.",
    "You are a meticulous code reviewer who catches every edge case and bug.",
    "You are a computer science professor writing algorithms to first-year students.",
    "You are a security-focused engineer auditing code for vulnerabilities.",
    "You are a systems programmer working close to the hardware, optimizing for memory and performance.",
    "You are a junior developer just learning best practices, careful and methodical in your approach.",
    "You are a technical interviewer evaluating a candidate's problem-solving process step by step.",
    "You are an open-source maintainer who values clean, well-documented, idiomatic code.",
    "You are a data scientist writing code to prototype and validate an algorithmic idea quickly.",
    "You are a full-stack developer balancing readability with performance across the entire code.",
    "You are a compiler engineer who thinks in terms of parsing, syntax trees, and formal language rules.",
    "You are a DevOps engineer writing scripts that must be robust and reliable in automated pipelines.",
    "You are a research engineer implementing an algorithm straight from an academic paper, prioritizing correctness over polish.",
]

NEGATION_PROMPT = (
    "Rewrite only the return/output specification using a logically equivalent negative "
    "formulation. Before rewriting, verify that the rewritten sentence is true if and only "
    "if the original sentence is true. Do not modify any other part of the "
    "problem. Output only the entire rewritten problem text."
)

IDENTIFIER_CONSTRAINT = [
    "Use {variable1} as your loop variable and {variable2} as your output list.",
    "Your loop variable must be named {variable1} and your result variable must be named {variable2}.",
    "It is mandatory to name the counter variable {variable1} and the output variable {variable2}.",
    "Name your helper variable {variable1} and its return value {variable2}.",
    "Call your temporary variable {variable1} and your final answer {variable2}.",
    "The input variable should be named {variable1}; the output variable should be named {variable2}.",
    "Set the flag variable to {variable1} and the buffer variable to {variable2}.",
    "Your accumulator must be named {variable1}, and your index variable must be named {variable2}.",
    "Name your first value {variable1} and your second value {variable2}.",
    "Call your main variable {variable1} and your supporting variable {variable2}.",
    "The primary variable should be called {variable1}; the secondary variable should be called {variable2}.",
    "Set the string variable to {variable1} and the secondary variable to {variable2}.",
    "Strictly use {variable1} for the sum variable and {variable2} for the average variable, with no exceptions.",
    "Your storage variable must be called {variable1}, and your key variable must be called {variable2}.",
    "Name the function {variable1} and its base variable {variable2}.",
    ]

PERMUTATION_PROMPT = (
    "Rewrite the programming problem below by reordering independent constraints, requirements, "
    "or setup sentences into a different but logically equivalent order. Do not change, remove, "
    "add, or reword any requirement, constraint, or detail — only reorder sentences or clauses "
    "that do not depend on one another. "
    "Output only the rewritten problem text with no preamble."
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
    "You must use only while loops for all iteration. Do not use for loops.",
    "You must use only for loops for all iteration. Do not use while loops.",
    "You must use nested loops to solve this problem. A single loop is not allowed.",
    "You must use only a single loop with no nesting. Nested loops are not allowed.",
)

FULL_COMMENTING_CONSTRAINT = (
    "Add a concise inline comment to every line of your solution explaining what that line does. "
    "Every non-blank line of code must have a comment."
)

DUMMY_VARIABLE_CONSTRAINT = (
    "At the start of your solution, declare a variable named _unused_flag and set it to True. Declare a variable named unused_iteration and set it to 0. Do not use these variables anywhere else in your solution.",
    "Before writing any logic, declare a variable named _dummy_check and set it to False. Declare a variable named _placeholder_count and set it to 0. Do not use these variables anywhere else in your code.",
    "At the top of your solution, add a variable _skip_marker set to None and a variable _reserved_slot set to -1. Do not reference either variable again.",
    "Declare two unused variables at the beginning of your solution: _idle_state set to True and _padding_value set to 0. Neither should appear anywhere else in the code.",
    "As the first lines of your solution, initialize _noop_flag to False and _spare_index to 0. These declarations must remain unused for the rest of your code.",
    "Your solution must begin with the declaration of _sentinel_flag (set to True) and _extra_counter (set to 1). Do not use these variables in any subsequent logic.",
    "Include _placeholder_bool set to False and _filler_number set to 99 at the start of your code. These must not be referenced or used anywhere else.",
    "At the beginning of your solution, create a variable _stub_flag with the value True and a variable _reserve_count with the value 0. Leave both unused for the remainder of the solution.",
    "First, declare _inactive_marker as None and _buffer_index as -1. Do not use either variable elsewhere in your solution.",
    "Add two dummy declarations at the start of your solution: _unused_toggle = False and _slack_variable = 0. These variables should not be used anywhere else.",
    "At the very beginning of your code, declare _dormant_flag and set it to True, and declare _padding_counter and set it to 5. Do not use these anywhere else in your solution.",
    "Your code must start with two unused variables: _quiet_flag set to False and _phantom_index set to -1. Neither variable should be used again.",
    "Declare _idle_marker set to None and _spare_counter set to 0 before any other code. These must remain unreferenced for the rest of the solution.",
    "At the top of your solution, initialize _hollow_flag to True and _extra_slot to 1. Do not use these variables anywhere else in your code.",
    "Begin your solution by declaring _void_flag set to False and _reserve_index set to 0. These variables must not be used anywhere else in the solution.",
)

GUARD_CLAUSE_CONSTRAINT = (
    "All conditional logic must be expressed as guard clauses (early returns) rather than nested if-else statements. "
    "Do not use nested if-else statements."
)

RECURSION_CONSTRAINT = (
    "You must implement your solution using recursion. "
    "Do not use any iterative loops (for or while)."
)

NO_BUILTIN_LIBS = (
    "You must not import or use any standard library modules; implement everything from scratch using only core language syntax."
)

LIBRARY_SUGGESTION_PROMPT = (
    "Given the following programming problem, name exactly one Python standard-library module or "
    "built-in function that would be genuinely useful for solving it efficiently (for example: itertools, "
    "collections, bisect, functools, heapq, math, re). "
    "Respond with only the name, nothing else \u2014 no explanation, no punctuation, no surrounding text."
)

SCOPE_NARROWING = [
    "Additional guarantee: none of the input values will ever be zero.",
    "Additional guarantee: all numeric inputs are guaranteed to be strictly positive.",
    "Additional guarantee: the input size is always a power of two.",
    "Additional guarantee: the input contains no duplicate values.",
    "Additional guarantee: all inputs fit within the range of a 32-bit signed integer.",
    "Additional guarantee: the input is already sorted in non-decreasing order.",
]

ROUND_TRIP_PROMPT = (
    "Solve the following programming problem internally, without showing any code. "
    "Then write a clear, complete natural-language description of your solution approach, "
    "as if writing a detailed docstring or explanation for someone else to implement. "
    "Your description must fully specify the required input format, the required output format, "
    "and the logic needed to solve the problem, such that someone reading only your description, "
    "with no access to the original problem text, could write correct code from it alone. "
    "Output only that description, with no preamble and no code."
)