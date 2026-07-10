import random

def _pick(variants: list[str]) -> str:
    return random.choice(variants)

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
    "m2p"
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
    "The silent astronaut carefully navigated her lunar rover through the crystal caverns."
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
    "Definitions: iterate means loop through each element; concatenate means append one string to the end of another."
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
    "You are a research engineer implementing an algorithm straight from an academic paper, prioritizing correctness over polish."
]
IDENTIFIER_CONSTRAINT = "Use " + _pick(WORDS) + " as your loop variable and " + _pick(WORDS) + " as your output list."


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



