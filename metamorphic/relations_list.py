RELATIONS = {
    0: {
        "module": "metamorphic.relations.mr0",
        "class": "MR0",
        "name": "do_nothing",
        "description": "Control relation that returns the original prompt and output unchanged.",
    },
    1: {
        "module": "metamorphic.relations.mrs",
        "class": "MR1",
        "name": "dead_context",
        "description": "Insert irrelevant sentence into prompt. Output should be identical.",
    },
    2: {
        "module": "metamorphic.relations.mrs",
        "class": "MR2",
        "name": "constraint_redundant_spec",
        "description": "Inject a redundant language constraint already implied by syntax.",
    },
    3: {
        "module": "metamorphic.relations.mrs",
        "class": "MR3",
        "name": "format_structural_equivalence",
        "description": "Convert the prompt into a structured bulleted list.",
    },
    4: {
        "module": "metamorphic.relations.mrs",
        "class": "MR4",
        "name": "persona_role_prompting",
        "description": "Prepend a persona to the prompt.",
    },
    5: {
        "module": "metamorphic.relations.mrs",
        "class": "MR5",
        "name": "context_extraneous_narrative",
        "description": "Inject an irrelevant backstory without changing requirements.",
    },
    6: {
        "module": "metamorphic.relations.mrs",
        "class": "MR6",
        "name": "polarity_negative_formulation",
        "description": "Rewrite a return constraint into an equivalent negative form.",
    },
    7: {
        "module": "metamorphic.relations.mrs",
        "class": "MR7",
        "name": "identifier_explicit_naming",
        "description": "Add explicit internal naming constraints for variables.",
    },
    8: {
        "module": "metamorphic.relations.mrs",
        "class": "MR8",
        "name": "output_formatting_constraint",
        "description": "Add constraints about outputting raw code only.",
    },
    9: {
        "module": "metamorphic.relations.mrs",
        "class": "MR9",
        "name": "hint_trivial_algorithm",
        "description": "Add a trivial algorithmic suggestion.",
    },
    10: {
        "module": "metamorphic.relations.mrs",
        "class": "MR10",
        "name": "order_requirement_permutation",
        "description": "Permute the order of independent constraints.",
    },
    11: {
        "module": "metamorphic.relations.mrs",
        "class": "MR11",
        "name": "verbosity_lexical_expansion",
        "description": "Add brief definitions for common programming terms.",
    },
}