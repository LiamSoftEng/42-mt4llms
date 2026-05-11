RELATIONS = {
    0: {
        "module": "metamorphic.relations.mr0",
        "class": "MR0",
        "name": "do_nothing",
        "description": "Control relation that returns the original prompt and output unchanged.",
    },
    1: {
        "module": "metamorphic.relations.mr1",
        "class": "MR1",
        "name": "dead_context",
        "description": "Insert irrelevant sentence into prompt. Output should be identical.",
    },
}