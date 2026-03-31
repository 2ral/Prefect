from modelizer.dependencies.fuzzingbook import convert_and_validate_ebnf_grammar, Grammar


DATEUTIL_GRAMMAR: Grammar = {
    "<start>": ["<iso8601datetime>"],

    "<iso8601datetime>": [
        "<iso8601date>",
        "<iso8601date>T<iso8601time>"
    ],

    "<iso8601date>": [
        "<iso8601calendardate>",
        "<iso8601weekdate>",
        "<iso8601ordinaldate>"
    ],

    "<iso8601calendardate>": [
        "<iso8601year>-<iso8601month>",
        "<iso8601year>-<iso8601month>-<iso8601day>",
        "<iso8601year><iso8601month><iso8601day>"
    ],

    "<iso8601year>": [
        "<sign><digit><digit><digit><digit>",
        "<digit><digit><digit><digit>"
    ],

    "<sign>": ["+", "-"],

    "<iso8601month>": [f"{i:02}" for i in range(1, 13)],

    "<iso8601day>": [f"{i:02}" for i in range(1, 32)],

    "<iso8601weekdate>": [
        "<iso8601year>W<iso8601week>",
        "<iso8601year>-W<iso8601week>",
        "<iso8601year>W<iso8601week>-<iso8601weekday>",
        "<iso8601year>-W<iso8601week>-<iso8601weekday>"
    ],

    "<iso8601week>": [f"{i:02}" for i in range(1, 54)],

    "<iso8601weekday>": [str(i) for i in range(1, 8)],

    "<iso8601ordinaldate>": [
        "<iso8601year>",
        "<iso8601year><iso8601ordinalday>",
        "<iso8601year>-<iso8601ordinalday>"
    ],

    "<iso8601ordinalday>": [f"{i:03}" for i in range(1, 367)],

    "<iso8601time>": [
        "<T><iso8601hour>",
        "<T><iso8601hour>:<iso8601minute>",
        "<T><iso8601hour>:<iso8601minute>:<iso8601second>",
        "<T><iso8601hour>:<iso8601minute>:<iso8601second><fractionopt>",
        "<T><iso8601hour>:<iso8601minute>:<iso8601second><fractionopt><iso8601timezone>",
        "<T><iso8601hour>:<iso8601minute>:<iso8601second><iso8601timezone>"
    ],

    "<T>": ["", "T"],

    # 24:00:00 is allowed to represent midnight at the end of a day
    "<iso8601hour>": [f"{i:02}" for i in range(0, 25)],

    "<iso8601minute>": [f"{i:02}" for i in range(0, 60)],

    # xx:yy:60 is allowed to represent leap seconds
    "<iso8601second>": [f"{i:02}" for i in range(0, 61)],

    "<fractionopt>": [
        "",
        ".<iso8601fraction>",
        ",<iso8601fraction>"
    ],

    "<iso8601fraction>": ["<digit>", "<digit><digit>", "<digit><digit><digit>"],

    "<iso8601timezone>": [
        "",
        "Z",
        "+<iso8601hour>",
        "+<iso8601hour>:<iso8601minute>",
        "-<iso8601hour>",
        "-<iso8601hour>:<iso8601minute>"
    ],

    "<digit>": [str(i) for i in range(10)]
}

DATEUTIL_GRAMMAR = convert_and_validate_ebnf_grammar(DATEUTIL_GRAMMAR)
