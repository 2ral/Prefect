from modelizer.dependencies.fuzzingbook import convert_and_validate_ebnf_grammar, Grammar


RE2_GRAMMAR: Grammar = {
    "<start>": ["<regex>"],

    "<regex>": ["<alternation>"],

    "<alternation>": ["<concatenation>", "<concatenation>|<alternation>"],

    "<concatenation>": ["<repetition>", "<repetition><concatenation>"],

    "<repetition>": [
        "<atom>",
        "<atom><rep_symbols>",
        "<atom><rep_range>",
    ],

    "<rep_range>": [
        "<l_brack><repeat_range><r_brack>",
        "<l_brack><repeat_min><comma><r_brack>",
        "<l_brack><repeat_min_max><r_brack>",

        "<l_brack><repeat_range><r_brack><questionmark>",
        "<l_brack><repeat_min><comma><r_brack><questionmark>",
        "<l_brack><repeat_min_max><r_brack><questionmark>"],

    "<rep_symbols>": ["*", "+", "?", "*?", "??"],

    "<l_brack>": ["{"],
    "<r_brack>": ["}"],
    "<comma>": [","],
    "<questionmark>": ["?"],

    "<repeat_range>": ["<repeat_min>", "<repeat_min_max>"],
    "<repeat_min>": ["<number>"],
    "<repeat_min_max>": ["0,5", "0,7", "3,6", "4, 10"],
    "<number>": [str(c) for c in list(range(0, 25))],
    # "<number>": ["<digit>", "<digit><digit>"],
    # "<digit>": list("0123456789"),

    "<atom>": [
        "<group>",
        "<char_class>",
        "<escaped_char>",
        "<literal_char>",
    ],

    "<group>": ["(<regex>)"],

    "<char_class>": [
        "[<char_class_items>]",
        "[^<char_class_items>]",
        "<posix_class>",
        "<negated_posix_class>",
        "<unicode_class>",
        "<negated_unicode_class>"
    ],

    # "<char_class_items>": [
    #     "<char_class_item>",
    #     "<char_class_item><char_class_items>"
    # ],

    "<char_class_items>": [
        "<char_class_item>",
    ],

    "<char_class_item>": [
        "<char_range>",
        "<char_class_char>"
    ],

    "<char_range>": ["A-Z", "a-z", "0-Z", "0-9"],

    "<char_class_char>": [
        "<escaped_char>",
        "<literal_char>"
    ],

    "<posix_class>": ["[[:<posix_class_name>:]]"],
    "<negated_posix_class>": ["[[:^<posix_class_name>:]]"],
    "<posix_class_name>": [
        "alnum", "alpha", "ascii", "blank", "cntrl", "digit",
        "graph", "lower", "print", "punct", "space", "upper",
        "word", "xdigit"
    ],

    "<unicode_class>": ["p{<unicode_class_name>}"],
    "<negated_unicode_class>": ["P{<unicode_class_name>}"],
    "<unicode_class_name>": [
        "L", "Ll", "Lu", "Lt", "Lm", "Lo",
        "M", "Mn", "Mc", "Me",
        "N", "Nd", "Nl", "No",
        "P", "Pc", "Pd", "Ps", "Pe", "Pi", "Pf", "Po",
        "S", "Sm", "Sc", "Sk",  # "So",      # Special symbols, get rid of So (mostly emoticons)
        # "Z", "Zs", "Zl", "Zp",             # maybe get rid of Z classes
        # "C", "Cc", "Cf", "Cs", "Co", "Cn"  # maybe get rid of C clases (\n, \t, surrogate, ...)
    ],

    "<escaped_char>": [
        "\\a", "\\f", "\\n", "\\r", "\\t", "\\v",
        "\\\\", "\\.", "\\*", "\\+", "\\?", "\\|", "\\(", "\\)", "\\[", "\\]", "\\{", "\\}",
        "\\d", "\\D", "\\s", "\\S", "\\w", "\\W"
    ],

    "<literal_char>": [
        "<printable_char>"
    ],

    "<printable_char>": [
        c for c in (
            list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.")
            # simplification - toggle command for more complex chars
            # + list("!\"#$%&'()*+,-./:;<=>?@[]^_`{|}~")
        )   # if c not in ".^$*+?{}[]|()" # excluded bc special meaning in regex
    ]
}

RE2_GRAMMAR = convert_and_validate_ebnf_grammar(RE2_GRAMMAR)
