from string import ascii_lowercase, digits

from modelizer.dependencies.fuzzingbook import convert_and_validate_ebnf_grammar, Grammar


BC_GRAMMAR: Grammar = {
    "<start>": ["<program>"],

    "<program>": [
        "<statement> ;",
        "<statement> ; <program>"
    ],

    "<statement>": [
        "<expression>",
        "<lower_tier_statement>",
    ],

    # Lower tier more complex statements
    "<lower_tier_statement>": [
        "<assign_statement>",
        "<if_statement>",
        "<while_statement>",
        "<for_statement>",
        "{ <statement_list> }",
        "<rest>",
    ],

    "<statement_list>": ["<statement> ;", "<statement> ; <statement_list>"],

    # Other statements
    "<rest>": [
        # "print <print_list>",
        "<comment>",
        "<function_def>",
    ],

    # "<print_list>": ["<expression>", "<expression> , <print_list>"],
    "<comment>": ["# COMMENT\n"],

    # Control flow
    "<if_statement>": ["if ( <expression> ) <simple_statement>", "if ( <expression> ) <simple_statement> else <simple_statement>"],
    "<while_statement>": ["while ( <expression> ) { <while_body> }"],
    "<for_statement>": ["for ( <opt_expr> ; <opt_expr> ; <opt_expr> ) { <for_body> }"],
    "<opt_expr>": ["<expression>", ""],
    "<while_body>": ["<simple_statement> ;", "<simple_statement> ; <while_body>" "break ;"],
    "<for_body>": ["<simple_statement>", "<simple_statement> ; <for_body>", "break ;", "continue ;"],

    # Statements used in bodies of control flow
    "<simple_statement>": [
        "<expression>",
        "<assign_statement>",
        "<if_statement>",
        "<while_statement>",
        "<for_statement>",
        # "print <print_list>",
        "<stop_execution>"
    ],

    # Termination commands
    "<stop_execution>": ["halt", "quit"],

    # Math expressions and functions
    "<expression>": ["<or>"],
    "<or>": ["<and>", "<or> || <and>"],
    "<and>": ["<cmp>", "<and> && <cmp>"],
    "<cmp>": ["<sum>", "<sum> <cmp_op> <sum>"],
    "<cmp_op>": ["<", "<=", ">", ">=", "==", "!="],
    "<sum>": ["<prod>", "<sum> + <prod>", "<sum> - <prod>"],
    "<prod>": ["<power>", "<prod> * <power>", "<prod> / <power>", "<prod> % <power>"],
    "<power>": ["<unary>", "<unary> ^ <power>"],
    "<unary>": ["<number>", "( <expression> )", "<call>", "<builtin>", "- <unary>"],

    "<builtin>": [
        "sqrt ( <expression> )",
        "length ( <expression> )",
        "scale ( <expression> )",
        "s ( <expression> )",
        "c ( <expression> )",
        "l ( <expression> )",
        "e ( <expression> )",
        "j ( <expression> , <expression> )",
    ],

    # Numbers
    "<number>": ["<int>", "<int> . <int>"],
    "<int>": ["<digit>", "<digit><int>"],
    "<digit>": list(digits),

    # Variables and assignments
    "<assign_statement>": ["<id> = <expression>"],
    "<id>": ["<letter><name_opts>"],
    "<name_opts>": ["", "<letter><name_opts>", "<digit><name_opts>", "_<name_opts>"],
    "<letter>": list(ascii_lowercase),

    # Functions
    "<function_def>": ["define <id> ( <param_list_opt> ) { <auto_opt> <statement_list> }"],
    "<param_list_opt>": ["", "<param_list>"],
    "<param_list>": ["<id>", "<id> , <param_list>"],
    "<auto_opt>": ["auto <auto_list> ;", ""],
    "<auto_list>": ["<id>", "<id> , <auto_list>"],
    "<call>": [
        "<id> ( <arg_list_opt> )",
        # "return <opt_expr>",
    ],
    "<arg_list_opt>": ["<arg_list>", ""],
    "<arg_list>": ["<expression>", "<expression> , <arg_list>"],
}

BC_GRAMMAR = convert_and_validate_ebnf_grammar(BC_GRAMMAR)
