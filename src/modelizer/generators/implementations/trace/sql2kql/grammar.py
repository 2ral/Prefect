from string import ascii_letters, digits

from modelizer.dependencies.fuzzingbook import Grammar, convert_and_validate_ebnf_grammar


SQL_SELECT_GRAMMAR = {
    "<start>": ["<select>;"],

    "<select>": [
        "SELECT<distinct_opt>? <select_options> FROM <source><alias_opt>?<join_clause>?<where_clause>?<group_by_clause>?<order_by_clause>?"
    ],

    "<distinct_opt>": [" DISTINCT"],

    "<select_options>": [
        "<wildcard>",
        "<select_expr_list>",
    ],

    "<select_expr_list>": [
        "<select_expr>",
        "<select_expr>, <select_expr_list>"
    ],

    "<select_expr>": [
        "<column><alias_opt>",
        "<filter_function>(<column>)<alias_opt>"
    ],

    "<column>": [
        "<table_identifier>.<column_identifier>",
        "<column_identifier>"
    ],

    "<alias_opt>": [
        " AS <alias_identifier>"
    ],

    "<join_clause>": [
        " <join_type> JOIN <table_identifier><alias_opt> ON <table_identifier>.<column_identifier> = <table_identifier>.<column_identifier>"
    ],

    "<join_type>": [
        "INNER",
        "LEFT",
        "RIGHT",
        "FULL OUTER"
    ],

    "<where_clause>": [
        " WHERE <where_expr>"
    ],

    "<where_expr>": [
        "<predicate>",
        "<where_expr> AND <where_expr>",
        "<where_expr> OR <where_expr>",
        "(<where_expr>)"
    ],

    "<predicate>": [
        "<column> <comparison_operator> <value>",
        "NOT <column> <comparison_operator> <value>",
        "<column> IN (<value_list>)",
        "<column> NOT IN (<value_list>)",
        "<column> BETWEEN <value> AND <value>",
        "<column> NOT BETWEEN <value> AND <value>"
    ],

    "<group_by_clause>": [
        " GROUP BY <group_by_expr_list><having_clause>?"
    ],

    "<group_by_expr_list>": [
        "<column>",
        "<column>, <group_by_expr_list>"
    ],

    "<having_clause>": [
        " HAVING <function>(<column>) <comparison_operator> <value>"
    ],

    "<order_by_clause>": [
        " ORDER BY <order_by_criteria_list>"
    ],

    "<order_by_criteria_list>": [
        "<order_by_criteria>",
        "<order_by_criteria>, <order_by_criteria_list>"
    ],

    "<order_by_criteria>": [
        "<column>",
        "<column> ASC",
        "<column> DESC"
    ],


    "<value_list>": [
        "<value>",
        "<value>, <value_list>",
        "<select>"  # Allow subquery in IN (SELECT ...)
    ],

    "<comparison_operator>": [
        "=",
        "<",
        ">",
        "<=",
        ">=",
        "LIKE"
    ],

    "<wildcard>": ["*"],

    "<source>": [
        "<table_identifier>",
        "(<select>) <alias_identifier>"
    ],

    # Placeholders
    "<table_identifier>": ["TABLE"],
    "<alias_identifier>": ["ALIAS"],
    "<column_identifier>": ["COLUMN"],
    "<value>": ["VALUE"],

    "<filter_function>": [
        "COUNT",
        "SUM",
        "AVG",
        "MIN",
        "MAX"
    ],
    "<function>": [
        "COUNT",
        "SUM",
        "AVG",
        "MIN",
        "MAX"
    ]
}

SQL_MARKERS = {
    "<table_identifier>": "TABLE",
    "<column_identifier>": "COLUMN",
    "<alias_identifier>": "ALIAS",
    "<value>": "VALUE",
}

SQL_EXTENDED_GRAMMAR = {
    **SQL_SELECT_GRAMMAR,
    "<start>": ["<query>;"],
    "<query>": ["<select>", "<update>", "<delete>", "<insert>"],
    "<update>": ["UPDATE <table_identifier> SET <set_clause><where_clause>?"],
    "<delete>": ["DELETE FROM <table_identifier><where_clause>?"],
    "<insert>": ["INSERT INTO <table_identifier> (<column_list>) VALUES (<value_list>)"],
    "<column_list>": ["<column_identifier>", "<column_identifier>, <column_list>"],
    "<set_clause>": ["<column_identifier> = <value>", "<column_identifier> = <value>, <set_clause>"],
}


def refine_sql_grammar(sql_grammar: Grammar) -> Grammar:
    return {
        **sql_grammar,
        "<table_identifier>": ["<identifier>"],
        "<column_identifier>": ["<identifier>"],
        "<alias_identifier>": ["<identifier>"],
        "<identifier>": ["<letter><alphanumeric>*"],
        "<alphanumeric>": ["<letter>", "<digit>"],
        "<value>": ["<text_value>", "<number_value>"],
        "<number_value>": ["<integer_value>", "<float_value>"],
        "<text_value>": ["'<character>*'"],
        "<integer_value>": ["<digit>+"],
        "<float_value>": ["<digit>+.<digit>+"],
        "<character>": ["<letter>", "<digit>", "<symbol>"],
        "<letter>": list(ascii_letters),
        "<digit>": list(digits),
        "<symbol>": ["+", "-", "*", "/", "=", "!=", "<", ">", "<=", ">=", ",", ".", ";", ":", "(", ")", "[", "]", "{", "}"],
    }


SQL_SELECT_GRAMMAR_REFINED = refine_sql_grammar(SQL_SELECT_GRAMMAR)
SQL_EXTENDED_GRAMMAR_REFINED = refine_sql_grammar(SQL_EXTENDED_GRAMMAR)
SQL_SELECT_GRAMMAR = convert_and_validate_ebnf_grammar(SQL_SELECT_GRAMMAR)
SQL_SELECT_GRAMMAR_REFINED = convert_and_validate_ebnf_grammar(SQL_SELECT_GRAMMAR_REFINED)
