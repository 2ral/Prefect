from modelizer.dependencies.fuzzingbook import convert_and_validate_ebnf_grammar, Grammar

BOTTLE_GRAMMAR: Grammar = {
    "<start>": ["<req>", "<req>~<req>", "<req>~<req>~<req>", "<req>~<req>~<req>~<req>"],
    "<req>": ["http://127.0.0.1:80/<path>#<method>",
              "http://<credentials>@127.0.0.1:80/protected.html#GET",
              "http://<user_state>@127.0.0.1:80/<path_state>#GET"],

    # reg
    "<path>": ["<valid_path>", "<invalid_path>"],
    "<valid_path>": ['test/test.html', 'about.html', 'contact.html', 'index_copy.html', 'index.html', '', 'hello', 'random'],
    "<invalid_path>": ['abc', 'test/test.htmx', 'contact.htmx', 'index.htmx'],

    "<method>": ["GET", "HEAD", "POST", "OPTIONS", "PUT", "PATCH", "DELETE"],

    # auth_user
    "<credentials>": ["<valid_credentials>", "<invalid_credentials>"],
    "<valid_credentials>": ["Nico:CovLearn", "Marius:LearnCov", "Tural:CISPA"],
    "<invalid_credentials>": ["<user>:<password>"],

    "<user>": ["Nico", "Marius", "Tural", "NoOne", "Balrog", "admin"],
    "<password>": ["CovLearn", "LearnCov", "CISPA", "wasd", "ShallNotPass", "1234", "admin"],

    # auth_server
    "<path_state>": ["login", "hidden", "logout"],
    "<user_state>": ["Nico:", "Marius:", "BalrogShallNotPass:"]
}

BOTTLE_GRAMMAR = convert_and_validate_ebnf_grammar(BOTTLE_GRAMMAR)
