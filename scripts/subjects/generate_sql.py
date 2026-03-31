import argparse

from modelizer.configs import SEED

from modelizer.generators.implementations import SQL2KQL_Subject, SQL2KQL_Generator
from modelizer.dependencies.fuzzingbook import GrammarCoverageFuzzer, KPathGrammarFuzzer


DEFAULT_COUNT = 12000
DEFAULT_PATH = "/mount/"


# USAGE: python scripts/subjects/generate_sql.py -c 10 -d /mount/ -f coverage -min 1 -max 24

if __name__ == "__main__":
    # argparser for general params
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-c', '--count', type=int, help='How big the generated dataset should be, default is 10', metavar="")
    arg_parser.add_argument('-d', '--data-dir', type=str, help='Where the generated data should be stored', metavar="")
    arg_parser.add_argument('-f', '--fuzzer', type=str, choices=["coverage","kpath"], help="which fuzzer")
    arg_parser.add_argument('-min', '--min', type=int, help='min nonterminals', metavar="")
    arg_parser.add_argument('-max', '--max', type=int, help='max nonterminals', metavar="")
    args = arg_parser.parse_args()

    path = f"{args.data_dir}sql-{args.count}-{args.fuzzer[0]}-{args.min}_{args.max}.pkl"

    subj = SQL2KQL_Subject()
    if args.fuzzer == "coverage":
        gen = SQL2KQL_Generator(subject=subj, seed=SEED, fuzzer_factory=GrammarCoverageFuzzer, min_nonterminals=args.min, max_nonterminals=args.max)
    elif args.fuzzer == "kpath":
        gen = SQL2KQL_Generator(subject=subj, seed=SEED, fuzzer_factory=KPathGrammarFuzzer, min_nonterminals=args.min, max_nonterminals=args.max)

    # generation api calls
    gen.generate_samples(count=args.count)
    gen.export(filepath=path, to_csv=False)
    # subj.post_execution()
