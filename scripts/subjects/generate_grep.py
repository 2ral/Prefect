import argparse

from modelizer.configs import SEED

from modelizer.generators.implementations import Grep_Subject, Grep_Generator

DEFAULT_COUNT = 12000
DEFAULT_PATH = "/mount/"


# USAGE: python scripts/subjects/generate_grep.py -c 10 -d /mount/

if __name__ == "__main__":
    # argparser for general params
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-c', '--count', type=int, help='How big the generated dataset should be, default is 10', metavar="")
    arg_parser.add_argument('-d', '--data-dir', type=str, help='Where the generated data should be stored', metavar="")
    args = arg_parser.parse_args()

    path = f"{args.data_dir}grep-{args.count}.pkl"
    subj = Grep_Subject()
    gen = Grep_Generator(subject=subj, seed=SEED)

    # generation api calls
    gen.generate_samples(count=args.count)
    gen.export(filepath=path, to_csv=False)
    # subj.post_execution()
