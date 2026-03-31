import argparse

from modelizer.configs import SEED

from modelizer.generators.implementations import Bottle_Subject, Bottle_Generator


# USAGE: python scripts/subjects/generate_bottle.py -c 10 -d /mount/ -min 1 -max 24

if __name__ == "__main__":
    # argparser for general params
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-c', '--count', type=int, help='How big the generated dataset should be, default is 10', metavar="")
    arg_parser.add_argument('-d', '--data-dir', type=str, help='Where the generated data should be stored', metavar="")
    arg_parser.add_argument('-min', '--min', type=int, help='min nonterminals', metavar="")
    arg_parser.add_argument('-max', '--max', type=int, help='max nonterminals', metavar="")
    arg_parser.add_argument('-p', '--port', type=int, default=35000, help='server port', metavar="")
    args = arg_parser.parse_args()

    path = f"{args.data_dir}bottle-{args.count}-{args.min}_{args.max}.pkl"

    subj = Bottle_Subject(port=args.port)
    gen = Bottle_Generator(subject=subj, seed=SEED, min_nonterminals=args.min, max_nonterminals=args.max)

    # generation api calls
    gen.generate_samples(count=args.count)
    gen.export(filepath=path, to_csv=False)
    subj.post_execution()
