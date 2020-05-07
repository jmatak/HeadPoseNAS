import argparse
import pathlib

from model.representation import *
from search.search_random import search_random
from search.search_pso import search_pso
from search.search_genetic import search_genetic
from search.search_cmaes import search_cmaes
from search.search_bobyqa import search_bobyqa

tf.random.set_random_seed(1950)

log_config = {
    "random": "output/random",
    "bobyqa": "output/bobyqa",
    "cmaes": "output/cmaes",
    "genetic": "output/genetic",
    "pso": "output/pso",
}


def make_log_dir(log_dir):
    path = pathlib.Path(f"{log_dir}/")
    path.mkdir(parents=True, exist_ok=True)


def get_arguments():
    parser = argparse.ArgumentParser(description='Pose extractor trainer.')

    parser.add_argument('-alg', "--algorithm", type=str, default="cmaes")

    parser.add_argument('-ma', "--machine", type=str, default="ludmila")
    parser.add_argument('-d', "--train_set", type=str, default="300w_train")
    parser.add_argument('-v', "--val_set", type=str, default="300w_val")
    parser.add_argument('-m', "--model", type=str, default="inception")

    parser.add_argument('-la', "--layer", type=int, default=13)
    parser.add_argument('-lr', "--learning_rate", nargs='+',
                        default=[0.0005, 0.0002, 0.00009, 0.00004, 0.00001, 0.00001])
    parser.add_argument('-b', "--batch_size", type=int, default=32)
    parser.add_argument('-e', "--epochs", type=int, default=6)
    parser.add_argument('-g', "--gpu", type=bool, default=True)

    parser.add_argument('-c', "--continue_iter", type=bool, default=False)
    parser.add_argument('-max', "--max_iter", type=int, default=20)
    parser.add_argument('-p', "--pop_size", type=int, default=10)

    return parser.parse_args()


if __name__ == '__main__':
    args = get_arguments()
    state = NeuralSearchState()

    if args.algorithm == 'random':
        make_log_dir(log_config['random'])
        search_random(args, state, log_dir=log_config['random'])

    elif args.algorithm == 'pso':
        make_log_dir(log_config['pso'])
        search_pso(args, state, log_dir=log_config['pso'])

    elif args.algorithm == 'genetic':
        make_log_dir(log_config['genetic'])
        search_genetic(args, state, log_dir=log_config['genetic'])

    elif args.algorithm == 'cmaes':
        make_log_dir(log_config['cmaes'])
        search_cmaes(args, state, log_dir=log_config['cmaes'])

    elif args.algorithm == 'bobyqa':
        make_log_dir(log_config['bobyqa'])
        search_bobyqa(args, state, log_dir=log_config['bobyqa'])

    else:
        print("Please pick a valid algorithm.")
        exit(1)
