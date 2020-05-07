import pickle

import cma

from training.evaluator import *

tf.random.set_random_seed(1950)
random.seed(1950)
np.random.seed(1950)

SIGMA = 0.3

EVAL_LOG = "eval.txt"
BEST_LOG = "best.txt"
OPT_SAVE = "optimization.pkl"


def make_logger(log_dir):
    open(f"{log_dir}/{EVAL_LOG}", "w+").write("")
    open(f"{log_dir}/{BEST_LOG}", "w+").write("")


def write_best(best, best_val, state, log_dir):
    open(f"{log_dir}/{BEST_LOG}", "a+").write(json.dumps({
        "IND": list(best),
        "IND_DEC": state.repr_int(list(best)),
        "VAL": float(best_val),
    }) + "\n")


def search_cmaes(args, state, log_dir):
    args.logger = f"{log_dir}/{EVAL_LOG}"

    if not args.continue_iter:
        options = {'bounds': [0, 1], "maxiter": args.max_iter, "popsize": args.pop_size}
        es = cma.CMAEvolutionStrategy(state.get_random_individual(), SIGMA, options)
        make_logger(log_dir)
    else:
        es = pickle.load(open(f'{log_dir}/{OPT_SAVE}', 'rb'))

    while not es.stop():
        X = es.ask()
        es.tell(X, [evaluate(args, state, x) for x in X])
        es.logger.add()
        pickle.dump(es, open(f'{log_dir}/{OPT_SAVE}', 'wb'))
        best, best_val = es.result[0], es.result[1]
        write_best(best, best_val, state, log_dir)

    es.logger.disp()
