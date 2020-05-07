from training.evaluator import *

tf.random.set_random_seed(1950)
random.seed(1950)
np.random.seed(1950)

EVAL_LOG = "eval.txt"
BEST_LOG = "best.txt"


def make_logger(log_dir):
    open(f"{log_dir}/{EVAL_LOG}", "w+").write("")
    open(f"{log_dir}/{BEST_LOG}", "w+").write("")


def write_best(best, best_val, state, log_dir):
    open(f"{log_dir}/{BEST_LOG}", "a+").write(json.dumps({
        "IND": list(best),
        "IND_DEC": state.repr_int(list(best)),
        "VAL": float(best_val),
    }) + "\n")


def search_random(args, state, log_dir):
    args.logger = f"{log_dir}/{EVAL_LOG}"
    if not args.continue_iter:
        make_logger(log_dir)

    best, bestVal = None, 0
    for iteration in range(args.max_iter * args.pop_size):
        ind = state.get_random_individual()
        res = evaluate(args, state, ind)

        if res < bestVal or best is None:
            best = ind
            bestVal = res

        if iteration % args.pop_size == 0:
            write_best(best, bestVal, state, log_dir)
