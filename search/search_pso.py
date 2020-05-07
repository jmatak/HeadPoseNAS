from pyswarm import pso

from training.evaluator import *

tf.random.set_random_seed(1950)
random.seed(1950)
np.random.seed(1950)

EVAL_LOG = "eval.txt"
OMEGA = 0.5
PHI_P = 0.4
PHI_G = 0.6


def evaluator_wrapper(x, *args):
    return evaluate(args[0], args[1], x)


def make_logger(log_dir):
    open(f"{log_dir}/{EVAL_LOG}", "w+").write("")


def search_pso(arguments, state, log_dir):
    arguments.logger = f"{log_dir}/{EVAL_LOG}"
    if not arguments.continue_iter:
        make_logger(log_dir)

    lower = np.array([0 for _ in range(state.size)])
    upper = np.array([1 for _ in range(state.size)])

    pso(evaluator_wrapper, lower, upper, args=(arguments, state,),
        swarmsize=arguments.pop_size,
        omega=OMEGA, phip=PHI_P, phig=PHI_G,
        maxiter=arguments.max_iter,
        debug=False)