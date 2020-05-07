import pybobyqa

from training.evaluator import *

tf.random.set_random_seed(1950)
random.seed(1950)
np.random.seed(1950)

EVAL_LOG = "eval.txt"


def evaluator_wrapper(x, *args):
    return evaluate(args[0], args[1], x)


def make_logger(log_dir):
    open(f"{log_dir}/{EVAL_LOG}", "w+").write("")


def search_bobyqa(arguments, state, log_dir):
    arguments.logger = f"{log_dir}/{EVAL_LOG}"
    if not arguments.continue_iter:
        make_logger(log_dir)

    lower = np.array([0 for _ in range(state.size)])
    upper = np.array([1 for _ in range(state.size)])

    for i in range(arguments.pop_size):
        x0 = np.array(state.get_random_individual())

        soln = pybobyqa.solve(evaluator_wrapper, x0,
                              args=(arguments, state,),
                              maxfun=arguments.max_iter,
                              bounds=(lower, upper),
                              seek_global_minimum=True)
        print(soln)
