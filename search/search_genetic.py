import copy
import pickle

from training.evaluator import *

tf.random.set_random_seed(1950)
random.seed(1950)
np.random.seed(1950)

EVAL_LOG = "eval.txt"
BEST_LOG = "best.txt"
OPT_SAVE = "optimization.pkl"

MUTATION_P = 0.3
MUTATION_SCALE = 0.3
TOURNAMENT_K = 3


def make_logger(log_dir):
    open(f"{log_dir}/{EVAL_LOG}", "w+").write("")
    open(f"{log_dir}/{BEST_LOG}", "w+").write("")


def write_best(best, best_val, state, log_dir):
    open(f"{log_dir}/{BEST_LOG}", "a+").write(json.dumps({
        "IND": list(best),
        "IND_DEC": state.repr_int(list(best)),
        "VAL": float(best_val),
    }) + "\n")


def tournament_selection(population, size=TOURNAMENT_K):
    indices = np.random.permutation(len(population))[:size]
    tournament = [population[i] for i in indices]
    tournament.sort(key=lambda x: x[1])
    return tournament[0]


def mutate(ind, mutation_p=MUTATION_P, scale=MUTATION_SCALE):
    new_ind, _ = ind
    new_ind = copy.copy(new_ind)
    for i in range(len(new_ind)):
        if random.random() < mutation_p:
            new_ind[i] += float(np.random.normal(0, scale, 1))
            if new_ind[i] < 0: new_ind[i] = 0
            if new_ind[i] > 1: new_ind[i] = 1

    return new_ind


def search_genetic(args, state, log_dir):
    args.logger = f"{log_dir}/{EVAL_LOG}"

    if not args.continue_iter:
        pop = [(state.get_random_individual(), 0) for _ in range(args.pop_size)]
        pop = [(x[0], evaluate(args, state, x[0])) for x in pop]
        make_logger(log_dir)
    else:
        pop, pop_size, max_iter = pickle.load(open(f"{log_dir}/{OPT_SAVE}", "rb"))
        args.pop_size = pop_size
        args.max_iter -= max_iter

    half = args.pop_size // 2
    for iteration in range(args.max_iter):
        tournament_pop = []
        for _ in range(half):
            selected = tournament_selection(pop)
            selected = mutate(selected)
            tournament_pop.append((selected, evaluate(args, state, selected)))

        pop = sorted(pop, key=lambda x: x[1])
        best, bestVal = pop[0]
        write_best(best, bestVal, state, log_dir)
        pickle.dump((pop, args.pop_size, iteration), file=open(f"{log_dir}/{OPT_SAVE}", "wb"))

        pop = pop[:half] + tournament_pop
