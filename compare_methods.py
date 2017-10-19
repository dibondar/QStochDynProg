"""
Perform comparison of different optimization methods
using random unitary propagation system as an example
"""
from multiprocessing import Pool
from operator import itemgetter

from get_randsys import get_rand_unitary_sys, np
from qstochdynprog import QStochDynProg
from monte_carlo_tree_search import MCTreeSearch
from monte_carlo_dyn_prog import MTCDynProg
from ga import GA


def counted(f):
    """
    A decorator that tracks how many times the function is called. Taken from
    https://stackoverflow.com/questions/21716940/is-there-a-way-to-track-the-number-of-times-a-function-is-called
    :param f: callable
    :return: callable
    """
    def wrapped(*args, **kwargs):
        wrapped.calls += 1
        return f(*args, **kwargs)
    wrapped.calls = 0
    return wrapped

def compare_methods(N):

    # generate a random system
    sys = get_rand_unitary_sys(N)

    max_cost_func = sys["max_cost_func"]

    # we wrap the propagator to track how many calls each method makes
    propagator = sys["propagator"] = counted(sys["propagator"])

    # initialize the dictionary where results are stored
    results = dict(max_cost_func=max_cost_func, N=N)

    # ###################### Dynamical programming ######################
    # results["dp_max"] = QStochDynProg(**sys).make_rounds(20).max_cost() / max_cost_func
    #
    # # save number of calls made during optimization
    # results["dp_calls"] = propagator.calls
    #
    # # reset the count
    # propagator.calls = 0
    #
    # ################ Hybrid monte-carlo tree seach and dynamical programming #############
    # results["mcdp_max"] = MTCDynProg(nsteps=10, **sys).make_rounds(2 ** 10).max_cost() / max_cost_func
    #
    # # save number of calls made during optimization
    # results["mcdp_calls"] = propagator.calls
    #
    # # reset the count
    # propagator.calls = 0

    ###################### Monte-Carlo tree search ######################
    results["mcts_max"] = MCTreeSearch(nsteps=2 ** 10, **sys).make_rounds(2 ** 9).max_cost() / max_cost_func

    # save number of calls made during optimization
    results["mcts_calls"] = propagator.calls

    # reset the count
    propagator.calls = 0

    ###################### Genetic algorithm ######################
    results["ga_max"] = GA(nsteps=2 ** 10, pop_size=2 ** 8, **sys).make_rounds(2 ** 8).max_cost() / max_cost_func

    # save number of calls made during optimization
    results["ga_calls"] = propagator.calls

    # reset the count
    propagator.calls = 0

    return results


if __name__=='__main__':

    p = Pool(8)
    results = p.map(compare_methods, range(50, 3, -4))

    # plot results
    import matplotlib.pyplot as plt

    N = map(itemgetter("N"), results)

    mcts_max = np.array(map(itemgetter("mcts_max"), results))
    mcts_calls = np.array(map(itemgetter("mcts_calls"), results))

    ga_max = np.array(map(itemgetter("ga_max"), results))
    ga_calls = np.array(map(itemgetter("ga_calls"), results))

    plt.subplot(121)
    plt.title("Normalized max values at the end of optimization")
    plt.plot(N, mcts_max, '*-', label="Monte-Carlo tree search")
    plt.plot(N, ga_max, '*-', label="Genetic algorithm")
    plt.legend()

    plt.subplot(122)
    plt.title("Algorithmic efficiency: Max-value per propagation")
    plt.plot(N, mcts_max / mcts_calls, '*-', label="Monte-Carlo tree search")
    plt.plot(N, ga_max / ga_calls, '*-', label="Genetic algorithm")
    plt.legend()

    plt.show()