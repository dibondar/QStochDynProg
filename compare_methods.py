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
    """
    Compare different optimization methods
    :param N: dimensionality of quantum system
    :return: a dictionary of results
    """
    # generate a random system
    sys = get_rand_unitary_sys(N)

    max_cost_func = sys["max_cost_func"]

    # we wrap the propagator to track how many calls each method makes
    propagator = sys["propagator"] = counted(sys["propagator"])

    # initialize the dictionary where results are stored
    results = dict(max_cost_func=max_cost_func, N=N)

    # ###################### Dynamical programming ######################
    results["dp_max"] = QStochDynProg(**sys).make_rounds(21).max_cost() / max_cost_func

    # save number of calls made during optimization
    results["dp_calls"] = propagator.calls

    # reset the count
    propagator.calls = 0

    ################ Hybrid monte-carlo tree seach and dynamical programming #############
    results["mcdp_max"] = MTCDynProg(nsteps=10, **sys).make_rounds(2 ** 13).max_cost() / max_cost_func

    # save number of calls made during optimization
    results["mcdp_calls"] = propagator.calls

    # reset the count
    propagator.calls = 0

    ###################### Monte-Carlo tree search ######################
    results["mcts_max"] = MCTreeSearch(nsteps=2 ** 10, **sys).make_rounds(2 ** 13).max_cost() / max_cost_func

    # save number of calls made during optimization
    results["mcts_calls"] = propagator.calls

    # reset the count
    propagator.calls = 0

    ###################### Genetic algorithm ######################
    results["ga_max"] = GA(nsteps=2 ** 11, pop_size=2 ** 7, **sys).make_rounds(2 ** 9).max_cost() / max_cost_func

    # save number of calls made during optimization
    results["ga_calls"] = propagator.calls

    # reset the count
    propagator.calls = 0

    return results


def plot_simulations(dir=''):
    """
    Plot results of simulations on a single plot
    """
    import matplotlib.pyplot as plt
    import glob
    import pickle

    def data2plot():
        """
        Generator yielding list to plot
        """
        for fname in glob.glob(dir + "*.simul"):
            with open(fname, "rb") as f:
                yield pickle.load(f)

    for R in data2plot():

        N = map(itemgetter("N"), R)

        dp_max = np.array(map(itemgetter("dp_max"), R))
        dp_calls = np.array(map(itemgetter("dp_calls"), R))

        mcdp_max = np.array(map(itemgetter("mcdp_max"), R))
        mcdp_calls = np.array(map(itemgetter("mcdp_calls"), R))

        mcts_max = np.array(map(itemgetter("mcts_max"), R))
        mcts_calls = np.array(map(itemgetter("mcts_calls"), R))

        ga_max = np.array(map(itemgetter("ga_max"), R))
        ga_calls = np.array(map(itemgetter("ga_calls"), R))

        # print np.mean(ga_calls / mcts_calls)

        plt.subplot(121)
        plt.title("Normalized max values at the end of optimization")
        plt.plot(N, mcts_max, 'r*-', alpha=0.4, label="Monte-Carlo tree search")
        plt.plot(N, ga_max, 'b*-', alpha=0.4, label="Genetic algorithm")
        plt.plot(N, mcdp_max, 'g*-', alpha=0.4, label="Monte-Carlo dynamical prog")
        plt.plot(N, dp_max, 'y*-', alpha=0.4, label="Dynamical prog")

        #plt.legend()

        plt.subplot(122)
        plt.title("Algorithmic efficiency: Max-value per propagation")
        plt.plot(N, mcts_max / mcts_calls, 'r*-', alpha=0.4, label="Monte-Carlo tree search")
        plt.plot(N, ga_max / ga_calls, 'b*-', alpha=0.4, label="Genetic algorithm")
        plt.plot(N, mcdp_max / mcdp_calls, 'g*-', alpha=0.4, label="Monte-Carlo dynamical prog")
        plt.plot(N, dp_max / dp_calls, 'y*-', alpha=0.4, label="Dynamical prog")
        #plt.legend()

    plt.show()

if __name__=='__main__':

    import pickle
    import tempfile

    #############################################################################
    #
    #   Calculate results and save
    #
    #############################################################################

    p = Pool(4)
    result = p.map(compare_methods, range(50, 3, -4))

    # save the results of simulations
    with tempfile.NamedTemporaryFile(suffix='.simul', delete=False, dir='') as f:
        pickle.dump(result, f)

    print("Results of simulation are saved in ", f.name)

    #############################################################################
    #
    #   Plot results
    #
    #############################################################################

    # display all the files
    # plot_simulations(data2plot())