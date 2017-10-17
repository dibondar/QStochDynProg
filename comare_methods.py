"""
Perform comparison of different optimization methods
using random unitary propagation system as an example
"""
from collections import namedtuple
from multiprocessing import Pool
from operator import itemgetter

from get_randsys import get_rand_unitary_sys
from qstochdynprog import QStochDynProg
from monte_carlo_tree_search import MCTreeSearch
from monte_carlo_dyn_prog import MTCDynProg

# Datatype to save comparison results
CResults = namedtuple("CResults", ["N", "max_cost_func", "dp", "mcts", "mcdp"])

def compare_methods(N):

    # generate a random system
    sys = get_rand_unitary_sys(N)

    max_cost_func = sys["max_cost_func"]

    return CResults(
        N=N,

        max_cost_func=max_cost_func,

        # Dynamical programming
        dp=QStochDynProg(**sys).make_rounds(20).max_cost() / max_cost_func,

        # Monte-Carlo tree search
        mcts=MCTreeSearch(nsteps=2000, **sys).make_rounds(1000).max_cost() / max_cost_func,

        # Hybrid monte-carlo tree seach and dynamical programming
        mcdp=MTCDynProg(nsteps=10, **sys).make_rounds(1000).max_cost() / max_cost_func,
    )


if __name__=='__main__':

    p = Pool(2)
    results = p.map(compare_methods, range(3, 50, 4))

    # plot results
    import matplotlib.pyplot as plt

    N = [x.N for x in results]

    plt.plot(N, [x.dp for x in results], '*-', label="Dynamical programming")
    plt.plot(N, [x.mcts for x in results], '*-', label="Monte-Carlo tree search")
    plt.plot(N, [x.mcdp for x in results], '*-', label="Hybrid monte-carlo tree seach and dynamical programming")
    plt.legend()

    plt.show()

