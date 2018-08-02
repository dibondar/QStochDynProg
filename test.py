from monte_carlo_tree_search import *
# import declaration of random system
from get_randsys import get_rand_unitary_sys
from random import seed
from multiprocessing import Pool


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


def calculate(N, n_opt_policy):
    """

    :param args:
    :return:
    """
    # seed the random number generators
    np.random.seed(1581)
    seed(6019)

    # generate the system
    sys = get_rand_unitary_sys(N, n_opt_policy)

    # agument the propagator so that we can count the number of calls to it
    propagator = sys["propagator"] = counted(sys["propagator"])

    # initialize the optimization method
    mcts = MCTreeSearch(nsteps=10, **sys)

    # number of steps in the simulation
    nsteps = 0

    while not np.isclose(mcts.max_cost_func, mcts.cost_opt_control_policy) and nsteps < 2e5:
        mcts.make_rounds()
        nsteps += 1

    if np.isclose(mcts.max_cost_func, mcts.cost_opt_control_policy):
        # return real length of the policy plus the number of calls
        return len(mcts.opt_control_policy), propagator.calls
    else:
        return None


if __name__=='__main__':


    #############################################################################
    #
    #
    #
    #############################################################################

    from itertools import product
    import matplotlib.pyplot as plt

    len_control_policy = np.arange(10, 5, -2)

    with Pool() as pool:
        results = pool.starmap(calculate, product([5], len_control_policy))

    # remove Nones from results
    results = [_ for _ in results if _ is not None]

    len_control_policy, propagator_calls = zip(*results)
    len_control_policy = np.array(len_control_policy)

    plt.semilogy(len_control_policy, propagator_calls, '*')
    plt.semilogy(len_control_policy, 2 ** (len_control_policy + 1) - 2)
    plt.show()