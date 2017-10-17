from monte_carlo_tree_search import MCTreeSearch, nx
from qstochdynprog import QStochDynProg


class MTCDynProg(MCTreeSearch):
    """
    A monte carlo tree search using the dynamical programing (a deterministic method)
    for the simulation stage
    """
    def simulation(self, control, state):
        """
        Perform the simulation stage of the Monte Carlo tree search.
        Overloading the simulation method of the MCTreeSearch class.

        :param control: initial control
        :param state: initial state
        :return: maximum value of cost function
        """

        # initialize dynamical programing
        dynprog = QStochDynProg(
            init_state=state.copy(),
            init_control=control,
            propagator=self.propagator,
            control_switching=self.control_switching,
            cost_func=self.cost_func,
        )

        # perform dynamical programing for nsteps
        dynprog.make_rounds(self.nsteps)

        # find the best value of the cost function
        return dynprog.max_cost()

###################################################################################################
#
#   Test
#
###################################################################################################


if __name__=='__main__':

    import matplotlib.pyplot as plt

    ###############################################################################################
    #
    #   Run the optimization
    #
    ###############################################################################################

    # import declaration of random system
    from get_randsys import get_rand_unitary_sys

    mcts = MCTreeSearch(nsteps=100, **get_rand_unitary_sys(5))

    for _ in range(1):

        mcts.make_rounds(100)

        plt.title("Landscape")
        plt.xlabel("time variable (dt)")
        plt.ylabel("Value of objective function")
        nx.draw(
            mcts.decision_graph,
            pos=mcts.get_pos_iteration_cost(),
            node_color=mcts.get_node_color(),
            edge_color=mcts.get_edge_color(),
            arrows=False,
            alpha=0.8,
            node_shape='s',
            with_labels=False,
            linewidths=0,
        )

        # Display bounds on the cost func values
        ax = plt.gca()
        xlim = ax.get_xlim()

        ax.plot(xlim, [mcts.min_cost_func, mcts.min_cost_func], 'b')
        ax.annotate("lower bound on cost function", xy=(xlim[0], mcts.min_cost_func), color='b')

        ax.plot(xlim, [mcts.max_cost_func, mcts.max_cost_func], 'r')
        ax.annotate("upper bound on cost function", xy=(xlim[0], mcts.max_cost_func), color='r')

        plt.axis('on')
        plt.show()