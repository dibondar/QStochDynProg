from random import choice
import numpy as np
import networkx as nx
from operator import itemgetter


class MCTreeSearch(object):
    """
    The Monte Carlo tree search methodology to maximize the cost function
    """
    get_weight = itemgetter('weight')
    get_control = itemgetter('control')
    get_iteration = itemgetter('iteration')

    def __init__(self, init_state, init_control, propagator, control_switching, cost_func):
        """
        Constructor
        :param init_state: (object) The initial state of a system

        :param init_control: (object) The initial vale of a control.
            It is needed to determine which controls can be used at the next time step.

        :param propagator: A callable object must accept two arguments: The value of control C
            and the state of a system S. Returns the new state of by acting onto state S by control C.

        :param control_switching: A graph specifying the rules of switching of controls.

        :param cost_func: An objective function to be maximized.
        """

        # Save parameters
        self.propagator = propagator
        self.control_switching = control_switching
        self.cost_func = cost_func

        self.init_state = init_state
        self.init_control = init_control

        # Number of time step in optimization iteration
        self.current_iteration = 0

        # Initialize the decision graph
        self.decision_graph = nx.DiGraph()

        # add the root node
        self.decision_graph.add_node(
            0,
            weight=self.cost_func(init_state),
            control=init_control,
            iteration=0,
        )

    def make_rounds(self, nrounds=1):
        """
        Generator performing nrounds of Monte Carlo rounds
        :param nsteps: a positive integer
        :return: generator yielding latest added node
        """
        # loop over rounds
        for _ in range(nrounds):

            # each round of Monte Carlo tree search consists of four steps
            leaf_node, control, state = self.selection()

            ##########################################################################
            #
            #  Expansion stage
            #
            ##########################################################################

            # extact current iteration value
            current_iteration = 1 + self.get_iteration(self.decision_graph.node[leaf_node])

            new_nodes = []

            # Loop over all controls attainable from the leaf_node node
            for C in self.control_switching[control]:

                # the name of a new node
                new_node = len(self.decision_graph)

                # add the new node and edge
                self.decision_graph.add_node(
                    new_node,
                    control=C,
                    iteration=current_iteration,
                    ##########################################################################
                    #
                    #   Simulation stage to get the weight of a new node
                    #
                    ##########################################################################
                    weight=self.simulation(C, state),
                )
                self.decision_graph.add_edge(new_node, leaf_node)

                new_nodes.append(new_node)

            ##########################################################################
            #
            #  Backpropagation stage
            #
            ##########################################################################

            for new_node in new_nodes:
                new_weight = self.get_weight(self.decision_graph.node[new_node])
                n = new_node

                while len(self.decision_graph[n]):
                    assert len(self.decision_graph[n]) == 1
                    # go one level up
                    n, = self.decision_graph[n]
                    if self.get_weight(self.decision_graph.node[n]) >= new_weight:
                        break
                    else:
                        # updade the value
                        self.decision_graph.node[n].update(weight=new_weight)




    def selection(self):
        """
        Perform the selection step of the Monte Carlo tree search
        :return: a tuple of 1) a leaf node from which the expansion stage takes over,
                2) the control that brought to that node, and 3) the current state
        """
        # in order to be able to move from the root to a leaf node,
        # the direction of the decsion graph needs to be reversed
        self.decision_graph.reverse(copy=False)

        # initialize the system state
        state = self.init_state.copy()
        control = self.init_control

        # start from the root and select successive child nodes down to a leaf node
        current_node = 0

        while len(self.decision_graph[current_node]):

            # extract where to go next
            candidate_nodes = self.decision_graph[current_node]

            # extract probabilities from weights
            p = np.fromiter(
                (self.get_weight(self.decision_graph.node[_]) for _ in candidate_nodes),
                np.float,
                len(candidate_nodes)
            )
            p /= p.sum()

            # make a random choice
            current_node = np.random.choice(candidate_nodes.keys(), p=p)

            # extract the control that brings to the selected node
            control = self.get_control(self.decision_graph.node[current_node])

            # propagate to the current node
            state = self.propagator(control, state)

        # restore the original directionality of the graph
        self.decision_graph.reverse(copy=False)

        return current_node, control, state

    def simulation(self, control, state, nsteps=10):
        """
        Perform the simulation stage of the Monte Carlo tree search
        :param control: initial control
        :param state: initial state
        :param nsteps: numer of random steps to take
        :return: maximum value of cost function
        """
        # make a local copy of the state
        state = state.copy()

        max_cost = -np.inf

        for _ in range(nsteps):
            # update state
            state = self.propagator(control, state)

            # randomly choose where to go next
            control = choice(self.control_switching[control].keys())

            max_cost = max(max_cost, self.cost_func(state))

        return max_cost

    ###################################################################################################
    #
    #   Plotting facilities
    #
    ###################################################################################################

    def get_node_color(self):
        """
        Decision graph plotting utility.
        :return:
        """
        return [self.get_weight(n) for n in self.decision_graph.node.values()]

    def get_edge_color(self):
        """
        Decision graph plotting utility.
        :return:
        """
        return [self.get_control(self.decision_graph.node[n]) for n,_ in self.decision_graph.edges()]

    def get_pos_iteration_cost(self):
        """
        Decision graph plotting utility.
        :return:
        """
        return {
            node:(self.get_iteration(prop), self.get_weight(prop))
            for node, prop in self.decision_graph.node.items()
        }

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
    from get_randsys import *

    init_state = np.zeros(N)
    init_state[0] = 1.

    mcts = MCTreeSearch(
        init_state,
        field[field.size / 2],
        CPropagator(field_switching),
        field_switching,
        CCostFunc()
    )

    for _ in range(5):

        mcts.make_rounds(10)

        plt.title("Landscape")
        plt.xlabel("time variable (dt)")
        plt.ylabel("Value of objective function")
        nx.draw(
            mcts.decision_graph,
            pos=mcts.get_pos_iteration_cost(),
            node_color=mcts.get_node_color(),
            edge_color=mcts.get_edge_color(),
            arrows=False,
            alpha=0.6,
            node_shape='s',
            with_labels=False,
            linewidths=0,
        )
        plt.axis('on')
        plt.show()