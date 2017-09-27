import numpy as np
import networkx as nx
from collections import namedtuple, defaultdict
from operator import itemgetter


class QStochDynProg:
    """
    Quantum stochastic dynamical programming for severly constrained control.
    """

    # data structure to save the state of a system
    CState = namedtuple('State', ['cost_func', 'node', 'control', 'state'])

    get_cost_function = itemgetter('cost_func')
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

        # Initialize heaps for performing the optimization
        S = self.CState(
            cost_func=self.cost_func(init_state),
            node=0,
            control=init_control,
            state=init_state
        )
        self.previous_heap = [S]

        # Number of time step in optimization iteration
        self.current_iteration = 0

        # Landscape saved as a graph, where vertices are states
        self.landscape = nx.DiGraph()
        self.landscape.add_node(S.node, cost_func=S.cost_func, iteration=self.current_iteration)

    def next_time_step(self):
        """
        Go to the next time step in the time-domain optimization
        :return:
        """
        self.current_iteration += 1

        # initialize the heap
        current_heap = []

        # Loop over all states selected at the previous step
        for S in self.previous_heap:
            # Loop over all controls attainable from current control
            for C in self.control_switching[S.control]:

                # update the state
                new_state = self.propagator(C, S.state)

                new_S = self.CState(
                    cost_func=self.cost_func(new_state),
                    node=len(self.landscape),
                    control=C,
                    state=new_state
                )

                self.landscape.add_node(new_S.node, cost_func=new_S.cost_func, iteration=self.current_iteration)
                self.landscape.add_edge(new_S.node, S.node, control=new_S.control)

                current_heap.append(new_S)

        self.previous_heap = current_heap

    def get_pos_iteration_cost(self):
        """
        Landscape plotting utility.
        :return:
        """
        return {
            node:(self.get_iteration(prop), self.get_cost_function(prop))
            for node, prop in self.landscape.node.items()
        }

    def get_pos_cost_iteration(self):
        """
        Landscape plotting utility.
        :return:
        """
        return {
            node:(self.get_cost_function(prop), self.get_iteration(prop))
            for node, prop in self.landscape.node.items()
        }

    def get_node_color(self):
        """
        Landscape plotting utility.
        :return:
        """
        return [self.get_cost_function(n) for n in self.landscape.node.values()]

    def get_edge_color(self):
        """
        Landscape plotting utility.
        :return:
        """
        return [d['control'] for _,_,d in self.landscape.edges(data=True) if 'control' in d]

    def get_costs_in_iteration(self):
        """
        :return: A list of list. The outer list is corresponds to the iteration.
        The inner list contains values of the cost function obtained at the given iteration.
        """
        # Group cost functions by iterations
        cost_per_iteration = defaultdict(list)

        for n in self.landscape.node.values():
            cost_per_iteration[self.get_iteration(n)].append(self.get_cost_function(n))

        # sort by iteration
        cost_per_iteration = sorted(cost_per_iteration.items())

        return cost_per_iteration

    def get_optimal_policy(self):
        """
        Find the optimal control policy to maximize the objective function
        :return: max value of the cost function
            and list of controls that take from the initial condition to the optimal solution
        """
        # Find the maximal node
        max_cost, max_node = max(
             (self.get_cost_function(prop), node) for node, prop in self.landscape.node.items()
        )

        # Initialize variables
        opt_policy_controls = []
        current_node = self.landscape[max_node]

        # Walk from best node backwards to the initial condition
        while current_node:

            assert len(current_node) == 1, "Algorithm implemented incorrectly"

            # Assertion above guarantees that there will be only one element
            next_node, prop = current_node.items()[0]

            # Add extracted value of the control
            opt_policy_controls.append(prop['control'])

            # Extract next node
            current_node = self.landscape[next_node]

        # reverse the order in the list
        opt_policy_controls.reverse()

        return max_cost, opt_policy_controls

    def get_landscape_connectedness(self, **kwargs):
        """
        :param kwargs: the same as in https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram.html
        :return: list of list. The outermost list is a list of levels. The innermost list contains the sizes of
            each connected component.
        """
        costs, nodes = zip(
            *sorted(
                (self.get_cost_function(prop), node) for node, prop in self.landscape.node.items()
            )
        )

        costs = np.array(costs)

        # create the histogram of cost function values
        _, bin_edges = np.histogram(costs, **kwargs)

        levels = (
            nodes[indx:] for indx in np.searchsorted(costs, bin_edges[1:-1])
        )

        # make an undirected shallow copy of self.landscape
        landscape = nx.Graph(self.landscape)

        return [
            sorted(
                (len(cc) for cc in nx.connected_components(landscape.subgraph(nbunch))),
                reverse=True
            )
            for nbunch in levels
        ]

###################################################################################################
#
#   Test
#
###################################################################################################

if __name__=='__main__':

    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    #np.random.seed(1839127)

    from itertools import product
    from scipy.linalg import expm

    ###############################################################################################
    #
    #    Crate graph for switching the values of the field
    #
    ###############################################################################################

    field_switching = nx.Graph()

    # field = np.linspace(0, 9, 4)
    # for k in range(1, field.size):
    #     field_switching.add_edges_from(
    #         product([field[k]], field[k-1:k+2])
    #     )
    # # add separatelly the lower point
    # field_switching.add_edges_from(
    #     [(field[0], field[0]), (field[0], field[1])]
    # )

    field = np.linspace(0, 9, 3)

    field_switching.add_edges_from(
            product(field, field)
    )

    # nx.draw_circular(
    #     field_switching,
    #     labels=dict((n, str(n)) for n in field_switching.nodes())
    # )
    # plt.show()

    ###############################################################################################
    #
    #   Create the dictionary of propagators
    #
    ###############################################################################################

    # Number of levels in the quantum system
    N = 50

    class CPropagator:
        """
        Propagator with precalculated matrix exponents
        """
        def __init__(self, field_switching):
            # Generate the unperturbed hamiltonian
            H0 = np.random.rand(N, N) + 1j * np.random.rand(N, N)
            H0 += H0.conj().T

            # Generate the dipole matrix
            V = np.random.rand(N, N) + 1j * np.random.rand(N, N)
            V += V.conj().T

            # precalculate the matrix exponents
            self._propagators = {
               f:expm(-1j * (H0 + f * V)) for f in field_switching
            }

        def __call__(self, f, state):
            return self._propagators[f].dot(state)

    ###############################################################################################
    #
    #   Create the objective (cost) function
    #
    ###############################################################################################

    class CCostFunc:
        """
        Objective function
        """
        def __init__(self):
            self.O = np.random.rand(N, N) + 1j * np.random.rand(N, N)
            self.O += self.O.conj().T

        def __call__(self, state):
            return np.einsum('ij,i,j', self.O, state.conj(), state).real

    ###############################################################################################
    #
    #   Run the optimization
    #
    ###############################################################################################

    init_state = np.zeros(N)
    init_state[0] = 1.

    opt = QStochDynProg(
        init_state,
        field[field.size / 2],
        CPropagator(field_switching),
        field_switching,
        CCostFunc()
    )

    for _ in range(12):
       opt.next_time_step()

    ###############################################################################################
    #
    #   Plot results
    #
    ###############################################################################################

    # plt.title("Landscape")
    # plt.xlabel("time variable (dt)")
    # plt.ylabel("Value of objective function")
    # nx.draw(
    #     opt.landscape,
    #     pos=opt.get_pos_iteration_cost(),
    #     node_color=opt.get_node_color(),
    #     edge_color=opt.get_edge_color(),
    #     arrows=False,
    #     alpha=0.6,
    #     node_shape='s',
    #     with_labels=False,
    #     linewidths=0,
    # )
    # plt.axis('on')
    # plt.show()

    # Plot histogram per iteration
    ax = plt.figure().add_subplot(111, projection='3d')

    ax.set_title("Histograms of values of cost functions per iteration")

    # extract list of list of cost function values
    for iter_num, costs in opt.get_costs_in_iteration()[3:]:

        print(max(costs))
        # form a histogram for the given iteration
        hist, bin_edges = np.histogram(costs, normed=True)
        # set bin positions
        bin_position = 0.5 *(bin_edges[1:] + bin_edges[:-1])
        # plot
        ax.bar(bin_position, hist, zs=iter_num, zdir='y', alpha=0.9)

    ax.set_xlabel("Value of cost function")
    ax.set_zlabel("probability distribuation")
    ax.set_ylabel("Iteration")

    plt.show()


    # # Display the connectedness analysis
    # connect_info = opt.get_landscape_connectedness()
    #
    # plt.subplot(121)
    # plt.title("Number of disconnected pieces")
    # plt.semilogy([len(_) for _ in connect_info], '*-')
    # plt.ylabel('Number of disconnected pieces')
    # plt.xlabel('Level set number')
    #
    # plt.subplot(122)
    # plt.title("Size of largest connected piece")
    # plt.semilogy([max(_) for _ in connect_info], '*-')
    # plt.ylabel("Size of largest connected piece")
    # plt.xlabel('Level set number')
    #
    # plt.show()