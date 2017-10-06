import numpy as np
import heapq
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

    def get_best_nodes_per_iteration(self, nnodes=1):
        """
        Find best nnodes per iteration
        :param nnodes: an integer indicating how many nodes per iterations are to be extracted (default 1).
        :return: The list of lists of best nodes
        """
        best_nodes_per_iteration = defaultdict(list)

        # Group all nodes by iteration
        for node, prop in self.landscape.node.items():
            best_nodes_per_iteration[self.get_iteration(prop)].append(node)

        # sort by iteration
        best_nodes_per_iteration = sorted(best_nodes_per_iteration.items())

        # extract nodes by ignoring iteration number
        best_nodes_per_iteration = zip(*best_nodes_per_iteration)[1]

        # Extracting sought best nnotes per iteration
        best_nodes_per_iteration = [
            heapq.nlargest(nnodes, nodes, key=lambda n: self.get_cost_function(self.landscape.node[n]))
            for nodes in best_nodes_per_iteration
        ]

        return best_nodes_per_iteration

    def get_best_path_per_iteration(self, npath=1):
        """
        Get values of the cost function along a best path per each iteration
        :param npath: an integer indicating how many paths per iterations are to be extracted (default 1).
        :return: A generator returning the list of cost functions along a path
        """
        # get the dict of best nodes
        for best_nodes in self.get_best_nodes_per_iteration(npath):
            for node in best_nodes:
                # extract list of nodes using depth first search
                # Note: depth first search must yield the same result as breath first search
                nodes = list(nx.dfs_preorder_nodes(self.landscape, node))
                nodes.reverse()
                yield [self.get_cost_function(self.landscape.node[_]) for _ in nodes]

    def get_costs_per_iteration(self):
        """
        :return: A list of tuples of the form [(iteration number, [list of values of obtained at this iteration],...]).
        """
        # Group cost functions by iterations
        cost_per_iteration = defaultdict(list)

        for n in self.landscape.node.values():
            cost_per_iteration[self.get_iteration(n)].append(self.get_cost_function(n))

        # sort by iteration
        cost_per_iteration = sorted(cost_per_iteration.items())

        return cost_per_iteration

    def get_best_paths(self, npaths=-1):
        """
        Extract npaths paths leading to the highest values of the cost function
        :param npaths: an integer specifying number of paths to generate (default all path)
        :return: generator returning the list of nodes.
        """
        # set the default number of paths if npaths is negative
        npaths = (len(self.landscape) if npaths < 0 else npaths)

        # Get npaths nodes with the largest cost function, which represent the end of policies
        best_nodes = heapq.nlargest(
            npaths,
            ((self.get_cost_function(prop), node) for node, prop in self.landscape.node.items())
        )

        # Extract the nodes
        best_nodes = zip(*best_nodes)[1]

        for end_node in best_nodes:
            # extract list of nodes using depth first search
            # Note: depth first search must yield the same result as breath first search
            nodes = list(nx.dfs_preorder_nodes(self.landscape, end_node))
            nodes.reverse()
            yield nodes

    def get_best_paths_cost_func(self, npaths=-1):
        """
        Get values of the cost function along a best path
        :param npaths: an integer specifying number of paths to generate (default all path)
        :return: A generator of lists of cost functions along a path
        """
        for nodes in self.get_best_paths(npaths):
            yield [self.get_cost_function(self.landscape.node[_]) for _ in nodes]

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

    opt = QStochDynProg(
        init_state,
        field[field.size / 2],
        CPropagator(field_switching),
        field_switching,
        CCostFunc()
    )

    for iter_num in range(10):
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

    ###############################################################################################

    # Plot histogram per iteration
    ax = plt.figure().add_subplot(111, projection='3d')

    ax.set_title("Histograms of values of cost functions per iteration")

    # extract list of list of cost function values
    for iter_num, costs in opt.get_costs_per_iteration()[6:]:

        print(max(costs))
        # form a histogram for the given iteration
        hist, bin_edges = np.histogram(costs, normed=True)
        # set bin positions
        bin_position = 0.5 *(bin_edges[1:] + bin_edges[:-1])
        # plot
        ax.bar(bin_position, hist, zs=iter_num, zdir='y', alpha=0.8)

    ax.set_xlabel("Value of cost function")
    ax.set_zlabel("probability distribuation")
    ax.set_ylabel("Iteration")

    plt.show()

    ###############################################################################################

    # nbestpath = 5
    #
    # plt.title("Cost functions along best %d optimization trajectories" % nbestpath)
    #
    # for num, path in enumerate(opt.get_best_paths_cost_func(nbestpath)):
    #     plt.plot(path, label=str(num))
    #
    # plt.legend()
    # plt.xlabel("Iteration")
    # plt.ylabel("Cost function")
    #
    # plt.show()

    ###############################################################################################

    # plt.title("Best path per iteration")
    #
    # for path in opt.get_best_path_per_iteration():
    #     plt.plot(path)
    #
    # plt.xlabel("Iteration")
    # plt.ylabel("Cost function")
    #
    # plt.show()

    ###############################################################################################

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