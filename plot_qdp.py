################################################################################
#
#   Plot figures in the paper
#
################################################################################

from monte_carlo_tree_search import MCTreeSearch
from qstochdynprog import QStochDynProg, np, nx
from get_randsys import get_rand_unitary_sys
from copy import deepcopy
from random import seed

import matplotlib.pyplot as plt


# set the common axis
_, axes = plt.subplots(2, 3, sharex=True, sharey=True)

################################################################################
#
#   Plot Monte Carlo tree search
#
################################################################################

# send random seed
seed(122)
np.random.seed(6711)

# create the data for plotting
mcts = MCTreeSearch(nsteps=100, **get_rand_unitary_sys(5))
data = [deepcopy(mcts.make_rounds(1)) for _ in range(axes.shape[1])]

vmin = min(mcts.get_original_weight(_) for _ in mcts.decision_graph.node.values())
vmax = max(mcts.get_original_weight(_) for _ in mcts.decision_graph.node.values())

for title, mcts, ax in zip(["(a)", "(b)", "(c)"], data, axes[0]):

    ax.set_title(title)
    nx.draw(
        mcts.decision_graph,
        pos=mcts.get_pos_iteration_cost(),
        node_color=mcts.get_node_color_weight(),
        edge_color=mcts.get_edge_color_weight(),
        arrows=False,
        #alpha=0.8,
        node_shape='o',
        with_labels=True,
        linewidths=1,
        width=2,

        # matplotlib params
        vmin=vmin,
        vmax=vmax,
        edge_vmin=vmin,
        edge_vmax=vmax,
        ax=ax,
    )

del mcts

################################################################################
#
#   Plot the dynamical programing
#
################################################################################

# send random seed
seed(122)
np.random.seed(6711)

# create the data for plotting
dp = QStochDynProg(nsteps=100, **get_rand_unitary_sys(5))
data = [deepcopy(dp.make_rounds(1)) for _ in range(axes.shape[1])]

for title, dp, ax in zip(["(d)", "(e)", "(f)"], data, axes[1]):

    ax.set_title(title)
    nx.draw(
        dp.landscape,
        pos=dp.get_pos_iteration_cost(),
        node_color=dp.get_node_color(),
        edge_color=dp.get_edge_color(),
        arrows=False,
        #alpha=0.8,
        node_shape='o',
        with_labels=False,
        linewidths=1,
        width=1,

        # matplotlib params
        vmin=vmin,
        vmax=vmax,
        edge_vmin=vmin,
        edge_vmax=vmax,
        ax=ax,
    )


plt.show()
