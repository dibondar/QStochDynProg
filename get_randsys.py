"""
Generate random system
"""
import networkx as nx
import numpy as np
from itertools import product
from scipy.linalg import expm

# np.random.seed(14)

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

field = np.linspace(0, 9, 2)

field_switching.add_edges_from(
    product(field, field)
)

###############################################################################################
#
#   Create the dictionary of propagators
#
###############################################################################################

# Number of levels in the quantum system
N = 3


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
            f: expm(-1j * (H0 + f * V)) for f in field_switching
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


###################################################################################################
#
#   Test
#
###################################################################################################


if __name__=='__main__':

    import matplotlib.pyplot as plt

    nx.draw_circular(
        field_switching,
        labels=dict((n, str(n)) for n in field_switching.nodes())
    )
    plt.show()