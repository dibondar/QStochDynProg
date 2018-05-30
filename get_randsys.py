import networkx as nx
import numpy as np
from itertools import product
from scipy.linalg import expm


def get_rand_unitary_sys(N):
    """
    Generate random unitary evolution
    :param N: a number of levels of the quantum system
    :return: init_state, init_control, propagator, control_switching, cost_func
    """

    ###############################################################################################
    #
    #    Crate graph for switching the values of the randomly generated unitary controls
    #
    ###############################################################################################

    control_switching = nx.Graph()

    # employ two controls and allow all possible switching between them
    control_switching.add_edges_from(
        product([0,1], [0,1])
    )

    # select the initial control
    init_control = 0

    ###############################################################################################
    #
    #    Utility function generating random hermitian matrix
    #
    ###############################################################################################

    def get_rand_herm():
        H = np.random.rand(N, N) + 1j * np.random.rand(N, N)
        H += H.conj().T
        return H

    ###############################################################################################
    #
    #   Create the propagator
    #
    ###############################################################################################

    # pre-calculate the matrix exponents
    _propagators = {
        control: expm(1j * get_rand_herm()) for control in control_switching
    }

    def propagator(self, control, state):
        """
        Propagate the state
        :param control:
        :param state: initial wave function
        :return: updated wave function
        """
        return self._propagators[control].dot(state)

    ###############################################################################################
    #
    #   Create the objective (cost) function
    #
    ###############################################################################################

    observable = get_rand_herm()

    # extracting observable extrema
    spectra_observable = np.linalg.eigvalsh(observable)
    min_cost_func = spectra_observable.min()
    max_cost_func = spectra_observable.max()

    def cost_func(self, state):
        return np.einsum('ij,i,j', self.observable, state.conj(), state).real

    ###############################################################################################
    #
    #   Generate initial state
    #
    ###############################################################################################

    init_state = np.zeros(N, dtype=np.complex)
    init_state[0] = 1.

    return {
        "init_state": init_state,
        "init_control": init_control,

        "_propagators": _propagators,
        "propagator": propagator,

        "control_switching": control_switching,

        "observable": observable,
        "cost_func": cost_func,

        "min_cost_func" : min_cost_func,
        "max_cost_func" : max_cost_func,
    }

