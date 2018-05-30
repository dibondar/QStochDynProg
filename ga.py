import array
import random
from types import MethodType, FunctionType

import numpy as np

from deap import algorithms
from deap import base
from deap import creator
from deap import tools


class GA(object):
    """
    Genetic algorithm using the DEAP library https://github.com/DEAP/deap.

    This implementation is not universal.
    It works only for binary choice of controls with no switching rules.
    """
    def __init__(self, init_state, propagator, cost_func, nsteps, pop_size, **kwargs):
        """
        Constructor
        :param init_state: (object) The initial state of a system

        :param propagator: A callable object must accept three arguments: Self, the value of control C,
            and the state of a system S. Returns the new state of by acting onto state S by control C.

        :param control_switching: (optional) A graph specifying the rules of switching of controls.

        :param cost_func: An objective function to be maximized. It must accept three two arguments: self and state

        :param nsteps: number of steps to take during the simulation stage (aka, the length of genome)

        :param pop_size: the population size for GA (number of individuals per generation)

        :param min_cost_func: (optional) minimal value of attainable by the cost function

        :param max_cost_func: (optional) maximal value of attainable by the cost function
        """
        self.init_state = init_state
        self.nsteps = nsteps
        self.pop_size = pop_size

        self.propagator = MethodType(propagator, self)
        self.cost_func = MethodType(cost_func, self)

        # save all the other attributes
        for name, value in kwargs.items():
            # if the value supplied is a function, then dynamically assign it as a method;
            if isinstance(value, FunctionType):
                setattr(self, name, MethodType(value, self))
            # otherwise bind it as a property
            else:
                setattr(self, name, value)

        ########################################################################################
        #
        #   Initialize GA. This implementation is based on
        #       https://github.com/DEAP/deap/blob/master/examples/ga/onemax_short.py
        #
        ########################################################################################

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", array.array, typecode='b', fitness=creator.FitnessMax)

        self.toolbox = base.Toolbox()

        # Attribute generator
        self.toolbox.register("attr_bool", random.randint, 0, 1)

        # Structure initializers
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.attr_bool, self.nsteps)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        self.toolbox.register("evaluate", self.ga_obj_func)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

        self.pop = self.toolbox.population(n=self.pop_size)
        self.hof = tools.HallOfFame(1)
        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("avg", np.mean)
        self.stats.register("std", np.std)
        self.stats.register("min", np.min)
        self.stats.register("max", np.max)

    def ga_obj_func(self, individual):
        """
        The Fitness function for the genetic algorithm
        :param individual: the genome of an individual (string of controls)
        :return: value,
        """
        # make a local copy of the intial state
        state = self.init_state.copy()

        max_cost =  self.cost_func(state)

        for control in individual:
            # update state
            state = self.propagator(control, state)

            max_cost = max(max_cost, self.cost_func(state))

        return max_cost,

    def make_rounds(self, nrounds=1, verbose=False):
        """
        Performing nrounds of Monte Carlo rounds
        :param nsteps: a positive integer
        :param verbose: a boolean flag whether to print stats per each generation of GA
        :return: self
        """

        self.pop, self.log = algorithms.eaSimple(
            self.pop, self.toolbox, cxpb=0.5, mutpb=0.2, ngen=nrounds,
            stats=self.stats, halloffame=self.hof, verbose=verbose,
        )

        return self

    ###################################################################################################
    #
    #   Analysis
    #
    ###################################################################################################

    def max_cost(self):
        """
        Return the found maximal value of the cost fuction
        :return: max val
        """
        # Extract the best individual from the hol of fame
        return max(self.ga_obj_func(individual)[0] for individual in self.hof)

###################################################################################################
#
#   Test
#
###################################################################################################


if __name__=='__main__':
    ###############################################################################################
    #
    #   Run the optimization
    #
    ###############################################################################################

    # import declaration of random system
    from get_randsys import get_rand_unitary_sys

    ga = GA(nsteps=100, pop_size=200, **get_rand_unitary_sys(10))

    print("Maximal attainable value ", ga.max_cost_func)

    ga.make_rounds(nrounds=100, verbose=True)

