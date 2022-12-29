import numpy as np
import random
from selection_methods import proportional_method, stochastic_residual_method, threshold_method, tournament_method, \
    rank_method
from test_functions import circle_function
from mutation import mutation_bin_gen, mutation_bin_fen, mutation_tri_fen, mutation_tri_gen, mutation_real_fen
from crossover import pmx


class Agent:
    def __init__(self, vector: np.ndarray):
        self.vector = vector
        self.fitness_value = None
        self.scaled_fitness_value = None

    def __repr__(self):
        if self.fitness_value is not None:
            if isinstance(self.vector[0], str):
                return f"[{self.vector}, {round(self.fitness_value, 3)}]"
            return f"[{self.vector.round(decimals=3)}, {round(self.fitness_value, 3)}]"
        else:
            return f"[{self.vector.round(decimals=3)}"


def generate_starting_population(n_agents: int, limitations: np.ndarray) -> np.ndarray:
    population = np.empty(n_agents, dtype=object)
    n_dimensions = len(limitations)
    for i in range(n_agents):
        init_vector = np.zeros(n_dimensions)
        for j, limit in enumerate(limitations):
            init_vector[j] = random.uniform(limit[0], limit[1])
        population[i] = Agent(init_vector)

    return population


# Maybe min is better?
def get_best_from_population(population: np.ndarray) -> tuple[tuple, float]:
    return max(population, key=lambda agent: agent.fitness_value)


def calculate_fitness_function(function: callable, population: np.ndarray):
    for agent in population:
        agent.fitness_value = function(agent.vector)


def main(fitness_function: callable,
         coding: callable, decoding: callable,
         selection_method: callable,
         substitution_strategy: callable,
         crossover: callable,
         mutation: callable,
         scaling: callable,
         iterations: int,
         n_agents: int) -> tuple[tuple, float]:
    population = generate_starting_population(n_agents)
    calculate_fitness_function(fitness_function, population)

    for i in range(1, iterations):
        parents = selection_method(population)
        coding(parents)
        children = crossover(parents)
        mutation(children)
        decoding(children)
        decoding(parents)
        calculate_fitness_function(fitness_function, children)
        scaling(np.concatenate(population, children))
        substitution_strategy(population, children)

    return get_best_from_population(population)


# main()

limits = np.array([[-10, 10], [-1, 1]])
pop = generate_starting_population(10, limits)
calculate_fitness_function(circle_function, pop)
for i in range(10):
    mutation_real_fen(pop, limits, mutation_probability=1)

# # for testing
# def fit_func(agents):
#     for agent in agents:
#         agent.fitness_value = np.sum(agent.vector)
#
#
# AGENTS = np.empty(10, dtype=object)
# for I in range(10):
#     AGENTS[I] = Agent(np.random.random(3))
#
# fit_func(AGENTS)
# proportional_method(AGENTS)
# stochastic_residual_method(AGENTS)
# tournament_method(AGENTS)
# threshold_method(AGENTS, True, 5)
# # rank_method(AGENTS, True, True, 0, 1, 1) # moze zadziała z dobrze dobranymi parametrami xd
popul = np.array([Agent(np.array([4, 2, 8, 7, 5, 9, 1, 3, 6])), Agent(np.array([1, 9, 7, 5, 4, 6, 8, 2, 3]))])
pmx(popul)
print(popul)