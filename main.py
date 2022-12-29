import copy

import numpy as np
import random
from selection_methods import proportional_method, stochastic_residual_method, threshold_method, tournament_method, \
    rank_method
from test_functions import circle_function, quadratic_function, dummy, cross_in_tray_function, bukin_function, \
    holder_table_function, egg_holder_function, griewank_function, drop_wave_function, levy_function_n13, \
    rastrigin_function
from mutation import mutation_bin_gen, mutation_bin_fen, mutation_tri_fen, mutation_tri_gen, mutation_real_fen
from crossover import pmx, arithmetic_crossover, mixed_crossover
from substitution_strategy import full_sub_strategy, \
    part_reproduction_elite_sub_strategy, part_reproduction_random_sub_strategy, \
    part_reproduction_similar_agents_gen_sub_strategy, part_reproduction_similar_agents_fen_sub_strategy
from scaling import linear, sigma_clipping, exponential
from wrappers import OptimizationTask


class Agent:
    def __init__(self, vector: np.ndarray):
        self.vector = vector
        self.fitness_value = None
        self.scaled_fitness_value = None

    def __repr__(self):
        if self.fitness_value is not None:
            if isinstance(self.vector[0], str):
                return f"[{self.vector}, {round(self.fitness_value, 3)}]"
            return f"[{self.vector}, {round(self.fitness_value, 3)}, {self.scaled_fitness_value}]"
        else:
            return f"[{self.vector}, {None}"

    def __eq__(self, other):
        return np.array_equal(self.vector, other.vector)

    def fen_similarity(self, other):
        return np.sum(np.abs(np.subtract(self.vector, other.vector)))

    def gen_similarity(self, other):
        diff_counter = 0
        for i in range(len(self.vector)):
            for j in range(len(self.vector[i])):
                if self.vector[i][j] != other.vector[i][j]:
                    diff_counter += 1
        return diff_counter


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


def get_average_fitness(population: np.ndarray):
    fitness_extractor = np.vectorize(lambda agent: agent.fitness_value)
    fitness_array = fitness_extractor(population)
    return np.average(fitness_array)


def main(task: OptimizationTask,
         coding: callable, decoding: callable,
         selection_method: callable,
         substitution_strategy: callable,
         crossover: callable,
         mutation: callable,
         scaling: callable,
         iterations: int,
         n_agents: int) -> Agent:
    population = generate_starting_population(n_agents, task.limits)
    calculate_fitness_function(task, population)

    for i in range(1, iterations):
        # print("-----------------POPULATION---------------------")
        # print(population)
        parents = selection_method(population)
        # print("----------------- parents ---------------------")
        # print(parents)
        coding(parents)
        children = crossover(parents)
        mutation(children, task.limits)
        decoding(children)
        decoding(parents)
        calculate_fitness_function(task, children)
        scaling(population, children)
        population = substitution_strategy(population, children)

    return get_best_from_population(population)


# main()
limits = np.array([[-15, 15], [-15, 15]])
task = OptimizationTask(cross_in_tray_function, limits)
xd = main(task, dummy, dummy, proportional_method, part_reproduction_random_sub_strategy, arithmetic_crossover,
          mutation_real_fen, linear, 200, 30)
print(xd)
# parents = generate_starting_population(5, limits)
# calculate_fitness_function(circle_function, parents)
# children = generate_starting_population(1, limits)
# calculate_fitness_function(circle_function, children)
#
# print(parents)
# print(children)
# new_population = part_reproduction_similar_agents_fen_sub_strategy(parents, children)
# print(new_population)
#
#
# parents = np.array([Agent(np.array([1])), Agent(np.array([2])), Agent(np.array([3])), Agent(np.array([4]))])
# parents[0].fitness_value = 169.
# parents[1].fitness_value = 576.
# parents[2].fitness_value = 64.
# parents[3].fitness_value = 361.

# children = np.array([])
# exponential(parents, children)
# print(parents)
# print(children)


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
