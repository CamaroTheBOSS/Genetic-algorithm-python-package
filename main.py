import copy

import numpy as np
import random
from selection_methods import proportional_method, stochastic_residual_method, threshold_method, tournament_method, \
    rank_method
from test_functions import circle_function, quadratic_function, dummy, cross_in_tray_function, bukin_function, \
    holder_table_function, egg_holder_function, griewank_function, drop_wave_function, levy_function_n13, \
    rastrigin_function, salesman_function
from mutation import mutation_bin_gen, mutation_bin_fen, mutation_tri_fen, mutation_tri_gen, mutation_real_fen
from crossover import pmx, arithmetic_crossover, mixed_crossover, binary_crossover, ox, cx
from substitution_strategy import full_sub_strategy, \
    part_reproduction_elite_sub_strategy, part_reproduction_random_sub_strategy, \
    part_reproduction_similar_agents_gen_sub_strategy, part_reproduction_similar_agents_fen_sub_strategy
from scaling import linear, sigma_clipping, exponential
from wrappers import OptimizationTask, WrappedCallback, Coding
from travelling_salesman import read_tsp_data
from coding import binary_coding, gray_coding, binary_decoding, gray_decoding


class Agent:
    def __init__(self, vector: np.ndarray, limitations: np.array):
        self.vector = vector
        self.fitness_value = None
        self.scaled_fitness_value = None
        self.limits = limitations

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
        population[i] = Agent(init_vector, limitations)

    return population


def generate_starting_population_for_salesman_problem(n_agents, limitations: np.ndarray):
    population = np.empty(n_agents, dtype=object)
    values = np.arange(0, len(limitations), 1)
    for i in range(n_agents):
        init_vector = np.copy(values)
        np.random.shuffle(init_vector)
        population[i] = Agent(init_vector)
    return population


# Maybe min is better?
def get_best_from_population(population: np.ndarray) -> Agent:
    return max(population, key=lambda agent: agent.fitness_value)


def calculate_fitness_function(population: np.ndarray, task: callable):
    for agent in population:
        agent.fitness_value = task(agent.vector, *task.parameters)


def get_average_fitness(population: np.ndarray):
    fitness_extractor = np.vectorize(lambda agent: agent.fitness_value)
    fitness_array = fitness_extractor(population)
    return np.average(fitness_array)


def main(task: OptimizationTask,
         coding: Coding,
         selection_method: WrappedCallback,
         substitution_strategy: WrappedCallback,
         crossover: WrappedCallback,
         mutation: WrappedCallback,
         scaling: WrappedCallback,
         iterations: int,
         n_agents: int) -> Agent:
    if task.salesman:
        population = generate_starting_population_for_salesman_problem(n_agents, task.limits)
    else:
        population = generate_starting_population(n_agents, task.limits)
    calculate_fitness_function(population, task)

    for i in range(1, iterations):
        parents = selection_method(population)
        coding(parents)
        children = crossover(parents)
        mutation(children, args=(task.limits,))
        coding.decode(children)
        coding.decode(parents)
        calculate_fitness_function(children, task)
        scaling(population, args=(children,))
        population = substitution_strategy(population, args=(children,))

    return get_best_from_population(population)


# main()
# limits = np.array([[-15, 15], [-15, 15]])
# task = OptimizationTask(cross_in_tray_function, limits)
# sub_strat = WrappedCallback(part_reproduction_elite_sub_strategy)
# xd = main(task,
#           Coding(dummy, dummy),
#           WrappedCallback(proportional_method),
#           WrappedCallback(part_reproduction_elite_sub_strategy),
#           WrappedCallback(arithmetic_crossover),
#           WrappedCallback(mutation_real_fen),
#           WrappedCallback(linear),
#           200,
#           30)
# print(xd)

# POP = generate_starting_population(10, np.array([[-20, 20], [-20, 20], [-20, 20]]))
# print('przed', POP)
# gray_coding(POP, 16)
# gray_decoding(POP, 16)
# print('po', POP)
# kids = binary_crossover(POP, 2)
# print("")
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
