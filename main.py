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
from utils import read_tsp_data


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


def generate_starting_population(n_agents: int, limitations: np.ndarray, data_type: str = 'float') -> np.ndarray:
    population = np.empty(n_agents, dtype=object)
    n_dimensions = len(limitations)
    for i in range(n_agents):
        init_vector = np.zeros(n_dimensions, dtype=object)
        for j, limit in enumerate(limitations):
            if data_type == 'float':
                init_vector[j] = random.uniform(limit[0], limit[1])
            elif data_type == 'bin':
                init_vector[j] = '0b'+''.join(np.random.randint(2, size=int(np.log2(limit[1]))).astype(str))
        population[i] = Agent(init_vector, limitations)

    return population


def generate_starting_population_for_salesman_problem(n_agents, limitations: np.ndarray):
    population = np.empty(n_agents, dtype=object)
    values = np.arange(0, len(limitations), 1)
    for i in range(n_agents):
        init_vector = np.copy(values)
        np.random.shuffle(init_vector)
        population[i] = Agent(init_vector, limitations)
    return population


# def generate_starting_population_for_knapsack_problem(n_agents: int, data: np.ndarray, capacity: int):
#     population = np.empty(n_agents, dtype=object)
#     for i in range(n_agents):
#         init_vector = np.random.randint(low=0, high=2, size=(len(data),))
#         while np.sum(np.multiply(init_vector, data[:, 1])) > capacity:
#             idx = np.random.choice(np.where(init_vector == 1)[0])
#             init_vector[idx] = 0
#
#         population[i] = Agent(init_vector, limitations)
#     print(population)
#    return population


# Maybe min is better?
def get_best_from_population(population: np.ndarray) -> Agent:
    return max(population, key=lambda agent: agent.fitness_value)


def calculate_fitness_function(population: np.ndarray, task: callable):
    for agent in population:
        agent.fitness_value = task(agent.vector, task.parameters)


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
    if task.problem_type == 'salesman':
        population = generate_starting_population_for_salesman_problem(n_agents, task.limits)
    elif task.problem_type == 'knapsack':
        population = generate_starting_population(n_agents, task.limits, 'bin')
    else:
        population = generate_starting_population(n_agents, task.limits)
    calculate_fitness_function(population, task)

    for i in range(1, iterations):
        parents = selection_method(population, args=selection_method.parameters)
        coding(parents, args=coding.parameters)
        children = crossover(parents, args=crossover.parameters)
        mutation(children, args=(task.limits,) + mutation.parameters)
        coding.decode(children, args=coding.parameters)
        coding.decode(parents, args=coding.parameters)
        calculate_fitness_function(children, task)
        scaling(population, args=(children,) + scaling.parameters)
        population = substitution_strategy(population, args=(children,) + substitution_strategy.parameters)

    return get_best_from_population(population)
