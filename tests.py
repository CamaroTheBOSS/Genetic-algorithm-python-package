import numpy as np
import random
from selection_methods import proportional_method, stochastic_residual_method, threshold_method, tournament_method, \
    rank_method
from test_functions import circle_function, quadratic_function, dummy, cross_in_tray_function, bukin_function, \
    holder_table_function, egg_holder_function, griewank_function, drop_wave_function, levy_function_n13, \
    rastrigin_function, salesman_function, knapsack_function
from mutation import mutation_bin_gen, mutation_bin_fen, mutation_tri_fen, mutation_tri_gen, mutation_real_fen, \
    mutation_salesman_problem
from crossover import pmx, arithmetic_crossover, mixed_crossover
from substitution_strategy import full_sub_strategy, \
    part_reproduction_elite_sub_strategy, part_reproduction_random_sub_strategy, \
    part_reproduction_similar_agents_gen_sub_strategy, part_reproduction_similar_agents_fen_sub_strategy
from scaling import linear, sigma_clipping, exponential
from wrappers import OptimizationTask, Coding, WrappedCallback
from main import main
from utils import read_tsp_data, read_knapsack_data


class Config:
    def __init__(self, coding: Coding,
                 selection_method: WrappedCallback,
                 substitution_strategy: WrappedCallback,
                 crossover: WrappedCallback,
                 mutation: WrappedCallback,
                 scaling: WrappedCallback,
                 iterations: int,
                 n_agents: int):
        self.coding = coding
        self.selection_method = selection_method
        self.substitution_strategy = substitution_strategy
        self.crossover = crossover
        self.mutation = mutation
        self.scaling = scaling
        self.iterations = iterations
        self.n_agents = n_agents

    def get(self):
        return [self.coding, self.selection_method, self.substitution_strategy, self.crossover,
                self.mutation, self.scaling, self.iterations, self.n_agents]


def get_feedback(task: OptimizationTask, result):
    print("---------------------------------------------------------------------")
    print(f"Desired: {task.target_x} -> {task.target_y}")
    print(f"Got: {list(result.vector)} -> {result.fitness_value}")
    print(f"Feedback: {1 / (1 + 2 * abs(task.target_y - result.fitness_value))}")


def cross_in_tray_test(config: Config):
    task = OptimizationTask(cross_in_tray_function, np.array([[-15, 15], [-15, 15]]),
                            target_x=[[1.34, 1.34], [-1.34, 1.34], [1.34, -1.34], [-1.34, -1.34]], target_y=2.06261)
    result = main(task, *config.get())
    get_feedback(task, result)


def bukin_test(config: Config):
    task = OptimizationTask(bukin_function, np.array([[-15, -5], [-3, 3]]),
                            target_x=[-10, 1], target_y=0)
    result = main(task, *config.get())
    get_feedback(task, result)


def drop_wave_test(config: Config):
    task = OptimizationTask(drop_wave_function, np.array([[-5.12, 5.12], [-5.12, 5.12]]),
                            target_x=[0, 0], target_y=1)
    result = main(task, *config.get())
    get_feedback(task, result)


def egg_holder_test(config: Config):
    task = OptimizationTask(egg_holder_function, np.array([[-512, -512], [-512, 512]]),
                            target_x=[512, 404.2319], target_y=959.6407)
    result = main(task, *config.get())
    get_feedback(task, result)


def griewank_test(config: Config):
    task = OptimizationTask(griewank_function, np.array([[-600, 600], [-600, 600]]),
                            target_x=[0, 0], target_y=0)
    result = main(task, *config.get())
    get_feedback(task, result)


def holder_table_test(config: Config):
    task = OptimizationTask(holder_table_function, np.array([[-10, 10], [-10, 10]]),
                            target_x=[[8.055, 9.665], [-8.055, 9.665], [8.055, -9.665], [-8.055, -9.665]], target_y=19.2085)
    result = main(task, *config.get())
    get_feedback(task, result)


def levy_test(config: Config):
    task = OptimizationTask(levy_function_n13, np.array([[-10, 10], [-10, 10]]),
                            target_x=[1, 1], target_y=0)
    result = main(task, *config.get())
    get_feedback(task, result)


def rastrigin_test(config: Config):
    task = OptimizationTask(rastrigin_function, np.array([[-5.12, 5.12], [-5.12, 5.12]]),
                            target_x=[0, 0], target_y=0)
    result = main(task, *config.get())
    get_feedback(task, result)


def salesman_test(config: Config, data: np.ndarray, target_x: list, target_y: float):
    task = OptimizationTask(salesman_function, np.full((len(data), 2), np.asarray([0, len(data)])),
                            target_x=target_x, target_y=target_y, problem_type='salesman', args=(data,))
    result = main(task, *config.get())
    get_feedback(task, result)


def knapsack_test(config: Config, data: np.ndarray, capacity: int, target_x: list, target_y: float):
    task = OptimizationTask(knapsack_function, np.asarray([[0, 2**len(data)]]),
                            target_x=target_x, target_y=target_y, problem_type='knapsack', args=(data, capacity,))
    result = main(task, *config.get())
    get_feedback(task, result)


def test(function: bool = True, salesman: bool = True, knapsack: bool = True):
    if function:
        config = Config(
            Coding(dummy, dummy),
            WrappedCallback(proportional_method),
            WrappedCallback(part_reproduction_elite_sub_strategy),
            WrappedCallback(arithmetic_crossover),
            WrappedCallback(mutation_real_fen),
            WrappedCallback(linear),
            200,
            30)

        cross_in_tray_test(config)
        bukin_test(config)
        drop_wave_test(config)
        egg_holder_test(config)
        griewank_test(config)
        holder_table_test(config)
        levy_test(config)
        rastrigin_test(config)

    if salesman:
        config = Config(
            Coding(dummy, dummy),
            WrappedCallback(tournament_method),
            WrappedCallback(part_reproduction_elite_sub_strategy),
            WrappedCallback(pmx),
            WrappedCallback(mutation_salesman_problem, parameters=(0.2,)),
            WrappedCallback(linear),
            100,
            50)

        data = read_tsp_data("salesman/test1.tsp")
        salesman_test(config, data, [], -1368)
        # data = read_tsp_data("salesman/simple.tsp")
        # salesman_test(config, data, [1, 2, 3, 4], -90)

    if knapsack:
        config = Config(
            Coding(dummy, dummy),
            WrappedCallback(tournament_method),
            WrappedCallback(part_reproduction_elite_sub_strategy),
            WrappedCallback(pmx),
            WrappedCallback(mutation_salesman_problem, parameters=(0.2,)),
            WrappedCallback(linear),
            100,
            50)

        data, capacity = read_knapsack_data("knapsack/test1.txt")
        knapsack_test(config, data, capacity, [1, 1, 1, 1, 0, 1, 0, 0, 0, 0], 92 + 57 + 49 + 68 + 43)


test(function=False, salesman=True, knapsack=True)
