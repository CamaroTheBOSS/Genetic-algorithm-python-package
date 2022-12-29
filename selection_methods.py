import numpy as np
from copy import deepcopy


def proportional_method(population: np.ndarray, selection_probability: np.ndarray = None) -> np.ndarray:
    n = len(population)
    if selection_probability is None:
        fitness_values = np.array([agent.fitness_value for agent in population])
        fitness_sum = np.sum(fitness_values)
        selection_probability = fitness_values/fitness_sum
    distribution = np.cumsum(selection_probability)
    selected = np.empty_like(population)
    for i in range(n):
        random_value = np.random.rand()
        chosen_idx = np.argwhere(distribution == np.amin(distribution, where=distribution >= random_value, initial=distribution[-1]))
        chosen_idx = chosen_idx.reshape(-1)
        selected[i] = deepcopy(population[chosen_idx[0]])

    return selected


def stochastic_residual_method(population: np.ndarray, selection_probability: np.ndarray = None) -> np.ndarray:

    n = len(population)
    if selection_probability is None:
        fitness_values = np.array([agent.fitness_value for agent in population])
        fitness_sum = np.sum(fitness_values)
        selection_probability = fitness_values/fitness_sum
    exp_n_of_copies = n * selection_probability
    exp_n_of_copies_ = np.floor(exp_n_of_copies).astype(int)
    vacates = n - np.sum(exp_n_of_copies_)
    distribution = np.cumsum(exp_n_of_copies - exp_n_of_copies_)
    selected = np.empty_like(population)

    # inserting copies
    idx_of_copies = np.argwhere(exp_n_of_copies_ > 0).reshape(-1)
    result_idx = 0
    for population_idx in idx_of_copies:
        for i in range(exp_n_of_copies_[population_idx]):
            selected[result_idx] = deepcopy(population[population_idx])
            result_idx += 1

    for i in range(vacates):
        random_value = np.random.rand()
        chosen_idx = np.argwhere(distribution/distribution[-1] == np.amin(distribution/distribution[-1], where=distribution/distribution[-1] <= random_value, initial=distribution[-1]/distribution[-1]))
        chosen_idx = chosen_idx.reshape(-1)
        selected[result_idx] = deepcopy(population[chosen_idx][0])
        result_idx += 1

    return selected


def tournament_method(population: np.ndarray) -> np.ndarray:

    n = len(population)
    selected = np.empty_like(population)
    for i in range(n):
        # r >= 2 or r >= 1?
        r = np.random.randint(2, n)
        contestants = np.random.choice(population, r)
        fitness_values = np.array([agent.fitness_value for agent in contestants])
        selected[i] = deepcopy(contestants[np.argmax(fitness_values)])

    return selected


def rank_method(population: np.ndarray, linear_mapping: bool, proportional: bool, a: float, k: float, b: float) -> np.ndarray:

    sorted_population = np.array(sorted(population, key=lambda x: x.fitness_value, reverse=True))
    ranks = np.zeros_like(sorted_population)
    rank = 0
    for i, agent in enumerate(sorted_population):
        if i > 0:
            if agent.fitness_value == sorted_population[i-1].fitness_value:
                ranks[i] = ranks[i-1]
            else:
                ranks[i] = rank
                rank += 1
        else:
            ranks[i] = rank
            rank += 1

    if linear_mapping:
        selection_probability = a + k * (1 - ranks/np.max(ranks))
    else:
        selection_probability = a + k * (np.max(ranks) - ranks) ** b

    # if conditions not met => wrong a, k or b parameter
    if np.min(selection_probability) < 0 or np.max(selection_probability) > 1 or np.sum(selection_probability) != 1:
        raise ValueError

    for i in range(len(ranks)):
        if i > 0:
            if selection_probability[i] > selection_probability[i-1]:
                raise ValueError

    if proportional:
        selected = proportional_method(sorted_population, selection_probability)

    else:
        selected = stochastic_residual_method(sorted_population, selection_probability)

    return selected


def threshold_method(population: np.ndarray, proportional: bool, threshold: float) -> np.ndarray:

    sorted_population = np.array(sorted(population, key=lambda x: x.fitness_value, reverse=True))
    ranks = np.zeros_like(sorted_population)
    rank = 0
    for i, agent in enumerate(sorted_population):
        if i > 0:
            if agent.fitness_value == sorted_population[i-1].fitness_value:
                ranks[i] = ranks[i-1]
            else:
                ranks[i] = rank
                rank += 1
        else:
            ranks[i] = rank
            rank += 1

    # tutaj powinien byc chyba inny threshold dla każdego agenta, ale w wykladzie nic o tym nie ma
    selection_probability = np.where(ranks < threshold, 1/threshold, 0)

    # nie ma tez informacji co robimy dalej wiec zakladam ze to
    if proportional:
        selected = proportional_method(sorted_population, selection_probability)
    else:
        selected = stochastic_residual_method(sorted_population, selection_probability)

    return selected